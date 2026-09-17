"""Regressions for constrained thermal sampling, reference frames and RNG streams."""

import random
from pathlib import Path

import numpy as np
import pytest

from analysis.fragment_energy import fragment_energy_partition
from core.collision import Collision
from core.fragment import Fragment
from core.molecule import Molecule
from dynamics.thermostat_driver import apply_thermostat, initialize_gle_state
from parallel.trajectory_runner import run_parallel_trajectories
from sampling.random_seed import set_sampling_seed
from thermostats.gle import GLEThermostat
from utils.constants import FS_TO_AU_TIME, R_GAS_HARTREE_PER_K


@pytest.mark.parametrize("mass", [[1.0, 1.0], [1.0, 4.0]])
def test_andersen_preserves_constrained_maxwell_covariance(mass):
    mass = np.array(mass)
    molecule = Molecule(["H", "He"], mass, np.zeros(6), np.zeros(6))
    molecule.remove_com = True
    kt = 300 * R_GAS_HARTREE_PER_K
    rng = np.random.default_rng(918273)
    random.seed(19283)
    samples = []
    # Test invariance of the equilibrium distribution with independent inputs.
    # This avoids autocorrelation-dependent tolerances in a thermostat chain.
    for _ in range(12000):
        molecule.p = rng.normal(size=6) * np.sqrt(molecule.wmass * kt)
        molecule.project_center_of_mass_momentum()
        apply_thermostat(molecule, "andersen", 1.0, 300, np.log(2) * FS_TO_AU_TIME)
        samples.append(molecule.p.copy())
    samples = np.array(samples)
    expected = np.kron(np.diag(mass) - np.outer(mass, mass) / mass.sum(), np.eye(3)) * kt
    np.testing.assert_allclose(np.cov(samples.T), expected, atol=0.025 * kt, rtol=0.03)
    np.testing.assert_allclose(samples.reshape(-1, 2, 3).sum(axis=1), 0, atol=1e-14)
    assert np.mean(samples**2 / molecule.wmass) * 6 / kt == pytest.approx(3, rel=0.03)


@pytest.mark.parametrize("algorithm", ["shake", "rattle"])
def test_andersen_samples_rigid_rotor_at_target_temperature(algorithm):
    mass = np.array([1.0, 2.0])
    molecule = Molecule(["H", "H"], mass, np.array([-2/3, 0, 0, 1/3, 0, 0]), np.zeros(6), nfix=1)
    molecule.remove_com = True
    molecule.add_rigid_constraint_group([0, 1], degrees_of_freedom_removed=1)
    molecule.prepare_rigid_constraints(algorithm=algorithm)
    kt = 300 * R_GAS_HARTREE_PER_K
    rng = np.random.default_rng(1287)
    random.seed(2287)
    energies = []
    for _ in range(10000):
        molecule.p = rng.normal(size=6) * np.sqrt(molecule.wmass * kt)
        molecule.project_center_of_mass_momentum()
        molecule.project_rigid_momenta()
        apply_thermostat(molecule, "andersen", 1.0, 300, np.log(2) * FS_TO_AU_TIME)
        assert molecule._constraint_solver.maximum_velocity_error(molecule.q, molecule.p, molecule.wmass) < 1e-10
        energies.append(0.5 * np.sum(molecule.p**2 / molecule.wmass))
    assert np.mean(energies) == pytest.approx(kt, rel=0.03)
    assert np.var(energies) == pytest.approx(kt**2, rel=0.06)


def _rotation(seed):
    axes, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return axes


@pytest.mark.parametrize("linear", [False, True])
def test_equilibrium_rotational_energy_is_independent_of_reference_and_lab_frames(linear):
    q = np.array([[-2., -1, 0], [2, -1, 0], [0, 1, 0]])
    if linear:
        q[:, 1:] = 0
    mass = np.array([1., 2., 3.])
    q -= np.average(q, weights=mass, axis=0)
    p = mass[:, None] * np.cross([.3, -.7, .5], q)
    expected = .5 * np.sum(p**2 / mass[:, None])
    fragment = {"indices": [0, 1, 2], "atoms": ["X"] * 3, "formula": "X3"}
    for lab_seed in range(4):
        for ref_seed in range(4, 8):
            lab = _rotation(lab_seed)
            reference = q @ _rotation(ref_seed).T + [10, -3, 8]
            reference_before = reference.copy()
            result = fragment_energy_partition(
                (q @ lab.T + [2, -1, 3]).ravel(),
                (p @ lab.T + mass[:, None] * [1, 2, 3]).ravel(), mass, fragment,
                equilibrium_geometries={"X3": reference},
                potential_energy=0.0, reference_potential_energy=0.0,
            )
            assert result["rotational_energy"] == pytest.approx(expected, abs=1e-12)
            assert result["vibrational_energy"] == pytest.approx(0, abs=1e-12)
            np.testing.assert_array_equal(reference, reference_before)


def test_fragment_stored_rotational_energy_matches_cartesian_kinetic_energy():
    q = np.array([[-4., -2, 0], [4, -2, 0], [0, 2, 0]]) @ _rotation(12).T
    fragment = Fragment(["H"] * 3, np.array([1., 2., 3.]), q.ravel(), np.zeros(9))
    fragment.rigid = True
    fragment.MDprimitive = False
    fragment.rotsampling = {0: ("Q", 3)}
    fragment.random_rot = True
    fragment.polyatom_sampling(sampling_seed=318, seed_metadata_file=None)
    assert fragment.erot == pytest.approx(.5 * np.sum(fragment.p**2 / fragment.wmass), rel=1e-12)


def test_collision_seed_reproduces_whole_collision_without_matching_fragments():
    a = Fragment.diatom_init("a", ["H", "H"], req=.74, omega=4400)
    b = Fragment.diatom_init("b", ["H", "H"], req=.74, omega=4400)
    for fragment in (a, b):
        fragment.vibsampling = {0: (0, "Q", 0)}
        fragment.rotsampling = {0: ("Q", 0)}
    collision = Collision(a, b, {"qchem": "Psi4", "functional": "hf"})
    collision.specify_collision_sampling(Rini=12, bmax=1, Ecoll=10)
    collision.sample_bimolecular_reactants(sampling_seed=1729, seed_metadata_file=None)
    q0, p0 = collision.q.copy(), collision.p.copy()
    qa, qb = q0.reshape(2, 2, 3)
    assert not np.allclose(qa[1] - qa[0], qb[1] - qb[0])
    assert not np.isclose(np.linalg.norm(qa[1] - qa[0]), np.linalg.norm(qb[1] - qb[0]))
    collision.sample_bimolecular_reactants(sampling_seed=1729, seed_metadata_file=None)
    np.testing.assert_allclose(collision.q, q0, atol=1e-13)
    np.testing.assert_allclose(collision.p, p0, atol=1e-13)


def test_default_gle_streams_are_reproducible_and_distinct(tmp_path):
    a_file = tmp_path / "GLE-A"
    a_file.write_text("1\n1.0 0.2\n-0.2 1.0\n")
    def draws():
        set_sampling_seed(9137, print_report=False)
        streams = []
        for _ in range(2):
            gle = GLEThermostat(.1, 1., .01, 6, a_file=str(a_file), c_file=None)
            streams.append(gle.step(np.zeros(6), np.ones(6)))
        return streams
    first, repeat = draws(), draws()
    np.testing.assert_array_equal(first, repeat)
    assert not np.array_equal(first[0], first[1])


class _SeedProbe:
    pass


def _seed_probe_factory(_itraj):
    return _SeedProbe()


def _seed_probe(_system, context, **kwargs):
    # Emulate the samplers' handling of an optional explicit seed.
    set_sampling_seed(kwargs.get("sampling_seed"), print_report=False)
    value = (random.random(), np.random.random(), kwargs["thermo_param"]["seed"])
    (Path(context.trajectory_dir) / "probe.txt").write_text(repr(value))


def test_parallel_explicit_seeds_are_distinct_and_schedule_independent(tmp_path):
    runs = []
    params = {"sampling_seed": 8172, "thermostat": "gle", "thermo_param": {"seed": 72}}
    for workers in (1, 3):
        results = run_parallel_trajectories(
            _seed_probe_factory, _seed_probe, total_trajectories=4,
            nparallel_jobs=workers, run_kwargs=params, progress_report=False,
            base_output_dir=str(tmp_path / str(workers)), scratch_base_dir=str(tmp_path / "scratch"),
        )
        assert all(result.status == "finished" for result in results)
        runs.append([(Path(result.trajectory_dir) / "probe.txt").read_text() for result in results])
    assert runs[0] == runs[1]
    assert len(set(runs[0])) == 4
    assert params["thermo_param"]["seed"] == 72


def test_gle_checkpoint_restores_auxiliary_state_and_future_noise(tmp_path):
    a_file = tmp_path / "GLE-A"
    a_file.write_text("1\n1.0 0.2\n-0.2 1.0\n")
    params = {"wopt": 1., "a_file": str(a_file), "c_file": None}
    molecule = Molecule(["He"], np.ones(1), np.zeros(3), np.zeros(3))
    set_sampling_seed(6174, print_report=False)
    initialize_gle_state(molecule, .1, params, 300)
    molecule.p = molecule._gle_state.step(molecule.p, molecule.wmass)
    backfile = tmp_path / "backup.xyz"
    molecule._save_restart_state(backfile, 1, 0., "gle", params, 300, .1)
    state = molecule._load_restart_state(backfile, 1)
    expected = molecule._gle_state.step(molecule.p, molecule.wmass)
    expected_aux = molecule._gle_state.gp.copy()
    initialize_gle_state(molecule, .1, params, 300)
    molecule._restore_restart_state(state, "gle", params, 300, .1)
    actual = molecule._gle_state.step(molecule.p, molecule.wmass)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(molecule._gle_state.gp, expected_aux)
