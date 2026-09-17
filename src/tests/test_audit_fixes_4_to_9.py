import random
from pathlib import Path

import numpy as np
import pytest

from analysis.spectrum import frequency_axis_cm1
from core.molecule import Molecule
from integrators.predcorr import PredCorr
from parallel.trajectory_runner import derive_trajectory_seed, run_parallel_trajectories
from sampling.thermal import thermal_rot_canonical_top
from sampling.thermal import thermal_rot_quantum_spherical_top
from sampling.rel_init_coords import setRelativeInitCoords
from utils.clustering import cluster_chemical_formulas
from utils.constants import ANGSTROM_TO_BOHR, FS_TO_AU_TIME, HARTREE_TO_CM1
from utils.distance import test_to_stop_specific as reaction_stop_specific
from utils.graph_cluster import create_adjacency_list


class _ParallelRandomProbe:
    pass


def _parallel_random_probe_factory(_itraj):
    return _ParallelRandomProbe()


def _parallel_random_probe_run(_system, context, **_kwargs):
    output = Path(context.trajectory_dir) / "random_probe.dat"
    output.write_text(
        f"{random.random():.17g} {np.random.random():.17g}\n",
        encoding="utf-8",
    )


def test_canonical_rigid_top_has_correct_component_variances_and_mean_energy():
    random.seed(918273)
    rt = 0.003
    moments = np.array([2.0, 5.0, 11.0])
    samples = np.array([thermal_rot_canonical_top(rt, moments) for _ in range(30000)])

    assert np.var(samples, axis=0) == pytest.approx(moments * rt, rel=0.05)
    energies = 0.5 * np.sum(samples * samples / moments, axis=1)
    assert np.mean(energies) == pytest.approx(1.5 * rt, rel=0.03)


def test_canonical_linear_top_skips_zero_moment_axis():
    random.seed(123)
    rt = 0.002
    moments = np.array([0.0, 4.0, 4.0])
    samples = np.array([thermal_rot_canonical_top(rt, moments) for _ in range(1000)])

    assert np.all(samples[:, 0] == 0.0)
    assert np.var(samples[:, 1:], axis=0) == pytest.approx([4.0 * rt, 4.0 * rt], rel=0.12)


def test_quantum_diatomic_rotor_matches_discrete_canonical_population():
    random.seed(112358)
    inertia_temperature = 1.0
    samples = np.array(
        [thermal_rot_quantum_spherical_top(0.001, inertia_temperature / 0.001) for _ in range(30000)]
    )
    states = np.arange(16)
    expected = (2 * states + 1) * np.exp(-states * (states + 1) / (2.0 * inertia_temperature))
    expected /= expected.sum()
    observed = np.bincount(samples, minlength=len(states))[: len(states)] / len(samples)

    assert np.abs(observed - expected).sum() < 0.035


def test_relative_coordinate_helper_honors_thermal_collision_energy(monkeypatch):
    monkeypatch.setattr("sampling.rel_init_coords.thermal_collision_energy", lambda rt: 0.125)
    mass_a = np.array([2.0])
    mass_b = np.array([3.0])
    _bimp, _q, p = setRelativeInitCoords(
        mass_a,
        mass_b,
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        Ecoll=None,
        Ecoll_thermal=True,
        temp=300.0,
        bmax=0.0,
        Rini=5.0,
    )
    velocity_a = p[:3] / mass_a[0]
    velocity_b = p[3:] / mass_b[0]
    reduced_mass = mass_a[0] * mass_b[0] / (mass_a[0] + mass_b[0])

    assert 0.5 * reduced_mass * np.dot(velocity_a - velocity_b, velocity_a - velocity_b) == pytest.approx(0.125)


@pytest.mark.parametrize(
    "channels",
    [
        {},
        {"empty": []},
        {"bad_operator": [((0, 1), "EQ", 1.0)]},
        {"bad_pair": [((0, 2), "GT", 1.0)]},
    ],
)
def test_reaction_stop_rejects_vacuous_or_invalid_channels(channels):
    q = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        reaction_stop_specific(q, channels)


def test_reaction_stop_is_checked_between_print_steps(monkeypatch, tmp_path):
    molecule = Molecule(
        atoms=["H", "H"],
        mass=np.array([1.0, 1.0]),
        q_ini=np.array([0.0, 0.0, 0.0, 0.5 * ANGSTROM_TO_BOHR, 0.0, 0.0]),
        p_ini=np.zeros(6),
    )
    molecule.fname = "stop_test"
    molecule.qchem = {"wfu": False}
    molecule._set_trajectory_scratch_dir = lambda traj_file, restart=False: None
    molecule.get_energy = lambda file_wf=None: (0.0, 0.0, 0.0)

    propagation_count = {"value": 0}

    def propagate(system, integrator, dt, propagator):
        propagation_count["value"] += 1
        system.q[3] = 2.0 * ANGSTROM_TO_BOHR

    monkeypatch.setattr("core.molecule.initialize_integrator", lambda *args, **kwargs: None)
    monkeypatch.setattr("core.molecule.apply_integrator", propagate)
    monkeypatch.setattr("core.molecule.prepare_wavefunction_directory", lambda *args, **kwargs: tmp_path)

    molecule.run_trajectory(
        maxstep=5,
        iprint=100,
        traj_file=str(tmp_path / "traj.xyz"),
        backfile=str(tmp_path / "backup.xyz"),
        pairs_to_stop={"dissociation": [((0, 1), "GT", 1.0)]},
    )

    assert propagation_count["value"] == 1


def test_hydrogen_bonds_and_monatomic_products_are_retained():
    atoms = ["H", "H", "Cl"]
    q = np.array(
        [
            0.0, 0.0, 0.0,
            0.74 * ANGSTROM_TO_BOHR, 0.0, 0.0,
            20.0 * ANGSTROM_TO_BOHR, 0.0, 0.0,
        ]
    )
    adjacency = create_adjacency_list(
        atoms,
        q,
        bond_th_HX=1.2 * ANGSTROM_TO_BOHR,
        bond_th_XX=2.0 * ANGSTROM_TO_BOHR,
    )

    assert adjacency[0] == [1]
    assert adjacency[1] == [0]
    assert cluster_chemical_formulas(q, atoms, eps=4.2, minPts=2) == "H2 + Cl"


def test_spectrum_frequency_axis_converts_fs_and_cycles_to_wavenumbers():
    dt_fs = 0.5
    axis = frequency_axis_cm1(2, dt_fs)
    expected_nyquist = np.pi * HARTREE_TO_CM1 / (dt_fs * FS_TO_AU_TIME)

    assert axis == pytest.approx([0.0, expected_nyquist])


def test_saved_spectrum_history_contains_physical_velocities():
    molecule = Molecule(
        atoms=["H"],
        mass=np.array([2.0]),
        q_ini=np.zeros(3),
        p_ini=np.array([4.0, 6.0, 8.0]),
    )
    molecule.save_velocity_and_distance_matrix(time_fs=0.25)

    assert molecule.vsave[0] == pytest.approx([2.0, 3.0, 4.0])
    assert molecule.tsave == [0.25]


def test_parallel_trajectory_seeds_are_reproducible_and_distinct():
    seeds_a = [derive_trajectory_seed(20260728, index) for index in range(20)]
    seeds_b = [derive_trajectory_seed(20260728, index) for index in range(20)]

    assert seeds_a == seeds_b
    assert len(set(seeds_a)) == len(seeds_a)
    assert seeds_a != [derive_trajectory_seed(20260729, index) for index in range(20)]
    assert all(0 <= seed < 2**32 for seed in seeds_a)


def test_parallel_rng_is_independent_of_worker_scheduling(tmp_path):
    def run_probe(directory, workers):
        results = run_parallel_trajectories(
            _parallel_random_probe_factory,
            _parallel_random_probe_run,
            total_trajectories=4,
            nparallel_jobs=workers,
            run_name="probe",
            base_output_dir=str(directory / "output"),
            scratch_base_dir=str(directory / "scratch"),
            base_seed=8675309,
            progress_report=False,
        )
        values = []
        for result in results:
            assert result.status == "finished"
            probe_file = Path(result.trajectory_dir) / "random_probe.dat"
            values.append(probe_file.read_text(encoding="utf-8"))
        return values

    serial_schedule = run_probe(tmp_path / "serial", workers=1)
    parallel_schedule = run_probe(tmp_path / "parallel", workers=3)

    assert serial_schedule == parallel_schedule
    assert len(set(serial_schedule)) == len(serial_schedule)


def test_checkpoint_roundtrip_restores_step_precision_and_rng_state(tmp_path):
    backfile = tmp_path / "backup.xyz"
    molecule = Molecule(
        atoms=["H"],
        mass=np.array([1.0]),
        q_ini=np.array([1.234567890123, -0.123456789012, 9.876543210987]),
        p_ini=np.array([0.111111111111, -0.222222222222, 0.333333333333]),
    )
    random.seed(13579)
    np.random.seed(24680)
    with open(backfile, "w", encoding="utf-8") as handle:
        molecule.print_trajectory(handle, 7, 1.0, 0.0, 0.0, 0.0, 0.0)
    molecule._save_restart_state(backfile, 7, -1.2345, thermostat=None, dt=1.0)
    expected_python = random.random()
    expected_numpy = np.random.random()

    random.seed(1)
    np.random.seed(1)
    restarted = Molecule(
        atoms=["H"],
        mass=np.array([1.0]),
        q_ini=np.zeros(3),
        p_ini=np.zeros(3),
    )
    step = restarted.restart_init("restart", backfile=backfile, restart=True)
    state = restarted._load_restart_state(backfile, step)
    initial_energy = restarted._restore_restart_state(state, None, None, None, 1.0)

    assert step == 7
    assert restarted.q == pytest.approx(molecule.q, abs=3.0e-12)
    assert restarted.p == pytest.approx(molecule.p, abs=1.0e-12)
    assert initial_energy == pytest.approx(-1.2345)
    assert random.random() == pytest.approx(expected_python)
    assert np.random.random() == pytest.approx(expected_numpy)


def test_predictor_corrector_history_roundtrips_through_restart_state(tmp_path):
    molecule = Molecule(
        atoms=["H", "H"],
        mass=np.array([1.0, 1.0]),
        q_ini=np.zeros(6),
        p_ini=np.zeros(6),
    )
    original = PredCorr(order=3, ndim=6)
    original.step = 9
    original.save_veloc[:] = np.arange(18, dtype=float).reshape(3, 6)
    original.save_force[:] = -original.save_veloc
    backfile = tmp_path / "predcorr.xyz"

    molecule._save_restart_state(
        backfile,
        completed_step=12,
        initial_energy=-2.0,
        thermostat=None,
        dt=0.5,
        integrator="predcorr",
        integrator_order=3,
        propagator=original,
    )
    state = molecule._load_restart_state(backfile, completed_step=12)
    restored = PredCorr(order=3, ndim=6)
    molecule._restore_integrator_restart_state(state, restored, "predcorr", 3)
