"""Physics, experimental energy inputs and neutral-file/QCT integration."""

from copy import deepcopy
import json
from pathlib import Path
import random

import numpy as np
import pytest

from core.molecule import Molecule
from photoionization import EnergyDistribution, prepare_ionic_state
from photoionization.distributions import EnergyConstraints
from photoionization.sources import HessianSource, MDTrajectorySource
from utils.constants import BOHR_TO_ANGSTROM, HARTREE_TO_EV


Q = np.array([-1., .2, .3, 1., -.1, .4])
P = np.array([2., -.3, .7, -.4, .8, -.2])
MASS = np.array([1000., 1500.])


def prepare(**kwargs):
    args = dict(photon_energy=15., level=1, ionic_energy=11., seed=37)
    args.update(kwargs)
    return prepare_ionic_state(Q, P, MASS, -100., -100. + 10. / HARTREE_TO_EV, **args)


def test_level_zero_preserves_every_cartesian_component():
    state = prepare(level=0, ionic_energy=None)
    np.testing.assert_array_equal(state.q, Q)
    np.testing.assert_array_equal(state.p, P)
    assert not np.shares_memory(state.p, P)
    assert state.ionic_energy_ev == pytest.approx(10.)
    assert state.electron_energy_ev == pytest.approx(5.)
    assert state.energy_residual_ev == pytest.approx(0., abs=1e-11)


def test_level_one_scales_all_momenta_and_closes_energy_balance():
    state = prepare()
    expected_tn = .5 * np.sum(P**2 / np.repeat(MASS, 3)) * HARTREE_TO_EV
    expected_scale = np.sqrt((expected_tn + 1.) / expected_tn)
    np.testing.assert_array_equal(state.q, Q)
    np.testing.assert_allclose(state.p, expected_scale * P, rtol=1e-11)
    assert state.electron_energy_ev == 4.
    assert state.ionic_kinetic_ev == pytest.approx(expected_tn + 1.)
    assert abs(state.energy_residual_ev) < 1e-11
    # Level 1 does not project away COM motion or preserve angular momentum.
    np.testing.assert_allclose(state.ionic_linear_momentum, expected_scale * state.neutral_linear_momentum)
    np.testing.assert_allclose(state.ionic_angular_momentum, expected_scale * state.neutral_angular_momentum)


@pytest.mark.parametrize("ionic", [11., ([10., 11., 12.], [0., 1., 0.])])
@pytest.mark.parametrize("electron", [4., ([3., 4., 5.], [0., 1., 0.])])
def test_all_constant_distribution_combinations(ionic, electron):
    state = prepare(ionic_energy=ionic, electron_energy=electron)
    assert state.ionic_energy_ev + state.electron_energy_ev == pytest.approx(15.)
    if np.isscalar(ionic) or np.isscalar(electron):
        assert state.ionic_energy_ev == pytest.approx(11.)
    else:
        assert 10. < state.ionic_energy_ev < 12.
    assert abs(state.energy_residual_ev) < 1e-10


@pytest.mark.parametrize("kwargs", [
    {"ionic_energy": 11.}, {"electron_energy": 4.},
    {"ionic_energy": ([10., 12.], [1., 1.])},
    {"electron_energy": ([3., 5.], [1., 1.])},
])
def test_one_energy_input_derives_the_other(kwargs):
    state = prepare(ionic_energy=None, **kwargs) if "ionic_energy" not in kwargs else prepare(**kwargs)
    assert state.electron_energy_ev + state.ionic_energy_ev == pytest.approx(15.)


def test_binding_energy_is_not_ionic_kinetic_energy_and_offset_aligns_pes():
    raw = prepare_ionic_state(Q, P, MASS, 0., 0., photon_energy=15.,
                             ionic_energy=11., ion_energy_offset=10., level=1)
    np.testing.assert_allclose(raw.p, prepare().p, atol=1e-11)
    assert raw.ionic_kinetic_ev != pytest.approx(11.)
    assert raw.ion_energy_offset_ev == 10.


@pytest.mark.parametrize("kwargs,match", [
    ({"electron_energy": 5.}, "must equal"),
    ({"ionic_energy": 1.}, "negative"),
    ({"ionic_energy": 16.}, "negative"),
    ({"ionic_energy": None}, "requires"),
    ({"level": 0}, "Level 0"),
    ({"level": 4}, "levels 0, 1, 2 and 3"),
    ({"photon_energy": 0.}, "positive"),
    ({"photon_energy": np.nan}, "finite"),
    ({"ionic_energy": True}, "finite number"),
    ({"electron_energy": ([0., 1.], [1., 1.])}, "support"),
    ({"ionic_energy": ([0., 1.], [1., 1.])}, "overlapping"),
])
def test_incompatible_inputs_fail_without_mutation(kwargs, match):
    before = Q.copy(), P.copy()
    with pytest.raises(ValueError, match=match):
        prepare(**kwargs)
    np.testing.assert_array_equal(Q, before[0])
    np.testing.assert_array_equal(P, before[1])


def test_zero_momenta_are_not_given_an_arbitrary_direction():
    with pytest.raises(ValueError, match="zero atomic momenta"):
        prepare_ionic_state(Q, np.zeros_like(P), MASS, 0., 0., level=1,
                            photon_energy=15., ionic_energy=11.)
    state = prepare_ionic_state(Q, np.zeros_like(P), MASS, 0., 10. / HARTREE_TO_EV,
                                level=1, photon_energy=15., ionic_energy=10.)
    np.testing.assert_array_equal(state.p, 0.)


def test_exact_zero_target_and_closed_level_zero_state():
    tn = .5 * np.sum(P**2 / np.repeat(MASS, 3)) * HARTREE_TO_EV
    state = prepare_ionic_state(Q, P, MASS, 0., (11. + tn) / HARTREE_TO_EV,
                                level=1, photon_energy=15., ionic_energy=11.)
    assert state.ionic_kinetic_ev < 1e-12
    with pytest.raises(ValueError, match="photon energy window"):
        prepare(level=0, ionic_energy=None, photon_energy=9.)


@pytest.mark.parametrize("grid,density", [
    ([1.], [1.]), ([1., 1.], [1., 1.]), ([2., 1.], [1., 1.]),
    ([0., 1.], [0., 0.]), ([0., 1.], [1., -1.]),
    ([0., np.inf], [1., 1.]), ([0., 1.], [1., np.nan]),
    ([-1., 1.], [1., 1.]), ([0., 1.], [1.]),
])
def test_invalid_distributions_rejected(grid, density):
    with pytest.raises(ValueError):
        EnergyDistribution(grid, density)


def test_nonuniform_grid_normalization_and_monte_carlo_cdf():
    # Linear density 2x on [0,1] has CDF x^2, including a nonuniform input grid.
    distribution = EnergyDistribution([0., .03, .6, 1.], [0., .06, 1.2, 2.])
    draws = distribution.sample(np.random.default_rng(314159), 2500)
    ordered = np.sort(draws)
    assert np.max(np.abs(ordered**2 - (np.arange(draws.size) + .5) / draws.size)) < .035
    assert np.all(distribution.pdf([-1., 2.]) == 0.)
    table = EnergyDistribution.from_input(np.column_stack((distribution.energies, distribution.densities)))
    np.testing.assert_allclose(table.pdf(ordered), distribution.pdf(ordered))


def test_two_splines_are_sampled_conditionally_not_independently():
    # f(I)=2I and f(e)=2e with photon=1 gives normalized Beta(2,2).
    constraints = EnergyConstraints(1., ([0., 1.], [0., 2.]), ([0., 1.], [0., 2.]))
    rng = np.random.default_rng(2345)
    draws = np.array([constraints.sample(rng) for _ in range(2500)])
    np.testing.assert_allclose(draws[:, 0] + draws[:, 1], 1.)
    assert draws[:, 2] == pytest.approx(np.full(2500, 2. / 3.))
    ordered = np.sort(draws[:, 0])
    assert np.max(np.abs(3 * ordered**2 - 2 * ordered**3 - (np.arange(2500) + .5) / 2500)) < .035


def test_density_truncation_zero_gaps_and_reproducibility():
    constraints = EnergyConstraints(15., ionic_energy=([0., 15.], [1., 1.]))
    rng = np.random.default_rng(999)
    draws = np.array([constraints.sample(rng, minimum_binding=10.) for _ in range(100)])
    assert np.all((draws[:, 0] >= 10.) & (draws[:, 0] <= 15.))
    np.testing.assert_allclose(draws[:, 2], 1. / 3.)
    distribution = EnergyDistribution([0., 1., 2., 3., 4., 5.], [1., 0., 0., 0., 0., 1.])
    samples = distribution.sample(np.random.default_rng(765), 200)
    assert np.all((samples < 1.) | (samples > 4.))
    np.testing.assert_array_equal(samples, distribution.sample(np.random.default_rng(765), 200))


@pytest.fixture
def backend(tmp_path):
    pes = tmp_path / "pes"
    pes.mkdir()
    (pes / "pes_interface.py").write_text(
        "import numpy as np\n"
        "class PESCalculator:\n"
        "    def __init__(self, pes_dir, config):\n"
        f"        self.shift = config['charge'] * 10. / {HARTREE_TO_EV}\n"
        "    def energy(self, q, atoms):\n"
        "        return self.shift + .001 * np.dot(q, q)\n"
        "    def force(self, q, atoms):\n"
        "        return -.002 * q\n")
    molecule = Molecule(["H", "He"], MASS, Q, P)
    molecule.qchem = {"qchem": "PES", "pes_path": str(pes), "charge": 0, "multiplicity": 1}
    ion = dict(molecule.qchem, charge=1, multiplicity=2)
    return molecule, ion


def write_md(path, frames):
    with path.open("w") as handle:
        for index, (q, p) in enumerate(frames):
            handle.write(f"2\nstep= {10 * index}\n")
            for symbol, position, momentum in zip(["H", "He"], q.reshape(-1, 3), p.reshape(-1, 3)):
                values = np.concatenate((position * BOHR_TO_ANGSTROM, momentum))
                handle.write(symbol + " " + " ".join(f"{x:.17g}" for x in values) + "\n")


def test_md_source_keeps_saved_phase_points_and_source_object(backend, tmp_path):
    molecule, config = backend
    path = tmp_path / "neutral.xyz"
    write_md(path, [(Q, P), (Q + 7., P * 2.)])
    molecule.qchem["scratch_dir"] = "user_neutral_scratch"
    molecule._nosehoover_state = {"old": [3.]}
    before = deepcopy(molecule.__dict__)
    config_before = deepcopy(config)
    result = molecule.tpepico(config, photon_energy=15., neutral_md_file=path,
                              level=0, n_samples=6, md_start=1, seed=9,
                              output_dir=tmp_path / "launches")
    for ion, state in zip(result.ions, result.initial_states):
        np.testing.assert_allclose(state.q, Q + 7., atol=1e-14)
        np.testing.assert_array_equal(state.p, P * 2.)
        assert state.source_index == 1
        assert ion.qchem["charge"] == 1
        assert "scratch_dir" not in ion.qchem
        assert ion._nosehoover_state is None
    np.testing.assert_array_equal(molecule.q, before["q"])
    np.testing.assert_array_equal(molecule.p, before["p"])
    assert molecule.qchem == before["qchem"]
    assert molecule._nosehoover_state == before["_nosehoover_state"]
    assert config == config_before
    metadata = json.loads((Path(result.output_dir) / "initial_states.json").read_text())
    assert len(metadata["initial_states"]) == 6
    saved = np.load(Path(result.output_dir) / "initial_states.npz")
    np.testing.assert_array_equal(saved["p"], [s.p for s in result.initial_states])
    with pytest.raises(ValueError, match="ensemble"):
        _ = result.ion


def test_hessian_source_uses_supplied_matrix_without_neutral_md_or_hessian_calculation(backend, tmp_path, monkeypatch):
    molecule, config = backend
    molecule.q_ini = np.array([-1., 0., 0., 1., 0., 0.])
    # One mass-weighted stretch at omega=.02, with exact external zero modes.
    direction = np.array([1., 0., 0., -1., 0., 0.])
    inverse_reduced_mass = 1 / MASS[0] + 1 / MASS[1]
    hessian = np.outer(direction, direction) * .02**2 / inverse_reduced_mass
    hessfile = tmp_path / "neutral.hess"
    np.savetxt(hessfile, hessian)
    def forbidden(*args, **kwargs):
        raise AssertionError("No neutral dynamics or Hessian calculation is allowed")
    monkeypatch.setattr(Molecule, "run_trajectory", forbidden)
    monkeypatch.setattr("normalmode.hessian.getHessian", forbidden)
    random.seed(15)
    np.random.seed(16)
    random_before, numpy_before = random.getstate(), np.random.get_state()
    result = molecule.tpepico(config, photon_energy=15., neutral_hessian=hessfile,
                              level=1, electron_energy=4., n_samples=3, seed=192,
                              qct_options={"fix_quantum": [(0, 2)]})
    assert random.getstate() == random_before
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_before[1])
    repeated = molecule.tpepico(config, photon_energy=15., neutral_hessian=hessfile,
                                level=1, electron_energy=4., n_samples=3, seed=192,
                                qct_options={"fix_quantum": [(0, 2)]})
    q0 = molecule.q_ini.reshape(-1, 3).copy()
    q0 -= np.average(q0, weights=MASS, axis=0)
    for state, other in zip(result.initial_states, repeated.initial_states):
        dq = state.q - q0.reshape(-1)
        total_neutral = state.neutral_kinetic_ev / HARTREE_TO_EV + .5 * dq @ hessian @ dq
        assert total_neutral == pytest.approx(.02 * 2.5, abs=1e-12)
        np.testing.assert_array_equal(state.q, other.q)
        np.testing.assert_array_equal(state.p, other.p)
    assert not np.allclose(result.initial_states[0].q, result.initial_states[1].q)


def test_real_ionic_trajectory_starts_from_prepared_momenta(backend, tmp_path, monkeypatch):
    molecule, config = backend
    path = tmp_path / "neutral.xyz"
    write_md(path, [(Q, P)])
    monkeypatch.chdir(tmp_path)
    result = molecule.tpepico(config, photon_energy=15., neutral_md_file=path,
                              level=1, ionic_energy=11., seed=6,
                              dynamics={"maxstep": 2, "timestep": .001, "Rstop": 100.})
    ion, state = result.ion, result.initial_states[0]
    assert ion.termination_step == 2
    assert not np.array_equal(ion.q, state.q)
    np.testing.assert_array_equal(ion.q_ini, state.q)
    np.testing.assert_array_equal(ion.p_ini, state.p)
    assert ion.qchem["charge"] == 1
    assert len(list(Path(result.output_dir).glob("tpepico_*.xyz"))) >= 1


@pytest.mark.parametrize("extra,match", [
    ({}, "exactly one"),
    ({"neutral_hessian": np.eye(6), "neutral_md_file": "unused"}, "exactly one"),
    ({"neutral_md_file": "unused", "qct_options": {}}, "apply only"),
    ({"neutral_md_file": "unused", "n_samples": 0}, "positive integer"),
])
def test_source_selection_validation(backend, extra, match):
    molecule, config = backend
    with pytest.raises(ValueError, match=match):
        molecule.tpepico(config, photon_energy=15., **extra)


@pytest.mark.parametrize("text,match", [
    ("", "No MD frames"),
    ("2\nstep= 0\nH 0 0 0\nHe 0 0 1\n", "px py pz"),
    ("2\nstep= 0\nHe 0 0 0 0 0 0\nH 0 0 1 0 0 0\n", "identities/order"),
    ("2\nstep= 0\nH 0 0 0 0 0 0\n", "px py pz"),
])
def test_md_file_validation(backend, tmp_path, text, match):
    molecule, _ = backend
    path = tmp_path / "bad.xyz"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        MDTrajectorySource(molecule, path)


def test_nonlinear_qct_from_xyz_and_supplied_hessian(tmp_path):
    from normalmode.eckart import get_eckart_projector
    mass = np.array([16000., 1000., 1000.])
    q = np.array([0., 0., 0., 2., 0., 0., -.5, 1.8, 0.])
    molecule = Molecule(['O', 'H', 'H'], mass, q + 100., np.zeros(9))
    xyz = tmp_path / 'neutral.xyz'
    xyz.write_text('3\nneutral equilibrium\n' + ''.join(
        atom + ' ' + ' '.join(f'{x:.17g}' for x in position * BOHR_TO_ANGSTROM) + '\n'
        for atom, position in zip(molecule.atoms, q.reshape(-1, 3))))
    projector = np.eye(9) - get_eckart_projector(mass, q)
    root_mass = np.sqrt(np.repeat(mass, 3))
    hessian = .02**2 * root_mass[:, None] * projector * root_mass[None, :]
    source = HessianSource(molecule, hessian, xyz, options={'fix_quantum': [(0, 2)]})
    sampled_q, sampled_p, _ = source.draw(np.random.default_rng(34), 0)
    centered = q.reshape(-1, 3) - np.average(q.reshape(-1, 3), weights=mass, axis=0)
    dq = sampled_q - centered.reshape(-1)
    energy = .5 * np.sum(sampled_p**2 / np.repeat(mass, 3)) + .5 * dq @ hessian @ dq
    assert energy == pytest.approx(.02 * (2.5 + .5 + .5), abs=1e-12)
    np.testing.assert_allclose(sampled_p.reshape(-1, 3).sum(axis=0), 0., atol=1e-12)
    np.testing.assert_array_equal(molecule.q_ini, q + 100.)


def test_threshold_constant_electron_and_distribution_output(backend, tmp_path):
    molecule, config = backend
    path = tmp_path / 'neutral.xyz'
    write_md(path, [(Q, P)])
    result = molecule.tpepico(
        config, photon_energy=11., neutral_md_file=path, level=1,
        electron_energy=0., ionic_energy=([10., 12.], [1., 1.]),
        seed=12, output_dir=tmp_path / 'threshold')
    state = result.initial_states[0]
    assert state.ionic_energy_ev == 11.
    assert state.electron_energy_ev == 0.
    record = json.loads((Path(result.output_dir) / 'initial_states.json').read_text())
    assert record['energy_inputs']['ionic']['energies_ev'] == [10., 12.]
    assert record['energy_inputs']['electron'] == 0.
    assert record['rng_initial_state']['bit_generator'] == 'PCG64'


@pytest.mark.parametrize('ionic', [11., ([10., 11., 12.], [0., 1., 0.])])
@pytest.mark.parametrize('electron', [4., ([3., 4., 5.], [0., 1., 0.])])
def test_photoionization_object_all_energy_inputs(backend, tmp_path, ionic, electron):
    from smite import Photoionization
    molecule, config = backend
    before = deepcopy(config)
    path = tmp_path / 'neutral.xyz'
    write_md(path, [(Q, P)])
    photoion = Photoionization(
        molecule, qchem=config, fname='chosen_ion', neutra_frankcondon_geom=molecule.q_ini,
        neutral_traj=path, photon_energy=15., level=1,
        electron_energy=electron, ionic_energy=ionic, sampling_seed=123)
    state = photoion.sample_initial_conditions()
    assert photoion.initial_state is state
    assert photoion.photoionization_initial_state is state
    assert photoion.qchem['charge'] == 1
    assert photoion.qchem['multiplicity'] == 2
    assert photoion.fname == 'chosen_ion'
    np.testing.assert_allclose(photoion.q, Q, atol=1e-14)
    np.testing.assert_array_equal(photoion.p, state.p)
    np.testing.assert_array_equal(molecule.p, P)
    assert abs(state.energy_residual_ev) < 1e-10
    assert config == before


def test_photoionization_setup_does_not_evaluate_or_propagate_neutral(backend, monkeypatch):
    from smite import Photoionization
    molecule, config = backend
    def forbidden(*args, **kwargs):
        raise AssertionError('Sampling setup must not perform neutral calculations')
    monkeypatch.setattr('photoionization.driver.Potential_Energy', forbidden)
    monkeypatch.setattr(Molecule, 'run_trajectory', forbidden)
    photoion = Photoionization(
        molecule, qchem=config,
        neutral_hessian='previously_saved.hess', neutra_frankcondon_geom=molecule.q_ini,
        photon_energy=15., level=1, ionic_energy=11.)
    assert photoion.sampling_set
    assert photoion.initial_state is None


def test_photoionization_sampling_stream_repeated_runs_and_explicit_seed(backend, tmp_path):
    from smite import Photoionization
    molecule, config = backend
    path = tmp_path / 'neutral.xyz'
    write_md(path, [(Q, P)])
    def system():
        return Photoionization(molecule, config, neutra_frankcondon_geom=molecule.q_ini,
            neutral_traj=path, photon_energy=15., level=1,
            ionic_energy=([10., 12.], [1., 1.]), sampling_seed=428)
    a, b = system(), system()
    records = []
    for _ in range(2):
        records.append(a.sample_initial_conditions())
        expected = b.sample_initial_conditions()
        np.testing.assert_array_equal(records[-1].p, expected.p)
    assert not np.array_equal(records[0].p, records[1].p)
    a._nosehoover_state = {'old': 123}
    explicit = a.sample_initial_conditions(sampling_seed=9)
    np.testing.assert_array_equal(explicit.p, a.sample_initial_conditions(sampling_seed=9).p)
    assert a._nosehoover_state is None
    np.testing.assert_array_equal(a.sample_initial_conditions().p, b.sample_initial_conditions().p)


@pytest.mark.parametrize('mode,expected_step', [('channel', 2), ('global', 2), ('persistent', 3), ('time', 1)])
def test_photoionization_object_uses_every_step_stopping(backend, tmp_path, monkeypatch, mode, expected_step):
    from smite import Photoionization
    molecule, config = backend
    q = np.array([0., 0., 0., .5 / BOHR_TO_ANGSTROM, 0., 0.])
    path = tmp_path / 'neutral.xyz'
    write_md(path, [(q, P)])
    photoion = Photoionization(molecule, config, fname='ion', neutra_frankcondon_geom=molecule.q_ini,
                              neutral_traj=path, photon_energy=15., level=0)
    propagation = []
    def propagate(system, integrator, dt, propagator):
        propagation.append(integrator)
        system.q[3] += .5 / BOHR_TO_ANGSTROM
    monkeypatch.setattr('core.molecule.apply_integrator', propagate)
    monkeypatch.chdir(tmp_path)
    channels = {'fragmentation': [((0, 1), 'GT', 1.2), ((0, 1), 'LT', 3.)]}
    stopping = {'Rstop': 1.2} if mode == 'global' else {'pairs_to_stop': channels}
    returned = photoion.sample_and_run_dynamics(
        integrator='leapfrog', integrator_order=4, timestep=.5,
        maxstep=1 if mode == 'time' else 10, iprint=100,
        reaction_persistence_steps=2 if mode == 'persistent' else 1,
        traj_file='ion_traj.xyz', backfile='ion_backup.xyz', **stopping)
    assert returned is photoion
    assert photoion.termination_step == expected_step
    assert photoion.termination_time_fs == .5 * expected_step
    assert len(propagation) == expected_step
    assert all(integrator == 'leapfrog' for integrator in propagation)
    if mode == 'time':
        assert photoion.termination_reason == 'maxstep'
    else:
        assert photoion.termination_reason == 'stop_condition'
        if mode != 'global':
            assert photoion.termination_channel == 'fragmentation'
    # Even with iprint=100, the terminal frame is written at the stopping step.
    lines = (tmp_path / 'ion_traj.xyz').read_text().splitlines()
    assert len(lines) == 8
    assert int(lines[5].split()[1]) == expected_step
    np.testing.assert_array_equal(photoion.initial_state.p, P)
    np.testing.assert_array_equal(molecule.p, P)


def test_photoionization_restart_uses_checkpoint_without_resampling(backend, tmp_path, monkeypatch):
    from smite import Photoionization
    molecule, config = backend
    path = tmp_path / 'neutral.xyz'
    write_md(path, [(Q, P)])
    photoion = Photoionization(molecule, config, fname='restart_ion', neutra_frankcondon_geom=molecule.q_ini,
                              neutral_traj=path, photon_energy=15., level=0)
    monkeypatch.chdir(tmp_path)
    photoion.sample_and_run_dynamics(maxstep=1, timestep=.001, Rstop=100.,
                                    output_dir=tmp_path / 'ionic_run')
    initial_state = photoion.initial_state
    def forbidden(*args, **kwargs):
        raise AssertionError('Restart must not draw another neutral or ionic state')
    monkeypatch.setattr(photoion, 'sample_initial_conditions', forbidden)
    path.unlink()
    photoion.sample_and_run_dynamics(restart=True, maxstep=2, timestep=.001, Rstop=100.)
    assert photoion.initial_state is initial_state
    assert photoion.termination_step == 2
    assert photoion.termination_reason == 'maxstep'
    assert Path(photoion.traj_file).is_file()
    assert Path(photoion.backfile).is_file()
    assert Path(photoion.preparation_output_dir, 'initial_states.json').is_file()


def test_photoionization_object_hessian_path(backend, tmp_path, monkeypatch):
    from smite import Photoionization
    molecule, config = backend
    molecule.q_ini = np.array([-1., 0., 0., 1., 0., 0.])
    direction = np.array([1., 0., 0., -1., 0., 0.])
    hessian = np.outer(direction, direction) * .02**2 / (1 / MASS[0] + 1 / MASS[1])
    photoion = Photoionization(
        molecule, config, neutra_frankcondon_geom=molecule.q_ini, neutral_hessian=hessian, qct_options={'init_vib_type': 'ZPE'},
        photon_energy=15., level=1, electron_energy=4.)
    monkeypatch.chdir(tmp_path)
    photoion.sample_and_run_dynamics(maxstep=0, Rstop=100., sampling_seed=32)
    assert photoion.initial_state.source_index == 0
    assert photoion.initial_state.electron_energy_ev == 4.
    np.testing.assert_array_equal(photoion.q, photoion.initial_state.q)
    np.testing.assert_array_equal(photoion.p, photoion.initial_state.p)


def test_photoionization_requires_exactly_one_stop_mode(backend):
    from smite import Photoionization
    molecule, config = backend
    photoion = Photoionization(molecule, config, neutra_frankcondon_geom=molecule.q_ini,
                              neutral_hessian=np.eye(6), photon_energy=15.)
    for stop in ({}, {'Rstop': 20., 'pairs_to_stop': {'break': [((0, 1), 'GT', 2.)]}}):
        with pytest.raises(ValueError, match='exactly one'):
            photoion.sample_and_run_dynamics(**stop)


@pytest.mark.parametrize('inputs,match', [
    ({'neutral_traj': 'neutral.xyz'}, 'requires photon_energy'),
    ({'photon_energy': 15.}, 'neutral_hessian must be given'),
    ({'photon_energy': 15., 'neutral_traj': 'neutral.xyz', 'level': 1}, 'requires electron_energy'),
    ({'photon_energy': 15., 'neutral_traj': 'neutral.xyz', 'level': 0, 'ionic_energy': 11.}, 'Level 1'),
])
def test_photoionization_constructor_rejects_incomplete_preparation(backend, inputs, match):
    from smite import Photoionization
    molecule, config = backend
    with pytest.raises(ValueError, match=match):
        Photoionization(molecule, config, neutra_frankcondon_geom=molecule.q_ini, **inputs)


@pytest.mark.parametrize('source', [{'neutral_traj': 'neutral.xyz'}, {'neutral_hessian': np.eye(6)}])
def test_photoionization_geometry_is_required_for_both_sources(backend, source):
    from smite import Photoionization
    molecule, config = backend
    with pytest.raises(ValueError, match='neutra_frankcondon_geom must be given'):
        Photoionization(molecule, config, photon_energy=15., **source)


def test_photoionization_trajectory_takes_precedence_over_hessian(backend, tmp_path, monkeypatch):
    from smite import Photoionization
    molecule, config = backend
    path = tmp_path / 'neutral.xyz'
    write_md(path, [(Q, P)])
    geometry_file = tmp_path / 'reference.xyz'
    shifted = Q + 10.
    geometry_file.write_text('2\nneutral reference\n' + ''.join(
        atom + ' ' + ' '.join(str(x * BOHR_TO_ANGSTROM) for x in row) + '\n'
        for atom, row in zip(molecule.atoms, shifted.reshape(-1, 3))))
    def forbidden(*args, **kwargs):
        raise AssertionError('A supplied trajectory must select MD without reading the Hessian')
    monkeypatch.setattr('photoionization.driver.HessianSource', forbidden)
    photoion = Photoionization(
        molecule, config, neutra_frankcondon_geom=geometry_file,
        neutral_traj=path, neutral_hessian='not_read.hess',
        qct_options={'init_vib_type': 'ZPE'}, photon_energy=15., level=0)
    np.testing.assert_allclose(photoion.q, shifted)
    state = photoion.sample_initial_conditions()
    np.testing.assert_allclose(state.q, Q)
    np.testing.assert_array_equal(state.p, P)
    assert photoion.neutral_hessian is None


def test_photoionization_checks_geometry_atom_order_in_md_mode(backend, tmp_path):
    from smite import Photoionization
    molecule, config = backend
    geometry_file = tmp_path / 'wrong.xyz'
    geometry_file.write_text('2\nwrong order\nHe 0 0 0\nH 1 0 0\n')
    with pytest.raises(ValueError, match='atom identities/order'):
        Photoionization(molecule, config, neutra_frankcondon_geom=geometry_file,
                        neutral_traj='neutral.xyz', photon_energy=15.)


def _photoion_with_neutral_modes(backend):
    from smite import Photoionization
    molecule, config = backend
    reference = np.array([-1., 0., 0., 1., 0., 0.])
    direction = np.array([1., 0., 0., -1., 0., 0.])
    hessian = np.outer(direction, direction) * .02**2 / (1 / MASS[0] + 1 / MASS[1])
    photoion = Photoionization(
        molecule, config, neutra_frankcondon_geom=reference,
        neutral_hessian=hessian, photon_energy=15., level=1, electron_energy=4.)
    return photoion, reference, hessian


def test_photoion_qct_requires_explicit_mode_choices(backend, monkeypatch):
    photoion, _, _ = _photoion_with_neutral_modes(backend)
    def forbidden(*args, **kwargs):
        raise AssertionError('Energy evaluations must not start before mode sampling is configured')
    monkeypatch.setattr('photoionization.driver.Potential_Energy', forbidden)
    with pytest.raises(ValueError, match='Specify_Mode_Sampling'):
        photoion.sample_and_run_dynamics(maxstep=1, Rstop=100.)


def test_photoion_mode_sampling_uses_neutral_quantum_numbers_and_runs(backend, tmp_path, monkeypatch):
    photoion, reference, hessian = _photoion_with_neutral_modes(backend)
    q_before = photoion.neutral.q.copy()
    returned = photoion.Specify_Mode_Sampling(
        init_vib_type='ZPE', init_rot_type='Jfix', jrot=0,
        fix_quantum=[(0, 2)], sampling_seed=481)
    assert returned is photoion
    assert photoion.neutral_vibsampling[0][1:] == ('Q', 2)
    monkeypatch.chdir(tmp_path)
    photoion.sample_and_run_dynamics(maxstep=1, timestep=.001, Rstop=100.)
    centered = reference.reshape(-1, 3) - np.average(reference.reshape(-1, 3), weights=MASS, axis=0)
    state = photoion.initial_state
    dq = state.q - centered.reshape(-1)
    neutral_energy = state.neutral_kinetic_ev / HARTREE_TO_EV + .5 * dq @ hessian @ dq
    assert neutral_energy == pytest.approx(.02 * 2.5, abs=1e-12)
    assert photoion.termination_step == 1
    np.testing.assert_array_equal(photoion.neutral.q, q_before)


def test_photoion_thermal_mode_sampling_and_overrides(backend):
    photoion, _, _ = _photoion_with_neutral_modes(backend)
    photoion.specify_mode_sampling(init_vib_type='Temp', init_rot_type='Temp', temp=300.)
    assert photoion.neutral_vibsampling[0][1:] == ('T', 300.)
    assert photoion.neutral_rotsampling[0] == ('T', 300.)
    state = photoion.sample_initial_conditions(sampling_seed=81)
    assert np.linalg.norm(state.neutral_angular_momentum) > 0.
    assert abs(state.energy_residual_ev) < 1e-10
    photoion.Specify_Mode_Sampling(init_vib_type='Temp', temp=300., fix_energy=[(0, .04)])
    assert photoion.neutral_vibsampling[0][1:] == ('E', .04)
    photoion.Specify_Mode_Sampling(fix_wigner=[0])
    assert photoion.neutral_vibsampling[0][1:] == ('W', 0)


@pytest.mark.parametrize('options,match', [
    ({'init_vib_type': 'Temp'}, 'temp'),
    ({'init_rot_type': 'Temp', 'temp': -5.}, 'temp'),
    ({'fix_quantum': [(1, 0)]}, 'invalid normal-mode index'),
    ({'fix_quantum': [(0, 1.5)]}, 'integer vibrational'),
])
def test_photoion_rejects_invalid_mode_setup_without_losing_previous_settings(backend, options, match):
    photoion, _, _ = _photoion_with_neutral_modes(backend)
    photoion.Specify_Mode_Sampling(fix_quantum=[(0, 3)])
    before = deepcopy(photoion._photoionization_sampling['qct_options'])
    with pytest.raises(ValueError, match=match):
        photoion.Specify_Mode_Sampling(**options)
    assert photoion._photoionization_sampling['qct_options'] == before


def test_photoion_md_does_not_accept_qct_mode_resampling(backend, tmp_path):
    from smite import Photoionization
    molecule, config = backend
    path = tmp_path / 'neutral.xyz'
    write_md(path, [(Q, P)])
    photoion = Photoionization(molecule, config, neutra_frankcondon_geom=Q,
                              neutral_traj=path, photon_energy=15., level=0)
    with pytest.raises(ValueError, match='neutral_traj already supplies'):
        photoion.Specify_Mode_Sampling(init_vib_type='Temp', temp=300.)
    state = photoion.sample_initial_conditions()
    np.testing.assert_array_equal(state.p, P)
