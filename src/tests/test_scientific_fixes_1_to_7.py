import math

import numpy as np
import pytest

import dynamics.thermostat_driver as thermostat_driver
import integrators.sprk as sprk_module
import integrators.stormerverlet as stormer_module
import sampling.morse_diatom as morse_module
import thermostats.andersen as andersen_module
from core.molecule import Molecule
from normalmode.thermochemistry import (
    AMU_TO_ELECMASS,
    RGAS_AU,
    eval_thermo,
)
from sampling.morse_diatom import (
    enumerate_bound_morse_states,
    morse_action_for_energy,
    morse_energy,
    sample_thermal_morse_state,
)
from thermostats.gle import GLEThermostat, cholesky_stabilized
from thermostats.nosehoover import NoseHoover
from utils.atomic_masses import get_mass_vector
from utils.constants import (
    ANGSTROM_TO_BOHR,
    BOHR_TO_ANGSTROM,
    FS_TO_AU_TIME,
    KJMOL_TO_HARTREE,
)


def test_gle_covariance_square_root_reconstructs_correlated_matrix():
    covariance = np.array([[4.0, 2.0], [2.0, 3.0]])
    factor = cholesky_stabilized(covariance)
    assert factor @ factor.T == pytest.approx(covariance, abs=1.0e-13)

    with pytest.raises(ValueError, match="not positive semidefinite"):
        cholesky_stabilized(np.array([[1.0, 2.0], [2.0, 1.0]]))


def _harmonic_position_error(stepper, timestep):
    q = np.array([1.0])
    p = np.array([0.0])
    mass = np.ones(1)
    for _ in range(round(1.0 / timestep)):
        q, p = stepper(timestep, mass, q, p)
    return abs(float(q[0]) - math.cos(1.0))


def test_stormer_and_sprk2_have_second_order_convergence(monkeypatch):
    monkeypatch.setattr(stormer_module, "force_calc", lambda _qc, q, _atoms: -q)
    monkeypatch.setattr(sprk_module, "force_calc", lambda _qc, q, _atoms: -q)

    stormer = lambda h, mass, q, p: stormer_module.stormer_verlet(
        None, h, mass, q, p, None
    )
    sprk2_state = sprk_module.SPRK(2)
    sprk2 = lambda h, mass, q, p: sprk2_state.sprk(
        None, h, mass, q, p, None
    )

    for stepper in (stormer, sprk2):
        coarse = _harmonic_position_error(stepper, 0.05)
        fine = _harmonic_position_error(stepper, 0.025)
        assert coarse / fine == pytest.approx(4.0, rel=0.02)


def test_nfix_is_a_count_not_a_cartesian_suffix():
    nose = NoseHoover(
        nfix=3,
        wmass=np.ones(6),
        tau=100.0,
        target_temp=300.0,
    )
    momenta = np.arange(1.0, 7.0)
    propagated = nose.step(momenta.copy(), dt=1.0)
    scale = propagated / momenta
    assert scale == pytest.approx(np.full(6, scale[0]))
    assert not np.isclose(scale[0], 1.0)

    gle = GLEThermostat.__new__(GLEThermostat)
    gle.ns = 0
    gle.gp = np.zeros((6, 1))
    gle.gT = np.zeros((1, 1))
    gle.gS = np.zeros((1, 1))
    gle.rng = np.random.default_rng(7)
    propagated = gle.step(np.arange(1.0, 7.0), np.ones(6), nfix=3)
    assert propagated == pytest.approx(np.zeros(6))


def test_com_projection_and_fragment_dof_count_are_physical():
    molecule = Molecule(
        atoms=["H", "He"],
        mass=np.array([1.0, 4.0]),
        q_ini=np.zeros(6),
        p_ini=np.array([2.0, -1.0, 3.0, 5.0, 7.0, -2.0]),
    )
    molecule.remove_com = True
    molecule._nfix_includes_com = False

    assert molecule.thermostat_removed_dof() == 3
    molecule.project_center_of_mass_momentum()
    total_momentum = molecule.p.reshape((-1, 3)).sum(axis=0)
    assert total_momentum == pytest.approx(np.zeros(3), abs=1.0e-14)


def test_andersen_collisions_are_independent_per_atom(monkeypatch):
    draws = iter([0.1, 0.9])
    monkeypatch.setattr(
        andersen_module.random, "uniform", lambda _low, _high: next(draws)
    )
    monkeypatch.setattr(
        andersen_module.random,
        "normalvariate",
        lambda mu, sigma: 1.0,
    )

    initial = np.arange(1.0, 7.0)
    propagated = andersen_module.thermo_andersen(
        3,
        initial.copy(),
        np.ones(6),
        dt=1.0,
        collision_time=1.0,
        Ttarg=300.0,
    )
    assert not np.allclose(propagated[:3], initial[:3])
    assert propagated[3:] == pytest.approx(initial[3:])


def test_andersen_public_parameter_is_converted_from_fs(monkeypatch):
    captured = {}

    def fake_andersen(nfix, p, wmass, dt, collision_time, temperature):
        captured["nfix"] = nfix
        captured["collision_time"] = collision_time
        return p

    monkeypatch.setattr(thermostat_driver, "thermo_andersen", fake_andersen)
    molecule = Molecule(
        atoms=["H", "H"],
        mass=np.ones(2),
        q_ini=np.zeros(6),
        p_ini=np.zeros(6),
    )
    thermostat_driver.apply_thermostat(
        molecule,
        "andersen",
        thermo_param=2.0,
        thermo_temp=300.0,
        dt=10.0,
    )
    assert captured["collision_time"] == pytest.approx(2.0 * FS_TO_AU_TIME)


def test_atomic_rotation_and_rotational_symmetry_thermochemistry():
    temperature = 298.15
    atom = eval_thermo(
        [],
        [0.0, 0.0, 0.0],
        39.948,
        1,
        temperature,
        101325.0,
        50.0,
    )
    assert atom.pfrot == 1.0
    assert atom.urot == 0.0
    assert atom.srot == 0.0
    assert atom.cprot == 0.0

    sigma_one = eval_thermo(
        [],
        [10.0, 20.0, 30.0],
        18.0,
        1,
        temperature,
        101325.0,
        50.0,
        symmetry_number=1.0,
    )
    sigma_two = eval_thermo(
        [],
        [10.0, 20.0, 30.0],
        18.0,
        1,
        temperature,
        101325.0,
        50.0,
        symmetry_number=2.0,
    )
    assert sigma_two.pfrot / sigma_one.pfrot == pytest.approx(0.5)
    assert (sigma_two.grot - sigma_one.grot) == pytest.approx(
        RGAS_AU * temperature * math.log(2.0)
    )
    assert AMU_TO_ELECMASS == pytest.approx(1822.88848628)


def _hcl_morse_parameters():
    masses = get_mass_vector(["H", "Cl"])
    reduced_mass = masses[0] * masses[1] / np.sum(masses)
    re = 1.2746 * ANGSTROM_TO_BOHR
    beta = 1.867 * BOHR_TO_ANGSTROM
    dissociation_energy = 431.0 * KJMOL_TO_HARTREE
    return reduced_mass, beta, dissociation_energy, re


def test_fixed_morse_energy_is_inverted_exactly():
    reduced_mass, beta, dissociation_energy, re = _hcl_morse_parameters()
    requested_energy = 0.08
    jrot = 3.0
    nvib = morse_action_for_energy(
        reduced_mass,
        requested_energy,
        jrot,
        beta,
        dissociation_energy,
        re,
    )
    recovered = morse_energy(
        reduced_mass,
        nvib,
        jrot,
        beta,
        dissociation_energy,
        re,
    )
    assert recovered == pytest.approx(requested_energy, abs=1.0e-13)


def test_thermal_morse_weights_use_coupled_energies_and_degeneracies(monkeypatch):
    reduced_mass, beta, dissociation_energy, re = _hcl_morse_parameters()
    temperature = 500.0
    states = enumerate_bound_morse_states(
        reduced_mass, beta, dissociation_energy, re
    )
    rt = morse_module.R_GAS_HARTREE_PER_K * temperature
    log_weights = np.array(
        [math.log(state[3]) - state[2] / rt for state in states]
    )
    expected_total = float(np.exp(log_weights - np.max(log_weights)).sum())
    captured = {}

    def choose_first(low, high):
        captured["low"] = low
        captured["high"] = high
        return low

    monkeypatch.setattr(morse_module.random, "uniform", choose_first)
    sampled = sample_thermal_morse_state(
        temperature, reduced_mass, beta, dissociation_energy, re
    )
    assert captured["low"] == 0.0
    assert captured["high"] == pytest.approx(expected_total)
    assert sampled == pytest.approx(states[0][:3])

