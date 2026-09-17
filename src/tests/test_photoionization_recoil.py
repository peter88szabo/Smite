"""Levels 2 and 3: momentum-conserving rescaling and photoelectron recoil.

These assert the *defining properties* of each level rather than merely that the
code runs: Level 2 exists to conserve linear and angular momentum exactly, and
Level 3 exists to give the ion the momentum the photon brought in and the
electron carried away. If those two statements hold to machine precision the
levels are doing their job.
"""

import numpy as np
import pytest

from photoionization import prepare_ionic_state
from photoionization.preparation import eckart_split, solve_internal_scale
from photoionization.recoil import (
    distribute_recoil,
    photon_momentum,
    recoil_weights,
    sample_electron_momentum,
)
from utils.constants import AU_SPEED_OF_LIGHT, HARTREE_TO_EV

U = 1822.88848628

# Water-like, bent, so the internal subspace has 3N - 6 = 3 dimensions.
WATER_MASS = np.array([15.999, 1.008, 1.008]) * U
WATER_Q = np.array([0.0, 0.0, 0.22, 0.0, 1.43, -0.88, 0.0, -1.43, -0.88])
# Deliberately carries net translation AND rotation, which is what Level 1 breaks.
WATER_P = np.array([3.0, -1.5, 2.0, -0.7, 0.9, -1.1, 1.2, 0.4, 0.6])

GAP_EV = 10.0
NEUTRAL_V = -76.0
IONIC_V = NEUTRAL_V + GAP_EV / HARTREE_TO_EV


def prepare(level, q=WATER_Q, p=WATER_P, mass=WATER_MASS, **kwargs):
    args = dict(photon_energy=15.0, ionic_energy=11.0, seed=17)
    if level == 0:
        args.pop("ionic_energy")
    args.update(kwargs)
    return prepare_ionic_state(q, p, mass, NEUTRAL_V, IONIC_V, level=level, **args)


def linear_momentum(p):
    return p.reshape(-1, 3).sum(axis=0)


def angular_momentum(q, p, mass):
    xyz = q.reshape(-1, 3)
    centered = xyz - np.average(xyz, axis=0, weights=mass)
    return np.cross(centered, p.reshape(-1, 3)).sum(axis=0)


# ---------------------------------------------------------------------------
# Level 2: the rescale must not touch P or L
# ---------------------------------------------------------------------------

def test_level_two_conserves_linear_and_angular_momentum_exactly():
    """The defining property. Level 1 fails this; that is why Level 2 exists."""
    state = prepare(2)

    np.testing.assert_allclose(linear_momentum(state.p), linear_momentum(WATER_P), atol=1e-12)
    np.testing.assert_allclose(angular_momentum(WATER_Q, state.p, WATER_MASS),
                               angular_momentum(WATER_Q, WATER_P, WATER_MASS), atol=1e-12)
    assert state.scaled_subspace == "internal"
    assert state.energy_residual_ev == pytest.approx(0.0, abs=1e-10)


def test_level_one_does_not_conserve_them_which_is_the_defect_being_fixed():
    state = prepare(1)
    assert not np.allclose(linear_momentum(state.p), linear_momentum(WATER_P), atol=1e-6)
    assert state.scaled_subspace == "all"


def test_level_two_energy_lands_on_the_target():
    """T_i = T_n + hv - eps - dV, assembled from projected pieces."""
    state = prepare(2)
    expected = state.neutral_kinetic_ev + 15.0 - state.electron_energy_ev - GAP_EV
    assert state.ionic_kinetic_ev == pytest.approx(expected, abs=1e-9)
    # The split is orthogonal, so the parts must add up.
    assert (state.external_kinetic_ev + state.internal_kinetic_ev
            == pytest.approx(state.ionic_kinetic_ev, abs=1e-9))


def test_level_two_matches_level_one_when_there_is_nothing_external_to_protect():
    """With P = L = 0 the whole momentum is internal, so both levels agree."""
    _, a_int = eckart_split(WATER_Q, WATER_P, WATER_MASS)
    p_internal = a_int * np.sqrt(np.repeat(WATER_MASS, 3))

    one = prepare(1, p=p_internal)
    two = prepare(2, p=p_internal)
    np.testing.assert_allclose(two.p, one.p, atol=1e-10)


def test_level_two_handles_a_linear_molecule():
    """rank(R) = 5 there; the pseudoinverse covers it with no special case."""
    mass = np.array([12.0, 15.999]) * U
    q = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 1.14])
    p = np.array([0.5, 0.3, 2.0, -0.2, 0.1, -1.4])

    state = prepare(2, q=q, p=p, mass=mass)
    np.testing.assert_allclose(linear_momentum(state.p), linear_momentum(p), atol=1e-12)
    np.testing.assert_allclose(angular_momentum(q, state.p, mass),
                               angular_momentum(q, p, mass), atol=1e-12)


def test_monatomic_ion_has_no_internal_subspace_to_rescale():
    with pytest.raises(ValueError, match="No internal .vibrational. motion"):
        prepare(2, q=np.zeros(3), p=np.array([1.0, 2.0, 3.0]), mass=np.array([40.0 * U]))


# ---------------------------------------------------------------------------
# Level 3: the ion must absorb exactly the photon minus the electron
# ---------------------------------------------------------------------------

def test_level_three_shifts_linear_momentum_by_exactly_the_recoil():
    state = prepare(3)
    delta_p = state.photon_momentum - state.electron_momentum

    np.testing.assert_allclose(linear_momentum(state.p),
                               linear_momentum(WATER_P) + delta_p, atol=1e-12)


def test_mass_weighted_recoil_leaves_angular_momentum_untouched():
    r"""dL = (sum_i w_i x_i) x dp, and the mass-weighted centroid IS the COM."""
    state = prepare(3, recoil_site="com")
    np.testing.assert_allclose(angular_momentum(WATER_Q, state.p, WATER_MASS),
                               angular_momentum(WATER_Q, WATER_P, WATER_MASS), atol=1e-12)


def test_a_localised_core_hole_torques_the_molecule():
    r"""dL = x_j x dp when all the recoil lands on atom j."""
    state = prepare(3, recoil_site=1)
    delta_p = state.photon_momentum - state.electron_momentum

    xyz = WATER_Q.reshape(-1, 3)
    centered = xyz - np.average(xyz, axis=0, weights=WATER_MASS)
    expected = angular_momentum(WATER_Q, WATER_P, WATER_MASS) + np.cross(centered[1], delta_p)

    np.testing.assert_allclose(angular_momentum(WATER_Q, state.p, WATER_MASS), expected, atol=1e-12)


def test_recoil_induced_internal_excitation_survives_the_rescale():
    """The test that catches a naive implementation.

    Rescaling the *combined* internal momentum would wipe out the vibrational
    excitation the recoil deposits. Mass-weighted recoil is purely translational
    and must deposit exactly none; a localised hole must deposit some.
    """
    delocalised = prepare(3, recoil_site="com")
    localised = prepare(3, recoil_site=1)

    assert delocalised.recoil_internal_kinetic_ev == pytest.approx(0.0, abs=1e-18)
    assert localised.recoil_internal_kinetic_ev > 0.0
    # Same seed, same electron -- so any difference is the site, not the sampling.
    np.testing.assert_allclose(localised.electron_momentum, delocalised.electron_momentum)


def test_level_three_still_closes_the_energy_balance():
    for site in ("com", 0, 1, [0.5, 0.25, 0.25]):
        state = prepare(3, recoil_site=site)
        assert state.energy_residual_ev == pytest.approx(0.0, abs=1e-10)


def test_recoil_is_recorded_only_when_it_was_applied():
    assert prepare(2).recoil_momentum is None
    assert prepare(3).recoil_momentum is not None
    assert prepare(0).scaled_subspace == "none"


def test_same_seed_reproduces_the_whole_launch():
    np.testing.assert_array_equal(prepare(3).p, prepare(3).p)
    assert not np.array_equal(prepare(3, seed=17).p, prepare(3, seed=18).p)


# ---------------------------------------------------------------------------
# The recoil kernel on its own
# ---------------------------------------------------------------------------

def test_electron_momentum_magnitude_is_sqrt_two_epsilon():
    rng = np.random.default_rng(0)
    p_e = sample_electron_momentum(rng, 700.0)
    assert np.linalg.norm(p_e) == pytest.approx(np.sqrt(2.0 * 700.0 / HARTREE_TO_EV))


def test_photon_momentum_is_energy_over_c():
    p_g = photon_momentum(1000.0, (0.0, 0.0, 5.0))
    assert np.linalg.norm(p_g) == pytest.approx((1000.0 / HARTREE_TO_EV) / AU_SPEED_OF_LIGHT)
    np.testing.assert_allclose(p_g / np.linalg.norm(p_g), [0.0, 0.0, 1.0])


def test_isotropic_sampling_has_no_preferred_direction():
    rng = np.random.default_rng(3)
    directions = np.array([sample_electron_momentum(rng, 10.0) for _ in range(4000)])
    mean = directions.mean(axis=0) / np.linalg.norm(directions[0])
    assert np.abs(mean).max() < 0.05


def test_recoil_energy_matches_the_closed_form_in_the_delocalised_limit():
    r"""E_rec = |p_e|^2 / 2M = (m_e/M) eps, the textbook recoil shift."""
    rng = np.random.default_rng(5)
    epsilon = 2700.0
    p_e = sample_electron_momentum(rng, epsilon)
    weights = recoil_weights("com", WATER_MASS)
    p_rec = distribute_recoil(-p_e, weights)

    energy = 0.5 * np.sum(p_rec * p_rec / np.repeat(WATER_MASS, 3)) * HARTREE_TO_EV
    assert energy == pytest.approx(epsilon / WATER_MASS.sum() * 1.0, rel=1e-10)
    assert energy == pytest.approx(0.0822, abs=5e-4)          # ~82 meV at 2.7 keV


def test_weights_always_sum_to_one_so_total_momentum_is_exact():
    for site in ("com", 0, 2, [3.0, 1.0, 1.0]):
        assert recoil_weights(site, WATER_MASS).sum() == pytest.approx(1.0)


@pytest.mark.parametrize("site, match", [
    (7, "outside 0..2"),
    ("dyson", "Unknown recoil_site"),
    ([1.0, -1.0, 1.0], "nonnegative"),
    ([0.0, 0.0, 0.0], "sum to a positive"),
    ([1.0, 1.0], "3 finite values"),
])
def test_invalid_recoil_sites_are_rejected(site, match):
    with pytest.raises(ValueError, match=match):
        recoil_weights(site, WATER_MASS)


def test_anisotropic_emission_is_refused_rather_than_silently_ignored():
    with pytest.raises(ValueError, match="not implemented"):
        sample_electron_momentum(np.random.default_rng(0), 10.0, beta=2.0)


# ---------------------------------------------------------------------------
# Through the driver, not just the kernel
# ---------------------------------------------------------------------------

@pytest.fixture
def backend(tmp_path):
    """Toy PES whose ionic surface sits exactly 10 eV above the neutral."""
    from core.molecule import Molecule

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
    mass = np.array([1000.0, 1500.0])
    q = np.array([-1.0, 0.2, 0.3, 1.0, -0.1, 0.4])
    p = np.array([2.0, -0.3, 0.7, -0.4, 0.8, -0.2])
    molecule = Molecule(["H", "He"], mass, q, p)
    molecule.qchem = {"qchem": "PES", "pes_path": str(pes), "charge": 0, "multiplicity": 1}
    return molecule, dict(molecule.qchem, charge=1, multiplicity=2), q, p, mass


def test_level_two_through_the_driver_conserves_momentum(backend):
    from photoionization.driver import tpepico

    molecule, ion, q, p, mass = backend
    result = tpepico(molecule, ion, photon_energy=15.0, level=2, ionic_energy=11.0,
                     neutral_hessian=np.eye(6) * 0.002, seed=11)
    state = result.initial_states[0]

    np.testing.assert_allclose(state.ionic_linear_momentum, state.neutral_linear_momentum, atol=1e-12)
    np.testing.assert_allclose(state.ionic_angular_momentum, state.neutral_angular_momentum, atol=1e-12)
    assert state.scaled_subspace == "internal"


def test_level_three_through_the_driver_records_the_recoil(backend, tmp_path):
    """Also exercises json.dump(..., allow_nan=False) on the new fields."""
    import json
    from photoionization.driver import tpepico

    molecule, ion, q, p, mass = backend
    out = tmp_path / "run"
    result = tpepico(molecule, ion, photon_energy=15.0, level=3, ionic_energy=11.0,
                     neutral_hessian=np.eye(6) * 0.002, seed=11, recoil_site=0,
                     photon_direction=(1.0, 0.0, 0.0), output_dir=str(out))

    metadata = json.loads((out / "initial_states.json").read_text())
    assert metadata["recoil_included"] is True
    assert metadata["recoil_site"] == 0

    state = result.initial_states[0]
    delta_p = np.asarray(state.photon_momentum) - np.asarray(state.electron_momentum)
    np.testing.assert_allclose(state.ionic_linear_momentum,
                               state.neutral_linear_momentum + delta_p, atol=1e-12)
    # All the recoil is on atom 0, so it must torque the molecule.
    assert not np.allclose(state.ionic_angular_momentum, state.neutral_angular_momentum, atol=1e-9)


@pytest.mark.parametrize("kwargs, match", [
    ({"level": 2, "recoil_site": 0}, "recoil_site applies to Level 3"),
    ({"level": 1, "photon_direction": (1.0, 0.0, 0.0)}, "photon_direction applies to Level 3"),
    ({"level": 0, "recoil_site": 1, "ionic_energy": None}, "recoil_site applies to Level 3"),
])
def test_recoil_arguments_are_refused_below_level_three(backend, kwargs, match):
    """Accepting them silently would imply recoil was modelled when it was not."""
    from photoionization.driver import tpepico

    molecule, ion, _, _, _ = backend
    args = dict(photon_energy=15.0, ionic_energy=11.0,
                neutral_hessian=np.eye(6) * 0.002, seed=11)
    args.update(kwargs)
    if args.get("ionic_energy") is None:
        args.pop("ionic_energy")
    with pytest.raises(ValueError, match=match):
        tpepico(molecule, ion, **args)


# ---------------------------------------------------------------------------
# The quadratic
# ---------------------------------------------------------------------------

def test_without_recoil_the_quadratic_collapses_to_the_level_one_form():
    r"""B = C = 0 gives s = sqrt(T_target / T_int)."""
    a_n = np.array([0.3, -0.4, 0.5])
    zero = np.zeros(3)
    t_n = 0.5 * a_n @ a_n
    s = solve_internal_scale(a_n, zero, 4.0 * t_n, 1e-12)
    assert s == pytest.approx(2.0)


def test_the_quadratic_hits_the_requested_energy():
    a_n = np.array([0.3, -0.4, 0.5])
    a_r = np.array([0.05, 0.02, -0.01])
    target = 0.25
    s = solve_internal_scale(a_n, a_r, target, 1e-12)
    combined = s * a_n + a_r
    assert 0.5 * combined @ combined == pytest.approx(target)


def test_a_negative_discriminant_is_a_physical_rejection():
    """The recoil alone deposits more internal energy than the budget allows."""
    a_n = np.array([1.0e-6, 0.0, 0.0])
    a_r = np.array([0.0, 0.5, 0.0])
    with pytest.raises(ValueError, match="less internal energy than the recoil"):
        solve_internal_scale(a_n, a_r, 1.0e-6, 1e-12)


def test_a_negative_target_names_its_own_cause():
    with pytest.raises(ValueError, match="exceeds the ionic energy budget"):
        solve_internal_scale(np.array([1.0, 0.0, 0.0]), np.zeros(3), -1.0, 1e-12)


def test_the_scale_may_go_negative_when_the_recoil_overshoots():
    """Permitted, not rejected: the sign of an internal momentum has no meaning
    in a Franck-Condon ensemble, where +a and -a are equally represented."""
    a_n = np.array([1.0, 0.0, 0.0])
    a_r = np.array([2.0, 0.0, 0.0])
    s = solve_internal_scale(a_n, a_r, 0.5, 1e-12)
    assert s < 0.0
    combined = s * a_n + a_r
    assert 0.5 * combined @ combined == pytest.approx(0.5)
