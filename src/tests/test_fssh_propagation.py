"""Propagation of the electronic amplitudes in surface hopping.

The amplitudes evolve under an anti-Hermitian generator, so the exact evolution
is unitary and the electronic norm is a conserved quantity. Hop probabilities
are ratios of populations, so a drifting norm corrupts them silently -- there is
no exception, just wrong branching ratios. These tests pin the conservation law
itself rather than any particular integrator.
"""

import numpy as np
import pytest

from fssh.fssh import _cdot, _effective_hamiltonian, _eval_hop_prob, _update_c


def _system(num_states, couplings, energies, speed=0.02):
    """A coupling array and velocity reproducing the requested ``v . d_ij``."""
    v = np.zeros((1, 3))
    v[0, 0] = speed

    d = np.zeros((num_states, num_states, 1, 3), dtype=complex)
    for (i, j), vd in couplings.items():
        d[i, j, 0, 0] = vd / speed
        d[j, i, 0, 0] = -vd / speed

    return v, d, np.asarray(energies, dtype=float)


def test_effective_hamiltonian_is_hermitian():
    v, d, epot = _system(3, {(0, 1): 0.7, (0, 2): -0.3, (1, 2): 1.1}, [0.0, 0.004, 0.02])
    hamiltonian = _effective_hamiltonian(v, d, 3, epot)

    assert hamiltonian == pytest.approx(hamiltonian.conj().T)
    # The couplings sit in the imaginary part, the energies on the diagonal.
    assert np.diag(hamiltonian).imag == pytest.approx(np.zeros(3))
    assert np.diag(hamiltonian).real == pytest.approx(epot)


@pytest.mark.parametrize("vd", [0.5, 2.0, 8.0])
@pytest.mark.parametrize("dtq", [1.0, 5.0])
def test_norm_is_conserved_to_machine_precision(vd, dtq):
    """Even at a coupling and timestep where Runge-Kutta diverged to NaN."""
    v, d, epot = _system(2, {(0, 1): vd}, [0.0, 0.004])

    c = np.array([np.sqrt(0.9), np.sqrt(0.1)], dtype=complex)
    for _ in range(200):
        c = _update_c(v, c, d, 2, epot, dtq)

    assert np.all(np.isfinite(c))
    assert np.sum(np.abs(c) ** 2) == pytest.approx(1.0, abs=1.0e-12)


def test_norm_is_conserved_for_more_than_two_states():
    v, d, epot = _system(4, {(0, 1): 1.3, (1, 2): -0.8, (2, 3): 0.5, (0, 3): 0.2},
                         [0.0, 0.003, 0.009, 0.015])

    c = np.zeros(4, dtype=complex)
    c[1] = 1.0
    for _ in range(500):
        c = _update_c(v, c, d, 4, epot, 2.0)

    assert np.sum(np.abs(c) ** 2) == pytest.approx(1.0, abs=1.0e-12)


def test_propagation_reproduces_the_equation_of_motion():
    """In the small-step limit the update must match ``cdot`` from the TDSE.

    ``_cdot`` is the literal statement of the adiabatic equation of motion, so
    this is what ties the unitary propagator to the physics it is solving.
    """
    v, d, epot = _system(3, {(0, 1): 0.6, (1, 2): -0.4, (0, 2): 0.1}, [0.0, 0.004, 0.02])

    c = np.array([0.7, 0.5 + 0.2j, 0.3 - 0.1j], dtype=complex)
    c /= np.sqrt(np.sum(np.abs(c) ** 2))

    dtq = 1.0e-6
    explicit = c + dtq * _cdot(v, c, d, 3, epot)
    propagated = _update_c(v, c, d, 3, epot, dtq)

    assert propagated == pytest.approx(explicit, abs=1.0e-11)


def test_matches_scipy_matrix_exponential():
    expm = pytest.importorskip("scipy.linalg").expm
    v, d, epot = _system(2, {(0, 1): 1.7}, [0.0, 0.006])

    c = np.array([np.sqrt(0.6), np.sqrt(0.4)], dtype=complex)
    dtq = 3.0

    reference = expm(-1.0j * _effective_hamiltonian(v, d, 2, epot) * dtq) @ c
    assert _update_c(v, c, d, 2, epot, dtq) == pytest.approx(reference, abs=1.0e-12)


def test_uncoupled_states_only_pick_up_a_phase():
    """With no coupling the populations must not move at all."""
    v, d, epot = _system(2, {}, [0.0, 0.01])

    c = np.array([np.sqrt(0.3), np.sqrt(0.7)], dtype=complex)
    populations = np.abs(c) ** 2

    for _ in range(100):
        c = _update_c(v, c, d, 2, epot, 4.0)

    assert np.abs(c) ** 2 == pytest.approx(populations, abs=1.0e-12)


def test_hop_probabilities_never_exceed_one():
    """The first-order expression is only a probability while it stays below 1.

    Left unclamped, a large ``dtq`` makes the cumulative sum in ``_check_hop``
    exceed unity, which forces a hop every sub-step and destroys the branching
    statistics without any error being raised.
    """
    import fssh.fssh as module

    v, d, epot = _system(3, {(0, 1): 40.0, (0, 2): 25.0}, [0.0, 0.004, 0.01])
    c = np.array([np.sqrt(0.5), np.sqrt(0.3), np.sqrt(0.2)], dtype=complex)

    module._HOP_PROBABILITY_WARNED = False
    with pytest.warns(RuntimeWarning, match="exceeded one"):
        g = _eval_hop_prob(v, c, d, 3, 0, dtq=5.0)

    assert np.all(g >= 0.0)
    assert g.sum() <= 1.0 + 1.0e-12
    assert g[0] == 0.0                       # never onto the active state


def test_rescaling_preserves_the_relative_branching():
    """When the total is rescaled, the ratio between target states must survive."""
    import fssh.fssh as module

    v, d, epot = _system(3, {(0, 1): 40.0, (0, 2): 25.0}, [0.0, 0.004, 0.01])
    c = np.array([np.sqrt(0.5), np.sqrt(0.3), np.sqrt(0.2)], dtype=complex)

    module._HOP_PROBABILITY_WARNED = True          # silence, already covered above
    small = _eval_hop_prob(v, c, d, 3, 0, dtq=1.0e-4)
    large = _eval_hop_prob(v, c, d, 3, 0, dtq=5.0)

    assert small.sum() < 1.0                        # unclamped reference
    assert large.sum() == pytest.approx(1.0)
    assert large[2] / large[1] == pytest.approx(small[2] / small[1], rel=1.0e-9)


def test_probabilities_below_one_are_left_untouched():
    v, d, epot = _system(2, {(0, 1): 0.8}, [0.0, 0.05])
    c = np.array([np.sqrt(0.9), np.sqrt(0.1)], dtype=complex)

    g = _eval_hop_prob(v, c, d, 2, 0, dtq=1.0e-4)
    assert 0.0 < g[1] < 1.0e-3                      # far from the clamp
    assert g.sum() < 1.0


def test_hop_probability_tracks_the_population_it_transfers():
    """g_{a->s} must equal the fractional population actually leaving state a."""
    v, d, epot = _system(2, {(0, 1): 0.8}, [0.0, 0.05])
    dtq = 1.0e-4

    c = np.array([np.sqrt(0.9), np.sqrt(0.1)], dtype=complex)
    for _ in range(5):
        population = np.abs(c[0]) ** 2
        predicted = _eval_hop_prob(v, c, d, 2, 0, dtq)[1]
        c = _update_c(v, c, d, 2, epot, dtq)
        observed = (population - np.abs(c[0]) ** 2) / population

        assert predicted == pytest.approx(observed, rel=1.0e-3)


# --------------------------------------------------------------------------
# Energy-based decoherence correction
# --------------------------------------------------------------------------

from fssh.fssh import LONG_DECAY_TIME, _decoherence


def _amplitudes(populations):
    c = np.sqrt(np.asarray(populations, dtype=float)).astype(complex)
    return c / np.sqrt(np.sum(np.abs(c) ** 2))


def _decay_time(gap, kinetic_energy, de_corr):
    """tau_k = (1 + C / E_kin) / |E_k - E_a|, the Granucci-Persico expression."""
    return (1.0 + de_corr / kinetic_energy) / gap


def test_disabled_by_default_leaves_the_amplitudes_untouched():
    c = _amplitudes([0.7, 0.3])
    mass = np.full((2, 3), 1000.0)
    v = np.full((2, 3), 0.01)

    unchanged = _decoherence(c, np.array([0.0, 0.05]), v, mass, 0, 1.0, de_corr=0.0)
    assert unchanged is c or unchanged == pytest.approx(c)


def test_inactive_population_follows_the_analytic_decay():
    """|c_k|^2 must fall as exp(-2 dtq / tau_k), not exp(-dtq / tau_k)."""
    de_corr, dtq = 0.1, 2.0
    epot = np.array([0.0, 0.05])
    mass = np.full((3, 3), 1000.0)
    v = np.full((3, 3), 0.01)
    kinetic_energy = 0.5 * float(np.sum(mass * v * v))

    c = _amplitudes([0.7, 0.3])
    before = float(np.abs(c[1]) ** 2)

    after = _decoherence(c, epot, v, mass, 0, dtq, de_corr)

    tau = _decay_time(abs(epot[1] - epot[0]), kinetic_energy, de_corr)
    assert float(np.abs(after[1]) ** 2) == pytest.approx(
        before * np.exp(-2.0 * dtq / tau), rel=1.0e-12
    )


def test_the_trace_is_restored_to_one():
    c = _amplitudes([0.5, 0.3, 0.2])
    mass = np.full((4, 3), 1000.0)
    v = np.full((4, 3), 0.02)

    after = _decoherence(c, np.array([0.0, 0.04, 0.09]), v, mass, 0, 3.0, 0.1)
    assert float(np.sum(np.abs(after) ** 2)) == pytest.approx(1.0, abs=1.0e-12)


def test_population_lost_by_inactive_states_lands_on_the_active_one():
    c = _amplitudes([0.5, 0.3, 0.2])
    mass = np.full((4, 3), 1000.0)
    v = np.full((4, 3), 0.02)
    before = np.abs(c) ** 2

    after = np.abs(_decoherence(c, np.array([0.0, 0.04, 0.09]), v, mass, 0, 3.0, 0.1)) ** 2

    assert after[1] < before[1] and after[2] < before[2]
    assert after[0] > before[0]
    assert after[0] - before[0] == pytest.approx(
        (before[1] - after[1]) + (before[2] - after[2]), abs=1.0e-12
    )


@pytest.mark.parametrize("gap, faster", [(0.20, True), (0.01, False)])
def test_states_further_apart_in_energy_decohere_faster(gap, faster):
    c = _amplitudes([0.6, 0.4])
    mass = np.full((3, 3), 1000.0)
    v = np.full((3, 3), 0.01)

    wide = np.abs(_decoherence(c, np.array([0.0, 0.20]), v, mass, 0, 2.0, 0.1)[1]) ** 2
    narrow = np.abs(_decoherence(c, np.array([0.0, 0.01]), v, mass, 0, 2.0, 0.1)[1]) ** 2
    assert wide < narrow


def test_faster_nuclei_decohere_more_quickly():
    """tau = (1 + C/E_kin)/|dE|, so a large E_kin shrinks tau.

    Slow nuclei give a large C/E_kin, a long decay time and almost no
    decoherence; in the limit of a motionless system tau diverges. Fast nuclei
    approach the bare hbar/|dE|, the shortest the correction allows.
    """
    c = _amplitudes([0.6, 0.4])
    epot = np.array([0.0, 0.05])
    mass = np.full((3, 3), 1000.0)

    slow = np.abs(_decoherence(c, epot, np.full((3, 3), 0.005), mass, 0, 2.0, 0.1)[1]) ** 2
    fast = np.abs(_decoherence(c, epot, np.full((3, 3), 0.05), mass, 0, 2.0, 0.1)[1]) ** 2

    assert fast < slow                       # more population lost when fast
    assert slow < np.abs(c[1]) ** 2          # but both do decay


@pytest.mark.parametrize(
    "epot, v",
    [
        (np.array([0.0, 0.0]), np.full((3, 3), 0.01)),      # degenerate states
        (np.array([0.0, 0.05]), np.zeros((3, 3))),          # motionless nuclei
    ],
)
def test_a_diverging_decay_time_is_guarded(epot, v):
    """A vanishing gap or vanishing kinetic energy must not divide by zero."""
    c = _amplitudes([0.6, 0.4])
    mass = np.full((3, 3), 1000.0)

    after = _decoherence(c, epot, v, mass, 0, 2.0, 0.1)

    assert np.all(np.isfinite(after))
    assert float(np.sum(np.abs(after) ** 2)) == pytest.approx(1.0, abs=1.0e-12)
    # The substituted time is long enough that a normal step changes nothing.
    assert np.abs(after[1]) ** 2 == pytest.approx(
        np.abs(c[1]) ** 2 * np.exp(-4.0 / LONG_DECAY_TIME), rel=1.0e-9
    )


def test_repeated_application_drives_the_active_state_to_one():
    """Over many steps the coherences are destroyed, which is the whole point."""
    c = _amplitudes([0.5, 0.5])
    epot = np.array([0.0, 0.05])
    mass = np.full((3, 3), 1000.0)
    v = np.full((3, 3), 0.01)

    for _ in range(400):
        c = _decoherence(c, epot, v, mass, 0, 2.0, 0.1)

    assert float(np.abs(c[0]) ** 2) == pytest.approx(1.0, abs=1.0e-3)
    assert float(np.sum(np.abs(c) ** 2)) == pytest.approx(1.0, abs=1.0e-12)


def test_the_amplitude_form_matches_the_density_matrix_transformation():
    """The correction is defined on rho; we apply it to c. They must agree.

    Granucci and Persico write the correction as a transformation of the density
    matrix. This writes that transformation out element by element -- including
    the sqrt(rho_aa' / rho_aa) factor the coherences between the active state
    and the rest pick up -- and checks that the amplitude form reproduces it.
    """
    de_corr, dtq, active = 0.1, 2.0, 0
    epot = np.array([0.0, 0.04, 0.09])
    mass = np.full((4, 3), 1000.0)
    v = np.full((4, 3), 0.02)
    kinetic_energy = 0.5 * float(np.sum(mass * v * v))

    c = np.array([0.6, 0.5 + 0.2j, 0.3 - 0.4j], dtype=complex)
    c /= np.sqrt(np.sum(np.abs(c) ** 2))
    rho = np.outer(c, c.conj())

    tau = {
        k: _decay_time(abs(epot[k] - epot[active]), kinetic_energy, de_corr)
        for k in range(3) if k != active
    }

    # The density-matrix form of the correction, written out explicitly.
    expected = rho.copy()
    for k in tau:
        expected[k, k] = rho[k, k] * np.exp(-2.0 * dtq / tau[k])
    expected[active, active] = 1.0 - sum(expected[k, k] for k in tau)
    for k in tau:
        for l in tau:
            if k != l:
                expected[k, l] = rho[k, l] * np.exp(-dtq / tau[k]) * np.exp(-dtq / tau[l])
    scale = np.sqrt(expected[active, active] / rho[active, active])
    for k in tau:
        expected[k, active] = rho[k, active] * np.exp(-dtq / tau[k]) * scale
        expected[active, k] = rho[active, k] * np.exp(-dtq / tau[k]) * scale

    corrected = _decoherence(c, epot, v, mass, active, dtq, de_corr)
    assert np.outer(corrected, corrected.conj()) == pytest.approx(expected, abs=1.0e-12)
