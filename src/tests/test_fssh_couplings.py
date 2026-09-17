"""Construction of the coupling matrix, for any number of states.

``de_cutoff`` is a *per-pair* gap filter: states i and j are coupled only when
|E_j - E_i| is below it. With two states there is a single pair, so a global
check and a per-pair check coincide and any confusion between them is invisible.
These tests use three and more states, where they do not coincide.

The curvature approximation does not decay where there is no genuine avoided
crossing, so a coupling handed to a well-separated pair does not stay harmlessly
small -- it drives spurious hopping. That is what the cutoff exists to prevent.
"""

import numpy as np
import pytest

import fssh.fssh as fssh_module
from fssh.fssh import _calculate_diff_energy, _get_couplings


N_ATOMS = 2
Q = np.array([0.0, 0.0, 0.0, 1.4, 0.0, 0.0])


@pytest.fixture
def surfaces(monkeypatch):
    """Install gradients and Hessians for ``num_states`` and count the calls."""
    calls = {"force": 0, "hessian": 0}

    def install(num_states, seed=0):
        rng = np.random.default_rng(seed)
        grad = rng.normal(0.0, 0.05, (num_states, N_ATOMS * 3))
        hess = rng.normal(0.0, 0.02, (num_states, N_ATOMS * 3, N_ATOMS * 3))
        hess = 0.5 * (hess + np.transpose(hess, (0, 2, 1)))

        def force(q, states):
            calls["force"] += 1
            return -grad.ravel()

        def hessian(q, states):
            calls["hessian"] += 1
            return hess.ravel()

        monkeypatch.setattr(fssh_module, "PES_Force", force)
        monkeypatch.setattr(fssh_module, "PES_Hessian", hessian)
        return calls

    return install


def pair_norms(d, num_states):
    return {(i, j): float(np.linalg.norm(d[i, j]))
            for i in range(num_states) for j in range(i + 1, num_states)}


# ---------------------------------------------------------------------------
# The regression
# ---------------------------------------------------------------------------

def test_a_close_pair_does_not_couple_the_distant_ones(surfaces):
    """States 0 and 1 nearly degenerate, state 2 two hartree away.

    Before this was fixed, the presence of the close 0-1 pair caused couplings
    to be handed to 0-2 and 1-2 as well, whose gaps are twenty times the cutoff.
    """
    surfaces(3)
    epot = np.array([0.00, 0.01, 2.01])

    norms = pair_norms(_get_couplings(Q, epot, 3, de_cutoff=0.1), 3)

    assert norms[(0, 1)] > 0.0          # genuinely close: coupled
    assert norms[(0, 2)] == 0.0         # gap 2.01 Ha: must not be
    assert norms[(1, 2)] == 0.0         # gap 2.00 Ha: must not be


@pytest.mark.parametrize("num_states", [2, 3, 4, 5, 6])
def test_only_pairs_inside_the_cutoff_are_ever_coupled(surfaces, num_states):
    """The defining property, at every state count."""
    surfaces(num_states)
    # A tight cluster at the bottom, the rest spread far above it.
    epot = np.concatenate([np.array([0.0, 0.02]),
                           0.5 + np.arange(num_states - 2) * 1.5])[:num_states]
    cutoff = 0.1

    d = _get_couplings(Q, epot, num_states, de_cutoff=cutoff)
    for (i, j), norm in pair_norms(d, num_states).items():
        if abs(epot[j] - epot[i]) >= cutoff:
            assert norm == 0.0, f"pair {i}-{j} is outside the cutoff but coupled"


@pytest.mark.parametrize("num_states", [2, 3, 4, 5])
def test_the_matrix_is_antisymmetric_at_every_state_count(surfaces, num_states):
    surfaces(num_states)
    epot = np.linspace(0.0, 0.05, num_states)
    d = _get_couplings(Q, epot, num_states, de_cutoff=0.5)

    np.testing.assert_allclose(d, -np.transpose(d, (1, 0, 2, 3)), atol=0.0)
    for i in range(num_states):
        assert np.all(d[i, i] == 0.0)


def test_widening_the_cutoff_only_ever_adds_pairs(surfaces):
    """Monotonicity: nothing that was coupled becomes uncoupled."""
    surfaces(4)
    epot = np.array([0.0, 0.02, 0.30, 1.20])

    previous = None
    for cutoff in (0.05, 0.1, 0.4, 2.0):
        norms = pair_norms(_get_couplings(Q, epot, 4, de_cutoff=cutoff), 4)
        coupled = {pair for pair, norm in norms.items() if norm > 0.0}
        if previous is not None:
            assert previous <= coupled
        previous = coupled


# ---------------------------------------------------------------------------
# The cost gate, which is a separate concern from the filter
# ---------------------------------------------------------------------------

def test_no_gradient_or_hessian_is_computed_when_nothing_is_close(surfaces):
    """The expensive part is skipped entirely, not computed and discarded."""
    calls = surfaces(3)
    epot = np.array([0.0, 1.0, 2.0])

    d = _get_couplings(Q, epot, 3, de_cutoff=0.1)

    assert calls == {"force": 0, "hessian": 0}
    assert np.all(d == 0.0)


def test_one_close_pair_is_enough_to_pay_for_the_hessian(surfaces):
    calls = surfaces(3)
    epot = np.array([0.0, 0.01, 2.0])

    _get_couplings(Q, epot, 3, de_cutoff=0.1)

    assert calls == {"force": 1, "hessian": 1}


# ---------------------------------------------------------------------------
# Two states: the case that always worked, pinned so it stays working
# ---------------------------------------------------------------------------

def test_two_states_are_unaffected_by_the_fix(surfaces):
    surfaces(2)
    close = _get_couplings(Q, np.array([0.0, 0.01]), 2, de_cutoff=0.1)
    far = _get_couplings(Q, np.array([0.0, 2.00]), 2, de_cutoff=0.1)

    assert np.linalg.norm(close[0, 1]) > 0.0
    assert np.linalg.norm(far[0, 1]) == 0.0


def test_the_gap_helper_returns_every_unordered_pair():
    gaps = _calculate_diff_energy(np.array([0.0, 1.0, 3.0]))
    np.testing.assert_allclose(sorted(gaps), [1.0, 2.0, 3.0])
    assert gaps.size == 3
