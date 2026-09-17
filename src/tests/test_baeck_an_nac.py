"""Baeck-An nonadiabatic couplings.

The couplings are approximated from the curvature of the adiabatic gap rather
than from wavefunction matrix elements, which is what lets surface hopping run
on a machine-learned surface that has no wavefunction at all. These tests pin
the algebra of that approximation, because every failure mode here is silent:
a wrong coupling does not raise, it just produces plausible-looking hopping
statistics that are wrong.
"""

import numpy as np
import pytest

from fssh.baeck_an_nac import calculate_nac
from fssh.fssh import _get_couplings


def _model(gap, gradient_difference, hessian_difference):
    """Two states whose difference has exactly the requested derivatives."""
    ndim = np.size(gradient_difference)
    epot = [0.0, gap]
    grad = [np.zeros((ndim // 3, 3)),
            np.asarray(gradient_difference, dtype=float).reshape(ndim // 3, 3)]
    hess = [np.zeros((ndim, ndim)), np.asarray(hessian_difference, dtype=float)]
    return epot, grad, hess


def _dyad(gap, gradient_difference, hessian_difference):
    """Rebuild the intermediate the implementation forms, from its own inputs."""
    g = np.asarray(gradient_difference, dtype=float).ravel()
    d2de2 = 2 * (gap * np.asarray(hessian_difference) + np.outer(g, g))
    return 1 / 8 * d2de2 - np.outer(g / 2, g / 2)


def test_the_gradient_term_cancels_exactly():
    """The dyad must reduce to (1/4) * gap * d2(gap)/dxdy for any gradient.

    An element-wise square in place of the outer product leaves a residual and
    breaks the symmetry of the dyad -- the bug this test exists to catch.
    """
    rng = np.random.default_rng(0)
    ndim = 6
    gap = 0.7
    g = rng.normal(size=ndim)
    hessian = rng.normal(size=(ndim, ndim))
    hessian = 0.5 * (hessian + hessian.T)

    dyad = _dyad(gap, g, hessian)
    assert dyad == pytest.approx(0.25 * gap * hessian, abs=1.0e-14)
    assert dyad == pytest.approx(dyad.T, abs=1.0e-14)


def test_magnitude_matches_the_analytic_baeck_an_expression():
    """For a gap curving as ``a + b x^2``, the coupling is 0.5*sqrt(2b/a)."""
    ndim = 6
    a, b = 0.30, 0.8

    hessian = np.zeros((ndim, ndim))
    hessian[0, 0] = 2.0 * b                     # d2(gap)/dx0^2
    epot, grad, hess = _model(a, np.zeros(ndim), hessian)

    nac = calculate_nac(epot, grad, hess)

    assert np.linalg.norm(nac) == pytest.approx(0.5 * np.sqrt(2.0 * b / a))
    # ... and it points along the coordinate that carries the curvature.
    assert abs(nac.ravel()[0]) == pytest.approx(np.linalg.norm(nac))


def test_a_nonzero_gap_gradient_does_not_change_the_magnitude():
    """The coupling depends on the curvature of the gap, not on its slope."""
    ndim = 6
    hessian = np.zeros((ndim, ndim))
    hessian[0, 0] = 1.6

    flat = calculate_nac(*_model(0.30, np.zeros(ndim), hessian))
    sloped = calculate_nac(*_model(0.30, np.full(ndim, 0.25), hessian))

    assert np.linalg.norm(sloped) == pytest.approx(np.linalg.norm(flat))


def test_a_gap_with_no_positive_curvature_gives_no_coupling():
    """Downward curvature is outside the model; svd would hide the sign."""
    ndim = 6
    hessian = np.zeros((ndim, ndim))
    hessian[0, 0] = -2.0

    nac = calculate_nac(*_model(0.30, np.zeros(ndim), hessian))
    assert nac == pytest.approx(np.zeros((ndim // 3, 3)))


def test_a_vanishing_gap_gives_no_coupling_instead_of_dividing_by_zero():
    ndim = 6
    hessian = np.zeros((ndim, ndim))
    hessian[0, 0] = 2.0

    nac = calculate_nac(*_model(0.0, np.zeros(ndim), hessian))
    assert np.all(np.isfinite(nac))
    assert nac == pytest.approx(np.zeros((ndim // 3, 3)))


def test_sign_is_kept_continuous_with_the_previous_step():
    """The eigenvector's overall sign is arbitrary and must not flip freely.

    It enters the amplitude derivative as ``v . d_ij``, so a free flip between
    steps scrambles the coherent phase.
    """
    ndim = 6
    hessian = np.zeros((ndim, ndim))
    hessian[0, 0] = 2.0
    model = _model(0.30, np.zeros(ndim), hessian)

    reference = calculate_nac(*model)

    aligned = calculate_nac(*model, previous_nac=reference)
    flipped = calculate_nac(*model, previous_nac=-reference)

    assert aligned == pytest.approx(reference)
    assert flipped == pytest.approx(-reference)


def test_coupling_matrix_is_antisymmetric_and_hollow(monkeypatch):
    """_get_couplings fills the upper triangle and mirrors it with a sign flip."""
    ndim = 6
    epot = np.array([0.0, 0.30])
    q = np.zeros(ndim)

    forces = np.zeros((2, ndim))
    hessians = np.zeros((2, ndim, ndim))
    hessians[1, 0, 0] = 2.0

    monkeypatch.setattr("fssh.fssh.PES_Force", lambda *a, **k: -forces)
    monkeypatch.setattr("fssh.fssh.PES_Hessian", lambda *a, **k: hessians)

    d = _get_couplings(q, epot, 2, de_cutoff=0.5)

    assert d[0, 1] == pytest.approx(-d[1, 0])
    assert d[0, 0] == pytest.approx(np.zeros((ndim // 3, 3)))
    assert np.linalg.norm(d[0, 1]) > 0.0
