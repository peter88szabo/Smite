"""Tully's models, and what they say about the Baeck-An coupling approximation.

Run from this directory::

    python -m pytest validation/tully -q

Kept out of ``src/tests`` because these are validation checks against an
external benchmark rather than unit tests of the package.
"""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pytest

from tully_models import (
    MODELS,
    adiabatic_energies,
    baeck_an_coupling,
    exact_coupling,
    extended_coupling,
    mixing_angle,
)


X = np.linspace(-10.0, 10.0, 4001)


# --------------------------------------------------------------------------
# The models themselves
# --------------------------------------------------------------------------

def test_simple_avoided_crossing_has_the_published_gap_at_the_crossing():
    """The two surfaces are 2C apart at x = 0."""
    lower, upper = adiabatic_energies("SAC", np.array([0.0]))
    assert upper[0] - lower[0] == pytest.approx(2 * 0.005)


def test_dual_avoided_crossing_gap_matches_the_closed_form():
    lower, upper = adiabatic_energies("DAC", np.array([0.0]))
    # V11 = 0, V22 = E0 - A = -0.05, V12 = C = 0.015
    expected = 2.0 * np.sqrt(0.025**2 + 0.015**2)
    assert upper[0] - lower[0] == pytest.approx(expected)


def test_extended_coupling_stays_on_after_the_crossing():
    """The defining feature of model III: V12 tends to 2B, not to zero."""
    _, _, v12 = extended_coupling(np.array([-20.0, 0.0, 20.0]))
    assert v12[0] == pytest.approx(0.0, abs=1.0e-8)
    assert v12[1] == pytest.approx(0.10)
    assert v12[2] == pytest.approx(0.20, abs=1.0e-6)


@pytest.mark.parametrize("model", list(MODELS))
def test_the_surfaces_never_cross(model):
    lower, upper = adiabatic_energies(model, X)
    assert np.all(upper >= lower)


@pytest.mark.parametrize("model", list(MODELS))
def test_the_analytic_coupling_matches_a_numerical_derivative(model):
    """d12 = -dtheta/dx, so differentiating the mixing angle must reproduce it."""
    step = 1.0e-6
    x = np.linspace(-8.0, 8.0, 401)
    # Skip the kink at x = 0 in SAC and ECR, where V has a discontinuous slope.
    x = x[np.abs(x) > 1.0e-2]

    numerical = np.abs(
        (mixing_angle(model, x + step) - mixing_angle(model, x - step)) / (2 * step)
    )
    assert exact_coupling(model, x) == pytest.approx(numerical, rel=1.0e-4, abs=1.0e-8)


def test_the_simple_model_peaks_at_the_coupling_maximum():
    """A known result for model I: |d12| at the crossing equals B."""
    assert exact_coupling("SAC", np.array([0.0]))[0] == pytest.approx(1.6, rel=1.0e-6)


def test_the_dual_model_has_two_coupling_maxima():
    coupling = exact_coupling("DAC", X)
    left = X[np.argmax(np.where(X < 0, coupling, 0))]
    right = X[np.argmax(np.where(X > 0, coupling, 0))]
    assert left == pytest.approx(-right, abs=0.05)
    assert abs(left) > 1.0


# --------------------------------------------------------------------------
# What Baeck-An reproduces, and what it does not
# --------------------------------------------------------------------------

@pytest.mark.parametrize("model, tolerance", [("SAC", 0.15), ("DAC", 0.05)])
def test_baeck_an_is_accurate_at_an_avoided_crossing(model, tolerance):
    """Where the method is designed to work, it works: within 15% and 5%.

    This is the peak of |d12|, which is where hops actually happen.
    """
    exact = exact_coupling(model, X)
    approximate = baeck_an_coupling(model, X)
    peak = int(np.argmax(exact))

    assert approximate[peak] / exact[peak] == pytest.approx(1.0, abs=tolerance)


@pytest.mark.parametrize("model", ["SAC", "DAC"])
def test_baeck_an_goes_silent_across_most_of_the_coupling_region(model):
    """It captures the peak and little else.

    The gap curvature turns negative away from the crossing, where Baeck-An
    prescribes no coupling at all, so the tails of d12 are lost. This is a
    property of the approximation, recorded here so it is not mistaken for a
    regression.
    """
    exact = exact_coupling(model, X)
    approximate = baeck_an_coupling(model, X)
    appreciable = exact > 0.05 * exact.max()

    silent = np.mean(approximate[appreciable] == 0.0)
    assert silent > 0.5


def test_baeck_an_fails_on_the_extended_coupling_model():
    """Model III has no localised avoided crossing, so the premise is absent.

    The approximation overestimates the peak by about 70%, and worse, it pins at
    a near-constant value of roughly 0.45 across the whole approach region while
    the exact coupling decays away. Between x = -2 and the crossing it is some
    two orders of magnitude too large, which would drive heavy spurious hopping.
    Past the crossing it collapses to zero instead. A system of this character
    should not be run with curvature-derived couplings.
    """
    exact = exact_coupling("ECR", X)
    approximate = baeck_an_coupling("ECR", X)
    peak = int(np.argmax(exact))

    assert approximate[peak] / exact[peak] > 1.5

    # The approach region, where the trajectory spends its time before hopping.
    approach = (X > -4.0) & (X < 0.0)
    assert np.median(approximate[approach] / exact[approach]) > 10.0

    # And nothing at all on the far side.
    beyond = X > 2.0
    assert np.all(approximate[beyond] == 0.0)


@pytest.mark.parametrize("model", list(MODELS))
def test_the_approximation_is_never_negative_or_nan(model):
    approximate = baeck_an_coupling(model, X)
    assert np.all(np.isfinite(approximate))
    assert np.all(approximate >= 0.0)


# --------------------------------------------------------------------------
# The exact quantum reference shipped alongside
# --------------------------------------------------------------------------

def test_the_reference_data_is_self_consistent():
    reference = np.loadtxt(HERE / "reference_exact_quantum.dat")
    assert reference.shape[1] == 4

    populations = reference[:, 2:]
    assert populations.sum(axis=1) == pytest.approx(np.ones(len(reference)), abs=1.0e-4)
    # Transfer to the upper surface grows with the collision momentum.
    assert np.all(np.diff(populations[:, 1]) > 0.0)
