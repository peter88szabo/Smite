"""Tully's three one-dimensional two-state scattering models.

J. C. Tully, *J. Chem. Phys.* **93**, 1061 (1990).

These are the standard benchmarks for surface hopping. Each is a 2x2 diabatic
Hamiltonian in one coordinate, so everything -- adiabatic energies, gradients,
Hessians, and the exact nonadiabatic coupling -- is available in closed form.
That last point is what makes them useful here: Smite approximates the coupling
from the curvature of the adiabatic gap (Baeck-An), and against these models the
approximation can be compared with the exact answer directly, with no
trajectories and no ambiguity about which part of the method is at fault.

For a two-state diabatic Hamiltonian the adiabatic states are a rotation of the
diabatic ones through the mixing angle

    theta(x) = 1/2 * arctan2( 2 V12, V11 - V22 )

and the coupling is its derivative::

    d12(x) = <psi_1| d/dx |psi_2> = -dtheta/dx

The sign is a phase convention; only the magnitude is physical.

Parameters follow Tully's paper. Model parameters were cross-checked against an
independent implementation before use.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


# --------------------------------------------------------------------------
# The diabatic Hamiltonians and their derivatives
# --------------------------------------------------------------------------

def simple_avoided_crossing(x):
    """Model I: a single avoided crossing at the origin."""
    A, B, C, D = 0.01, 1.6, 0.005, 1.0
    x = np.asarray(x, dtype=float)

    v11 = np.where(x > 0.0, A * (1.0 - np.exp(-B * np.abs(x))),
                   -A * (1.0 - np.exp(-B * np.abs(x))))
    v12 = C * np.exp(-D * x**2)
    return v11, -v11, v12


def d_simple_avoided_crossing(x):
    A, B, C, D = 0.01, 1.6, 0.005, 1.0
    x = np.asarray(x, dtype=float)

    dv11 = A * B * np.exp(-B * np.abs(x))          # symmetric in x
    dv12 = -2.0 * C * D * x * np.exp(-D * x**2)
    return dv11, -dv11, dv12


def dual_avoided_crossing(x):
    """Model II: two crossings, so the trajectory samples Stueckelberg phase."""
    A, B, C, D, E0 = 0.1, 0.28, 0.015, 0.06, 0.05
    x = np.asarray(x, dtype=float)

    v11 = np.zeros_like(x)
    v22 = -A * np.exp(-B * x**2) + E0
    v12 = C * np.exp(-D * x**2)
    return v11, v22, v12


def d_dual_avoided_crossing(x):
    A, B, C, D, _ = 0.1, 0.28, 0.015, 0.06, 0.05
    x = np.asarray(x, dtype=float)

    dv11 = np.zeros_like(x)
    dv22 = 2.0 * A * B * x * np.exp(-B * x**2)
    dv12 = -2.0 * C * D * x * np.exp(-D * x**2)
    return dv11, dv22, dv12


def extended_coupling(x):
    """Model III: the coupling stays on after the crossing, causing reflection.

    Designed to break naive surface hopping: trajectories that hop are often
    frustrated on the way out, so it is the model that exercises the
    frustrated-hop treatment hardest.
    """
    A, B, C = 6.0e-4, 0.10, 0.90
    x = np.asarray(x, dtype=float)

    v11 = np.full_like(x, A)
    v12 = np.where(x < 0.0, B * np.exp(C * x), B * (2.0 - np.exp(-C * x)))
    return v11, -v11, v12


def d_extended_coupling(x):
    A, B, C = 6.0e-4, 0.10, 0.90
    x = np.asarray(x, dtype=float)

    dv11 = np.zeros_like(x)
    dv12 = np.where(x < 0.0, B * C * np.exp(C * x), B * C * np.exp(-C * x))
    return dv11, dv11, dv12


MODELS = {
    "SAC": (simple_avoided_crossing, d_simple_avoided_crossing,
            "simple avoided crossing"),
    "DAC": (dual_avoided_crossing, d_dual_avoided_crossing,
            "dual avoided crossing"),
    "ECR": (extended_coupling, d_extended_coupling,
            "extended coupling with reflection"),
}


# --------------------------------------------------------------------------
# Adiabatic quantities, in closed form
# --------------------------------------------------------------------------

def adiabatic_energies(model: str, x) -> Tuple[np.ndarray, np.ndarray]:
    """The two adiabatic surfaces, lower first."""
    v11, v22, v12 = MODELS[model][0](x)
    half_sum = 0.5 * (v11 + v22)
    half_gap = np.sqrt(0.25 * (v11 - v22) ** 2 + v12**2)
    return half_sum - half_gap, half_sum + half_gap


def mixing_angle(model: str, x) -> np.ndarray:
    v11, v22, v12 = MODELS[model][0](x)
    return 0.5 * np.arctan2(2.0 * v12, v11 - v22)


def exact_coupling(model: str, x) -> np.ndarray:
    """``|d12(x)|``, the exact nonadiabatic coupling, from ``dtheta/dx``.

    Differentiating ``theta = 1/2 arctan2(2 V12, V11 - V22)`` analytically::

        dtheta/dx = [ (V11 - V22) dV12/dx - V12 d(V11 - V22)/dx ]
                    / [ (V11 - V22)^2 + 4 V12^2 ]
    """
    v11, v22, v12 = MODELS[model][0](x)
    dv11, dv22, dv12 = MODELS[model][1](x)

    difference = v11 - v22
    d_difference = dv11 - dv22
    numerator = difference * dv12 - v12 * d_difference
    denominator = difference**2 + 4.0 * v12**2
    return np.abs(numerator / denominator)


def adiabatic_gradients(model: str, x, step: float = 1.0e-5):
    """dE/dx for both surfaces, by central differences on the closed form."""
    lower_plus, upper_plus = adiabatic_energies(model, np.asarray(x) + step)
    lower_minus, upper_minus = adiabatic_energies(model, np.asarray(x) - step)
    return ((lower_plus - lower_minus) / (2.0 * step),
            (upper_plus - upper_minus) / (2.0 * step))


def adiabatic_curvatures(model: str, x, step: float = 1.0e-4):
    """d2E/dx2 for both surfaces, which is what Baeck-An consumes."""
    x = np.asarray(x, dtype=float)
    lower_plus, upper_plus = adiabatic_energies(model, x + step)
    lower_mid, upper_mid = adiabatic_energies(model, x)
    lower_minus, upper_minus = adiabatic_energies(model, x - step)
    return ((lower_plus - 2.0 * lower_mid + lower_minus) / step**2,
            (upper_plus - 2.0 * upper_mid + upper_minus) / step**2)


def baeck_an_coupling(model: str, x, step: float = 1.0e-4) -> np.ndarray:
    """Baeck-An's estimate of ``|d12|`` from the gap and its curvature.

        |d12| = 1/2 sqrt( (d2/dx2)(dE) / dE )

    Zero wherever the gap curves the wrong way, which is what the
    implementation in ``fssh.baeck_an_nac`` also returns there.
    """
    lower, upper = adiabatic_energies(model, x)
    gap = upper - lower

    lower_curvature, upper_curvature = adiabatic_curvatures(model, x, step)
    gap_curvature = upper_curvature - lower_curvature

    argument = np.where(gap > 0.0, gap_curvature / np.where(gap > 0.0, gap, 1.0), 0.0)
    return np.where(argument > 0.0, 0.5 * np.sqrt(np.abs(argument)), 0.0)
