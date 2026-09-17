"""Conservation diagnostics for classical trajectories."""

import numpy as np

from utils.constants import HARTREE_TO_KJMOL


def total_linear_momentum(p):
    return np.sum(np.asarray(p, dtype=float).reshape((-1, 3)), axis=0)


def total_angular_momentum(q, p, mass):
    coordinates = np.asarray(q, dtype=float).reshape((-1, 3))
    momenta = np.asarray(p, dtype=float).reshape((-1, 3))
    masses = np.asarray(mass, dtype=float).reshape(-1)
    if len(coordinates) != len(masses) or len(momenta) != len(masses):
        raise ValueError("Coordinates, momenta, and masses must contain the same number of atoms")
    center_of_mass = np.average(coordinates, axis=0, weights=masses)
    return np.sum(np.cross(coordinates - center_of_mass, momenta), axis=0)


def collision_conservation_residuals(
    collision,
    trajectory_initial_energy_hartree=None,
    trajectory_final_energy_hartree=None,
):
    """Return initial/final conserved quantities and their final-minus-initial residuals.

    The energy residual is reported only when the trajectory integrator supplied
    total initial and final energies on the same full-dimensional Hamiltonian.
    Fragment asymptotic energies are deliberately not used for this diagnostic:
    their difference includes the physical reaction energy.
    """
    q_initial = getattr(collision, "q_collision_initial", None)
    p_initial = getattr(collision, "p_collision_initial", None)
    if q_initial is None or p_initial is None:
        q_initial = getattr(collision, "q_ini", None)
        p_initial = getattr(collision, "p_ini", None)
    if q_initial is None or p_initial is None:
        raise ValueError("Initial collision coordinates and momenta are unavailable")

    linear_initial = total_linear_momentum(p_initial)
    linear_final = total_linear_momentum(collision.p)
    angular_initial = total_angular_momentum(q_initial, p_initial, collision.mass)
    angular_final = total_angular_momentum(collision.q, collision.p, collision.mass)
    energy_residual = None
    if trajectory_initial_energy_hartree is not None and trajectory_final_energy_hartree is not None:
        energy_residual = (
            float(trajectory_final_energy_hartree)
            - float(trajectory_initial_energy_hartree)
        )

    return {
        "linear_momentum_initial": linear_initial,
        "linear_momentum_final": linear_final,
        "linear_momentum_residual": linear_final - linear_initial,
        "linear_momentum_residual_norm": float(np.linalg.norm(linear_final - linear_initial)),
        "angular_momentum_initial": angular_initial,
        "angular_momentum_final": angular_final,
        "angular_momentum_residual": angular_final - angular_initial,
        "angular_momentum_residual_norm": float(np.linalg.norm(angular_final - angular_initial)),
        "trajectory_initial_energy_hartree": trajectory_initial_energy_hartree,
        "trajectory_final_energy_hartree": trajectory_final_energy_hartree,
        "trajectory_energy_residual_hartree": energy_residual,
        "trajectory_energy_residual_kjmol": (
            None if energy_residual is None else energy_residual * HARTREE_TO_KJMOL
        ),
    }
