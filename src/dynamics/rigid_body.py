"""Reversible free rigid-body drift in Cartesian coordinates and momenta.

The nonlinear rotor uses symmetric principal-axis Hamiltonian splitting
(Dullweber, Leimkuhler, McLachlan, JCP 107, 5840 (1997),
https://doi.org/10.1063/1.474310). Laboratory angular momentum is constant
during the free drift. A linear rotor has an exact rotation about that vector.
"""

import numpy as np

from utils.geometry import reference_rotation


def inertia_tensor(positions, masses):
    weighted = masses[:, None] * positions
    return np.eye(3) * np.sum(weighted * positions) - positions.T @ weighted


def _rotation(axis, angle):
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    # 2 sin(theta/2)^2 avoids cancellation in 1-cos(theta).
    return np.eye(3) + np.sin(angle) * skew + 2.0 * np.sin(0.5 * angle)**2 * (skew @ skew)


def project_rigid_momenta(positions, momenta, masses):
    """Project onto rigid velocities while retaining linear/angular momentum."""
    total_mass = np.sum(masses)
    relative = positions - np.average(positions, axis=0, weights=masses)
    velocity = np.sum(momenta, axis=0) / total_mass
    angular_momentum = np.sum(np.cross(relative, momenta), axis=0)
    omega = np.linalg.pinv(inertia_tensor(relative, masses), rcond=1.0e-12) @ angular_momentum
    return masses[:, None] * (velocity + np.cross(omega, relative))


def free_rigid_body_step(reference, positions, momenta, masses, dt):
    """Advance a rigid group with fixed laboratory linear/angular momentum."""
    center = np.average(positions, axis=0, weights=masses)
    relative = positions - center
    velocity = np.sum(momenta, axis=0) / np.sum(masses)
    angular_momentum = np.sum(np.cross(relative, momenta), axis=0)

    reference = reference - np.average(reference, axis=0, weights=masses)
    moments, body_axes = np.linalg.eigh(inertia_tensor(reference, masses))
    active = moments > 1.0e-12 * np.max(moments)
    rotation = reference_rotation(reference, relative, masses)
    axes = rotation @ body_axes

    if np.count_nonzero(active) == 2:
        magnitude = np.linalg.norm(angular_momentum)
        if magnitude > 0.0:
            angle = dt * magnitude / np.mean(moments[active])
            axes = _rotation(angular_momentum / magnitude, angle) @ axes
    elif np.count_nonzero(active) == 3:
        # Each subflow rotates about one instantaneous principal axis.
        # Its angular-momentum component is constant within that subflow.
        for index, fraction in ((0, 0.5), (1, 0.5), (2, 1.0), (1, 0.5), (0, 0.5)):
            axis = axes[:, index]
            angle = fraction * dt * np.dot(angular_momentum, axis) / moments[index]
            axes = _rotation(axis, angle) @ axes
    else:
        raise ValueError("A rigid group requires at least two nonzero principal moments")

    relative_new = (reference @ body_axes) @ axes.T
    omega = axes[:, active] @ ((axes[:, active].T @ angular_momentum) / moments[active])
    positions_new = relative_new + center + dt * velocity
    momenta_new = masses[:, None] * (velocity + np.cross(omega, relative_new))
    return positions_new, momenta_new
