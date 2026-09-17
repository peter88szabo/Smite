"""Mass-weighted alignment of corresponding Cartesian atoms."""

import numpy as np


def reference_rotation(reference_centered, current_centered, masses):
    """Return the proper rotation mapping reference vectors into the current frame.

    Both arrays have shape (N, 3), are centered at their respective mass
    centers, and must use the same atom order. Reflections are excluded.
    """
    covariance = reference_centered.T @ (masses[:, None] * current_centered)
    left, _values, right = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.copysign(1.0, np.linalg.det(right.T @ left.T))
    return right.T @ correction @ left.T
