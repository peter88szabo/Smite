"""Stable central finite-difference Hessians from analytic gradients."""

import numpy as np


def central_difference_hessian(q, gradient, *, dx=0.002):
    """Differentiate ``gradient(q)`` with a symmetric central stencil.

    Coordinates are never mutated in place, which prevents failed backend
    calls from leaving callers at a displaced geometry.  The final
    symmetrization suppresses finite-difference/SCF noise while retaining the
    conservative Hessian expected by vibrational and optimizer code.
    """
    q0 = np.asarray(q, dtype=float).reshape(-1).copy()
    dx = float(dx)
    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError("hessian_dx must be a finite positive number")

    ndim = q0.size
    hessian = np.empty((ndim, ndim), dtype=float)
    for coordinate in range(ndim):
        q_plus = q0.copy()
        q_minus = q0.copy()
        q_plus[coordinate] += dx
        q_minus[coordinate] -= dx
        grad_plus = np.asarray(gradient(q_plus), dtype=float).reshape(-1)
        grad_minus = np.asarray(gradient(q_minus), dtype=float).reshape(-1)
        if grad_plus.shape != q0.shape or grad_minus.shape != q0.shape:
            raise ValueError("Gradient evaluator returned a shape incompatible with the coordinates")
        if not np.all(np.isfinite(grad_plus)) or not np.all(np.isfinite(grad_minus)):
            raise ValueError("Gradient evaluator returned non-finite values during Hessian calculation")
        hessian[:, coordinate] = (grad_plus - grad_minus) / (2.0 * dx)
    return 0.5 * (hessian + hessian.T)
