import numpy as np
from scipy.linalg import svd
from numpy.typing import NDArray
from numpy import float64, complex128


def calculate_nac(
    epot: list[NDArray[float64]],
    grad: list[NDArray[float64]],
    hess: list[NDArray[float64]],
) -> NDArray[complex128]:
    """
    This computes the nonadiabatic coupling vector (NAC) between two states using the energy, gradient, and Hessian.
    The following wavefunction-based NACs, usually computed as follows, are approximated :

    . math::
    d_{ij}(\mathbf{R}) = \frac{\langle \Psi_{i}(\mathbf{R};r) | \nabla_\mathbf{R} H_p(\mathbf{R};r) | \Psi_{j}(\mathbf{R};r) \rangle}{\epsilon_{j}(\mathbf{R}) - \epsilon_i(\mathbf{R})}

    The NACs are used for hopping from state i to state j. i=0 and j=1

    Reference:
    Westermayr, J.; Gastegger, M.; Marquetand, P. J. Phys. Chem. Lett. 2020, 11 (10), 3828–3834. https://doi.org/10.1021/acs.jpclett.0c00527.

    Warning:
    Only for crossing between same-symmetry states!

    Parameters
    ----------
    epot : list[NDArray[float64]]
        Energy of the states.
    grad : list[NDArray[float64]]
        Gradient of the states.
    hess : list[NDArray[float64]]
        Hessian of the states.

    Returns
    -------
    NDArray[complex128]
        Nonadiabatic coupling vector.
    """
    if len(grad[0].shape) == 1:
        raise NotImplementedError(
            "Reshape the 1d array using either .reshape(-1,1) for multiple particles and one dof or .reshape(1,-1)"
            + " for one particle and multiple dof."
        )
    else:
        n_atoms, n_dof = grad[0].shape[0], 3

    de = epot[1] - epot[0]
    dde_dx = grad[1] - grad[0]
    d2de_dx2 = hess[1] - hess[0]

    d2de2_dx2 = 2 * (
        de * d2de_dx2 + dde_dx.ravel() ** 2
    )  # The summation is element-wise
    nac_dyad = 1 / 8 * d2de2_dx2 - np.outer(1 / 2 * dde_dx, 1 / 2 * dde_dx)

    U, E, V = svd(nac_dyad)

    nac = V[np.argmax(E), :] * np.sqrt(np.max(E)) / de

    nac = np.reshape(nac, (n_atoms, n_dof))

    return nac
