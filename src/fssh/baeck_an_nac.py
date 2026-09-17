import numpy as np
from scipy.linalg import eigh
from numpy.typing import NDArray
from numpy import float64, complex128


# Below this adiabatic gap (Hartree) the Baeck-An expression divides by a number
# indistinguishable from zero and the two-state model has broken down anyway.
MINIMUM_GAP = 1.0e-10


def calculate_nac(
    epot: list[NDArray[float64]],
    grad: list[NDArray[float64]],
    hess: list[NDArray[float64]],
    previous_nac: NDArray[float64] = None,
) -> NDArray[complex128]:
    r"""
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
    previous_nac : NDArray[float64], optional
        The coupling returned at the previous step. Only its sign is used, to
        keep the arbitrary phase of the eigenvector continuous along a
        trajectory. Without it the sign can flip between steps and scramble the
        coherent evolution of the electronic amplitudes, because the coupling
        enters the amplitude derivative as ``v . d_ij``.

    Returns
    -------
    NDArray[complex128]
        Nonadiabatic coupling vector. Zero where the model does not apply: a
        vanishing gap, or a gap with no positive curvature.
    """
    if len(grad[0].shape) == 1:
        raise NotImplementedError(
            "Reshape the 1d array using either .reshape(-1,1) for multiple particles and one dof or .reshape(1,-1)"
            + " for one particle and multiple dof."
        )
    else:
        n_atoms, n_dof = grad[0].shape[0], 3

    de = epot[1] - epot[0]
    gap = abs(de)
    if not np.isfinite(gap) or gap < MINIMUM_GAP:
        return np.zeros((n_atoms, n_dof))

    dde_dx = (grad[1] - grad[0]).ravel()
    d2de_dx2 = hess[1] - hess[0]

    # Chain rule for the square of the gap. Both terms are rank-two in the
    # coordinates, so the gradient term is the outer product d(dE)/dx d(dE)/dy,
    # not an element-wise square.
    d2de2_dx2 = 2 * (de * d2de_dx2 + np.outer(dde_dx, dde_dx))
    nac_dyad = 1 / 8 * d2de2_dx2 - np.outer(1 / 2 * dde_dx, 1 / 2 * dde_dx)

    # The subtraction above cancels the gradient term exactly, leaving
    # (1/4) * dE * d2(dE)/dxdy. Written this way to stay close to the reference,
    # and symmetrized so rounding cannot make the eigensolver complain.
    nac_dyad = 0.5 * (nac_dyad + nac_dyad.T)

    # eigh, not svd: the singular values are absolute and would turn a gap that
    # curves downwards -- where Baeck-An prescribes no coupling at all -- into a
    # spurious real one.
    eigenvalues, eigenvectors = eigh(nac_dyad)
    curvature = eigenvalues[-1]
    if curvature <= 0.0:
        return np.zeros((n_atoms, n_dof))

    nac = eigenvectors[:, -1] * np.sqrt(curvature) / gap

    # The eigenvector's overall sign is arbitrary; tie it to the previous step.
    if previous_nac is not None:
        if np.dot(nac, np.asarray(previous_nac, dtype=float).ravel()) < 0.0:
            nac = -nac

    nac = np.reshape(nac, (n_atoms, n_dof))

    return nac
