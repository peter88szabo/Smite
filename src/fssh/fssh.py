import warnings
from typing import Optional

import numpy as np
from scipy.linalg import eigh
from numpy.typing import NDArray
from numpy import float64, complex128

from .pes_adapter import PES_Energy, PES_Force, PES_Hessian
from .baeck_an_nac import calculate_nac

class FSSH:
    def __init__(self, num_states: int, active_state: int, de_cutoff: float = 0.5) -> None:
       self.c = self._initialize_amplitudes(num_states, active_state)
       self.de_cutoff = de_cutoff
       self.d: Optional[NDArray] = None

    def __call__(
        self,
        dtc: float,
        active_state: int,
        num_states: int,
        q: NDArray[float64],
        p: NDArray[float64],
        mass: NDArray[float64],
        dtq: Optional[float] = None,
    ):
        """
        Propagate the system using the fewest switches surface hopping algorithm.

        Parameters
        ----------
        dtc : float
            Classical timestep.
        active_state : int
            Index of the active state.
        num_states : int
            Number of states.
        q : NDArray[float64]
            Atomic coordinates.
        p : NDArray[float64]
            Atomic momenta.
        mass : NDArray[float64]
            Atomic masses.
        c : NDArray[complex128]
            Quantum amplitudes.
        de_cutoff : float, optional
            Energy difference cutoff for the non-adiabatic coupling calculation.
        dtq : float, optional
            Quantum timestep. If not specified, it is set to 1/10 of the classical timestep.

        Returns
        -------
        tuple[NDArray[float64], NDArray[complex128], int]
            New atomic momenta, quantum amplitudes, and active state.
        """
        dtq = dtq if dtq is not None else dtc * 1e-1
        # Define the number of quantum steps per classical step
        num_step = int(dtc / dtq)

        # Initialize the hopped state to -1, no hop has occured
        hopped_state = -1

        # Reshape momentum and masses
        p, mass = p.reshape(-1, 3), mass.reshape(-1, 3)

        # Transform momentum into velocity to simplify equations
        v = p / mass

        # Get the potential energy values
        epot = PES_Energy(
            q, list(range(num_states))
        )  # TODO: This should be generally implemented for any qc method

        # Update coupling every classical timestep, keeping the sign of each
        # coupling continuous with the previous step.
        self.d = _get_couplings(q, epot, num_states, self.de_cutoff, previous=self.d)

        # Propagate the quantum system and check for hops
        for _ in range(num_step):
            self.c, hopped_state = _quantum_step(
                v, self.c, self.d, epot, num_states, active_state, hopped_state, dtq
            )

        # Check if the hop is energetically allowed
        if hopped_state != -1:
            v, active_state = _check_frustrated_hop(
                q, v, self.d, mass, epot, active_state, hopped_state
            )

        p = mass * v  # Transform velocity back to momentum
        p = p.reshape(-1)

        return p, active_state

    def _initialize_amplitudes(self, num_states: int, starting_state: int) -> NDArray[complex128]:
        c = np.zeros(num_states, dtype=complex)
        c[starting_state] = 1.0
        return c


def _get_couplings(
    q: NDArray[float64],
    epot: NDArray[float64],
    num_states: int,
    de_cutoff: float,
    previous: NDArray[complex128] = None,
):
    """
    Calculate the non-adiabatic couplings between all states

    Parameters
    ----------
    q : NDArray[float64]
        Atomic coordinates.
    num_states : int
        Number of states.
    de_cutoff : float
        Energy difference cutoff for the non-adiabatic coupling calculation.

    Returns
    -------
    NDArray[complex128]
        Non-adiabatic coupling matrix.
    """
    n_atoms, n_dof = q.shape[0] // 3, 3
    states = list(range(num_states))

    # Initialize coupling matrix
    d = np.zeros((num_states, num_states, n_atoms, n_dof), dtype=complex128)
    # Calculate the energy difference between all states
    diff_epot = _calculate_diff_energy(epot)

    # Check if any energy difference between all states is lower than the cutoff
    if np.any(diff_epot < de_cutoff):
        grad = -PES_Force(
            q, states
        )  # TODO: These should be generally implemented for any qc method
        grad = grad.reshape(num_states, n_atoms, n_dof)  # reshape for nac calc
        hess = PES_Hessian(
            q, states
        )  # TODO: This should be generally implemented for any qc method
        hess = hess.reshape(
            num_states, n_atoms * n_dof, n_atoms * n_dof
        )  # reshape for nac calc
    else:
        grad = None
        hess = None

    for i in states:  # Rows are the states hopping from
        for j in states:  # Columns are the states hopping to
            if i < j:
                if grad is not None:
                    # Compute the upper triangle matrix elements
                    d[i, j] = calculate_nac(
                        [epot[i], epot[j]],
                        [grad[i], grad[j]],
                        [hess[i], hess[j]],
                        previous_nac=None if previous is None else previous[i, j].real,
                    )
                else:
                    d[i, j] = np.zeros_like(d[i, j])

    # Reduce the number of calls to approximate_nac()
    for i in states:
        for j in states:
            # Copy upper triangle to lower triangle
            # non-adiabatic coupling matrix is antisymmetric
            if i > j:
                d[i, j] = -d[j, i]

    return d


def _calculate_diff_energy(epot: NDArray[float64]) -> NDArray[float64]:
    """
    Calculates the difference in potential energy between all states.

    Parameters
    ----------
    epot : NDArray[float64]
        Potential energy values of all states.

    Returns
    -------
    NDArray[float64]
        Difference in potential energy between all states.
    """
    epot = epot[:, np.newaxis]
    diff = np.abs((epot - epot.T)[np.triu_indices(epot.shape[0], 1)])
    return diff


def _quantum_step(
    v: NDArray[float64],
    c: NDArray[complex128],
    d: NDArray[complex128],
    epot: NDArray[float64],
    num_states: int,
    active_state: int,
    hopped_state: int,
    dtq: float,
) -> tuple[NDArray[complex128], int]:
    """
    Perform a single quantum timestep.

    Parameters
    ----------
    v : NDArray[float64]
        Velocity vector.
    c : NDArray[complex128]
        Quantum amplitudes.
    d : NDArray[complex128]
        Non-adiabatic coupling matrix.
    epot : NDArray[float64]
        Potential energy values of all states.
    num_states : int
        Number of states.
    active_state : int
        Index of the active state.
    hopped_state : int
        Index of the hopped state.
    dtq : float
        Quantum timestep.

    Returns
    -------
    tuple[NDArray[complex128], int]
        Updated quantum amplitudes and hopped state
    """
    if hopped_state == -1:
        hop_prob = _eval_hop_prob(v, c, d, num_states, active_state, dtq)
        hopped_state = _check_hop(hop_prob, num_states, active_state)

    c = _update_c(v, c, d, num_states, epot, dtq)
    return c, hopped_state


def _check_frustrated_hop(
    q: NDArray[float64],
    v: NDArray[float64],
    d: NDArray[np.float64],
    m: NDArray[float64],
    epot: NDArray[float64],
    active_state: int,
    hopped_state: int,
) -> tuple[NDArray[float64], int]:
    """
    Check if the hop is frustrated and reflect the velocity vector if necessary.

    Parameters
    ----------
    q : NDArray[float64]
        Atomic coordinates.
    v : NDArray[float64]
        Velocity vector.
    d : NDArray[float64]
        Non-adiabatic coupling matrix.
    m : NDArray[float64]
        Atomic masses.
    epot : NDArray[float64]

    Returns
    -------
    tuple[NDArray[float64], int]
        Updated velocity vector and active state.
    """
    previous_active_state = active_state

    v, new_active_state = _rescale_velocity(
        q, v, d, m, epot, active_state, hopped_state
    )

    if new_active_state != previous_active_state:  # Hop is not frustrated
        return v, new_active_state
    else:  # Reflection of the velocity vector due to frustrated hop, active_state is not switched
        return v, active_state


def _cdot(
    v: NDArray[float64],
    c: NDArray[complex128],
    d: NDArray[complex128],
    num_states: int,
    epot: NDArray[float64],
):
    """
    Calculate the time derivative of the quantum amplitudes.

    Parameters
    ----------
    v : NDArray[float64]
        Velocity vector.
    c : NDArray[complex128]
        Quantum amplitudes.
    d : NDArray[complex128]
        Non-adiabatic coupling matrix.
    num_states : int
        Number of states.
    epot : NDArray[float64]
        Potential energy values of all states.

    Returns
    -------
    NDArray[complex128]
        Time derivative of the quantum amplitudes.
    """
    cdot = np.zeros(num_states, dtype=complex)

    for i in range(num_states):
        cdot[i] = epot[i] * c[i] / 1.0j - np.sum(
            v * d[i, :] * c[:, np.newaxis, np.newaxis]
        )  # Reshape c for multiplication
    return cdot


def _update_c(
    v: NDArray[float64],
    c: NDArray[complex128],
    d: NDArray[complex128],
    num_states: int,
    epot: NDArray[float64],
    dtq: float,
):
    """
    Update the quantum amplitudes with the exact unitary propagator.

    The generator of the adiabatic equation of motion is anti-Hermitian, so the
    evolution is a unitary rotation of the amplitudes and is applied here as
    ``exp(-i H dtq)`` via a Hermitian eigendecomposition. This conserves the
    electronic norm to machine precision for any ``dtq``.

    It replaces a fourth-order Runge-Kutta step, which approximates that unitary
    operator with a non-unitary one: the norm drifted by ~4e-2 even at weak
    coupling and small ``dtq``, and diverged to NaN in the strong-coupling regime
    that surface hopping exists to describe.

    Parameters
    ----------
    v : NDArray[float64]
        Velocity vector.
    c : NDArray[complex128]
        Quantum amplitudes.
    d : NDArray[complex128]
        Non-adiabatic coupling matrix.
    num_states : int
        Number of states.
    epot : NDArray[float64]
        Potential energy values of all states.
    dtq : float
        Quantum timestep.

    Returns
    -------
    NDArray[complex128]
        Updated quantum amplitudes.
    """
    hamiltonian = _effective_hamiltonian(v, d, num_states, epot)

    # H is Hermitian, so exp(-i H dtq) is unitary by construction and the
    # electronic norm is conserved to machine precision for any dtq.
    eigenvalues, eigenvectors = eigh(hamiltonian)
    phases = np.exp(-1.0j * eigenvalues * dtq)

    return eigenvectors @ (phases * (eigenvectors.conj().T @ c))


def _effective_hamiltonian(
    v: NDArray[float64],
    d: NDArray[complex128],
    num_states: int,
    epot: NDArray[float64],
) -> NDArray[complex128]:
    """Assemble the Hermitian generator of the amplitude evolution.

    The adiabatic equation of motion is ``cdot = A c`` with
    ``A = -i diag(E) - T`` and ``T_kj = v . d_kj``. The couplings are real and
    antisymmetric, so ``T`` is anti-Hermitian, ``A`` is anti-Hermitian, and
    ``H = i A`` -- assembled here -- is Hermitian::

        H_kk = E_k
        H_kj = -i (v . d_kj)      for k != j

    which makes ``exp(-i H dt)`` the exact, unitary propagator of ``A``.
    """
    hamiltonian = np.zeros((num_states, num_states), dtype=complex128)
    for i in range(num_states):
        hamiltonian[i, i] = epot[i]
        for j in range(num_states):
            if i != j:
                hamiltonian[i, j] = -1.0j * np.real(np.sum(v * d[i, j]))
    return hamiltonian


def _eval_hop_prob(
    v: NDArray[float64],
    c: NDArray[complex128],
    d: NDArray[complex128],
    num_states: int,
    active_state: int,
    dtq: float,
):
    """
    Evaluate the hopping probability between all states.

    Parameters
    ----------
    v : NDArray[float64]
        Velocity vector.
    c : NDArray[complex128]
        Quantum amplitudes.
    d : NDArray[complex128]
        Non-adiabatic coupling matrix.
    num_states : int
        Number of states.
    active_state : int
        Index of the active state.
    dtq : float
        Quantum timestep.

    Returns
    -------
    NDArray[float64]
        Hopping probabilities. Non-negative, and summing to at most one.
    """
    g = np.zeros(num_states)  # Real hopping probability in the adiabatic basis

    for state in range(num_states):
        if state != active_state:
            g[state] = (
                2
                * dtq
                * (
                    np.real(
                        np.sum(v * d[active_state, state])
                        * np.conjugate(c[active_state])
                        * c[state]
                    )
                    / (c[active_state].real ** 2 + c[active_state].imag ** 2)
                )
            )
            if g[state] < 0:
                g[state] = (
                    0  # As in Jain, Alguire, Subotnik (2016) 10.1021/acs.jctc.6b00673
                )
        else:
            g[state] = 0

    # The expression above is first order in dtq, so it is only a probability
    # while it stays below one. A total above one means the linearization has
    # broken down for this timestep: the hop is then certain, and the best that
    # can be recovered is the relative branching between the target states.
    # Rescaling preserves those ratios; the warning says how to make it moot.
    total = g.sum()
    if total > 1.0:
        _warn_hop_probability_exceeded_one()
        g = g / total

    return g


_HOP_PROBABILITY_WARNED = False


def _warn_hop_probability_exceeded_one():
    """Warn once per run rather than once per quantum sub-step."""
    global _HOP_PROBABILITY_WARNED
    if not _HOP_PROBABILITY_WARNED:
        _HOP_PROBABILITY_WARNED = True
        warnings.warn(
            "Total surface-hopping probability exceeded one and was rescaled. "
            "The first-order expression for the hop probability is only valid "
            "below one, so the branching ratios in this run are unreliable. "
            "Reduce dtq (the quantum timestep) until this warning stops.",
            RuntimeWarning,
            stacklevel=3,
        )


def _check_hop(hop_prob: NDArray[float64], num_states: int, active_state: int) -> int:
    """
    Check if a hop occurs between states.

    Parameters
    ----------
    hop_prob : NDArray[float64]
        Hopping probabilities.
    num_states : int
        Number of states.
    active_state : int
        Index of the active state.

    Returns
    -------
    int
        Index of the hopped state.
    """
    if hop_prob[active_state] != 0.0:
        raise ValueError(
            "Hopping probability from the active state to the active state is not zero."
        )

    zeta = np.random.uniform(0, 1)
    s = 0.0  # Cumulative sum of hopping probabilities

    for i in range(num_states):
        s = s + hop_prob[i]
        if zeta < s:
            return i
    return -1


def _rescale_velocity(
    q: NDArray[float64],
    v: NDArray[float64],
    d: NDArray[float64],
    m: NDArray[float64],
    epot: NDArray[float64],
    active_state: int,
    hopped_state: int,
) -> tuple[NDArray[float64], int]:
    """
    Rescale the velocity vector after a hop has been detected as in:
    Hammes-Schiffer & Tully (1994) 10.1063/1.467146

    Parameters
    ----------
    q : NDArray[float64]
        Atomic coordinates.
    v : NDArray[float64]
        Velocity vector.
    d : NDArray[float64]
        Non-adiabatic coupling matrix.
    m : NDArray[float64]
        Atomic masses.
    epot : NDArray[float64]
        Potential energy values of all states.
    active_state : int
        Index of the active state.
    hopped_state : int
        Index of the hopped state.

    Returns
    -------
    tuple[NDArray[float64], int]
        Updated velocity vector and active state.
    """
    # Copy the velocity vector to not modify the original state
    v = v.copy()

    d_ah = d[active_state, hopped_state]  # same as d_{lambda,k} in the paper

    a = 1 / 2 * np.sum(d_ah**2 / m)
    b = np.sum(v * d_ah)
    c = epot[active_state] - epot[hopped_state]

    discr = b**2 + 4 * a * c

    if discr >= 0:  # Hop is not frustrated
        # Hop occurs
        active_state = hopped_state

        gamma = (b - np.sign(b) * np.sqrt(discr)) / (2 * a)

    else:  # No hop occurs, hop is frustrated
        if _check_reverse_velocity(q, v, d_ah, active_state, hopped_state):
            gamma = (
                b / a
            )  # Reverse the velocity along the direction of the coupling vector
        else:
            gamma = 0.0

    v = v - gamma * d_ah / m

    # Check if v contains complex values.
    # Will result in error if passed to RATTLE containing complex values.
    if np.iscomplexobj(v):
        if not v.imag.any(): # Check if all imaginary parts are zero, they should be
            v = v.real
        else:
            raise TypeError("Velocity array contains complex typed values which has non-zero imaginary parts.")

    return v, active_state


def _check_reverse_velocity(
    q: NDArray[float64],
    v: NDArray[float64],
    d_ah: NDArray[float64],
    active_state: int,
    hopped_state: int,
) -> bool:
    """
    Check if the velocity vector is reflected due to a frustrated hop as in:
    Jasper & Truhlar, Chem. Phys. Lett. 2003, 369 (1), 60–67. https://doi.org/10.1016/S0009-2614(02)01990-5.

    Parameters
    ----------
    q : NDArray[float64]
        Atomic coordinates.
    v : NDArray[float64]
        Velocity vector.
    d_ah : NDArray[float64]
        Non-adiabatic coupling vector.
    active_state : int
        Index of the active state.
    hopped_state : int
        Index of the hopped state.

    Returns
    -------
    bool
        True if the velocity vector is reflected, False otherwise.
    """

    force1 = PES_Force(q, active_state).reshape(-1,3)
    force2 = PES_Force(q, hopped_state).reshape(-1,3)

    fd1 = np.sum(force1 * d_ah)
    fd2 = np.sum(force2 * d_ah)
    vd = np.sum(v * d_ah)

    if fd1 * fd2 < 0.0 and fd1 * vd < 0.0:
        return True
    return False


def _decoherence():
    pass
