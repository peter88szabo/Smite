from typing import Any

import numpy as np
from numba import jit
from numpy._typing import NDArray, _64Bit
from numpy import float64, ndarray, dtype, floating

from integrators.gradient import force_calc

class Rattle:
    def __init__(self, q_init, constrained_bonds):
        if constrained_bonds is None:
            raise NotImplementedError("The constrained bonds are not specified.")
        if isinstance(constrained_bonds, list):
            constrained_bonds = np.array(constrained_bonds)
        if len(q_init.shape) == 1:
            q_init = q_init.reshape(-1,3)

        self.q_init = q_init
        self.constrained_bonds = constrained_bonds
        self.fixed_internals = _generate_fixed_internals(q_init, constrained_bonds)

    def rattle(self,
        qcinput: dict,
        dt: float,
        mass: NDArray[float64],
        q: NDArray[float64],
        p: NDArray[float64],
        atoms: list,
        active_state: int,
        tol: float = 1e-8,
    ) -> tuple[NDArray[float64], NDArray[float64]]:
        """
        The RATTLE algorithm for performing molecular dynamics with holonomic constraints.
        Only implemented for constrained bonds, not angles.
        The implementation is derived from:
        RATTLE Recipe for General Holonomic Constraints: Angle and Torsion Constraints, R. Kutteh

        Parameters
        ----------
        qcinput : dict
            The input dictionary containing the quantum chemical information.
        dt : float
            The time step.
        mass : NDArray[float64]
            The mass of the atoms.
        q : NDArray[float64]
            The positions.
        p : NDArray[float64]
            The momenta.
        atoms : list
            The list of atoms.
        constrained_bonds : list[list] | NDArray[float64]
            The constrained bonds.
        tol : float
            The tolerance for the constraints.

        Returns
        -------
        tuple[NDArray[float64], NDArray[float64]]
            The updated positions and momenta.
        """

        # Transform coordinates and mass arrays into 2D matrices
        q, mass, p = q.reshape(-1, 3), mass.reshape(-1, 3), p.reshape(-1, 3)

        # Transform momenta to velocities to simplify the equations
        v = p / mass

        # RATTLE update
        q_new, v_new = _propagate(
            qcinput,
            mass,
            dt,
            q,
            v,
            atoms,
            active_state,
            self.fixed_internals,
            self.constrained_bonds,
            tol,
        )

        p_new = v_new * mass

        q_new, p_new = q_new.reshape(q.shape[0] * 3), p_new.reshape(p.shape[0] * 3)

        return q_new, p_new


def _generate_fixed_internals(
    q: NDArray[float64], constrained_bonds: NDArray[float64]
) -> NDArray[float64]:
    """
    Generate the fixed internal bond distances for the constrained bonds.

    Parameters
    ----------
    q : NDArray[float64]
        The positions.
    constrained_bonds : NDArray[float64]
        The constrained bonds.

    Returns
    -------
    NDArray[float64]
        The fixed internal bond distances.
    """
    num_bonds = constrained_bonds.shape[0]
    fixed_internals = np.zeros(num_bonds)

    for idx, bond in enumerate(constrained_bonds):
        fixed_internals[idx] = np.linalg.norm(q[bond[0], :] - q[bond[1], :])

    return fixed_internals


def _propagate(
    qcinput: dict,
    mass: NDArray[float64],
    dt: float,
    q: NDArray[float64],
    v: NDArray[float64],
    atoms: list,
    active_state: int,
    fixed_internals: NDArray[float64],
    constrained_bonds: NDArray[float64],
    tol: float,
) -> tuple[ndarray[Any, dtype[float64]], ndarray[Any, dtype[float64]]]:
    """
    Propagate the positions and velocities using the RATTLE algorithm.

    Parameters
    ----------
    qcinput : dict
        The input dictionary.
    mass : NDArray[float64]
        The mass of the atoms.
    dt : float
        The time step.
    q : NDArray[float64]
        The positions.
    v : NDArray[float64]
        The velocities.
    atoms : list
        The list of atoms.
    fixed_internals : NDArray[float64]
        The fixed internal bond distances.
    constrained_bonds : NDArray[float64]
        The constrained bonds.
    tol : float
        The tolerance for the constraints.

    Returns
    -------
    tuple[NDArray[float64], NDArray[float64]]
        The updated positions and velocities
    """
    q_t0 = q.copy()
    v_t0 = v.copy()

    # Unconstrained step
    q_dt, v_dt_half = leapfrog_halfstep(
        qcinput, active_state, mass, dt, q_t0, v_t0, atoms
    )

    q_constraints, v_constraints = (
        np.zeros(len(constrained_bonds)),
        np.zeros(len(constrained_bonds)),
    )
    q_constraints = update_distance_constraints(
        q_dt, fixed_internals, q_constraints, constrained_bonds
    )

    q_dt_corr = coords_corr(
        q_t0, q_dt, q_constraints, constrained_bonds, fixed_internals, mass, dt, tol
    )

    # Second half-step to update the velocities with the force calculated with the new t+dt coords
    _, v_dt = leapfrog_halfstep(
        qcinput, active_state, mass, dt, q_dt_corr, v_dt_half, atoms
    )

    # Determine the constraints on the velocities for the new coordinates
    v_constraints = update_velocity_constraints(
        q_dt_corr, v_dt, v_constraints, constrained_bonds
    )

    # Correct the velocities
    v_dt_corr = v_corr(q_dt, v_dt, v_constraints, constrained_bonds, mass, dt, tol)

    return q_dt_corr, v_dt_corr


def leapfrog_halfstep(
    qcinput: dict,
    active_state: int,
    mass: NDArray[float64],
    dt: float,
    q: NDArray[float64],
    v: NDArray[float64],
    atoms: list,
) -> tuple[NDArray[float64], NDArray[float64]]:
    """
    Perform a half-step leapfrog integration to update the positions and velocities.

    Parameters
    ----------
    qcinput : dict
        The input dictionary.
    mass : NDArray[float64]
        The mass of the atoms.
    dt : float
        The time step.
    q : NDArray[float64]
        The positions.
    v : NDArray[float64]
        The velocities.
    atoms : list
        The list of atoms.

    Returns
    -------
    tuple[NDArray[float64], NDArray[float64]]
        The updated positions and velocities
    """
    q_dt = q.copy()
    v_dt_half = v.copy()

    q_dt_arr = q_dt.reshape(q.shape[0] * 3)
    f = force_calc(qcinput, q_dt_arr, atoms, active_state)
    f = f.reshape(-1, 3)

    v_dt_half += 1 / (2 * mass) * dt * f  # Half-step
    q_dt += dt * v_dt_half

    return q_dt, v_dt_half


@jit(nopython=True)
def distance_constraint(
    a: NDArray[float64], b: NDArray[float64], fixed_internal: float
) -> float:
    """
    The distance constraint is the difference between the squared distance vector
    and the squared constrained bond distance.

    g = r_ij^2 - d_ij^2

    Parameters
    ----------
    a : NDArray[float64]
        The coordinates of atom a.
    b : NDArray[float64]
        The coordinates of atom b.
    fixed_internal : float
        The fixed internal bond distance.

    Returns
    -------
    float
        The constraint value.
    """
    constraint = np.dot((b - a), (b - a)) - fixed_internal**2
    return constraint


@jit(nopython=True)
def velocity_constraint(
    q_dt: list[NDArray[float64], NDArray[float64]],
    v_dt: list[NDArray[float64], NDArray[float64]],
) -> float:
    """
    The velocity constraints removes the velocity along the bond.

    Parameters
    ----------
    q_dt : tuple[NDArray[float64], NDArray[float64]]
        The coordinates at time t+dt.
    v_dt : tuple[NDArray[float64], NDArray[float64]]
        The velocities at time t+dt.

    Returns
    -------
    float
        The constraint value.
    """
    diff_coords_dt = q_dt[0] - q_dt[1]
    diff_velocities_dt = v_dt[0] - v_dt[1]

    # Are the position and velocity vectors perpendicular i.e. zero dot product?
    # If zero, there is no velocity along the constrained bond
    constraint = np.dot(diff_coords_dt, diff_velocities_dt)
    return constraint


@jit(nopython=True)
def gamma_iterate(
    q_t0: list[NDArray[float64], NDArray[float64]],
    q_dt: list[NDArray[float64], NDArray[float64]],
    constraint: float,
    m_a: float,
    m_b: float,
    fixed_internal: float,
    dt: float,
    tol: float,
) -> tuple[NDArray[float64], NDArray[float64]]:
    """
    Iterate over different values of gamma until the distance constraint between atoms a and b is below the tolerance.

    Parameters
    ----------
    q_t0 : tuple[NDArray[float64], NDArray[float64]]
        The coordinates at time t0.
    q_dt : tuple[NDArray[float64], NDArray[float64]]
        The coordinates at time t+dt.
    fixed_internal : float
        The fixed internal bond distance.
    constraint : float
        The constraint value.
    m_a : float
        The mass of atom a.
    m_b : float
        The mass of atom b.
    dt : float
        The time step.
    tol : float
        The tolerance for the constraints.

    Returns
    -------
    tuple[NDArray[float64], NDArray[float64]]
        The updated coordinates.
    """
    # Define a number of constants
    reciprocal_reduced_mass = 1 / m_a + 1 / m_b
    diff_q_t0 = q_t0[0] - q_t0[1]
    coefficient_a = dt**2 * 1 / m_a
    coefficient_b = dt**2 * 1 / m_b

    # Calculate the gamma for the first correction step
    gamma = gamma_step(diff_q_t0, q_dt, constraint, reciprocal_reduced_mass, dt)

    # Update coords by reducing the constraint below the tol
    while abs(constraint) > tol:
        q_dt[0] -= coefficient_a * gamma * diff_q_t0
        q_dt[1] -= coefficient_b * gamma * -diff_q_t0

        constraint = distance_constraint(q_dt[0], q_dt[1], fixed_internal)

        gamma = gamma_step(diff_q_t0, q_dt, constraint, reciprocal_reduced_mass, dt)

    return q_dt[0], q_dt[1]


@jit(nopython=True)
def gamma_step(
    diff_q_t0: NDArray[float64],
    q_dt: list[NDArray[float64], NDArray[float64]],
    constraint: float,
    reciprocal_reduced_mass: float,
    dt: float,
) -> float:
    """
    Take a single step to update the gamma (distance) constraint parameter.

    Parameters
    ----------
    diff_q_t0 : NDArray[float64]
        The difference in coordinates at time t0.
    q_dt : NDArray[float64]
        The coordinates at time t+dt.
    constraint : float
        The constraint value.
    reciprocal_reduced_mass : float
        The reciprocal reduced mass.
    dt : float
        The time step.

    Returns
    -------
    float
        The updated gamma value.
    """
    dp_diff_vectors = np.dot(diff_q_t0, q_dt[0] - q_dt[1])

    gamma = 1 / dt**2 * constraint / (2 * reciprocal_reduced_mass * dp_diff_vectors)
    return gamma


@jit(nopython=True)
def eta_step(
    diff_q_dt: NDArray[float64],
    dotprod_diff_q_dt: float,
    v_dt: list[NDArray[float64], NDArray[float64]],
    reciprocal_reduced_mass: float,
    dt: float,
) -> float:
    """
    Take a single step to update the eta (velocity) constraint parameter.

    Parameters
    ----------
    diff_q_dt : NDArray[float64]
        The difference in coordinates at time t+dt.
    dotprod_diff_q_dt : float
        The dot product of the difference in coordinates at time t+dt.
    v_dt : NDArray[float64]
        The velocities at time t+dt.
    reciprocal_reduced_mass : float
        The reciprocal reduced mass.
    dt : float
        The time step.

    Returns
    -------
    float
        The updated eta value.
    """
    diff_velocities_dt = v_dt[0] - v_dt[1]
    eta = (
        1
        / dt
        * np.dot(diff_q_dt, diff_velocities_dt)
        / (dotprod_diff_q_dt * reciprocal_reduced_mass)
    )

    return eta


@jit(nopython=True)
def eta_iterate(
    q_dt: list[NDArray[float64], NDArray[float64]],
    v_dt: list[NDArray[float64], NDArray[float64]],
    constraint: float,
    m_a: float,
    m_b: float,
    dt: float,
    tol: float,
) -> tuple[NDArray[float64], NDArray[float64]]:
    """
    Iterate over different values of eta until the velocity constraint is satisfied.
    Here, diff_coords_dt is the constraint force as determined by the optimized coordinates.

    Parameters
    ----------
    q_dt : tuple[NDArray[float64], NDArray[float64]]
        The list of coordinates at time t+dt for atoms a and b.
    v_dt : tuple[NDArray[float64], NDArray[float64]]
        The list of velocities at time t+dt for atoms a and b.
    constraint : float
        The constraint value.
    m_a : float
        The mass of atom a.
    m_b : float
        The mass of atom b.
    dt : float
        The time step.
    tol : float
        The tolerance for the constraints.

    Returns
    -------
    tuple[NDArray[float64], NDArray[float64]
        The updated velocities for atoms a and b.
    """
    # Define a number of constants
    reciprocal_reduced_mass = 1 / m_a + 1 / m_b
    diff_q_dt = q_dt[0] - q_dt[1]
    dotprod_diff_q_dt = np.dot(diff_q_dt, diff_q_dt)
    coefficient_a = dt * 1 / m_a
    coefficient_b = dt * 1 / m_b

    eta = eta_step(diff_q_dt, dotprod_diff_q_dt, v_dt, reciprocal_reduced_mass, dt)

    while np.abs(constraint) > tol:
        v_dt[0] -= coefficient_a * eta * diff_q_dt
        v_dt[1] -= coefficient_b * eta * -diff_q_dt

        constraint = velocity_constraint(q_dt, v_dt)

        eta = eta_step(diff_q_dt, dotprod_diff_q_dt, v_dt, reciprocal_reduced_mass, dt)

    return v_dt[0], v_dt[1]


@jit(nopython=True)
def update_distance_constraints(
    q_dt: NDArray[float64],
    fixed_internals: NDArray[float64],
    constraints: NDArray[float64],
    constrained_bonds: NDArray[float64],
) -> np.ndarray:
    """
    Update all distance constraints as specified by the bonds dictionary.

    Parameters
    ----------
    q_dt : NDArray[float64]
        The coordinates at time t+dt.
    fixed_internals : NDArray[float64]
        The fixed internal bond distances.
    constraints : NDArray[float64]
    constrained_bonds : NDArray[float64]
        The constrained bonds.

    Returns
    -------
    NDArray[float64]
        The updated constraints.
    """

    num_bonds = constrained_bonds.shape[0]

    for bond in range(num_bonds):
        atom_1 = constrained_bonds[bond, 0]
        atom_2 = constrained_bonds[bond, 1]

        constraint = distance_constraint(
            a=q_dt[atom_1, :], b=q_dt[atom_2, :], fixed_internal=fixed_internals[bond]
        )
        constraints[bond] = constraint

    return constraints


@jit(nopython=True)
def update_velocity_constraints(
    q_dt: NDArray[float64],
    v_dt: NDArray[float64],
    constraints: NDArray[float64],
    constrained_bonds: NDArray[float64],
) -> NDArray:
    """
    Update all velocity constraints as specified by the bonds dictionary.

    Parameters
    ----------
    q_dt : NDArray[float64]
        The coordinates at time t+dt.
    v_dt : NDArray[float64]
        The velocities at time t+dt.
    constraints : NDArray[float64]
        The constraints.
    constrained_bonds : NDArray[float64]
        The constrained bonds.

    Returns
    -------
    NDArray[float64]
        The updated constraints.
    """
    num_bonds = constrained_bonds.shape[0]

    for bond in range(num_bonds):
        atom_1 = constrained_bonds[bond, 0]
        atom_2 = constrained_bonds[bond, 1]

        constraint = velocity_constraint(
            q_dt=[q_dt[atom_1, :], q_dt[atom_2, :]],
            v_dt=[v_dt[atom_1, :], v_dt[atom_2, :]],
        )
        constraints[bond] = constraint

    return constraints


@jit(nopython=True)
def coords_corr(
    q_t0: NDArray[float64],
    q_dt: NDArray[float64],
    constraints: NDArray[float64],
    constrained_bonds: NDArray[float64],
    fixed_internals: NDArray[float64],
    mass: NDArray[float64],
    dt: float,
    tol: float,
) -> NDArray[float64]:
    """
    Correct the coordinates to satisfy the constraints.

    Parameters
    ----------
    q_t0 : NDArray[float64]
        The coordinates at time t0.
    q_dt : NDArray[float64]
        The coordinates at time t+dt.
    constraints : NDArray[float64]
        The constraints.
    constrained_bonds : NDArray[float64]
        The constrained bonds.
    fixed_internals : NDArray[float64]
        The fixed internal bond distances.
    mass : NDArray[float64]
        The mass of the atoms.
    dt : float
        The time step.
    tol : float
        The tolerance for the constraints.

    Returns
    -------
    NDArray[float64]
        The corrected coordinates.
    """
    num_bonds = constrained_bonds.shape[0]

    while np.any(np.abs(constraints) > tol):
        for bond in range(num_bonds):
            atom_1 = constrained_bonds[bond, 0]
            atom_2 = constrained_bonds[bond, 1]

            q_dt[atom_1, :], q_dt[atom_2, :] = gamma_iterate(
                q_t0=[q_t0[atom_1, :], q_t0[atom_2, :]],
                q_dt=[q_dt[atom_1, :], q_dt[atom_2, :]],
                constraint=constraints[bond],
                m_a=mass[atom_1, atom_1],
                m_b=mass[atom_2, atom_2],
                fixed_internal=fixed_internals[bond],
                dt=dt,
                tol=tol,
            )
            constraints = update_distance_constraints(
                q_dt, fixed_internals, constraints, constrained_bonds
            )

    return q_dt


@jit(nopython=True)
def v_corr(
    q_dt: NDArray[float64],
    v_dt: NDArray[float64],
    constraints: NDArray[float64],
    constrained_bonds: NDArray[float64],
    mass: NDArray[float64],
    dt: float,
    tol: float,
) -> NDArray[float64]:
    """
    Correct the velocities to satisfy the constraints.

    Parameters
    ----------
    q_dt : NDArray[float64]
        The coordinates at time t+dt.
    v_dt : NDArray[float64]
        The velocities at time t+dt.
    constraints : NDArray[float64]
        The constraints.
    constrained_bonds : NDArray[float64]
        The constrained bonds.
    mass : NDArray[float64]
        The mass of the atoms.
    dt : float
        The time step.
    tol : float
        The tolerance for the constraints.

    Returns
    -------
    NDArray[float64]
        The corrected velocities.
    """
    q_dt = q_dt.copy()
    v_dt = v_dt.copy()

    num_bonds = constrained_bonds.shape[0]

    while np.any(np.abs(constraints) > tol):
        for bond in range(num_bonds):
            atom_1 = constrained_bonds[bond, 0]
            atom_2 = constrained_bonds[bond, 1]

            v_dt[atom_1, :], v_dt[atom_2, :] = eta_iterate(
                q_dt=[q_dt[atom_1, :], q_dt[atom_2, :]],
                v_dt=[v_dt[atom_1, :], v_dt[atom_2, :]],
                constraint=constraints[bond],
                m_a=mass[atom_1, atom_1],
                m_b=mass[atom_2, atom_2],
                dt=dt,
                tol=tol,
            )

            constraints = update_velocity_constraints(
                q_dt, v_dt, constraints, constrained_bonds
            )
    return v_dt
