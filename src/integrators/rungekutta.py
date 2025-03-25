from numpy._typing import NDArray

from integrators.gradient import force_calc


def rk4(
    qcinput: dict, dt: float, wmass: NDArray, q: NDArray, p: NDArray, atoms: list[str], active_state: int
) -> tuple[NDArray, NDArray]:
    """
    Runge-Kutta 4 integrator for conservative systems.
    """

    # Step 1
    f1 = force_calc(qcinput, q, atoms, active_state)
    v1 = p / wmass

    # Step 2
    q2 = q + 0.5 * dt * v1
    f2 = force_calc(qcinput, q2, atoms, active_state)
    v2 = (p + 0.5 * dt * f1) / wmass

    # Step 3
    q3 = q + 0.5 * dt * v2
    f3 = force_calc(qcinput, q3, atoms, active_state)
    v3 = (p + 0.5 * dt * f2) / wmass

    # Step 4
    q4 = q + dt * v3
    f4 = force_calc(qcinput, q4, atoms, active_state)
    v4 = (p + dt * f3) / wmass

    # Final update of q and p
    q += dt * (v1 + 2.0 * (v2 + v3) + v4) / 6.0
    p += dt * (f1 + 2.0 * (f2 + f3) + f4) / 6.0

    return q, p
