from numpy._typing import NDArray

from integrators.gradient import force_calc


def velverlet(
    qcinput: dict, dt: float, wmass: NDArray, q: NDArray, p: NDArray, atoms: list[str], active_state: int
) -> tuple[NDArray, NDArray]:
    force = force_calc(qcinput, q, atoms, active_state)

    q = q + (p / wmass) * dt + 0.5 * (force / wmass) * dt**2

    new_force = force_calc(qcinput, q, atoms, active_state)

    p = p + 0.5 * (force + new_force) * dt

    return q, p
