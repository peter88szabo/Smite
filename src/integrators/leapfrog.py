from numpy._typing import NDArray

from integrators.gradient import force_calc


def leapfrog(
    qcinput: dict, dt: float, wmass: NDArray, q: NDArray, p: NDArray, atoms: list[str]
) -> tuple[NDArray, NDArray]:
    force = force_calc(qcinput, q, atoms)
    p = p + 0.5 * force * dt

    q = q + p / wmass * dt

    force = force_calc(qcinput, q, atoms)
    p = p + 0.5 * force * dt
    return q, p
