from numpy._typing import NDArray

from integrators.gradient import force_calc


def stormer_verlet(
    qcinput: dict, dt: float, wmass: NDArray, q: NDArray, p: NDArray, atoms: list[str], active_state: int
) -> tuple[NDArray, NDArray]:
    # Update positions (q) using current momenta (p)
    q = q + p / wmass * dt

    # Calculate force at the new positions
    force = force_calc(qcinput, q, atoms, active_state)

    # Update momenta (p) using the new forces
    p = p + force * dt

    return q, p
