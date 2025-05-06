from numpy._typing import NDArray

from integrators.gradient import force_calc


class SPRK:
    def __init__(self, order: int):
        """
        Initialize the SPRK Integrator with the desired order.

        Parameters:
        - order: Order of the integrator (2, 4, 6, or 8 currently supported).
        """
        if order == 2:
            # Strang Splitting (2nd Order)
            self.a_coeffs = [0.5, 0.5]
            self.b_coeffs = [1.0, 0.0]
        elif order == 4:
            # Forest-Ruth (4th Order)
            self.a_coeffs = [
                0.6756035959798289,
                -0.1756035959798288,
                -0.1756035959798288,
                0.6756035959798289,
            ]
            self.b_coeffs = [
                0.0,
                1.3512071919596578,
                -1.7024143839193156,
                1.3512071919596578,
            ]
        else:
            raise ValueError("Unsupported order. Choose 2, 4, 6, or 8.")

        self.order = order

    def __call__(
        self,
        qcinput: dict,
        dt: float,
        wmass: NDArray,
        q: NDArray,
        p: NDArray,
        atoms: list[str],
        active_state: int
    ) -> tuple[NDArray, NDArray]:
        num_stages = len(self.a_coeffs)

        for stage in range(num_stages):
            force = force_calc(qcinput, q, atoms, active_state)
            p = p + self.b_coeffs[stage] * force * dt

            # Update position using coefficient a
            q = q + self.a_coeffs[stage] * p / wmass * dt

        return q, p
