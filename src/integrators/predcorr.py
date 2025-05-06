import numpy as np
from numpy._typing import NDArray

from integrators.verlet import velverlet
from integrators.gradient import force_calc


class PredCorr:
    def __init__(self, order: int, ndim: int):
        self.order = order
        self.ab_coeff = None
        self.am_coeff = None
        self.step = -1
        self.save_veloc = np.zeros((order, ndim))
        self.save_force = np.zeros((order, ndim))

        self.get_adbash_admoul_coeffs()

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
        self.step += 1

        if self.step < self.order:
            return self.predcorr_initialize(qcinput, dt, wmass, q, p, atoms, active_state)
        elif self.step >= self.order:
            return self.call_predcorr(qcinput, dt, wmass, q, p, atoms, active_state)

    def get_adbash_admoul_coeffs(self) -> None:
        """
        Coefficients for the Adams-Bashford and
        Adams-Moulton integrators in arbitary order

        Reference:
        Baba Seidu, Int. J. Comp. Appl. Math., 2011, Vol. 6., pp. 215-220
        """
        # Adams-Bashforth coefficients
        a = np.array(
            [[j**i for j in range(self.order)] for i in range(self.order)], dtype=float
        )
        b = np.array([(-1) ** i / (i + 1) for i in range(self.order)], dtype=float)

        # Solve for Adams-Bashforth coefficients
        ab_coeff = np.linalg.solve(a, b)

        # Adams-Moulton coefficients
        a = np.array(
            [[(j - 1) ** i for j in range(self.order)] for i in range(self.order)],
            dtype=float,
        )
        b = np.array([(-1) ** i / (i + 1) for i in range(self.order)], dtype=float)

        # Solve for Adams-Moulton coefficients
        am_coeff = np.linalg.solve(a, b)

        # reverse order of vector elements
        self.ab_coeff = ab_coeff[::-1]
        self.am_coeff = am_coeff[::-1]

    def predcorr_initialize(
        self,
        qcinput: dict,
        dt: float,
        wmass: NDArray,
        q: NDArray,
        p: NDArray,
        atoms: list[str],
        active_state: int
    ) -> tuple[NDArray, NDArray]:
        """
        Initializer for arbitrary order Adams-Bashford---Adams-Moulton
        predictor-corrector integrator for Hamiltonian systems
        """
        if (self.step + 1) > self.order:
            raise ValueError(
                "the initializer step number must be smaller or equal than the order of the Predictor Corrector"
            )

        q, p = velverlet(qcinput, dt, wmass, q, p, atoms, active_state)

        self.save_veloc[self.step, :] = p / wmass
        self.save_force[self.step, :] = force_calc(qcinput, q, atoms, active_state)

        return q, p

    def call_predcorr(
        self,
        qcinput: dict,
        dt: float,
        wmass: NDArray,
        q: NDArray,
        p: NDArray,
        atoms: list[str],
        active_state: int
    ) -> tuple[NDArray, NDArray]:
        """
        Arbitrary order Adams-Bashford--Adams-Moulton
        predictor-corrector integrator for Hamiltonian systems.
        """
        q_old = np.copy(q)
        p_old = np.copy(p)

        # ----------------------------------------------------------
        # Adams-Bashford predictor step
        # ----------------------------------------------------------
        sum_q = np.zeros_like(q)
        sum_p = np.zeros_like(p)
        for j in range(self.order):
            sum_q += self.ab_coeff[j] * self.save_veloc[j, :]
            sum_p += self.ab_coeff[j] * self.save_force[j, :]

        q_pred = q_old + sum_q * dt
        p_pred = p_old + sum_p * dt

        force_pred = force_calc(qcinput, q_pred, atoms, active_state)

        # ----------------------------------------------------------
        # Adams-Moulton corrector step
        # ----------------------------------------------------------
        sum_q = np.zeros_like(q)
        sum_p = np.zeros_like(p)
        for j in range(1, self.order):
            sum_q += self.am_coeff[j - 1] * self.save_veloc[j, :]
            sum_p += self.am_coeff[j - 1] * self.save_force[j, :]

        # The last segment with updated force and momentum from the predictor step
        sum_q += self.am_coeff[-1] * p_pred / wmass
        sum_p += self.am_coeff[-1] * force_pred

        q_new = q_old + sum_q * dt
        p_new = p_old + sum_p * dt

        # ----------------------------------------------------------
        # Save the new velocity and force variables
        # ----------------------------------------------------------
        force_new = force_calc(
            qcinput, q_new, atoms, active_state
        )  # Recalculate force with corrected q

        # Shift indices of previous elements and discard the last element
        for j in range(1, self.order):
            self.save_veloc[j - 1, :] = self.save_veloc[j, :]
            self.save_force[j - 1, :] = self.save_force[j, :]

        # Update the last element with newly propagated q and p
        self.save_veloc[-1, :] = p_new / wmass
        self.save_force[-1, :] = force_new

        return q_new, p_new
