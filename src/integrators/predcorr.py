import numpy as np
from fractions import Fraction
from integrators.verlet       import velverlet
from integrators.gradient     import force_calc

class PredCorr:
    def __init__(self, order, ndim):
        if not isinstance(order, int) or order < 1:
            raise ValueError("Predictor-corrector order must be a positive integer")

        self.order = order
        self.ab_coeff = None
        self.am_coeff = None
        self.am_history_coeff = None
        self.am_predict_coeff = None
        self.step = -1
        self.save_veloc = np.zeros((order, ndim))
        self.save_force = np.zeros((order, ndim))

        self.get_adbash_admoul_coeffs()

    def predcorr(self, qcinput, dt, wmass, q, p, atoms):
        self.step += 1

        if(self.step < self.order):
            return self.predcorr_initialize(qcinput, dt, wmass, q, p, atoms)
        elif(self.step >= self.order):
            return self.call_predcorr(qcinput, dt, wmass, q, p, atoms) 



    @staticmethod
    def _solve_fraction_system(matrix, rhs):
        n = len(rhs)
        augmented = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]

        for col in range(n):
            pivot = None
            for row in range(col, n):
                if augmented[row][col] != 0:
                    pivot = row
                    break
            if pivot is None:
                raise ValueError("Singular Adams coefficient matrix")

            if pivot != col:
                augmented[col], augmented[pivot] = augmented[pivot], augmented[col]

            scale = augmented[col][col]
            augmented[col] = [value / scale for value in augmented[col]]

            for row in range(n):
                if row == col:
                    continue
                scale = augmented[row][col]
                if scale == 0:
                    continue
                augmented[row] = [
                    augmented[row][i] - scale * augmented[col][i]
                    for i in range(n + 1)
                ]

        return [augmented[row][-1] for row in range(n)]

    @classmethod
    def _adams_coefficients(cls, offsets):
        order = len(offsets)
        matrix = [
            [Fraction(offset) ** power for offset in offsets]
            for power in range(order)
        ]
        rhs = [Fraction(1, power + 1) for power in range(order)]
        return cls._solve_fraction_system(matrix, rhs)

    def get_adbash_admoul_coeffs(self):
        '''
        Coefficients for arbitrary-order Adams-Bashforth and
        Adams-Moulton predictor-corrector integration.

        Reference:
        Baba Seidu, Int. J. Comp. Appl. Math., 2011, Vol. 6., pp. 215-220
        '''
        # AB coefficients are computed newest-to-oldest:
        # f_n, f_{n-1}, ..., f_{n-order+1}. Store them oldest-to-newest to
        # match save_veloc/save_force history order.
        ab_offsets = [0 - j for j in range(self.order)]
        ab_coeff = self._adams_coefficients(ab_offsets)
        self.ab_coeff = np.array([float(value) for value in reversed(ab_coeff)])

        # AM coefficients are computed as:
        # f_{n+1}^{pred}, f_n, f_{n-1}, ..., f_{n-order+2}.
        # Keep the predicted endpoint coefficient separate so the history
        # alignment is explicit and does not depend on reversed indexing.
        am_offsets = [1 - j for j in range(self.order)]
        am_coeff = self._adams_coefficients(am_offsets)
        self.am_predict_coeff = float(am_coeff[0])
        self.am_history_coeff = np.array([float(value) for value in reversed(am_coeff[1:])])
        self.am_coeff = np.append(self.am_history_coeff, self.am_predict_coeff)

        return 

    def predcorr_initialize(self, qcinput, dt, wmass, q, p, atoms):
        '''
        Initializer for arbitrary order Adams-Bashford---Adams-Moulton
        predictor-corrector integrator for Hamiltonian systems
        '''
        if (self.step + 1) > self.order:
            raise ValueError("the initializer step number must be smaller or equal than the order of the Predictor Corrector")

        q,p = velverlet(qcinput, dt, wmass, q, p, atoms)

        self.save_veloc[self.step,:] = p / wmass
        self.save_force[self.step,:] = force_calc(qcinput, q, atoms) 

        return q, p

    def call_predcorr(self, qcinput, dt, wmass, q, p, atoms):
        """
        Arbitrary order Adams-Bashford--Adams-Moulton
        predictor-corrector integrator for Hamiltonian systems.
        """
        q_old = np.copy(q)
        p_old = np.copy(p)
        q_pred = np.zeros_like(q)
        p_pred = np.zeros_like(p)

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

        force_pred = force_calc(qcinput, q_pred, atoms)

        # ----------------------------------------------------------
        # Adams-Moulton corrector step
        # ----------------------------------------------------------
        sum_q = np.zeros_like(q)
        sum_p = np.zeros_like(p)
        for j in range(self.order - 1):
            sum_q += self.am_history_coeff[j] * self.save_veloc[j + 1, :]
            sum_p += self.am_history_coeff[j] * self.save_force[j + 1, :]

        # Predicted endpoint derivative f_{n+1}^{pred}.
        sum_q += self.am_predict_coeff * p_pred / wmass
        sum_p += self.am_predict_coeff * force_pred

        q_new = q_old + sum_q * dt
        p_new = p_old + sum_p * dt

        # ----------------------------------------------------------
        # Save the new velocity and force variables
        # ----------------------------------------------------------
        force_new = force_calc(qcinput, q_new, atoms)  # Recalculate force with corrected q

        # Shift indices of previous elements and discard the last element
        for j in range(1, self.order):
            self.save_veloc[j - 1, :] = self.save_veloc[j, :]
            self.save_force[j - 1, :] = self.save_force[j, :]

        # Update the last element with newly propagated q and p
        self.save_veloc[-1, :] = p_new / wmass
        self.save_force[-1, :] = force_new

        return q_new, p_new
