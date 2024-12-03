import math
import numpy as np
from integrators.verlet       import velverlet
from integrators.gradient     import force_calc

class PredCorr:
    def __init__(self, order, ndim):
        self.order = order
        self.ab_coeff = None
        self.am_coeff = None
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



    def get_adbash_admoul_coeffs(self):
        '''
        Coefficients for the Adams-Bashford and
        Adams-Moulton integrators in arbitary order

        Reference:
        Baba Seidu, Int. J. Comp. Appl. Math., 2011, Vol. 6., pp. 215-220
        '''
        # Adams-Bashforth coefficients
        a = np.array([[j ** i for j in range(self.order)] for i in range(self.order)], dtype=float)
        b = np.array([(-1) ** i / (i + 1) for i in range(self.order)], dtype=float)

        # Solve for Adams-Bashforth coefficients
        ab_coeff = np.linalg.solve(a, b)

        # Adams-Moulton coefficients
        a = np.array([[(j - 1) ** i for j in range(self.order)] for i in range(self.order)], dtype=float)
        b = np.array([(-1) ** i / (i + 1) for i in range(self.order)], dtype=float)

        # Solve for Adams-Moulton coefficients
        am_coeff = np.linalg.solve(a, b)

        #reverse order of vector elements
        self.ab_coeff = ab_coeff[::-1]
        self.am_coeff = am_coeff[::-1]

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
        for j in range(1, self.order):
            sum_q += self.am_coeff[j-1] * self.save_veloc[j,:]
            sum_p += self.am_coeff[j-1] * self.save_force[j,:]

        # The last segment with updated force and momentum from the predictor step
        sum_q += self.am_coeff[-1] * p_pred / wmass
        sum_p += self.am_coeff[-1] * force_pred

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

