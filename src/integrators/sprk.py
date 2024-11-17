import numpy as np
from integrators.gradient import force_calc 

class SPRK:
    def __init__(self, order):
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
                0.6756035959798289, -0.1756035959798288,
                -0.1756035959798288, 0.6756035959798289
            ]
            self.b_coeffs = [
                0.0, 1.3512071919596578,
                -1.7024143839193156, 1.3512071919596578
            ]
        elif order == 6:
            # Yoshida's 6th Order
            self.a_coeffs = [
                0.78451361047755726381949763,
                0.23557321335935813368479318,
                -1.17767998417887100694641568,
                1.0 - 2 * (0.78451361047755726381949763 +
                           0.23557321335935813368479318 +
                           -1.17767998417887100694641568)
            ]
            self.b_coeffs = [0.5 * a for a in self.a_coeffs]
        elif order == 8:
            # Yoshida's 8th Order
            self.a_coeffs = [
                0.74167036435061295344822780,
                -0.40910082580003159399730010,
                0.19075471029623837995387626,
                -0.57386247111608226665638773,
                0.29906418130365592384446354,
                0.33462491824529818378495798,
                1.0 - 2 * (0.74167036435061295344822780 +
                           -0.40910082580003159399730010 +
                           0.19075471029623837995387626 +
                           -0.57386247111608226665638773 +
                           0.29906418130365592384446354 +
                           0.33462491824529818378495798)
            ]
            self.b_coeffs = [0.5 * a for a in self.a_coeffs]
        else:
            raise ValueError("Unsupported order. Choose 2, 4, 6, or 8.")

        self.order = order

    def sprk(self, qcinput, dt, wmass, q, p, atoms):
        num_stages = len(self.a_coeffs)

        for stage in range(num_stages):
            force = force_calc(qcinput, q, atoms)  
            p = p + self.b_coeffs[stage] * force * dt
            
            # Update position using coefficient a
            q = q + self.a_coeffs[stage] * p / wmass * dt

        return (q, p)

