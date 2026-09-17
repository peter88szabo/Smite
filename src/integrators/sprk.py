import numpy as np
from integrators.gradient import force_calc 

class SPRK:
    def __init__(self, order):
        """
        Initialize the SPRK Integrator with the desired order.

        Parameters:
        - order: Order of the integrator (2 or 4).
        """
        if order == 2:
            # Strang Splitting (2nd Order)
            self.a_coeffs = [0.5, 0.5]
            self.b_coeffs = [0.0, 1.0]
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
        else:
            raise ValueError("Unsupported SPRK order. Choose 2 or 4.")

        self.order = order

    def sprk(self, qcinput, dt, wmass, q, p, atoms):
        num_stages = len(self.a_coeffs)

        for stage in range(num_stages):
            force = force_calc(qcinput, q, atoms)  
            p = p + self.b_coeffs[stage] * force * dt
            
            # Update position using coefficient a
            q = q + self.a_coeffs[stage] * p / wmass * dt

        return (q, p)

