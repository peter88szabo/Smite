import numpy as np
from integrators.gradient     import force_calc

def rk4(qcinput, dt, wmass, q, p, atoms):
    """
    Runge-Kutta 4 integrator for conservative systems.
    """
    
    # Step 1
    f1 = force_calc(qcinput, q, atoms)
    v1 = p / wmass
    
    # Step 2
    q2 = q + 0.5 * dt * v1
    f2 = force_calc(qcinput, q2, atoms)
    v2 = (p + 0.5 * dt * f1) / wmass
    
    # Step 3
    q3 = q + 0.5 * dt * v2
    f3 = force_calc(qcinput, q3, atoms)
    v3 = (p + 0.5 * dt * f2) / wmass
    
    # Step 4
    q4 = q + dt * v3
    f4 = force_calc(qcinput, q4, atoms)
    v4 = (p + dt * f3) / wmass
    
    # Final update of q and p
    q += dt * (v1 + 2.0 * (v2 + v3) + v4) / 6.0
    p += dt * (f1 + 2.0 * (f2 + f3) + f4) / 6.0

    return (q, p)

