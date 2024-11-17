import numpy as np
from integrators.gradient     import force_calc

def rk4(qcinput, dt, wmass, q, p, atoms):
    """
    Runge-Kutta 4 integrator for conservative systems.
    """
    qold = np.copy(q)
    pold = np.copy(p)
    
    # Step 1
    f1 = force_calc(qcinput, q, atoms)
    g1 = p / wmass
    
    # Step 2
    q2 = qold + 0.5 * dt * g1
    p2 = pold + 0.5 * dt * f1
    f2 = force_calc(qcinput, q2, atoms)
    g2 = p2 / wmass
    
    # Step 3
    q3 = qold + 0.5 * dt * g2
    p3 = pold + 0.5 * dt * f2
    f3 = force_calc(qcinput, q3, atoms)
    g3 = p3 / wmass
    
    # Step 4
    q4 = qold + dt * g3
    p4 = pold + dt * f3
    f4 = force_calc(qcinput, q4, atoms)
    g4 = p4 / wmass
    
    # Final update of q and p
    q = qold + dt * (g1 + 2.0 * (g2 + g3) + g4) / 6.0
    p = pold + dt * (f1 + 2.0 * (f2 + f3) + f4) / 6.0

    return q, p

