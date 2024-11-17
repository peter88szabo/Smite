import numpy as np
from integrators.gradient import force_calc

def stormer_verlet(qcinput, dt, wmass, q, p, atoms):
    # Update positions (q) using current momenta (p)
    q = q + p / wmass * dt

    # Calculate force at the new positions
    force = force_calc(qcinput, q, atoms)

    # Update momenta (p) using the new forces
    p = p + force * dt

    return (q, p)

