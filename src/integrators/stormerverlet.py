import numpy as np
from integrators.gradient import force_calc

def stormer_verlet(qcinput, dt, wmass, q, p, atoms):
    """Second-order kick-drift-kick Störmer-Verlet propagation."""
    force = force_calc(qcinput, q, atoms)

    # Half kick at q_n.
    p = p + 0.5 * force * dt

    # Full drift to q_{n+1}, followed by the force at the new position.
    q = q + p / wmass * dt
    force = force_calc(qcinput, q, atoms)

    # Half kick to p_{n+1}.
    p = p + 0.5 * force * dt

    return (q, p)

