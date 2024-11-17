import numpy as np
from integrators.gradient     import force_calc

def velverlet(qcinput, dt, wmass, q, p, atoms):
    force = force_calc(qcinput, q, atoms)

    q = q + (p / wmass) * dt + 0.5 * (force / wmass) * dt**2

    new_force = force_calc(qcinput, q, atoms)

    p = p + 0.5 * (force + new_force) * dt

    return (q, p)



def velverlet(qcinput, dt, wmass, q, p, atoms):
    force = force_calc(qcinput, q, atoms)
    p = p + 0.5*force*dt

    q = q + p/wmass*dt

    force = force_calc(qcinput, q, atoms)
    p = p + 0.5*force*dt
    return (q, p)




