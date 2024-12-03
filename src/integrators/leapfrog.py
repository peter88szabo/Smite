import numpy as np
from integrators.gradient     import force_calc


def leapfrog(qcinput, dt, wmass, q, p, atoms):
    force = force_calc(qcinput, q, atoms)
    p = p + 0.5*force*dt

    q = q + p/wmass*dt

    force = force_calc(qcinput, q, atoms)
    p = p + 0.5*force*dt
    return (q, p)




