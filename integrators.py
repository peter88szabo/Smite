import math
import numpy as np
from gradient import force_calc

def velverlet(qcinput, dt, mass, q, p, atoms):
    force = force_calc(qcinput, q, atoms)
    p = p + 0.5*force*dt

    q = q + p/mass*dt

    force = force_calc(qcinput, q, atoms)
    p = p + 0.5*force*dt
    return (q,p)




