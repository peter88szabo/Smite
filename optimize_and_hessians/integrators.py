import math
import numpy as np
from gradient import force_calc

def velverlet(qchem, dt,mass,q, p, atoms, charge, multiplicity, functional, base):
    force = force_calc(qchem, q, atoms, charge, multiplicity, functional, base)
    p = p + 0.5*force*dt

    q = q + p/mass*dt

    force = force_calc(qchem, q, atoms, charge, multiplicity, functional, base)
    p = p + 0.5*force*dt
    return (q,p)




