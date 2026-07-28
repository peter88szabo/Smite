import numpy as np
import random
from utils.constants import R_GAS_HARTREE_PER_K

def random_initialize_momenta(wmass, Temp_init):
    """
    Initialize atomic momenta to fulfill
    the Maxwell-Boltzmann distribution

    The formula for velocities:
    v(i)=sqrt(RT/m(i))*N

    where N is a random number with normal distirbution

    But we calculate here momenta instead of velocities
    """

    RT = R_GAS_HARTREE_PER_K * Temp_init


    p = np.zeros(len(wmass))

    for i in range(len(wmass)):
        p[i] = np.sqrt(wmass[i] * RT) * random.normalvariate(mu=0.0, sigma=1.0)

    return np.array(p)
