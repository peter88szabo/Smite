import numpy as np
import random

def thermo_andersen(nfix, p, wmass, dt, prob, Ttarg):
    """
    Set atomic momenta to fulfill
    the Maxwell-Boltzmann distribution
    same procedure as to initialize the momenta from MB distribution

    The formula for velocities:
    v(i)=sqrt(RT/m(i))*N

    where N is a random number with normal distirbution

    But we calculate here momenta instead of velocities
    """

    RT = (8.3144598/1000.0/2625.5) * Ttarg  #Rgas in Hartree/K

    if random.uniform(0.0,1.0) < prob*dt :
       for i in range(len(p)-nfix):
           p[i] = np.sqrt(wmass[i] * RT) * random.normalvariate(mu=0.0, sigma=1.0) 

    return p
