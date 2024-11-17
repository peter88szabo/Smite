import numpy as np
import random

def traj_temperature(nfix, p, wmass):
    '''
    Actual temperature of the system

    N*R*T/2 = Ekin

    where N is the number of degrees of freedom of the system

    nfix is used for the number of the fixeddegrees of freedom
    (e.g. nfix=3, if translational modes are frozen)
    '''
    Rgas = (8.3144598/1000.0/2625.5) #in Hartree/K

    Ekin=sum(p*p/wmass)*0.5

    return 2.0*Ekin/float(len(p)-nfix)/Rgas


def random_initialize_momenta(p, wmass, Temp_init):
    """
    Initialize atomic momenta to fulfill
    the Maxwell-Boltzmann distribution

    The formula for velocities:
    v(i)=sqrt(RT/m(i))*N

    where N is a random number with normal distirbution

    But we calculate here momenta instead of velocities
    """

    RT = (8.3144598/1000.0/2625.5) * Temp_init  #Rgas in Hartree/K

    for i in range(len(p)):
        p[i] = np.sqrt(wmass[i] * RT) * random.normalvariate(mu=0.0, sigma=1.0)

    return p


def thermo_berendsen(nfix, p, wmass, dt, tau, Ttarg):
    '''
    Berendsen thermostate
    Ttarg = target temperature
    tau = time constant,scaling (coupling) parameter
    tau has to be: tau > dt
    '''

    if(tau < dt):
       raise ValueError("Wrong tau paramter in Berendsen thermostat")

    Tact = traj_temperature(nfix, p, wmass)

    dum = dt * (Ttarg-Tact)/Tact/tau

    return p * np.sqrt(1.0 + dum)
