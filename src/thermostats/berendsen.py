import numpy as np
import random
from utils.constants import R_GAS_HARTREE_PER_K

def traj_temperature(nfix, p, wmass):
    '''
    Actual temperature of the system

    N*R*T/2 = Ekin

    where N is the number of degrees of freedom of the system

    nfix is used for the number of the fixeddegrees of freedom
    (e.g. nfix=3, if translational modes are frozen)
    '''
    Ekin=sum(p*p/wmass)*0.5

    ndof = len(p) - nfix
    if ndof <= 0:
        raise ValueError("Number of active degrees of freedom must be positive in traj_temperature()")

    return 2.0*Ekin/float(ndof)/R_GAS_HARTREE_PER_K


def thermo_berendsen(nfix, p, wmass, dt, tau, Ttarg):
    '''
    Berendsen thermostat
    Ttarg = target temperature
    tau = time constant,scaling (coupling) parameter
    tau has to be: tau > dt
    '''
    if(tau < dt):
       print("tau and dt: ", tau, dt)
       raise ValueError("Wrong tau paramter in Berendsen thermostat")

    Tact = traj_temperature(nfix, p, wmass)

    dum = dt * (Ttarg-Tact)/Tact/tau

    return p * np.sqrt(1.0 + dum)
