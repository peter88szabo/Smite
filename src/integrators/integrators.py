import math
import numpy as np

from utils.atomic_masses      import get_mass_vector
from integrators.gradient     import force_calc
from utils.format_and_print   import parseXYZ, print_trajectory
from utils.format_and_print   import print_trajectory


def velverlet(qcinput, dt, wmass, q, p, atoms):
    force = force_calc(qcinput, q, atoms)
    p = p + 0.5*force*dt

    q = q + p/wmass*dt

    force = force_calc(qcinput, q, atoms)
    p = p + 0.5*force*dt
    return (q, p)


if __name__ == '__main__':
    c1   = 0.52917721092         # [bohr]     * c1 = [Ansgtrom]
    c3   = 1838.6836605e0        # [g/mol]    * c3 = [electron mass unit]
    c5   = 219474.e0            # [Hartree]  * c5 = [cm-1]
    c6   = 41.341105             # [fs]       * c6 = [time in au]
    c7   = 2625.5                # [Hartree]  * c7 = [kJ/mol]
    c9   = 1.0e8/c1             # [freqcm-1] *c9=[freq(bohr^(-1))]
    c10  = 137.035999074        # [speed of light in atomic unit]
    Rgas = 8.3144598/1000.0/c7 #Hartree/K


    xyz_water = '''
      O    -0.011100  0.0000  -0.00788
      H     0.007500  0.0000   0.95111
      H     0.899200  0.0000  -0.30990
       '''

    Natoms, atoms, q_eq = parseXYZ(xyz_water)
    q = np.array(q_eq) / 0.52917721092

    mass = get_mass_vector(atoms)
    wmass = np.repeat(mass, 3)

    p = np.zeros(len(q))

    print("atoms of water molecule:", Natoms)
    print(atoms)
    print("mass:", mass)
    print("wmass:", wmass)
    print("q: ", q)
    print("p: ", p)


    filename = "asdasd.xzy"
    V = 1000.0
    dE = 2000.0
    dt = 0.1
    istep = 1


    qcinput = {
    'qchem': 'XTB',
    'path': '/home/peter/orca_6_0_0/xtb',
    'nproc': 1,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 1,
    'additional': '--acc 10',
    'wfu': False
    }


    q, p = velverlet(qcinput, dt, wmass, q, p, atoms)
    print("q: ", q)
    print("p: ", p)

    trajfile=open(filename,'w')
    print_trajectory(trajfile, atoms, q, p, V, dE, dt, istep)




