import numpy as np
import math
import random
import time
import sys
import logging
import os

#-----------------------------------------------------
# Source modules
#-----------------------------------------------------
from format_and_print import parseXYZ, makeXYZ
from format_and_print import print_trajectory
from format_and_print import check_and_create_file

from gradient    import Energy
from hessian     import getHessian 
from cenmass     import cenmass
from polyatom    import polyatom_init
from integrators import velverlet
from thermal     import thermal_vibr_mode
from nmodeprint  import print_normalmode
from eckart      import eckart_transform
from normalmode  import print_frequencies
#-----------------------------------------------------

#set number of procs
#lib.num_threads(8)
#print("number of processors:",lib.num_threads())

#time-stamp start
t_start = time.time()

c1 = 0.52917721092  # [bohr]    * c1 = [Ansgtrom] 
c3 = 1838.6836605e0 # [g/mol]   * c3 = [electron mass unit]
c6 = 41.341105      # [fs]      * c6 = [time in au]
c7 = 2625.5         # [Hartree] * c7 = [kJ/mol] 
c4 = 627.51         # [Hartree] * c4 = [kcal/mol]
c9=1.0e8/c1         #     [speed of light in atomic unit]
c10=137.035999074
Rgas = 8.3144598/1000.0/c7 #Hartree/K 

mH = 1.00782503223*c3
mC = 12.011*c3
mN = 14.007*c3
mO = 15.999*c3

#=========== Input parameters ====================
#qchem = 'PySCF'
qchem = 'Sparrow'

charge = 0
multiplicity = 1
#functional = 'b3lyp'
functional = 'dftb3'
base = ''
wfu = False

xyz = '''
  C   1.516     0.107   -0.213
  C  -1.516     0.106    0.213
  C   0.718    -1.061    0.274
  C  -2.092     1.010   -0.562
  C   2.092     1.011    0.562
  C  -0.717    -1.062   -0.274
  H   1.207    -1.989   -0.036
  H  -2.653     1.839   -0.150
  H   0.696    -1.060    1.367
  H  -2.021     0.951   -1.643
  H   2.020     0.952    1.643
  H  -0.695    -1.060   -1.367
  H   2.652     1.840    0.150
  H  -1.205    -1.990    0.036
  H   1.600     0.205   -1.293
  H  -1.600     0.204    1.293
 '''
Natoms, atoms, q_eq = parseXYZ(xyz)

q_eq = np.array(q_eq) / c1


#-----------------------------------------------------------------
#mass = [mC, mC, mC, mC, mC, mC, mH, mH, mH, mH, mH, mN, mO, mO]
mass =  [mC]*6 + [mH]*10

wmass = []   # auxiliary mass vector, same length as q and p
for ww in mass:
    wmass += [ww]*3
wmass = np.array(wmass)
#-----------------------------------------------------------------

hessFile = 'hessian_Cope15diene_DFTB3.hess'

hess = getHessian(qchem, hessFile, xyz, charge, multiplicity, functional, base)

#-----------------------------------------------------------------
# MD run parameters
#-----------------------------------------------------------------
temp = 100.0 #K
RT = Rgas * temp #in Hartree/K
dt = 0.3*c6 #fs --> automic time
nstep = 10000  
iprint = 10

jrot = 0

seed = 222286
random.seed(seed)
#-----------------------------------------------------------------

#=========== End parameters ====================

q = [0.0]*len(q_eq)
p = [0.0]*len(q_eq)

fix_vib = [41, 6]

hess_eckart = eckart_transform(mass, q_eq, hess)
q, p, freq, nvib = polyatom_init(RT, jrot, fix_vib, mass, q_eq, hess_eckart)

freq = np.array(freq)
nvib = np.array(nvib)

real_freq = freq[6:len(freq)]


assert len(real_freq) == len(nvib), "Vectors 'real_freq' and 'nvib' must have the same"


Ezero = 0.5*np.sum(real_freq)
Evib = np.sum(real_freq * (nvib + 0.5))


with open("quantum_states.dat", "w") as file:
        file.write("%20s %15.4f \n" % ("Ezero [kcal/mol] = ", Ezero*c4) )
        file.write("%20s %15.4f \n" % ("Evib  [kcal/mol] = ", Evib*c4) )
        file.write("%20s %15.4f \n" % ("Eexc  [kcal/mol] = ", (Evib-Ezero)*c4) )
        file.write("%20s %10.1f \n" % ("Temperature [K] = ", temp))
        for i in range(0, len(nvib)):
            file.write("%6d %8.0f %15.2f \n" % (i, nvib[i], freq[i+6]/c10*c9/(math.pi * 2)))


print_normalmode("mode_36.xyz",36, 20.0, atoms, mass, q_eq, hess_eckart)
print_normalmode("mode_37.xyz",37, 20.0, atoms, mass, q_eq, hess_eckart)
print_normalmode("mode_38.xyz",38, 20.0, atoms, mass, q_eq, hess_eckart)
print_normalmode("mode_39.xyz",39, 20.0, atoms, mass, q_eq, hess_eckart)
print_normalmode("mode_40.xyz",40, 20.0, atoms, mass, q_eq, hess_eckart)
print_normalmode("mode_41.xyz",41, 20.0, atoms, mass, q_eq, hess_eckart)

#sys.exit()

#=========================================================================================
# Run dynamics
#=========================================================================================

fname = "Cope15diene_"
traj_file = fname+"traj.xyz"
#traj_file = check_and_create_file(traj_file, '.xyz')

with open(traj_file, "a") as file_trj:
    file_wf = fname + ".delete_this_moldenfile"
    T0, V0, E0 = Energy(file_wf, qchem, q, p, atoms, wmass, charge, multiplicity, functional, base, wfu)
    for i in range(0,nstep):
        #----------------------------------------------------------------------------------
        if i % iprint == 0 or i == 0:
            file_wf = fname + "step_" + str(i) + ".molden"
            T, V, Ene = Energy(file_wf, qchem, q, p, atoms, wmass, charge, multiplicity, functional, base, wfu) 
            dE = Ene - E0
            print_trajectory(file_trj, atoms, q, p, V, dE, dt, i)
            print("%20s %10d %15s %20.2f" % ("trajectory step: " , i, "time[fs]: ", i*dt/c6))
            #-----------------------------------------------------------------------------------
        q,p = velverlet(qchem, dt, wmass, q, p, atoms, charge, multiplicity, functional, base)
#=========================================================================================

t_end = time.time()
print((t_end - t_start), "s")

