import numpy as np
import random
import time
import sys
import logging
import os

#-----------------------------------------------------
# Source modules
#-----------------------------------------------------
from format_and_print import parseXYZ, makeXYZ
from format_and_print import parseCheckPoint
from format_and_print import print_trajectory
from format_and_print import check_and_create_file

from gradient    import Energy
from hessian     import getHessian 
from cenmass     import cenmass
from polyatom    import polyatom_init
from integrators import velverlet
from thermal     import thermal_vibr_mode
from nmodeprint  import print_normalmode
from nmodeprint  import print_normalmode_with_fix_quantum 
from eckart      import eckart_transform
from normalmode  import print_frequencies
#-----------------------------------------------------

#set number of procs
#lib.num_threads(4)
#print("number of processors:",lib.num_threads())
#print()

#time-stamp start
t_start = time.time()

c1 = 0.52917721092  # [bohr]    * c1 = [Ansgtrom] 
c3 = 1838.6836605e0 # [g/mol]   * c3 = [electron mass unit]
c6 = 41.341105      # [fs]      * c6 = [time in au]
c7 = 2625.5         # [Hartree] * c7 = [kJ/mol] 
Rgas = 8.3144598/1000.0/c7 #Hartree/K 

mH = 1.00782503223*c3
mC = 12.011*c3
mN = 14.007*c3
mO = 15.999*c3

#=========== Input parameters ====================
qchem = 'Orca'
functional = 'b3lyp'
base = 'pc-2'
charge = 0
multiplicity = 2
path = '/home/peter/Programs/Orca5.04/orca'
#path = '/home/peter/Programs/Orca_5.0.4/orca'
nproc = 4

#-----------------------------------------------------------------
# MD run parameters
#-----------------------------------------------------------------
temp = 300.0 #K
RT = Rgas * temp #in Hartree/K
dt = 0.5*c6 #fs --> automic time
nstep = 200
iprint = 5

jrot = 0
fix_vib = [0, 3]

restart = False

seed = 122986
random.seed(seed)
#-----------------------------------------------------------------

qcinput = [0]*8
qcinput[0] = qchem
qcinput[1] = path
qcinput[2] = nproc
qcinput[3] = functional
qcinput[4] = base
qcinput[5] = charge
qcinput[6] = multiplicity
qcinput[7] = ''

# B3LYP pc-2 opt 
xyz = '''
  C   0.05945380790845      0.20453663646409      0.10232295614842
  H   0.05129027268353     -0.13431973543756      1.12566939847397
  H   0.88249226237116     -0.05194703450250     -0.54491588227273
  H   -0.75746034296314      0.79848013347598     -0.27470147234966
 '''
Natoms, atoms, q_eq = parseXYZ(xyz)

q_eq = np.array(q_eq) / c1


#-----------------------------------------------------------------
mass =  [mC]*1 + [mH]*3

wmass = []   # auxiliary mass vector, same length as q and p
for ww in mass:
    wmass += [ww]*3
wmass = np.array(wmass)
#-----------------------------------------------------------------
fname = "Nitro-Benzene_"
traj_file = fname+"traj.xyz"
vibfile = fname+"freq_and_sampled_vib_quanta.vib"
backfile = fname+"checkpoint.xyz"


############################################################################################
if not restart:
    start_step = 0
    #hessFile = "hessian_Nitro-Benzene_b3lyp_pc-1.hess"
    hessFile = 'hessian_' + fname + functional + '_' + base + '_'+ qchem + '.hess'

    hess = getHessian(qcinput, hessFile, xyz)


    q = [0.0]*len(q_eq)
    p = [0.0]*len(q_eq)

    print()
    print("-----------------------------------")
    print("fixmode = ", fix_vib[0])
    print("fixvib = ", fix_vib[1])
    print("-----------------------------------")
    print()
    fnm = "mode_" + "0_extra" + ".xyz"
    print_normalmode_with_fix_quantum(fnm, fix_vib[0], fix_vib[1], atoms, mass, q_eq, hess)

    #hess_eckart = eckart_transform(mass, q_eq, hess)
    q, p, freq, nvib = polyatom_init(vibfile, RT, jrot, fix_vib, mass, q_eq, hess)

    for i in range(3*Natoms-6):
        fnm = "mode_" + str(i) + ".xyz"
        print_normalmode(fnm, i, 20.0, atoms, mass, q_eq, hess)
else:
    last_step, Natoms, atoms, q, p = parseCheckPoint(backfile)
    start_step = last_step
    print()
    print("The trajectory is restarted from a CheckPoint file, where the last step was: ", last_step)
    print("Still ", (nstep - last_step), " steps to finish the trajectory")
    print()
############################################################################################


#=========================================================================================
# Run dynamics
#=========================================================================================
'''
with open(traj_file, "a") as file_trj:
    file_wf = fname + ".delete_this_moldenfile"
    T0, V0, E0 = Energy(qcinput, file_wf, q, p, atoms, wmass)
    os.remove(file_wf)
    for i in range(start_step,nstep):
        #----------------------------------------------------------------------------------
        if (i % iprint == 0 and i > start_step) or i == 0:
            file_wf = fname + "step_" + str(i) + ".molden"
            T, V, Ene = Energy(qcinput, file_wf, q, p, atoms, wmass) 
            dE = Ene - E0
            print_trajectory(file_trj, atoms, q, p, V, dE, dt, i)

            backup_file=open(backfile,'w')
            print_trajectory(backup_file, atoms, q, p, V, dE, dt, i)
            backup_file.close()

            print("%20s %10d %15s %20.2f" % ("trajectory step: " , i, "time[fs]: ", i*dt/c6))
        #-----------------------------------------------------------------------------------
        q,p = velverlet(qcinput, dt, wmass, q, p, atoms)
#=========================================================================================
'''

t_end = time.time()
print((t_end - t_start), "s")

