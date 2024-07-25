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
from format_and_print import parseCheckPoint
from format_and_print import print_trajectory
from format_and_print import check_and_create_file

from gradient         import Energy
from hessian          import getHessian 
from cenmass          import cenmass
from polyatom         import polyatom_init
from integrators      import velverlet
from thermal          import thermal_vibr_mode
from nmodeprint       import print_normalmode
from eckart           import eckart_transform
from normalmode       import print_frequencies
from rel_init_coords  import setRelativeInitCoords 
from diatom           import diatom_init_harm
#-----------------------------------------------------

#set number of procs
#lib.num_threads(4)
#print("number of processors:",lib.num_threads())
#print()

#time-stamp start
t_start = time.time()

c1  = 0.52917721092  # [bohr]    * c1 = [Ansgtrom] 
c3  = 1838.6836605e0 # [g/mol]   * c3 = [electron mass unit]
c5  = 219474.e0      # [Hartree] * c5 = [cm-1]
c6  = 41.341105      # [fs]      * c6 = [time in au]
c7  = 2625.5         # [Hartree] * c7 = [kJ/mol] 
c9  = 1.0e8/c1       # [frequency in cm-1]*c9=[freq(bohr^(-1))]
c10 = 137.035999074  # [speed of light in atomic unit]
Rgas = 8.3144598/1000.0/c7 #Hartree/K 

mH = 1.00782503223*c3
mC = 12.011*c3
mN = 14.007*c3
mO = 15.999*c3

#-----------------------------------------------------
mH = 1.00782503223*c3
mC = 12.011*c3
mN = 14.007*c3
mO = 15.999*c3

#=========== Input parameters ====================
qchem = 'Orca'
functional = 'r2SCAN-3c'
base = ''
charge = 0
multiplicity_A = 1
multiplicity = 2
path = '/home/peter/Programs/Orca5.04/orca'
nproc = 6

#-----------------------------------------------------------------
# MD run parameters
#-----------------------------------------------------------------
temp = 300.0 #K
RT = Rgas * temp #in Hartree/K
dt = 0.4*c6 #fs --> automic time
nstep = 500   
iprint = 2

#Reactant A (polyatomic) Isoprene
jrot_A =  0
fix_vib = [0, 3]

#Reactant B (diatatomic) OH
jrot_B = 5
nvib_B = 0
omega_B = 4000.0 * (math.pi * 2) * c10 /c9 #cm-1 --> a.u.
req_B = 1.0/c1 #Angstrom
mass_B =  [mO] + [mH]
atoms_B =  ['O'] + ['H']

#
Ecoll =  5.0/c7 # kJ/mol --> Hartree
Rini = 5.0/c1 #Ansgtrom --> bohr
bmax = 3.0/c1 #Ansgtrom --> bohr

restart = False

seed = 220029
random.seed(seed)
#-----------------------------------------------------------------

qcinput_A = [0]*8
qcinput_A[0] = qchem
qcinput_A[1] = path
qcinput_A[2] = nproc
qcinput_A[3] = functional
qcinput_A[4] = base
qcinput_A[5] = charge
qcinput_A[6] = multiplicity_A
qcinput_A[7] = ''

qcinput = [0]*8
qcinput[0] = qchem
qcinput[1] = path
qcinput[2] = nproc
qcinput[3] = functional
qcinput[4] = base
qcinput[5] = charge
qcinput[6] = multiplicity
qcinput[7] = ''



#Sparrow PM6 optimized structure
xyz_A = '''
  C         0.01275          0.00000         1.35292
  C        -0.04520         -0.00000         0.02009
  C        -1.29563         -0.00000        -0.76814
  C        -1.28239          0.00000        -2.10714
  C        -2.58923         -0.00000        -0.01147
  H         0.87318         -0.00000        -0.57632
  H         0.94110          0.00000         1.90529
  H        -0.85649          0.00000         1.99530
  H        -2.16634         -0.00000        -2.71174
  H        -0.37079         -0.00000        -2.69792
  H        -3.46327         -0.00000        -0.67466
  H        -2.66984         -0.88694         0.63214
  H        -2.66984          0.88694         0.63214
 '''

Natoms_A, atoms_A, q_eq_A = parseXYZ(xyz_A)

q_eq_A = np.array(q_eq_A) / c1


#-----------------------------------------------------------------
mass_A =  [mC]*5 + [mH]*8
#-----------------------------------------------------------------
fname = "Isoprene_"
traj_file = fname+"traj.xyz"
vibfile = fname+"freq_and_sampled_vib_quanta.vib"
backfile = fname+"checkpoint.xyz"


############################################################################################
if not restart:
    start_step = 0
    hessFile = 'hessian_' + fname + functional + '_' + base + '_'+ qchem + '.hess'

    hess_A = getHessian(qcinput_A, hessFile, xyz_A)
    import shutil
    shutil.rmtree('orca_tmp', ignore_errors=True)

    #q = [0.0]*len(q_eq_A)
    #p = [0.0]*len(q_eq_A)

    hess_eckart = eckart_transform(mass_A, q_eq_A, hess_A)
    #q_A, p_A, freq, nvib = polyatom_init(vibfile, RT, jrot, fix_vib, mass, q_eq, hess_eckart)
    q_A, p_A, freq_A, nvib_A = polyatom_init(vibfile, RT, jrot_A, fix_vib, mass_A, q_eq_A, hess_eckart)

    #for i in range(3*Natoms_A-6):
    #    fnm = "mode_" + str(i) + ".xyz"
    #    print_normalmode(fnm, i, 20.0, atoms_A, mass_A, q_eq_A, hess_eckart)

    q_B, p_B = diatom_init_harm(req_B, omega_B, mass_B, jrot_B, nvib_B)

    q, p, atoms, mass, bimp = setRelativeInitCoords(Ecoll, bmax, Rini, atoms_A, atoms_B, mass_A, mass_B, q_A, q_B, p_A, p_B)

    with open('init_geom.xyz', "a") as file_trj:
        print_trajectory(file_trj, atoms, q, p, 0.0, 0.0, 1.0, 0)

    wmass = []   # auxiliary mass vector, same length as q and p
    for ww in mass:
        wmass += [ww]*3
    wmass = np.array(wmass)


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
with open(traj_file, "a") as file_trj:
    #os.remove("orca_tmp/*")
    file_wf = fname + ".delete_this_moldenfile"
    T0, V0, E0 = Energy(qcinput, file_wf, q, p, atoms, wmass)
    #os.remove(file_wf)
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

            with open('Run_traj.log', "a") as file_log:
                file_log.write("%20s %10d %15s %20.2f\n" % ("trajectory step: " , i, "time[fs]: ", i*dt/c6))

            print("%20s %10d %15s %20.2f" % ("trajectory step: " , i, "time[fs]: ", i*dt/c6))
        #-----------------------------------------------------------------------------------
        q,p = velverlet(qcinput, dt, wmass, q, p, atoms)
#=========================================================================================


t_end = time.time()
print((t_end - t_start), "s")

