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
#qchem = 'Sparrow'
#functional = 'b3lyp d3'
#functional = 'HF-3c'
#functional = 'dftb3'
functional = 'r2SCAN-3c'
#base = 'pc-1'
#base = 'def2-SVP'
base = ''
charge = 0
multiplicity = 1
path = '/home/peter/Programs/Orca5.04/orca'
#path = ''
#path = '/home/peter/Programs/Orca_5.0.4/orca'
nproc = 4
wfu = False
additional = ''

#-----------------------------------------------------------------
# MD run parameters
#-----------------------------------------------------------------
temp = 100.0 #K
RT = Rgas * temp #in Hartree/K
dt = 1.0*c6 #fs --> automic time
nstep = 5000
iprint = 5

jrot = 0
fix_vib = [41, 6]

restart = False

seed = 122902
random.seed(seed)
#-----------------------------------------------------------------

qcinput = [0]*9
qcinput[0] = qchem
qcinput[1] = path
qcinput[2] = nproc
qcinput[3] = functional
qcinput[4] = base
qcinput[5] = charge
qcinput[6] = multiplicity
qcinput[7] = additional
qcinput[8] = wfu


#dftb3 optimized 15-diene
xyz = '''
   C   1.51758    0.10942    -0.21957
   C  -1.51763    0.10856     0.21955
   C   0.71961   -1.06314     0.27662
   C  -2.09723    1.01377    -0.56789
   C   2.09663    1.01516     0.56792
   C  -0.71893   -1.06352    -0.27663
   H   1.21079   -1.99409    -0.03590
   H  -2.65641    1.84185    -0.15082
   H   0.69556   -1.06397     1.37082
   H  -2.02253    0.95284    -1.64754
   H   2.02185    0.95414     1.64753
   H  -0.69485   -1.06435    -1.37083
   H   2.65525    1.84351     0.15080
   H  -1.20953   -1.99480     0.03583
   H   1.60145    0.20577    -1.30224
   H  -1.60152    0.20485     1.30226
 '''
Natoms, atoms, q_eq = parseXYZ(xyz)

q_eq = np.array(q_eq) / c1


#-----------------------------------------------------------------
mass =  [mC]*6 + [mH]*10

wmass = []   # auxiliary mass vector, same length as q and p
for ww in mass:
    wmass += [ww]*3
wmass = np.array(wmass)
#-----------------------------------------------------------------
fname = "Cope-15diene_"
traj_file = fname+"traj.xyz"
vibfile = fname+"freq_and_sampled_vib_quanta.vib"
backfile = fname+"checkpoint.xyz"


############################################################################################
if not restart:
    start_step = 0
    hessFile = 'hessian_' + fname + functional + '_' + base + '_'+ qchem + '.hess'

    hess = getHessian(qcinput, hessFile, xyz)


    q = [0.0]*len(q_eq)
    p = [0.0]*len(q_eq)


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

print("\n******************************************************")
if os.path.exists(traj_file):
   os.remove(traj_file)
   print(f"{traj_file} already exisits, it has been deleted to create a new one.")
print("******************************************************")


#=========================================================================================
# Run dynamics
#=========================================================================================
with open(traj_file, "a") as file_trj:
    file_wf = fname + ".delete_this_moldenfile"
    T0, V0, E0 = Energy(qcinput, file_wf, q, p, atoms, wmass)
    if os.path.exists(file_wf):
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


t_end = time.time()
print((t_end - t_start), "s")

