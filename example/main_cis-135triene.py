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
from distance    import test_to_stop        
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
#qchem = 'Orca'
qchem = 'Sparrow'
#functional = 'b3lyp d3'
#functional = 'HF-3c'
functional = 'pm6'
#functional = 'r2SCAN-3c'
#base = 'pc-1'
#base = 'def2-SVP'
base = ''
charge = 0
multiplicity = 1
#path = '/home/peter/Programs/Orca5.04/orca'
path = ''
#path = '/home/peter/Programs/Orca_5.0.4/orca'
nproc = 4
wfu = False
additional = ''

#-----------------------------------------------------------------
# MD run parameters
#-----------------------------------------------------------------
temp = 100.0 #K
RT = Rgas * temp #in Hartree/K
dt = 0.5*c6 #fs --> automic time
nstep = 20000
iprint = 5

jrot = 0
fix_vib = [33, 7]

restart = False

seed = 122112
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


#IRC endpoint of openchain cis-1,3,5 triene
xyz = '''
  C      -1.185385      1.500364     -0.174799
  C       0.057100      1.525641      0.290486
  C       1.182421      0.671307     -0.090281
  C       1.182420     -0.671306     -0.090282
  C       0.057100     -1.525640      0.290485
  C      -1.185388     -1.500364     -0.174796
  H       0.285347      2.226153      1.090336
  H       2.144030      1.164815     -0.199258
  H       2.144029     -1.164815     -0.199261
  H       0.285350     -2.226154      1.090332
  H      -1.972016     -2.076478      0.294389
  H      -1.462109     -0.924082     -1.042658
  H      -1.972017      2.076476      0.294382
  H      -1.462101      0.924083     -1.042665
 '''
Natoms, atoms, q_eq = parseXYZ(xyz)

q_eq = np.array(q_eq) / c1


#-----------------------------------------------------------------
mass =  [mC]*6 + [mH]*8

wmass = []   # auxiliary mass vector, same length as q and p
for ww in mass:
    wmass += [ww]*3
wmass = np.array(wmass)
#-----------------------------------------------------------------
fname = "Ringclosing-135triene_"
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
    #    tstop = test_to_stop(i,Natoms, q, 0, 5, 1.6)
    #    if tstop == True:
    #       print("\n Reactive event found")
    #       break
#=========================================================================================


t_end = time.time()
print((t_end - t_start), "s")

