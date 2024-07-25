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
Rgas = 8.3144598/1000.0/c7 #Hartree/K 

mH = 1.00782503223*c3
mC = 12.011*c3
mN = 14.007*c3
mO = 15.999*c3

#=========== Input parameters ====================
temp = 300.0 #K
RT = Rgas * temp #in Hartree/K
dt = 0.5*c6 #fs --> automic time
nstep = 10
iprint = 5
seed = 423486
random.seed(seed)

#qchem = 'PySCF'
qchem = 'Sparrow'

charge = 0
multiplicity = 1
#functional = 'b3lyp'
functional = 'pm6'
base = 'sto-3g'

#----------Fragment A-----------------------------
charge_A = 0
multiplicity_A = 1
functional_A = functional 
base_A = base

xyzA = '''
 C   0.041504   0.000119   0.018112
 C   0.007547   0.000107   1.424817
 C   1.229617   0.000023   2.125206
 C   2.450070  -0.000018   1.415311
 C   2.460308   0.000066   0.003460
 C   1.245690   0.000041  -0.709808
 H  -0.968265   0.000141   1.932043
 H   1.231453  -0.000043   3.223826
 H   3.400466  -0.000138   1.966557
 H   3.414912  -0.000010  -0.540322
 H   1.201740   0.000054  -1.808689
 N  -1.304719  -0.000072  -0.762361
 O  -1.235931  -0.000065  -2.082955
 O  -2.416230  -0.000207  -0.045999
 '''

nA, atoms_A, qeq_A = parseXYZ(xyzA)
qeq_A = np.array(qeq_A) / c1
mass_A =  [mC]*6 + [mH]*5 + [mN] + [mO]*2

wmass_A = []   # auxiliary mass vector, same length as q and p
for ww in mass_A:
    wmass_A += [ww]*3
wmass_A = np.array(wmass_A)

hessFile_A = 'hessian.hess'
hess_A = getHessian(qchem, hessFile, xyz_A, charge, multiplicity, functional, base)

hess_eckart_A = eckart_transform(mass_A, qeq_A, hess_A)

fix_vib = [24, 12]

q_A, p_A, freq_A, nvib_A = polyatom_init(RT, jrot_A, fix_vib_A, mass_A, qeq_A, hess_eckart_A)
#-----------------------------------------------------------------

#=========== End parameters ====================



#=========================================================================================
# Run dynamics
#=========================================================================================

fname = "NitroBenzene_"
traj_file = fname+"traj.xyz"
#traj_file = check_and_create_file(traj_file, '.xyz')

with open(traj_file, "a") as file_trj:
    file_wf = fname + ".delete_this_moldenfile"
    T0, V0, E0 = Energy(file_wf, qchem, q, p, atoms, wmass, charge, multiplicity, functional, base)
    for i in range(0,nstep):
        #----------------------------------------------------------------------------------
        if i % iprint == 0 or i == 0:
            file_wf = fname + "step_" + str(i) + ".molden"
            T, V, Ene = Energy(file_wf, qchem, q, p, atoms, wmass, charge, multiplicity, functional, base) 
            dE = Ene - E0
            print_trajectory(file_trj, atoms, q, p, V, dE, dt, i)
            print("%20s %10d %15s %20.2f" % ("trajectory step: " , i, "time[fs]: ", i*dt/c6))
            #-----------------------------------------------------------------------------------
        q,p = velverlet(qchem, dt, wmass, q, p, atoms, charge, multiplicity, functional, base)
#=========================================================================================

t_end = time.time()
print((t_end - t_start), "s")

