import math
import numpy as np
import random
import time
import sys
import logging
import os
import shutil

#-----------------------------------------------------
# Source modules
#-----------------------------------------------------
from format_and_print import parseXYZ, makeXYZ
from format_and_print import parseCheckPoint
from format_and_print import print_trajectory
from format_and_print import print_end
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
from distance         import test_to_stop        
from rundynamics      import run_trajectory
from rel_init_coords  import setRelativeInitCoords
from diatom           import diatom_init_harm
from clustering       import cluster_chemical_formulas
from distance_matrix  import compute_connectivity 
from distance_matrix  import add_matrix_and_get_index

#-----------------------------------------------------

#set number of procs
#lib.num_threads(4)
#print("number of processors:",lib.num_threads())
#print()

#time-stamp start
t_start = time.time()

c1 = 0.52917721092  # [bohr]    * c1 = [Ansgtrom] 
c3 = 1838.6836605e0 # [g/mol]   * c3 = [electron mass unit]
c5  = 219474.e0      # [Hartree] * c5 = [cm-1]
c6 = 41.341105      # [fs]      * c6 = [time in au]
c7 = 2625.5         # [Hartree] * c7 = [kJ/mol] 
c9  = 1.0e8/c1       # [frequency in cm-1]*c9=[freq(bohr^(-1))]
c10 = 137.035999074  # [speed of light in atomic unit]
Rgas = 8.3144598/1000.0/c7 #Hartree/K 

mH = 1.00782503223*c3
mC = 12.011*c3
mN = 14.007*c3
mO = 15.999*c3

#=========== Input parameters ====================
#qchem = 'Orca'
qchem = 'XTB'
#functional = 'b3lyp d3'
#functional = 'HF-3c'
functional = ''
#functional = 'r2SCAN-3c'
#base = 'pc-1'
#base = 'def2-SVP'
base = ''
charge = 0 
multiplicity = 2
#path = '/home/peter/Programs/sparrow/install/bin/sparrow'
#path = '/home/peter/Programs/Orca.5.04/orca'
path = '/home/peter/Programs/xtb-6.6.1/bin/xtb'
nproc = 8
wfu = False
additional = '--acc 10'
#additional = '--acc 20 --gfn 1'
#additional = '--gfnff'

#-----------------------------------------------------------------
# MD run parameters
#-----------------------------------------------------------------
temp = 300.0 #K
RT = Rgas * temp #in Hartree/K
dt = 0.3*c6 #fs --> automic time
nstep = 10001 
iprint = 2

ntraj = 20


#Reactant A (polyatomic) Isoprene
jrot_A =  10
fix_vib = [999, 0]
multiplicity_A = 1

#Reactant B (diatatomic) OH
jrot_B = 2
nvib_B = 0
omega_B = 4000.0 * (math.pi * 2) * c10 /c9 #cm-1 --> a.u.
req_B = 1.00/c1 #Angstrom
mass_B =  [mO] + [mH]
atoms_B =  ['O'] + ['H']

#
Ecoll =  2.5/c7 # kJ/mol --> Hartree
Rini = 9.0/c1 #Ansgtrom --> bohr
bmax = 9.0/c1 #Ansgtrom --> bohr

restart = False

seed = 221442
random.seed(seed)


stopcond = {}
stopcond['atomA'] = 2
stopcond['atomB'] = 13
stopcond['rdist'] = 10.0 #0.9*Rini*c1 #Angstrom
#-----------------------------------------------------------------

qcinput_A = [0]*9
qcinput_A[0] = qchem
qcinput_A[1] = path
qcinput_A[2] = nproc
qcinput_A[3] = functional
qcinput_A[4] = base
qcinput_A[5] = charge
qcinput_A[6] = multiplicity_A
qcinput_A[7] = additional
qcinput_A[8] = wfu

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

#energy: -14.722245720047 gnorm: 0.000764575836 xtb: 6.6.1 (8d0f1dd)
xyz_A = '''
C            0.02012641732028        0.00000006175965        1.34209029266400
C           -0.07220144507590        0.00000001647193        0.01710129122274
C           -1.30867616316804        0.00000008084485       -0.75371041848674
C           -1.26600101272938       -0.00000006313200       -2.08574799229951
C           -2.61129971073827        0.00000000910147       -0.01059470902742
H            0.83384966683544       -0.00000004944119       -0.57654919142490
H            0.97625961435649       -0.00000008770044        1.83720037583573
H           -0.84391447302206        0.00000003526300        1.98409942271942
H           -2.16103364837372        0.00000003331697       -2.68428156221756
H           -0.33278514660391        0.00000001414596       -2.62340022207185
H           -3.44782134013576       -0.00000005697376       -0.70430590601218
H           -2.68424632281343       -0.88185741243400        0.62429430932029
H           -2.68424643585175        0.88185741877753        0.62429430977797
 '''

Natoms_A, atoms_A, q_eq_A = parseXYZ(xyz_A)

q_eq_A = np.array(q_eq_A) / c1


#-----------------------------------------------------------------
mass_A =  [mC]*5 + [mH]*8
#-----------------------------------------------------------------

prod_con_list = []
prod_q_list = [] # np.column_stack((vector1, vector2))

for itraj in range(1,ntraj+1):
    fbase = "Isoprene+OH_" + qchem
    fname = fbase + "_traj_" + str(itraj)
    vibfile = fbase+"_freq_and_sampled_vib_quanta.vib"
    backfile = fname+"_checkpoint.xyz"
    prodfile = "Products_"+ fbase + ".out" 

    ############################################################################################
    if not restart:
        start_step = 0
        hessFile = 'hessian_' + fbase + functional + '_' + base + '_'+ qchem + '.hess'

        hess_A = getHessian(qcinput_A, hessFile, xyz_A)
        shutil.rmtree('orca_tmp', ignore_errors=True)

        #q = [0.0]*len(q_eq_A)
        #p = [0.0]*len(q_eq_A)

        hess_eckart = eckart_transform(mass_A, q_eq_A, hess_A)
        q_A, p_A, freq_A, nvib_A = polyatom_init(vibfile, RT, jrot_A, fix_vib, mass_A, q_eq_A, hess_eckart)

        #for i in range(3*Natoms_A-6):
        #    fnm = "mode_" + str(i) + ".xyz"
        #    print_normalmode(fnm, i, 20.0, atoms_A, mass_A, q_eq_A, hess_eckart)

        q_B, p_B = diatom_init_harm(req_B, omega_B, mass_B, jrot_B, nvib_B)

        q, p, atoms, mass, bimp = setRelativeInitCoords(Ecoll, bmax, Rini, atoms_A, atoms_B, mass_A, mass_B, q_A, q_B, p_A, p_B)

        with open('init_geom.xyz', "a") as file_trj:
             print_trajectory(file_trj, atoms, q, p, bimp, 0.0, 1.0, 0)

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



    #cluster analysis paramters:
    eps = 2.1/c1 #in Bohr 
    minPts = 1 #minum number of points to be a cluster
   #connectivity matrix parameters:
    H_max_dist = 1.5/c1  # Maximum bond distance for H to any atom (adjust as needed)
    nonH_max_dist = 2.0/c1    # Maximum bond distance for non-H atom to another non-H atom (adjust as needed)


    #initialize_reactants:
    maxstep, q, p = run_trajectory(qcinput, q, p, dt, atoms, wmass, fname, restart, start_step, nstep, iprint, stopcond)

    #-------------------------------------------
    #End analysis:
    connect_mat = compute_connectivity(q, atoms, H_max_dist, nonH_max_dist)

    channel, prod_con_list, prod_q_list = add_matrix_and_get_index(connect_mat, q, prod_con_list, prod_q_list)

    formula = cluster_chemical_formulas(q, atoms, eps, minPts)

    with open(prodfile, "a") as end_file:
        print_end(end_file, atoms, q, p, dt, maxstep, channel, formula)


#========================================================================================


t_end = time.time()
print((t_end - t_start), "s")

