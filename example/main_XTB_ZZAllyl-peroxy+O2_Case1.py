import math
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
temp = 200.0 #K
RT = Rgas * temp #in Hartree/K
dt = 0.3*c6 #fs --> automic time
nstep = 18000
iprint = 2


#Reactant A (polyatomic) Isoprene-OH-OO peroxy allyl radical
jrot_A =  0
fix_vib = [0, 3]
multiplicity_A = 2

#Reactant B (diatatomic) O2
jrot_B = 3
nvib_B = 0
omega_B = 1580.0 * (math.pi * 2) * c10 /c9 #cm-1 --> a.u.
req_B = 1.20/c1 #Angstrom
mass_B =  [mO] + [mO]
atoms_B =  ['O'] + ['O']

#
Ecoll =  2.5/c7 # kJ/mol --> Hartree
Rini = 8.5/c1 #Ansgtrom --> bohr
bmax = 6.0/c1 #Ansgtrom --> bohr

restart = False

seed = 215221
random.seed(seed)
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


#ZZAllyl-peroxy+O2_Case1 M062X avtz optim Con 21 (11 csak a futtatos jelolesben 21)
xyz_A = '''
C   -0.19595538763398      0.07192231509414      0.00714857985445
C   -0.01703579534250     -0.10344222305327      1.49976587011423
C   -1.00831293517504     -0.74304790940588      2.21484017141187
C   -1.09565313995228     -0.96572164058650      3.69139746432409
O   -2.44905662107855     -1.09778426038872      4.10589699809888
O   -3.03738052053937      0.19479040763344      4.06761535894273
C   1.14919758827461      0.41511797958774      2.03274034295381
O   1.44878216770210      0.30476609400103      3.35160340084751
H   -1.82826168270363     -1.17741133496109      1.65145938615843
H   1.87669555170293      0.91631236085033      1.40830140902315
H   0.76154446389898      0.10310849679981     -0.51109606581980
H   -0.77393280198763     -0.75201110449544     -0.40695658995199
H   -0.72775230388759      0.99760660592287     -0.21471742835545
H   -0.61505142152353     -0.17733195303782      4.26524823267036
H   -0.65062609284075     -1.92286967316247      3.98246076165767
H   2.29766286152308      0.71228566068994      3.53361144883480
H   -3.35772393043684      0.25312017851188      3.15850065923520
 '''

Natoms_A, atoms_A, q_eq_A = parseXYZ(xyz_A)

q_eq_A = np.array(q_eq_A) / c1


#-----------------------------------------------------------------
mass_A =  [mC]*5 +[mO]*3 + [mH]*9
#-----------------------------------------------------------------
fname = "ZZAllyl-peroxy+O2_Case1_" + qchem 
vibfile = fname+"_freq_and_sampled_vib_quanta.vib"
backfile = fname+"_checkpoint.xyz"

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
    mass = mass_A + mass_B

    wmass = []   # auxiliary mass vector, same length as q and p
    for ww in mass:
        wmass += [ww]*3
    wmass = np.array(wmass)

    last_step, Natoms, atoms, q, p = parseCheckPoint(backfile)
    start_step = last_step
    print()
    print("The trajectory is restarted from a CheckPoint file, where the last step was: ", last_step)
    print("Still ", (nstep - last_step), " steps to finish the trajectory")
    print()
############################################################################################



stopcond = {}
stopcond['atomA'] = 2
stopcond['atomB'] = 18 
stopcond['rdist'] = 10.0 #0.9*Rini*c1 #Angstrom

run_trajectory(qcinput, q, p, dt, atoms, wmass, fname, restart, start_step, nstep, iprint, stopcond)


t_end = time.time()
print((t_end - t_start), "s")

