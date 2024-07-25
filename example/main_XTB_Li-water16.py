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
from thermostat  import thermo_berendsen, thermo_andersen
from thermostat  import traj_temperature 
from thermostat  import random_initialize_momenta


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

mH  = 1.00782503223*c3
mC  = 12.011*c3
mN  = 14.007*c3
mO  = 15.999*c3
mLi = 6.941*c3

#=========== Input parameters ====================
#qchem = 'PySCF'
qchem = 'XTB'
#qchem = 'Sparrow_Py'
#functional = 'b3lyp'
#functional = 'HF-3c'
#functional = 'PM6'
functional = ''
#functional = 'AM1'
#functional = 'r2SCAN-3c'
#base = 'pc-1'
#base = 'def2-SVP'
base = ''
charge = 1
multiplicity = 1
#path = '/home/peter/Programs/sparrow/install/bin/sparrow'
#path = '/home/peter/Programs/Orca.5.0.4/orca'
path = '/home/peter/Programs/xtb-6.6.1/bin/xtb' 
#path = '' 
nproc = 4

wfu = True
additional = ''

#-----------------------------------------------------------------
# MD run parameters
#-----------------------------------------------------------------
dt_NVT = 1.0*c6 #fs --> automic time
dt_NVE = 0.5*c6 #fs --> automic time
nstep_NVT = 2000
nstep_NVE = 14000
iprint_NVT = 5
iprint_NVE = 2


tau_Berendsen = 20.0 * c6 #fs --> au  
Temp_init = 200.0
Temp_targ = 298.0

restart = False

seed = 222112
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


xyz = '''
Li           0.86991676558325        1.48302187645282        2.14276261851947
O           -1.35536978105074        1.35407368646476        1.27648855264062
O           -2.21055277277043        4.31420027202642        1.36590499832334
O           -2.60719575451089        1.73271769923538        3.93851259814342
O            0.36256624378615       -0.97938799356083        1.71593031121526
O            1.55210564177920       -0.35586344592905       -0.73354216828313
O            2.78954354493230        4.35211116453598        1.73523936346730
O            0.11417115265965        3.57952336667755        2.38036836928488
O            1.67017554365573        2.31297581962824        0.51922472687424
O            0.73312289696457        7.62383904790158        1.51425165088761
O            0.30107205789143        3.65603273947972        4.96484081453718
O            4.88843526866227        2.66657636479554        0.59641459312900
O           -1.25960100685173        1.33975913740428        6.60744661075083
O            1.07922589792730        1.14596348367410        4.32001313815721
O            1.21870787544763        1.93632839551321       -2.11045834597725
O            3.34155903516353        0.54392668832498        2.71752963979636
O           -1.96698222455404        6.66760921073010       -0.08823598453069
H           -1.43767669216991        0.47190446066930        0.93268231205455
H           -1.87868483751667        1.47230118759983        2.08178952938558
H           -2.18648327997632        3.48178848168904        0.84957660491760
H           -2.16735433705115        5.06149226720012        0.75368993862383
H           -3.47332102631460        2.20895680032196        4.13932664493334
H           -1.99922211013357        2.11967576936143        4.62527108861365
H            0.43409993783929       -1.92356264427587        1.61288906038578
H            0.72088112802321       -0.59505868453672        0.89477749637223
H           -0.82600765664130        3.80883040098589        2.04530160897517
H            0.75696781714860        4.26493757467432        2.15878046309076
H            1.59560793493361        0.25683244821050       -1.48009619145022
H            2.37240103872586       -0.23395870869190       -0.26532416031958
H            3.43153356146209        1.04535556522667        3.53466136521969
H            2.66737186456545        0.00541575642853        3.11221564985025
H            2.40183377223673        1.76964855402300        0.84416293046189
H            2.12710448038186        3.17946732891807        0.68209417850211
H            0.10335034181718        3.76057848161260        3.96497275372450
H           -0.24647560562160        2.93452829574974        5.27908412027283
H            4.94871464787188        2.70150915379188        1.56727008902581
H            5.83526719048664        2.79933106884846        0.49739948679216
H            0.32542186727132        2.01953326128742       -2.36756407622598
H            1.34506852568568        2.16810067877025       -1.21558991612964
H           -1.45457309289657        1.53890175647100        7.55117791429233
H           -2.02018490492281        0.87270865770894        6.28277307858270
H            2.61370725321094        5.11575856546695        2.32150089165158
H            3.66001778293435        4.55413098532262        1.43272687154398
H            0.80114902750002        0.50970153408701        4.93608790861669
H            1.02605412927333        2.00484865532427        4.77107650900282
H           -1.62162505160926        6.91572474854613       -0.97653271717329
H           -2.13193601348513        7.50672711334292        0.38313251113043
H            0.06633594955325        7.05209588009284        1.11575685429259
H            0.66578007809736        8.49792717650578        1.20623552749567
 '''
Natoms, atoms, q_eq = parseXYZ(xyz)

q_eq = np.array(q_eq) / c1


#-----------------------------------------------------------------
mass =  [mLi]*1 + [mO]*16 + [mH]*32

wmass = []   # auxiliary mass vector, same length as q and p
for ww in mass:
    wmass += [ww]*3
wmass = np.array(wmass)
#-----------------------------------------------------------------
fname = "Li-16water_"
traj_file_NVT = "NVT_" + fname + "traj.xyz"
traj_file_NVE = "NVE_" + fname + "traj.xyz"
backfile_NVT  = "NVT_" + fname + "checkpoint.xyz"
backfile_NVE  = "NVE_" + fname + "checkpoint.xyz"


############################################################################################
if not restart:
    start_step = 0

    q = [0.0]*len(q_eq)
    p = [0.0]*len(q_eq)

    q = q_eq
    p = np.array(random_initialize_momenta(p, wmass, Temp_init))

else:
    last_step, Natoms, atoms, q, p = parseCheckPoint(backfile_NVT)
    start_step = last_step
    print()
    print("The trajectory is restarted from a CheckPoint file, where the last step was: ", last_step)
    print("Still ", (nstep - last_step), " steps to finish the trajectory")
    print()
############################################################################################

print("\n******************************************************")
if os.path.exists(traj_file_NVT):
   os.remove(traj_file_NVT)
   print(f"{traj_file_NVT} already exisits, it has been deleted to create a new one.")
print("******************************************************")


#=========================================================================================
# Run NVT dynamics
#=========================================================================================
with open(traj_file_NVT, "a") as file_trj:
    file_wf = "NVT_" + fname + ".delete_this_moldenfile"

    T0, V0, E0 = Energy(qcinput, file_wf, q, p, atoms, wmass)

    if os.path.exists(file_wf):
       os.remove(file_wf)

    for i in range(start_step,nstep_NVT):
        #----------------------------------------------------------------------------------
        if (i % iprint_NVT == 0 and i > start_step) or i == 0:
            file_wf = "NVT_"+ fname + "step_" + str(i) + ".molden"
            T, V, Ene = Energy(qcinput, file_wf, q, p, atoms, wmass) 
            dE = Ene - E0

            Temp_act = traj_temperature(0, p, wmass)

            print_trajectory(file_trj, atoms, q, p, V, dE, dt_NVT, i)

            backup_file=open(backfile_NVT,'w')
            print_trajectory(backup_file, atoms, q, p, V, dE, dt_NVT, i)
            backup_file.close()

            print("%20s %10d %12s %15.2f %13s %15.4f %8s %12.4f" % ("NVT trajectory step: " , i, "time[fs]: ", i*dt_NVT/c6, "dE[kJ/mol]: ", dE*c7, "T[K]: ", Temp_act))
        #-----------------------------------------------------------------------------------
        q,p = velverlet(qcinput, dt_NVT, wmass, q, p, atoms)

        p = thermo_berendsen(0, p, wmass, dt_NVT, tau_Berendsen, Temp_targ) 
        q,p = cenmass(q, p, mass) #prevent flying ice-cube
        #p = thermo_andersen(p, wmass, dt_NVT, prob, Temp_targ)

print()


#=========================================================================================
# Run NVE dynamics
#=========================================================================================


with open(traj_file_NVE, "a") as file_trj:
    file_wf = "NVE_" + fname + ".delete_this_moldenfile"

    T0, V0, E0 = Energy(qcinput, file_wf, q, p, atoms, wmass)

    if os.path.exists(file_wf):
       os.remove(file_wf)

    for i in range(start_step,nstep_NVE):
        #----------------------------------------------------------------------------------
        if (i % iprint_NVE == 0 and i > start_step) or i == 0:
            file_wf = "NVE_" + fname + "step_" + str(i) + ".molden"
            T, V, Ene = Energy(qcinput, file_wf, q, p, atoms, wmass)
            dE = Ene - E0

            Temp_act = traj_temperature(0, p, wmass)

            print_trajectory(file_trj, atoms, q, p, V, dE, dt_NVE, i)

            backup_file=open(backfile_NVE,'w')
            print_trajectory(backup_file, atoms, q, p, V, dE, dt_NVE, i)
            backup_file.close()

            print("%20s %10d %12s %15.2f %8s %15.4f %8s %12.4f" % ("NVE trajectory step: " , i, "time[fs]: ", i*dt_NVE/c6, "dE: ", dE*c7, "T[K]: ", Temp_act))
        #-----------------------------------------------------------------------------------
        q,p = velverlet(qcinput, dt_NVE, wmass, q, p, atoms)



t_end = time.time()
print((t_end - t_start), "s")

