import os

from format_and_print import print_trajectory
from format_and_print import qvec_to_xyz_matrix
from gradient         import Energy
from integrators      import velverlet
from distance         import test_to_stop        

#================================================================================================================
def run_trajectory(qcinput, q, p, dt, atoms, wmass, fname, restart, start_step, nstep, iprint, stopcond):
#================================================================================================================
    c6 = 41.341105      # [fs]      * c6 = [time in au]

    traj_file = fname+"_traj.xyz"
    backfile = fname+"_checkpoint.xyz"

   #--------------------------------------------------------------------------------------
    if not restart: 
        print("\n*************************************************************************")
        if os.path.exists(traj_file):
            os.remove(traj_file)
            print(f"{traj_file} already exisits, it has been deleted to create a new one.")
        print("***************************************************************************")
   #--------------------------------------------------------------------------------------


    with open(traj_file, "a") as file_trj:

       #to print wavefunction (it happens when qcinput[8] = True
        file_wf = fname + ".delete_this_moldenfile"

        T0, V0, E0 = Energy(qcinput, file_wf, q, p, atoms, wmass)

        if os.path.exists(file_wf):
            os.remove(file_wf)

        for i in range(start_step,nstep):
            #----------------------------------------------------------------------------------
            if (iprint > 0 and i % iprint == 0 and i > start_step) or i == 0:
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

            tstop = test_to_stop(i, q, stopcond['atomA'], stopcond['atomB'], stopcond['rdist'])

            if i > 500 and tstop == True:
                print("\n Reactive event found")
                break

        maxstep = i
    return (maxstep, q, p)

#==============================================================================================

