from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == '__main__':
    import random
    from smite import Molecule, Fragment, Collision


    #See Lam's recent publication:
    #Environ. Sci.: Atmos., 2023,3, 1678-1684, https://doi.org/10.1039/D3EA00143A
    #H2O2 XTB opt
    xyz_H2O2 = '''
    O            0.05611300699723        0.22647282338165       -0.07718637907274
    O            0.16803213662644        0.16159915200254        1.33740570520030
    H            0.85462287631102       -0.23506870591601       -0.36444208691326
    H           -0.63093201993469        0.62266873053181        1.62422676078569
     '''

    qcinput_doublet = {
    'qchem': 'XTB',
    'path': '/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/xtb',
    'nproc': 4,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 2,
    'additional': '--gfnff --iterations 1000 --tblite',
    'wfu': False 
    }

    qcinput_singlet = {
    'qchem': 'XTB',
    'path': '/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/xtb',
    'nproc': 2,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 1,
    'additional': '--iterations 1000',
    'wfu': False
    }

    qcinput_Orca_doublet = {
    'qchem': 'Orca',
    'path': '/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/orca',
    'nproc': 2,
    #'functional': 'HF-3c',
    #'functional': 'wB97X-3c',
    'functional': 'PM6',
    'basis': '',
    'charge': 0,
    'multiplicity': 2,
    'additional': '',
    'wfu': False
    }

    qcinput_PySCF = {
    'qchem': 'PySCF',
    'path': '',
    'nproc': 4,
    'functional': 'hf',
    'basis': 'sto-3g',
    'charge': 0,
    'multiplicity': 2,
    'additional': '',
    'wfu': False
    }
    
    qcinput_Sparrow_bin = {
    'qchem': 'Sparrow_bin',
    'path': '/home/peter/Programs/sparrow/install/bin/sparrow',
    'nproc': 2,
    'functional': 'PM6',
    'basis': '',
    'charge': 0,
    'multiplicity': 2,
    'additional': '-I 200 --density_rmsd_criterion 1e-3 --self_consistence_criterion 1e-5',
    'wfu': False
    }


    qcinput_Sparrow_Py = {
    'qchem': 'Sparrow_Py',
    'path': '',
    'nproc': 4,
    'functional': 'PM6',
    'basis': '',
    'charge': 0,
    'multiplicity': 2,
    'additional': '',
    'wfu': False
    }

    fix_quantum = [(14, 0),
                   (15, 0)]

    fix_energy  = [(0, 0.0),
                   (1, 0.0),
                   (2, 0.0)]

    fix_temp    = [(5, 330.0),
                   (6, 430.0)]

    seed = 28220222
    random.seed(seed)


    

    H2O2  = Fragment.Polyatom_Init(fname='H2O2', qchem=qcinput_singlet, xyz=xyz_H2O2, random_rot=True)
    #zzallyl.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=0, fix_quantum=fix_quantum)
    #zzallyl.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Temp', temp=300.0)
    H2O2.Specify_Mode_Sampling(init_vib_type='Temp', init_rot_type='Temp', temp=300.0)

    req_OH = 0.96 #Angstrom
    omega_OH = 3808.2 #cm-1

    OH = Fragment.Diatom_Init(fname='OH', atoms=['O','H'], req=req_OH, omega=omega_OH, random_rot=True, diatom='harmonic')
    #OH.Specify_Mode_Sampling(init_rot_type='Jfix', nvib=0, jrot=0)
    OH.Specify_Mode_Sampling(init_rot_type='Temp', nvib=0, temp=300.0)

    print("----------------------- H2O2 --------------------")
    H2O2.print_mode_sampling()
    #print("------------------------ OH ---------------------\n")
    #OH.print_mode_sampling()
    print("-------------------------------------------------\n")

    pairs_to_test = {
    'reaction_C1-H1': [((0, 2), 'GT', 6.0), ((4, 2), 'LT', 1.5)],  
    'reaction_C1-H2': [((1, 3), 'GT', 6.0), ((4, 3), 'LT', 1.5)],  
    # add more channels and pairs as needed
    }

    print("\nReaction of H2O2 + OH\n")
  

    #------------------------------------------------------------------------------------------------------------
    def setup_reactions(fragA, fragB, qchem, Rini, bmax, bsampling=True, Ecoll=10.0, Ecoll_thermal=False, temp=300.0, slurm=False, pairs_to_stop=None, Rstop=None):
        import os

        cores_per_traj = qchem['nproc']

        if slurm:
            total_cores = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count()))
            print("SLURM mode is active.")
        else:
            total_cores = os.cpu_count()
            print("Running in local mode.")

        ntraj_paralell = total_cores // cores_per_traj
        print(f"Running with {ntraj_paralell} parallel trajectories")
        print(f"Total cores to use: {total_cores}, Cores per trajectory: {cores_per_traj}")

        reactions = [] #for differnet trajectories

        for itraj in range(ntraj_paralell):
            react = Collision(fragA, fragB, qchem=qchem, itraj=itraj)

            react.Specify_Collision_Sampling(Rini=Rini, bmax=bmax, bsampling=bsampling, Ecoll=Ecoll, Ecoll_thermal=Ecoll_thermal, temp=temp,
                                             pairs_to_stop=pairs_to_stop)

            reactions.append(react)

        return reactions
    #------------------------------------------------------------------------------------------------------------


    #------------------------------------------------------------------------------------------------------------
    def run_paralell_reactions(reactions, delay_between_jobs = 1, integrator='leapfrog', integrator_order=4,
                               timestep=0.5, maxstep=100, iprint=1, spectrum=False):   
    #------------------------------------------------------------------------------------------------------------
        from concurrent.futures import ProcessPoolExecutor
        import time

        ntraj_paralell = len(reactions)
        print(f"\nnumber of paralell collisions: {ntraj_paralell}\n")
        with ProcessPoolExecutor(max_workers=ntraj_paralell) as executor:
            futures = []
            it = -1
            for reaction in reactions:
                # Add delay before launching the next job if specified
                it += 1 
                if delay_between_jobs > 0:
                    time.sleep(delay_between_jobs)

                print(f"\n############### Trajectory {it} is running using ################\n")
                futures.append(executor.submit(reaction.sample_and_run_collision,
                                                integrator, integrator_order,
                                                timestep, maxstep, iprint)
                               )


        # Wait for all tasks to complete
        for future in futures:
            try:
                future.result()  # Get result or raise an exception if any occurred
            except Exception as e:
                print(f"Error in trajectory: {e}")
       #------------------------------------------------------------------------------------------------------------


    reactions = setup_reactions(H2O2, OH, qchem=qcinput_PySCF, Rini=4.0, bmax=2.0, bsampling=True, Ecoll_thermal=True, temp=300.0, pairs_to_stop=pairs_to_test)

    print("About to call run_paralell_reactions()")
    run_paralell_reactions(reactions, delay_between_jobs = 1, integrator='leapfrog', integrator_order=4,
                                   timestep=0.7, maxstep=10, iprint=1)
 
