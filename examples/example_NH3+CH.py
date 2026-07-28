from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == '__main__':
    import random
    from smite import Molecule, Fragment, Collision


    '''
    https://pubs.acs.org/action/showCitFormats?doi=10.1021/acs.jpclett.3c02360&ref=pdf
    J Phys Chem Lett. 2023 Sep 28;14(38):8500-8506. doi: 10.1021/acs.jpclett.3c02360
    '''

    #xtb opt
    xyz_NH3 = '''
     N           -0.00737607617840       -0.01255290510709       -0.00537570671164
     H            0.00705402855711        0.01192834477483        1.00647705419603
     H            0.95144560228079        0.01173880263613       -0.32878525460555
     H           -0.46557355465950        0.82987675769613       -0.32898109287884
    '''

    qcinput_doublet = {
    'qchem': 'XTB',
    'path': '/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/xtb',
    'nproc': 4,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 2,
    'additional': '--gfnff --acc 50 --iterations 1000 --tblite',
    'wfu': False 
    }

    qcinput_singlet = {
    'qchem': 'XTB',
    'path': '/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/xtb',
    'nproc': 4,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 1,
    'additional': '--gfnff --acc 50 --iterations 1000',
    'wfu': False
    }

    qcinput_Orca_doublet = {
    'qchem': 'Orca',
    'path': '/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/orca',
    'nproc': 4,
    'functional': 'HF-3c',
    #'functional': 'wB97X-3c',
    #'functional': 'PBEh-3c',
    'basis': '',
    'charge': 0,
    'multiplicity': 2,
    'additional': '',
    'wfu': False
    }

    qcinput_PySCF_singlet = {
    'qchem': 'PySCF',
    'path': '',
    'nproc': 4,
    'functional': 'wb97x',
    'basis': 'sto-3g',
    'charge': 0,
    'multiplicity': 1,
     'additional': '',
    'wfu': False
    }


    qcinput_PySCF_doublet = {
    'qchem': 'PySCF',
    'path': '',
    'nproc': 4,
    'functional': 'wb97x',
    'basis': 'sto-3g',
    'charge': 0,
    'multiplicity': 2,
     'additional': '',
    'wfu': False
    }

    import os
    nproc = qcinput_PySCF_singlet['nproc']
    os.environ['OMP_NUM_THREADS'] = str(nproc)
    
    qcinput_Sparrow_bin = {
    'qchem': 'Sparrow_bin',
    'path': '/home/peter/Programs/sparrow/install/bin/sparrow',
    'nproc': 4,
    'functional': 'PM6',
    'basis': '',
    'charge': 0,
    'multiplicity': 1,
    'additional': '-I 200 --density_rmsd_criterion 1e-3 --self_consistence_criterion 1e-5',
    'wfu': False
    }

    fix_quantum = [(14, 0),
                   (15, 0)]

    fix_energy  = [(0, 0.0),
                   (1, 0.0),
                   (2, 0.0)]

    fix_temp    = [(5, 330.0),
                   (6, 430.0)]

    seed = 87655922
    random.seed(seed)


    NH3  = Fragment.Polyatom_Init(fname='NH3', qchem=qcinput_PySCF_singlet, xyz=xyz_NH3, random_rot=True)
    #zzallyl.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=0, fix_quantum=fix_quantum)
    #zzallyl.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Temp', temp=300.0)
    NH3.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=10)

    #doublet CH radical optimized by XTB
    req_CH = 1.094 #Angstrom
    omega_CH = 2759.29 #cm-1
     
    CH = Fragment.Diatom_Init(fname='CH', atoms=['C','H'], req=req_CH, omega=omega_CH, random_rot=True, diatom='harmonic')
    #OH.Specify_Mode_Sampling(init_rot_type='Jfix', nvib=0, jrot=0)
    CH.Specify_Mode_Sampling(init_rot_type='Temp', nvib=0, temp=300.0)

    print("----------------------- NH3 --------------------")
    NH3.print_mode_sampling()
    print("-------------NH3 Sampling is done---------------")

    pairs_to_test = {
    'reaction_N-H1': [((1, 0), 'GT', 14.0)],  
    'reaction_N-H2': [((2, 0), 'GT', 14.0)],  
    'reaction_N-H3': [((3, 0), 'GT', 14.0)],  
    # add more channels and pairs as needed
    }

    print("\nReaction of NH3 + CH\n")

    reaction =  Collision(NH3, CH, qchem=qcinput_PySCF_doublet) 
    reaction.Specify_Collision_Sampling(Rini=6.0, bmax=5.0, bsampling=True, Ecoll_thermal=True, temp=100.0)

    reaction.sample_and_run_collision(integrator='leapfrog', integrator_order=4, timestep=0.7, maxstep=10000, iprint=4, pairs_to_stop=pairs_to_test)


   

