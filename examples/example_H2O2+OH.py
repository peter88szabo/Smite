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
    'nproc': 4,
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
    'nproc': 4,
    #'functional': 'HF-3c',
    'functional': 'wB97X-3c',
    #'functional': 'PBEh-3c',
    'basis': '',
    'charge': 0,
    'multiplicity': 2,
    'additional': '',
    'wfu': False,
    'scf_guess_mode': 'CMatrix'
    }

    qcinput_PySCF = {
    'qchem': 'PySCF',
    'path': '',
    'nproc': 4,
    'functional': 'PBE',
    'basis': 'sto-3g',
    'charge': 0,
    'multiplicity': 1,
    'additional': '',
    'wfu': True
    }
    
    qcinput_Sparrow_bin = {
    'qchem': 'Sparrow_bin',
    'path': '/home/peter/Programs/sparrow/install/bin/sparrow',
    'nproc': 4,
    'functional': 'PM6',
    'basis': '',
    'charge': 0,
    'multiplicity': 2,
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

    reaction =  Collision(H2O2, OH, qchem=qcinput_Orca_doublet) 
    reaction.Specify_Collision_Sampling(Rini=5.2, bmax=5.0, bsampling=True, Ecoll_thermal=True, temp=300.0)

    reaction.sample_and_run_collision(integrator='leapfrog', integrator_order=4, timestep=0.7, maxstep=10000, iprint=4, pairs_to_stop=pairs_to_test)


   

