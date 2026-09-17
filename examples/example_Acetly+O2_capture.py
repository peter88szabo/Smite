from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == '__main__':
    import random
    from smite import Molecule, Fragment, Collision

    #ZZAllyl-peroxy+O2_Case1 M062X avtz optim Conformer 21
    xyz_acetyl = '''
    C           -0.00518562527576        0.00000001989751       -0.00950000391940
    C            0.08627378663703        0.00000010570051        1.47034849520302
    O            1.01528257754469       -0.00000011452468        2.21435669252749
    H           -1.04057015714535       -0.00000004761165       -0.34096832906429
    H            0.50037517519675        0.88551384184127       -0.39711897761138
    H            0.50037524304264       -0.88551380530297       -0.39711887713545

     '''

    qcinput_XTB = {
    'qchem': 'XTB',
    'path': '/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/xtb',
    'nproc': 4,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 2,
    'additional': '--gfnff',
    'wfu': False
    }

    qcinput_Orca = {
    'qchem': 'Orca',
    'path': '/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/orca',
    'nproc': 4,
    #'functional': 'HF-3c',
    #'functional': 'wB97X-3c',
    #'functional': 'PBEh-3c',
    'functional': 'b97-3c',
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
    'functional': 'B3LYP',
    'basis': 'pc1',
    'charge': 0,
    'multiplicity': 2,
    'additional': '',
    'wfu': False
    }

    qcinp = qcinput_Orca

    seed = 12979102
    random.seed(seed)


    acetyl  = Fragment.Polyatom_Init(fname='Acetyl', qchem=qcinp, xyz=xyz_acetyl, random_rot=True)
    acetyl.Specify_Mode_Sampling(init_vib_type='Temp', init_rot_type='Temp', temp=300.0)

    req_O2 = 1.2 #Angstrom
    omega_O2 = 1580.0 #cm-1

    oxygen = Fragment.Diatom_Init(fname='O2', atoms=['O','O'], req=req_O2, omega=omega_O2, random_rot=True, diatom='harmonic')
    oxygen.Specify_Mode_Sampling(init_rot_type='Temp', nvib=0, temp=300.0)

    print("--------- Acetyl radical--------")
    acetyl.print_mode_sampling()
    print("--------- Acetly radical DONE--------")

    pairs_to_test = {
    'capture_1': [((1, 6), 'LT', 1.5)],  
    'capture_2': [((1, 7), 'LT', 1.5)],  
    'non-react': [((1, 7), 'GT', 7.5)],  
    # add more channels and pairs as needed
    }

    print("\nReaction of Acetly + O2")
    print()


    reaction =  Collision(acetyl, oxygen, qchem=qcinp) 
    reaction.Specify_Collision_Sampling(Rini=6.5, bmax=3.0, bsampling=True, Ecoll_thermal=True, temp=300.0)

    reaction.sample_and_run_collision(integrator='leapfrog', integrator_order=4, timestep=0.5, maxstep=10000, iprint=1, pairs_to_stop=pairs_to_test)


   




