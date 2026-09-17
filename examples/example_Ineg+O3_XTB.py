from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == '__main__':
    import random
    from smite import Molecule, Fragment, Collision

   #O3 optimized with Psi4 r2scan-3c:
    xyz_O3 = '''
     O   -1.045794039043    0.361044294544    0.000000000000
     O    0.218502671235    0.373548742271    0.000000000000
     O    0.827291367808   -0.734593036815    0.000000000000
     '''

    qcinput_neutral = {
    'qchem': 'XTB',
    'path': '/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/xtb',
    'nproc': 4,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 1,
    'additional': '--gfn1',
    'wfu': False
    }


    qcinput_negative = {
    'qchem': 'XTB',
    'path': '/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/xtb',
    'nproc': 4,
    'functional': '',
    'basis': '',
    'charge': -1,
    'multiplicity': 1,
    'additional': '--gfn1 --etemp 1000.0',
    'wfu': False
    }

    #seed = 4272yy_seed01
    #random.seed(seed)


    Ozon  = Fragment.Polyatom_Init(fname='O3', qchem=qcinput_neutral, xyz=xyz_O3, random_rot=True)
    Ozon.Specify_Mode_Sampling(init_vib_type='Temp', init_rot_type='Temp', temp=300.0)


    Iodide = Fragment.Atom_Init(fname='Iodide', atoms=['I'])

    print("--------- Ozon --------")
    Ozon.print_mode_sampling()
    print("--------- Ozon DONE--------")

    pairs_to_test = {
    'capture_alpha': [((0, 3), 'LT', 2.7)],  
    'capture_delta': [((2, 3), 'LT', 2.7)],  
    'nonreact': [((0, 3), 'GT', 8.0), ((1, 3), 'GT', 8.0), ((2, 3), 'GT', 8.0)],  
    'react_IO+O2_A': [((0, 1), 'GT', 10.0), ((0, 3), 'LT', 3.5)],  
    'react_IO+O2_B': [((1, 2), 'GT', 10.0), ((2, 3), 'LT', 3.5)],  
    # add more channels and pairs as needed
    }

    print("\nReaction of Ineg + O3")
    print()


    reaction =  Collision(Ozon, Iodide, qchem=qcinput_negative) 
    reaction.Specify_Collision_Sampling(Rini=7.0, bmax=5.0, bsampling=1, Ecoll_thermal=True, temp=300.0)

    reaction.sample_and_run_collision(integrator='verlet', timestep=0.5, maxstep=10000, iprint=4, pairs_to_stop=pairs_to_test)

