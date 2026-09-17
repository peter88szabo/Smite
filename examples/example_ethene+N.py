from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == '__main__':
    import random
    from smite import Molecule, Fragment, Collision

    '''
    Phys. Chem. Chem. Phys., 2011,13, 8515-8525
    https://doi.org/10.1039/C0CP02439B
    '''
    xyz_ethene = '''
    C            0.00355319473205        0.00000000000006        0.01089711077076
    C            0.00198987070772        0.00000000000009        1.32724967003551
    H            0.91813365392387       -0.00000000000003       -0.55970678610572
    H            0.91539774402372       -0.00000000000004        1.89979746520437
    H           -0.91279771819434       -0.00000000000004        1.89759827028969
    H           -0.90982174519301       -0.00000000000003       -0.56133573019461
    '''

    qcinput_singlet = {
    'qchem': 'XTB',
    'path': '/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/xtb',
    'nproc': 4,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 1,
    'additional': '--gfnff --acc 50 --iterations 300 --tblite',
    'wfu': False
    }

    qcinput_doublet = {
    'qchem': 'XTB',
    'path': '/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/xtb',
    'nproc': 4,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 2,
    'additional': '--gfnff --acc 50 --iterations 300 --tblite',
    'wfu': False
    }


    qcinput_Orca = {
    'qchem': 'Orca',
    'path': '/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/orca',
    'nproc': 4,
    'functional': 'HF-3c',
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
    'multiplicity': 1,
    'additional': '-I 200 --density_rmsd_criterion 1e-3 --self_consistence_criterion 1e-5',
    'wfu': False
    }

    seed = 12949102
    random.seed(seed)


    fix_quantum = [(14, 0),
                   (15, 0)]

    fix_energy  = [(0, 0.0),
                   (1, 0.0),
                   (2, 0.0)]

    fix_temp    = [(5, 330.0),
                   (6, 430.0)]

    fix_quantum_ethene = [(0, 6)]

    ethene  = Fragment.Polyatom_Init(fname='ethene', qchem=qcinput_singlet, xyz=xyz_ethene, random_rot=True)
    ethene.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=3)

    nitrogen = Fragment.Atom_Init(fname='N', atoms=['N'])

    print("--------- Ethene mode sampling---------")
    ethene.print_mode_sampling()
    print("--------------------------------------")


    pairs_to_test = {
    'Habstr_1': [((0, 1), 'GT', 7.5)],
    'Habstr_2': [((0, 2), 'GT', 7.5)],
    'Nonreact': [((0, 1), 'LT', 2.5), ((0, 2), 'LT', 2.5), ((0, 3), 'GT', 8.0)]
    # add more channels and pairs as needed
    }

    print("\nReaction of Ethene + N(2D)\n")

    reaction = Collision(ethene, nitrogen, qchem=qcinput_doublet) 
    reaction.Specify_Collision_Sampling(Rini=6.0, bmax=3.0, bsampling=True, Ecoll=20.0, temp=300.0)

    reaction.sample_and_run_collision(integrator='leapfrog', integrator_order=4, timestep=0.7, maxstep=10000, iprint=2, pairs_to_stop=pairs_to_test)

