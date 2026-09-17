from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == '__main__':
    import random
    from smite import Molecule, Fragment, Collision

    xyz_formicacid = '''
    O            0.02519086968664        0.00000000004832        0.05577741304340
    C           -0.13442694410193       -0.00000000026232        1.36849020460625
    H            0.85176199884867        0.00000000012085        1.85801158304635
    O           -1.18087820488110        0.00000000007319        1.94046212548985
    H           -0.83617471955228        0.00000000001996       -0.39680932618584
    '''


    qcinput_xtb_doublet = {
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

    #'additional': '--gfnff',

    qcinput_xtb_singlet = {
    'qchem': 'XTB',
    'path': '/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/xtb',
    'nproc': 4,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 1,
    'additional': '--iterations 1000 --tblite',
    'wfu': False 
    }

    qcinput_Orca_singlet = {
    'qchem': 'XTB',
    'path': '/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/orca',
    'nproc': 4,
    'functional': 'b97-3c',
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
    'functional': 'b97-3c',
    #'functional': 'wB97X-3c',
    #'functional': 'PBEh-3c',
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


    qcinp_doublet = qcinput_Orca_doublet 
    qcinp_singlet = qcinput_xtb_singlet

    fix_quantum = [(14, 0),
                   (15, 0)]

    fix_energy  = [(0, 0.0),
                   (1, 0.0),
                   (2, 0.0)]

    fix_temp    = [(5, 330.0),
                   (6, 430.0)]

    seed = 20292933
    random.seed(seed)

    formicacid  = Fragment.Polyatom_Init(fname='formicacid', qchem=qcinp_singlet, xyz=xyz_formicacid, random_rot=True)
    #zzallyl.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=0, fix_quantum=fix_quantum)
    #zzallyl.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Temp', temp=300.0)
    formicacid.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=0)

    req_OH = 0.96 #Angstrom
    omega_OH = 3808.2 #cm-1

    OH = Fragment.Diatom_Init(fname='OH', atoms=['O','H'], req=req_OH, omega=omega_OH, random_rot=True, diatom='harmonic')
    #OH.Specify_Mode_Sampling(init_rot_type='Jfix', nvib=0, jrot=0)
    OH.Specify_Mode_Sampling(init_rot_type='Temp', nvib=0, temp=300.0)

    print("----------------------- Formicacid ---------------------")
    formicacid.print_mode_sampling()
    print("------------- Formicacid Sampling is done --------------")

    pairs_to_test = {
    'CH-bite': [((2, 5), 'LT', 1.4), ((1, 5), 'GT', 6.8)],
    'OH-bite': [((4, 5), 'LT', 1.4), ((1, 5), 'GT', 6.8)],
    'Nonreact': [((1, 5), 'GT', 6.8), ((0, 4), 'LT', 1.4),  ((1, 2), 'LT', 1.4)],
    # add more channels and pairs as needed
    }

    print("\nReaction of Formicacid + OH\n")

    reaction =  Collision(formicacid, OH, qchem=qcinp_doublet) 
    reaction.Specify_Collision_Sampling(Rini=6.5, bmax=5.0, bsampling=True, Ecoll_thermal=True, temp=300.0)

    reaction.sample_and_run_collision(integrator='leapfrog', integrator_order=4, timestep=0.5, maxstep=6000, iprint=2, pairs_to_stop=pairs_to_test)


   



