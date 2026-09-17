from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == '__main__':
    import random
    from smite import Molecule, Fragment, Collision

    xyz_isoprene = '''
      C         0.01275          0.00000         1.35292
      C        -0.04520         -0.00000         0.02009
      C        -1.29563         -0.00000        -0.76814
      C        -1.28239          0.00000        -2.10714
      C        -2.58923         -0.00000        -0.01147
      H         0.87318         -0.00000        -0.57632
      H         0.94110          0.00000         1.90529
      H        -0.85649          0.00000         1.99530
      H        -2.16634         -0.00000        -2.71174
      H        -0.37079         -0.00000        -2.69792
      H        -3.46327         -0.00000        -0.67466
      H        -2.66984         -0.88694         0.63214
      H        -2.66984          0.88694         0.63214
     '''

    qcinput_doublet = {
    'qchem': 'XTB',
    'path': '/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/xtb',
    'nproc': 4,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 2,
    'additional': '--acc 50 --iterations 1000 --tblite',
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
    'additional': '--acc 50 --iterations 1000',
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

    fix_quantum = [(14, 0),
                   (15, 0)]

    fix_energy  = [(0, 0.0),
                   (1, 0.0),
                   (2, 0.0)]

    fix_temp    = [(5, 330.0),
                   (6, 430.0)]

    seed = 12356133
    random.seed(seed)

    isoprene  = Fragment.Polyatom_Init(fname='isoprene', qchem=qcinput_singlet, xyz=xyz_isoprene, random_rot=True)
    #zzallyl.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=0, fix_quantum=fix_quantum)
    #zzallyl.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Temp', temp=300.0)
    isoprene.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=10)

    req_OH = 0.96 #Angstrom
    omega_OH = 3808.2 #cm-1

    OH = Fragment.Diatom_Init(fname='OH', atoms=['O','H'], req=req_OH, omega=omega_OH, random_rot=True, diatom='harmonic')
    #OH.Specify_Mode_Sampling(init_rot_type='Jfix', nvib=0, jrot=0)
    OH.Specify_Mode_Sampling(init_rot_type='Temp', nvib=0, temp=300.0)

    print("----------------------- Isoprene ---------------------")
    isoprene.print_mode_sampling()
    print("------------- Isoprene Sampling is done --------------")

    pairs_to_test = {
    'capture_1': [((0, 13), 'LT', 1.4)],
    'capture_2': [((1, 13), 'LT', 1.4)],
    'capture_3': [((2, 13), 'LT', 1.4)],
    'capture_4': [((3, 13), 'LT', 1.4)],
    'reaction_1': [((2, 13), 'GT', 8.0)],
    # add more channels and pairs as needed
    }

    print("\nReaction of Isoprene + OH\n")

    reaction =  Collision(isoprene, OH, qchem=qcinput_doublet) 
    #reaction =  Collision(isoprene, OH, qchem=qcinput_doublet) 
    reaction.Specify_Collision_Sampling(Rini=7.0, bmax=4.0, bsampling=True, Ecoll_thermal=True, temp=300.0)

    reaction.sample_and_run_collision(integrator='symplectic', integrator_order=4, timestep=0.7, maxstep=10000, iprint=2, pairs_to_stop=pairs_to_test)


   



