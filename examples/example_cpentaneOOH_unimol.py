from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == '__main__':
    import random
    from smite import Molecule, Fragment, Collision

    #xtb optimized (with spinpol) cyclo-pentane-OOH 
    xyz = '''
    C            0.01081395105239        0.01558146911373       -0.10103220609204
    C            0.14946988284407       -0.28886528558037        1.34834341099794
    C            1.40861348995409        0.25709181478668        1.90847504986098
    C            1.90101925234989        1.22688021474615        0.82688719375338
    C            1.36604991655964        0.63670186467838       -0.47940666335403
    H           -0.62934553084510       -0.74528594519601        1.92876195980860
    H            2.13970940391761       -0.54848073971897        2.06378902049673
    H            1.25901303519659        0.74240591688463        2.87556940456133
    H            2.98462204898756        1.33271411246099        0.81813254638113
    H            1.45215680121995        2.20639229963877        0.99051474533383
    H            2.02803698296961       -0.14994762507947       -0.84246585761089
    H            1.24775215243452        1.38684658307065       -1.25847828760131
    H           -0.23149082416073       -0.86820981024594       -0.70805099770180
    O           -1.11261749575066        0.82632302269654       -0.42383003430539
    O           -1.06591675844812        2.07390158937051        0.27397656674813
    H           -2.00238130828186        2.29417751837391        0.35500814872365
     '''

    qcinput_XTB = {
    'qchem': 'XTB',
    'path': '/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/xtb',
    'nproc': 4,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 2,
    'additional': '--iterations 200 --tblite',
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
    'multiplicity': 2,
    'additional': '',
    'wfu': True
    }
    
    qcinput_Sparrow_bin = {
    'qchem': 'Sparrow_bin',
    'path': '/home/peter/Programs/sparrow/install/bin/sparrow',
    'nproc': 4,
    'functional': 'DFTB3',
    'basis': '',
    'charge': 0,
    'multiplicity': 2,
    'additional': '-I 200 --density_rmsd_criterion 1e-3 --self_consistence_criterion 1e-5',
    'wfu': False
    }


    fix_quantum = [(41, 3)]

    fix_energy  = [(0, 0.0),
                   (1, 0.0),
                   (2, 0.0)]

    fix_temp    = [(5, 330.0),
                   (6, 430.0)]

    seed = 13349112
    random.seed(seed)

    QOOH  = Fragment.Polyatom_Init(fname='CPOOH', qchem=qcinput_XTB, xyz=xyz, random_rot=False)

   #diene.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=10, fix_quantum=fix_quantum, fix_temp=fix_temp)
    QOOH.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=0)

    QOOH.sample_and_run_trajectory(traj_file='CPOOH_OH-exc_exc_3OH.xyz',integrator='symplectic', integrator_order=4, timestep=0.7, maxstep=8000, iprint=1, Rstop=10.0, spectrum=True) 

    QOOH.vibrational_spectrum(dt=0.5, print_maxfreq=7000.0)




   



