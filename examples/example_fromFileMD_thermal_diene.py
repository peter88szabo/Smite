from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == '__main__':
    import random
    from smite import Molecule, Fragment, Collision

    xyz_water = '''
      O    -0.011100  0.0000  -0.00788
      H     0.007500  0.0000   0.95111
      H     0.899200  0.0000  -0.30990
       '''

    #IRC endpoint of openchain cis-1,3,5 triene
    xyz_diene = '''
      C      -1.185385      1.500364     -0.174799
      C       0.057100      1.525641      0.290486
      C       1.182421      0.671307     -0.090281
      C       1.182420     -0.671306     -0.090282
      C       0.057100     -1.525640      0.290485
      C      -1.185388     -1.500364     -0.174796
      H       0.285347      2.226153      1.090336
      H       2.144030      1.164815     -0.199258
      H       2.144029     -1.164815     -0.199261
      H       0.285350     -2.226154      1.090332
      H      -1.972016     -2.076478      0.294389
      H      -1.462109     -0.924082     -1.042658
      H      -1.972017      2.076476      0.294382
      H      -1.462101      0.924083     -1.042665
     '''

    qcinput_XTB = {
    'qchem': 'XTB',
    'path': '/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/xtb',
    'nproc': 4,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 1,
    'additional': '--acc 50 --iterations 200',
    'wfu': False
    }

    qcinput_Orca = {
    'qchem': 'Orca',
    'path': '/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/orca',
    'nproc': 4,
    'functional': 'HF-3c',
    'basis': '',
    'charge': 0,
    'multiplicity': 1,
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
    'functional': 'DFTB3',
    'basis': '',
    'charge': 0,
    'multiplicity': 1,
    'additional': '-I 200 --density_rmsd_criterion 1e-3 --self_consistence_criterion 1e-5',
    'wfu': False
    }


    rigid = False
    random_rot = False #True

    fix_quantum = [(14, 1),
                   (15, 1)]

    fix_energy  = [(0, 0.0),
                   (1, 0.0),
                   (2, 0.0)]

    fix_temp    = [(5, 330.0),
                   (6, 430.0)]

    fix_quantum_water = [(0, 6)]

    seed = 12949102
    random.seed(seed)

    #water  = Fragment.Polyatom_Init(fname=fname_water, qchem=qcinput_singlet, xyz=xyz_water, random_rot=False)
    #water.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=3)

    diene  = Fragment.Polyatom_Init(fname='diene', qchem=qcinput_XTB, xyz=xyz_diene, fromMD=True, random_rot=False)

   #diene.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=10, fix_quantum=fix_quantum, fix_temp=fix_temp)
    #diene.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=0, fix_quantum=fix_quantum)
    #diene.Specify_Mode_Sampling(MDfile='NVT_production.xyz', init_rot_type='Jfix', jrot=0) # temp is for the intiail momentum selection
    diene.Specify_Mode_Sampling(MDfile='NVT_production.xyz') # temp is for the intiail momentum selection

    print('\nNVT segment equilibriate at 500 K:')
    diene.sample_and_run_trajectory(traj_file='NVE_initial_from_MDfile.xyz',
                                    integrator='leapfrog',
                                    integrator_order=4,
                                    timestep=0.5,
                                    maxstep=1000,
                                    iprint=1,
                                    Rstop=12.0)
                                    #thermostat='berendsen',
                                    #thermo_param=10.0,
                                    #thermo_temp=500.0) 




   




