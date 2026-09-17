from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == '__main__':
    import random
    from smite import Molecule, Fragment, Collision

    #ZdeltaOH-peroxy+O2_Case1 M062X avtz opt
    xyz_ZdeltaOH = '''
      C   0.10213079516265     -0.40335082664479      0.07543528539646
      C   0.00231507387603      0.00802344726355      1.33653797561024
      C   1.29261263667164     -0.10109601567464     -0.78577068343470
      H   0.81871300831184      0.57777501379411      1.77060573809632
      C   -1.15009257741011     -0.22944443636352      2.26272853544681
      C   -1.00797013579701     -1.17237892534706     -0.58491050082761
      O   -2.16167483039548     -0.29909812294270     -0.72700722676734
      O   -3.12707714552250     -0.89834017698997     -1.35106072849739
      H   -0.73814320117304     -1.50106582256588     -1.58678218934808
      H   -1.33749817030000     -2.03251034450817     -0.00291414960058
      H   1.73717180355810     -1.02300433466165     -1.16706734211551
      H   0.99681014299398      0.49236522207854     -1.65371501497305
      H   2.05296779799310      0.44647846921573     -0.23347319460176
      H   -1.87279302286029     -0.93109039434383      1.83784949130262
      O   -0.70869446056454     -0.66187057091895      3.53955774422386
      H   -1.68093871610262      0.70722576340998      2.43674947835721
      H   -0.19992899844176     -1.46881794480076      3.43166678173249
     '''

    qcinput_XTB = {
    'qchem': 'XTB',
    'path': '/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/xtb',
    'nproc': 4,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 2,
    'additional': ' --ptb --iterations 200',
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
    'multiplicity': 2,
    'additional': '-I 200 --density_rmsd_criterion 1e-3 --self_consistence_criterion 1e-5',
    'wfu': False
    }


    rigid = False
    random_rot = False #True

    seed = 12249112
    random.seed(seed)

    ZdeltaOH  = Fragment.Polyatom_Init(fname='ZdeltaOH', qchem=qcinput_XTB, xyz=xyz_ZdeltaOH, random_rot=False)

    ZdeltaOH.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=0)

    print('\nNVT segment equilibriate at 500 K:')
    ZdeltaOH.sample_and_run_trajectory(traj_file='NVT_equilibriate.xyz',integrator='leapfrog', integrator_order=4, timestep=0.5, maxstep=500, iprint=1, Rstop=10.0, thermostat='berendsen', thermo_param=2.0, thermo_temp=300.0) 
    print('\nNVT segment production at 500 K:')
    ZdeltaOH.run_trajectory(traj_file='NVT_production.xyz', integrator='leapfrog', integrator_order=4, timestep=0.3, maxstep=3000, iprint=1, Rstop=10.0, thermostat='berendsen', thermo_param=5.0, thermo_temp=300.0) 
    print('\nNVE segment:')
    ZdeltaOH.run_trajectory(traj_file='NVE_final.xyz', integrator='symplectic', integrator_order=4, timestep=0.3, maxstep=9000, iprint=1, Rstop=10.0, spectrum=True) 

    #print(diene.vsave[46])

    ZdeltaOH.vibrational_spectrum(dt=0.3, print_maxfreq=8000.0)




   




