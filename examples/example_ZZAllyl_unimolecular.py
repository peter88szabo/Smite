from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == '__main__':
    import random
    from smite import Molecule, Fragment, Collision

    #ZZAllyl-peroxy+O2_Case1 M062X avtz optim Conformer 21
    xyz_ZZAllyl = '''
    C   -0.19595538763398      0.07192231509414      0.00714857985445
    C   -0.01703579534250     -0.10344222305327      1.49976587011423
    C   -1.00831293517504     -0.74304790940588      2.21484017141187
    C   -1.09565313995228     -0.96572164058650      3.69139746432409
    O   -2.44905662107855     -1.09778426038872      4.10589699809888
    O   -3.03738052053937      0.19479040763344      4.06761535894273
    C   1.14919758827461      0.41511797958774      2.03274034295381
    O   1.44878216770210      0.30476609400103      3.35160340084751
    H   -1.82826168270363     -1.17741133496109      1.65145938615843
    H   1.87669555170293      0.91631236085033      1.40830140902315
    H   0.76154446389898      0.10310849679981     -0.51109606581980
    H   -0.77393280198763     -0.75201110449544     -0.40695658995199
    H   -0.72775230388759      0.99760660592287     -0.21471742835545
    H   -0.61505142152353     -0.17733195303782      4.26524823267036
    H   -0.65062609284075     -1.92286967316247      3.98246076165767
    H   2.29766286152308      0.71228566068994      3.53361144883480
    H   -3.35772393043684      0.25312017851188      3.15850065923520
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

    ZZAllyl  = Fragment.Polyatom_Init(fname='ZZAllyl', qchem=qcinput_XTB, xyz=xyz_ZZAllyl, random_rot=False)

    ZZAllyl.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=0)

    print('\nNVT segment equilibriate at 500 K:')
    ZZAllyl.sample_and_run_trajectory(traj_file='NVT_equilibriate.xyz',integrator='leapfrog', integrator_order=4, timestep=0.5, maxstep=500, iprint=1, Rstop=10.0, thermostat='berendsen', thermo_param=2.0, thermo_temp=300.0) 
    print('\nNVT segment production at 500 K:')
    ZZAllyl.run_trajectory(traj_file='NVT_production.xyz', integrator='leapfrog', integrator_order=4, timestep=0.5, maxstep=2000, iprint=1, Rstop=10.0, thermostat='berendsen', thermo_param=5.0, thermo_temp=300.0) 
    print('\nNVE segment:')
    ZZAllyl.run_trajectory(traj_file='NVE_final.xyz', integrator='symplectic', integrator_order=4, timestep=0.3, maxstep=15000, iprint=1, Rstop=10.0, spectrum=True) 

    #print(diene.vsave[46])

    ZZAllyl.vibrational_spectrum(dt=0.3, print_maxfreq=8000.0)




   




