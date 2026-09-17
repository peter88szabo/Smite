from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == '__main__':
    import random
    from smite import Molecule, Fragment, Collision

    #trans-IsoOH Case-2 conf-11 (r2SCAN-3c optimized, but the same as with M062X-D3)
    #Coordinates from ORCA-job IsoOH_Case-2_M062Xopt_11_Compound_1 E -271.040327889081
    xyz_IsoOH = '''
    C          -0.04133087943945      0.07114250078738      1.38999071346232
    C           0.02070690184692      0.01666510282459      0.00788362371526
    C          -1.26856364750845      0.00193604142078     -0.78526109220447
    C           1.25671559933393     -0.01291622892551     -0.64027788592838
    C           1.47848498555495     -0.08846749550178     -2.10972256189703
    H          -0.99051468103380      0.09713273461484      1.91341807774138
    H           0.86502010385865      0.09006595320454      1.98672613761931
    H           2.15887766668238      0.00324193447454     -0.02997803564743
    H           0.52959627377747     -0.17538849005273     -2.65884146455451
    H           1.98306008511518      0.81705697518545     -2.47372434820703
    H          -1.35703924866629      0.88913674900855     -1.42133762484310
    H          -1.32867369902273     -0.87570639962484     -1.43815472591279
    H          -2.13252287462749     -0.01861509393368     -0.11705314031457
    O           2.36925369873426     -1.16389173263393     -2.45883424444901
    H           2.02962571539443     -1.96081955084824     -2.03469142857991
     '''

    qcinput_Orca_doublet = {
    'qchem': 'PySCF',
    'path': '',
    'nproc': 4,
    'functional': 'r2SCAN',
    'basis': 'def2-SVP',
    'charge': 0,
    'multiplicity': 2,
    'additional': 'TightSCF',
    'wfu': False
    }

    #seed = 4272yy_seed01
    #random.seed(seed)


    IsoOH  = Fragment.Polyatom_Init(fname='IsoOH', qchem=qcinput_Orca_doublet, xyz=xyz_IsoOH, random_rot=True)
    IsoOH.Specify_Mode_Sampling(init_vib_type='Temp', init_rot_type='Temp', temp=300.0)

    #r2SCAN-3c optimized:
    req_O2 = 1.21001 #Angstrom
    omega_O2 = 1610.19 #cm-1

    oxygen = Fragment.Diatom_Init(fname='O2', atoms=['O','O'], req=req_O2, omega=omega_O2, random_rot=True, diatom='harmonic')
    #oxygen.Specify_Mode_Sampling(init_rot_type='Jfix', nvib=0, jrot=0)
    oxygen.Specify_Mode_Sampling(init_rot_type='Temp', nvib=0, temp=300.0)

    print("--------- Trans-Isoprene-OH radical--------")
    IsoOH.print_mode_sampling()
    print("--------- Trans-Isoprene-OH radical DONE--------")

    pairs_to_test = {
    'capture_delta_1': [((0, 15), 'LT', 1.5)],  
    'capture_delta_2': [((0, 16), 'LT', 1.5)],  
    'capture_beta_1': [((3, 15), 'LT', 1.5)],  
    'capture_beta_2': [((3, 16), 'LT', 1.5)],  
    'nonreact_1': [((0, 16), 'GT', 11.0), ((0, 15), 'GT', 11.0)],  
    'nonreact_2': [((3, 16), 'GT', 11.0), ((3, 15), 'GT', 11.0)],  
    'react_break-O-H': [((13, 14), 'GT', 5.0)],  
    'react_break-Calpha-H1': [((4, 8), 'GT', 5.0)],  
    'react_break-Calpha-H2': [((4, 9), 'GT', 5.0)],  
    # add more channels and pairs as needed
    }

    print("\nReaction of Trans-Isoprene-OH + O2")
    print()


    reaction =  Collision(IsoOH, oxygen, qchem=qcinput_Orca_doublet) 
    reaction.Specify_Collision_Sampling(Rini=8.0, bmax=5.0, bsampling=1, Ecoll_thermal=True, temp=300.0)

    #reaction.sample_and_run_collision(integrator='verlet', integrator_order=4, timestep=0.5, maxstep=10000, iprint=2, pairs_to_stop=pairs_to_test)
    reaction.sample_and_run_collision(integrator='verlet', integrator_order=4, timestep=0.5, maxstep=6, iprint=1, pairs_to_stop=pairs_to_test)

