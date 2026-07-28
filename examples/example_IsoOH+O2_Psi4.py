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
    #Eventually Optimized with Psi4 r2SCAN-3c
    xyz_IsoOH = '''
     C   -0.711214232886    0.304600841541    2.228590453231
     C   -0.649403026600    0.251101879894    0.846407032294
     C   -1.937566841572    0.234708225750    0.052663782834
     C    0.586138668908    0.223973692562    0.196793900946
     C    0.809083282573    0.147130435363   -1.272039967709
     H   -1.659747672567    0.328715351531    2.752098903981
     H    0.194771809813    0.324705582680    2.824787548487
     H    1.487909101543    0.240987350792    0.806059791385
     H   -0.138917724343    0.062853587797   -1.821475247145
     H    1.317531379755    1.049724737740   -1.635721892814
     H   -2.025520235037    1.121080008792   -0.583556877823
     H   -1.995082447941   -0.642723760046   -0.599703881092
     H   -2.801310939284    0.213392687817    0.720077488100
     O    1.696583862212   -0.932610848287   -1.619638096114
     H    1.352593961402   -1.727485618788   -1.195490098756
     '''

    qcinput_Orca_doublet = {
    'qchem': 'Psi4',
    'path': '',
    'nproc': 4,
    'functional': 'r2scan-3c',
    'basis': '',
    'charge': 0,
    'multiplicity': 2,
    'additional': '',
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
    reaction.sample_and_run_collision(integrator='verlet', integrator_order=4, timestep=0.5, maxstep=100, iprint=1, pairs_to_stop=pairs_to_test)

