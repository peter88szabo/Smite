from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == '__main__':
    import random
    from smite import Molecule, Fragment, Collision

    #ZZAllyl-peroxy+O2_Case1 M062X avtz optim Conformer 21
    xyz_zzallyl = '''
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

    qcinput_Orca_doublet = {
    'qchem': 'Orca',
    'path': '/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/orca',
    'nproc': 8,
    #'functional': 'HF-3c',
    #'functional': 'wB97X-3c',
    'functional': 'PBEh-3c',
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
    'functional': 'B3LYP',
    'basis': 'pc1',
    'charge': 0,
    'multiplicity': 2,
    'additional': '',
    'wfu': False
    }

    seed = 12979102
    random.seed(seed)


    zzallyl  = Fragment.Polyatom_Init(fname='ZZAllyl', qchem=qcinput_Orca_doublet, xyz=xyz_zzallyl, random_rot=True)
    zzallyl.Specify_Mode_Sampling(init_vib_type='Temp', init_rot_type='Temp', temp=300.0)

    req_O2 = 1.2 #Angstrom
    omega_O2 = 1580.0 #cm-1

    oxygen = Fragment.Diatom_Init(fname='O2', atoms=['O','O'], req=req_O2, omega=omega_O2, random_rot=True, diatom='harmonic')
    oxygen.Specify_Mode_Sampling(init_rot_type='Temp', nvib=0, temp=300.0)

    print("--------- ZZ-OH-Allyl Isoprenyl radical--------")
    zzallyl.print_mode_sampling()
    print("--------- ZZ-OH-Allyl Isoprenyl radical DONE--------")

    pairs_to_test = {
    'capture_gamma_1': [((2, 17), 'LT', 1.5)],  
    'capture_gamma_2': [((2, 18), 'LT', 1.5)],  
    'capture_alpha_1': [((6, 17), 'LT', 1.5)],  
    'capture_alpha_2': [((6, 18), 'LT', 1.5)],  
    'reaction_1': [((2, 18), 'GT', 8.0), ((2, 17), 'GT', 8.0)],  
    'reaction_2': [((4, 5), 'GT', 8.0)],  
    # add more channels and pairs as needed
    }

    print("\nReaction of ZZ-OH-allyl + O2")
    print()


    reaction =  Collision(zzallyl, oxygen, qchem=qcinput_Orca_doublet) 
    reaction.Specify_Collision_Sampling(Rini=7.1, bmax=7.0, bsampling=True, Ecoll_thermal=True, temp=300.0)

    reaction.sample_and_run_collision(integrator='leapfrog', integrator_order=4, timestep=0.5, maxstep=10000, iprint=1, pairs_to_stop=pairs_to_test)


   




