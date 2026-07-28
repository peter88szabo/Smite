from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == '__main__':
    import random
    from smite import Molecule, Fragment, Collision

    #XTB opt
    xyz_CH3COOH = '''
    C            0.00700781793894       -0.00000078610485        0.24261918108015
    O            0.07594452838859        0.00000132887781        1.43599490113932
    O            1.12505835596856       -0.00000042392672       -0.50611261212202
    C           -1.30193648061866       -0.00000011021804       -0.51761725732002
    H            0.94701006550162        0.00000000929729       -1.45495299221528
    H           -1.16686766525846        0.00000078443204       -1.59667834619087
    H           -1.87485831067234       -0.87674152671934       -0.22345511612350
    H           -1.87485831124826        0.87674072436181       -0.22345375824779
     '''

    #XTB opt
    xyz_H2CO = '''
    C            0.00000024685549        0.00000000279304        0.16750577050769
    O           -0.00000105411306       -0.00000000031030        1.35957352167911
    H            0.92650723091444       -0.00000000124137       -0.43020363175959
    H           -0.92650542365688       -0.00000000124137       -0.43020566042721
    '''

    #XTB opt protonated NH2-OH
    xyz_NH2OH2p = '''
    N            0.00785673124647       -0.05894218841253       -0.07270805199625
    H           -0.08940321476448        0.18748208713487        0.91444326956006
    H            1.00381766287360       -0.16936222298766       -0.27478616608802
    O           -0.61673554662125       -1.37861438400971       -0.24918877408073
    H           -1.58536689966567       -1.25309729815698       -0.05939913308404
    H           -0.51778973306867       -1.60590799356799       -1.21280414431101
     '''
     
    #XTB opt protonated NH2-OH
    xyz_NH3OHp = '''
    N            0.03161048256318       -0.00844063454729        0.02231995540351
    H            0.01922293423543       -0.06013077447815        1.05965286829739
    H            1.00547937160736       -0.06021180227194       -0.33511245587379
    H           -0.50516192821981       -0.81351388630543       -0.35715903254268
    O           -0.64315920903544        1.11252852046792       -0.45487317432386
    H           -0.23797365115072        1.95265957713489       -0.16815816096058
     '''

    #XTB opt (non-protonated) HN=CH-CH=C=O
    #See Fig 7 in PCCP, 2016, 18 (22), pp.14980-1499
    xyz_preuracil = '''
    C            0.15374311553823        0.00060119474978       -0.09080883034258
    O            0.33484761994069       -0.00000423331199        1.05323154344058
    C            0.02758608630975       -0.00012536859168       -1.38589023771524
    H           -0.95065050157436       -0.00051687560896       -1.83699999732908
    C            1.19039627526772       -0.00060544942459       -2.27108851144749
    H            2.15843208053511        0.00006856988395       -1.78252576119791
    N            1.19145321332042        0.00046002879353       -3.52649502408131
    H            0.27786311066246        0.00012213350996       -3.97557918132698
    '''

    #XTB opt (non-protonated) HN=CH-CH=C=O
    xyz_CONH = '''
    C           -0.07971861286770        0.00000000007783        0.00909255795711
    O            0.08803235558551       -0.00000000000217        1.15094863406913
    N           -0.03260252922933       -0.00000000000080       -1.18068619109313
    H           -0.68608721348847       -0.00000000007485       -1.92918000093311
    '''



    qcinput_neutral = {
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

    qcinput_positive = {
    'qchem': 'XTB',
    'path': '/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/xtb',
    'nproc': 4,
    'functional': '',
    'basis': '',
    'charge': 1,
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

    seed = 12529222
    random.seed(seed)


    H2CO  = Fragment.Polyatom_Init(fname='H2CO', qchem=qcinput_neutral, xyz=xyz_H2CO, random_rot=True)
    #zzallyl.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=0, fix_quantum=fix_quantum)
    #zzallyl.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Temp', temp=300.0)
    H2CO.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=10)

    print("----------------------- H2CO --------------------")
    H2CO.print_mode_sampling()
    print("----------------------------------------------------\n")

    NH2OH2p  = Fragment.Polyatom_Init(fname='NH2OH2p', qchem=qcinput_positive, xyz=xyz_NH2OH2p, random_rot=True)
    NH2OH2p.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=10)
    print("----------------------- NH2OH2p --------------------")
    NH2OH2p.print_mode_sampling()
    print("----------------------------------------------------")

    pairs_to_test = {
    'reaction_C1-H1': [((0, 4), 'GT', 7.0), ((10, 4), 'LT', 2.0)],  
    'reaction_C1-H2': [((0, 5), 'GT', 7.0), ((10, 5), 'LT', 2.0)],  
    'reaction_C1-H3': [((0, 6), 'GT', 7.0), ((10, 6), 'LT', 2.0)],  
    'reaction_C2-H1': [((2, 7), 'GT', 7.0), ((10, 7), 'LT', 2.0)],  
    'reaction_C2-H2': [((2, 8), 'GT', 7.0), ((10, 8), 'LT', 2.0)],  
    'reaction_C2-H3': [((2, 9), 'GT', 7.0), ((10, 9), 'LT', 2.0)]  
    # add more channels and pairs as needed
    }

    print("\nReaction of H2CO + H2N-OH2+\n")

    reaction =  Collision(H2CO, NH2OH2p, qchem=qcinput_positive) 
    reaction.Specify_Collision_Sampling(Rini=8.0, bmax=4.0, bsampling=True, Ecoll_thermal=True, temp=300.0)

    #reaction.sample_and_run_collision(integrator='leapfrog', integrator_order=4, timestep=0.7, maxstep=10000, iprint=4, pairs_to_stop=pairs_to_test)
    reaction.sample_and_run_collision(integrator='leapfrog', integrator_order=4, timestep=0.7, maxstep=10000, iprint=4, Rstop=15.0)


   




