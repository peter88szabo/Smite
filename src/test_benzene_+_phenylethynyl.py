if __name__ == '__main__':
    import random
    from smite import Molecule, Fragment, Collision

    #XTB opt
    xyz_phenylethynyl = '''
    C            0.00846094144860        0.00000010123263        0.00575080471213
    C            0.00020829471033        0.00000004868907        1.39643854840350
    C            1.20261709640749       -0.00000002564473        2.08039127475346
    C            2.40127432774593       -0.00000002694817        1.38624195630578
    C            2.40241926989333       -0.00000000043977        0.00100593863455
    C            1.20845356463297        0.00000002767754       -0.69730488233994
    H           -0.93857487064063        0.00000004290333        1.92958492175492
    H            1.20034177487815       -0.00000002568821        3.16055952730583
    H            3.33674196706751       -0.00000005373687        1.92593936679697
    H            3.33638735247113       -0.00000000803387       -0.54161826032785
    H            1.19958685121661        0.00000002302902       -1.77688862455272
    C           -1.22883576972214       -0.00000010085011       -0.70818578523304
    C           -2.26453080010931       -0.00000000218988       -1.31191478621358
     '''

    #XTB opt
    xyz_benzene = '''
    C            0.01468162579394       -0.00000000000306        0.00846038034826
    C            0.01472600581974        0.00000000000245        1.39308559033407
    C            1.21380050763925        0.00000000000030        2.08539800813318
    C            2.41294500703128       -0.00000000000277        1.39311651570462
    C            2.41294777552013        0.00000000000178        0.00849048210558
    C            1.21383664999024        0.00000000000115       -0.68377576126132
    H           -0.92099473058520        0.00000000000279        1.93331655448251
    H            1.21378310292763        0.00000000000140        3.16585529784655
    H            3.34864396174925       -0.00000000000401        1.93335866013916
    H            3.34860933395883        0.00000000000293       -0.53178716413194
    H            1.21379325199189        0.00000000000116       -1.76425468808933
    H           -0.92108849183697       -0.00000000000412       -0.53176587561133
    '''


    qcinput_singlet = {
    'qchem': 'XTB',
    'path': '/home/peter/orca_6_0_0/xtb',
    'nproc': 4,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 1,
    'additional': '--gfnff --acc 50 --iterations 1000 --spinpol --tblite',
    'wfu': False
    }

    qcinput_doublet = {
    'qchem': 'XTB',
    'path': '/home/peter/orca_6_0_0/xtb',
    'nproc': 4,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 2,
    'additional': '--gfnff --acc 50 --iterations 1000 --spinpol --tblite',
    'wfu': False
    }

    qcinput_Orca_doublet = {
    'qchem': 'Orca',
    'path': '/home/peter/orca_6_0_0/orca',
    'nproc': 4,
    #'functional': 'HF-3c',
     'functional': 'wB97X-3c',
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

    seed = 15029222
    random.seed(seed)


    benzene  = Fragment.Polyatom_Init(fname='benzene', qchem=qcinput_singlet, xyz=xyz_benzene, random_rot=True)
    #zzallyl.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=0, fix_quantum=fix_quantum)
    #zzallyl.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Temp', temp=300.0)
    benzene.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=0)

    print("----------------------- benzene --------------------")
    benzene.print_mode_sampling()
    print("----------------------------------------------------\n")

    phenylethynyl  = Fragment.Polyatom_Init(fname='phenylethynyl', qchem=qcinput_doublet, xyz=xyz_phenylethynyl, random_rot=True)
    phenylethynyl.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=10)
    print("----------------------- phenylethynyl --------------------")
    phenylethynyl.print_mode_sampling()
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

    print("\nReaction of benzene + phenyl-ethynyl radical\n")

    reaction =  Collision(benzene, phenylethynyl, qchem=qcinput_doublet) 
    reaction.Specify_Collision_Sampling(Rini=8.0, bmax=5.0, bsampling=True, Ecoll_thermal=True, temp=300.0)

    #reaction.sample_and_run_collision(integrator='leapfrog', integrator_order=4, timestep=0.7, maxstep=10000, iprint=4, pairs_to_stop=pairs_to_test)
    reaction.sample_and_run_collision(integrator='leapfrog', integrator_order=4, timestep=1.0, maxstep=10000, iprint=4, Rstop=14.0)


   




