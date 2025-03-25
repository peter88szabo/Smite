if __name__ == '__main__':
    import random
    from smite import Molecule, Fragment, Collision

    #ZZAllyl-peroxy+O2_Case1 M062X avtz optim Conformer 21
    xyz_acetone = '''
     C     0.000000     0.000000     0.000000
     C    -0.707107     1.224745    -0.500000
     C    -0.844154     1.462117    -1.974745
     O    -1.170755     2.027808     0.292792
     H     0.000000     0.000000     1.070000
     H     1.008807     0.000000    -0.356663
     H    -0.504403    -0.873651    -0.356667
     H    -1.372651     2.377501    -2.141068
     H     0.127819     1.525914    -2.417576
     H    -1.385390     0.652263    -2.417576
     '''

    xtb = "/home/jenne/xtb-dist/bin/xtb"

    qcinput_doublet = {
    'qchem': 'XTB',
    'path': xtb,
    'nproc': 4,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 2,
    'additional': '--gfnff --acc 50 --iterations 1000 --spinpol --tblite',
    'wfu': False 
    }

    qcinput_singlet = {
    'qchem': 'XTB',
    'path': xtb,
    'nproc': 4,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 1,
    'additional': '--gfnff --acc 50 --iterations 1000',
    'wfu': False
    }

    qcinput_Orca_doublet = {
    'qchem': 'Orca',
    'path': '/home/peter/orca_6_0_0/orca',
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

    seed = 12229222
    random.seed(seed)


    acetone  = Fragment.Polyatom_Init(fname='acetone', qchem=qcinput_singlet, xyz=xyz_acetone, random_rot=True)
    #zzallyl.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=0, fix_quantum=fix_quantum)
    #zzallyl.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Temp', temp=300.0)
    acetone.Specify_Mode_Sampling(init_vib_type='Temp', init_rot_type='Temp', temp=300.0)

    req_OH = 0.96 #Angstrom
    omega_OH = 3808.2 #cm-1

    OH = Fragment.Diatom_Init(fname='OH', atoms=['O','H'], req=req_OH, omega=omega_OH, random_rot=True, diatom='harmonic')
    #OH.Specify_Mode_Sampling(init_rot_type='Jfix', nvib=0, jrot=0)
    OH.Specify_Mode_Sampling(init_rot_type='Temp', nvib=0, temp=300.0)

    print("----------------------- Acetone --------------------")
    acetone.print_mode_sampling()
    print("-------------Acetone Sampling is done---------------")

    pairs_to_test = {
    'reaction_C1-H1': [((0, 4), 'GT', 7.0), ((10, 4), 'LT', 2.0)],  
    'reaction_C1-H2': [((0, 5), 'GT', 7.0), ((10, 5), 'LT', 2.0)],  
    'reaction_C1-H3': [((0, 6), 'GT', 7.0), ((10, 6), 'LT', 2.0)],  
    'reaction_C2-H1': [((2, 7), 'GT', 7.0), ((10, 7), 'LT', 2.0)],  
    'reaction_C2-H2': [((2, 8), 'GT', 7.0), ((10, 8), 'LT', 2.0)],  
    'reaction_C2-H3': [((2, 9), 'GT', 7.0), ((10, 9), 'LT', 2.0)]  
    # add more channels and pairs as needed
    }

    print("\nReaction of Acetone + OH\n")

    reaction =  Collision(acetone, OH, qchem=qcinput_doublet) 
    reaction.Specify_Collision_Sampling(Rini=6.0, bmax=4.0, bsampling=True, Ecoll_thermal=True, temp=300.0)

    reaction.sample_and_run_collision(integrator='sprk', integrator_order=4, timestep=0.7, maxstep=10000, iprint=4, pairs_to_stop=pairs_to_test)


   




