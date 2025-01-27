if __name__ == '__main__':
    import random
    from smite import Molecule, Fragment, Collision


    #See Lam's recent publication:
    #Environ. Sci.: Atmos., 2023,3, 1678-1684, https://doi.org/10.1039/D3EA00143A
    #HO2 XTB opt
    xyz_HO2 = '''
    O            0.07025861839762        0.21828722337439       -0.00806467583261
    O            0.15206618560286        0.17083416697104        1.29066285487068
    H            0.85644321593421       -0.23611812087724       -0.38682093982377
     '''

    #Reg: 
    #Klippensten et al, Combustion and Flame, Vol. 243, 2022, pp-111975
    #https://doi.org/10.1016/j.combustflame.2021.111975

    qcinput_doublet = {
    'qchem': 'XTB',
    'path': '/home/peter/orca_6_0_0/xtb',
    'nproc': 4,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 2,
    'additional': '--gffn --iterations 1000 --spinpol --tblite',
    'wfu': False 
    }

    qcinput_triplet = {
    'qchem': 'XTB',
    'path': '/home/peter/orca_6_0_0/xtb',
    'nproc': 4,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 3,
    'additional': '--iterations 1000 --spinpol --tblite',
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
    'multiplicity': 2,
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

    seed = 17229222
    random.seed(seed)


    HO2A  = Fragment.Polyatom_Init(fname='HO2A', qchem=qcinput_doublet, xyz=xyz_HO2, random_rot=True)
    #zzallyl.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=0, fix_quantum=fix_quantum)
    #zzallyl.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Temp', temp=300.0)
    HO2A.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=0)

    print("----------------------- HO2-A --------------------")
    HO2A.print_mode_sampling()
    print("-------------------------------------------------\n")

    HO2B  = Fragment.Polyatom_Init(fname='HO2B', qchem=qcinput_doublet, xyz=xyz_HO2, random_rot=True)
    #zzallyl.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=0, fix_quantum=fix_quantum)
    #zzallyl.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Temp', temp=300.0)
    HO2B.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=0)

    print("----------------------- HO2-B --------------------")
    HO2B.print_mode_sampling()
    print("-------------------------------------------------\n")


    pairs_to_test = {
    'reaction_C1-H1': [((0, 2), 'GT', 6.0), ((4, 2), 'LT', 1.5)],  
    'reaction_C1-H2': [((1, 3), 'GT', 6.0), ((4, 3), 'LT', 1.5)],  
    # add more channels and pairs as needed
    }

    print("\nReaction of HO2 + HO2\n")

    reaction =  Collision(HO2A, HO2B, qchem=qcinput_triplet) 
    reaction.Specify_Collision_Sampling(Rini=5.2, bmax=5.0, bsampling=True, Ecoll_thermal=True, temp=300.0)

    reaction.sample_and_run_collision(integrator='leapfrog', integrator_order=4, timestep=0.7, maxstep=10000, iprint=4, pairs_to_stop=pairs_to_test)


   




