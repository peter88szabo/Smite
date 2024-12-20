if __name__ == '__main__':
    import random
    from smite import Molecule, Fragment, Collision

    '''
    R. I. Kaiser et al 2001 ApJ 561 858
    DOI 10.1086/323261
    '''
    xyz_propene = '''
    C           -0.02068925344180       -0.00000000767811        0.02975680595526
    C            0.02012144314018        0.00000005041067        1.34903574412311
    H            0.90067707918524       -0.00000000693691       -0.53739143264635
    H            0.94880080487519       -0.00000004109096        1.89366505323869
    H           -0.88026648726973        0.00000002912193        1.94238206905807
    C           -1.28921825180245       -0.00000000831580       -0.76728356929627
    H           -1.08253669200043        0.00000002688200       -1.83378057121858
    H           -1.88829331125620       -0.87855356796066       -0.53094257149026
    H           -1.88829333143001        0.87855352556784       -0.53094252772368
    '''

    qcinput_singlet = {
    'qchem': 'XTB',
    'path': '/home/peter/orca_6_0_0/xtb',
    'nproc': 4,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 1,
    'additional': '--gfnff --acc 50 --iterations 300 --spinpol --tblite',
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
    'additional': '--gfnff --acc 50 --iterations 300 --spinpol --tblite',
    'wfu': False
    }


    qcinput_Orca = {
    'qchem': 'Orca',
    'path': '/home/peter/orca_6_0_0/orca',
    'nproc': 4,
    'functional': 'HF-3c',
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

    seed = 12949102
    random.seed(seed)


    fix_quantum = [(14, 0),
                   (15, 0)]

    fix_energy  = [(0, 0.0),
                   (1, 0.0),
                   (2, 0.0)]

    fix_temp    = [(5, 330.0),
                   (6, 430.0)]

    fix_quantum_propene = [(0, 6)]

    propene  = Fragment.Polyatom_Init(fname='propene', qchem=qcinput_singlet, xyz=xyz_propene, random_rot=True)
    propene.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=3)

    carbon = Fragment.Atom_Init(fname='C', atoms=['C'])

    print("--------- Ethene mode sampling---------")
    propene.print_mode_sampling()
    print("--------------------------------------")


    pairs_to_test = {
    'Habstr_1': [((0, 1), 'GT', 7.5)],
    'Habstr_2': [((0, 2), 'GT', 7.5)],
    'Nonreact': [((0, 1), 'LT', 2.5), ((0, 2), 'LT', 2.5), ((0, 3), 'GT', 8.0)]
    # add more channels and pairs as needed
    }

    print("\nReaction of propene + C(3P)\n")

    reaction = Collision(propene, carbon, qchem=qcinput_triplet) 
    reaction.Specify_Collision_Sampling(Rini=6.0, bmax=3.0, bsampling=True, Ecoll=20.0, temp=300.0)

    reaction.sample_and_run_collision(integrator='leapfrog', integrator_order=4, timestep=0.7, maxstep=10000, iprint=2, pairs_to_stop=pairs_to_test)


