if __name__ == '__main__':
    import random
    from smite import Molecule, Fragment, Collision

    xyz_water = '''
      O    -0.011100  0.0000  -0.00788
      H     0.007500  0.0000   0.95111
      H     0.899200  0.0000  -0.30990
    '''

    qcinput_doublet = {
    'qchem': 'XTB',
    'path': '/home/peter/orca_6_0_0/xtb',
    'nproc': 4,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 2,
    'additional': '--acc 50 --iterations 1000 --spinpol --tblite',
    'wfu': False 
    }

    qcinput_singlet = {
    'qchem': 'XTB',
    'path': '/home/peter/orca_6_0_0/xtb',
    'nproc': 4,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 1,
    'additional': '--acc 50 --iterations 1000 --spinpol --tblite',
    'wfu': False
    }

    qcinput_singlet_plus = {
    'qchem': 'XTB',
    'path': '/home/peter/orca_6_0_0/xtb',
    'nproc': 4,
    'functional': '',
    'basis': '',
    'charge': 1,
    'multiplicity': 1,
    'additional': '--acc 50 --iterations 1000 --spinpol --tblite',
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

    fix_quantum_water = [(0, 6)]

    water  = Fragment.Polyatom_Init(fname=fname_water, qchem=qcinput_singlet, xyz=xyz_water, random_rot=random_rot)
    water.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=3)

    clorine = Fragment.Atom_Init(fname='Cl', atoms=['Cl'])

    print("--------- H2O mode sampling---------")
    water.print_mode_sampling()
    print("--------------------------------------")


    pairs_to_test = {
    'Habstr_1': [((0, 1), 'GT', 8.0)],
    'Habstr_2': [((0, 2), 'GT', 8.0)],
    'Nonreact': [((0, 1), 'LT', 2.5), ((0, 2), 'LT', 2.5), ((0, 3), 'GT', 8.0)]
    # add more channels and pairs as needed
    }

    print("\nReaction of H2O + Cl\n")

    reaction = Collision(water, clorine, qchem=qcinput_doublet) 
    reaction.Specify_Collision_Sampling(Rini=7.0, bmax=3.0, bsampling=True, Ecoll=20.0, temp=300.0)

    reaction.sample_and_run_collision(integrator='symplectic', integrator_order=4, timestep=0.5, maxstep=10000, iprint=4, pairs_to_stop=pairs_to_test)


