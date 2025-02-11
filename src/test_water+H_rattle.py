if __name__ == '__main__':
    import random
    from smite import Molecule, Fragment, Collision

    xyz_water = '''
      O    -0.011100  0.0000  -0.00788
      H     0.007500  0.0000   0.95111
      H     0.899200  0.0000  -0.30990
    '''

    qcinput = {
    'qchem': 'XTB',
    'path': '/home/jenne/repos/Smite/xtb-dist/bin/xtb',
    'nproc': 4,
    'functional': '',
    'basis': '',
    'charge': 1,
    'multiplicity': 1,
    'additional': '--acc 50 --iterations 1000 --spinpol --tblite',
    'wfu': False
    }

    seed = 12949103
    random.seed(seed)


    fix_quantum = [(14, 0),
                   (15, 0)]

    fix_energy  = [(0, 0.0),
                   (1, 0.0),
                   (2, 0.0)]

    fix_temp    = [(5, 330.0),
                   (6, 430.0)]

    fix_quantum_water = [(0, 6)]

    water  = Fragment.Polyatom_Init(fname="H2O", qchem=qcinput, xyz=xyz_water, random_rot=True, rigid=True)
    water.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=3, rigid=True)

    hydrogen = Fragment.Atom_Init(fname="H", atoms=['H'])


    pairs_to_test = {
    'Habstr_1': [((0, 1), 'GT', 8.0)],
    'Habstr_2': [((0, 2), 'GT', 8.0)],
    'Nonreact': [((0, 1), 'LT', 2.5), ((0, 2), 'LT', 2.5), ((0, 3), 'GT', 8.0)]
    # add more channels and pairs as needed
    }

    print("\nReaction of H2O + H\n")

    constrained_bonds = [(0,1),(0,2),(1,2)]

    reaction = Collision(water, hydrogen, qchem=qcinput)
    reaction.Specify_Collision_Sampling(Rini=7.0, bmax=3.0, bsampling=True, Ecoll=20.0, temp=300.0)

    reaction.sample_and_run_collision(integrator='rattle', timestep=0.5, maxstep=10000, iprint=1, pairs_to_stop=pairs_to_test, constrained_bonds=constrained_bonds)


