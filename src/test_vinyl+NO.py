if __name__ == '__main__':
    import random
    from smite import Molecule, Fragment, Collision

    #vinyl radical optimized by XTB-spinpol
    xyz_vinyl = '''
    C           -0.06765881401168        0.36069437750287        0.21726087063540
    C           -0.02989301947210        0.05415854223373        1.47236819314274
    H            0.88885859725297        0.01810805651872        2.06187426829113
    H           -0.92319008305886       -0.19134188210751        2.02827103745251
    H            0.58952031928966        0.63203190585219       -0.57444136952178
    '''


    qcinput_vinyl = {
    'qchem': 'XTB',
    'path': '/home/peter/orca_6_0_0/xtb',
    'nproc': 8,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 2,
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
    'multiplicity': 2,
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




    seed = 12222222
    random.seed(seed)

    fix_quantum = [(14, 0),
                   (15, 0)]

    fix_energy  = [(0, 0.0),
                   (1, 0.0),
                   (2, 0.0)]

    fix_temp    = [(5, 330.0),
                   (6, 430.0)]

    fix_quantum_water = [(0, 6)]

    vinyl = Fragment.Polyatom_Init(fname='vinyl', qchem=qcinput_vinyl, xyz=xyz_vinyl, random_rot=True)
    vinyl.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=0)

    req_NO = 1.1557 #Angstrom for ground state as Pi-doublet from XTB spinpol
    omega_NO = 1948.36 #cm-1

    NO = Fragment.Diatom_Init(fname='NO', atoms=['N','O'], req=req_NO, omega=omega_NO, random_rot=True, diatom='harmonic')
    NO.Specify_Mode_Sampling(init_rot_type='Jfix', nvib=0, jrot=30)

    print("--------- Vinyl radical--------")
    vinyl.print_mode_sampling()
    print("--------- Vinyl radical DONE--------")


    pairs_to_test = {
    'capture_ON-C1': [((0, 5), 'LT', 1.5)],  
    'capture_ON-C2': [((1, 5), 'LT', 1.5)],  
    'capture_NO-C1': [((0, 6), 'LT', 1.5)],  
    'capture_NO-C2': [((1, 6), 'LT', 1.5)],  
    'HCCH + HNO_1': [((0, 5), 'GT', 9.0), ((0, 6), 'GT', 9.0), ((5, 2), 'LT', 1.8)],  
    'HCCH + HNO_2': [((0, 5), 'GT', 9.0), ((0, 6), 'GT', 9.0), ((5, 3), 'LT', 1.8)],  
    'HCCH + HON_1': [((0, 5), 'GT', 9.0), ((0, 6), 'GT', 9.0), ((6, 2), 'LT', 1.8)],  
    'HCCH + HON_2': [((0, 5), 'GT', 9.0), ((0, 6), 'GT', 9.0), ((6, 3), 'LT', 1.8)],  
    # add more channels and pairs as needed
    }

    qcinput_vinyl_singlet = {
    'qchem': 'XTB',
    'path': '/home/peter/orca_6_0_0/xtb',
    'nproc': 4,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 1,
    'additional': '--acc 100 --iterations 1000 --spinpol --tblite',
    'wfu': False
    }

    print("\nReaction of Vinyil + NO")
    print()

    reaction =  Collision(vinyl, NO, qchem=qcinput_vinyl_singlet)
    #reaction =  Collision(vinyl, NO, qchem=qcinput_Orca)
    reaction.Specify_Collision_Sampling(Rini=7.0, bmax=4.0, bsampling=True, Ecoll_thermal=True, temp=300.0)

    reaction.sample_and_run_collision(integrator='symplectic', integrator_order=4, timestep=1.0, maxstep=10000, pairs_to_stop=pairs_to_test, iprint=1)


   




