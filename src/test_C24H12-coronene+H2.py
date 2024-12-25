if __name__ == '__main__':
    import random
    from smite import Molecule, Fragment, Collision

    '''
    https://doi.org/10.1002/ntls.20240010
    '''
    xyz_C24H12 = '''
C           -0.67015605342988        1.20280625722957        0.12958788457530
C           -2.05556959928655        1.22571879436343        0.42508076835987
C           -2.72471988926605        0.03050974095212        0.76217404276924
C           -1.98823218572723       -1.17644722491012        0.79760810764023
C           -0.65562130347892       -1.19848651426389        0.51337752977709
C            0.04012639037815       -0.01521641751081        0.17246422173030
C            1.42240574783411       -0.00924035331086       -0.12700672993872
C            2.06884321761705        1.14539998594253       -0.45265811822155
C            1.38132815852719        2.38038378292295       -0.50318414539220
C            0.00189545218119        2.40319737101209       -0.20896710182370
C           -4.14836427108844        2.47183621279330        0.67623584742479
C           -4.79415590923015        1.25890317960483        1.01114114109290
C           -4.10798252802979        0.08222358137924        1.05256192159883
C           -2.76893160201667        2.44902252621438        0.38201867130047
C           -4.18944191067129        4.86146031394170        0.30005808798425
C           -4.83587938940471        3.70681999534348        0.62570944890090
C           -2.09688004061746        3.64941370021474        0.04346366330085
C           -2.80716247553873        4.86743642187337        0.00058748477137
C           -2.11141473120820        6.05070656791103       -0.34032556801751
C           -0.71146658668010        3.62650111155709       -0.25202927377198
C           -0.04231626889553        4.82171022427333       -0.58912250033694
C           -0.77880385830853        6.02866726234928       -0.62455613206181
C            2.02711981778124        3.59331684620148       -0.83808930488840
C            1.34094641153108        4.76999641485403       -0.87951023506218
H           -2.50737321190493       -2.08853547837653        1.05668630757502
H           -0.10508667964803       -2.12826544884904        0.54430616153201
H            1.96043456059439       -0.94617635092573       -0.09222413429310
H            3.12576188360224        1.13528414475124       -0.67927330368548
H           -5.85132565361407        1.28375098519886        1.23543696427168
H           -4.61436656426152       -0.83743949063839        1.31010589434668
H           -4.72747071064571        5.79839628024487        0.26527519967460
H           -5.89279814762692        3.71693584641998        0.85232423826587
H           -2.66194932277717        6.98048552372761       -0.37125402918787
H           -0.25966274344234        6.94075557339958       -0.88363393198954
H            3.08428959308137        3.56846910365721       -1.06238499267359
H            1.84733040367093        5.68965953045204       -1.13705408554768
    '''

    qcinput_singlet = {
    'qchem': 'XTB',
    'path': '/home/peter/orca_6_0_0/xtb',
    'nproc': 4,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 1,
    'additional': '',
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

    seed = 14579112
    random.seed(seed)


    fix_quantum = [(14, 0),
                   (15, 0)]

    fix_energy  = [(0, 0.0),
                   (1, 0.0),
                   (2, 0.0)]

    fix_temp    = [(5, 330.0),
                   (6, 430.0)]

    fix_quantum_C24H12 = [(0, 6)]

    C24H12  = Fragment.Polyatom_Init(fname='C24H12', qchem=qcinput_singlet, xyz=xyz_C24H12, surface=True, surf_3atom=[0, 13, 19])
    C24H12.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=0)


    #XTB opt
    req_H2 = 0.78 #Angstrom
    omega_H2 = 3755.53 #cm-1

    H2 = Fragment.Diatom_Init(fname='H2', atoms=['H','H'], req=req_H2, omega=omega_H2, random_rot=True, diatom='harmonic')
    #OH.Specify_Mode_Sampling(init_rot_type='Jfix', nvib=0, jrot=0)
    H2.Specify_Mode_Sampling(init_rot_type='Jfix', nvib=0, jrot=2.0)

    print("--------- PAH: C24H12 mode sampling---------")
    C24H12.print_mode_sampling()
    print("--------------------------------------------")


    pairs_to_test = {
    'Nonreactive': [((0, 37), 'GT', 14.0)],
    # add more channels and pairs as needed
    }

    print("\nReaction of C24H12 + H2\n")

    reaction = Collision(C24H12, H2, qchem=qcinput_singlet) 
    reaction.Specify_Collision_Sampling(Rini=10.0, bmax=3.0, bsampling=True, Ecoll_thermal=True, temp=1000.0, surf_skew_max=45.0, surf_skew_fix=True, surf_side=1)

    reaction.sample_and_run_collision(integrator='leapfrog', integrator_order=4, timestep=0.7, maxstep=10000, iprint=2, pairs_to_stop=pairs_to_test)


