import numpy as np
import random
import math
from normalmode     import getNormalmode, print_frequencies
from nmodeprint     import print_normalmode
from rotation_axis  import rotate_fragment, locate_frags_to_rotate
from atomic_masses  import get_mass_vector 
from eckart         import eckart_transform
#from atomic_overlap import check_atomic_overlap

#     [Anstrom]*c1=[bohr]
c1=1.0e0/0.5291772e0
#     [Hartree]*c4=[kcal/mol]
c4= 627.51
#     [Hartree]*c5=[cm-1]
c5=219474.e0
#     [frequency in cm-1]*c9=[freq(bohr^(-1))]
c7 = 2625.5         # [Hartree] * c7 = [kJ/mol] 

c9=1.0e8*c1
#     [speed of light in atomic unit]
c10=137.035999074

Rgas = 8.3144598/1000.0/c7 #Hartree/K 


#-------------------------------------------------------
def thermal_vibr_mode(temp, ome):
#-------------------------------------------------------
    RT = Rgas * temp
    rand = random.uniform(0,1)
    vib = -RT*np.log(1.0-rand)/abs(ome)
    nvib = int(vib) #harmonic oscillator quantum number 
    return nvib
#-------------------------------------------------------

#------------------------------------------------------------------------------------------
def initialize_vibrational_modes(freq, init_type='ZPE', temp=300.0, nvib=0, ene=0.0):
#------------------------------------------------------------------------------------------
    """
    Initializes a dictionary with a given number of modes, each set to ('Q', nvib) or ('T', temp).
    each mode is initialized as a quantized mode with ZPE energy or according to a temperature

    freq: Frequencies of vibrational modes.

    """
    freq = np.array(freq) /c10*c9/(math.pi * 2) #converting to cm-1
    freq = np.round(freq, 3)

    nmodes = len(freq)

    if init_type == 'ZPE':
        return {i: (freq[i], 'Q', 0) for i in range(nmodes)}
    elif init_type == 'Qvib':
        return {i: (freq[i], 'Q', nvib) for i in range(nmodes)}
    elif init_type == 'Temp':
        return {i: (freq[i], 'T', temp) for i in range(nmodes)}
    elif init_type == 'Ene':
        return {i: (freq[i], 'E', ene) for i in range(nmodes)}
    else:
        raise ValueError("init_type must be 'ZPE', 'Ene' or 'Temp'")
#-------------------------------------------------------------------------------------------


def specify_modes(vib_modes, **kwargs):
    """
    Sets the mode type and value for a given mode index.

    Modes how to sample: these could be signed as with Q, T, E, R
    to give quantum number (Q), Temperature (T), energy (E), or rotor (R)

    {1: ('Q', 0),
     2: ('Q', 3),
     3: ('T', 500.0),
     4: ('T', 300.0),
     5: ('E', 30.0),
     6: ('E', 11.0),
     8: ('R', (1,2) ),
     9: ('R', (2,3) )}

    Parameters:
    vib_modes (dict): The dictionary of vibrational (and/or rotor) modes.

    kwargs:
    fix_quantum = [(1, 0),
                   (2, 0)]

    fix_energy  = [(3, 600.0),
                   (4, 400.0)]

    fix_temp    = [(5, 300.0),
                   (6, 400.0)]

    rotor_list   = [(   1, (1, 2)),
                    (   2, (4, 5)),
                    (None, (7, 9)),
       # Add more pairs as needed]

       'None' is used for rotors which cannot assigned to a vibration
        Such rotors will be appended to vib_modes with a negative integer key 
    """

    if 'fix_quantum' in kwargs:
        for mode_index, qnumber in kwargs['fix_quantum']:
            if mode_index in vib_modes:
                frequency = vib_modes[mode_index][0]
                vib_modes[mode_index] = (frequency, 'Q', qnumber)

    if 'fix_energy' in kwargs:
        for mode_index, energy in kwargs['fix_energy']:
            if mode_index in vib_modes:
                frequency = vib_modes[mode_index][0]
                vib_modes[mode_index] = (frequency, 'E', energy)

    if 'fix_temp' in kwargs:
        for mode_index, tempr in kwargs['fix_temp']:
            if mode_index in vib_modes:
                frequency = vib_modes[mode_index][0]
                vib_modes[mode_index] = (frequency, 'T', tempr)

    if 'rotor_list' in kwargs:
        for mode_index, (atom1, atom2) in kwargs['rotor_list']:
            if mode_index is not None and mode_index in vib_modes:
                frequency = vib_modes[mode_index][0]
                vib_modes[mode_index] = (frequency, 'R', (atom1, atom2))
            else:
                new_index = min(vib_modes.keys()) - 1 
                frequency = None  
                vib_modes[new_index] = (frequency, 'R', (atom1, atom2))

    return vib_modes

#--------------------------------------------------------------------------------------------------------------------------------------------------
def init_vib_rot_modes(freq, init_type='ZPE', temp=300.0, **kwargs):
#--------------------------------------------------------------------------------------------------------------------------------------------------
    
    #initialize all vibrational modes the same way (either with ZPE quantum state 'nvib = 0' or according to a temperature)
    vib_modes = initialize_vibrational_modes(freq=freq, init_type=init_type, temp=temp)

    #then if any specific mode is fixed (energy, quantum number or temperature or chosen as a rotor), everything in kwargs:
    if kwargs:
        vib_modes = specify_modes(vib_modes, **kwargs)

    return vib_modes
#--------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------
def polyatom_vib_rot_sampling(mass, atoms, q_eq, ww, L, vib_modes, **kwargs):
    '''
     q_eq: equilbiriom coords
     ww: freq
     L: eigenvector of Hessians (Normal mode <--> Cartesian space transformator)
    '''
#-------------------------------------------------------------------------------------------------------------------------------------------------
    #phase_sampling together with temperature, and other fixed and rotor_list fetched as kwargs

    temp = kwargs.get('temp', 300.0)
    phase_sampling = kwargs.get('phase_sampling', 'linear')
    verbosity  = kwargs.get('verbosity', False)
    traj_index =  kwargs.get('traj_index', -999)
    bond_th_HX = kwargs.get('bond_th_HX', 1.4/0.5291772) # bond threshold in Angstrom for H-X, where X = any non H-atom
    bond_th_XX = kwargs.get('bond_th_XX', 2.0/0.5291772) # bond threshold in Angstrom for X-X bonds to be considered as a part of a fragment

    # ww: frequencies (in atomic units)
    # L : transfomration matrix (normal mode --> Cartesian), consist of the eigenvectors of the hessian

    #ww, ww_low, L = getNormalmode(mass, hessian)
    #ww_all = np.append(ww_low, ww)

    '''
    # for all modes that are not rotors (or even rotors but with 0.0 energy or ZPE) we do normal mode sampling
    # then after this we sample one-by-one the internal rotors too
    # then discard the geoms where there is some overlap between atoms --> this happens not here

    '''
    #------------------------------------------------------------------------------------------
    energy = []
    nvib = []
    #the main loop running over the normal modes
    for imode in range(len(ww)):

        #In case of imaginary freqs (when ww is imaginary, it has a negative sign):
        ww[imode] = abs(ww[imode])

        freq = vib_modes[imode][0] 
        sampling_mode = vib_modes[imode][1] # it must be 'Q', 'E', 'T', or 'R' 
        excitation = vib_modes[imode][2]  #either quantum number, energy or temperature value

        if sampling_mode == 'Q':
            nv = excitation 
            nvib += [nv]
            energy += [ww[imode]*(nv + 0.5)]
        elif sampling_mode == 'T':
            nv = thermal_vibr_mode(excitation, ww[imode])
            nvib += [nv]
            energy += [ww[imode]*(nv + 0.5)]
        elif sampling_mode == 'E':
            energy += [excitation]
            nv = excitation/ww[imode] - 0.5 #non-integer quantum number
            nvib += [nv]
        elif sampling_mode == 'R':
            #make sure that the corresponding torsional modes (rotors) has only ZPE energy
            nv = 0 
            nvib += [nv]
            energy += [ww[imode]*(nv + 0.5)]
        else:
            raise ValueError("sampling_mode must be 'Q', 'E', 'T' or 'R' ") 
    #------------------------------------------------------------------------------------------

    Evib = sum(np.array(energy))
    Ezero = 0.5*sum(np.array(ww))

    if verbosity == True:
        output_file = "sampled_mode_energies.txt" 
        with open(output_file, "a") as f:  # Open the file in append mode
            f.write(f"{'traj index:':15} {traj_index} {'      Ezero':15} {'       Evib':15} {'       Eexc':15}\n")
            f.write(f"{'kcal/mol -->':15} {Ezero*c4:15.3f} {Evib*c4:15.3f} {(Evib-Ezero)*c4:15.3f}\n")
            f.write(f"{'kJ/mol   -->':15} {Ezero*c7:15.3f} {Evib*c7:15.3f} {(Evib-Ezero)*c7:15.3f}\n")
            f.write(f"{'cm-1     -->':15} {Ezero*c5:15.3f} {Evib*c5:15.3f} {(Evib-Ezero)*c5:15.3f}\n\n")


    #energy to amolitude
    #Amplitude of nornam modes (they are already chose accordingly the input, Q, E, T, R)
    ampl = [math.sqrt(2.0 * energy[i])/ww[i] for i in range(len(ww))]

    if phase_sampling == 'cosine':
        #normal mode coords which is a 1D harmonic oscillator
        q_norm = [a * math.cos(random.uniform(0, 2 * math.pi)) for a in ampl]
    elif phase_sampling == 'linear':
        q_norm = [a * random.uniform(-1, 1) for a in ampl]
    else:
        raise ValueError("Your given {sampling} method is not available. Options to choose: cosine or linear\n")

    #Normal mode to Cartesian transformation:
    q_cart_vib = np.transpose(q_eq) + np.matmul(L, np.transpose(np.array(q_norm)))


    q_temp_rovib = q_cart_vib #temporary variable that changes iteratively for each rotor

    #----------------------------- Rotor sampling ----------------------------------------------------

    for mode_index, (freq, sampling, value) in vib_modes.items():

        if sampling == 'R':
            atom1 = value[0]
            atom2 = value[1]

            #use the equilbrium structure to locate the fragment to rotate:
            ind_rot_center, rot_frag_index, spectator_frag_index = locate_frags_to_rotate(atoms, q_eq, atom_ax1=atom1, atom_ax2=atom2,
                                                                   bond_th_HX=bond_th_HX, bond_th_XX=bond_th_XX, verbosity=verbosity, traj_index=traj_index)

            theta = (random.uniform(0.0, 360.0)) #the function eats the angle in degree

            #here we rewrite the same temporary q_temp_rovib vector after sampling each rotor
            q_temp_rovib = rotate_fragment(atoms = atoms, q=q_temp_rovib, atom_ax1=atom1, atom_ax2=atom2,
                                           auto_frag=False, rot_angle=theta, frag_index=rot_frag_index, atom_rotcenter=ind_rot_center, **kwargs)

    # we need something to reject the overlapping configurations
    # if check_atomic_overlap(atoms, q_temp_rovib):
    #     q_cart_samp = q_cart_vib 
    # else:
    #     q_cart_samp = q_temp_rovib
    #----------------------------- End of rotor sampling ----------------------------------------------------

    q = q_temp_rovib

    return(q)
#-------------------------------------------------------------------------------------------


if __name__ == "__main__":

    from hessian import getHessian

    def parseXYZ(xyz):
        lines = xyz.strip().split('\n')
        Natoms = sum(1 for line in lines if line.strip())

        q = []
        atoms = []
        for line in lines:
            parts = line.split()
            a, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append(a)
            q.extend([x, y, z])

        q = np.array(q)
        return Natoms, atoms, q

    def print_trajectory(trajfile, atoms, q, what, angle):
        trajfile.write(str(len(atoms)) + "\n")
        trajfile.write("%7s %20s %10.2f \n" % ("which= ", what, angle))
        au2Ang=0.5291772e0
        for i in range(0, len(atoms)):
            jx = 3 * i
            jy = 3 * i + 1
            jz = 3 * i + 2
            trajfile.write("%3s %15.5f  %15.5f %15.5f \n" % (atoms[i], q[jx]*au2Ang, q[jy]*au2Ang, q[jz]*au2Ang))

    '''
    freq = np.array([100.0928, 200, 300, 400, 500, 600, 700, 800, 900, 1000])*c10/c9*(math.pi * 2)


    vibrational_modes = initialize_vibrational_modes(freq, 'Temp', temp = 500.0)
    for i in vibrational_modes:
        print(i,":",vibrational_modes[i])
    print()

    vibrational_modes = initialize_vibrational_modes(freq, 'Temp')
    for i in vibrational_modes:
        print(i,":",vibrational_modes[i])
    print()

    vibrational_modes = initialize_vibrational_modes(freq, 'ZPE')
    for i in vibrational_modes:
        print(i,":",vibrational_modes[i])




    vib_modes = initialize_vibrational_modes(freq, init_type='ZPE')
    print("\nbefore setting modes")
    for i in vib_modes:
        print(i,":",vib_modes[i])

    fix_quantum = [(1, 19),
                   (2, 29)]

    fix_energy  = [(3, 650.0),
                   (4, 450.0)]

    fix_temp    = [(5, 330.0),
                   (6, 430.0)]

    rotor_list = [(7, (1, 2)),
                  (8, (4, 5)),
                  (None, (7, 9)),
                  (None, (12, 13)),
                  (None, (14, 15)),]

    #vib_modes = specify_modes(vibrational_modes, fix_quantum=fix_quantum, fix_energy=fix_energy, fix_temp=fix_temp, rotor_list=rotor_list)
    vib_modes = init_vib_rot_modes(freq, init_type='Temp', temp=210.0, fix_quantum=fix_quantum, fix_energy=fix_energy, fix_temp=fix_temp, rotor_list=rotor_list)
    #vib_modes = init_vib_rot_modes(freq, init_type='Temp', temp=210.0, rotor_list=rotor_list)
    #vib_modes = init_vib_rot_modes(freq, init_type='ZPE', temp=210.0)

    print("\nfix quantum: ", fix_quantum)
    print("fix energy: ", fix_energy)
    print("fix temp: ", fix_temp)
    print("rotor list: ", rotor_list)

# Print the updated vibrational modes
    print("\nafter setting modes")
    for i in vib_modes:
        print(i,":",vib_modes[i])
    '''

    seed = 211422
    random.seed(seed)


    c3   = 1838.6836605e0        # [g/mol]    * c3 = [electron mass unit]

    mH  = 1.00782503223*c3
    mC  = 12.011*c3
    mN = 14.007*c3
    mO  = 15.999*c3
    mKr = 83.798*c3

    qchem = 'XTB'
    functional = ''
    base = ''
    charge = 0
    multiplicity = 1
    path = '/home/peter/Programs/xtb-6.6.1/bin/xtb'
    nproc = 8
    wfu = False
    additional = '--acc 10'
    #additional = '--acc 20 --gfn 1'
    #additional = '--gfnff'


    qcinput = {
    'qchem': qchem,
    'path': path,
    'nproc': nproc,
    'functional': functional,
    'basis': base,
    'charge': charge,
    'multiplicity': multiplicity,
    'additional': additional,
    'wfu': False,
    }


    filename = 'diene'


    '''

    qchem = 'Orca'
    functional = 'B3LYP'
    base = 'pc-1'
    charge = -1
    multiplicity = 1
    path = '/home/peter/Programs/Orca.5.0.4/orca'
    nproc = 1
    wfu = False
    additional = ''


    qcinput = [0]*9
    qcinput[0] = qchem
    qcinput[1] = path
    qcinput[2] = nproc
    qcinput[3] = functional
    qcinput[4] = base
    qcinput[5] = charge
    qcinput[6] = multiplicity
    qcinput[7] = additional
    qcinput[8] = wfu

    filename = 'BrO3'
    '''



    hessFile = 'hessian_' + filename + '.hess'

   #IRC endpoint of openchain cis-1,3,5 triene
    xyz = '''
      C      -1.185385      1.500364     -0.174799
      C       0.057100      1.525641      0.290486
      C       1.182421      0.671307     -0.090281
      C       1.182420     -0.671306     -0.090282
      C       0.057100     -1.525640      0.290485
      C      -1.185388     -1.500364     -0.174796
      H       0.285347      2.226153      1.090336
      H       2.144030      1.164815     -0.199258
      H       2.144029     -1.164815     -0.199261
      H       0.285350     -2.226154      1.090332
      H      -1.972016     -2.076478      0.294389
      H      -1.462109     -0.924082     -1.042658
      H      -1.972017      2.076476      0.294382
      H      -1.462101      0.924083     -1.042665
     '''


    xyzTS = '''
    Br  0.12749716604946      0.13686847794079      0.65733607166257
    O   -0.28818586475166     -0.93233785502120     -0.76651616950362
    O   0.33471173146352     -0.36883118082864     -2.07899979157369
    O   -0.17420633526133      0.79641820500903     -2.38381561008524

     '''

    Natoms, atoms, q_eq = parseXYZ(xyz)

    q_eq = q_eq / 0.52917721092

    mass = get_mass_vector(atoms)
    #mass =  [mC]*6 + [mH]*8

    #hessFile = 'hessian_diene.hess'
    #hessFile = 'R2_m062xD3_ma-def2-TZVP_opt_freq.hess'

    hessian = getHessian(qcinput, hessFile, xyz)

    ww, ww_low, L = print_normalmode(fname=filename, atoms=atoms, mass=mass, q_eq=q_eq, hessian=hessian,
                                         give_freq_and_Lmat=True, Amp=30.0, is_eckart=True)

    #ww, ww_low, L = getNormalmode(mass, hessian, structure='TS')

    ww_all = np.append(ww_low, ww)

    print_frequencies(filename, ww_all)

    '''
    KWARGS list for print_normalmode:
    give_freq_and_Lmat = kwargs.get('give_freq_and_Lmat', False)
    linear = kwargs.get('linear', False)
    imode = kwargs.get('imode', None)
    Amp = kwargs.get('Amp', 0)
    '''


    #fix_quantum = [(0, 8)]

    fix_quantum = [(35, 6),
                   (34, 6)]

    fix_energy  = [(3, 650.0),
                   (4, 450.0)]

    fix_temp    = [(5, 330.0),
                   (6, 430.0)]

    rotor_list = [(None, (1, 2))]


    #vib_modes = init_vib_rot_modes(ww, init_type='Temp', temp=210.0, fix_quantum=fix_quantum)
    #vib_modes = init_vib_rot_modes(ww, init_type='Temp', temp=210.0, fix_quantum=fix_quantum, rotor_list=rotor_list)
    #vib_modes = init_vib_rot_modes(ww, init_type='Ene', temp=210.0, rotor_list=rotor_list)
    vib_modes = init_vib_rot_modes(ww, init_type='ZPE', temp=1210.0, fix_quantum=fix_quantum, rotor_list=rotor_list)
    #vib_modes = init_vib_rot_modes(ww, init_type='ZPE')

    # Print the updated vibrational modes
    print(f"\n{filename} sampling:\n")
    for i in vib_modes:
        print(i,":",vib_modes[i])

    with open('trajectory.txt', 'w') as trajfile:
        for i in range(1000):

            q = polyatom_vib_rot_sampling(mass=mass, atoms=atoms, q_eq=q_eq, ww=ww, L=L, vib_modes=vib_modes, verbosity=True, traj_index=i)
            #q = polyatom_vib_rot_sampling(mass=mass, atoms=atoms, q_eq=q_eq, ww=ww, L=L, vib_modes=vib_modes)

            print_trajectory(trajfile, atoms, q, "Sampling: ", i)



