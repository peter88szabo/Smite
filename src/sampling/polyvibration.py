import numpy as np
import random
import math
import os
from utils.cenmass             import cenmass
from utils.constants           import ANGSTROM_TO_BOHR, ATOMIC_MASS_GMOL_TO_AU
from utils.constants           import AU_ANGULAR_FREQUENCY_TO_CM1, CM1_TO_AU_ANGULAR_FREQUENCY
from utils.constants           import BOHR_TO_ANGSTROM, HARTREE_TO_KCAL_MOL, HARTREE_TO_CM1
from utils.constants           import HARTREE_TO_KJMOL, R_GAS_HARTREE_PER_K
from normalmode.normalmode     import getNormalmode, print_frequencies
from normalmode.nmodeprint     import print_normalmode
from normalmode.eckart         import eckart_transform
from sampling.thermal          import thermal_vibr_mode
#from atomic_overlap import check_atomic_overlap


def sample_wigner_ground_mode(omega):
    """Sample one harmonic mode from the ground-state Wigner distribution.

    In atomic units and in the mass-weighted normal coordinate convention,
    W0(Q, P) = pi^-1 exp(-omega Q^2 - P^2 / omega). Therefore the normal
    coordinate Q has variance 1/(2 omega), and its conjugate momentum P, here
    equal to Qdot for unit effective mass, has variance omega/2.
    """
    omega = abs(float(omega))
    if omega <= 0.0:
        raise ValueError("Wigner sampling requires a positive real frequency")
    xi_a = random.uniform(0.0, 1.0)
    xi_b = random.uniform(0.0, 1.0)
    radius = math.sqrt(-math.log(max(1.0e-300, 1.0 - xi_a)))
    angle = 2.0 * math.pi * xi_b
    q0 = 1.0 / math.sqrt(omega)
    q_norm = q0 * radius * math.cos(angle)
    v_norm = math.sqrt(omega) * radius * math.sin(angle)
    energy = 0.5 * (v_norm * v_norm + (omega * q_norm) * (omega * q_norm))
    effective_quantum = energy / omega - 0.5
    return q_norm, v_norm, energy, effective_quantum


def project_sampled_mode_energies(
    mass,
    q_eq,
    ww,
    L,
    q,
    p,
    *,
    requested_energies=None,
    requested_quanta=None,
    traj_index=-999,
    label="sampled polyatomic state",
    output_file="sampled_mode_energy_diagnostics.dat",
    print_report=True,
    write_file=True,
):
    """Project sampled Cartesian q,p back onto harmonic normal modes.

    The normal-mode matrix used in this code maps normal coordinates to
    Cartesians as q = q_eq + L Q. Its columns are mass-orthonormal, therefore
    the inverse projection is Q = L.T M dq and Qdot = L.T p.
    """
    mass = np.asarray(mass, dtype=float).reshape(-1)
    q_eq = np.asarray(q_eq, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    ww = np.asarray(ww, dtype=float).reshape(-1)
    L = np.asarray(L, dtype=float)
    if q.size != q_eq.size or p.size != q.size:
        raise ValueError("Mode-energy diagnostics require q_eq, q, and p with identical Cartesian size")
    if q.size != 3 * mass.size:
        raise ValueError("Mode-energy diagnostics require 3N Cartesian coordinates")
    if L.shape != (q.size, ww.size):
        raise ValueError(f"Mode matrix shape {L.shape} does not match expected {(q.size, ww.size)}")

    wmass = np.repeat(mass, 3)
    q_eq, _p_zero = cenmass(q_eq, np.zeros_like(q_eq), mass)
    q, p = cenmass(q, p, mass)
    dq = q - q_eq
    normal_q = L.T @ (wmass * dq)
    normal_v = L.T @ p
    omega = np.abs(ww)
    energies = 0.5 * (normal_v * normal_v + (omega * normal_q) * (omega * normal_q))
    effective_quanta = np.full_like(energies, np.nan, dtype=float)
    valid = omega > 0.0
    effective_quanta[valid] = energies[valid] / omega[valid] - 0.5

    requested_energies = (
        np.asarray(requested_energies, dtype=float).reshape(-1)
        if requested_energies is not None else np.full_like(energies, np.nan)
    )
    requested_quanta = (
        np.asarray(requested_quanta, dtype=float).reshape(-1)
        if requested_quanta is not None else np.full_like(energies, np.nan)
    )
    if requested_energies.size != energies.size:
        raise ValueError("requested_energies length does not match number of modes")
    if requested_quanta.size != energies.size:
        raise ValueError("requested_quanta length does not match number of modes")

    if print_report:
        print("\nBack-projected sampled normal-mode energies:")
        print("state:", label)
        print("traj index:", traj_index)
        print(
            "%5s %12s %14s %14s %14s %12s %12s"
            % ("mode", "freq_cm-1", "E_req_kcal", "E_rec_kcal", "dE_kcal", "n_req", "n_rec")
        )
        for imode in range(energies.size):
            diff = energies[imode] - requested_energies[imode]
            print(
                "%5d %12.2f %14.6f %14.6f %14.6f %12.4f %12.4f"
                % (
                    imode,
                    ww[imode] * AU_ANGULAR_FREQUENCY_TO_CM1,
                    requested_energies[imode] * HARTREE_TO_KCAL_MOL,
                    energies[imode] * HARTREE_TO_KCAL_MOL,
                    diff * HARTREE_TO_KCAL_MOL,
                    requested_quanta[imode],
                    effective_quanta[imode],
                )
            )
        print(
            "total requested/recovered [kcal/mol]: "
            f"{np.nansum(requested_energies) * HARTREE_TO_KCAL_MOL:.6f} "
            f"{np.sum(energies) * HARTREE_TO_KCAL_MOL:.6f}\n"
        )

    if write_file and output_file:
        new_file = not os.path.exists(output_file)
        with open(output_file, "a", encoding="utf-8") as handle:
            if new_file:
                handle.write(
                    "# traj_index mode freq_cm-1 requested_energy_Eh recovered_energy_Eh "
                    "delta_energy_Eh requested_quantum recovered_quantum label\n"
                )
            for imode in range(energies.size):
                handle.write(
                    f"{int(traj_index):8d} {imode:6d} "
                    f"{ww[imode] * AU_ANGULAR_FREQUENCY_TO_CM1:14.6f} "
                    f"{requested_energies[imode]:18.10e} {energies[imode]:18.10e} "
                    f"{energies[imode] - requested_energies[imode]:18.10e} "
                    f"{requested_quanta[imode]:14.6f} {effective_quanta[imode]:14.6f} "
                    f"{label}\n"
                )

    return {
        "normal_coordinate": normal_q,
        "normal_velocity": normal_v,
        "energy": energies,
        "effective_quantum": effective_quanta,
        "requested_energy": requested_energies,
        "requested_quantum": requested_quanta,
    }


#------------------------------------------------------------------------------------------
def initialize_vibrational_modes(freq, init_vib_type='ZPE', temp=300.0):
#------------------------------------------------------------------------------------------
    """
    Initializes a dictionary with a given number of modes, each set to ('Q', nvib=0) or ('T', temp).
    each mode is initialized as a quantized mode with ZPE energy or according to a temperature

    freq: Frequencies of vibrational modes.

    """
    freq = np.array(freq) * AU_ANGULAR_FREQUENCY_TO_CM1
    freq = np.round(freq, 3)

    nmodes = len(freq)

    init_vib_type_normalized = str(init_vib_type).lower()

    if init_vib_type_normalized == 'zpe':
        return {i: (freq[i], 'Q', 0) for i in range(nmodes)}
    elif init_vib_type_normalized == 'temp':
        return {i: (freq[i], 'T', temp) for i in range(nmodes)}
    elif init_vib_type_normalized == 'wigner':
        return {i: (freq[i], 'W', 0) for i in range(nmodes)}
    else:
        raise ValueError("init_type must be 'ZPE', 'Temp', or 'Wigner'")
#-------------------------------------------------------------------------------------------


def specify_vib_modes(vib_modes, **kwargs):
    """
    Sets the mode type and value for a given mode index.

    Modes how to sample: these could be signed as with Q, T, E, R
    to give quantum number (Q), Temperature (T), energy (E)

    {1: ('Q', 0),
     2: ('Q', 3),
     3: ('T', 500.0),
     4: ('T', 300.0),
     5: ('E', 30.0),
     6: ('E', 11.0),

    Parameters:
    vib_modes (dict): The dictionary of vibrational modes.

    kwargs:
    fix_quantum = [(1, 0),
                   (2, 0)]

    fix_energy  = [(3, 600.0),
                   (4, 400.0)]

    fix_temp    = [(5, 300.0),
                   (6, 400.0)]
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

    if 'fix_wigner' in kwargs:
        for mode_index in kwargs['fix_wigner']:
            if mode_index in vib_modes:
                frequency = vib_modes[mode_index][0]
                vib_modes[mode_index] = (frequency, 'W', 0)

    return vib_modes


#--------------------------------------------------------------------------------------------------------------------------------------------------
def polyatom_vibration_sampling(mass, atoms, q_eq, ww, L, vib_modes, **kwargs):
    '''
     q_eq: equilbiriom coords
     ww: freq
     L: eigenvector of Hessians (Normal mode <--> Cartesian space transformator)
    '''
#-------------------------------------------------------------------------------------------------------------------------------------------------
    #phase_sampling together with temperature, and other fixed fetched as kwargs

    temp = kwargs.get('temp', 300.0)
    phase_sampling = kwargs.get('phase_sampling', 'linear')
    verbosity  = kwargs.get('verbosity', False)
    traj_index =  kwargs.get('traj_index', -999)
    mode_energy_diagnostics = kwargs.get('mode_energy_diagnostics', True)
    mode_energy_file = kwargs.get('mode_energy_file', "sampled_mode_energy_diagnostics.dat")
    bond_th_HX = kwargs.get('bond_th_HX', 1.4 * ANGSTROM_TO_BOHR) # bond threshold in Angstrom for H-X, where X = any non H-atom
    bond_th_XX = kwargs.get('bond_th_XX', 2.0 * ANGSTROM_TO_BOHR) # bond threshold in Angstrom for X-X bonds to be considered as a part of a fragment
    #------------------------------------------------------------------------------------------

    wmass = np.repeat(mass, 3)

    freq_cutoff = 100.0 #cm-1

    energy = []
    requested_energy = []
    nvib = []
    requested_nvib = []
    freq_mode = np.zeros_like(ww)
    q_norm = np.zeros(len(ww), dtype=float)
    v_norm = np.zeros(len(ww), dtype=float)
    #the main loop running over the normal modes
    print("\nVibrational quantum number of sampled modes (freq in cm-1):")
    for imode in range(len(ww)):

        #In case of imaginary freqs (when ww is imaginary, it has a negative sign):
        ww[imode] = abs(ww[imode])
        freq_mode[imode] = ww[imode] * AU_ANGULAR_FREQUENCY_TO_CM1

        freq = vib_modes[imode][0] 
        sampling_mode = vib_modes[imode][1] # it must be 'Q', 'E', 'T', or 'R' 
        excitation = vib_modes[imode][2]  #either quantum number, energy or temperature value

        if sampling_mode == 'Q':
            nv = excitation 
            nvib += [nv]
            energy += [ww[imode]*(nv + 0.5)]
        elif sampling_mode == 'T':
            RT = R_GAS_HARTREE_PER_K * excitation
            #Truhlar like rounding/correction of too low frequencies
            mode_freq = (freq_cutoff * CM1_TO_AU_ANGULAR_FREQUENCY) if freq_mode[imode] < freq_cutoff else ww[imode]
            nv = thermal_vibr_mode(RT, mode_freq)
            nvib += [nv]
            energy += [ww[imode]*(nv + 0.5)]
        elif sampling_mode == 'E':
            energy += [excitation]
            nv = excitation/ww[imode] - 0.5 #non-integer quantum number
            nvib += [nv]
        elif sampling_mode == 'W':
            if excitation not in (None, 0, 0.0):
                raise ValueError("Wigner sampling currently supports only the vibrational ground state")
            q_norm[imode], v_norm[imode], sampled_energy, nv = sample_wigner_ground_mode(ww[imode])
            energy += [sampled_energy]
            nvib += [nv]
        else:
            raise ValueError("sampling_mode must be 'Q', 'E', 'T', or 'W'") 

        if sampling_mode == 'W':
            requested_energy += [0.5 * ww[imode]]
            requested_nvib += [0]
        else:
            requested_energy += [energy[-1]]
            requested_nvib += [nv]

        label = "Wigner" if sampling_mode == 'W' else str(nv)
        print(f"{imode:<5d}  {freq_mode[imode]:>10.2f}  {label}")
    #------------------------------------------------------------------------------------------

    Evib = sum(np.array(energy))
    Ezero = 0.5*sum(np.array(ww))

    print(f"\n{'traj index:':15} {traj_index} {'     Ezero':15} {'      Evib':15} {'      Eexc':15}")
    print(f"{'kcal/mol -->':15} {Ezero*HARTREE_TO_KCAL_MOL:15.3f} {Evib*HARTREE_TO_KCAL_MOL:15.3f} {(Evib-Ezero)*HARTREE_TO_KCAL_MOL:15.3f}")
    print(f"{'kJ/mol   -->':15} {Ezero*HARTREE_TO_KJMOL:15.3f} {Evib*HARTREE_TO_KJMOL:15.3f} {(Evib-Ezero)*HARTREE_TO_KJMOL:15.3f}")
    print(f"{'cm-1     -->':15} {Ezero*HARTREE_TO_CM1:15.3f} {Evib*HARTREE_TO_CM1:15.3f} {(Evib-Ezero)*HARTREE_TO_CM1:15.3f}\n")

    if verbosity == True:
        output_file = "sampled_mode_energies.txt" 
        with open(output_file, "a") as f:  # Open the file in append mode
            f.write(f"{'traj index:':15} {traj_index} {'      Ezero':15} {'       Evib':15} {'       Eexc':15}\n")
            f.write(f"{'kcal/mol -->':15} {Ezero*HARTREE_TO_KCAL_MOL:15.3f} {Evib*HARTREE_TO_KCAL_MOL:15.3f} {(Evib-Ezero)*HARTREE_TO_KCAL_MOL:15.3f}\n")
            f.write(f"{'kJ/mol   -->':15} {Ezero*HARTREE_TO_KJMOL:15.3f} {Evib*HARTREE_TO_KJMOL:15.3f} {(Evib-Ezero)*HARTREE_TO_KJMOL:15.3f}\n")
            f.write(f"{'cm-1     -->':15} {Ezero*HARTREE_TO_CM1:15.3f} {Evib*HARTREE_TO_CM1:15.3f} {(Evib-Ezero)*HARTREE_TO_CM1:15.3f}\n\n")


    #energy to amolitude
    #Amplitude of normal modes for the original fixed-energy QCT modes.
    ampl = [
        0.0 if vib_modes[i][1] == 'W' else math.sqrt(2.0 * energy[i]) / ww[i]
        for i in range(len(ww))
    ]



    if phase_sampling != 'linear':
        raise NotImplementedError(f"phase_sampling='{phase_sampling}' is not implemented in polyatom_vibration_sampling()")

    for i in range(len(ampl)):
        if vib_modes[i][1] == 'W':
            continue
        phase = random.uniform(0, 2 * math.pi)
        q_norm[i] = ampl[i] * math.cos(phase)
        v_norm[i] = -ampl[i] * ww[i] * math.sin(phase)

    #Normal mode to Cartesian transformation:
    q_cart_vib = np.transpose(q_eq) + np.matmul(L, np.transpose(np.array(q_norm)))
    v_cart_vib = np.matmul(L, np.transpose(v_norm))
    p_cart_vib = [wmass[i] * v_cart_vib[i] for i in range(len(v_cart_vib)) ]

    q,p = cenmass(q_cart_vib, p_cart_vib, mass)

    if mode_energy_diagnostics:
        project_sampled_mode_energies(
            mass,
            q_eq,
            ww,
            L,
            q,
            p,
            requested_energies=requested_energy,
            requested_quanta=requested_nvib,
            traj_index=traj_index,
            label="after vibrational sampling",
            output_file=mode_energy_file,
            print_report=True,
            write_file=True,
        )

    #q,p,ai,am = poly_rotation_init(jrot, q_eq, mass, q, p)

    #Randomly roteate the molecule about its center of mass
    #q,p = euler_rot(q, p)


    return(q,p)
#-------------------------------------------------------------------------------------------


if __name__ == "__main__":

    from normalmode.hessian        import getHessian
    from utils.atomic_masses       import get_mass_vector 

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
        for i in range(0, len(atoms)):
            jx = 3 * i
            jy = 3 * i + 1
            jz = 3 * i + 2
            trajfile.write("%3s %15.5f  %15.5f %15.5f \n" % (atoms[i], q[jx]*BOHR_TO_ANGSTROM, q[jy]*BOHR_TO_ANGSTROM, q[jz]*BOHR_TO_ANGSTROM))

    '''
    freq = np.array([100.0928, 200, 300, 400, 500, 600, 700, 800, 900, 1000]) * CM1_TO_AU_ANGULAR_FREQUENCY


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


    #vib_modes = specify_modes(vibrational_modes, fix_quantum=fix_quantum, fix_energy=fix_energy, fix_temp=fix_temp)
    vib_modes = init_vib_rot_modes(freq, init_type='Temp', temp=210.0, fix_quantum=fix_quantum, fix_energy=fix_energy, fix_temp=fix_temp)
    #vib_modes = init_vib_rot_modes(freq, init_type='Temp', temp=210.0)
    #vib_modes = init_vib_rot_modes(freq, init_type='ZPE', temp=210.0)

    print("\nfix quantum: ", fix_quantum)
    print("fix energy: ", fix_energy)
    print("fix temp: ", fix_temp)

# Print the updated vibrational modes
    print("\nafter setting modes")
    for i in vib_modes:
        print(i,":",vib_modes[i])
    '''

    seed = 211422
    random.seed(seed)


    mH  = 1.00782503223 * ATOMIC_MASS_GMOL_TO_AU
    mC  = 12.011 * ATOMIC_MASS_GMOL_TO_AU
    mN = 14.007 * ATOMIC_MASS_GMOL_TO_AU
    mO  = 15.999 * ATOMIC_MASS_GMOL_TO_AU
    mKr = 83.798 * ATOMIC_MASS_GMOL_TO_AU

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

    q_eq = q_eq * ANGSTROM_TO_BOHR

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



    #vib_modes = init_vib_rot_modes(ww, init_type='Temp', temp=210.0, fix_quantum=fix_quantum)
    #vib_modes = init_vib_rot_modes(ww, init_type='Temp', temp=210.0, fix_quantum=fix_quantum)
    #vib_modes = init_vib_rot_modes(ww, init_type='Ene', temp=210.0)
    vib_modes = init_vib_rot_modes(ww, init_type='ZPE', temp=1210.0, fix_quantum=fix_quantum)
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
