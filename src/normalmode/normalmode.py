import numpy as np
import math
from normalmode.eckart    import eckart_transform
from utils.constants      import AU_ANGULAR_FREQUENCY_TO_CM1, CM1_TO_AU_ANGULAR_FREQUENCY
from utils.constants      import BOHR_TO_ANGSTROM

from math import sin, cos

def eigval_to_freq(lambd):

    if lambd < 0.0:
        omega = -math.sqrt(abs(lambd))
    else:
        omega = math.sqrt(lambd)
    return omega


def getNormalmode(mass, hessian, linear=False, **kwargs):

    is_eckart = kwargs.get('is_eckart', True)
    if is_eckart:
        q_eq = kwargs.get('q_eq')

    nlow = 5 if linear else 6

    wmass = np.repeat(mass, 3)  # weights by coordinates

    M = np.diag(1.0 / np.sqrt(wmass))

   #mass-weighted Hessian:
    hess_mw = np.matmul(np.matmul(M,hessian),M)

   #projecting out the translation and rotation
    if is_eckart: 
        hess_mw = eckart_transform(mass, q_eq, hess_mw)
    else:
        hess_mw = 0.5 * (hess_mw + hess_mw.T)

    lambd, Lraw = np.linalg.eigh(hess_mw)

    #Lraw = np.array(L[:,6:])
    Lraw = np.array(Lraw)
    for i in range(len(mass)):
        for j in range(Lraw.shape[1]):
            Lraw[3*i  ,j] /= math.sqrt(mass[i])
            Lraw[3*i+1,j] /= math.sqrt(mass[i])
            Lraw[3*i+2,j] /= math.sqrt(mass[i])



   # Classify eigenvalues and locate their indices
    tolerance = (0.01 * CM1_TO_AU_ANGULAR_FREQUENCY)**2
    negative_eigval_ind = []
    zero_eigval_ind = []
    positive_eigval_ind = []

    for idx, eigenvalue in enumerate(lambd):
        gr = eigval_to_freq(eigenvalue) * AU_ANGULAR_FREQUENCY_TO_CM1
        if eigenvalue < 0 and abs(eigenvalue) > tolerance:
            negative_eigval_ind.append(idx)
        elif abs(eigenvalue) <= tolerance:
            zero_eigval_ind.append(idx)
        else:
            positive_eigval_ind.append(idx)

    ww = []
    ww_low = []
    Lfilter = []

    # A nonlinear structure has six external modes; a linear structure has five.
    if len(zero_eigval_ind) == nlow:
        for i in range(lambd.size):
            if i in zero_eigval_ind:  # If the eigenvalue is near zero
                ww_low.append(eigval_to_freq(lambd[i]))
            else:  # For non-zero eigenvalues
                Lfilter.append(Lraw[:, i])  
                ww.append(eigval_to_freq(lambd[i]))
        print(f"\nHessian has an optimal structure:")
        print(f"Found {nlow} zero eigenvalues of Hessian. Number of zero freqs (eigvals): {len(zero_eigval_ind)}")
        print(f"Normal modes are defined by the eigenvectors of non-zero eigenvalues.")
        print(f"(In case of TS structure, the largest negative eigval/eigvect is kept)\n")
    else:
        # A distorted structure may not have the expected number of numerically
        # zero external modes. Fall back to removing nlow eigenvectors.
        print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   Warning   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   ")
        print(f"Expected {nlow} zero eigenvalues of Hessian; found {len(zero_eigval_ind)}")
        print(f"Normal modes are defined by discarding the eigenvectors of the {nlow} lowest eigenvalues.")
        print(f"(In case of TS structure, the largest negative eigval/eigvect is kept)")
        print(f"Check the nature of the discarded normalmodes.\n")
        for i in range(0, nlow):
            ww_low.append(eigval_to_freq(lambd[i]))
        for i in range(nlow, lambd.size):
            Lfilter.append(Lraw[:, i])  
            ww.append(eigval_to_freq(lambd[i]))

    Lfilter = np.array(Lfilter).T  # Transpose to match the shape of original Lraw

    return ww, ww_low, Lfilter


def print_frequencies(fname, ww, linear=False):
    nlow = 5 if linear else 6

    print()
    print(f"---------- Low Frequencies ----------")
    print(f"%6s %6s %12s" % ("index1","index2", "freq[cm-1]"))
    for i in range(0, nlow):
        if ww[i] < 0.0 and abs(ww[i]) > 0.01 * CM1_TO_AU_ANGULAR_FREQUENCY:
            print(f"%6d %6d %12.2f %10s" % (i, i-nlow, ww[i] * AU_ANGULAR_FREQUENCY_TO_CM1, " <-- Imag"))
        else:
            print(f"%6d %6d %12.2f" % (i, i-nlow, ww[i] * AU_ANGULAR_FREQUENCY_TO_CM1))
    print(f"---------- High Frequencies ---------")
    for i in range(nlow, len(ww)):
        if ww[i] < 0.0:
            print(f"%6d %6d %12.2f %10s" % (i, i-nlow, ww[i] * AU_ANGULAR_FREQUENCY_TO_CM1, " <-- Imag"))
        else:
            print(f"%6d %6d %12.2f" % (i, i-nlow, ww[i] * AU_ANGULAR_FREQUENCY_TO_CM1))
    print(f"-----------------------------------\n")


    filename = "vibrational_freq_" + fname + ".dat"

    with open(filename, "w") as file:
        file.write("---------- Low Frequencies ----------\n")
        file.write("%6s %6s %12s \n" % ("index1","index2", "freq[cm-1]"))
        for i in range(0, nlow):
            if ww[i] < 0.0 and abs(ww[i]) > 0.01 * CM1_TO_AU_ANGULAR_FREQUENCY:
                file.write("%6d %6d %12.2f %10s \n" % (i, i-nlow, ww[i] * AU_ANGULAR_FREQUENCY_TO_CM1, " <-- Imag"))
            else:
                file.write("%6d %6d %12.2f \n" % (i, i-nlow, ww[i] * AU_ANGULAR_FREQUENCY_TO_CM1))
        file.write("---------- High Frequencies --------- \n")
        for i in range(nlow, len(ww)):
            if ww[i] < 0.0:
                file.write("%6d %6d %12.2f %10s \n" % (i, i-nlow, ww[i] * AU_ANGULAR_FREQUENCY_TO_CM1, " <-- Imag"))
            else:
                file.write("%6d %6d %12.2f \n" % (i, i-nlow, ww[i] * AU_ANGULAR_FREQUENCY_TO_CM1))

        file.write("-----------------------------------\n")


def _xyz_from_q(atoms, q_bohr):
    q_angstrom = np.asarray(q_bohr, dtype=float).reshape(-1) * BOHR_TO_ANGSTROM
    lines = []
    for i, atom in enumerate(atoms):
        j = 3 * i
        lines.append(
            f"{atom:2s} {q_angstrom[j]:16.10f} {q_angstrom[j+1]:16.10f} {q_angstrom[j+2]:16.10f}"
        )
    return "\n".join(lines)


def frequency_analysis(qcinput, atoms, q, fname, hessFile=None, linear=False, is_eckart=True,
                       force_hessian_recalc=False,
                       Amp_modeanim=30.0, print_nmode=True,
                       print_thermo=True, temp=298.15, pressure=101325.0,
                       multiplicity=None, qrrho_cutoff=50.0,
                       symmetry_number=1.0, chirality_number=1.0,
                       electronic_energy=None):
    """Run the same Hessian/frequency workflow used for polyatomic initialization.

    Coordinates are expected in Bohr. The printed output, frequency file, and
    optional normal-mode animation files are produced through the same functions
    used by Fragment.Polyatom_Init.
    """
    from normalmode.hessian import getHessian
    from normalmode.nmodeprint import print_normalmode
    from normalmode.thermochemistry import thermochemistry_analysis
    from utils.atomic_masses import get_mass_vector

    atoms = list(atoms)
    q = np.asarray(q, dtype=float).reshape(-1)
    if len(q) != 3 * len(atoms):
        raise ValueError("frequency_analysis requires 3 * len(atoms) Cartesian coordinates")

    if hessFile is None:
        hessFile = "hessian_" + fname + ".hess"

    xyz = _xyz_from_q(atoms, q)
    hess_qcinput = dict(qcinput)
    if force_hessian_recalc:
        hess_qcinput["force_hessian_recalc"] = True
    hessian = getHessian(qcinput=hess_qcinput, hessFile=hessFile, xyz=xyz)
    mass = get_mass_vector(atoms)

    if print_nmode:
        freq, freq_low, Lmat = print_normalmode(
            fname=fname,
            atoms=atoms,
            mass=mass,
            q_eq=q,
            hessian=hessian,
            give_freq_and_Lmat=True,
            Amp=Amp_modeanim,
            is_eckart=is_eckart,
            linear=linear,
        )
    else:
        freq, freq_low, Lmat = getNormalmode(
            mass=mass,
            hessian=hessian,
            linear=linear,
            q_eq=q,
            is_eckart=is_eckart,
        )

    freq_all = np.append(freq_low, freq)
    print_frequencies(fname, freq_all, linear=linear)

    thermo_result = None
    if print_thermo:
        if multiplicity is None:
            multiplicity = qcinput.get("multiplicity", 1)
        if electronic_energy is None:
            try:
                from integrators.gradient import Potential_Energy

                electronic_energy = float(Potential_Energy(qcinput, None, q, atoms))
            except Exception:
                electronic_energy = None
        thermo_result = thermochemistry_analysis(
            atoms=atoms,
            q_bohr=q,
            mass=mass,
            freqs_au=freq,
            temp=temp,
            pressure=pressure,
            multiplicity=multiplicity,
            qrrho_cutoff=qrrho_cutoff,
            symmetry_number=symmetry_number,
            chirality_number=chirality_number,
            electronic_energy=electronic_energy,
            print_report=True,
        )

    return {
        "atoms": atoms,
        "q": q,
        "hessian": hessian,
        "freq": freq,
        "freq_low": freq_low,
        "freq_all": freq_all,
        "Lmat": Lmat,
        "hessFile": hessFile,
        "freqFile": "vibrational_freq_" + fname + ".dat",
        "thermochemistry": thermo_result,
    }
