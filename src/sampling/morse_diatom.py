import math
import random

import numpy as np

from utils.cenmass import cenmass
from utils.constants import HARTREE_TO_CM1, HARTREE_TO_KCAL_MOL, HARTREE_TO_KJMOL, R_GAS_HARTREE_PER_K
from sampling.thermal import thermal_rot_quantum_spherical_top, thermal_vibr_mode
from sampling.polyrotation import add_rotational_momentum


def morse_omega0(beta, De, redmass):
    return beta * math.sqrt(2.0 * De / redmass)


def morse_energy(redmass, nv, jrot, beta, De, re):
    """
    Quasiclassical rovibrational energy of the rotating Morse oscillator.
    Ref.: Porter, Raff, Miller, JCP 63, 2214 (1975), Eq. (48).
    """
    omega0 = morse_omega0(beta, De, redmass)
    erot_rigid = jrot * (jrot + 1.0) / (2.0 * redmass * re * re)
    eharm = omega0 * (nv + 0.5)
    eanharm = eharm * eharm / (4.0 * De)
    enonrig = erot_rigid * erot_rigid / (De * beta * beta * re * re)
    ecoup = 3.0 * (1.0 - 1.0 / (beta * re)) * erot_rigid * eharm / (2.0 * beta * re * De)
    return erot_rigid + eharm - eanharm - enonrig - ecoup


def prmconst(angmom2, beta, De, re, Enj, redmass):
    """
    Parameters a, b, c for the sampled Morse variable xi = exp[-beta(r-re)].
    Ref.: Porter, Raff, Miller, JCP 63, 2214 (1975), Eq. (6).
    """
    gg = 1.0 / (beta * re)

    ap = (1.0 - 3.0 * gg * (1.0 - gg)) / (2.0 * redmass * re * re)
    bp = (1.0 - 1.5 * gg) * gg * 2.0 / (redmass * re * re)
    cp = gg * (1.0 - 3.0 * gg) / (2.0 * redmass * re * re)

    a = Enj - De - ap * angmom2
    b = 2.0 * De - bp * angmom2
    c = -De + cp * angmom2

    return a, b, c


def rprsetup(a, b, c, beta, re, redmass):
    """
    Sample the relative Morse coordinate r and conjugate momentum pr
    using the phase-space construction of Porter, Raff, and Miller.
    """
    qn = 2.0 * math.pi * random.uniform(0.0, 1.0)

    discr = b * b - 4.0 * a * c
    if discr <= 0.0:
        raise ValueError("Invalid Morse sampling parameters: b^2 - 4ac must be positive")

    csi = -2.0 * a / (b + math.sin(qn) * math.sqrt(discr))
    if csi <= 0.0:
        raise ValueError("Invalid Morse sampling parameters: sampled Morse variable must be positive")

    r = re - math.log(csi) / beta

    pr_sq = 2.0 * redmass * (a + b * csi + c * csi * csi)
    if pr_sq < -1.0e-12:
        raise ValueError("Invalid Morse sampling parameters: sampled pr^2 became negative")
    pr = math.sqrt(max(pr_sq, 0.0))

    if random.uniform(0.0, 1.0) < 0.5:
        pr = -pr

    return r, pr


def qpsetup(mass, r, pr):
    """
    Build Cartesian q/p for a diatom placed along the x-axis,
    then shift the COM to the origin.
    """
    q1 = np.array([r, 0.0, 0.0])
    q2 = np.array([0.0, 0.0, 0.0])

    p1 = np.array([pr, 0.0, 0.0])
    p2 = np.array([-pr, 0.0, 0.0])

    q = np.append(q1, q2)
    p = np.append(p1, p2)

    return cenmass(q, p, mass)


def sample_diatom_rotational_state(rot_modes, inertia):
    sampling_mode = rot_modes[0][0]
    excitation = rot_modes[0][1]

    print("\n-------------------------------------------------------------------------")
    print("Diatom Rotational Sampling:")
    if sampling_mode == 'Q':
        jrot = excitation
        print(f"Fix Quantum Number: Jrot = {jrot}")
    elif sampling_mode == 'T':
        print("Thermal Sampling:")
        print(f"Temp = {excitation:>12.2f} K")
        RT = R_GAS_HARTREE_PER_K * excitation
        jrot = thermal_rot_quantum_spherical_top(RT, inertia)
        print(f"jrot = {jrot}")
    else:
        raise ValueError("sampling_mode must be 'Q' or 'T'")

    angmomabs = math.sqrt(jrot * (jrot + 1.0))
    angle = random.uniform(0.0, 2.0 * math.pi)
    angmom = np.array([0.0, angmomabs * math.sin(angle), angmomabs * math.cos(angle)])
    erot = jrot * (jrot + 1.0) / (2.0 * inertia)
    print(f"Erot = {erot * HARTREE_TO_CM1:>12.2f} cm-1  {erot * HARTREE_TO_KJMOL:>12.3f} kJ/mol")
    print("-------------------------------------------------------------------------\n")

    return jrot, angmom, erot


def _print_morse_vibrational_summary(sampling_mode, excitation, nvib, ezero, evib):
    print("\n-------------------------------------------------------------------------")
    print("Diatom Morse Vibrational Sampling:")
    if sampling_mode == 'Q':
        print(f"Fix Quantum Number: nvib = {nvib}")
    elif sampling_mode == 'T':
        print("Thermal Sampling:")
        print(f"Temp = {excitation:>12.2f} K")
        print(f"nvib = {nvib}")
    elif sampling_mode == 'E':
        print("Fixed Energy Sampling:")
        print(f"Effective nvib = {nvib:.6f}")

    print(f"\n{'traj index:':15} {-999} {'     Ezero':15} {'      Evib':15} {'      Eexc':15}")
    print(f"{'kcal/mol -->':15} {ezero*HARTREE_TO_KCAL_MOL:15.3f} {evib*HARTREE_TO_KCAL_MOL:15.3f} {(evib-ezero)*HARTREE_TO_KCAL_MOL:15.3f}")
    print(f"{'kJ/mol   -->':15} {ezero*HARTREE_TO_KJMOL:15.3f} {evib*HARTREE_TO_KJMOL:15.3f} {(evib-ezero)*HARTREE_TO_KJMOL:15.3f}")
    print(f"{'cm-1     -->':15} {ezero*HARTREE_TO_CM1:15.3f} {evib*HARTREE_TO_CM1:15.3f} {(evib-ezero)*HARTREE_TO_CM1:15.3f}")
    print("-------------------------------------------------------------------------\n")


def diatom_vibration_morse_sampling(vib_modes, rot_modes, req, beta, De, mass):
    """
    Sample a rotating Morse oscillator in atomic units following
    Porter, Raff, and Miller (JCP 63, 2214, 1975).

    The sampled relative coordinate/momentum depends on the rotational
    quantum number, so the rovibrational state is built in one coupled step.
    """
    if len(mass) != 2:
        raise ValueError("Lengths of mass vector in diatom Morse sampling must be 2")

    redmass = mass[0] * mass[1] / (mass[0] + mass[1])
    omega0 = morse_omega0(beta, De, redmass)
    inertia_eq = redmass * req * req

    jrot, angmom, erot = sample_diatom_rotational_state(rot_modes, inertia_eq)

    sampling_mode = vib_modes[0][1]
    excitation = vib_modes[0][2]

    if sampling_mode == 'Q':
        nvib = excitation
    elif sampling_mode == 'T':
        RT = R_GAS_HARTREE_PER_K * excitation
        nvib = thermal_vibr_mode(RT, omega0)
    elif sampling_mode == 'E':
        # Keep the same external interface as the harmonic sampler:
        # a specified energy corresponds to a non-integer Morse action variable.
        nvib = excitation / omega0 - 0.5
    else:
        raise ValueError("sampling_mode must be 'Q', 'T', or 'E' for Morse diatom sampling")

    ezero = morse_energy(redmass, 0.0, jrot, beta, De, req)
    Enj = morse_energy(redmass, nvib, jrot, beta, De, req)
    _print_morse_vibrational_summary(sampling_mode, excitation, nvib, ezero, Enj)
    a, b, c = prmconst(jrot * (jrot + 1.0), beta, De, req, Enj, redmass)
    r, pr = rprsetup(a, b, c, beta, req, redmass)
    q, p = qpsetup(mass, r, pr)

    inertia = redmass * r * r
    angvel = np.array([0.0, angmom[1] / inertia, angmom[2] / inertia])
    p = add_rotational_momentum(angvel, mass, q, p)
    q, p = cenmass(q, p, mass)

    return q, p, angmom, inertia, jrot, nvib, Enj
