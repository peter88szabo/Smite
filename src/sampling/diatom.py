import numpy as np;
import random
import math

from utils.cenmass         import cenmass
from utils.constants       import HARTREE_TO_CM1, HARTREE_TO_EV, HARTREE_TO_KCAL_MOL
from utils.constants       import HARTREE_TO_KJMOL, R_GAS_HARTREE_PER_K

from sampling.thermal      import thermal_vibr_mode 
from sampling.thermal      import thermal_rot_quantum_spherical_top
from sampling.polyvibration import sample_wigner_ground_mode
from sampling.polyrotation import angular_momentum 
from sampling.polyrotation import angmom_correction_after_vibrational_sampling 
from sampling.polyrotation import add_rotational_momentum 




def _print_diatom_vibrational_summary(title, sampling_mode, excitation, nvib, ezero, evib):
    print("\n-------------------------------------------------------------------------")
    print(title)
    if sampling_mode == 'Q':
        print(f"Fix Quantum Number: nvib = {nvib}")
    elif sampling_mode == 'T':
        print("Thermal Sampling:")
        print(f"Temp = {excitation:>12.2f} K")
        print(f"nvib = {nvib}")
    elif sampling_mode == 'E':
        print("Fixed Energy Sampling:")
        print(f"Effective nvib = {nvib:.6f}")
    elif sampling_mode == 'W':
        print("Ground-state Wigner Sampling:")
        print(f"Effective nvib = {nvib:.6f}")

    print(f"\n{'traj index:':15} {-999} {'     Ezero':15} {'      Evib':15} {'      Eexc':15}")
    print(f"{'kcal/mol -->':15} {ezero*HARTREE_TO_KCAL_MOL:15.3f} {evib*HARTREE_TO_KCAL_MOL:15.3f} {(evib-ezero)*HARTREE_TO_KCAL_MOL:15.3f}")
    print(f"{'kJ/mol   -->':15} {ezero*HARTREE_TO_KJMOL:15.3f} {evib*HARTREE_TO_KJMOL:15.3f} {(evib-ezero)*HARTREE_TO_KJMOL:15.3f}")
    print(f"{'cm-1     -->':15} {ezero*HARTREE_TO_CM1:15.3f} {evib*HARTREE_TO_CM1:15.3f} {(evib-ezero)*HARTREE_TO_CM1:15.3f}")
    print("-------------------------------------------------------------------------\n")


def diatom_rotation_rigidrot_sampling(rot_modes, mass, q, p):

    q, p = cenmass(q, p, mass)

    rdist = distance_between_ij(0, 1, q)

    redmass = mass[0]*mass[1] / (mass[0] + mass[1])
    Inertia = redmass * rdist * rdist

    #---------------------------------------------------------------------------------------
    #obtain a rotational quantum number (fixed J, fixed energy or thermal sampling)
    #---------------------------------------------------------------------------------------
    sampling_mode = rot_modes[0][0] # it must be 'Q', 'E', 'T'
    excitation = rot_modes[0][1] #it's either the Jrot quantum number, energy or temperature

    print(f"\n-------------------------------------------------------------------------")
    print(f"Diatom Rotational Sampling:")
    if sampling_mode == 'Q':
        jrot = excitation
        angmomabs = math.sqrt(jrot * (jrot + 1))
    elif sampling_mode == 'T':
        print(f"Thermal Sampling:")
        print(f"Temp = {excitation:>12.2f} K")
        print(f"Quantum number sampled directly from thermal distribution")

        RT = R_GAS_HARTREE_PER_K * excitation
        jrot = thermal_rot_quantum_spherical_top(RT,Inertia)
        print("jrot = ", jrot)
        angmomabs = math.sqrt(jrot * (jrot + 1))
        print(f"angmom = {angmomabs:>12.5f} a.u.")
    else:
        raise ValueError("sampling_mode must be 'Q' or 'T'")
    #---------------------------------------------------------------------------------------

    ai = [1.0e20, Inertia, Inertia]

    angle = random.uniform(0, 2 * math.pi)
    angmom = [
        0.0,
        angmomabs * math.sin(angle),
        angmomabs * math.cos(angle)
    ]

    Erot1 = sum([angmom[i]**2/ai[i]/2.0 for i in range(len(angmom))])
    Erot = jrot*(jrot+1.0)/Inertia/2.0
    #print(f"Erot1 = {Erot1*HARTREE_TO_CM1:>12.2f} cm-1  {Erot1*HARTREE_TO_KJMOL:>12.3f} kJ/mol  {Erot1*HARTREE_TO_EV:>12.5f} eV")
    print(f"Erot = {Erot*HARTREE_TO_CM1:>12.2f} cm-1  {Erot*HARTREE_TO_KJMOL:>12.3f} kJ/mol  {Erot*HARTREE_TO_EV:>12.5f} eV")

    print(f"-------------------------------------------------------------------------\n")

    # add_rotational_momentum now adds p_rot = m (omega x r), so the
    # physical angular velocity components must carry the same sign as L = I omega.
    wx = 0.0
    wy = angmom[1] / ai[1]
    wz = angmom[2] / ai[2]

    angvel = np.array([wx, wy, wz])

    p = add_rotational_momentum(angvel, mass, q, p)

    return (p, angmom, Inertia) #it does not change the positions only the momentum


def diatom_vibration_harmonic_sampling(vib_modes, req, omega, mass):
    if len(mass) != 2:
        raise ValueError("Lengths of mass vector in diatom() must be 2")


    sampling_mode = vib_modes[0][1] # it must be 'Q', 'E', 'T'
    excitation = vib_modes[0][2] #it's either the nvib quantum number, energy or temperature

    redmass = mass[0]*mass[1] / (mass[0] + mass[1])

    if sampling_mode == 'Q':
        nvib = excitation
        energy = omega * (nvib + 0.5)
    elif sampling_mode == 'T':
        RT = R_GAS_HARTREE_PER_K * excitation
        nvib = thermal_vibr_mode(RT, omega)
        energy = omega * (nvib + 0.5)
    elif sampling_mode == 'E':
        nvib = excitation/omega - 0.5 #non-integer quantum number
        energy = excitation
    elif sampling_mode == 'W':
        if excitation not in (None, 0, 0.0):
            raise ValueError("Wigner sampling currently supports only the vibrational ground state")
        q_norm, v_norm, energy, nvib = sample_wigner_ground_mode(omega)
    else:
        raise ValueError("Either nvib=xxx, temp=xxx, energy=xxx, or Wigner mode must be given as input in diatom_vibration_harmonic_sampling()")

    ezero = 0.5 * omega
    _print_diatom_vibrational_summary(
        title="Diatom Harmonic Vibrational Sampling:",
        sampling_mode=sampling_mode,
        excitation=excitation,
        nvib=nvib,
        ezero=ezero,
        evib=energy,
    )

    q = []
    p = []

    if sampling_mode == 'W':
        dr = q_norm / math.sqrt(redmass)
        pr = math.sqrt(redmass) * v_norm
    else:
       #Phase of vibration (angle of distance as cos(time*omega) --> cos(2pi*rnd)
        phase = random.uniform(0, 2 * math.pi)

        dr = math.sqrt( 2 * ( nvib + 0.5 ) / ( redmass * omega)) * math.cos(phase)
        pr = - math.sqrt(2 * ( nvib + 0.5 ) * redmass * omega) * math.sin(phase)
     
    r = req + dr

    q1 = np.array([r, 0.0, 0.0])
    q2 = np.array([0.0, 0.0, 0.0])

    vr = pr / redmass

   #center of mass coordinate system: velocity is distributed
    g = mass[1] / sum(mass)

    p1 = np.array([pr, 0.0, 0.0])
    p2 = np.array([-pr, 0.0, 0.0])

    q = np.append(q1, q2)
    p = np.append(p1, p2)

    q, p = cenmass(q, p, mass)

    return (q, p)

def distance_between_ij(i,j,q):
    tx = q[3*i]   - q[3*j]
    ty = q[3*i+1] - q[3*j+1]
    tz = q[3*i+2] - q[3*j+2]

    dist = np.sqrt(tx*tx + ty*ty + tz*tz)
    return dist
