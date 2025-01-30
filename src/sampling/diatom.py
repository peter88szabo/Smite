import numpy as np;
import random
import math

from utils.cenmass         import cenmass

from sampling.thermal      import thermal_vibr_mode 
from sampling.thermal      import thermal_rot_quantum_spherical_top
from sampling.polyrotation import angular_momentum 
from sampling.polyrotation import angmom_correction_after_vibrational_sampling 
from sampling.polyrotation import add_rotational_momentum 

#     [Anstrom]*c1=[bohr]
c1=1.0e0/0.5291772e0
#     [kcal/mol]*c2=[Hartree]
c2=1.e0/627.51e0
#     [g/mol]*c3=[electron mass unit]
c3=1838.6836605e0
#     [Hartree]*c4=[eV]
c4=27.2114
#     [Hartree]*c5=[cm-1]
c5=219474.e0
#     [femto-sec]*c6=[time in au]
c6=41.341105
#     [frequency in cm-1]*c9=[freq(bohr^(-1))]
c9=1.0e8*c1
#     [speed of light in atomic unit]
c10=137.035999074


c7 = 2625.5         # [Hartree] * c7 = [kJ/mol]

Rgas = 8.3144598/1000.0/c7 #Hartree/K




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

        Rgas = (8.3144598/1000.0/2625.5) #in Hartree/K
        RT = Rgas * excitation #excitation is the temperature here
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
    #print(f"Erot1 = {Erot1*c5:>12.2f} cm-1  {Erot1*c7:>12.3f} kJ/mol  {Erot1*c4:>12.5f} eV")
    print(f"Erot = {Erot*c5:>12.2f} cm-1  {Erot*c7:>12.3f} kJ/mol  {Erot*c4:>12.5f} eV")

    print(f"-------------------------------------------------------------------------\n")

    wx = 0.0
    wy = -angmom[1] / ai[1]
    wz = -angmom[2] / ai[2]

    angvel = np.array([wx, wy, wz])

    p = add_rotational_momentum(angvel, mass, q, p)

    return (p, angmom, Inertia) #it does not change the positions only the momentum


def diatom_vibration_harmonic_sampling(vib_modes, req, omega, mass):
    if len(mass) != 2:
        raise ValueError("Lengths of mass vector in diatom() must be 2")


    sampling_mode = vib_modes[0][1] # it must be 'Q', 'E', 'T'
    excitation = vib_modes[0][2] #it's either the nvib quantum number, energy or temperature

    if sampling_mode == 'Q':
        print(f"Diatom Vibrational Sampling: Fix nvib = {nvib:<d10}")
        energy = omega * (nvib + 0.5)
    elif sampling_mode == 'T':
        print(f"Diatom Vibrational Sampling: Thermal Temp = {excitation:<d12.2} K")
        Rgas = (8.3144598/1000.0/2625.5) #in Hartree/K
        RT = Rgas * excitation #excitation is the temperature here
        nvib = thermal_vibr_mode(RT, omega)
        energy = omega * (nvib + 0.5)
    elif sampling_mode == 'E':
        nvib = excitation/omega - 0.5 #non-integer quantum number
    else:
        raise ValueError("Either nvib=xxx or temp=xxx or energy=xxx must be given as input in diatom_vibration_harmonic_sampling()")

    q = []
    p = []

    redmass = mass[0]*mass[1] / (mass[0] + mass[1])

   #Phase of vibration (angle of distance as cos(time*omega) --> cos(2pi*rnd)
    dr_angle = random.uniform(0, 2 * math.pi)

    dr = math.sqrt( 2 * ( nvib + 0.5 ) / ( redmass * omega)) * math.cos(dr_angle)

    pr_angle = random.uniform(0, 2 * math.pi)
    pr = - math.sqrt(2 * ( nvib + 0.5 ) * redmass * omega) * math.sin(pr_angle)
     
    r = req + dr

    q1 = np.array([r, 0.0, 0.0])
    q2 = np.array([0.0, 0.0, 0.0])

    vr = pr / redmass

   #center of mass coordinate system: velocity is distributed
    g = mass[1] / sum(mass)

    p1 = np.array([mass[0] * vr * g, 0.0, 0.0])
    p2 = np.array([mass[1] * (vr - p1[0]/mass[0]), 0.0, 0.0])

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



