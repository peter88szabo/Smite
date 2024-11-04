import numpy as np;
import random
import math

from utils.cenmass         import cenmass

from sampling.thermal      import thermal_vibr_mode 
from sampling.thermal      import thermal_rot_quantum_spherical_top
from sampling.polyrotation import angular_momentum 
from sampling.polyrotation import angmom_correction_after_vibrational_sampling 
from sampling.polyrotation import add_rotational_momentum 



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

    if sampling_mode == 'Q':
        jrot = excitation
        angmomabs = math.sqrt(jrot * (jrot + 1))
    elif sampling_mode == 'T':
        Rgas = (8.3144598/1000.0/2625.5) #in Hartree/K
        RT = Rgas * excitation #excitation is the temperature here
        jrot = thermal_rot_quantum_spherical_top(RT,Inertia)
        angmomabs = math.sqrt(jrot * (jrot + 1))
    else:
        raise ValueError("sampling_mode must be 'Q' or 'T'")
    #---------------------------------------------------------------------------------------

    ai = [1.0e20, Inertia, Inertia]

    angle = random.uniform(0, 2 * math.pi)
    angmom = [
        0.0,
        agnmomabs * math.sin(angle),
        agnmomabs * math.cos(angle)
    ]

    wx = 0.0
    wy = -angmom[1] / ai[1]
    wz = -angmom[2] / ai[2]

    angvel = [wx, wy, wz]

    p = add_rotational_momentum(angvel, mass, q, p)

    return (p, am, Inertia) #it does not change the positions only the momentum


def diatom_vibration_harmonic_sampling(vib_modes, req, omega, mass):
    if len(mass) != 2:
        raise ValueError("Lengths of mass vector in diatom() must be 2")


    sampling_mode = vib_modes[0][1] # it must be 'Q', 'E', 'T'
    excitation = vib_modes[0][2] #it's either the nvib quantum number, energy or temperature

    if sampling_mode == 'Q':
        energy = omega * (nvib + 0.5)
    elif sampling_mode == 'T':
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



