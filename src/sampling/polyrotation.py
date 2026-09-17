import numpy as np
import random
import math
from utils.constants import HARTREE_TO_CM1, HARTREE_TO_EV, HARTREE_TO_KJMOL, R_GAS_HARTREE_PER_K

from sampling.thermal   import thermal_rot_quantum_spherical_top
from sampling.thermal   import thermal_rot_classic_spherical_top 
from sampling.thermal   import thermal_rot_asymmetric_top_equipart 
from sampling.thermal   import thermal_rot_canonical_top


#from cenmass import cenmass
#from euler import euler_rot
#from thermal import thermal_vibr_mode  

from math import sin, cos

def _inertia_properties(qq, w):
    """Return the physical inertia tensor and principal-axis decomposition.

    Coordinates must be relative to the molecular center of mass.
    ``principal_axes[:, i]`` is principal axis ``i`` expressed in the
    laboratory Cartesian frame.
    """
    q_xyz = np.asarray(qq, dtype=float).reshape((-1, 3))
    mass = np.asarray(w, dtype=float).reshape(-1)
    if q_xyz.shape[0] != mass.size:
        raise ValueError("Inertia calculation requires one mass per Cartesian atom")

    inertia = np.zeros((3, 3), dtype=float)
    identity = np.eye(3)
    for atom_mass, position in zip(mass, q_xyz):
        inertia += atom_mass * (
            np.dot(position, position) * identity - np.outer(position, position)
        )

    principal_moments, principal_axes = np.linalg.eigh(inertia)
    # This equals the inverse for nonlinear molecules and remains defined for
    # a linear molecule's zero principal moment.
    inertia_inverse = np.linalg.pinv(inertia, rcond=1.0e-12, hermitian=True)
    return principal_moments, principal_axes, inertia, inertia_inverse


def calcI(qq, w):
    """Return principal moments and the laboratory-frame inverse inertia."""
    principal_moments, _principal_axes, _inertia, inertia_inverse = _inertia_properties(qq, w)
    return principal_moments, inertia_inverse



def angular_momentum(q, p):
    am = [0.0,0.0,0.0]
    for i in range(len(q)//3):
        jx = 3*i
        jy = 3*i+1
        jz = 3*i+2
        am[0] += q[jy]*p[jz] - q[jz]*p[jy]
        am[1] += q[jz]*p[jx] - q[jx]*p[jz]
        am[2] += q[jx]*p[jy] - q[jy]*p[jx]
    return am

def angmom_correction_after_vibrational_sampling(am_rot, q, p):
    '''
    To rotate the molecule after vibrational sampling
    we have to take into account that the sampling 
    generates distorted (non-equilbiriom molecule)
    with a given momentum. These q,p coordinates
    generate naturally an angular momentum
    stemming from vibrations:

    Lvib = q x p

    when we sample the vibration w.r.t a fix rotational
    quantum numbmer or the corresponding angular momentum
     
    we would overshoot the magnitude of the angomom without
    the correction w.r.t the vibrational angular momemtnum (Lvib)

    Hence, if we wish to do the rotational sampling with a fix Lrot
    angular momentum the real angular momentum used in rotational
    sampling must be (to start the trajectory according to Lrot): 

    Lsampling = Lrot - Lvib 
    '''
    #q and p must the coordinates after vibratinal sampling
    am_vib = angular_momentum(q, p)

    am_corrected = am_rot - am_vib 

    return am_corrected

def add_rotational_momentum(angvel, mass, q, p):

    wx,wy,wz = angvel.tolist()

    ang = []
    for i in range(len(mass)):
        jx = 3*i
        jy = 3*i+1
        jz = 3*i+2
        qx, qy, qz = q[jx], q[jy], q[jz]

        ang = np.array([
                wy*qz - wz*qy,
                wz*qx - wx*qz,
                wx*qy - wy*qx
              ])

        # Rotational momentum of atom i is p_rot = m_i (omega x r_i).
        # This must be added to the current momentum so that the assembled
        # coordinates carry the target angular momentum L = I omega.
        pang = mass[i] * ang

        # Add the rotational contribution with the physical sign.
        p[jx] += pang[0]
        p[jy] += pang[1]
        p[jz] += pang[2]

    return p


def initialize_rotational_modes(init_rot_type='Jfix', temp=300.0, jrot=0, krot=0):
    """
    Each mode rotational mode is initialized as
    a quantized mode with Fixed J quantum numer or w.r.t a temperature
    """
    nmodes = 1 #at the moment we have only one mode, later we can take care of k-rotors too

    if init_rot_type == 'Jfix':
        return {i: ('Q', jrot) for i in range(nmodes)}
    elif init_rot_type == 'Temp':
        return {i: ('T', temp) for i in range(nmodes)}
    else:
        raise ValueError("init_type must be 'Jfix' or 'Temp'")



def polyatom_rotation_sampling(rot_modes, mass, q, p):

    # Ixyz contains principal moments. Iinv acts in the laboratory frame,
    # while principal_axes maps principal-axis components into that frame.
    Ixyz, principal_axes, _inertia, Iinv = _inertia_properties(q, mass)

    #---------------------------------------------------------------------------------------
    #obtain a rotational quantum number (fixed J, fixed energy or thermal sampling)
    #---------------------------------------------------------------------------------------
    sampling_mode = rot_modes[0][0] # it must be 'Q', 'E', 'T'
    excitation = rot_modes[0][1] #it's either the Jrot quantum number, energy or temperature

    print("\n-------------------------------------------------------------------------")
    print(f"Polyatomic Rotational Sampling:")
    if sampling_mode == 'Q':
        print(f"Fix Quantum Number:")
        print(f"Jrot = {excitation}")
        jrot = excitation
        amrot = math.sqrt(jrot * (jrot + 1))

        # Sample the target angular-momentum direction uniformly on the sphere.
        cos_theta = random.uniform(-1.0, 1.0)
        sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
        phi = random.uniform(0.0, 2.0 * math.pi)

        am = np.array([
            amrot * sin_theta * cos(phi),
            amrot * sin_theta * sin(phi),
            amrot * cos_theta,
         ])

    elif sampling_mode == 'T':
        print(f"Thermal Sampling:")
        print(f"Temp = {excitation:>12.2f} K")
        print(f"Angular momentum has been sampled directly (not the quantum number)")
        print(f"from the canonical rigid-top distribution")

        RT = R_GAS_HARTREE_PER_K * excitation #excitation is the temperature here
       #am = thermal_rot_asymmetric_top_equipart(RT, Ixyz) 
        am_principal = thermal_rot_canonical_top(RT, Ixyz)
        # The thermal sampler returns components along the principal axes.
        # Convert them to the laboratory frame before combining them with
        # Cartesian vibrational angular momentum.
        am = principal_axes @ am_principal
        am_abs = np.linalg.norm(am)
        print(f"Princ. Mom. Inertia = [{Ixyz[0]:<12.5f} {Ixyz[1]:>12.5f} {Ixyz[2]:>12.5f}] a.u.")
        print(f"Angmom =  [{am[0]:<12.5f} {am[1]:>12.5f} {am[2]:>12.5f}] a.u.")
        print(f"Angmom_abs = {am_abs:>12.5f} a.u.")
    else:
        raise ValueError("sampling_mode must be 'Q', 'T'")
    #---------------------------------------------------------------------------------------

    Erot = 0.5 * float(am @ Iinv @ am)
    print(f"Erot = {Erot*HARTREE_TO_CM1:>12.2f} cm-1  {Erot*HARTREE_TO_KJMOL:>12.3f} kJ/mol  {Erot*HARTREE_TO_EV:>12.5f} eV")
    print("-------------------------------------------------------------------------\n")

    am_corrected = angmom_correction_after_vibrational_sampling(am, q, p)

    #calculate angular velocities (w) from the inverse of intertia tensor
    # I * w = L   --->  I(-1) * L = w
    angvel = np.matmul(Iinv, np.transpose(am_corrected))

    #Add rotational momentum to the vibrational one
    p = add_rotational_momentum(angvel, mass, q, p)

    # Cartesian momenta are the source of truth. Guard against future
    # sign/frame regressions by checking that they carry the requested total J.
    am_actual = np.asarray(angular_momentum(q, p), dtype=float)
    angmom_tolerance = 1.0e-9 * max(1.0, float(np.linalg.norm(am)))
    if not np.allclose(am_actual, am, rtol=1.0e-9, atol=angmom_tolerance):
        raise RuntimeError(
            "Rotational sampling failed to realize the requested angular momentum: "
            f"target={am}, actual={am_actual}"
        )

    return (p, am, Ixyz)
#asd
