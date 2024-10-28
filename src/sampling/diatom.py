import numpy as np;
import random
import math
from utils.cenmass import cenmass
from utils.euler   import euler_rot


def diatom_init_harm(req, omega, mass, jrot, nvib):
    if len(mass) != 2:
        raise ValueError("Lengths of mass vector in diatom() must be 2")
    q = []
    p = []

    redmass = mass[0]*mass[1] / (mass[0] + mass[1])

   #Phase of vibration (angle of distance as cos(time*omega) --> cos(2pi*rnd)
    dr_angle = random.uniform(0, 2 * math.pi)

    dr = math.sqrt( 2 * ( nvib + 0.5 ) / ( redmass * omega)) * math.cos(dr_angle)

   # do the same for momentum
    pr_angle = random.uniform(0, 2 * math.pi)
    pr = - math.sqrt(2 * ( nvib + 0.5 ) * redmass * omega) * math.sin(pr_angle)
     
    r = req + dr

   #we just put in an arbitrary coordinate system
    q1 = [r, 0.0, 0.0]
    q2 = [0.0, 0.0, 0.0]

    vr = pr / redmass

   #center of mass coordinate system: velocity is distributed
    g = mass[1] / sum(mass)

    p1 = [mass[0] * vr * g, 0.0, 0.0]
    p2 = [mass[1] * (vr - p1[0]/mass[0]), 0.0, 0.0]

   #after this we shift everthing into the center of mass
    qq, pp = cenmass(q1+q2, p1+p2, mass)

#------------------We are done with the vibrational part ---------------------------
# Here comes the rotational sampling:

   #we convert the rotational quantum number (jrot) --> angmom
    agnmomabs = math.sqrt(jrot * ( jrot + 1))

   #this is the moment of inertia:
   #this is 2D object, thus the one of the components must be infinit
    ai = [1.0e20, redmass * r * r, redmass * r * r]

  #sample the phase of rotation:
    angle = random.uniform(0, 2 * math.pi)

  #2D rotor, hence one component of angmom is 0
    angmom = [
        0.0,
        agnmomabs * math.sin(angle),
        agnmomabs * math.cos(angle)
    ]

  # the angular velocities:
    wx = 0.0
    wy = -angmom[1] / ai[1]
    wz = -angmom[2] / ai[2]

 #concerting the angular velocities from internal coord to cartesian components
    for i in range(len(mass)):
        jx = i
        jy = i + 1
        jz = i + 2
        ang = np.array([
            wy * qq[jz] - wz * qq[jy],
            wz * qq[jx] - wx * qq[jz],
            wx * qq[jy] - wy * qq[jx]
        ])

       #from velocity to momentum
        pang = ang * mass[i]

      #pp are originally the center of mass (cartesian) vibrational coordinates
      # here we add the rotationally sampled coordinated to the vibrational ones
        pp[jx] -= pang[0]
        pp[jy] -= pang[1]
        pp[jz] -= pang[2]


    #the rotationally and vibrationally sampled molecule (this happened in internal coord that transforemed to Cartesian)
    #then we rotatate randomly w.r.t the Euler angles in the 3D space (coords and momenta as well)
    #around the the COM

    #q, p = euler_rot(qq, pp)

    return (qq, pp)

def distance_between_ij(i,j,q):
    tx = q[3*i+2] - q[3*j+2]
    ty = q[3*i+1] - q[3*j+1]
    tz = q[3*i]   - q[3*j]

    dist = np.sqrt(tx*tx + ty*ty + tz*tz)
    return dist




'''

import math
from format_and_print import print_trajectory

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

mH = 1.00782503223*c3
mC = 12.011*c3
mN = 14.007*c3
mO = 15.999*c3

Natoms = 2

mass =  [mH] + [mH]
atoms = ['H'] + ['O']

#-----------------------------------------------------
req = 1.0 # Angstrom
omega = 3000.0 #cm-1
jrot = 0
nvib = 1
seed = 222986
traj_file = "diatom_traj.xyz"
#-----------------------------------------------------
req = req * c1 
omega = omega * (math.pi * 2) * c10 /c9

random.seed(seed)



with open(traj_file, "a") as file_trj:
    for i in range(0,2000):
        q,p = diatom_init_harm(req, omega, mass, jrot, nvib)
        print_trajectory(file_trj, atoms, q, p, 0.0, 0.0, 1.0, i)
        dist = distance_between_ij(0,1,q) 
        print("%10d %20.3f" % (i, dist/c1))

'''
