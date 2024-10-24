import numpy as np
import random
import math
#from cenmass import cenmass
#from euler import euler_rot
#from thermal import thermal_vibr_mode  

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

from math import sin, cos

def calcI(qq, w):
    Ixx = 0.0
    Iyy = 0.0
    Izz = 0.0
    Ixy = 0.0
    Ixz = 0.0
    Iyz = 0.0
    for i in range(len(w)):
        jx = 3*i
        jy = 3*i+1
        jz = 3*i+2

        Ixx += w[i] * (qq[jy]**2 + qq[jz]**2)
        Iyy += w[i] * (qq[jx]**2 + qq[jz]**2)
        Izz += w[i] * (qq[jx]**2 + qq[jy]**2)
        Ixy += w[i] * qq[jx]*qq[jy]
        Ixz += w[i] * qq[jx]*qq[jz]
        Iyz += w[i] * qq[jy]*qq[jz]
    I = np.array([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz]
        ])
    #principial axis and eigenvalues
    ai, dontneed = np.linalg.eigh(I)
    Iinv = np.linalg.inv(I)
    return (ai, Iinv)



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

    am_corred = am_rot - am_vib 

    return am_corrted

def add_rotational_momentum(angvel, mass, q, p):

    wx,wy,wz = angvel.tolist()

    ang = []
    for i in range(len(mass)):
        jx = 3*i
        jy = 3*i+1
        jz = 3*i+2

        ang = np.array([
                wy*q[jz] - wz*q[jy],
                wz*q[jx] - wx*q[jz],
                wx*q[jy] - wy*q[jx]
              ])

        #rotational momentum of the ith atom
        pang = mass[i] * ang

        #add to the vibrational momentum the rotational one
        p[jx] -= pang[0]
        p[jy] -= pang[1]
        p[jz] -= pang[2]


    return p




def polyatom_rotation_sampling(jrot, mass, q, p):
    ai,Iinv = calcI(q, mass)

    #angmom from rotation quantum numbers
    amrot = math.sqrt(jrot * (jrot + 1))

    #set random direction for angular momentum
    theta = random.uniform(0, math.pi)
    phi =  random.uniform(0, 2*math.pi)

    am = np.array([
            amrot*sin(theta)*cos(phi),
            amrot*sin(theta)*sin(phi),
            amrot*cos(theta)
         ])


    #Erot = sum([am[i]**2/ai[i]/2.0 for i in range(len(am))])
    #print("Erot[cm-1] =" , Erot*c5, "  Erot[eV] =", Erot*c4)

    am_corrected = angmom_correction_after_vibrational_sampling(am, q, p)

    #calculate angular velocities (w) from the inverse of intertia tensor
    # I * w = L   --->  I(-1) * L = w
    angvel = -np.matmul(Iinv, np.transpose(am_corrected))

    #Add rotational momentum to the vibrational one
    p = add_rotational_momentum(angvel, mass, q, p)

    return (p, ai, am)
#asd
