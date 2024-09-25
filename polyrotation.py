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
    ai, dontneed = np.linalg.eigh(I)
    Iinv = np.linalg.inv(I)
    return (ai, Iinv)



def angmomcorr(qq, pp):
    am = [0.0,0.0,0.0]
    for i in range(len(qq)//3):
        jx = 3*i
        jy = 3*i+1
        jz = 3*i+2
        am[0] += qq[jy]*pp[jz] - qq[jz]*pp[jy]
        am[1] += qq[jz]*pp[jx] - qq[jx]*pp[jz]
        am[2] += qq[jx]*pp[jy] - qq[jy]*pp[jx]
    return am

def poly_rotation_init(jrot, q_eq, mass, q, p):
    ai,Iinv = calcI(q_eq, mass)

    amjr = math.sqrt(jrot * (jrot + 1))

  #set random direction for angular momentum

    theta = random.uniform(0, math.pi)
    phi =  random.uniform(0, 2 * math.pi)

    am = np.array([
            amjr*sin(theta)*cos(phi),
            amjr*sin(theta)*sin(phi),
            amjr*cos(theta)
         ])


    Erot = sum([am[i]**2/ai[i]/2.0 for i in range(len(am))])

    #print("Erot[cm-1] =" , Erot*c5, "  Erot[eV] =", Erot*c4)

    #print('angmom before corr: ', am)

    am -= np.array(angmomcorr(q,p))

    #print('angmom after corr: ', am)
    #print()

    #calculate angular velocities

    angvel = -np.matmul(Iinv, np.transpose(am))

    wx,wy,wz = angvel.tolist()

    ang = []
    for i in range(len(q)//3):
        jx = 3*i
        jy = 3*i+1
        jz = 3*i+2

        ang = np.array([
                wy*q[jz] - wz*q[jy],
                wz*q[jx] - wx*q[jz],
                wx*q[jy] - wy*q[jx]
              ])

        pang = mass[i] * ang

        p[jx] -= pang[0]
        p[jy] -= pang[1]
        p[jz] -= pang[2]
    return (q,p, ai, am)
#asd
