import numpy as np
import random
import math


#====================================================
def thermal_collision_energy(RT):
#----------------------------------------------------
#   Maxwell-Boltzmann distribution for the 
#   collision energy (P(Ecoll)) is a
#   second ordered Gamma distribution, that can
#   be sampled with two independent random numbers.
#----------------------------------------------------
    rand1 = random.uniform(0,1)
    rand2 = random.uniform(0,1)

#   collision energy:
    Ecoll = -RT*np.log(rand1*rand2)

    return Ecoll
#====================================================


#====================================================
def thermal_vibr_mode(RT,ome):

    rand = random.uniform(0,1)

    vib = -RT*np.log(1.0-rand)/abs(ome)

#   harmonic oscillator quantum number:
    nvib = int(vib)

    return nvib
#====================================================

#====================================================
def thermal_rot_quantum_spherical_top(RT,Inertia):
#----------------------------------------------------
#   it's useful for spherical top or diatomic cases
#
#   for diatomic molecule:
#   Inertia = redmass*re*re
#
#   jrot here is the rotational quantum number
#----------------------------------------------------
    rand = random.uniform(0,1)

    alp = np.log(1.0-rand)*2.0*Inertia*RT

    roty = 0.5*(np.sqrt(1.0-4.0*alp)-1.0)

#   diatomic rotational quantum number:
    jrot = int(round(roty))

    return jrot
#====================================================


#====================================================
def thermal_rot_classic_spherical_top(RT,Inertia):
#----------------------------------------------------
#   it's useful for spherical top or diatomic cases
#
#   for diatomic molecule:
#   Inertia = redmass*re*re
#
#   according to 
#   E. Martinez-Nunez et al:
#   J. Phys. Chem. A 2005, 109, 5415-5423
# 
#   jrot is the classical angular momentum
#---------------------------------------------------
    rand = random.uniform(0,1)

#   symmetric top angualar momentum:
    jrot = np.sqrt(-2.0*Inertia*RT*np.log(1-rand))

    return jrot
#====================================================


#====================================================
def thermal_rot_asymmetric_top_equipart(RT,Ixyz):
#----------------------------------------------------
#   Each rotational axis has  RT/2 energy
#   Lx^2/(2Ix)=RT/2 --> Lx
#   Ly^2/(2Iy)=RT/2 --> Ly
#   Lz^2/(2Iz)=RT/2 --> Lz
#---------------------------------------------------
    Lrot = np.zeros(3)

    for i in range(len(Ixyz)):

        # for linear molecules one axis is skipped
        if Ixyz[i] < 10**10 :    

            Lrot[i] = np.sqrt(RT*abs(Ixyz[i]))

            rand = random.uniform(0,1)

            # random sign
            if rand < 0.5: Lrot[i] = -Lrot[i]

    return Lrot
#====================================================


#====================================================
def thermal_rot_symmetric_top(RT,Ixyz):
#----------------------------------------------------
#    oblate:  Ix = Iy < Iz  (plate shaped) 
#       or    
#    prolate: Ix < Iy = Iz  (cigar shaped)
#----------------------------------------------------
    Lrot = np.zeros(3)

    rand = np.random.normal(0.0,sigma)

    dd1 = abs(Ixyz[1]-Ixyz[0])
    dd2 = abs(Ixyz[2]-Ixyz[1])

    if dd1 < dd2 : top="oblate"
    if dd1 > dd2 : top="prolate"

    ##########################################################

    if top == "oblate" :
        #------------------------------------------
        # Lz obeys the following distribution:
        # P(Lz)=exp(-Lz^2/(2*Iz*RT)) where -inf<Lz<inf
        # This is a Gaussian distribution
        # thats variance is sigma**2 = Iz*RT
        # and its mean mu = 0.0
        #------------------------------------------

        # random number with normal distribution
        # thats variance is 1.0 and mean is 0.0
        rnd_gauss = np.random.randn()

        # therefore, Lz component can obtain as
        Lrot[2] = np.sqrt(Ixyz[2]*RT)*rnd_gauss

        #------------------------------------------
        # L=Ltot obeys the following distiribution:
        # L*exp(L^2/(2*Iavg*RT)) where Lz < L < inf
        # (remark: Ix=Iy < Iz)
        # 
        # where Iavg is the average inertia
        # because sometimes the molecule is not 
        # perfectly symmetric: Ix != Iy
        #
        # Iavg = sqrt(Ix*Iy)
        #------------------------------------------
        Iavg = np.sqrt(abs(Ixyz[0]*Ixyz[1]))

        rand = random.uniform(0,1)
        L = np.sqrt(Lrot[2]**2 - 2.0*Iavg*RT*np.log(1-rand))


        rand = random.uniform(0,1)
        angle = random.uniform(0,2*math.pi)

        Lrot[0] = np.sqrt(L**2 - Lrot[2]**2)*np.cos(angle)
        Lrot[1] = np.sqrt(L**2 - Lrot[2]**2)*np.sin(angle)

    ################################################

    if top == "prolate" :
        #------------------------------------------
        # Lx obeys the following distribution:
        # P(Lx)=exp(-Lx^2/(2*Ix*RT)) where -inf<Lx<inf
        # This is a Gaussian distribution
        # thats variance is sigma**2 = Ix*RT
        # and its mean mu = 0.0
        #------------------------------------------

        # random number with normal distribution
        # thats variance is 1.0 and mean is 0.0
        rnd_gauss = np.random.randn()

        # therefore, Lx component can obtain as
        Lrot[0] = np.sqrt(Ixyz[0]*RT)*rnd_gauss

        #------------------------------------------
        # L=Ltot obeys the following distiribution:
        # L*exp(L^2/(2*Iavg*RT)) where Lx < L < inf
        # (remark: Ix < Iy = Iz)
        #
        # where Iavg is the average inertia
        # because sometimes the molecule is not
        # perfectly symmetric: Iy != Iz
        #
        # Iavg = sqrt(Iy*Iz)
        #------------------------------------------
        Iavg = np.sqrt(abs(Ixyz[1]*Ixyz[2]))

        rand = random.uniform(0,1)
        L = np.sqrt(Lrot[0]**2 - 2.0*Iavg*RT*np.log(1-rand))

        rand = random.uniform(0,1)
        angle = random.uniform(0,2*math.pi)

        Lrot[2] = np.sqrt(L**2 - Lrot[0]**2)*np.cos(angle)
        Lrot[1] = np.sqrt(L**2 - Lrot[0]**2)*np.sin(angle)
    ##########################################################


    return Lrot
#====================================================

#s = np.random.normal(mu, sigma)


def partfu_vibr(RT,ome):

    alpha = ome/RT
    pf = np.exp(-alpha/2.0)/(1.0-np.exp(-alpha))

    return pf

#=========================================================================
#T=300.0
#R=8.314/1000.0
#RT=R*T

#redmass=1.0
#ome=1.0
#re=1.0
#Inertia=1.0/(redmass*re*re)

#nmax=20000
#seed=12345
#random.seed(seed)

#for i in range(nmax):
#    Ecoll = thermal_collision_energy(RT)
#    nvib  = thermal_vibr_mode(RT,ome)
#    jrot  = thermal_rot_quantum_spherical_top(RT,Inertia)
#    Jstop = thermal_rot_classic_spherical_top(RT,Inertia)

#    print(Ecoll,nvib,jrot,Jstop)


#pf=partfu_vibr(RT,ome)
#for i in range(10):
#    prop = np.exp(-ome*(i+0.5)/RT)/pf

#    print i,prop

#E0=0.0
#for i in range(1000):
#    E=E0+i*0.03

#    prop=(1/RT**1.5)*np.sqrt(E/3.14159)*np.exp(-E/RT)*2.0
#    print E,prop

# print "RT:",RT
