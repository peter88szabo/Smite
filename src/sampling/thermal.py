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
def thermal_rot_quantum_spherical_top(RT, Inertia, tail_tolerance=1.0e-12):
    """Sample an exact canonical rotational quantum number for a diatomic rotor.

    The discrete rigid-rotor population is

    ``P(J) ∝ (2J + 1) exp[-J(J + 1) / (2 I RT)]``.

    The previous implementation sampled a continuous approximation and rounded
    it, which biases low-J populations.  We instead construct the normalized
    discrete distribution until its remaining tail is negligible, then draw
    from its cumulative weights.  ``Inertia`` and ``RT`` are in atomic units.
    """
    RT = float(RT)
    Inertia = float(Inertia)
    tail_tolerance = float(tail_tolerance)
    if not np.isfinite(RT) or RT < 0.0:
        raise ValueError("RT must be finite and non-negative")
    if not np.isfinite(Inertia) or Inertia <= 0.0:
        raise ValueError("Inertia must be finite and positive for quantum rotational sampling")
    if not 0.0 < tail_tolerance < 1.0:
        raise ValueError("tail_tolerance must lie between 0 and 1")
    if RT == 0.0:
        return 0

    inertia_temperature = Inertia * RT
    weights = [1.0]
    total = 1.0
    jrot = 0
    while True:
        # w(J+1)/w(J), obtained directly from the degeneracy-weighted
        # canonical probability to avoid evaluating large exponentials.
        ratio = ((2.0 * jrot + 3.0) / (2.0 * jrot + 1.0)) * math.exp(
            -(jrot + 1.0) / inertia_temperature
        )
        next_weight = weights[-1] * ratio
        weights.append(next_weight)
        total += next_weight
        jrot += 1

        # The ratios decrease after the distribution mode.  Bound the
        # unrepresented geometric tail before stopping.
        next_ratio = ((2.0 * jrot + 3.0) / (2.0 * jrot + 1.0)) * math.exp(
            -(jrot + 1.0) / inertia_temperature
        )
        if next_ratio < 1.0:
            tail_bound = next_weight * next_ratio / (1.0 - next_ratio)
            if tail_bound <= tail_tolerance * total:
                break

    threshold = random.random() * total
    cumulative = 0.0
    for jrot, weight in enumerate(weights):
        cumulative += weight
        if threshold < cumulative:
            return jrot
    return len(weights) - 1  # protects against roundoff at the CDF endpoint
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
#
# Linear case is missing
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
def thermal_rot_canonical_top(RT, Ixyz, zero_moment_tolerance=1.0e-12):
    """Sample body-frame angular momentum from the canonical rigid-top law.

    For principal moments ``I_i``, the rotational Hamiltonian is
    ``sum_i L_i**2/(2 I_i)``. Consequently each active component is an
    independent normal variate with variance ``I_i * RT``. This expression is
    valid for asymmetric, symmetric, and spherical tops. A zero principal
    moment (the molecular axis of a linear rotor) has no associated rotation
    and is returned as exactly zero.
    """
    moments = np.asarray(Ixyz, dtype=float)
    if moments.shape != (3,):
        raise ValueError("Ixyz must contain exactly three principal moments")
    if not np.isfinite(RT) or RT < 0.0:
        raise ValueError("RT must be a finite, non-negative thermal energy")
    if np.any(~np.isfinite(moments)):
        raise ValueError("Principal moments must be finite")
    if np.any(moments < -zero_moment_tolerance):
        raise ValueError("Principal moments cannot be negative")

    moments = np.where(moments <= zero_moment_tolerance, 0.0, moments)
    Lrot = np.zeros(3)
    for i, moment in enumerate(moments):
        if moment > 0.0 and RT > 0.0:
            Lrot[i] = random.gauss(0.0, math.sqrt(moment * RT))
    return Lrot


def thermal_rot_symmetric_top(RT, Ixyz):
    """Backward-compatible name for the canonical general-top sampler."""
    return thermal_rot_canonical_top(RT, Ixyz)




'''
import random
import numpy as np
import matplotlib.pyplot as plt

def thermal_rot_symmetric_top(RT, Ixyz):
    Lrot = np.zeros(3)
    
    # Sample the Lz component
    rnd_gauss = random.gauss(0.0, 1.0)
    Lrot[2] = np.sqrt(Ixyz[2] * RT) * rnd_gauss
    
    return Lrot

RT = 1.0
Ixyz = [2.0, 2.0, 3.0]
num_samples = 1000

samples_Lz = [thermal_rot_symmetric_top(RT, Ixyz)[2] for _ in range(num_samples)]

# Generate Lz values for the theoretical distribution
Lz_vals = np.linspace(min(samples_Lz), max(samples_Lz), 500)
Iz = Ixyz[2]
P_Lz = np.exp(-Lz_vals**2 / (2 * Iz * RT)) / np.sqrt(2 * np.pi * Iz * RT)  # Normalized Gaussian

plt.figure(figsize=(8, 5))
plt.hist(samples_Lz, bins=30, density=True, alpha=0.6, color='skyblue', label='Sampled Lz Histogram')
plt.plot(Lz_vals, P_Lz, 'r-', linewidth=2, label='Theoretical Distribution (P(Lz))')

plt.xlabel('Lz')
plt.ylabel('Probability Density')
plt.title('Comparison of Sampled and Theoretical Distributions of Lz')
plt.legend()
plt.show()
'''



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
