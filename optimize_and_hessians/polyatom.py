import numpy as np
import random
import math
from cenmass import cenmass
from euler import euler_rot
from thermal import thermal_vibr_mode  
from normalmode import getNormalmode, print_frequencies
from polyrotation import poly_rotation_init

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

def polyatom_init(RT, jrot, fix_vib, mass, q_eq, hessian):

    ww, ww_low, L = getNormalmode(mass, hessian)

    ww_all = np.append(ww_low, ww)
    print_frequencies(ww_all)


    print()
    print("++++++Thermal sampled vib modes++++++")
    print("%6s %6s %12s %8s" % ("index1","index2", "freq[cm-1]", "    nvib"))
    nvib = []
    for i in range(len(ww)): 
        if i == fix_vib[0]:
            nv = fix_vib[1] #value of fixed quantum number
            nvib += [nv]
            print("%6d %6d %12.1f %8d %13s" % (i+6, i, ww[i]/c10*c9/(math.pi * 2), nv, " <-- fix mode"))
        else:
            nv = thermal_vibr_mode(RT,ww[i])
            nvib += [nv]
            print("%6d %6d %12.1f %8d" % (i+6, i, ww[i]/c10*c9/(math.pi * 2), nv))
    print("---------------------------------------")


    energy = [ww[i]*(nvib[i] + 0.5) for i in range(len(ww))]
    ampl = [math.sqrt(2.0 * energy[i])/ww[i] for i in range(len(ww))]

    print()
    print('ZPE[Hartree] = ', 0.5*sum(np.array(ww)))
    #print('energy eV', np.array(energy)*c4)
    print()

    q_norm = [a * math.cos(random.uniform(0, 2 * math.pi)) for a in ampl]
    p_norm = [- ampl[i] * ww[i] * math.sin(random.uniform(0, 2 * math.pi)) for i in range(len(ampl))]

    L_tr = np.transpose(L)

    q_desc = np.transpose(q_eq) + np.matmul(L, np.transpose(np.array(q_norm)))
    p_desc = np.matmul(L, np.transpose(p_norm))

    q,p = cenmass(q_desc, p_desc, mass)


    q,p = poly_rotation_init(jrot, q_eq, mass, q, p)

    # Randomly roteate the molecule about its center of mass
    #q,p = euler_rot(q, p)
    return(q,p, ww_all, nvib)
