import numpy as np
import math

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

def getNormalmode(mass, hessian):

    wmass = []   # weights by coordinates
    for ww in mass:
        wmass += [ww]*3
    M = np.linalg.matrix_power(np.sqrt(np.diag(wmass)),-1)

    hess_mw = np.matmul(np.matmul(M,hessian),M)
    lambd, L = np.linalg.eigh(hess_mw)

    L = np.array(L[:,6:])
    for i in range(len(mass)):
        for j in range(L.shape[1]):
            L[3*i  ,j] /= math.sqrt(mass[i])
            L[3*i+1,j] /= math.sqrt(mass[i])
            L[3*i+2,j] /= math.sqrt(mass[i])

    ww = []
    ww_low = []
    for i in range(0,6):
        if lambd[i] < 0.0:
            ww_low += [-math.sqrt(abs(lambd[i]))]
        else:
            ww_low += [math.sqrt(lambd[i])]
    for i in range(6,lambd.size):
        if lambd[i] < 0.0:
            ww += [-math.sqrt(abs(lambd[i]))]
        else:
            ww += [math.sqrt(lambd[i])]

    return ww, ww_low, L


def print_frequencies(ww):
    with open("vibrational_freq.dat", "w") as file:
        file.write("----------Low Frequencies----------\n")
        file.write("%6s %6s %12s \n" % ("index1","index2", "freq[cm-1]"))
        for i in range(0,6):
            if ww[i] < 0.0:
                file.write("%6d %6d %12.2f %10s \n" % (i, i-6, ww[i]/c10*c9/(math.pi * 2), " <-- Imag"))
            else:
                file.write("%6d %6d %12.2f \n" % (i, i-6, ww[i]/c10*c9/(math.pi * 2)))
        file.write("----------High Frequencies--------- \n")
        for i in range(6,len(ww)):
            if ww[i] < 0.0:
                file.write("%6d %6d %12.2f %10s \n" % (i, i-6, -ww[i]/c10*c9/(math.pi * 2), " <-- Imag"))
            else:
                file.write("%6d %6d %12.2f \n" % (i, i-6, ww[i]/c10*c9/(math.pi * 2)))

        file.write("-----------------------------------\n")
