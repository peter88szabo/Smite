import numpy as np
import math
from eckart         import eckart_transform

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

def eigval_to_freq(lambd):

    if lambd < 0.0:
        omega = -math.sqrt(abs(lambd))
    else:
        omega = math.sqrt(lambd)
    return omega


def getNormalmode(mass, hessian, linear=False, **kwargs):

    is_eckart = kwargs.get('is_eckart', True)
    if is_eckart:
        q_eq = kwargs.get('q_eq')

    nlow = 6 - int(linear)
    nmode = 3*len(mass) - nlow

    wmass = np.repeat(mass, 3)  # weights by coordinates

    M = np.diag(1.0 / np.sqrt(wmass))

   #mass-weighted Hessian:
    hess_mw = np.matmul(np.matmul(M,hessian),M)

   #projecting out the translation and rotation
    if is_eckart: 
        hess_mw = eckart_transform(mass, q_eq, hess_mw)

    lambd, Lraw = np.linalg.eigh(hess_mw)

    #Lraw = np.array(L[:,6:])
    Lraw = np.array(Lraw)
    for i in range(len(mass)):
        for j in range(Lraw.shape[1]):
            Lraw[3*i  ,j] /= math.sqrt(mass[i])
            Lraw[3*i+1,j] /= math.sqrt(mass[i])
            Lraw[3*i+2,j] /= math.sqrt(mass[i])



   # Classify eigenvalues and locate their indices
    tolerance = (0.01*c10/c9*(math.pi * 2))**2
    negative_eigval_ind = []
    zero_eigval_ind = []
    positive_eigval_ind = []

    print(f"Frequencies before ordering, right after diag(hessian):")
    for idx, eigenvalue in enumerate(lambd):
        gr =  eigval_to_freq(eigenvalue) / c10 * c9 / (math.pi * 2)
        fr =  round(gr, 2)
        print(f"{idx:5d} {fr:12.2f}")
        if eigenvalue < 0 and abs(eigenvalue) > tolerance:
            negative_eigval_ind.append(idx)
        elif abs(eigenvalue) <= tolerance:
            zero_eigval_ind.append(idx)
        else:
            positive_eigval_ind.append(idx)
    print()
    print(f"Indices of negative eigenvalues: {negative_eigval_ind}")
    print(f"Indices of zero eigenvalues (within tolerance): {zero_eigval_ind}")
    print(f"Indices of positive eigenvalues: {positive_eigval_ind}\n")

    ww = []
    ww_low = []
    Lfilter = []

    #in case of real equilibrium or TS structure we have 6 zero eigenvalues:
    if len(zero_eigval_ind) == 6:
        for i in range(lambd.size):
            if i in zero_eigval_ind:  # If the eigenvalue is near zero
                ww_low.append(eigval_to_freq(lambd[i]))
            else:  # For non-zero eigenvalues
                Lfilter.append(Lraw[:, i])  
                ww.append(eigval_to_freq(lambd[i]))
        print(f"\nHessian has an optimal structure:")
        print(f"Found 6 zero eigenvalue of Hessian. Number of zero freqs (eigvals): {len(zero_eigval_ind)}")
        print(f"Normal modes are defined by the eigenvectors of non-zero eigenvalues.")
        print(f"(In case of TS structure, the largest negative eigval/eigvect is kept)\n")
    else:
    #however, sometimes when the structure is distorted we do not have necessarly 6 zero eigenvalues:
        print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   Warning   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   ")
        print(f"Less than 6 zero eigenvalue of Hessian. Number of zero freqs (eigvals): {len(zero_eigval_ind)}")
        print(f"Normal modes are defined by discarding the eigenvectors of the six lowest eigenvalues.")
        print(f"(In case of TS structure, the largest negative eigval/eigvect is kept)")
        print(f"Check the nature of the discarded normalmodes.\n")
        for i in range(0, 6):
            ww_low.append(eigval_to_freq(lambd[i]))
        for i in range(6, lambd.size):
            Lfilter.append(Lraw[:, i])  
            ww.append(eigval_to_freq(lambd[i]))

    Lfilter = np.array(Lfilter).T  # Transpose to match the shape of original Lraw

    return ww, ww_low, Lfilter


def print_frequencies(fname, ww):

    print()
    print(f"---------- Low Frequencies ----------")
    print(f"%6s %6s %12s" % ("index1","index2", "freq[cm-1]"))
    for i in range(0,6):
        if ww[i] < 0.0 and abs(ww[i])>0.01*c10/c9*(math.pi * 2):
            print(f"%6d %6d %12.2f %10s" % (i, i-6, ww[i]/c10*c9/(math.pi * 2), " <-- Imag"))
        else:
            print(f"%6d %6d %12.2f" % (i, i-6, ww[i]/c10*c9/(math.pi * 2)))
    print(f"---------- High Frequencies ---------")
    for i in range(6,len(ww)):
        if ww[i] < 0.0:
            print(f"%6d %6d %12.2f %10s" % (i, i-6, ww[i]/c10*c9/(math.pi * 2), " <-- Imag"))
        else:
            print(f"%6d %6d %12.2f" % (i, i-6, ww[i]/c10*c9/(math.pi * 2)))
    print(f"-----------------------------------\n")


    filename = "vibrational_freq_" + fname + ".dat"

    with open(filename, "w") as file:
        file.write("---------- Low Frequencies ----------\n")
        file.write("%6s %6s %12s \n" % ("index1","index2", "freq[cm-1]"))
        for i in range(0,6):
            if ww[i] < 0.0 and abs(ww[i])>0.01*c10/c9*(math.pi * 2):
                file.write("%6d %6d %12.2f %10s \n" % (i, i-6, ww[i]/c10*c9/(math.pi * 2), " <-- Imag"))
            else:
                file.write("%6d %6d %12.2f \n" % (i, i-6, ww[i]/c10*c9/(math.pi * 2)))
        file.write("---------- High Frequencies --------- \n")
        for i in range(6,len(ww)):
            if ww[i] < 0.0:
                file.write("%6d %6d %12.2f %10s \n" % (i, i-6, ww[i]/c10*c9/(math.pi * 2), " <-- Imag"))
            else:
                file.write("%6d %6d %12.2f \n" % (i, i-6, ww[i]/c10*c9/(math.pi * 2)))

        file.write("-----------------------------------\n")
