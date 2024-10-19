import numpy as np
import math
import random
from cenmass import cenmass

def setRelativeInitCoords(Ecoll, bmax, Rini, atomA, atomB, massA, massB, qA, qB, pA, pB):
    bimp = bmax * math.sqrt(random.uniform(0.0,1.0))
    sepx = math.sqrt(Rini*Rini - bimp*bimp)


   #shif the A and B molecule to their center of mass:
    qA, pA = cenmass(qA, pA, massA)
    qB, pB = cenmass(qB, pB, massB)

    for i in range(len(massB)):
        jx = 3 * i
        jz = 3 * i + 2
        qB[jx] += sepx
        qB[jz] += bimp

    wA = sum(massA)
    wB = sum(massB)

    redmass = wA*wB/(wA+wB)

    velRel = math.sqrt(2.0*Ecoll/redmass)
    velA = velRel*wB / (wA+wB)
    velB = velA - velRel

   #it has only velocity along the Z-axis
    for i in range(len(massA)):
        pA[3 * i] += velA * massA[i]

    for i in range(len(massB)):
        pB[3 * i] += velB * massB[i]

    q = np.append(qA, qB)
    p = np.append(pA, pB)
    mass = np.append(massA, massB)
    atoms = atomA + atomB #this is just simple array, not numpy

    return(q,p, atoms, mass, bimp)
