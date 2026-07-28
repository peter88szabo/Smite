import numpy as np
import math
import random
from utils.cenmass import cenmass

def setRelativeInitCoords(massA, massB, qA, qB, pA, pB, Ecoll=0.1, Ecoll_thermal=False, bmax=8.0, bsampling=False, temp=300.0, Rini=18.0):
    if bsampling:
        bimp = bmax * math.sqrt(random.uniform(0.0,1.0))
    else:
        bimp = bmax #fix impact parameter for opacity function P(b) calculations
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

    return(bimp, q, p)
