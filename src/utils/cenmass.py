import numpy as np


def cenmass(q, p, w):
    vcm = np.zeros(3)
    qcm = np.zeros(3)

   #deterime the center of mass
    for i in range(len(w)):
        vcm[0] += p[3 * i]
        vcm[1] += p[3 * i + 1]
        vcm[2] += p[3 * i + 2]

        qcm[0] += q[3 * i]     * w[i]
        qcm[1] += q[3 * i + 1] * w[i]
        qcm[2] += q[3 * i + 2] * w[i]

    vcm /= sum(w)
    qcm /= sum(w)

   #shif the molecule to its center of mass
    for i in range(len(w)):
        p[3 * i]     -= vcm[0] * w[i]
        p[3 * i + 1] -= vcm[1] * w[i]
        p[3 * i + 2] -= vcm[2] * w[i]

        q[3 * i]     -= qcm[0]
        q[3 * i + 1] -= qcm[1]
        q[3 * i + 2] -= qcm[2]

    return q, p



def cenmassQ(q, w):
    qcm = np.zeros(3)

   #deterime the center of mass
    for i in range(len(w)):
        qcm[0] += q[3 * i]     * w[i]
        qcm[1] += q[3 * i + 1] * w[i]
        qcm[2] += q[3 * i + 2] * w[i]

    qcm /= sum(w)

   #shif the molecule to its center of mass
    for i in range(len(w)):
        q[3 * i]     -= qcm[0]
        q[3 * i + 1] -= qcm[1]
        q[3 * i + 2] -= qcm[2]

    return q

