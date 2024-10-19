import numpy as np
import math

def distances(q):
    natoms = len(q) // 3
    d = np.zeros((natoms,natoms))
    for i in range(natoms):
        for j in range(i):
            tx = q[3*j]   - q[3*i]
            ty = q[3*j+1] - q[3*i+1]
            tz = q[3*j+2] - q[3*i+2]

            d[i][j] = math.sqrt(tx*tx + ty*ty + tz*tz)
    return d


def test_to_stop(istep,  q, iatom, jatom, tol):
    bohr_to_angst = 0.52917721092

    

    dist = distances(q)

    if iatom < jatom:
        iatom,jatom = jatom,iatom
    if iatom == jatom:
        raise TypeError("two indices cannot be the same")

    res = False

    tol = tol/bohr_to_angst

    if dist[iatom][jatom] > tol:
        res = True

#    for i in range(natoms):
#        for j in range(i):
#            print(i,j, dist[i][j]*bohr_to_angst)

#    print()
#    print(istep, "i,j: ", iatom, jatom)
#    print("dist[bohr] = ", dist[iatom][jatom], ", dist[Angst] = ", dist[iatom][jatom]* bohr_to_angst, ", tol[Angst] = ", tol*bohr_to_angst)

    return res


