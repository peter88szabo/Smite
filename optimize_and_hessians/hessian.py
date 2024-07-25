from pyscf            import gto, dft, hessian
from gradient         import Sparrow_Force
from format_and_print import parseXYZ

import numpy as np
import os

def PySCF_Hessian(Natoms, xyz, charge, multiplicity, functional, base):
    mol = gto.M(
            atom = xyz,
            basis = base,
            charge = charge,
            spin = multiplicity -1,
            verbose=0)

    if multiplicity != 1:
        mf = dft.UKS(mol).run(xc = functional)
    else:
        mf = dft.RKS(mol).run(xc = functional)

    h = mf.Hessian().kernel()
    hess = h.transpose(0,2,1,3).reshape(Natoms*3,Natoms*3)
    return hess

def Sparrow_Hessian(q, atoms, charge, multiplicity, method):
    '''
    Numerical hessian from Sparrow by calling the analytical gradient
    '''

    dx = 0.002
    ndim = len(q)
    hess = np.zeros((ndim, ndim))

    for i in range(ndim):
        q[i] += dx
        gradp1 = -Sparrow_Force(q, atoms, charge, multiplicity, method) #negative sign because it's force not gradient

        q[i] -= 2*dx
        gradm1 = -Sparrow_Force(q, atoms, charge, multiplicity, method)

        hess[i,:] = 0.5 * (gradp1 - gradm1) / dx

        q[i] += dx #restore partial coordinate

    return hess


def getHessian(qchem, hessFile, xyz, charge, multiplicity, functional, base):

    c1 = 0.52917721092  

    Natoms, atoms, qcoord = parseXYZ(xyz)

    qcoord = qcoord / c1 #angtstrom to bohr

    if os.path.exists(hessFile):
        print()
        print("Loading hessian from file")
        print()
        hess = np.loadtxt(hessFile, delimiter=' ')
        if hess.shape == (Natoms*3,Natoms*3):
            return hess

    print("Recalculating Hessian from scratch because the program could not find it in the file: ", hessFile)
    print("or the size of the Hessian matrix is wrong in the given file. Its size must be (Natoms*3,Natoms*3)")

    if qchem == 'PySCF':
        hess = PySCF_Hessian(Natoms, xyz, charge, multiplicity, functional, base)
    elif qchem == 'Sparrow':
        hess = Sparrow_Hessian(qcoord, atoms, charge, multiplicity, functional)
    else:
        raise ValueError("Non-Existing Quantum Chemical method is input. Avaiable packages: PySCF, Sparrow")

    np.savetxt(hessFile, hess, delimiter=' ', newline='\n')


    return hess

