from format_and_print import parseXYZ
from orcarun          import Orca_Hessian
from sparrowbin       import Sparrowbin_Force, Sparrowbin_Energy
from pyscfrun         import PySCF_Hessian
from sparrowpy        import SparrowPy_Hessian
from sparrowbin       import Sparrowbin_Hessian
from xtbrun           import XTB_Hessian

import numpy as np
import os


def getHessian(qcinput, hessFile, xyz):

    Natoms, atoms, qcoord = parseXYZ(xyz)

    qcoord = qcoord / 0.52917721092 #angtstrom to bohr

    qchem = qcinput[0] 

    if os.path.exists(hessFile):
        print()
        print("Loading hessian from file")
        print()
        hess = np.loadtxt(hessFile, delimiter=' ')
        if hess.shape == (Natoms*3,Natoms*3):
            return hess
        else: 
            print("\nSize of the Hessian matrix in the file is wrong. It must be (3*Natoms, 3*Natoms)\n")


    print("\nRecalculating Hessian from scratch. The program could not find Hessian in the file: ", hessFile, "\n")

    if qchem == 'PySCF':
        hess = PySCF_Hessian(Natoms, xyz, qcinput)
    elif qchem == 'Orca':
        hess = Orca_Hessian(Natoms, xyz, qcinput)
    elif qchem == 'Sparrow_Py':
        hess = SparrowPy_Hessian(qcoord, atoms, qcinput)
    elif qchem == 'Sparrow_bin':
        hess = Sparrowbin_Hessian(qcoord, atoms, qcinput)
    elif qchem == 'XTB':
        hess = XTB_Hessian(qcoord, atoms, qcinput)
    else:
        raise ValueError("Non-Existing Quantum Chemical method in input. Avaiable packages: Orca, PySCF, Sparrow_Py, Sparrow_bin, XTB")

    np.savetxt(hessFile, hess, delimiter=' ', newline='\n')


    return hess

