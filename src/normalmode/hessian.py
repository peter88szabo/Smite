from qchem_interfaces.orcarun      import Orca_Hessian
from qchem_interfaces.sparrowbin   import Sparrowbin_Force, Sparrowbin_Energy
from qchem_interfaces.pyscfrun     import PySCF_Hessian
from qchem_interfaces.sparrowpy    import SparrowPy_Hessian
from qchem_interfaces.sparrowbin   import Sparrowbin_Hessian
from qchem_interfaces.xtbrun       import XTB_Hessian

from utils.format_and_print import parseXYZ

import numpy as np
import os


def getHessian(qcinput, hessFile, xyz):

    Natoms, atoms, qcoord = parseXYZ(xyz)

    qcoord = qcoord / 0.52917721092 #angtstrom to bohr

    qchem = qcinput['qchem'] 

    if os.path.exists(hessFile):
        print(f"\nLoading hessian from file\n")
        hess = np.loadtxt(hessFile, delimiter=' ')
        if hess.shape == (Natoms*3,Natoms*3):
            return hess
        else: 
            print("\nSize of the Hessian matrix in the file is wrong. It must be (3*Natoms, 3*Natoms)\n")


    print(f"\nThe program could not find Hessian in the file: {hessFile}")
    print(f"Recalculating Hessian from scratch\n")

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
    elif qchem == 'PES':
        hess = PES_Hessian(qcoord)
    else:
        raise ValueError("Non-Existing Quantum Chemistry interface in input. Avaiable packages: Orca, PySCF, Sparrow_Py, Sparrow_bin, XTB")

    np.savetxt(hessFile, hess, delimiter=' ', newline='\n')


    return hess

