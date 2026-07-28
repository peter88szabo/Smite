from qchem_interfaces.psi4run      import Psi4_Hessian
from qchem_interfaces.orcarun      import Orca_Hessian
from qchem_interfaces.sparrowbin   import Sparrowbin_Force, Sparrowbin_Energy
from qchem_interfaces.pyscfrun     import PySCF_Hessian
from qchem_interfaces.sparrowpy    import SparrowPy_Hessian
from qchem_interfaces.sparrowbin   import Sparrowbin_Hessian
from qchem_interfaces.xtbrun       import XTB_Hessian
from qchem_interfaces.pesrun       import PES_Hessian
from qchem_interfaces.qchem_validation import validate_qchem_input

from utils.format_and_print import parseXYZ
from utils.constants import ANGSTROM_TO_BOHR

import numpy as np
import os


def getHessian(qcinput, hessFile, xyz):
    qcinput = validate_qchem_input(qcinput)

    Natoms, atoms, qcoord = parseXYZ(xyz)

    qcoord = qcoord * ANGSTROM_TO_BOHR #angtstrom to bohr

    qchem = qcinput['qchem'] 

    force_recalc = bool(qcinput.get("force_hessian_recalc", False))
    quiet = bool(qcinput.get("quiet_hessian", False))
    save_hessian = bool(qcinput.get("save_hessian", True))

    if os.path.exists(hessFile) and not force_recalc:
        if not quiet:
            print(f"\nLoading hessian from file\n")
        hess = np.loadtxt(hessFile, delimiter=' ')
        if hess.shape == (Natoms*3,Natoms*3):
            return hess
        else: 
            if not quiet:
                print("\nSize of the Hessian matrix in the file is wrong. It must be (3*Natoms, 3*Natoms)\n")
    elif os.path.exists(hessFile) and force_recalc:
        if not quiet:
            print(f"\nIgnoring existing Hessian file because force_hessian_recalc=True: {hessFile}")


    if not quiet:
        print(f"\nThe program could not find Hessian in the file: {hessFile}")
        print(f"Recalculating Hessian from scratch\n")

    if qchem == 'PySCF':
        hess = PySCF_Hessian(Natoms, xyz, qcinput)
    elif qchem == 'Psi4':
        hess = Psi4_Hessian(qcoord, atoms, qcinput)
    elif qchem == 'Orca':
        hess = Orca_Hessian(Natoms, xyz, qcinput)
    elif qchem == 'Sparrow_Py':
        hess = SparrowPy_Hessian(qcoord, atoms, qcinput)
    elif qchem == 'Sparrow_bin':
        hess = Sparrowbin_Hessian(qcoord, atoms, qcinput)
    elif qchem == 'XTB':
        hess = XTB_Hessian(qcoord, atoms, qcinput)
    elif qchem == 'PES':
        hess = PES_Hessian(qcoord, atoms, qcinput)
    else:
        raise ValueError("Non-Existing Quantum Chemistry interface in input. Avaiable packages: Orca, PySCF, Sparrow_Py, Sparrow_bin, XTB")

    if save_hessian:
        np.savetxt(hessFile, hess, delimiter=' ', newline='\n')


    return hess
