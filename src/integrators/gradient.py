import numpy as np

from qchem_interfaces.orcarun     import Orca_Force, Orca_Energy
from qchem_interfaces.pyscfrun    import PySCF_Force, PySCF_Energy
from qchem_interfaces.sparrowbin  import Sparrowbin_Force, Sparrowbin_Energy
from qchem_interfaces.sparrowpy   import SparrowPy_Force, SparrowPy_Energy
from qchem_interfaces.xtbrun      import XTB_Force, XTB_Energy


def Potential_Energy(qcinput, file_wf, q, atoms):

    qchem = qcinput['qchem']

    if qchem == 'PySCF':
        V = PySCF_Energy(file_wf, q, atoms, qcinput)
    elif qchem == 'Orca':
        V = Orca_Energy(file_wf, q, atoms, qcinput)
    elif qchem == 'Sparrow_Py':
        V = SparrowPy_Energy(file_wf, q, atoms, qcinput)
    elif qchem == 'Sparrow_bin':
        V = Sparrowbin_Energy(file_wf, q, atoms, qcinput)
    elif qchem == 'XTB':
        V = XTB_Energy(file_wf, q, atoms, qcinput)
    elif qchem == 'PES':
        V = PES_Energy(q, atoms)
    else:
        raise ValueError("Non-Existing Quantum Chemical method is input. Avaiable packages: Orca, PySCF, Sparrow, XTB")

    return V

def Energy(qcinput, file_wf, q, p, atoms, wmass):

    qchem = qcinput['qchem']

    if qchem == 'PySCF':
        V = PySCF_Energy(file_wf, q, atoms, qcinput)
    elif qchem == 'Orca':
        V = Orca_Energy(file_wf, q, atoms, qcinput)
    elif qchem == 'Sparrow_Py':
        V = SparrowPy_Energy(file_wf, q, atoms, qcinput) 
    elif qchem == 'Sparrow_bin':
        V = Sparrowbin_Energy(file_wf, q, atoms, qcinput) 
    elif qchem == 'XTB':
        V = XTB_Energy(file_wf, q, atoms, qcinput) 
    elif qchem == 'PES':
        V = PES_Energy(q, atoms) 
    else:
        raise ValueError("Non-Existing Quantum Chemical method is input. Avaiable packages: Orca, PySCF, Sparrow, XTB")

    T = sum(0.5*np.array(p)*np.array(p)/np.array(wmass))
    Etot = T+V
    return(T, V, Etot)


def force_calc(qcinput, q, atoms):
    qchem = qcinput['qchem']

    if qchem == 'PySCF':
        force = PySCF_Force(q, atoms, qcinput) 
    elif qchem == 'Orca':
        force = Orca_Force(q, atoms, qcinput)
    elif qchem == 'Sparrow_Py':
        force = SparrowPy_Force(q, atoms, qcinput)
    elif qchem == 'Sparrow_bin':
        force = Sparrowbin_Force(q, atoms, qcinput)
    elif qchem == 'XTB':
        force = XTB_Force(q, atoms, qcinput)
    elif qchem == 'PES':
        force = PES_Force(q, atoms)
    else:
        raise ValueError("Non-Existing Quantum Chemical method is input. Avaiable packages: Orca, PySCF, Sparrow, XTB")

    return force


