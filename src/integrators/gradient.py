import numpy as np
from numpy._typing import NDArray

from qchem_interfaces.psi4run     import Psi4_Force, Psi4_Energy
from qchem_interfaces.orcarun     import Orca_Force, Orca_Energy
from qchem_interfaces.pyscfrun    import PySCF_Force, PySCF_Energy
from qchem_interfaces.sparrowbin  import Sparrowbin_Force, Sparrowbin_Energy
from qchem_interfaces.sparrowpy   import SparrowPy_Force, SparrowPy_Energy
from qchem_interfaces.xtbrun      import XTB_Force, XTB_Energy
<<<<<<< HEAD
from qchem_interfaces.gp_pes      import PES_Force, PES_Energy

def Potential_Energy(qcinput, file_wf, q, atoms, active_state):
=======
from qchem_interfaces.pesrun      import PES_Force, PES_Energy
from qchem_interfaces.energy_cache import cached_energy, cached_force
from qchem_interfaces.qchem_validation import validate_qchem_input


def _can_use_energy_cache(qcinput, file_wf):
    return not qcinput.get('wfu', False) and file_wf is None


def Potential_Energy(qcinput, file_wf, q, atoms):

    qcinput = validate_qchem_input(qcinput)
>>>>>>> ecad59b (2026)
    qchem = qcinput['qchem']
    if _can_use_energy_cache(qcinput, file_wf):
        V = cached_energy(qcinput, q, atoms)
        if V is not None:
            return V

    if qchem == 'PySCF':
        V = PySCF_Energy(file_wf, q, atoms, qcinput)
    elif qchem == 'Psi4':
        V = Psi4_Energy(file_wf, q, atoms, qcinput)
    elif qchem == 'Orca':
        V = Orca_Energy(file_wf, q, atoms, qcinput)
    elif qchem == 'Sparrow_Py':
        V = SparrowPy_Energy(file_wf, q, atoms, qcinput)
    elif qchem == 'Sparrow_bin':
        V = Sparrowbin_Energy(file_wf, q, atoms, qcinput)
    elif qchem == 'XTB':
        V = XTB_Energy(file_wf, q, atoms, qcinput)
    elif qchem == 'PES':
<<<<<<< HEAD
        V = PES_Energy(q, active_state)
=======
        V = PES_Energy(file_wf, q, atoms, qcinput)
>>>>>>> ecad59b (2026)
    else:
        raise ValueError("Non-Existing Quantum Chemical method is input. Avaiable packages: Orca, PySCF, Psi4, Sparrow, XTB")

    return V

<<<<<<< HEAD
def Energy(qcinput: dict, file_wf: str, q: NDArray, p: NDArray, atoms: list[str], wmass: NDArray, active_state: int) -> tuple[float,float,float]:
    V = Potential_Energy(qcinput, file_wf, q, atoms, active_state)
=======
def Energy(qcinput, file_wf, q, p, atoms, wmass):

    qcinput = validate_qchem_input(qcinput)
    qchem = qcinput['qchem']
    if _can_use_energy_cache(qcinput, file_wf):
        V = cached_energy(qcinput, q, atoms)
        if V is not None:
            T = sum(0.5*np.array(p)*np.array(p)/np.array(wmass))
            Etot = T+V
            return(T, V, Etot)

    if qchem == 'PySCF':
        V = PySCF_Energy(file_wf, q, atoms, qcinput)
    elif qchem == 'Psi4':
        V = Psi4_Energy(file_wf, q, atoms, qcinput)
    elif qchem == 'Orca':
        V = Orca_Energy(file_wf, q, atoms, qcinput)
    elif qchem == 'Sparrow_Py':
        V = SparrowPy_Energy(file_wf, q, atoms, qcinput) 
    elif qchem == 'Sparrow_bin':
        V = Sparrowbin_Energy(file_wf, q, atoms, qcinput) 
    elif qchem == 'XTB':
        V = XTB_Energy(file_wf, q, atoms, qcinput) 
    elif qchem == 'PES':
        V = PES_Energy(file_wf, q, atoms, qcinput)
    else:
        raise ValueError("Non-Existing Quantum Chemical method is input. Avaiable packages: Orca, PySCF, Psi4, Sparrow, XTB")

>>>>>>> ecad59b (2026)
    T = sum(0.5*np.array(p)*np.array(p)/np.array(wmass))
    Etot = T+V
    return T, V, Etot


<<<<<<< HEAD
def force_calc(qcinput: dict, q: NDArray, atoms: list[str], active_state: int) -> NDArray:
=======
def force_calc(qcinput, q, atoms):
    qcinput = validate_qchem_input(qcinput)
>>>>>>> ecad59b (2026)
    qchem = qcinput['qchem']
    force = cached_force(qcinput, q, atoms)
    if force is not None:
        return force

    # Keep this as one if/elif chain. A standalone `if` here makes the later
    # final `else` execute for PySCF as well, so MD force evaluation aborts
    # even after the PySCF force has already been computed.
    if qchem == 'PySCF':
        force = PySCF_Force(q, atoms, qcinput) 
    elif qchem == 'Psi4':
        force = Psi4_Force(q, atoms, qcinput)
    elif qchem == 'Orca':
        force = Orca_Force(q, atoms, qcinput)
    elif qchem == 'Sparrow_Py':
        force = SparrowPy_Force(q, atoms, qcinput)
    elif qchem == 'Sparrow_bin':
        force = Sparrowbin_Force(q, atoms, qcinput)
    elif qchem == 'XTB':
        force = XTB_Force(q, atoms, qcinput)
    elif qchem == 'PES':
<<<<<<< HEAD
        force = PES_Force(q, active_state)
=======
        force = PES_Force(q, atoms, qcinput)
>>>>>>> ecad59b (2026)
    else:
        raise ValueError("Non-Existing Quantum Chemical method is input. Avaiable packages: Orca, PySCF, Psi4, Sparrow, XTB")

    return force
