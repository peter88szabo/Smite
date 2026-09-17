import os
import numpy as np
from utils.constants import BOHR_TO_ANGSTROM
from qchem_interfaces.backend_common import QCBackendError, backend_scratch_dir, parse_error
from qchem_interfaces.energy_cache import store_energy
from qchem_interfaces.wavefunction_output import prepare_wavefunction_output


def configure_pyscf_runtime(lib, qcinput):
    nproc = int(qcinput.get('nproc', 1))
    if nproc > 0:
        lib.num_threads(nproc)
    if qcinput.get('scratch_dir'):
        lib.param.TMPDIR = backend_scratch_dir(qcinput, 'pyscf', 'pyscf_tmp')


def makeXYZ(atoms, q):
    natoms = len(atoms)

    xyz = ""

    for i in range(natoms):
        xyz += atoms[i] + " "*6+str(q[i*3]*BOHR_TO_ANGSTROM) \
            + " "*6+str(q[i*3+1]*BOHR_TO_ANGSTROM) \
            + " "*6+str(q[i*3+2]*BOHR_TO_ANGSTROM) \
            + "\n"
    return xyz


def _import_pyscf():
    try:
        from pyscf import gto, dft, lib
        from pyscf.tools import molden
    except ImportError as exc:
        raise QCBackendError(
            "PySCF is not importable. Install PySCF in this Python environment "
            "or choose a different qchem backend."
        ) from exc
    return gto, dft, lib, molden


def _build_pyscf_mean_field(q, atoms, qcinput):
    gto, dft, lib, _molden = _import_pyscf()
    configure_pyscf_runtime(lib, qcinput)

    xyz = makeXYZ(atoms, q)
    mol = gto.M(
            atom=xyz,
            basis=qcinput['basis'],
            charge=qcinput['charge'],
            spin=qcinput['multiplicity'] - 1,
            output=os.devnull,
            verbose=0)

    if qcinput['multiplicity'] != 1:
        mf = dft.UKS(mol)
    else:
        mf = dft.RKS(mol)
    mf.xc = qcinput['functional']
    try:
        mf.kernel()
    except Exception as exc:
        raise QCBackendError(f"PySCF SCF calculation failed: {exc}") from exc
    if not getattr(mf, "converged", True):
        raise QCBackendError("PySCF SCF calculation did not converge.")
    if not np.isfinite(float(mf.e_tot)):
        raise QCBackendError(f"PySCF returned a non-finite energy: {mf.e_tot}")
    return mol, mf

def PySCF_Force(q, atoms, qcinput):
    _mol, mf = _build_pyscf_mean_field(q, atoms, qcinput)
    try:
        g = mf.nuc_grad_method()
        grad_matrix = np.asarray(g.kernel(), dtype=float)
    except Exception as exc:
        raise parse_error("PySCF", f"gradient calculation failed: {exc}", cause=exc) from exc
    expected = (len(atoms), 3)
    if grad_matrix.shape != expected:
        raise parse_error("PySCF", f"gradient shape is {grad_matrix.shape}, expected {expected}")
    grad = grad_matrix.reshape(-1)
    force = -np.array(grad)
    store_energy(qcinput, q, atoms, mf.e_tot, force=force)
    return force 

def PySCF_Energy(filename, q, atoms, qcinput):
    _gto, _dft, _lib, molden = _import_pyscf()
    wfu          =   qcinput['wfu']
    mol, mf = _build_pyscf_mean_field(q, atoms, qcinput)
    E = float(mf.e_tot)

    if wfu == True:
        filename = prepare_wavefunction_output(filename, "PySCF")
        with open(filename, "w") as file_wf:
            molden.header(mol, file_wf)
            molden.orbital_coeff(mol, file_wf, mf.mo_coeff, ene=mf.mo_energy, occ=mf.mo_occ)

    return E

def PySCF_Hessian(Natoms, xyz, qcinput):
    gto, dft, lib, _molden = _import_pyscf()

    functional   =   qcinput['functional']
    basis        =   qcinput['basis']
    charge       =   qcinput['charge']
    multiplicity =   qcinput['multiplicity']
    configure_pyscf_runtime(lib, qcinput)

    mol = gto.M(
            atom = xyz,
            basis = basis,
            charge = charge,
            spin = multiplicity -1,
            output=os.devnull,
            verbose=0)

    if multiplicity != 1:
        mf = dft.UKS(mol)
    else:
        mf = dft.RKS(mol)
    mf.xc = functional
    try:
        mf.kernel()
    except Exception as exc:
        raise QCBackendError(f"PySCF Hessian SCF calculation failed: {exc}") from exc
    if not getattr(mf, "converged", True):
        raise QCBackendError("PySCF Hessian SCF calculation did not converge.")
    h = mf.Hessian().kernel()
    hess = h.transpose(0,2,1,3).reshape(Natoms*3,Natoms*3)
    return hess


    # Compute the analytical Hessian
    #hessian_calculator = hessian.RKS(mf) if multiplicity == 1 else hessian.UKS(mf)
    #h = hessian_calculator.kernel()

    # Reshape the Hessian to a 2D matrix
    #hess = h.transpose(0, 2, 1, 3).reshape(Natoms * 3, Natoms * 3)
    #return hess
