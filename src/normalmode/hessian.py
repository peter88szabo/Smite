from qchem_interfaces.psi4run      import Psi4_Hessian
from qchem_interfaces.orcarun      import Orca_Hessian
from qchem_interfaces.sparrowbin   import Sparrowbin_Force, Sparrowbin_Energy
from qchem_interfaces.pyscfrun     import PySCF_Hessian
from qchem_interfaces.sparrowpy    import SparrowPy_Hessian
from qchem_interfaces.sparrowbin   import Sparrowbin_Hessian
from qchem_interfaces.xtbrun       import XTB_Hessian
from qchem_interfaces.pesrun       import PES_Hessian
from qchem_interfaces.qchem_validation import validate_qchem_input
from qchem_interfaces.energy_cache import electronic_model_fingerprint

from utils.format_and_print import parseXYZ
from utils.constants import ANGSTROM_TO_BOHR

import numpy as np
import os
import hashlib
import json


_HESSIAN_METADATA_VERSION = 1


def _hessian_metadata_path(hess_file):
    return f"{hess_file}.meta.json"


def _hessian_provenance(qcinput, atoms, qcoord):
    qcoord = np.ascontiguousarray(np.asarray(qcoord, dtype=np.float64).reshape(-1))
    atom_bytes = "\0".join(map(str, atoms)).encode("utf-8")
    geometry_hash = hashlib.sha256(atom_bytes + qcoord.tobytes()).hexdigest()
    return {
        "version": _HESSIAN_METADATA_VERSION,
        "atoms": list(atoms),
        "coordinate_hash": geometry_hash,
        "coordinates_unit": "bohr",
        "dimension": int(qcoord.size),
        "electronic_model": electronic_model_fingerprint(qcinput),
    }


def _load_matching_hessian(hess_file, provenance, expected_shape):
    metadata_path = _hessian_metadata_path(hess_file)
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata != provenance:
            return None, "provenance does not match the requested geometry/model"
        hessian = np.loadtxt(hess_file, delimiter=" ")
    except (OSError, ValueError, json.JSONDecodeError):
        return None, "metadata is missing or unreadable"
    if hessian.shape != expected_shape:
        return None, f"matrix has shape {hessian.shape}, expected {expected_shape}"
    if not np.all(np.isfinite(hessian)):
        return None, "matrix contains non-finite values"
    return 0.5 * (hessian + hessian.T), None


def getHessian(qcinput, hessFile, xyz):
    qcinput = validate_qchem_input(qcinput)

    Natoms, atoms, qcoord = parseXYZ(xyz)

    qcoord = qcoord * ANGSTROM_TO_BOHR #angtstrom to bohr

    qchem = qcinput['qchem'] 

    force_recalc = bool(qcinput.get("force_hessian_recalc", False))
    quiet = bool(qcinput.get("quiet_hessian", False))
    save_hessian = bool(qcinput.get("save_hessian", True))
    expected_shape = (Natoms * 3, Natoms * 3)
    provenance = _hessian_provenance(qcinput, atoms, qcoord)

    if os.path.exists(hessFile) and not force_recalc:
        hess, cache_error = _load_matching_hessian(hessFile, provenance, expected_shape)
        if hess is not None:
            if not quiet:
                print(f"\nLoading Hessian with matching provenance from file\n")
            return hess
        if not quiet:
            print(f"\nIgnoring existing Hessian: {cache_error}.\n")
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

    hess = np.asarray(hess, dtype=float)
    if hess.shape != expected_shape:
        raise ValueError(f"Hessian has shape {hess.shape}, expected {expected_shape}")
    if not np.all(np.isfinite(hess)):
        raise ValueError("Computed Hessian contains non-finite values")
    hess = 0.5 * (hess + hess.T)

    if save_hessian:
        np.savetxt(hessFile, hess, delimiter=' ', newline='\n')
        with open(_hessian_metadata_path(hessFile), "w", encoding="utf-8") as handle:
            json.dump(provenance, handle, sort_keys=True, indent=2)


    return hess
