import numpy as np


SUPPORTED_QCHEM = {
    "XTB",
    "Orca",
    "PySCF",
    "Psi4",
    "Sparrow_bin",
    "Sparrow_Py",
    "PES",
}

QCHEM_ALIASES = {
    "orca": "Orca",
    "ORCA": "Orca",
    "xtb": "XTB",
    "pyscf": "PySCF",
    "psi4": "Psi4",
    "sparrow": "Sparrow_bin",
    "Sparrow": "Sparrow_bin",
    "sparrow_bin": "Sparrow_bin",
    "sparrow_py": "Sparrow_Py",
    "pes": "PES",
}

DEFAULTS = {
    "charge": 0,
    "multiplicity": 1,
    "nproc": 1,
    "wfu": False,
    "additional": "",
    "basis": "",
    "functional": "",
    "path": "",
}

COMMON_OPTIONAL_KEYS = {
    "_last_energy_cache",
    "_qchem_validated",
    "backend_retry_after_cleanup",
    "force_hessian_recalc",
    "hessian_dx",
    "hessian_keyword",
    "orca_hessian_keyword",
    "pes_name",
    "pes_path",
    "quiet_hessian",
    "oh_h_index",
    "oh_h_atom_index",
    "retry_after_cleanup",
    "scf_guess_mode",
    "save_hessian",
    "scratch_dir",
    "spin_multiplicities",
    "spin_mults",
    "spinmulti2",
    "spinmult1",
    "spinmult2",
    "state_a",
    "state_b",
    "what",
}

PATH_REQUIRED = {"XTB", "Orca", "Sparrow_bin"}
FUNCTIONAL_REQUIRED = {"Orca", "PySCF", "Psi4", "Sparrow_bin", "Sparrow_Py"}


def normalize_qchem_name(qchem):
    if qchem in SUPPORTED_QCHEM:
        return qchem
    if qchem in QCHEM_ALIASES:
        return QCHEM_ALIASES[qchem]
    if isinstance(qchem, str):
        lowered = qchem.strip().lower()
        if lowered in QCHEM_ALIASES:
            return QCHEM_ALIASES[lowered]
    raise ValueError(
        "Unsupported qchem interface "
        f"{qchem!r}. Choose from: {', '.join(sorted(SUPPORTED_QCHEM))}."
    )


def validate_qchem_input(qcinput, *, require_path=True, strict=False, allowed_extra_keys=None):
    if not isinstance(qcinput, dict):
        raise TypeError("qchem input must be a dictionary")

    if qcinput.get("_qchem_validated") is True:
        return qcinput

    if "qchem" not in qcinput:
        raise ValueError("qchem input must define key 'qchem'")

    allowed_extra_keys = set(allowed_extra_keys or ())
    raw_keys = set(qcinput)
    normalized = dict(DEFAULTS)
    normalized.update(qcinput)
    qchem = normalize_qchem_name(normalized["qchem"])
    normalized["qchem"] = qchem

    allowed_keys = {"qchem"} | set(DEFAULTS) | COMMON_OPTIONAL_KEYS | allowed_extra_keys
    unknown = sorted(raw_keys - allowed_keys)
    if strict and unknown:
        raise ValueError(f"qchem input has unknown key(s): {', '.join(unknown)}")
    normalized["_unknown_qchem_keys"] = unknown

    try:
        normalized["charge"] = int(normalized["charge"])
        normalized["multiplicity"] = int(normalized["multiplicity"])
        normalized["nproc"] = int(normalized["nproc"])
    except (TypeError, ValueError) as exc:
        raise ValueError("qchem keys 'charge', 'multiplicity', and 'nproc' must be integers") from exc

    if normalized["multiplicity"] < 1:
        raise ValueError("qchem key 'multiplicity' must be >= 1")
    if normalized["nproc"] < 1:
        raise ValueError("qchem key 'nproc' must be >= 1")
    if normalized["charge"] > 50 or normalized["charge"] < -50:
        raise ValueError("qchem key 'charge' is outside the supported sanity range -50..50")
    if not isinstance(normalized["wfu"], bool):
        raise ValueError("qchem key 'wfu' must be True or False")

    if normalized.get("additional") is None:
        normalized["additional"] = ""

    if "hessian_dx" in normalized:
        try:
            normalized["hessian_dx"] = float(normalized["hessian_dx"])
        except (TypeError, ValueError) as exc:
            raise ValueError("qchem key 'hessian_dx' must be a finite positive number") from exc
        if not np.isfinite(normalized["hessian_dx"]) or normalized["hessian_dx"] <= 0.0:
            raise ValueError("qchem key 'hessian_dx' must be a finite positive number")

    if require_path and qchem in PATH_REQUIRED and not normalized.get("path"):
        raise ValueError(f"{qchem} requires qchem key 'path' with the executable path")

    if qchem in FUNCTIONAL_REQUIRED and not normalized.get("functional"):
        raise ValueError(f"{qchem} requires qchem key 'functional'")

    if qchem == "PES":
        if normalized.get("wfu"):
            raise ValueError("PES does not support wfu=True because no electronic wavefunction is produced")
        if not (normalized.get("path") or normalized.get("pes_path") or normalized.get("pes_name") or normalized.get("name")):
            raise ValueError("PES requires one of: 'path', 'pes_path', 'pes_name', or 'name'")

    scratch_dir = normalized.get("scratch_dir")
    if scratch_dir is not None and not isinstance(scratch_dir, str):
        raise ValueError("qchem key 'scratch_dir' must be a string when provided")

    spin_multiplicities = normalized.get("spin_multiplicities") or normalized.get("spin_mults")
    if spin_multiplicities is not None:
        try:
            mult_a, mult_b = spin_multiplicities
            mult_a = int(mult_a)
            mult_b = int(mult_b)
        except (TypeError, ValueError) as exc:
            raise ValueError("qchem spin_multiplicities must contain two integers") from exc
        if mult_a < 1 or mult_b < 1:
            raise ValueError("qchem spin_multiplicities values must be >= 1")
        normalized["spin_multiplicities"] = (mult_a, mult_b)

    normalized["_qchem_validated"] = True
    qcinput.clear()
    qcinput.update(normalized)
    return qcinput
