import hashlib
import os

import numpy as np


# These fields either contain a previous cache entry or validation bookkeeping;
# neither changes the electronic Hamiltonian evaluated by a backend.
_NON_MODEL_KEYS = frozenset({
    "_last_energy_cache",
    "_qchem_validated",
    "_unknown_qchem_keys",
    "force_hessian_recalc",
    "quiet_hessian",
    "save_hessian",
    "scratch_dir",
})


def _freeze(value):
    """Convert arbitrary qchem options to a deterministic, hashable structure."""
    if isinstance(value, dict):
        return tuple(
            sorted(
                (str(key), _freeze(item))
                for key, item in value.items()
                if key not in _NON_MODEL_KEYS
            )
        )
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return ("ndarray", array.dtype.str, array.shape, array.tobytes())
    if isinstance(value, np.generic):
        return _freeze(value.item())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    if isinstance(value, os.PathLike):
        return ("path", os.fspath(value))
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return value
    return (type(value).__module__, type(value).__qualname__, repr(value))


def electronic_model_fingerprint(qcinput):
    """Return the identity of the backend/model/state used for an evaluation.

    Coordinates and atom labels are checked separately.  This fingerprint makes
    cache reuse conditional on *all* electronic-structure options, including
    charge, multiplicity, method, basis, and backend-specific settings.
    """
    if not isinstance(qcinput, dict):
        raise TypeError("qcinput must be a dictionary")
    frozen = _freeze(qcinput)
    return hashlib.sha256(repr(frozen).encode("utf-8")).hexdigest()


def store_energy(qcinput, q, atoms, energy, force=None):
    qcinput["_last_energy_cache"] = {
        "q": np.asarray(q, dtype=float).copy(),
        "atoms": tuple(atoms),
        "model_fingerprint": electronic_model_fingerprint(qcinput),
        "energy": float(energy),
        "force": None if force is None else np.asarray(force, dtype=float).copy(),
    }


def cached_energy(qcinput, q, atoms):
    cache = qcinput.get("_last_energy_cache")
    if cache is None:
        return None

    if cache.get("model_fingerprint") != electronic_model_fingerprint(qcinput):
        return None

    q_arr = np.asarray(q, dtype=float)
    if tuple(atoms) != cache["atoms"]:
        return None
    if q_arr.shape != cache["q"].shape:
        return None
    if not np.array_equal(q_arr, cache["q"]):
        return None

    return cache["energy"]


def cached_force(qcinput, q, atoms):
    cache = qcinput.get("_last_energy_cache")
    if cache is None or cache.get("force") is None:
        return None

    if cache.get("model_fingerprint") != electronic_model_fingerprint(qcinput):
        return None

    q_arr = np.asarray(q, dtype=float)
    if tuple(atoms) != cache["atoms"]:
        return None
    if q_arr.shape != cache["q"].shape:
        return None
    if not np.array_equal(q_arr, cache["q"]):
        return None

    return cache["force"].copy()
