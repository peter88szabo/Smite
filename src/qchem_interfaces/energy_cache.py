import numpy as np


def store_energy(qcinput, q, atoms, energy, force=None):
    qcinput["_last_energy_cache"] = {
        "q": np.asarray(q, dtype=float).copy(),
        "atoms": tuple(atoms),
        "energy": float(energy),
        "force": None if force is None else np.asarray(force, dtype=float).copy(),
    }


def cached_energy(qcinput, q, atoms):
    cache = qcinput.get("_last_energy_cache")
    if cache is None:
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

    q_arr = np.asarray(q, dtype=float)
    if tuple(atoms) != cache["atoms"]:
        return None
    if q_arr.shape != cache["q"].shape:
        return None
    if not np.array_equal(q_arr, cache["q"]):
        return None

    return cache["force"].copy()
