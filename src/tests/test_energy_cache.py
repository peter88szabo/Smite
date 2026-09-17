import numpy as np

from qchem_interfaces.energy_cache import cached_energy, cached_force, store_energy


def test_cache_is_specific_to_electronic_model_and_state():
    qcinput = {
        "qchem": "Psi4",
        "functional": "hf",
        "basis": "sto-3g",
        "charge": 0,
        "multiplicity": 1,
    }
    q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.4])
    atoms = ["H", "H"]
    force = np.arange(6.0)
    store_energy(qcinput, q, atoms, -1.1, force=force)

    assert cached_energy(qcinput, q, atoms) == -1.1
    np.testing.assert_allclose(cached_force(qcinput, q, atoms), force)

    for key, value in (("multiplicity", 3), ("charge", 1), ("functional", "b3lyp"), ("basis", "6-31g")):
        changed = dict(qcinput)
        changed[key] = value
        assert cached_energy(changed, q, atoms) is None
        assert cached_force(changed, q, atoms) is None


def test_cache_ignores_its_own_validation_bookkeeping_only():
    qcinput = {"qchem": "PES", "pes_name": "test"}
    q = np.zeros(3)
    store_energy(qcinput, q, ["H"], 2.0)
    qcinput["_qchem_validated"] = True
    qcinput["_unknown_qchem_keys"] = ["irrelevant"]

    assert cached_energy(qcinput, q, ["H"]) == 2.0
