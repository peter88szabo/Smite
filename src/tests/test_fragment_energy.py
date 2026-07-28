from types import SimpleNamespace

import numpy as np
import pytest

from analysis.fragment_energy import (
    product_energy_partitions,
    product_potential_energies,
    product_total_energy,
    reactant_total_energy,
)


def _collision(qchem="mock"):
    q = np.array([
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        5.0, 0.0, 0.0,
        6.0, 0.0, 0.0,
    ])
    return SimpleNamespace(
        q=q.copy(),
        p=np.zeros_like(q),
        q_collision_initial=q.copy(),
        p_collision_initial=np.zeros_like(q),
        mass=np.ones(4),
        natom=4,
        atoms=["H", "H", "H", "H"],
        fragment_A=SimpleNamespace(natom=2),
        qchem={"qchem": qchem},
    )


def _fragments():
    return [
        {"indices": [0, 1], "atoms": ["H", "H"], "formula": "H2_A"},
        {"indices": [2, 3], "atoms": ["H", "H"], "formula": "H2_B"},
    ]


def _states():
    return {
        "products": [
            {"indices": [0, 1], "charge": 1, "multiplicity": 2},
            {"indices": [2, 3], "charge": -1, "multiplicity": 1},
        ],
        "reactants": {
            "A": {"charge": 0, "multiplicity": 1},
            "B": {"charge": 0, "multiplicity": 1},
        },
    }


def test_qchem_fragments_are_evaluated_independently_with_declared_states():
    collision = _collision()
    fragments = _fragments()
    calls = []

    def evaluator(qchem, q, atoms):
        calls.append((qchem["charge"], qchem["multiplicity"], len(atoms)))
        return 100.0 * qchem["charge"] + 10.0 * qchem["multiplicity"] + len(atoms) + np.sum(q)

    records = product_potential_energies(
        collision,
        fragments,
        _states(),
        equilibrium_geometries={"H2_A": [[-0.4, 0, 0], [0.4, 0, 0]], "H2_B": [[-0.4, 0, 0], [0.4, 0, 0]]},
        energy_evaluator=evaluator,
    )
    assert [record["potential_energy_source"] for record in records] == [
        "isolated_fragment_qchem",
        "isolated_fragment_qchem",
    ]
    np.testing.assert_allclose([record["potential_energy"] for record in records], [123.0, -77.0])
    np.testing.assert_allclose([record["reference_potential_energy"] for record in records], [122.0, -88.0])
    assert calls[:2] == [(1, 2, 2), (1, 2, 2)]
    assert calls[2:4] == [(-1, 1, 2), (-1, 1, 2)]

    partitions = product_energy_partitions(
        collision.q, collision.p, collision.mass, fragments, potential_energies=records
    )
    np.testing.assert_allclose(
        [partition["vibrational_energy"] for partition in partitions], [1.0, 11.0]
    )

    initial = reactant_total_energy(collision, _states(), energy_evaluator=evaluator)
    final = product_total_energy(collision, fragments, records, energy_evaluator=evaluator)
    assert initial["total_energy"] == pytest.approx(36.0)
    assert final["total_energy"] == pytest.approx(46.0)


def test_pes_uses_full_coordinates_with_a_distant_spectator():
    collision = _collision(qchem="PES")
    calls = []

    def evaluator(_qchem, q, atoms):
        q_xyz = q.reshape((-1, 3))
        calls.append((len(atoms), q_xyz.copy()))
        return float(np.sum(q_xyz * q_xyz))

    records = product_potential_energies(
        collision,
        _fragments(),
        _states(),
        equilibrium_geometries={"H2_A": [[-0.4, 0, 0], [0.4, 0, 0]], "H2_B": [[-0.4, 0, 0], [0.4, 0, 0]]},
        isolation_distance=80.0,
        energy_evaluator=evaluator,
    )
    assert {record["potential_energy_source"] for record in records} == {
        "full_pes_with_spectator_separated"
    }
    assert len(calls) == 4
    assert all(natom == 4 for natom, _q in calls)
    assert all(np.max(np.abs(q)) > 70.0 for _natom, q in calls)


def test_fragment_state_requires_charge_and_multiplicity():
    states = _states()
    del states["products"][0]["multiplicity"]
    with pytest.raises(ValueError, match="charge.*multiplicity"):
        product_potential_energies(_collision(), _fragments(), states, energy_evaluator=lambda *_: 0.0)
