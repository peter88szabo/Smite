import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _calculator(config):
    interface = Path(__file__).resolve().parents[2] / "peslib" / "OH+CH4" / "pes_interface.py"
    spec = importlib.util.spec_from_file_location("ohch4_test_interface", interface)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    calculator = module.PESCalculator.__new__(module.PESCalculator)
    calculator.config = dict(config)
    return calculator


def _geometry():
    atoms = ["C", "H", "H", "H", "H", "O", "H"]
    # H atom 1 is deliberately closer to O than the actual OH hydrogen at 6.
    return atoms, np.array(
        [
            0.0, 0.0, 0.0,
            0.1, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, -1.0,
            10.0, 0.0, 0.0,
        ]
    )


def test_ohch4_uses_fixed_oh_hydrogen_identity_not_nearest_hydrogen():
    atoms, q = _geometry()
    calculator = _calculator({"oh_h_index": 6})

    q_ordered, inverse = calculator._ordered_coordinates(q, atoms)

    # Native coordinate 6 is H(O), irrespective of geometric proximity.
    assert inverse[6] == 6
    np.testing.assert_allclose(q_ordered.reshape((-1, 3))[6], q.reshape((-1, 3))[6])


def test_ohch4_requires_explicit_oh_hydrogen_identity():
    atoms, q = _geometry()

    with pytest.raises(ValueError, match="oh_h_index"):
        _calculator({})._ordered_coordinates(q, atoms)


@pytest.mark.parametrize("index", [0, 5, 7, 1.5, True])
def test_ohch4_rejects_invalid_oh_hydrogen_identity(index):
    atoms, q = _geometry()

    with pytest.raises(ValueError):
        _calculator({"oh_h_index": index})._ordered_coordinates(q, atoms)
