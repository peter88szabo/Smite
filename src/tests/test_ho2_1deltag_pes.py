import importlib.util
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from qchem_interfaces import pesrun


PES_DIR = Path(__file__).resolve().parents[2] / "peslib" / "HO2_1Deltag"
BOHR_TO_ANGSTROM_NATIVE = 0.52917706


@pytest.fixture(scope="module")
def calculator():
    if shutil.which("gfortran") is None:
        pytest.skip("gfortran is required to build the HO2_1Deltag test library")

    subprocess.run(
        ["make"],
        cwd=PES_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    interface = PES_DIR / "pes_interface.py"
    spec = importlib.util.spec_from_file_location(
        "ho2_1deltag_test_interface", interface
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PESCalculator(PES_DIR)


def _bohr(geometry_angstrom):
    return np.asarray(geometry_angstrom, dtype=float).reshape(-1) / BOHR_TO_ANGSTROM_NATIVE


def _central_difference(calculator, q, atoms, step=1.0e-5):
    result = np.empty_like(q)
    for index in range(q.size):
        q_plus = q.copy()
        q_minus = q.copy()
        q_plus[index] += step
        q_minus[index] -= step
        result[index] = (
            calculator.energy(q_plus, atoms) - calculator.energy(q_minus, atoms)
        ) / (2.0 * step)
    return result


@pytest.mark.parametrize(
    "geometry_angstrom",
    [
        # General tabulated region.
        [[0.50, 1.00, 0.20], [0.00, 0.00, 0.00], [1.25, 0.00, 0.00]],
        # Equal O-H distances exercise the symmetrizing switching function.
        [[0.625, 1.00, 0.20], [0.00, 0.00, 0.00], [1.25, 0.00, 0.00]],
        # OH + O long range exercises radial, O-H spline, and angle derivatives.
        [[0.00, 0.00, 0.00], [1.00, 0.00, 0.00], [12.00, 1.00, 0.00]],
        # H + O2 at the example's approximately 10 Angstrom separation.
        [[10.00, 1.00, 0.00], [-0.60375, 0.00, 0.00], [0.60375, 0.00, 0.00]],
    ],
)
def test_analytic_cartesian_gradient_matches_energy_finite_difference(
    calculator, geometry_angstrom
):
    atoms = ["H", "O", "O"]
    q = _bohr(geometry_angstrom)

    _energy, gradient = calculator.energy_and_gradient(q, atoms)
    numerical = _central_difference(calculator, q, atoms)

    np.testing.assert_allclose(gradient, numerical, rtol=2.0e-5, atol=2.0e-8)
    np.testing.assert_allclose(
        gradient.reshape(3, 3).sum(axis=0),
        np.zeros(3),
        atol=2.0e-13,
    )


@pytest.mark.parametrize("permutation", [[1, 0, 2], [2, 0, 1]])
def test_atom_reordering_restores_cartesian_gradient(calculator, permutation):
    atoms = ["H", "O", "O"]
    q = _bohr([[0.50, 1.00, 0.20], [0.00, 0.00, 0.00], [1.25, 0.00, 0.00]])
    energy, gradient = calculator.energy_and_gradient(q, atoms)

    permutation = np.asarray(permutation)
    permuted_atoms = [atoms[index] for index in permutation]
    permuted_q = q.reshape(3, 3)[permutation].reshape(-1)
    permuted_energy, permuted_gradient = calculator.energy_and_gradient(
        permuted_q, permuted_atoms
    )

    assert permuted_energy == pytest.approx(energy, abs=1.0e-14)
    np.testing.assert_allclose(
        permuted_gradient.reshape(3, 3),
        gradient.reshape(3, 3)[permutation],
        atol=1.0e-13,
    )


def test_smite_pes_backend_returns_native_energy_and_negative_gradient(calculator):
    atoms = ["O", "O", "H"]
    q = _bohr([[0.00, 0.00, 0.00], [1.25, 0.00, 0.00], [0.50, 1.00, 0.20]])
    qcinput = {
        "qchem": "PES",
        "pes_name": "HO2_1Deltag",
        "pes_path": str(PES_DIR),
        "wfu": False,
    }

    pesrun._CALCULATORS.clear()
    energy = pesrun.PES_Energy(None, q, atoms, qcinput)
    force = pesrun.PES_Force(q, atoms, qcinput)
    direct_energy, direct_gradient = calculator.energy_and_gradient(q, atoms)

    assert energy == pytest.approx(direct_energy, abs=1.0e-14)
    np.testing.assert_allclose(force, -direct_gradient, atol=1.0e-13)


@pytest.mark.parametrize(
    ("coordinates", "atoms", "message"),
    [
        (np.zeros(6), ["H", "O"], "exactly three atoms"),
        (np.zeros(9), ["H", "H", "O"], "composition H/O/O"),
        (np.full(9, np.nan), ["H", "O", "O"], "finite"),
    ],
)
def test_invalid_inputs_fail_cleanly(calculator, coordinates, atoms, message):
    with pytest.raises(ValueError, match=message):
        calculator.energy(coordinates, atoms)
