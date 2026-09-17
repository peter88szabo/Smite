from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qchem_interfaces.pesrun import PES_Energy, PES_Force
from qchem_interfaces.qchem_validation import validate_qchem_input
from utils.constants import ANGSTROM_TO_BOHR


@dataclass
class PESValidationResult:
    atoms: list[str]
    energy: float
    force: np.ndarray
    coordinate_index: int
    analytic_gradient_component: float
    finite_difference_gradient_component: float
    absolute_error: float
    relative_error: float
    passed: bool
    tolerance: float


def parse_xyz_geometry(xyz):
    lines = [line.strip() for line in str(xyz).strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError("XYZ geometry is empty")
    if len(lines[0].split()) == 1:
        try:
            natoms = int(lines[0])
            lines = lines[2:2 + natoms]
        except ValueError:
            pass
    atoms = []
    coords = []
    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed XYZ line: {line!r}")
        atoms.append(parts[0])
        coords.extend([float(parts[1]), float(parts[2]), float(parts[3])])
    return atoms, np.asarray(coords, dtype=float) * ANGSTROM_TO_BOHR


def validate_pes_interface(
    qcinput,
    xyz=None,
    *,
    atoms=None,
    q=None,
    coordinate_index=0,
    displacement=1.0e-4,
    tolerance=1.0e-4,
    print_report=True,
):
    """Validate one PES interface point with energy, force, and finite difference.

    Coordinates are in bohr when atoms and q are supplied directly. XYZ input is
    interpreted in Angstrom. The force convention is checked through
    gradient = -force.
    """
    qcinput = validate_qchem_input(qcinput)
    if qcinput["qchem"] != "PES":
        raise ValueError("validate_pes_interface requires qcinput['qchem'] = 'PES'")

    if xyz is not None:
        atoms, q = parse_xyz_geometry(xyz)
    elif atoms is None or q is None:
        raise ValueError("Provide either xyz or both atoms and q")
    else:
        atoms = list(atoms)
        q = np.asarray(q, dtype=float).reshape(-1)

    expected = 3 * len(atoms)
    if q.size != expected:
        raise ValueError(f"Coordinate length is {q.size}, expected {expected} for {len(atoms)} atoms")
    coordinate_index = int(coordinate_index)
    if coordinate_index < 0 or coordinate_index >= q.size:
        raise ValueError(f"coordinate_index must be in the range 0..{q.size - 1}")

    energy = PES_Energy(None, q, atoms, qcinput)
    force = np.asarray(PES_Force(q, atoms, qcinput), dtype=float).reshape(-1)
    if force.shape != q.shape:
        raise ValueError(f"PES force shape {force.shape} does not match coordinate shape {q.shape}")

    dq = np.zeros_like(q)
    dq[coordinate_index] = float(displacement)
    e_plus = PES_Energy(None, q + dq, atoms, qcinput)
    e_minus = PES_Energy(None, q - dq, atoms, qcinput)
    fd_grad = (e_plus - e_minus) / (2.0 * float(displacement))
    analytic_grad = -force[coordinate_index]
    abs_error = abs(analytic_grad - fd_grad)
    scale = max(abs(analytic_grad), abs(fd_grad), 1.0e-12)
    rel_error = abs_error / scale
    passed = bool(abs_error <= float(tolerance))

    result = PESValidationResult(
        atoms=atoms,
        energy=float(energy),
        force=force,
        coordinate_index=coordinate_index,
        analytic_gradient_component=float(analytic_grad),
        finite_difference_gradient_component=float(fd_grad),
        absolute_error=float(abs_error),
        relative_error=float(rel_error),
        passed=passed,
        tolerance=float(tolerance),
    )

    if print_report:
        print("PES interface validation")
        print("energy [hartree]:", result.energy)
        print("coordinate index:", result.coordinate_index)
        print("analytic gradient:", result.analytic_gradient_component)
        print("finite-difference gradient:", result.finite_difference_gradient_component)
        print("absolute error:", result.absolute_error)
        print("relative error:", result.relative_error)
        print("tolerance:", result.tolerance)
        print("passed:", result.passed)
    return result
