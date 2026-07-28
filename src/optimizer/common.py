from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from integrators.gradient import Potential_Energy, force_calc
from optimizer.internal_coords import (
    InternalCoords,
    analyze_structure,
    bpg_matrix,
    compute_internals,
    internal_coords_from_bonds,
)
from optimizer.redundant_internals import (
    bpg_from_redundant,
    build_redundant_internals,
)
from utils.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, HARTREE_TO_KJMOL


@dataclass
class OptimizationResult:
    atoms: list[str]
    q: np.ndarray
    energy: float
    converged: bool
    nsteps: int
    method: str
    backend_optimizer: str
    trajectory_file: str | None = None
    gradient: np.ndarray | None = None
    message: str = ""
    coordinates: str = "cartesian"
    target: str = "minimum"
    negative_modes: int | None = None
    reaction_mode: np.ndarray | None = None
    hessian: np.ndarray | None = None

def step_limit(dX, max_component=0.1):
    dX = np.asarray(dX, dtype=float).copy()
    stpmax = len(dX) * max_component
    norm = np.linalg.norm(dX)
    if norm > stpmax:
        dX *= stpmax / norm

    largest = np.max(np.abs(dX)) if dX.size else 0.0
    if largest > max_component:
        dX *= max_component / largest
    return dX

def _gradient(qcinput, q, atoms):
    return -np.asarray(force_calc(qcinput, q, atoms), dtype=float)

def _energy(qcinput, q, atoms):
    return float(Potential_Energy(qcinput, None, q, atoms))

def _write_opt_frame(handle, atoms, q, step, energy, grad):
    grad_max = float(np.max(np.abs(grad))) if grad is not None else np.nan
    handle.write(str(len(atoms)) + "\n")
    handle.write(
        "%7s %10d %6s %20.8f %10s %20.8f\n"
        % ("step= ", step, "Ene = ", energy, "GradMax = ", grad_max)
    )
    for i, atom in enumerate(atoms):
        j = 3 * i
        handle.write(
            "%3s %15.8f  %15.8f %15.8f\n"
            % (atom, q[j] * BOHR_TO_ANGSTROM, q[j + 1] * BOHR_TO_ANGSTROM, q[j + 2] * BOHR_TO_ANGSTROM)
        )

def _xyz_from_q(atoms, q_bohr):
    q_angstrom = np.asarray(q_bohr, dtype=float).reshape(-1) * BOHR_TO_ANGSTROM
    lines = []
    for i, atom in enumerate(atoms):
        j = 3 * i
        lines.append(
            f"{atom:2s} {q_angstrom[j]:16.10f} {q_angstrom[j+1]:16.10f} {q_angstrom[j+2]:16.10f}"
        )
    return "\n".join(lines)

def _read_xyz_input(xyz):
    if xyz is None:
        return None
    if hasattr(xyz, "read"):
        return xyz.read()
    xyz = str(xyz)
    if "\n" not in xyz:
        try:
            with open(xyz, "r", encoding="utf-8") as handle:
                return handle.read()
        except OSError:
            pass
    return xyz

def _parse_xyz_input(xyz):
    text = _read_xyz_input(xyz)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Empty XYZ input")
    start = 0
    natoms = None
    try:
        natoms = int(lines[0].split()[0])
        start = 2
    except (ValueError, IndexError):
        start = 0

    coord_lines = lines[start:]
    if natoms is not None:
        if len(coord_lines) < natoms:
            raise ValueError(f"XYZ declares {natoms} atoms but only {len(coord_lines)} coordinate lines were found")
        coord_lines = coord_lines[:natoms]

    atoms = []
    q_angstrom = []
    for line in coord_lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid XYZ coordinate line: {line}")
        atoms.append(parts[0])
        q_angstrom.extend([float(parts[1]), float(parts[2]), float(parts[3])])
    return atoms, np.asarray(q_angstrom, dtype=float)

def _atoms_q_from_xyz(xyz):
    atoms, q_angstrom = _parse_xyz_input(xyz)
    q = np.asarray(q_angstrom, dtype=float).reshape(-1) * ANGSTROM_TO_BOHR
    atoms = list(atoms)
    if q.size != 3 * len(atoms):
        raise ValueError("XYZ coordinate length must be 3 * number of atoms")
    return atoms, q

def _validate_reaction_indices(indices, natoms, size, label):
    try:
        values = tuple(int(i) for i in indices)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a {size}-atom index tuple") from exc

    if len(values) != size:
        raise ValueError(f"{label} must contain exactly {size} atom indices")
    if len(set(values)) != size:
        raise ValueError(f"{label} atom indices must refer to different atoms")
    if any(i < 0 or i >= natoms for i in values):
        raise ValueError(f"{label} atom indices must be in the range 0..{natoms - 1}")
    return values

def _matches_reversed(reference, candidate):
    candidate = tuple(candidate)
    return candidate == reference or tuple(reversed(candidate)) == reference

def _converged(energy_change, step, grad, *, max_gradient, rms_gradient, max_step, rms_step, energy_tol):
    grad_max = float(np.max(np.abs(grad)))
    grad_rms = float(np.linalg.norm(grad) / np.sqrt(len(grad)))
    step_max = float(np.max(np.abs(step)))
    step_rms = float(np.linalg.norm(step) / np.sqrt(len(step)))
    return (
        grad_max <= max_gradient
        and grad_rms <= rms_gradient
        and step_max <= max_step
        and step_rms <= rms_step
        and abs(energy_change) <= energy_tol
    )

def _print_header(energy_tol, max_step, rms_step, max_gradient, rms_gradient):
    print(
        "Convergence thresholds: "
        f"dE={energy_tol * HARTREE_TO_KJMOL:.2e} kJ/mol, "
        f"max_step={max_step:.2e}, rms_step={rms_step:.2e}, "
        f"max_grad={max_gradient:.2e}, rms_grad={rms_gradient:.2e}"
    )
    print("%5s %18s %18s %11s %11s %11s %11s" % (
        "step", "E[Eh]", "dE[kJ/mol]", "max_step", "rms_step", "max_grad", "rms_grad"
    ))

def _print_step(istep, energy, energy_change, step, grad):
    print(
        "%5d %18.8f %18.6f %11.2e %11.2e %11.2e %11.2e"
        % (
            istep,
            energy,
            energy_change * HARTREE_TO_KJMOL,
            np.max(np.abs(step)),
            np.linalg.norm(step) / np.sqrt(len(step)),
            np.max(np.abs(grad)),
            np.linalg.norm(grad) / np.sqrt(len(grad)),
        )
    )

def _print_optimization_status(converged, message):
    status = "CONVERGED" if converged else "NOT CONVERGED"
    status_text = f" {status} "
    width = max(len(status_text) + 8, len(str(message)) if message else 0)
    bar = "=" * width
    print()
    print(bar)
    print(status_text.center(width, "*"))
    if message:
        print(message)
    print(bar)
    print()

def _solve_step(hess, grad):
    try:
        return np.linalg.solve(hess, -grad)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(hess, -grad, rcond=None)[0]

def _bfgs_update_hessian(hess, step, y):
    ys = float(np.dot(y, step))
    h_step = hess @ step
    shs = float(np.dot(step, h_step))
    if ys <= 1.0e-14 or shs <= 1.0e-14:
        return hess
    return hess + np.outer(y, y) / ys - np.outer(h_step, h_step) / shs

def _bofill_update_hessian(hess, step, y, *, denom_tol=1.0e-14):
    step = np.asarray(step, dtype=float)
    y = np.asarray(y, dtype=float)
    u = y - hess @ step
    ss = float(np.dot(step, step))
    us = float(np.dot(u, step))
    uu = float(np.dot(u, u))
    if ss <= denom_tol or uu <= denom_tol or abs(us) <= denom_tol:
        return hess, False

    sr1 = np.outer(u, u) / us
    psb = (np.outer(u, step) + np.outer(step, u)) / ss - (us / (ss * ss)) * np.outer(step, step)
    phi = (us * us) / (uu * ss)
    phi = min(1.0, max(0.0, phi))
    updated = hess + phi * sr1 + (1.0 - phi) * psb
    updated = 0.5 * (updated + updated.T)
    if not np.all(np.isfinite(updated)):
        return hess, False
    return updated, True

def _powell_update_hessian(hess, step, y, *, denom_tol=1.0e-14):
    step = np.asarray(step, dtype=float)
    y = np.asarray(y, dtype=float)
    u = y - hess @ step
    ss = float(np.dot(step, step))
    us = float(np.dot(u, step))
    if ss <= denom_tol:
        return hess, False
    updated = hess + (np.outer(u, step) + np.outer(step, u)) / ss - (us / (ss * ss)) * np.outer(step, step)
    updated = 0.5 * (updated + updated.T)
    if not np.all(np.isfinite(updated)):
        return hess, False
    return updated, True

def _exact_cartesian_hessian(qcinput, atoms, q, hess_file):
    from normalmode.hessian import getHessian

    hess_qcinput = dict(qcinput)
    hess_qcinput["force_hessian_recalc"] = True
    hess_qcinput["quiet_hessian"] = True
    hess_qcinput["save_hessian"] = False
    return np.asarray(
        getHessian(qcinput=hess_qcinput, hessFile=hess_file, xyz=_xyz_from_q(atoms, q)),
        dtype=float,
    )

def _bpg_for_internal_coordinates(x, ic, *, use_redundant_internals=False):
    if use_redundant_internals:
        system = build_redundant_internals(x, coordinates=ic)
        return bpg_from_redundant(system)
    return bpg_matrix(x, ic)

def _linear_bend_threshold(linear_bends, linear_bend_threshold_degrees):
    if linear_bend_threshold_degrees is not None:
        return float(linear_bend_threshold_degrees)
    if not linear_bends:
        return None
    return 179.5

def _extra_index_tuples(items, natoms, size, label):
    if items is None:
        return []
    values = []
    for item in items:
        try:
            entry = tuple(int(idx) for idx in item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} entries must be {size}-atom index tuples") from exc
        if len(entry) != size:
            raise ValueError(f"{label} entries must contain exactly {size} atom indices")
        if len(set(entry)) != size:
            raise ValueError(f"{label} entries must refer to different atoms")
        if any(idx < 0 or idx >= natoms for idx in entry):
            raise ValueError(f"{label} atom indices must be in the range 0..{natoms - 1}")
        values.append(entry)
    return values

def _append_reversible_unique(existing, extras):
    output = list(existing)
    for extra in extras:
        if any(_matches_reversed(extra, item) for item in output):
            continue
        output.append(tuple(extra))
    return output

def _print_extra_internal_coordinates(extra_bonds=None, extra_angles=None, extra_dihedrals=None):
    if not extra_bonds and not extra_angles and not extra_dihedrals:
        return
    print("Extra internal coordinates forced by input:")
    if extra_bonds:
        print(f"  bonds: {list(extra_bonds)}")
    if extra_angles:
        print(f"  angles: {list(extra_angles)}")
    if extra_dihedrals:
        print(f"  dihedrals: {list(extra_dihedrals)}")

def _internal_coordinates_with_extras(
    x,
    atoms,
    model,
    *,
    linear_threshold=None,
    extra_bonds=None,
    extra_angles=None,
    extra_dihedrals=None,
):
    base = analyze_structure(
        x,
        model=model,
        linear_bend_threshold_degrees=linear_threshold,
    )
    natoms = len(atoms)
    extra_bonds = _extra_index_tuples(extra_bonds, natoms, 2, "extra_bonds")
    extra_angles = _extra_index_tuples(extra_angles, natoms, 3, "extra_angles")
    extra_dihedrals = _extra_index_tuples(extra_dihedrals, natoms, 4, "extra_dihedrals")

    bonds = list(base.bonds)
    seen_bonds = {tuple(sorted(bond)) for bond in bonds}
    for i, j in extra_bonds:
        bond = tuple(sorted((i, j)))
        if bond not in seen_bonds:
            seen_bonds.add(bond)
            bonds.append(bond)

    if len(bonds) != len(base.bonds):
        ic = internal_coords_from_bonds(
            natoms,
            bonds,
            x=x,
            linear_bend_threshold_degrees=linear_threshold,
        )
    else:
        ic = base

    angles = _append_reversible_unique(ic.angles, extra_angles)
    dihedrals = _append_reversible_unique(ic.dihedrals, extra_dihedrals)
    return InternalCoords(
        nat=ic.nat,
        bonds=list(ic.bonds),
        angles=angles,
        linear_bends=list(ic.linear_bends),
        dihedrals=dihedrals,
        impropers=list(ic.impropers),
    )

def _initial_internal_coordinate_state(
    x,
    atoms,
    model,
    *,
    use_redundant_internals=False,
    linear_bends=False,
    linear_bend_threshold_degrees=None,
    extra_bonds=None,
    extra_angles=None,
    extra_dihedrals=None,
):
    linear_threshold = _linear_bend_threshold(linear_bends, linear_bend_threshold_degrees)
    ic = _internal_coordinates_with_extras(
        x,
        atoms,
        model,
        linear_threshold=linear_threshold,
        extra_bonds=extra_bonds,
        extra_angles=extra_angles,
        extra_dihedrals=extra_dihedrals,
    )
    if use_redundant_internals:
        system = build_redundant_internals(
            x,
            coordinates=ic,
        )
        return system.coordinates, system.q, bpg_from_redundant(system), system

    qs = compute_internals(x, ic)
    return ic, qs, bpg_matrix(x, ic), None

def _current_redundant_system(x, ic, *, use_redundant_internals=False):
    if not use_redundant_internals:
        return None
    return build_redundant_internals(x, coordinates=ic)
