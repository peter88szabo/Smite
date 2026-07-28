from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from normalmode.eckart import build_bmat, get_eckart_projector
from optimizer.common import (
    OptimizationResult,
    _bofill_update_hessian,
    _bpg_for_internal_coordinates,
    _current_redundant_system,
    _energy,
    _exact_cartesian_hessian,
    _gradient,
    _initial_internal_coordinate_state,
    _print_optimization_status,
    _print_extra_internal_coordinates,
    _write_opt_frame,
    _xyz_from_q,
)
from optimizer.coordinate_refs import (
    internal_coordinate_direction,
    parse_weighted_coordinate_reference,
)
from optimizer.internal_coords import (
    best_fit_dq_to_cart,
    bpg_matrix,
    compute_internals,
    default_connectivity_model,
    diff_internals,
    initial_internal_hessian_from_model,
    internal_gradient,
)
from optimizer.redundant_internals import (
    bpg_from_redundant,
    cartesian_hessian_to_redundant,
    initial_redundant_hessian_from_model,
)
from utils.atomic_masses import get_mass_vector
from utils.constants import HARTREE_TO_KJMOL


ORCA_TS_ENERGY_TOL = 5.0e-6
ORCA_TS_MAX_STEP = 4.0e-3
ORCA_TS_RMS_STEP = 2.0e-3
ORCA_TS_MAX_GRADIENT = 3.0e-4
ORCA_TS_RMS_GRADIENT = 1.0e-4


@dataclass(frozen=True)
class TSModeDiagnostics:
    index: int
    eigenvalue: float
    overlap: float | None
    source: str
    tracking_space: str

def _cartesian_reaction_direction_from_bond(q, atoms, reaction_bond):
    try:
        i, j = reaction_bond
    except (TypeError, ValueError) as exc:
        raise ValueError("reaction_bond must be a two-atom index pair, e.g. reaction_bond=(0, 1)") from exc

    natoms = len(atoms)
    if not isinstance(i, (int, np.integer)) or not isinstance(j, (int, np.integer)):
        raise ValueError("reaction_bond atom indices must be integers")
    i = int(i)
    j = int(j)
    if i == j:
        raise ValueError("reaction_bond atom indices must refer to two different atoms")
    if i < 0 or i >= natoms or j < 0 or j >= natoms:
        raise ValueError(f"reaction_bond atom indices must be in the range 0..{natoms - 1}")

    q = np.asarray(q, dtype=float).reshape(-1)
    if q.size != 3 * natoms:
        raise ValueError("Coordinate length must be 3 * number of atoms")

    ri = q[3 * i : 3 * i + 3]
    rj = q[3 * j : 3 * j + 3]
    bond = rj - ri
    bond_norm = float(np.linalg.norm(bond))
    if bond_norm <= 0.0:
        raise ValueError("reaction_bond atoms occupy the same position, so no direction can be built")

    unit = bond / bond_norm
    direction = np.zeros_like(q)
    direction[3 * i : 3 * i + 3] = -unit
    direction[3 * j : 3 * j + 3] = unit
    return direction

def _cartesian_reaction_direction_from_transfer(q, atoms, reaction_transfer):
    try:
        donor, transferred_atom, acceptor = reaction_transfer
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "reaction_transfer must be a three-atom index tuple, e.g. "
            "reaction_transfer=(donor, transferred_atom, acceptor)"
        ) from exc

    natoms = len(atoms)
    indices = (donor, transferred_atom, acceptor)
    if not all(isinstance(idx, (int, np.integer)) for idx in indices):
        raise ValueError("reaction_transfer atom indices must be integers")
    donor, transferred_atom, acceptor = (int(donor), int(transferred_atom), int(acceptor))
    if len({donor, transferred_atom, acceptor}) != 3:
        raise ValueError("reaction_transfer must refer to three different atoms")
    if any(idx < 0 or idx >= natoms for idx in (donor, transferred_atom, acceptor)):
        raise ValueError(f"reaction_transfer atom indices must be in the range 0..{natoms - 1}")

    q = np.asarray(q, dtype=float).reshape(-1)
    if q.size != 3 * natoms:
        raise ValueError("Coordinate length must be 3 * number of atoms")

    transferred_pos = q[3 * transferred_atom : 3 * transferred_atom + 3]
    donor_vec = q[3 * donor : 3 * donor + 3] - transferred_pos
    acceptor_vec = q[3 * acceptor : 3 * acceptor + 3] - transferred_pos
    donor_norm = float(np.linalg.norm(donor_vec))
    acceptor_norm = float(np.linalg.norm(acceptor_vec))
    if donor_norm <= 0.0 or acceptor_norm <= 0.0:
        raise ValueError("reaction_transfer atoms occupy coincident positions, so no direction can be built")

    direction = np.zeros_like(q)
    direction[3 * transferred_atom : 3 * transferred_atom + 3] = (
        acceptor_vec / acceptor_norm - donor_vec / donor_norm
    )
    return direction

def _internal_reaction_direction_from_reference(
    ic,
    *,
    reaction_bond=None,
    reaction_angle=None,
    reaction_dihedral=None,
):
    refs = sum(item is not None for item in (reaction_bond, reaction_angle, reaction_dihedral))
    if refs != 1:
        raise ValueError("Specify exactly one internal reaction reference")

    if reaction_bond is not None:
        return internal_coordinate_direction(ic, "bond", reaction_bond, label="reaction_bond")

    if reaction_angle is not None:
        return internal_coordinate_direction(ic, "angle", reaction_angle, label="reaction_angle")

    return internal_coordinate_direction(ic, "dihedral", reaction_dihedral, label="reaction_dihedral")

def _parse_reaction_coordinate_entry(entry, index):
    # Combined reaction-coordinate input uses one compact form:
    #     ("bond", (1, 14), 1.0)
    #     ("angle", (0, 1, 14))
    # The weight is optional.  If it is omitted, the default weight is 0.5
    # before the final combined vector is normalized.
    try:
        return parse_weighted_coordinate_reference(entry, index, default_weight=0.5)
    except ValueError as exc:
        raise ValueError(
            "Invalid reaction_coordinates entry %d: %s" % (index, exc)
        ) from exc

def _internal_reaction_direction_from_coordinates(ic, reaction_coordinates):
    if reaction_coordinates is None:
        raise ValueError("reaction_coordinates cannot be None")
    try:
        entries = list(reaction_coordinates)
    except TypeError as exc:
        raise ValueError("reaction_coordinates must be an iterable of coordinate references") from exc
    if not entries:
        raise ValueError("reaction_coordinates must contain at least one coordinate reference")

    combined = np.zeros(ic.nint, dtype=float)
    for index, entry in enumerate(entries, start=1):
        kind, atoms, weight, _default_weight = _parse_reaction_coordinate_entry(entry, index)
        if kind == "bond":
            direction = _internal_reaction_direction_from_reference(ic, reaction_bond=atoms)
        elif kind == "angle":
            direction = _internal_reaction_direction_from_reference(ic, reaction_angle=atoms)
        else:
            direction = _internal_reaction_direction_from_reference(ic, reaction_dihedral=atoms)
        combined += weight * direction

    norm = float(np.linalg.norm(combined))
    if norm <= 0.0:
        raise ValueError("reaction_coordinates produced a zero reaction direction")
    return combined / norm

def _reaction_coordinates_label(reaction_coordinates):
    if reaction_coordinates is None:
        return None
    pieces = []
    for index, entry in enumerate(reaction_coordinates, start=1):
        kind, atoms, weight, default_weight = _parse_reaction_coordinate_entry(entry, index)
        atom_text = " ".join(str(int(i)) for i in atoms)
        prefix = {"bond": "B", "angle": "A", "dihedral": "D"}[kind]
        if default_weight:
            pieces.append(f"{prefix} {atom_text} * default(0.5)")
        else:
            pieces.append(f"{prefix} {atom_text} * {weight:g}")
    return "Following TS mode: combined " + " + ".join(pieces)

def _reaction_reference_label(
    *,
    reaction_bond=None,
    reaction_angle=None,
    reaction_dihedral=None,
    reaction_transfer=None,
    reaction_coordinates=None,
):
    if reaction_coordinates is not None:
        return _reaction_coordinates_label(reaction_coordinates)
    if reaction_bond is not None:
        return "Following TS mode: B %d %d" % tuple(int(i) for i in reaction_bond)
    if reaction_angle is not None:
        return "Following TS mode: A %d %d %d" % tuple(int(i) for i in reaction_angle)
    if reaction_dihedral is not None:
        return "Following TS mode: D %d %d %d %d" % tuple(int(i) for i in reaction_dihedral)
    if reaction_transfer is not None:
        return "Following TS mode: atom-transfer %d %d %d" % tuple(int(i) for i in reaction_transfer)
    return None

def _format_threshold_guide_value(value):
    return f"{float(value):.2e}"

def _place_centered_text(line, center, text):
    start = int(round(center - (len(text) - 1) / 2.0))
    if start < 0:
        start = 0
    end = start + len(text)
    if end > len(line):
        line.extend(" " for _ in range(end - len(line)))
    line[start:end] = list(text)

def _print_ts_threshold_guide(energy_tol, max_step, rms_step, max_gradient, rms_gradient, header_line):
    values = (
        _format_threshold_guide_value(energy_tol * HARTREE_TO_KJMOL),
        _format_threshold_guide_value(max_step),
        _format_threshold_guide_value(rms_step),
        _format_threshold_guide_value(max_gradient),
        _format_threshold_guide_value(rms_gradient),
    )
    columns = ("dE[kJ/mol]", "max_step", "rms_step", "max_grad", "rms_grad")
    centers = []
    for column in columns:
        start = header_line.index(column)
        centers.append(start + (len(column) - 1) / 2.0)

    value_line = list(" " * len(header_line))
    arrow_bar_line = list(" " * len(header_line))
    arrow_line = list(" " * len(header_line))
    prefix = "Threshold Values:"
    value_line[:len(prefix)] = list(prefix)
    for center, value in zip(centers, values):
        _place_centered_text(value_line, center, value)
        _place_centered_text(arrow_bar_line, center, "|")
        _place_centered_text(arrow_line, center, "V")
    print("".join(value_line).rstrip())
    print("".join(arrow_bar_line).rstrip())
    print("".join(arrow_line).rstrip())

def _print_ts_header(energy_tol, max_step, rms_step, max_gradient, rms_gradient):
    print("Transition-state convergence tolerances:")
    print(f"  Energy Change:  {energy_tol: .2e} Eh")
    print(f"  Max. Gradient:  {max_gradient: .2e} Eh/bohr")
    print(f"  RMS Gradient:   {rms_gradient: .2e} Eh/bohr")
    print(f"  Max Step:       {max_step: .2e} bohr")
    print(f"  RMS Step:       {rms_step: .2e} bohr")
    print("  Strict Convergence          .... False")
    header_line = (
        "%5s %18s %18s %11s %11s %11s %11s %9s %7s %8s %9s"
        % (
            "step",
            "E[Eh]",
            "dE[kJ/mol]",
            "max_step",
            "rms_step",
            "max_grad",
            "rms_grad",
            "step_cap",
            "#imags",
            "TSmode",
            "overlap",
        )
    )
    print()
    _print_ts_threshold_guide(energy_tol, max_step, rms_step, max_gradient, rms_gradient, header_line)
    print(header_line)

def _print_ts_step(istep, energy, energy_change, step, grad, dmax, negative_modes, mode_diag):
    mode_text = "n/a"
    overlap_text = "n/a"
    if mode_diag is not None:
        mode_text = str(mode_diag.index)
        if mode_diag.overlap is not None:
            overlap_text = f"{mode_diag.overlap:.2f}"
        else:
            overlap_text = "n/a"
    print(
        "%5d %18.8f %18.6f %11.2e %11.2e %11.2e %11.2e %9.2e %7s %8s %9s"
        % (
            istep,
            energy,
            energy_change * HARTREE_TO_KJMOL,
            np.max(np.abs(step)),
            np.linalg.norm(step) / np.sqrt(len(step)),
            np.max(np.abs(grad)),
            np.linalg.norm(grad) / np.sqrt(len(grad)),
            dmax,
            str(negative_modes),
            mode_text,
            overlap_text,
        )
    )

def _normalize_mode_columns(modes):
    modes = np.asarray(modes, dtype=float)
    norms = np.linalg.norm(modes, axis=0)
    normalized = modes.copy()
    good = norms > 0.0
    normalized[:, good] /= norms[good]
    return normalized

def _mass_weighted_cartesian_modes_from_internal(evecs, bpg, atoms):
    dx_modes = bpg.b.T @ (bpg.inv_g @ np.asarray(evecs, dtype=float))
    masses = np.asarray(get_mass_vector(atoms), dtype=float)
    sqrt_m = np.sqrt(np.repeat(masses, 3))
    return sqrt_m[:, None] * dx_modes

def _mass_weight_cartesian_direction(atoms, direction):
    direction = np.asarray(direction, dtype=float).reshape(-1)
    masses = np.asarray(get_mass_vector(atoms), dtype=float)
    if direction.size != 3 * len(masses):
        raise ValueError("Cartesian reaction direction length must be 3 * number of atoms")
    return np.sqrt(np.repeat(masses, 3)) * direction

def _partitioned_rfo_step_from_shifts(evals, grad_eig, reaction_index, lambda_p, lambda_n):
    step_eig = np.zeros_like(grad_eig)
    denom = float(evals[reaction_index] - lambda_p)
    if abs(denom) > 1.0e-14:
        step_eig[reaction_index] = -float(grad_eig[reaction_index]) / denom

    min_indices = np.asarray([i for i in range(len(evals)) if i != reaction_index], dtype=int)
    if min_indices.size:
        min_evals = evals[min_indices]
        min_grad = grad_eig[min_indices]
        denom = min_evals - lambda_n
        good = np.abs(denom) > 1.0e-14
        step_eig[min_indices[good]] = -min_grad[good] / denom[good]
    return step_eig

def _trust_constrained_partitioned_rfo_step(evals, grad_eig, reaction_index, lambda_p, lambda_n, trust_radius):
    step_eig = _partitioned_rfo_step_from_shifts(evals, grad_eig, reaction_index, lambda_p, lambda_n)
    if trust_radius is None or trust_radius <= 0.0:
        return step_eig

    trust_radius = float(trust_radius)
    if float(np.linalg.norm(step_eig)) <= trust_radius:
        return step_eig

    def shifted_step(gamma):
        # Increasing lambda_p damps the uphill component.  Decreasing lambda_n
        # damps the minimization subspace.  This changes the RFO direction,
        # unlike post-scaling the completed step vector.
        return _partitioned_rfo_step_from_shifts(
            evals,
            grad_eig,
            reaction_index,
            lambda_p + gamma,
            lambda_n - gamma,
        )

    lo = 0.0
    hi = 1.0
    for _ in range(80):
        if float(np.linalg.norm(shifted_step(hi))) <= trust_radius:
            break
        hi *= 2.0
    else:
        return trust_radius * step_eig / float(np.linalg.norm(step_eig))

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(np.linalg.norm(shifted_step(mid))) > trust_radius:
            lo = mid
        else:
            hi = mid
    return shifted_step(hi)

def _repair_ts_minimization_eigenvalues(evals, reaction_index, *, floor=1.0e-4):
    repaired = np.asarray(evals, dtype=float).copy()
    floor = abs(float(floor))
    if floor <= 0.0:
        return repaired
    for idx, value in enumerate(repaired):
        if idx == reaction_index:
            continue
        if value < floor:
            repaired[idx] = abs(float(value)) + floor
    return repaired

def _partitioned_rfo_step(
    hess,
    grad,
    *,
    previous_mode=None,
    reaction_direction=None,
    tracking_vectors=None,
    previous_tracking_mode=None,
    reaction_tracking_direction=None,
    follow_lowest=False,
    trust_radius=0.1,
    repair_hessian=True,
    hessian_eigenvalue_floor=1.0e-4,
):
    hess = 0.5 * (np.asarray(hess, dtype=float) + np.asarray(hess, dtype=float).T)
    grad = np.asarray(grad, dtype=float).reshape(-1)
    evals, evecs = np.linalg.eigh(hess)

    tracking_mode = None
    tracking_basis = None
    mode_source = "lowest"
    mode_overlap = None
    tracking_space = "internal" if tracking_vectors is None else "tracking"
    if tracking_vectors is not None:
        tracking_basis = _normalize_mode_columns(tracking_vectors)
        if tracking_basis.shape[1] != evecs.shape[1]:
            raise ValueError("tracking_vectors must have one column per Hessian eigenvector")

    if follow_lowest:
        reaction_index = int(np.argmin(evals))
        mode_source = "lowest"
    elif reaction_tracking_direction is not None and tracking_basis is not None:
        ref = np.asarray(reaction_tracking_direction, dtype=float).reshape(-1)
        ref_norm = np.linalg.norm(ref)
        if ref_norm <= 0.0 or ref.shape[0] != tracking_basis.shape[0]:
            raise ValueError("reaction_tracking_direction must be a non-zero vector with tracking-coordinate shape")
        overlaps = np.abs(tracking_basis.T @ (ref / ref_norm))
        reaction_index = int(np.argmax(overlaps))
        mode_overlap = float(overlaps[reaction_index])
        mode_source = "reference"
        tracking_space = "mass_weighted_cartesian"
    elif previous_tracking_mode is not None and tracking_basis is not None:
        prev = np.asarray(previous_tracking_mode, dtype=float).reshape(-1)
        prev_norm = np.linalg.norm(prev)
        if prev_norm > 0.0 and prev.shape[0] == tracking_basis.shape[0]:
            overlaps = np.abs(tracking_basis.T @ (prev / prev_norm))
            reaction_index = int(np.argmax(overlaps))
            mode_overlap = float(overlaps[reaction_index])
            mode_source = "previous"
            tracking_space = "mass_weighted_cartesian"
        else:
            reaction_index = int(np.argmin(evals))
    elif reaction_direction is not None:
        ref = np.asarray(reaction_direction, dtype=float).reshape(-1)
        ref_norm = np.linalg.norm(ref)
        if ref_norm <= 0.0 or ref.shape != grad.shape:
            raise ValueError("reaction_direction must be a non-zero vector with the optimization-coordinate shape")
        ref = ref / ref_norm
        overlaps = np.abs(evecs.T @ ref)
        reaction_index = int(np.argmax(overlaps))
        mode_overlap = float(overlaps[reaction_index])
        mode_source = "reference"
    elif previous_mode is not None:
        prev = np.asarray(previous_mode, dtype=float).reshape(-1)
        prev_norm = np.linalg.norm(prev)
        if prev_norm > 0.0 and prev.shape == grad.shape:
            overlaps = np.abs(evecs.T @ (prev / prev_norm))
            reaction_index = int(np.argmax(overlaps))
            mode_overlap = float(overlaps[reaction_index])
            mode_source = "previous"
        else:
            reaction_index = int(np.argmin(evals))
    else:
        reaction_index = int(np.argmin(evals))

    reaction_mode = evecs[:, reaction_index].copy()
    if tracking_basis is not None:
        tracking_mode = tracking_basis[:, reaction_index].copy()

    if previous_tracking_mode is not None and tracking_mode is not None:
        prev = np.asarray(previous_tracking_mode, dtype=float).reshape(-1)
        if prev.shape == tracking_mode.shape and np.dot(tracking_mode, prev) < 0.0:
            reaction_mode *= -1.0
            tracking_mode *= -1.0
    elif previous_mode is not None and np.dot(reaction_mode, previous_mode) < 0.0:
        reaction_mode *= -1.0

    if mode_overlap is None and previous_tracking_mode is not None and tracking_mode is not None:
        prev = np.asarray(previous_tracking_mode, dtype=float).reshape(-1)
        prev_norm = np.linalg.norm(prev)
        mode_norm = np.linalg.norm(tracking_mode)
        if prev_norm > 0.0 and mode_norm > 0.0 and prev.shape == tracking_mode.shape:
            mode_overlap = float(abs(np.dot(tracking_mode, prev) / (mode_norm * prev_norm)))
    if mode_overlap is None and previous_mode is not None:
        prev = np.asarray(previous_mode, dtype=float).reshape(-1)
        prev_norm = np.linalg.norm(prev)
        mode_norm = np.linalg.norm(reaction_mode)
        if prev_norm > 0.0 and mode_norm > 0.0 and prev.shape == reaction_mode.shape:
            mode_overlap = float(abs(np.dot(reaction_mode, prev) / (mode_norm * prev_norm)))

    grad_eig = evecs.T @ grad
    step_evals = (
        _repair_ts_minimization_eigenvalues(
            evals,
            reaction_index,
            floor=hessian_eigenvalue_floor,
        )
        if repair_hessian
        else evals
    )
    # Baker partitioned RFO: maximize one selected mode with the positive
    # 2x2 RFO root, and minimize all other modes together with one coupled
    # RFO shift. This is intentionally not a separate scalar RFO step for
    # every non-reaction mode.
    lam_k = float(step_evals[reaction_index])
    grad_k = float(grad_eig[reaction_index])
    lambda_p = 0.5 * (lam_k + np.sqrt(lam_k * lam_k + 4.0 * grad_k * grad_k))
    lambda_n = 0.0

    min_indices = [i for i in range(len(evals)) if i != reaction_index]
    if min_indices:
        min_evals = step_evals[min_indices]
        min_grad = grad_eig[min_indices]
        rfo = np.zeros((len(min_indices) + 1, len(min_indices) + 1), dtype=float)
        rfo[:-1, :-1] = np.diag(min_evals)
        rfo[:-1, -1] = min_grad
        rfo[-1, :-1] = min_grad
        lambda_n = float(np.linalg.eigvalsh(rfo)[0])
    step_eig = _trust_constrained_partitioned_rfo_step(
        step_evals,
        grad_eig,
        reaction_index,
        lambda_p,
        lambda_n,
        trust_radius,
    )

    step = evecs @ step_eig

    negative_modes = int(np.sum(evals < -1.0e-8))
    diagnostics = TSModeDiagnostics(
        index=reaction_index,
        eigenvalue=float(evals[reaction_index]),
        overlap=mode_overlap,
        source=mode_source,
        tracking_space=tracking_space,
    )
    return step, reaction_mode, reaction_index, evals, negative_modes, tracking_mode, diagnostics

def _ts_converged(energy_change, step, grad, *, energy_tol, max_gradient, rms_gradient, max_step, rms_step):
    grad_max = float(np.max(np.abs(grad)))
    grad_rms = float(np.linalg.norm(grad) / np.sqrt(len(grad)))
    step_max = float(np.max(np.abs(step)))
    step_rms = float(np.linalg.norm(step) / np.sqrt(len(step)))
    return (
        abs(float(energy_change)) <= energy_tol
        and grad_max <= max_gradient
        and grad_rms <= rms_gradient
        and step_max <= max_step
        and step_rms <= rms_step
    )

def _near_ts_convergence_for_recalc(
    step,
    grad,
    *,
    max_gradient,
    rms_gradient,
    max_step,
    rms_step,
    factor,
):
    if step is None:
        return False
    factor = float(factor)
    if factor <= 0.0:
        return False
    grad_max = float(np.max(np.abs(grad)))
    grad_rms = float(np.linalg.norm(grad) / np.sqrt(len(grad)))
    step_max = float(np.max(np.abs(step)))
    step_rms = float(np.linalg.norm(step) / np.sqrt(len(step)))
    return (
        grad_max <= factor * max_gradient
        and grad_rms <= factor * rms_gradient
        and step_max <= factor * max_step
        and step_rms <= factor * rms_step
    )

def _format_ts_mode_diagnostics(diag):
    overlap = "n/a" if diag.overlap is None else f"{diag.overlap:.3f}"
    return (
        f"mode={diag.index} eig={diag.eigenvalue:.5e} "
        f"source={diag.source} overlap={overlap}"
    )

def _ts_adaptive_hessian_reason(
    *,
    mode_diag,
    rho,
    negative_modes,
    previous_negative_modes,
    trust_radius,
    initial_trust_radius,
    trust_radius_min,
    overlap_min,
    rho_min,
    rho_max,
    trust_fraction,
    negative_mode_change,
    trust_collapse,
):
    if mode_diag is not None and mode_diag.overlap is not None and mode_diag.overlap < overlap_min:
        return f"mode overlap {mode_diag.overlap:.3f} < {overlap_min:.3f}"
    if not np.isfinite(rho) or rho < rho_min or rho > rho_max:
        return f"poor model ratio rho={rho:.3f}"
    if (
        negative_mode_change
        and previous_negative_modes is not None
        and negative_modes is not None
        and negative_modes != previous_negative_modes
    ):
        return f"imaginary-mode count changed {previous_negative_modes}->{negative_modes}"
    if trust_collapse and trust_radius <= max(trust_radius_min, trust_fraction * initial_trust_radius):
        return f"trust radius collapsed to {trust_radius:.5e}"
    return None

def _ts_hessian_recalc_allowed(step, last_recalc_step, cooldown):
    if cooldown <= 0:
        return True
    return (step - last_recalc_step) >= cooldown

def _eckart_project_cartesian_gradient_hessian(atoms, q, grad=None, hess=None):
    """Project Cartesian derivatives into the vibrational subspace.

    The existing Eckart projector is built for mass-weighted coordinates, as in
    the normal-mode code.  Convert Cartesian derivatives to mass-weighted form,
    project translations/rotations there, then convert back to Cartesian units.
    """
    mass = np.asarray(get_mass_vector(atoms), dtype=float)
    if mass.size != len(atoms) or np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
        raise ValueError("Could not build Eckart projector because one or more atomic masses are invalid")

    wmass = np.repeat(mass, 3)
    inv_sqrt_m = 1.0 / np.sqrt(wmass)
    sqrt_m = np.sqrt(wmass)
    try:
        rmat = get_eckart_projector(mass, q)
    except np.linalg.LinAlgError:
        bmat = build_bmat(mass, q)
        rmat = bmat @ np.linalg.pinv(bmat.T @ bmat) @ bmat.T
    projector = np.eye(3 * len(atoms)) - rmat

    projected_grad = None
    if grad is not None:
        grad_mw = inv_sqrt_m * np.asarray(grad, dtype=float).reshape(-1)
        projected_grad = sqrt_m * (projector @ grad_mw)

    projected_hess = None
    if hess is not None:
        hess = np.asarray(hess, dtype=float)
        hess_mw = inv_sqrt_m[:, None] * hess * inv_sqrt_m[None, :]
        projected_hess_mw = projector @ hess_mw @ projector
        projected_hess = sqrt_m[:, None] * projected_hess_mw * sqrt_m[None, :]
        projected_hess = 0.5 * (projected_hess + projected_hess.T)

    return projected_grad, projected_hess

def _maybe_project_ts_derivatives(atoms, q, grad=None, hess=None, *, project_eckart=True):
    if not project_eckart:
        return grad, hess
    return _eckart_project_cartesian_gradient_hessian(atoms, q, grad=grad, hess=hess)

def _hessian_file_for_step(base_hess_file, step, *, force_unique=False):
    if step == 0 and not force_unique:
        return base_hess_file
    if "." in base_hess_file:
        stem, suffix = base_hess_file.rsplit(".", 1)
        return f"{stem}_step{step:04d}.{suffix}"
    return f"{base_hess_file}_step{step:04d}"

def _final_hessian_file(base_hess_file):
    if "." in base_hess_file:
        stem, suffix = base_hess_file.rsplit(".", 1)
        return f"{stem}_final.{suffix}"
    return f"{base_hess_file}_final"

def _internal_coordinate_second_derivative_correction(q, ic, grad_q, *, dx=1.0e-4):
    q = np.asarray(q, dtype=float).reshape(-1)
    grad_q = np.asarray(grad_q, dtype=float).reshape(-1)
    if grad_q.size != ic.nint:
        raise ValueError(f"Internal gradient length mismatch: got {grad_q.size}, expected {ic.nint}")

    ndim = q.size
    correction = np.zeros((ndim, ndim), dtype=float)
    for i in range(ndim):
        q_plus = q.copy()
        q_minus = q.copy()
        q_plus[i] += dx
        q_minus[i] -= dx
        b_plus = bpg_matrix(q_plus, ic).b
        b_minus = bpg_matrix(q_minus, ic).b
        db_dxi = (b_plus - b_minus) / (2.0 * dx)
        correction[i, :] = grad_q @ db_dxi

    correction = 0.5 * (correction + correction.T)
    if not np.all(np.isfinite(correction)):
        raise ValueError("Non-finite internal-coordinate Hessian correction")
    return correction

def _internal_hessian_from_cartesian(hess_x, bpg, q=None, ic=None, grad_q=None):
    hess_x = np.asarray(hess_x, dtype=float)
    effective_hess_x = hess_x
    if q is not None and ic is not None and grad_q is not None:
        correction = _internal_coordinate_second_derivative_correction(q, ic, grad_q)
        effective_hess_x = hess_x - correction

    hess_q = bpg.inv_g @ (bpg.b @ effective_hess_x @ bpg.b.T) @ bpg.inv_g
    return 0.5 * (hess_q + hess_q.T)

def _inject_model_ts_negative_curvature(hess_q, reaction_direction=None, *, magnitude=0.05):
    hess_q = 0.5 * (np.asarray(hess_q, dtype=float) + np.asarray(hess_q, dtype=float).T)
    magnitude = abs(float(magnitude))
    if magnitude <= 0.0:
        return hess_q

    if reaction_direction is not None:
        vec = np.asarray(reaction_direction, dtype=float).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm > 0.0 and vec.size == hess_q.shape[0]:
            vec = vec / norm
            current = float(vec @ hess_q @ vec)
            return hess_q - (current + magnitude) * np.outer(vec, vec)

    evals, evecs = np.linalg.eigh(hess_q)
    idx = int(np.argmin(evals))
    evals[idx] = -magnitude
    return (evecs * evals) @ evecs.T

def _count_negative_modes(hess, *, cutoff=1.0e-8):
    evals = np.linalg.eigvalsh(0.5 * (np.asarray(hess, dtype=float) + np.asarray(hess, dtype=float).T))
    return int(np.sum(evals < -cutoff))

def _final_ts_hessian(qcinput, atoms, q, hess_file, *, project_eckart=True):
    hess = _exact_cartesian_hessian(qcinput, atoms, q, hess_file)
    _unused_grad, hess = _maybe_project_ts_derivatives(
        atoms, q, hess=hess, project_eckart=project_eckart
    )
    return hess

def _ts_accept_step(
    qcinput,
    atoms,
    q,
    energy,
    grad,
    hess,
    step,
    *,
    project_eckart=True,
    min_gradient_improvement=0.0,
):
    del min_gradient_improvement
    trial_q = q + step
    trial_energy = _energy(qcinput, trial_q, atoms)
    trial_grad = _gradient(qcinput, trial_q, atoms)
    trial_grad, _unused_hess = _maybe_project_ts_derivatives(
        atoms, trial_q, grad=trial_grad, project_eckart=project_eckart
    )
    pred = float(np.dot(grad, step) + 0.5 * np.dot(step, hess @ step))
    actual = trial_energy - energy
    rho = actual / pred if abs(pred) > 1.0e-14 else np.nan
    return trial_q, trial_energy, trial_grad, step, rho

def _smite_cartesian_ts_optimize_geometry(
    qcinput,
    atoms,
    q,
    *,
    maxstep=100,
    energy_tol=ORCA_TS_ENERGY_TOL,
    max_step=ORCA_TS_MAX_STEP,
    rms_step=ORCA_TS_RMS_STEP,
    max_gradient=ORCA_TS_MAX_GRADIENT,
    rms_gradient=ORCA_TS_RMS_GRADIENT,
    trajectory_file="tsopt_traj.xyz",
    print_report=True,
    hess_file="hessian_ts.hess",
    hessian_recalc_interval=0,
    trust_radius=0.1,
    trust_radius_min=1.0e-4,
    trust_radius_max=0.3,
    reaction_direction=None,
    reaction_label=None,
    max_rejected_steps=8,
    project_eckart=True,
    final_hessian=True,
    min_gradient_improvement=0.0,
    adaptive_hessian_recalc=False,
    adaptive_hessian_overlap_min=0.55,
    adaptive_hessian_rho_min=0.05,
    adaptive_hessian_rho_max=5.0,
    adaptive_hessian_trust_fraction=0.15,
    adaptive_hessian_on_negative_mode_change=True,
    adaptive_hessian_on_trust_collapse=False,
    adaptive_hessian_retry_recalc=False,
    adaptive_hessian_recalc_cooldown=3,
    use_redundant_internals=False,
    skip_hessian_recalc_near_convergence=True,
    hessian_recalc_near_convergence_factor=3.0,
):
    q = np.asarray(q, dtype=float).reshape(-1).copy()
    atoms = list(atoms)
    if len(q) != 3 * len(atoms):
        raise ValueError("Coordinate length must be 3 * number of atoms")

    energy = _energy(qcinput, q, atoms)
    grad = _gradient(qcinput, q, atoms)
    hess = _exact_cartesian_hessian(qcinput, atoms, q, hess_file)
    grad, hess = _maybe_project_ts_derivatives(atoms, q, grad=grad, hess=hess, project_eckart=project_eckart)
    reaction_mode = None
    reaction_mode_is_lowest = False
    negative_modes = None
    initial_trust_radius = trust_radius
    previous_negative_modes = None
    adaptive_recalc_next_reason = None
    last_hessian_recalc_step = 0
    previous_cart_step = None

    if print_report:
        print("Transition-state optimization with Cartesian partitioned RFO")
        print("TS mode tracking coordinates: cartesian")
        if reaction_label is not None:
            print(reaction_label)
        _print_ts_header(energy_tol, max_step, rms_step, max_gradient, rms_gradient)

    traj_handle = open(trajectory_file, "w", encoding="utf-8") if trajectory_file else None
    try:
        if traj_handle is not None:
            _write_opt_frame(traj_handle, atoms, q, 0, energy, grad)

        for istep in range(1, maxstep + 1):
            fixed_recalc = hessian_recalc_interval and istep > 1 and (istep - 1) % hessian_recalc_interval == 0
            if fixed_recalc and skip_hessian_recalc_near_convergence:
                if _near_ts_convergence_for_recalc(
                    previous_cart_step,
                    grad,
                    max_gradient=max_gradient,
                    rms_gradient=rms_gradient,
                    max_step=max_step,
                    rms_step=rms_step,
                    factor=hessian_recalc_near_convergence_factor,
                ):
                    if print_report:
                        print(f"      TS scheduled Hessian recalculation skipped near convergence at step {istep - 1}")
                    fixed_recalc = False
            if fixed_recalc or adaptive_recalc_next_reason is not None:
                if print_report:
                    if fixed_recalc:
                        print(f"      TS scheduled Hessian recalculation at step {istep - 1}")
                    if adaptive_recalc_next_reason is not None:
                        print(f"      TS adaptive Hessian recalculation: {adaptive_recalc_next_reason}")
                hfile = _hessian_file_for_step(hess_file, istep - 1, force_unique=True)
                hess = _exact_cartesian_hessian(qcinput, atoms, q, hfile)
                _unused_grad, hess = _maybe_project_ts_derivatives(
                    atoms, q, hess=hess, project_eckart=project_eckart
                )
                last_hessian_recalc_step = istep - 1
                adaptive_recalc_next_reason = None

            step, trial_mode, _mode_idx, _evals, trial_negative_modes, _tracking_mode, mode_diag = _partitioned_rfo_step(
                hess,
                grad,
                previous_mode=None if reaction_mode_is_lowest else reaction_mode,
                reaction_direction=reaction_direction if reaction_mode is None and not reaction_mode_is_lowest else None,
                follow_lowest=reaction_mode_is_lowest,
                trust_radius=trust_radius,
            )
            trial_q, trial_energy, trial_grad, used_step, rho = _ts_accept_step(
                qcinput,
                atoms,
                q,
                energy,
                grad,
                hess,
                step,
                project_eckart=project_eckart,
                min_gradient_improvement=min_gradient_improvement,
            )
            reaction_mode = trial_mode
            if _mode_idx == int(np.argmin(_evals)):
                reaction_mode_is_lowest = True
            negative_modes = trial_negative_modes

            energy_change = trial_energy - energy
            if print_report:
                _print_ts_step(
                    istep, trial_energy, energy_change, used_step, trial_grad,
                    trust_radius, negative_modes, mode_diag
                )

            if adaptive_hessian_recalc:
                if _ts_hessian_recalc_allowed(
                    istep, last_hessian_recalc_step, adaptive_hessian_recalc_cooldown
                ):
                    adaptive_recalc_next_reason = _ts_adaptive_hessian_reason(
                        mode_diag=mode_diag,
                        rho=rho,
                        negative_modes=negative_modes,
                        previous_negative_modes=previous_negative_modes,
                        trust_radius=trust_radius,
                        initial_trust_radius=initial_trust_radius,
                        trust_radius_min=trust_radius_min,
                        overlap_min=adaptive_hessian_overlap_min,
                        rho_min=adaptive_hessian_rho_min,
                        rho_max=adaptive_hessian_rho_max,
                        trust_fraction=adaptive_hessian_trust_fraction,
                        negative_mode_change=adaptive_hessian_on_negative_mode_change,
                        trust_collapse=adaptive_hessian_on_trust_collapse,
                    )
                else:
                    adaptive_recalc_next_reason = None
            previous_negative_modes = negative_modes

            if traj_handle is not None:
                _write_opt_frame(traj_handle, atoms, trial_q, istep, trial_energy, trial_grad)

            if _ts_converged(
                energy_change,
                used_step,
                trial_grad,
                energy_tol=energy_tol,
                max_gradient=max_gradient,
                rms_gradient=rms_gradient,
                max_step=max_step,
                rms_step=rms_step,
            ):
                result_hess = None
                result_negative_modes = negative_modes
                if final_hessian:
                    result_hess = _final_ts_hessian(
                        qcinput,
                        atoms,
                        trial_q,
                        _final_hessian_file(hess_file),
                        project_eckart=project_eckart,
                    )
                    result_negative_modes = _count_negative_modes(result_hess)
                if print_report:
                    _print_optimization_status(True, "Transition-state convergence reached.")
                return OptimizationResult(
                    atoms=atoms,
                    q=trial_q,
                    energy=trial_energy,
                    converged=True,
                    nsteps=istep,
                    method="P-RFO",
                    backend_optimizer="smite",
                    trajectory_file=trajectory_file,
                    gradient=trial_grad,
                    message="Transition-state convergence reached.",
                    coordinates="cartesian",
                    target="transition_state",
                    negative_modes=result_negative_modes,
                    reaction_mode=reaction_mode,
                    hessian=result_hess,
                )

            y = trial_grad - grad
            hess, updated = _bofill_update_hessian(hess, used_step, y)
            if not updated and hessian_recalc_interval == 0:
                pass

            q = trial_q
            energy = trial_energy
            grad = trial_grad
    finally:
        if traj_handle is not None:
            traj_handle.close()

    if print_report:
        _print_optimization_status(False, "Maximum TS optimization steps reached without convergence.")
    return OptimizationResult(
        atoms=atoms,
        q=q,
        energy=energy,
        converged=False,
        nsteps=maxstep,
        method="P-RFO",
        backend_optimizer="smite",
        trajectory_file=trajectory_file,
        gradient=grad,
        message="Maximum TS optimization steps reached without convergence.",
        coordinates="cartesian",
        target="transition_state",
        negative_modes=negative_modes,
        reaction_mode=reaction_mode,
    )

def _ts_accept_internal_step(
    qcinput,
    atoms,
    q,
    energy,
    grad_q,
    hess_q,
    dq,
    qs,
    ic,
    bpg,
    best_fit_iters,
    *,
    best_fit_rms_tol=1.0e-7,
    project_eckart=True,
    min_gradient_improvement=0.0,
    cartesian_step_max=None,
    use_redundant_internals=False,
):
    del min_gradient_improvement
    q = np.asarray(q, dtype=float).reshape(-1)
    dq = np.asarray(dq, dtype=float).reshape(-1)
    scale = 1.0
    trial_q = best_fit_dq_to_cart(q, qs, dq, ic, bpg, n_iter=best_fit_iters, rms_tol=best_fit_rms_tol)
    if cartesian_step_max is not None and cartesian_step_max > 0.0:
        for _ in range(8):
            cart_step = trial_q - q
            cart_norm = float(np.linalg.norm(cart_step))
            if cart_norm <= cartesian_step_max:
                break
            scale *= cartesian_step_max / cart_norm
            trial_q = best_fit_dq_to_cart(
                q,
                qs,
                scale * dq,
                ic,
                bpg,
                n_iter=best_fit_iters,
                rms_tol=best_fit_rms_tol,
            )
    trial_energy = _energy(qcinput, trial_q, atoms)
    trial_grad_x = _gradient(qcinput, trial_q, atoms)
    trial_grad_x, _unused_hess = _maybe_project_ts_derivatives(
        atoms, trial_q, grad=trial_grad_x, project_eckart=project_eckart
    )
    accepted_dq = scale * dq
    pred = float(np.dot(grad_q, accepted_dq) + 0.5 * np.dot(accepted_dq, hess_q @ accepted_dq))
    actual = trial_energy - energy
    rho = actual / pred if abs(pred) > 1.0e-14 else np.nan
    trial_bpg = _bpg_for_internal_coordinates(
        trial_q,
        ic,
        use_redundant_internals=use_redundant_internals,
    )
    trial_grad_q = internal_gradient(trial_bpg, trial_grad_x)
    return trial_q, trial_energy, trial_grad_x, trial_grad_q, trial_bpg, trial_q - q, rho, accepted_dq

def _smite_internal_ts_optimize_geometry(
    qcinput,
    atoms,
    q,
    *,
    maxstep=100,
    energy_tol=ORCA_TS_ENERGY_TOL,
    max_step=ORCA_TS_MAX_STEP,
    rms_step=ORCA_TS_RMS_STEP,
    max_gradient=ORCA_TS_MAX_GRADIENT,
    rms_gradient=ORCA_TS_RMS_GRADIENT,
    max_step_internal=0.2,
    best_fit_iters=5,
    best_fit_rms_tol=1.0e-7,
    trajectory_file="tsopt_traj.xyz",
    print_report=True,
    hess_file="hessian_ts.hess",
    hessian_recalc_interval=0,
    trust_radius=None,
    trust_radius_min=1.0e-4,
    trust_radius_max=0.4,
    reaction_direction=None,
    reaction_bond=None,
    reaction_angle=None,
    reaction_dihedral=None,
    reaction_transfer=None,
    reaction_coordinates=None,
    max_rejected_steps=8,
    connectivity_kcn=16.0,
    connectivity_facmin=0.45,
    connect_fragments=True,
    project_eckart=True,
    final_hessian=True,
    min_gradient_improvement=0.0,
    internal_hessian_correction=True,
    mode_tracking_coordinates="internal",
    adaptive_hessian_recalc=False,
    adaptive_hessian_overlap_min=0.55,
    adaptive_hessian_rho_min=0.05,
    adaptive_hessian_rho_max=5.0,
    adaptive_hessian_trust_fraction=0.15,
    adaptive_hessian_on_negative_mode_change=True,
    adaptive_hessian_on_trust_collapse=False,
    adaptive_hessian_retry_recalc=False,
    adaptive_hessian_recalc_cooldown=3,
    use_redundant_internals=False,
    skip_hessian_recalc_near_convergence=True,
    hessian_recalc_near_convergence_factor=3.0,
    repair_ts_hessian=True,
    ts_hessian_eigenvalue_floor=1.0e-4,
    linear_bends=False,
    linear_bend_threshold_degrees=None,
    extra_bonds=None,
    extra_angles=None,
    extra_dihedrals=None,
    internal_hessian_model="simple",
    ts_initial_hessian="exact",
    ts_model_negative_curvature=0.05,
):
    q = np.asarray(q, dtype=float).reshape(-1).copy()
    atoms = list(atoms)
    if len(q) != 3 * len(atoms):
        raise ValueError("Coordinate length must be 3 * number of atoms")
    if trust_radius is None:
        trust_radius = max_step_internal
    mode_tracking_coordinates = str(mode_tracking_coordinates).lower()
    if mode_tracking_coordinates not in {"internal", "mass_weighted_cartesian"}:
        raise ValueError("mode_tracking_coordinates must be 'internal' or 'mass_weighted_cartesian'")
    # Baker EF/P-RFO uses the Hessian in the coordinates being optimized.  The
    # finite-difference primitive-coordinate correction below is not part of
    # that algorithm and can disturb the followed Hessian mode.
    internal_hessian_correction = False

    model = default_connectivity_model(
        atoms,
        kcn=connectivity_kcn,
        facmin=connectivity_facmin,
        connect_fragments=connect_fragments,
    )
    ic, qs, bpg, redundant_system = _initial_internal_coordinate_state(
        q,
        atoms,
        model,
        use_redundant_internals=use_redundant_internals,
        linear_bends=linear_bends,
        linear_bend_threshold_degrees=linear_bend_threshold_degrees,
        extra_bonds=extra_bonds,
        extra_angles=extra_angles,
        extra_dihedrals=extra_dihedrals,
    )
    if ic.nint == 0:
        raise ValueError("Internal-coordinate TS optimization found no internal coordinates")

    ts_initial_hessian = str(ts_initial_hessian).lower()
    if ts_initial_hessian not in {"exact", "model"}:
        raise ValueError("ts_initial_hessian must be 'exact' or 'model'")

    energy = _energy(qcinput, q, atoms)
    grad_x = _gradient(qcinput, q, atoms)
    hess_x = None
    if ts_initial_hessian == "exact":
        hess_x = _exact_cartesian_hessian(qcinput, atoms, q, hess_file)
        grad_x, hess_x = _maybe_project_ts_derivatives(
            atoms, q, grad=grad_x, hess=hess_x, project_eckart=project_eckart
        )
    else:
        grad_x, _unused_hess = _maybe_project_ts_derivatives(
            atoms, q, grad=grad_x, project_eckart=project_eckart
        )
    grad_q = internal_gradient(bpg, grad_x)
    if ts_initial_hessian == "model":
        if use_redundant_internals and redundant_system is not None:
            hess_q = initial_redundant_hessian_from_model(redundant_system, model=internal_hessian_model)
        else:
            hess_q = initial_internal_hessian_from_model(ic, q_values=qs, model=internal_hessian_model)
    elif use_redundant_internals and redundant_system is not None:
        hess_q = cartesian_hessian_to_redundant(redundant_system, hess_x)
    elif internal_hessian_correction:
        hess_q = _internal_hessian_from_cartesian(hess_x, bpg, q=q, ic=ic, grad_q=grad_q)
    else:
        hess_q = _internal_hessian_from_cartesian(hess_x, bpg)
    reaction_mode = None
    reaction_tracking_mode = None
    reaction_mode_is_lowest = False
    negative_modes = None
    initial_trust_radius = trust_radius
    previous_negative_modes = None
    adaptive_recalc_next_reason = None
    last_hessian_recalc_step = 0
    previous_cart_step = None

    reaction_tracking_direction = None
    reaction_reference_label = _reaction_reference_label(
        reaction_bond=reaction_bond,
        reaction_angle=reaction_angle,
        reaction_dihedral=reaction_dihedral,
        reaction_transfer=reaction_transfer,
        reaction_coordinates=reaction_coordinates,
    )
    if reaction_coordinates is not None:
        if mode_tracking_coordinates != "internal":
            raise ValueError("reaction_mode='coordinates' requires internal mode tracking")
        if reaction_direction is not None:
            raise ValueError("reaction_coordinates cannot be combined with reaction_direction")
        if reaction_bond is not None or reaction_angle is not None or reaction_dihedral is not None:
            raise ValueError(
                "reaction_coordinates cannot be combined with reaction_bond, reaction_angle, or reaction_dihedral"
            )
        reaction_direction = _internal_reaction_direction_from_coordinates(ic, reaction_coordinates)
    elif reaction_bond is not None or reaction_angle is not None or reaction_dihedral is not None:
        if mode_tracking_coordinates != "internal":
            raise ValueError("reaction_mode='bond', 'angle', or 'dihedral' requires internal mode tracking")
        if reaction_direction is not None:
            raise ValueError("Internal reaction references cannot be combined with reaction_direction")
        reaction_direction = _internal_reaction_direction_from_reference(
            ic,
            reaction_bond=reaction_bond,
            reaction_angle=reaction_angle,
            reaction_dihedral=reaction_dihedral,
        )
    if reaction_direction is not None:
        reaction_direction = np.asarray(reaction_direction, dtype=float)
        if mode_tracking_coordinates == "mass_weighted_cartesian":
            if reaction_direction.shape != q.shape:
                raise ValueError(
                    "mass_weighted_cartesian mode tracking requires a Cartesian-shaped reaction_direction"
                )
            reaction_tracking_direction = _mass_weight_cartesian_direction(atoms, reaction_direction)
            reaction_direction = None
        elif reaction_direction.shape == q.shape:
            reaction_direction = bpg.b @ reaction_direction

    if ts_initial_hessian == "model":
        hess_q = _inject_model_ts_negative_curvature(
            hess_q,
            reaction_direction=reaction_direction,
            magnitude=ts_model_negative_curvature,
        )

    if print_report:
        print("Transition-state optimization with internal-coordinate partitioned RFO")
        print("Update method: Bofill")
        coord_text = "Pulay redundant internal coordinates" if use_redundant_internals else "internal coordinates"
        print(f"Choice of coordinates: {coord_text}")
        print(
            f"Internal coordinates: {ic.nbonds} bonds, {ic.nangles} angles, "
            f"{ic.nlinear_bends} linear bends, {ic.ndihedrals} dihedrals, "
            f"{ic.nimpropers} impropers"
        )
        _print_extra_internal_coordinates(extra_bonds, extra_angles, extra_dihedrals)
        print(f"TS mode tracking coordinates: {mode_tracking_coordinates}")
        print(f"Initial TS Hessian: {ts_initial_hessian}")
        if ts_initial_hessian == "model":
            print(f"Initial internal Hessian model: {internal_hessian_model}")
            print(f"Injected model TS negative curvature: {abs(float(ts_model_negative_curvature)):.6f}")
        if reaction_reference_label is not None:
            print(reaction_reference_label)
        _print_ts_header(energy_tol, max_step, rms_step, max_gradient, rms_gradient)

    traj_handle = open(trajectory_file, "w", encoding="utf-8") if trajectory_file else None
    try:
        if traj_handle is not None:
            _write_opt_frame(traj_handle, atoms, q, 0, energy, grad_x)

        for istep in range(1, maxstep + 1):
            fixed_recalc = hessian_recalc_interval and istep > 1 and (istep - 1) % hessian_recalc_interval == 0
            if fixed_recalc and skip_hessian_recalc_near_convergence:
                if _near_ts_convergence_for_recalc(
                    previous_cart_step,
                    grad_x,
                    max_gradient=max_gradient,
                    rms_gradient=rms_gradient,
                    max_step=max_step,
                    rms_step=rms_step,
                    factor=hessian_recalc_near_convergence_factor,
                ):
                    if print_report:
                        print(f"      TS scheduled Hessian recalculation skipped near convergence at step {istep - 1}")
                    fixed_recalc = False
            if fixed_recalc or adaptive_recalc_next_reason is not None:
                if print_report:
                    if fixed_recalc:
                        print(f"      TS scheduled Hessian recalculation at step {istep - 1}")
                    if adaptive_recalc_next_reason is not None:
                        print(f"      TS adaptive Hessian recalculation: {adaptive_recalc_next_reason}")
                hfile = _hessian_file_for_step(hess_file, istep - 1, force_unique=True)
                hess_x = _exact_cartesian_hessian(qcinput, atoms, q, hfile)
                _unused_grad, hess_x = _maybe_project_ts_derivatives(
                    atoms, q, hess=hess_x, project_eckart=project_eckart
                )
                if internal_hessian_correction:
                    hess_q = _internal_hessian_from_cartesian(hess_x, bpg, q=q, ic=ic, grad_q=grad_q)
                elif use_redundant_internals:
                    redundant_system = _current_redundant_system(
                        q,
                        ic,
                        use_redundant_internals=use_redundant_internals,
                    )
                    hess_q = cartesian_hessian_to_redundant(redundant_system, hess_x)
                else:
                    hess_q = _internal_hessian_from_cartesian(hess_x, bpg)
                last_hessian_recalc_step = istep - 1
                adaptive_recalc_next_reason = None

            tracking_vectors = None
            if mode_tracking_coordinates == "mass_weighted_cartesian":
                trial_evals, trial_evecs = np.linalg.eigh(0.5 * (hess_q + hess_q.T))
                tracking_vectors = _mass_weighted_cartesian_modes_from_internal(trial_evecs, bpg, atoms)
                del trial_evals, trial_evecs
            dq, trial_mode, _mode_idx, _evals, trial_negative_modes, trial_tracking_mode, mode_diag = _partitioned_rfo_step(
                hess_q,
                grad_q,
                previous_mode=reaction_mode,
                reaction_direction=reaction_direction if reaction_mode is None and not reaction_mode_is_lowest else None,
                tracking_vectors=tracking_vectors,
                previous_tracking_mode=reaction_tracking_mode,
                reaction_tracking_direction=(
                    reaction_tracking_direction if reaction_tracking_mode is None and not reaction_mode_is_lowest else None
                ),
                follow_lowest=reaction_mode_is_lowest,
                trust_radius=trust_radius,
                repair_hessian=repair_ts_hessian,
                hessian_eigenvalue_floor=ts_hessian_eigenvalue_floor,
            )
            trial_q, trial_energy, trial_grad_x, trial_grad_q, trial_bpg, cart_step, rho, accepted_dq = _ts_accept_internal_step(
                qcinput,
                atoms,
                q,
                energy,
                grad_q,
                hess_q,
                dq,
                qs,
                ic,
                bpg,
                best_fit_iters,
                best_fit_rms_tol=best_fit_rms_tol,
                project_eckart=project_eckart,
                min_gradient_improvement=min_gradient_improvement,
                cartesian_step_max=trust_radius,
                use_redundant_internals=use_redundant_internals,
            )
            reaction_mode = trial_mode
            reaction_tracking_mode = trial_tracking_mode
            if _mode_idx == int(np.argmin(_evals)):
                reaction_mode_is_lowest = True
            negative_modes = trial_negative_modes

            energy_change = trial_energy - energy
            if print_report:
                _print_ts_step(
                    istep, trial_energy, energy_change, cart_step, trial_grad_x,
                    trust_radius, negative_modes, mode_diag
                )

            if adaptive_hessian_recalc:
                if _ts_hessian_recalc_allowed(
                    istep, last_hessian_recalc_step, adaptive_hessian_recalc_cooldown
                ):
                    adaptive_recalc_next_reason = _ts_adaptive_hessian_reason(
                        mode_diag=mode_diag,
                        rho=rho,
                        negative_modes=negative_modes,
                        previous_negative_modes=previous_negative_modes,
                        trust_radius=trust_radius,
                        initial_trust_radius=initial_trust_radius,
                        trust_radius_min=trust_radius_min,
                        overlap_min=adaptive_hessian_overlap_min,
                        rho_min=adaptive_hessian_rho_min,
                        rho_max=adaptive_hessian_rho_max,
                        trust_fraction=adaptive_hessian_trust_fraction,
                        negative_mode_change=adaptive_hessian_on_negative_mode_change,
                        trust_collapse=adaptive_hessian_on_trust_collapse,
                    )
                else:
                    adaptive_recalc_next_reason = None
            previous_negative_modes = negative_modes

            if traj_handle is not None:
                _write_opt_frame(traj_handle, atoms, trial_q, istep, trial_energy, trial_grad_x)

            if _ts_converged(
                energy_change,
                cart_step,
                trial_grad_x,
                energy_tol=energy_tol,
                max_gradient=max_gradient,
                rms_gradient=rms_gradient,
                max_step=max_step,
                rms_step=rms_step,
            ):
                result_hess = None
                result_negative_modes = negative_modes
                if final_hessian:
                    result_hess = _final_ts_hessian(
                        qcinput,
                        atoms,
                        trial_q,
                        _final_hessian_file(hess_file),
                        project_eckart=project_eckart,
                    )
                    if internal_hessian_correction:
                        result_hess_q = _internal_hessian_from_cartesian(
                            result_hess, trial_bpg, q=trial_q, ic=ic, grad_q=trial_grad_q
                        )
                    elif use_redundant_internals:
                        trial_redundant_system = _current_redundant_system(
                            trial_q,
                            ic,
                            use_redundant_internals=use_redundant_internals,
                        )
                        result_hess_q = cartesian_hessian_to_redundant(trial_redundant_system, result_hess)
                    else:
                        result_hess_q = _internal_hessian_from_cartesian(result_hess, trial_bpg)
                    result_negative_modes = _count_negative_modes(result_hess_q)
                if print_report:
                    _print_optimization_status(True, "Transition-state convergence reached.")
                return OptimizationResult(
                    atoms=atoms,
                    q=trial_q,
                    energy=trial_energy,
                    converged=True,
                    nsteps=istep,
                    method="P-RFO",
                    backend_optimizer="smite",
                    trajectory_file=trajectory_file,
                    gradient=trial_grad_x,
                    message="Transition-state convergence reached.",
                    coordinates="internal",
                    target="transition_state",
                    negative_modes=result_negative_modes,
                    reaction_mode=reaction_mode,
                    hessian=result_hess,
                )

            y = trial_grad_q - grad_q
            hess_q, _updated = _bofill_update_hessian(hess_q, accepted_dq, y)

            q = trial_q
            qs = compute_internals(q, ic)
            bpg = trial_bpg
            redundant_system = _current_redundant_system(
                q,
                ic,
                use_redundant_internals=use_redundant_internals,
            )
            energy = trial_energy
            grad_x = trial_grad_x
            grad_q = trial_grad_q
            previous_cart_step = cart_step
    finally:
        if traj_handle is not None:
            traj_handle.close()

    if print_report:
        _print_optimization_status(False, "Maximum TS optimization steps reached without convergence.")
    return OptimizationResult(
        atoms=atoms,
        q=q,
        energy=energy,
        converged=False,
        nsteps=maxstep,
        method="P-RFO",
        backend_optimizer="smite",
        trajectory_file=trajectory_file,
        gradient=grad_x,
        message="Maximum TS optimization steps reached without convergence.",
        coordinates="internal",
        target="transition_state",
        negative_modes=negative_modes,
        reaction_mode=reaction_mode,
    )
