from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimizer.common import (
    _atoms_q_from_xyz,
    _bpg_for_internal_coordinates,
    _current_redundant_system,
    _energy,
    _gradient,
    _initial_internal_coordinate_state,
    _print_optimization_status,
    _write_opt_frame,
)
from optimizer.internal_coords import (
    best_fit_dq_to_cart,
    compute_internals,
    default_connectivity_model,
    initial_internal_hessian_from_model,
    internal_gradient,
)
from optimizer.redundant_internals import bpg_from_redundant, initial_redundant_hessian_from_model
from optimizer.reporter import make_reporter
from qchem_interfaces.qchem_validation import validate_qchem_input
from utils.constants import HARTREE_TO_KJMOL


@dataclass
class SpinCrossingPoint:
    step: int
    q: np.ndarray
    energy_a: float
    energy_b: float
    gradient_a: np.ndarray
    gradient_b: np.ndarray
    effective_gradient: np.ndarray
    parallel_gradient: np.ndarray
    perpendicular_gradient: np.ndarray
    step_vector: np.ndarray


@dataclass
class SpinCrossingResult:
    atoms: list[str]
    q: np.ndarray
    energy_a: float
    energy_b: float
    converged: bool
    nsteps: int
    method: str
    multiplicities: tuple[int, int]
    trajectory_file: str | None = None
    gradient_a: np.ndarray | None = None
    gradient_b: np.ndarray | None = None
    effective_gradient: np.ndarray | None = None
    points: list[SpinCrossingPoint] | None = None
    message: str = ""
    coordinates: str = "cartesian"
    use_redundant_internals: bool = False

    @property
    def energy_gap(self) -> float:
        return float(self.energy_a - self.energy_b)


def _spin_multiplicities_from_qcinput(qcinput, spin_multiplicities=None):
    if spin_multiplicities is None:
        spin_multiplicities = (
            qcinput.get("spin_multiplicities")
            or qcinput.get("spin_mults")
            or qcinput.get("multiplicities")
            or qcinput.get("spinmult")
            or qcinput.get("spin_multiplicity")
        )
    if spin_multiplicities is None and "spinmult1" in qcinput:
        if "spinmult2" in qcinput:
            spin_multiplicities = (qcinput["spinmult1"], qcinput["spinmult2"])
        elif "spinmulti2" in qcinput:
            spin_multiplicities = (qcinput["spinmult1"], qcinput["spinmulti2"])
    if spin_multiplicities is None:
        raise ValueError(
            "Spin-crossing optimization requires two spin multiplicities. "
            "Pass spin_multiplicities=(m1, m2) or set qcinput['spin_multiplicities']."
        )
    try:
        mult_a, mult_b = spin_multiplicities
    except (TypeError, ValueError) as exc:
        raise ValueError("spin_multiplicities must be a two-integer sequence, e.g. (1, 3)") from exc
    mult_a = int(mult_a)
    mult_b = int(mult_b)
    if mult_a < 1 or mult_b < 1:
        raise ValueError("Spin multiplicities must be >= 1")
    return mult_a, mult_b


def _state_qcinput(qcinput, multiplicity, label):
    state = dict(qcinput)
    state["multiplicity"] = int(multiplicity)
    state.pop("spin_multiplicities", None)
    state.pop("spin_mults", None)
    state.pop("multiplicities", None)
    state.pop("spinmult", None)
    state.pop("spin_multiplicity", None)
    state.pop("spinmult1", None)
    state.pop("spinmult2", None)
    state.pop("spinmulti2", None)
    state.pop("_last_energy_cache", None)
    state.pop("_qchem_validated", None)
    state.pop("_unknown_qchem_keys", None)
    scratch_dir = state.get("scratch_dir")
    if scratch_dir:
        state["scratch_dir"] = f"{scratch_dir}_{label}_mult{int(multiplicity)}"
    return validate_qchem_input(state)


def _effective_gradient(energy_a, energy_b, grad_a, grad_b, *, fac_parallel=1.0, fac_perpendicular=140.0):
    grad_a = np.asarray(grad_a, dtype=float).reshape(-1)
    grad_b = np.asarray(grad_b, dtype=float).reshape(-1)
    dgrad = grad_a - grad_b
    denom = float(np.dot(dgrad, dgrad))
    if denom <= 1.0e-24:
        raise ValueError("Cannot build spin-crossing effective gradient because state gradients are parallel/equal")

    scale = float(np.dot(grad_a, dgrad)) / denom
    perpendicular = float(energy_a - energy_b) * dgrad
    parallel = grad_a - scale * dgrad
    effective = float(fac_perpendicular) * perpendicular + float(fac_parallel) * parallel
    return parallel, perpendicular, effective


def _step_limit(step, *, max_component=0.1):
    step = np.asarray(step, dtype=float).copy()
    stpmax = len(step) * float(max_component)
    norm = float(np.linalg.norm(step))
    if norm > stpmax and norm > 0.0:
        step *= stpmax / norm

    largest = float(np.max(np.abs(step))) if step.size else 0.0
    if largest > max_component:
        step *= float(max_component) / largest
    return step


def _bb_step(eq, x_1, x_2, g_1, g_2):
    dgrad = g_2 - g_1
    dx = x_2 - x_1
    xg = float(np.dot(dgrad, dx))
    if eq == 1:
        denom = float(np.dot(dgrad, dgrad))
        alpha = xg / denom if abs(denom) > 1.0e-16 else 0.7
    elif eq == 2:
        alpha = float(np.dot(dx, dx)) / xg if abs(xg) > 1.0e-16 else 0.7
    else:
        raise ValueError("BB step equation must be 1 or 2")
    if not np.isfinite(alpha) or alpha <= 0.0:
        alpha = 0.7
    alpha = min(float(alpha), 10.0)
    return -alpha * g_2


def _bfgs_inverse_update(inv_hess, dx, dgrad, *, damped=False):
    overlap = float(np.dot(dgrad, dx))
    scale = np.linalg.norm(dx) * np.linalg.norm(dgrad)
    if overlap <= max(1.0e-14, 1.0e-8 * scale):
        return inv_hess, False
    eye = np.eye(len(dx))
    xg = np.outer(dx, dgrad)
    gx = np.outer(dgrad, dx)
    updated = (eye - xg / overlap) @ inv_hess @ (eye - gx / overlap) + np.outer(dx, dx) / overlap
    if not np.all(np.isfinite(updated)):
        return inv_hess, False
    return 0.5 * (updated + updated.T), True


def _psb_inverse_update(inv_hess, dx, dgrad):
    y = dx - inv_hess @ dgrad
    yy = float(np.dot(y, y))
    yd = float(np.dot(y, dgrad))
    dd = float(np.dot(dgrad, dgrad))
    if dd <= 1.0e-14:
        return inv_hess, False
    updated = (
        inv_hess
        + (np.outer(y, dgrad) + np.outer(dgrad, y)) / dd
        - (yd / (dd * dd)) * np.outer(dgrad, dgrad)
    )
    if yy <= 1.0e-24 or not np.all(np.isfinite(updated)):
        return inv_hess, False
    return 0.5 * (updated + updated.T), True


def _bfgs_old_inverse_update(inv_hess, dx, dgrad):
    h_dg = inv_hess @ dgrad
    fac = float(np.dot(dgrad, dx))
    fae = float(np.dot(dgrad, h_dg))
    if abs(fac) <= 1.0e-14 or abs(fae) <= 1.0e-14:
        return inv_hess, False
    fac = 1.0 / fac
    fad = 1.0 / fae
    w = fac * dx - fad * h_dg
    updated = inv_hess + fac * np.outer(dx, dx) - fad * np.outer(h_dg, h_dg) + fae * np.outer(w, w)
    if not np.all(np.isfinite(updated)):
        return inv_hess, False
    return 0.5 * (updated + updated.T), True


def _dfp_inverse_update(inv_hess, dx, dgrad):
    h_dg = inv_hess @ dgrad
    overlap = float(np.dot(dx, dgrad))
    ghg = float(np.dot(dgrad, h_dg))
    if abs(overlap) <= 1.0e-14 or abs(ghg) <= 1.0e-14:
        return inv_hess, False
    updated = inv_hess + np.outer(dx, dx) / overlap - np.outer(h_dg, h_dg) / ghg
    if not np.all(np.isfinite(updated)):
        return inv_hess, False
    return 0.5 * (updated + updated.T), True


def _sr1_inverse_update(inv_hess, dx, dgrad):
    y = dx - inv_hess @ dgrad
    overlap = float(np.dot(y, dgrad))
    denom_scale = np.linalg.norm(y) * np.linalg.norm(dgrad)
    if abs(overlap) <= max(1.0e-14, 1.0e-8 * denom_scale):
        return inv_hess, False
    updated = inv_hess + np.outer(y, y) / overlap
    if not np.all(np.isfinite(updated)):
        return inv_hess, False
    return 0.5 * (updated + updated.T), True


def _broyden_inverse_update(inv_hess, dx, dgrad):
    # Good Broyden inverse update satisfying H_{k+1} dG = dX.
    h_dg = inv_hess @ dgrad
    y = dx - h_dg
    denom = float(np.dot(dgrad, dgrad))
    if denom <= 1.0e-14:
        return inv_hess, False
    updated = inv_hess + np.outer(y, dgrad) / denom
    if not np.all(np.isfinite(updated)):
        return inv_hess, False
    return updated, True


def _bofill_inverse_update(inv_hess, dx, dgrad):
    sr1_hess, sr1_ok = _sr1_inverse_update(inv_hess, dx, dgrad)
    psb_hess, psb_ok = _psb_inverse_update(inv_hess, dx, dgrad)
    if not sr1_ok:
        return psb_hess, psb_ok
    if not psb_ok:
        return sr1_hess, sr1_ok
    y = dx - inv_hess @ dgrad
    yy = float(np.dot(y, y))
    dd = float(np.dot(dgrad, dgrad))
    yd = float(np.dot(y, dgrad))
    phi = (yd * yd) / (yy * dd) if yy > 1.0e-24 and dd > 1.0e-24 else 0.0
    phi = min(1.0, max(0.0, phi))
    updated = phi * sr1_hess + (1.0 - phi) * psb_hess
    if not np.all(np.isfinite(updated)):
        return inv_hess, False
    return 0.5 * (updated + updated.T), True


def _quasi_newton_step(method, x_1, x_2, g_1, g_2, inv_hess):
    method = str(method).upper()
    dx = x_2 - x_1
    dgrad = g_2 - g_1
    updated = True
    if method in {"BFGS", "DAMPED_BFGS", "BFGS_DAMPED"}:
        inv_hess, updated = _bfgs_inverse_update(inv_hess, dx, dgrad, damped=True)
        step = -(inv_hess @ g_2)
    elif method in {"PLAIN_BFGS", "BFGS_PLAIN"}:
        inv_hess, updated = _bfgs_inverse_update(inv_hess, dx, dgrad, damped=False)
        step = -(inv_hess @ g_2)
    elif method == "BFGS_OLD":
        inv_hess, updated = _bfgs_old_inverse_update(inv_hess, dx, dgrad)
        step = -(inv_hess @ g_2)
    elif method in {"DFP"}:
        inv_hess, updated = _dfp_inverse_update(inv_hess, dx, dgrad)
        step = -(inv_hess @ g_2)
    elif method in {"SR1"}:
        inv_hess, updated = _sr1_inverse_update(inv_hess, dx, dgrad)
        step = -(inv_hess @ g_2)
    elif method in {"BROYDEN"}:
        inv_hess, updated = _broyden_inverse_update(inv_hess, dx, dgrad)
        step = -(inv_hess @ g_2)
    elif method in {"BOFILL"}:
        inv_hess, updated = _bofill_inverse_update(inv_hess, dx, dgrad)
        step = -(inv_hess @ g_2)
    elif method in {"PSB"}:
        inv_hess, updated = _psb_inverse_update(inv_hess, dx, dgrad)
        step = -(inv_hess @ g_2)
    elif method == "BBGRAD1":
        step = _bb_step(1, x_1, x_2, g_1, g_2)
    elif method == "BBGRAD2":
        step = _bb_step(2, x_1, x_2, g_1, g_2)
    else:
        raise ValueError(
            "method must be one of: BBGrad1, BBGrad2, BFGS, Plain_BFGS, "
            "BFGS_old, DFP, SR1, PSB, Bofill, Broyden"
        )
    return step, inv_hess, updated


def _line_search_internal(
    qcinput_a,
    qcinput_b,
    atoms,
    q,
    energy_a,
    energy_b,
    dq,
    qs,
    ic,
    bpg,
    *,
    best_fit_iters,
    best_fit_rms_tol,
    min_scale=1.0e-4,
):
    current_gap = abs(float(energy_a - energy_b))
    scale = 1.0
    while scale >= min_scale:
        trial_q = best_fit_dq_to_cart(
            q,
            qs,
            scale * dq,
            ic,
            bpg,
            n_iter=best_fit_iters,
            rms_tol=best_fit_rms_tol,
        )
        trial_energy_a = _energy(qcinput_a, trial_q, atoms)
        trial_energy_b = _energy(qcinput_b, trial_q, atoms)
        if abs(trial_energy_a - trial_energy_b) <= current_gap:
            return trial_q, scale * dq, trial_q - q, scale, True
        scale *= 0.5
    return np.asarray(q, dtype=float).copy(), np.zeros_like(dq), np.zeros_like(q), 0.0, False


def _line_search_cartesian(
    qcinput_a, qcinput_b, atoms, q, energy_a, energy_b, step, *, min_scale=1.0e-4
):
    """Backtrack a spin-crossing step until the energy gap is reduced."""
    current_gap = abs(float(energy_a - energy_b))
    scale = 1.0
    while scale >= min_scale:
        trial_q = np.asarray(q, dtype=float) + scale * np.asarray(step, dtype=float)
        trial_energy_a = _energy(qcinput_a, trial_q, atoms)
        trial_energy_b = _energy(qcinput_b, trial_q, atoms)
        if (
            np.isfinite(trial_energy_a)
            and np.isfinite(trial_energy_b)
            and abs(trial_energy_a - trial_energy_b) <= current_gap
        ):
            return trial_q, scale * step, True
        scale *= 0.5
    return np.asarray(q, dtype=float).copy(), np.zeros_like(step), False


def _converged(energy_gap, step, effective_gradient, *, energy_tol, max_step, rms_step, max_gradient, rms_gradient):
    step = np.asarray(step, dtype=float).reshape(-1)
    effective_gradient = np.asarray(effective_gradient, dtype=float).reshape(-1)
    step_max = float(np.max(np.abs(step))) if step.size else 0.0
    step_rms = float(np.linalg.norm(step) / np.sqrt(len(step))) if step.size else 0.0
    grad_max = float(np.max(np.abs(effective_gradient))) if effective_gradient.size else 0.0
    grad_rms = (
        float(np.linalg.norm(effective_gradient) / np.sqrt(len(effective_gradient)))
        if effective_gradient.size
        else 0.0
    )
    return (
        abs(float(energy_gap)) <= energy_tol
        and step_max <= max_step
        and step_rms <= rms_step
        and grad_max <= max_gradient
        and grad_rms <= rms_gradient
    )


def _print_header(energy_tol, max_step, rms_step, max_gradient, rms_gradient):
    print("Spin-crossing convergence tolerances:")
    print(f"  Energy Gap          TolE    .... {energy_tol: .2e} Eh")
    print(f"  Max. Eff. Gradient  TolMAXG .... {max_gradient: .2e} Eh/bohr")
    print(f"  RMS Eff. Gradient   TolRMSG .... {rms_gradient: .2e} Eh/bohr")
    print(f"  Max Step            TolMAXD .... {max_step: .2e} bohr")
    print(f"  RMS Step            TolRMSD .... {rms_step: .2e} bohr")
    header_line = (
        "%5s %18s %18s %18s %11s %11s %11s %11s"
        % ("step", "Ea[Eh]", "Eb[Eh]", "dE[kJ/mol]", "max_step", "rms_step", "max_geff", "rms_geff")
    )
    print()
    _print_threshold_guide(energy_tol, max_step, rms_step, max_gradient, rms_gradient, header_line)
    print(header_line)


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


def _print_threshold_guide(energy_tol, max_step, rms_step, max_gradient, rms_gradient, header_line):
    values = (
        _format_threshold_guide_value(energy_tol * HARTREE_TO_KJMOL),
        _format_threshold_guide_value(max_step),
        _format_threshold_guide_value(rms_step),
        _format_threshold_guide_value(max_gradient),
        _format_threshold_guide_value(rms_gradient),
    )
    columns = ("dE[kJ/mol]", "max_step", "rms_step", "max_geff", "rms_geff")
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


def _print_step(istep, energy_a, energy_b, step, effective_gradient):
    print(
        "%5d %18.8f %18.8f %18.6f %11.2e %11.2e %11.2e %11.2e"
        % (
            istep,
            energy_a,
            energy_b,
            (energy_a - energy_b) * HARTREE_TO_KJMOL,
            np.max(np.abs(step)),
            np.linalg.norm(step) / np.sqrt(len(step)),
            np.max(np.abs(effective_gradient)),
            np.linalg.norm(effective_gradient) / np.sqrt(len(effective_gradient)),
        )
    )


def optimize_spin_crossing(
    qcinput,
    xyz,
    *,
    spin_multiplicities=None,
    method="BFGS",
    maxstep=40,
    energy_tol=5.0e-5,
    max_step=4.0e-3,
    rms_step=2.5e-3,
    max_gradient=7.0e-4,
    rms_gradient=5.0e-4,
    step_limit=True,
    step_max_component=0.1,
    initial_inverse_hessian=0.7,
    coordinates="cartesian",
    max_step_internal=0.2,
    best_fit_iters=5,
    best_fit_rms_tol=1.0e-7,
    connectivity_kcn=16.0,
    connectivity_facmin=0.45,
    connect_fragments=True,
    use_redundant_internals=False,
    internal_hessian_model="simple",
    linear_bends=False,
    linear_bend_threshold_degrees=None,
    fac_parallel=1.0,
    fac_perpendicular=140.0,
    trajectory_file="spincross_traj.xyz",
    print_report=True,
    reporter=None,
    verbosity=1,
    log_file=None,
    log_append=False,
    _reporter_active=False,
):
    """Optimize a minimum-energy crossing point between two spin states.

    The two states use identical qchem settings except for ``multiplicity``.
    Supply the pair as ``spin_multiplicities=(m1, m2)`` or
    ``qcinput["spin_multiplicities"] = (m1, m2)``.

    This follows the old SpinCross MECP effective-gradient algorithm, but uses
    the same public input style as the other optimizer modules: ``qcinput`` plus
    an XYZ geometry string/path.  Coordinates are in the package's standard
    bohr/Hartree units internally.
    """
    reporter = make_reporter(
        print_report=print_report,
        reporter=reporter,
        verbosity=verbosity,
        log_file=log_file,
        append=log_append,
    )
    if not _reporter_active:
        options = locals().copy()
        options.pop("qcinput")
        options.pop("xyz")
        options["reporter"] = reporter
        options["print_report"] = print_report and reporter.should_report()
        options["_reporter_active"] = True
        with reporter.capture_prints():
            return optimize_spin_crossing(qcinput, xyz, **options)

    mult_a, mult_b = _spin_multiplicities_from_qcinput(qcinput, spin_multiplicities)
    qcinput_a = _state_qcinput(qcinput, mult_a, "state_a")
    qcinput_b = _state_qcinput(qcinput, mult_b, "state_b")
    atoms, q = _atoms_q_from_xyz(xyz)
    q = np.asarray(q, dtype=float).reshape(-1).copy()
    ndim = q.size
    if ndim != 3 * len(atoms):
        raise ValueError("Coordinate length must be 3 * number of atoms")

    method_label = str(method)
    coordinates = str(coordinates).lower()
    if coordinates not in {"cartesian", "internal"}:
        raise ValueError("coordinates must be 'cartesian' or 'internal'")
    if use_redundant_internals and coordinates != "internal":
        raise ValueError("use_redundant_internals=True requires coordinates='internal'")
    if len(atoms) < 2 and coordinates == "internal":
        coordinates = "cartesian"

    x_1 = q.copy()
    x_2 = q.copy()
    geff_1 = np.zeros(ndim, dtype=float)
    ic = qs = bpg = redundant_system = None
    opt_coord_1 = opt_coord_2 = None
    opt_geff_1 = None
    if coordinates == "internal":
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
        )
        if ic.nint == 0:
            coordinates = "cartesian"
        else:
            opt_coord_1 = qs.copy()
            opt_coord_2 = qs.copy()
            opt_geff_1 = np.zeros(ic.nint, dtype=float)

    inv_dim = ndim if coordinates == "cartesian" else ic.nint
    internal_hessian_model = str(internal_hessian_model).lower()
    use_model_inverse_hessian = (
        coordinates == "internal"
        and internal_hessian_model not in {"simple", "constant", "default"}
    )
    if use_model_inverse_hessian:
        if use_redundant_internals:
            initial_hess = initial_redundant_hessian_from_model(
                redundant_system,
                model=internal_hessian_model,
            )
        else:
            initial_hess = initial_internal_hessian_from_model(
                ic,
                q_values=qs,
                model=internal_hessian_model,
            )
        inv_hess = np.linalg.pinv(initial_hess, rcond=1.0e-8) * float(initial_inverse_hessian)
    else:
        inv_hess = np.eye(inv_dim, dtype=float) * float(initial_inverse_hessian)
    points: list[SpinCrossingPoint] = []

    if print_report:
        print("Spin-crossing optimization")
        print(f"Multiplicities: {mult_a}, {mult_b}")
        print(f"Update method: {method_label}")
        print(f"Coordinates: {'Pulay redundant internals' if use_redundant_internals else coordinates}")
        if coordinates == "internal":
            print(f"Initial internal Hessian model: {internal_hessian_model}")
        if coordinates == "internal":
            print(
                f"Internal coordinates: {ic.nbonds} bonds, {ic.nangles} angles, "
                f"{ic.nlinear_bends} linear bends, {ic.ndihedrals} dihedrals, "
                f"{ic.nimpropers} impropers"
            )
        _print_header(energy_tol, max_step, rms_step, max_gradient, rms_gradient)

    traj_handle = open(trajectory_file, "w", encoding="utf-8") if trajectory_file else None
    try:
        for istep in range(1, maxstep + 1):
            energy_a = _energy(qcinput_a, x_2, atoms)
            energy_b = _energy(qcinput_b, x_2, atoms)
            grad_a = _gradient(qcinput_a, x_2, atoms)
            grad_b = _gradient(qcinput_b, x_2, atoms)
            parallel, perpendicular, geff_2 = _effective_gradient(
                energy_a,
                energy_b,
                grad_a,
                grad_b,
                fac_parallel=fac_parallel,
                fac_perpendicular=fac_perpendicular,
            )

            if coordinates == "internal":
                if use_redundant_internals:
                    redundant_system = _current_redundant_system(
                        x_2,
                        ic,
                        use_redundant_internals=use_redundant_internals,
                    )
                    qs = redundant_system.q
                    bpg = bpg_from_redundant(redundant_system)
                else:
                    qs = compute_internals(x_2, ic)
                    bpg = _bpg_for_internal_coordinates(
                        x_2,
                        ic,
                        use_redundant_internals=use_redundant_internals,
                    )
                grad_a_opt = internal_gradient(bpg, grad_a)
                grad_b_opt = internal_gradient(bpg, grad_b)
                parallel_opt, perpendicular_opt, geff_opt_2 = _effective_gradient(
                    energy_a,
                    energy_b,
                    grad_a_opt,
                    grad_b_opt,
                    fac_parallel=fac_parallel,
                    fac_perpendicular=fac_perpendicular,
                )
                if istep == 1:
                    if use_model_inverse_hessian:
                        opt_step = -(inv_hess @ geff_opt_2)
                    else:
                        opt_step = -0.7 * geff_opt_2
                else:
                    opt_step, inv_hess, _updated = _quasi_newton_step(
                        method_label,
                        opt_coord_1,
                        opt_coord_2,
                        opt_geff_1,
                        geff_opt_2,
                        inv_hess,
                    )
                opt_norm = float(np.linalg.norm(opt_step))
                if opt_norm > max_step_internal and opt_norm > 0.0:
                    opt_step *= float(max_step_internal) / opt_norm
                x_3, accepted_opt_step, step, _scale, accepted = _line_search_internal(
                    qcinput_a,
                    qcinput_b,
                    atoms,
                    x_2,
                    energy_a,
                    energy_b,
                    opt_step,
                    qs,
                    ic,
                    bpg,
                    best_fit_iters=best_fit_iters,
                    best_fit_rms_tol=best_fit_rms_tol,
                )
            else:
                if istep == 1:
                    step = -0.7 * geff_2
                else:
                    step, inv_hess, _updated = _quasi_newton_step(method_label, x_1, x_2, geff_1, geff_2, inv_hess)
                if step_limit:
                    step = _step_limit(step, max_component=step_max_component)
                x_3, step, accepted = _line_search_cartesian(
                    qcinput_a, qcinput_b, atoms, x_2, energy_a, energy_b, step
                )

            if not accepted:
                message = "Spin-crossing line search failed to reduce the state-energy gap"
                if print_report:
                    _print_optimization_status(False, message)
                return SpinCrossingResult(
                    atoms=atoms, q=x_2, energy_a=energy_a, energy_b=energy_b,
                    converged=False, nsteps=istep - 1, method=method_label,
                    multiplicities=(mult_a, mult_b), trajectory_file=trajectory_file,
                    gradient_a=grad_a, gradient_b=grad_b, effective_gradient=geff_2,
                    points=points, message=message, coordinates=coordinates,
                    use_redundant_internals=bool(use_redundant_internals),
                )

            # Convergence, reporting, and trajectory records must all describe
            # the same accepted geometry, not the geometry before its step.
            trial_energy_a = _energy(qcinput_a, x_3, atoms)
            trial_energy_b = _energy(qcinput_b, x_3, atoms)
            trial_grad_a = _gradient(qcinput_a, x_3, atoms)
            trial_grad_b = _gradient(qcinput_b, x_3, atoms)
            trial_parallel, trial_perpendicular, trial_geff = _effective_gradient(
                trial_energy_a,
                trial_energy_b,
                trial_grad_a,
                trial_grad_b,
                fac_parallel=fac_parallel,
                fac_perpendicular=fac_perpendicular,
            )

            if print_report:
                _print_step(istep, trial_energy_a, trial_energy_b, step, trial_geff)
            if traj_handle is not None:
                _write_opt_frame(
                    traj_handle, atoms, x_3, istep,
                    0.5 * (trial_energy_a + trial_energy_b), trial_geff,
                )

            point = SpinCrossingPoint(
                step=istep,
                q=x_3.copy(),
                energy_a=trial_energy_a,
                energy_b=trial_energy_b,
                gradient_a=trial_grad_a.copy(),
                gradient_b=trial_grad_b.copy(),
                effective_gradient=trial_geff.copy(),
                parallel_gradient=trial_parallel.copy(),
                perpendicular_gradient=trial_perpendicular.copy(),
                step_vector=step.copy(),
            )
            points.append(point)

            if _converged(
                trial_energy_a - trial_energy_b,
                step,
                trial_geff,
                energy_tol=energy_tol,
                max_step=max_step,
                rms_step=rms_step,
                max_gradient=max_gradient,
                rms_gradient=rms_gradient,
            ):
                if print_report:
                    _print_optimization_status(True, "Spin-crossing convergence reached.")
                return SpinCrossingResult(
                    atoms=atoms,
                    q=x_3,
                    energy_a=trial_energy_a,
                    energy_b=trial_energy_b,
                    converged=True,
                    nsteps=istep,
                    method=method_label,
                    multiplicities=(mult_a, mult_b),
                    trajectory_file=trajectory_file,
                    gradient_a=trial_grad_a,
                    gradient_b=trial_grad_b,
                    effective_gradient=trial_geff,
                    points=points,
                    message="Spin-crossing convergence reached.",
                    coordinates=coordinates,
                    use_redundant_internals=bool(use_redundant_internals),
                )

            x_1 = x_2
            x_2 = x_3
            geff_1 = geff_2
            if coordinates == "internal":
                opt_coord_1 = opt_coord_2
                opt_coord_2 = qs + accepted_opt_step
                opt_geff_1 = geff_opt_2
    finally:
        if traj_handle is not None:
            traj_handle.close()

    energy_a = _energy(qcinput_a, x_2, atoms)
    energy_b = _energy(qcinput_b, x_2, atoms)
    grad_a = _gradient(qcinput_a, x_2, atoms)
    grad_b = _gradient(qcinput_b, x_2, atoms)
    _parallel, _perpendicular, geff = _effective_gradient(
        energy_a,
        energy_b,
        grad_a,
        grad_b,
        fac_parallel=fac_parallel,
        fac_perpendicular=fac_perpendicular,
    )
    if print_report:
        _print_optimization_status(False, "Maximum spin-crossing optimization steps reached without convergence.")
    return SpinCrossingResult(
        atoms=atoms,
        q=x_2,
        energy_a=energy_a,
        energy_b=energy_b,
        converged=False,
        nsteps=maxstep,
        method=method_label,
        multiplicities=(mult_a, mult_b),
        trajectory_file=trajectory_file,
        gradient_a=grad_a,
        gradient_b=grad_b,
        effective_gradient=geff,
        points=points,
        message="Maximum spin-crossing optimization steps reached without convergence.",
        coordinates=coordinates,
        use_redundant_internals=bool(use_redundant_internals),
    )


optimize_spincrossing = optimize_spin_crossing
