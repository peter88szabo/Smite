from __future__ import annotations

import numpy as np

from optimizer.common import (
    OptimizationResult,
    _bfgs_update_hessian,
    _bpg_for_internal_coordinates,
    _converged,
    _current_redundant_system,
    _energy,
    _gradient,
    _initial_internal_coordinate_state,
    _print_header,
    _print_optimization_status,
    _print_extra_internal_coordinates,
    _print_step,
    _solve_step,
    _write_opt_frame,
    step_limit,
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
    initial_redundant_hessian_from_model,
)
from qchem_interfaces.orcarun import Orca_GeomOpt, makeXYZ
from utils.constants import ANGSTROM_TO_BOHR


def _line_search_cartesian(qcinput, atoms, q, energy, grad, step, *, min_scale=1.0e-4):
    scale = 1.0
    directional = float(np.dot(grad, step))
    while scale >= min_scale:
        trial_q = q + scale * step
        trial_energy = _energy(qcinput, trial_q, atoms)
        if trial_energy <= energy + 1.0e-4 * scale * directional or trial_energy < energy:
            trial_grad = _gradient(qcinput, trial_q, atoms)
            return trial_q, trial_energy, trial_grad, scale * step, scale, True
        scale *= 0.5
    # Never silently accept an uphill step after Armijo backtracking fails.
    return q.copy(), float(energy), np.asarray(grad, dtype=float).copy(), np.zeros_like(step), 0.0, False

def _smite_cartesian_optimize_geometry(
    qcinput,
    atoms,
    q,
    *,
    method="BFGS",
    maxstep=100,
    energy_tol=5.0e-5,
    max_step=4.0e-3,
    rms_step=2.5e-3,
    max_gradient=7.0e-4,
    rms_gradient=5.0e-4,
    step_max_component=0.1,
    trajectory_file="geomopt_traj.xyz",
    print_report=True,
):
    method = method.upper()
    if method not in {"BFGS", "STEEPEST_DESCENT", "TWOPOINTGRAD_1", "TWOPOINTGRAD_2"}:
        raise ValueError("method must be one of: BFGS, STEEPEST_DESCENT, TWOPOINTGRAD_1, TWOPOINTGRAD_2")

    q = np.asarray(q, dtype=float).reshape(-1).copy()
    atoms = list(atoms)
    ndim = len(q)
    h_inv = np.eye(ndim)

    energy = _energy(qcinput, q, atoms)
    grad = _gradient(qcinput, q, atoms)
    previous_q = None
    previous_grad = None

    if print_report:
        _print_header(energy_tol, max_step, rms_step, max_gradient, rms_gradient)

    traj_handle = open(trajectory_file, "w", encoding="utf-8") if trajectory_file else None
    try:
        if traj_handle is not None:
            _write_opt_frame(traj_handle, atoms, q, 0, energy, grad)

        for istep in range(1, maxstep + 1):
            if method == "STEEPEST_DESCENT" or previous_q is None:
                step = step_limit(-0.7 * grad, max_component=step_max_component)
            elif method.startswith("TWOPOINTGRAD"):
                dG = grad - previous_grad
                dX = q - previous_q
                denom1 = float(np.dot(dG, dG))
                denom2 = float(np.dot(dG, dX))
                if method == "TWOPOINTGRAD_1":
                    alpha = denom2 / denom1 if abs(denom1) > 1.0e-16 else 0.7
                else:
                    alpha = float(np.dot(dX, dX)) / denom2 if abs(denom2) > 1.0e-16 else 0.7
                if not np.isfinite(alpha) or alpha <= 0.0:
                    alpha = 0.7
                alpha = min(alpha, 10.0)
                step = step_limit(-alpha * grad, max_component=step_max_component)
            else:
                step = step_limit(-h_inv @ grad, max_component=step_max_component)

            # A non-descent quasi-Newton direction cannot satisfy Armijo.
            if float(np.dot(grad, step)) >= 0.0:
                step = step_limit(-grad, max_component=step_max_component)

            trial_q, trial_energy, trial_grad, accepted_step, _scale, accepted = _line_search_cartesian(
                qcinput, atoms, q, energy, grad, step
            )
            if not accepted:
                message = "Line search failed to find a downhill Cartesian step"
                if print_report:
                    _print_optimization_status(False, message)
                return OptimizationResult(
                    atoms=atoms, q=q, energy=energy, converged=False, nsteps=istep - 1,
                    method=method, backend_optimizer="smite", trajectory_file=trajectory_file,
                    gradient=grad, message=message, coordinates="cartesian",
                )
            energy_change = trial_energy - energy

            if print_report:
                _print_step(istep, trial_energy, energy_change, accepted_step, trial_grad)

            if traj_handle is not None:
                _write_opt_frame(traj_handle, atoms, trial_q, istep, trial_energy, trial_grad)

            converged = _converged(
                energy_change,
                accepted_step,
                trial_grad,
                max_gradient=max_gradient,
                rms_gradient=rms_gradient,
                max_step=max_step,
                rms_step=rms_step,
                energy_tol=energy_tol,
            )

            if method == "BFGS":
                y = trial_grad - grad
                s = accepted_step
                ys = float(np.dot(y, s))
                if ys > 1.0e-14:
                    rho = 1.0 / ys
                    ident = np.eye(ndim)
                    sy = np.outer(s, y)
                    ys_outer = np.outer(y, s)
                    h_inv = (ident - rho * sy) @ h_inv @ (ident - rho * ys_outer) + rho * np.outer(s, s)

            previous_q = q
            previous_grad = grad
            q = trial_q
            energy = trial_energy
            grad = trial_grad

            if converged:
                if print_report:
                    _print_optimization_status(True, "Convergence reached.")
                return OptimizationResult(
                    atoms=atoms,
                    q=q,
                    energy=energy,
                    converged=True,
                    nsteps=istep,
                    method=method,
                    backend_optimizer="smite",
                    trajectory_file=trajectory_file,
                    gradient=grad,
                    message="Convergence reached.",
                    coordinates="cartesian",
                )
    finally:
        if traj_handle is not None:
            traj_handle.close()

    if print_report:
        _print_optimization_status(False, "Maximum optimization steps reached without convergence.")
    return OptimizationResult(
        atoms=atoms,
        q=q,
        energy=energy,
        converged=False,
        nsteps=maxstep,
        method=method,
        backend_optimizer="smite",
        trajectory_file=trajectory_file,
        gradient=grad,
        message="Maximum optimization steps reached without convergence.",
        coordinates="cartesian",
    )

def _line_search_internal(
    qcinput,
    atoms,
    q,
    energy,
    grad_x,
    dq,
    qs,
    ic,
    bpg,
    *,
    best_fit_iters,
    best_fit_rms_tol=1.0e-7,
    min_scale=1.0e-4,
):
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
        trial_energy = _energy(qcinput, trial_q, atoms)
        if trial_energy < energy:
            trial_grad_x = _gradient(qcinput, trial_q, atoms)
            return trial_q, trial_energy, trial_grad_x, scale * dq, trial_q - q, scale, True
        scale *= 0.5
    return (
        np.asarray(q, dtype=float).copy(), float(energy), np.asarray(grad_x, dtype=float).copy(),
        np.zeros_like(dq), np.zeros_like(q), 0.0, False,
    )

def _smite_internal_optimize_geometry(
    qcinput,
    atoms,
    q,
    *,
    method="BFGS",
    maxstep=100,
    energy_tol=5.0e-5,
    max_step=4.0e-3,
    rms_step=2.5e-3,
    max_gradient=7.0e-4,
    rms_gradient=5.0e-4,
    max_step_internal=0.2,
    best_fit_iters=5,
    best_fit_rms_tol=1.0e-7,
    trajectory_file="geomopt_traj.xyz",
    print_report=True,
    connectivity_kcn=16.0,
    connectivity_facmin=0.45,
    connect_fragments=True,
    use_redundant_internals=False,
    linear_bends=False,
    linear_bend_threshold_degrees=None,
    extra_bonds=None,
    extra_angles=None,
    extra_dihedrals=None,
    internal_hessian_model="simple",
):
    method = method.upper()
    if method not in {"BFGS", "STEEPEST_DESCENT"}:
        raise ValueError("Internal-coordinate optimization currently supports method='BFGS' or 'STEEPEST_DESCENT'")

    q = np.asarray(q, dtype=float).reshape(-1).copy()
    atoms = list(atoms)
    if len(q) != 3 * len(atoms):
        raise ValueError("Coordinate length must be 3 * number of atoms")
    if len(atoms) < 2:
        energy = _energy(qcinput, q, atoms)
        grad = _gradient(qcinput, q, atoms)
        single_atom_converged = np.linalg.norm(grad) <= max_gradient
        if print_report:
            _print_optimization_status(single_atom_converged, "Single-atom system has no internal coordinates.")
        return OptimizationResult(
            atoms=atoms,
            q=q,
            energy=energy,
            converged=single_atom_converged,
            nsteps=0,
            method=method,
            backend_optimizer="smite",
            trajectory_file=trajectory_file,
            gradient=grad,
            message="Single-atom system has no internal coordinates.",
            coordinates="internal",
        )

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
        return _smite_cartesian_optimize_geometry(
            qcinput,
            atoms,
            q,
            method=method,
            maxstep=maxstep,
            energy_tol=energy_tol,
            max_step=max_step,
            rms_step=rms_step,
            max_gradient=max_gradient,
            rms_gradient=rms_gradient,
            trajectory_file=trajectory_file,
            print_report=print_report,
        )

    energy = _energy(qcinput, q, atoms)
    grad_x = _gradient(qcinput, q, atoms)
    grad_q = internal_gradient(bpg, grad_x)
    if use_redundant_internals and redundant_system is not None:
        hess_q = initial_redundant_hessian_from_model(redundant_system, model=internal_hessian_model)
    else:
        hess_q = initial_internal_hessian_from_model(ic, q_values=qs, model=internal_hessian_model)

    if print_report:
        if use_redundant_internals:
            print("Choice of coordinates: Pulay redundant internal coordinates")
        print(
            f"Internal coordinates: {ic.nbonds} bonds, {ic.nangles} angles, "
            f"{ic.nlinear_bends} linear bends, {ic.ndihedrals} dihedrals, "
            f"{ic.nimpropers} impropers"
        )
        _print_extra_internal_coordinates(extra_bonds, extra_angles, extra_dihedrals)
        print(f"Initial internal Hessian model: {internal_hessian_model}")
        _print_header(energy_tol, max_step, rms_step, max_gradient, rms_gradient)

    traj_handle = open(trajectory_file, "w", encoding="utf-8") if trajectory_file else None
    try:
        if traj_handle is not None:
            _write_opt_frame(traj_handle, atoms, q, 0, energy, grad_x)

        for istep in range(1, maxstep + 1):
            if method == "STEEPEST_DESCENT":
                dq = -0.7 * grad_q
            else:
                dq = _solve_step(hess_q, grad_q)

            stp = float(np.linalg.norm(dq))
            if stp > max_step_internal and stp > 0.0:
                dq *= max_step_internal / stp

            old_q = q
            old_energy = energy
            old_grad_q = grad_q
            old_qs = qs

            q, energy, grad_x, accepted_dq, cart_step, scale, accepted = _line_search_internal(
                qcinput,
                atoms,
                q,
                energy,
                grad_x,
                dq,
                qs,
                ic,
                bpg,
                best_fit_iters=best_fit_iters,
                best_fit_rms_tol=best_fit_rms_tol,
            )
            if not accepted:
                message = "Line search failed to find a downhill internal-coordinate step"
                if print_report:
                    _print_optimization_status(False, message)
                return OptimizationResult(
                    atoms=atoms, q=old_q, energy=old_energy, converged=False, nsteps=istep - 1,
                    method=method, backend_optimizer="smite", trajectory_file=trajectory_file,
                    gradient=grad_x, message=message, coordinates="internal",
                )
            energy_change = energy - old_energy

            if use_redundant_internals:
                redundant_system = _current_redundant_system(
                    q,
                    ic,
                    use_redundant_internals=use_redundant_internals,
                )
                qs = redundant_system.q
                bpg = bpg_from_redundant(redundant_system)
            else:
                qs = compute_internals(q, ic)
                bpg = bpg_matrix(q, ic)
            grad_q = internal_gradient(bpg, grad_x)

            if print_report:
                _print_step(istep, energy, energy_change, cart_step, grad_x)

            if traj_handle is not None:
                _write_opt_frame(traj_handle, atoms, q, istep, energy, grad_x)

            if _converged(
                energy_change,
                cart_step,
                grad_x,
                max_gradient=max_gradient,
                rms_gradient=rms_gradient,
                max_step=max_step,
                rms_step=rms_step,
                energy_tol=energy_tol,
            ):
                if print_report:
                    _print_optimization_status(True, "Convergence reached.")
                return OptimizationResult(
                    atoms=atoms,
                    q=q,
                    energy=energy,
                    converged=True,
                    nsteps=istep,
                    method=method,
                    backend_optimizer="smite",
                    trajectory_file=trajectory_file,
                    gradient=grad_x,
                    message="Convergence reached.",
                    coordinates="internal",
                )

            if method == "BFGS":
                y = diff_internals(grad_q, old_grad_q, ic.ndihedrals, ic.nimpropers)
                hess_candidate = _bfgs_update_hessian(hess_q, accepted_dq, y)
                if np.all(np.isfinite(hess_candidate)):
                    hess_q = hess_candidate

    finally:
        if traj_handle is not None:
            traj_handle.close()

    if print_report:
        _print_optimization_status(False, "Maximum optimization steps reached without convergence.")
    return OptimizationResult(
        atoms=atoms,
        q=q,
        energy=energy,
        converged=False,
        nsteps=maxstep,
        method=method,
        backend_optimizer="smite",
        trajectory_file=trajectory_file,
        gradient=grad_x,
        message="Maximum optimization steps reached without convergence.",
        coordinates="internal",
    )

def _orca_native_optimize_geometry(qcinput, atoms, q, *, what="minimum"):
    qcinput = dict(qcinput)
    qcinput.setdefault("what", what)
    xyz = makeXYZ(atoms, q)
    converged, energy, opt_q_angstrom = Orca_GeomOpt(len(atoms), xyz, qcinput)
    opt_q = np.asarray(opt_q_angstrom, dtype=float) * ANGSTROM_TO_BOHR
    return OptimizationResult(
        atoms=list(atoms),
        q=opt_q,
        energy=float(energy),
        converged=bool(converged),
        nsteps=0,
        method="ORCA_OPT",
        backend_optimizer="backend",
        message="ORCA built-in optimizer finished.",
    )
