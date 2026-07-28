from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from optimizer.common import (
    OptimizationResult,
    _atoms_q_from_xyz,
    _energy,
    _exact_cartesian_hessian,
    _gradient,
    _xyz_from_q,
)
from optimizer.driver import (
    optimize_geometry,
)
from optimizer.reporter import make_reporter
from utils.atomic_masses import get_mass_vector
from utils.constants import BOHR_TO_ANGSTROM, HARTREE_TO_KCAL_MOL, HARTREE_TO_KJMOL


@dataclass
class IRCPoint:
    s: float
    q: np.ndarray
    energy: float
    gradient: np.ndarray
    grad_norm_mw: float
    step_size: float
    corrector_iterations: int = 0


@dataclass
class IRCBranch:
    direction: int
    points: list[IRCPoint]
    converged: bool
    message: str
    endpoint: OptimizationResult | None = None


@dataclass
class IRCResult:
    atoms: list[str]
    q_ts: np.ndarray
    ts_point: IRCPoint
    imaginary_mode_mw: np.ndarray
    forward: IRCBranch
    backward: IRCBranch
    trajectory: list[IRCPoint]
    trajectory_file: str | None = None
    profile_file: str | None = None
    plot_file: str | None = None
    endpoint_comparison: dict | None = None


def _mass_vector_amu(atoms):
    mass = np.asarray(get_mass_vector(atoms, gmol2au=1.0), dtype=float)
    if mass.size != len(atoms) or np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
        raise ValueError("IRC requires valid atomic masses for all atoms")
    return mass


def _sqrt_mass_by_coord(mass):
    return np.repeat(np.sqrt(np.asarray(mass, dtype=float)), 3)


def _cart_to_mw(q, sqrt_m):
    return np.asarray(q, dtype=float).reshape(-1) * sqrt_m


def _mw_to_cart(y, sqrt_m):
    return np.asarray(y, dtype=float).reshape(-1) / sqrt_m


def _grad_cart_to_mw(grad_x, sqrt_m):
    return np.asarray(grad_x, dtype=float).reshape(-1) / sqrt_m


def _hess_cart_to_mw(hess_x, sqrt_m):
    hess_x = np.asarray(hess_x, dtype=float)
    return hess_x / sqrt_m[:, None] / sqrt_m[None, :]


def _normalize(vec, *, label="vector"):
    vec = np.asarray(vec, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError(f"Cannot normalize zero or invalid {label}")
    return vec / norm


def _imaginary_mode_from_hessian(hess_x, sqrt_m):
    hess_mw = _hess_cart_to_mw(hess_x, sqrt_m)
    hess_mw = 0.5 * (hess_mw + hess_mw.T)
    evals, evecs = np.linalg.eigh(hess_mw)
    idx = int(np.argmin(evals))
    if evals[idx] >= 0.0:
        raise ValueError("IRC initialization requires a Hessian with at least one negative eigenvalue")
    return _normalize(evecs[:, idx], label="imaginary mode"), evals


def _prepare_imaginary_mode(atoms, q_ts, sqrt_m, imaginary_mode=None, hessian=None, mode_is_mass_weighted=True):
    if imaginary_mode is not None:
        mode = np.asarray(imaginary_mode, dtype=float).reshape(-1)
        if mode.shape != q_ts.shape:
            raise ValueError("imaginary_mode must have the same shape as q_ts")
        if not mode_is_mass_weighted:
            mode = _cart_to_mw(mode, sqrt_m)
        return _normalize(mode, label="imaginary mode"), None

    if hessian is None:
        raise ValueError("IRC requires either imaginary_mode or hessian")
    return _imaginary_mode_from_hessian(hessian, sqrt_m)


def _point_from_y(qcinput, atoms, y, sqrt_m, s, step_size, corrector_iterations=0):
    q = _mw_to_cart(y, sqrt_m)
    energy = _energy(qcinput, q, atoms)
    grad_x = _gradient(qcinput, q, atoms)
    grad_y = _grad_cart_to_mw(grad_x, sqrt_m)
    return IRCPoint(
        s=float(s),
        q=q,
        energy=energy,
        gradient=grad_x,
        grad_norm_mw=float(np.linalg.norm(grad_y)),
        step_size=float(step_size),
        corrector_iterations=int(corrector_iterations),
    )


def _relative_energy_kjmol(point, reference_energy):
    return (float(point.energy) - float(reference_energy)) * HARTREE_TO_KJMOL


def _relative_energy_kcalmol(point, reference_energy):
    return (float(point.energy) - float(reference_energy)) * HARTREE_TO_KCAL_MOL


def _write_branch_xyz(handle, atoms, point, branch_label, reference_energy):
    handle.write(str(len(atoms)) + "\n")
    handle.write(
        f"irc_part={branch_label} s={point.s:.8f} E_hartree={point.energy:.12f} "
        f"dE_from_TS_kJmol={_relative_energy_kjmol(point, reference_energy):.8f} "
        f"grad_norm_mw={point.grad_norm_mw:.8e} h={point.step_size:.8f} "
        f"corrector_iterations={point.corrector_iterations}\n"
    )
    q_angstrom = point.q * BOHR_TO_ANGSTROM
    for i, atom in enumerate(atoms):
        j = 3 * i
        handle.write(f"{atom:2s} {q_angstrom[j]:16.8f} {q_angstrom[j+1]:16.8f} {q_angstrom[j+2]:16.8f}\n")


def _write_trajectory_xyz(filename, atoms, trajectory, reference_energy):
    if not filename:
        return
    with open(filename, "w", encoding="utf-8") as handle:
        for point in trajectory:
            branch_label = "TS"
            if point.s < 0.0:
                branch_label = "backward"
            elif point.s > 0.0:
                branch_label = "forward"
            _write_branch_xyz(handle, atoms, point, branch_label, reference_energy)


def _write_profile_dat(filename, trajectory, reference_energy):
    if not filename:
        return
    with open(filename, "w", encoding="utf-8") as handle:
        handle.write("# IRC energy profile\n")
        handle.write("# Relative energies are measured from the TS reference energy.\n")
        handle.write("# Columns: index branch s E_hartree dE_from_TS_kcalmol dE_from_TS_kJmol grad_norm_mw step_size corrector_iterations\n")
        for index, point in enumerate(trajectory):
            branch = "TS"
            if point.s < 0.0:
                branch = "backward"
            elif point.s > 0.0:
                branch = "forward"
            handle.write(
                "%6d %10s %16.8f %22.12f %20.8f %20.8f %16.8e %12.6f %6d\n"
                % (
                    index,
                    branch,
                    point.s,
                    point.energy,
                    _relative_energy_kcalmol(point, reference_energy),
                    _relative_energy_kjmol(point, reference_energy),
                    point.grad_norm_mw,
                    point.step_size,
                    point.corrector_iterations,
                )
            )


def _write_profile_plot(filename, trajectory, reference_energy):
    if not filename:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("IRC: matplotlib is not available; skipping PNG energy profile plot.", flush=True)
        return None

    s_values = [point.s for point in trajectory]
    rel_values = [_relative_energy_kcalmol(point, reference_energy) for point in trajectory]
    grad_values = [point.grad_norm_mw for point in trajectory]

    fig, ax_energy = plt.subplots(figsize=(7.0, 4.5))
    ax_energy.plot(s_values, rel_values, marker="o", color="tab:blue", label="Energy")
    ax_energy.axhline(0.0, color="0.4", linewidth=0.8, linestyle="--")
    ax_energy.axvline(0.0, color="0.4", linewidth=0.8, linestyle=":")
    ax_energy.set_xlabel("IRC coordinate s [amu^0.5 bohr]")
    ax_energy.set_ylabel("E - E_TS [kcal/mol]")
    ax_energy.set_title("IRC energy profile")

    ax_grad = ax_energy.twinx()
    ax_grad.plot(s_values, grad_values, marker="s", color="tab:red", alpha=0.65, label="|grad|")
    ax_grad.set_ylabel("Mass-weighted gradient norm")

    lines1, labels1 = ax_energy.get_legend_handles_labels()
    lines2, labels2 = ax_grad.get_legend_handles_labels()
    ax_energy.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    return filename


def _lqa_predictor(grad_y, hess_y, step_size, substeps):
    displacement = np.zeros_like(grad_y)
    ds = step_size / max(1, int(substeps))
    for _ in range(max(1, int(substeps))):
        local_grad = grad_y + hess_y @ displacement
        tangent = -_normalize(local_grad, label="LQA gradient")
        displacement = displacement + ds * tangent
    return displacement


def _predictor_step(qcinput, atoms, y, sqrt_m, grad_y, step_size, *, predictor, hess_file, lqa_substeps):
    if predictor == "euler":
        return step_size * (-_normalize(grad_y, label="mass-weighted gradient"))
    if predictor != "lqa":
        raise ValueError("predictor must be 'euler' or 'lqa'")

    q = _mw_to_cart(y, sqrt_m)
    hess_x = _exact_cartesian_hessian(qcinput, atoms, q, hess_file)
    hess_y = _hess_cart_to_mw(hess_x, sqrt_m)
    return _lqa_predictor(grad_y, hess_y, step_size, lqa_substeps)


def _correct_irc_point(
    qcinput,
    atoms,
    y_previous,
    y_initial,
    sqrt_m,
    step_size,
    *,
    corrector_tol,
    corrector_maxiter,
    corrector_alpha,
):
    y = np.asarray(y_initial, dtype=float).copy()
    ndim = y.size

    for iteration in range(1, int(corrector_maxiter) + 1):
        radial = y - y_previous
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm <= 0.0:
            radial = np.zeros(ndim)
            radial[0] = 1.0
            radial_norm = 1.0
        u = radial / radial_norm

        q = _mw_to_cart(y, sqrt_m)
        grad_y = _grad_cart_to_mw(_gradient(qcinput, q, atoms), sqrt_m)
        g_perp = grad_y - u * float(np.dot(u, grad_y))
        g_perp_norm = float(np.linalg.norm(g_perp))
        if g_perp_norm < corrector_tol:
            return y_previous + step_size * u, iteration, True

        alpha = float(corrector_alpha)
        max_move = 0.5 * step_size
        if alpha * g_perp_norm > max_move:
            alpha = max_move / g_perp_norm
        y = y - alpha * g_perp

        radial = y - y_previous
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm <= 0.0:
            return y_initial, iteration, False
        y = y_previous + step_size * radial / radial_norm

    return y, int(corrector_maxiter), False


def _maybe_minimize_endpoint(qcinput, atoms, q, optimize_endpoints, endpoint_optimizer_kwargs):
    if not optimize_endpoints:
        return None
    kwargs = dict(endpoint_optimizer_kwargs or {})
    kwargs.setdefault("backend_optimizer", "smite")
    kwargs.setdefault("coordinates", "internal")
    kwargs.setdefault("method", "BFGS")
    kwargs.setdefault("trajectory_file", None)
    kwargs.setdefault("print_report", False)
    return optimize_geometry(qcinput, _xyz_from_q(atoms, q), **kwargs)


def _kabsch_aligned_rmsd_angstrom(q, reference_q):
    p = np.asarray(q, dtype=float).reshape(-1, 3) * BOHR_TO_ANGSTROM
    r = np.asarray(reference_q, dtype=float).reshape(-1, 3) * BOHR_TO_ANGSTROM
    p_centroid = p.mean(axis=0)
    r_centroid = r.mean(axis=0)
    p0 = p - p_centroid
    r0 = r - r_centroid
    cov = p0.T @ r0
    u, _s, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    p_aligned = p0 @ rot
    diff = p_aligned - r0
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _plain_rmsd_angstrom(q, reference_q):
    diff = (np.asarray(q, dtype=float).reshape(-1) - np.asarray(reference_q, dtype=float).reshape(-1))
    diff = (diff * BOHR_TO_ANGSTROM).reshape(-1, 3)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _compare_irc_endpoints(
    atoms,
    forward,
    backward,
    *,
    endpoint_reference_initial=None,
    endpoint_reference_final=None,
    endpoint_compare_align=True,
    endpoint_compare_tol=0.25,
    print_report=True,
):
    references = []
    for label, xyz in (("initial", endpoint_reference_initial), ("final", endpoint_reference_final)):
        if xyz is None:
            continue
        ref_atoms, ref_q = _atoms_q_from_xyz(xyz)
        if ref_atoms != atoms:
            raise ValueError(f"IRC endpoint reference {label!r} has different atom labels/order")
        references.append((label, ref_q))
    if not references:
        return None

    branches = []
    for label, branch in (("forward", forward), ("backward", backward)):
        q_branch = branch.endpoint.q if branch.endpoint is not None else branch.points[-1].q
        branches.append((label, q_branch, branch.endpoint is not None))

    rmsd_fn = _kabsch_aligned_rmsd_angstrom if endpoint_compare_align else _plain_rmsd_angstrom
    comparison = {
        "tolerance_angstrom": float(endpoint_compare_tol),
        "aligned": bool(endpoint_compare_align),
        "rmsd": {},
        "best_match": {},
    }
    for branch_label, q_branch, endpoint_optimized in branches:
        branch_data = {}
        for ref_label, ref_q in references:
            branch_data[ref_label] = rmsd_fn(q_branch, ref_q)
        comparison["rmsd"][branch_label] = branch_data
        comparison["best_match"][branch_label] = min(branch_data, key=branch_data.get)
        comparison[f"{branch_label}_endpoint_optimized"] = endpoint_optimized

    if print_report:
        print("IRC endpoint comparison:", flush=True)
        print(f"  RMSD alignment: {endpoint_compare_align}", flush=True)
        print(f"  match tolerance [Angstrom]: {float(endpoint_compare_tol):.4f}", flush=True)
        for branch_label in ("forward", "backward"):
            for ref_label, rmsd in comparison["rmsd"][branch_label].items():
                status = "match" if rmsd <= endpoint_compare_tol else "different"
                print(f"  {branch_label:8s} vs {ref_label:7s}: RMSD={rmsd:.6f} Angstrom ({status})", flush=True)
            print(
                f"  {branch_label:8s} best match: {comparison['best_match'][branch_label]}",
                flush=True,
            )
    return comparison


def _follow_branch(
    qcinput,
    atoms,
    y_start,
    sqrt_m,
    *,
    direction,
    step_size,
    step_size_min,
    step_size_max,
    max_steps,
    predictor,
    hess_file,
    lqa_substeps,
    corrector_tol,
    corrector_maxiter,
    corrector_alpha,
    endpoint_gradient_tol,
    adaptive,
    optimize_endpoints,
    endpoint_optimizer_kwargs,
    trajectory_file,
    print_report,
    reference_energy,
    initial_s=0.0,
):
    points = []
    branch_label = "forward" if direction > 0 else "backward"
    handle = open(trajectory_file, "w", encoding="utf-8") if trajectory_file else None

    try:
        current_y = np.asarray(y_start, dtype=float).copy()
        current_step = float(step_size)
        s = float(initial_s)
        point = _point_from_y(qcinput, atoms, current_y, sqrt_m, s, current_step)
        points.append(point)
        if handle is not None:
            _write_branch_xyz(handle, atoms, point, branch_label, reference_energy)

        for istep in range(1, int(max_steps) + 1):
            if point.grad_norm_mw < endpoint_gradient_tol:
                endpoint = _maybe_minimize_endpoint(
                    qcinput, atoms, point.q, optimize_endpoints, endpoint_optimizer_kwargs
                )
                return IRCBranch(direction, points, True, "IRC branch reached the endpoint gradient threshold.", endpoint)

            grad_y = _grad_cart_to_mw(point.gradient, sqrt_m)

            accepted = False
            last_iterations = 0
            while current_step >= step_size_min:
                predicted_delta = _predictor_step(
                    qcinput,
                    atoms,
                    current_y,
                    sqrt_m,
                    grad_y,
                    current_step,
                    predictor=predictor,
                    hess_file=f"{hess_file}_{branch_label}_step{istep:04d}.hess",
                    lqa_substeps=lqa_substeps,
                )
                y_pred = current_y + predicted_delta
                y_corr, n_corr, corr_ok = _correct_irc_point(
                    qcinput,
                    atoms,
                    current_y,
                    y_pred,
                    sqrt_m,
                    current_step,
                    corrector_tol=corrector_tol,
                    corrector_maxiter=corrector_maxiter,
                    corrector_alpha=corrector_alpha,
                )
                last_iterations = n_corr
                if corr_ok:
                    accepted = True
                    break
                current_step *= 0.5

            if not accepted:
                return IRCBranch(
                    direction,
                    points,
                    False,
                    "IRC branch stopped because the corrector could not converge above the minimum step size.",
                    None,
                )

            s += current_step
            current_y = y_corr
            point = _point_from_y(qcinput, atoms, current_y, sqrt_m, s, current_step, last_iterations)
            points.append(point)
            if handle is not None:
                _write_branch_xyz(handle, atoms, point, branch_label, reference_energy)
            if print_report:
                rel_energy = _relative_energy_kjmol(point, reference_energy)
                print(
                    f"{branch_label:8s} {istep:4d} s={s:10.5f} E={point.energy:18.10f} "
                    f"dE_TS={rel_energy:12.5f} kJ/mol "
                    f"|g_mw|={point.grad_norm_mw:10.4e} h={current_step:8.4f} corr={last_iterations:3d}"
                )

            if adaptive:
                if last_iterations <= 3:
                    current_step = min(step_size_max, 1.2 * current_step)
                elif last_iterations > 10:
                    current_step = max(step_size_min, 0.5 * current_step)

        endpoint = _maybe_minimize_endpoint(qcinput, atoms, points[-1].q, optimize_endpoints, endpoint_optimizer_kwargs)
        return IRCBranch(direction, points, False, "Maximum IRC steps reached.", endpoint)
    finally:
        if handle is not None:
            handle.close()


def follow_irc(
    qcinput,
    xyz,
    *,
    imaginary_mode=None,
    mode_is_mass_weighted=True,
    hessian=None,
    hess_file="hessian_irc_ts.hess",
    initial_displacement=0.02,
    step_size=0.05,
    step_forward=50,
    step_backward=50,
    max_steps=None,
    step_size_min=0.01,
    step_size_max=0.20,
    predictor="euler",
    lqa_substeps=5,
    corrector_tol=1.0e-4,
    corrector_maxiter=20,
    corrector_alpha=1.0,
    endpoint_gradient_tol=1.0e-4,
    adaptive=True,
    optimize_endpoints=False,
    endpoint_optimizer_kwargs=None,
    endpoint_reference_initial=None,
    endpoint_reference_final=None,
    endpoint_compare_align=True,
    endpoint_compare_tol=0.25,
    trajectory_prefix="irc",
    trajectory_file=None,
    profile_file="irc_energy_profile.dat",
    plot_file="irc_energy_profile.png",
    print_report=True,
    reporter=None,
    verbosity=1,
    log_file=None,
    log_append=False,
    _reporter_active=False,
):
    """Follow both IRC branches from a transition-state geometry.

    Input geometry is an XYZ string or XYZ file path with Angstrom coordinates.
    IRC propagation is done internally in mass-weighted Cartesian coordinates
    using atomic masses in amu, so `step_size` and `initial_displacement` have
    units amu**0.5 * bohr.

    The IRC energy profile table is always written when profile_file is not
    None.  Relative energies are reported in kcal/mol from the TS reference
    energy.  If matplotlib is available, plot_file is also written.
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
            return follow_irc(qcinput, xyz, **options)

    atoms, q_ts = _atoms_q_from_xyz(xyz)

    mass = _mass_vector_amu(atoms)
    sqrt_m = _sqrt_mass_by_coord(mass)
    y_ts = _cart_to_mw(q_ts, sqrt_m)

    if hessian is None and imaginary_mode is None:
        hessian = _exact_cartesian_hessian(qcinput, atoms, q_ts, hess_file)
    imag_mode_mw, evals = _prepare_imaginary_mode(
        atoms,
        q_ts,
        sqrt_m,
        imaginary_mode=imaginary_mode,
        hessian=hessian,
        mode_is_mass_weighted=mode_is_mass_weighted,
    )

    if max_steps is not None:
        step_forward = int(max_steps)
        step_backward = int(max_steps)
    step_forward = int(step_forward)
    step_backward = int(step_backward)
    if step_forward < 0 or step_backward < 0:
        raise ValueError("step_forward and step_backward must be non-negative")

    ts_point = _point_from_y(qcinput, atoms, y_ts, sqrt_m, 0.0, step_size)

    if print_report:
        print("Intrinsic reaction coordinate following")
        print(f"TS energy [Eh]={ts_point.energy:.12f}")
        print(f"step_size={step_size:.5f} initial_displacement={initial_displacement:.5f}")
        print(f"step_forward={step_forward} step_backward={step_backward}")
        print(f"adaptive step size: {bool(adaptive)}")
        print(f"endpoint optimization after IRC: {bool(optimize_endpoints)}")
        print(f"profile file: {profile_file}")
        print(f"plot file: {plot_file}")
        mode_source = "user-provided imaginary_mode" if imaginary_mode is not None else "lowest Hessian eigenvector"
        print(f"IRC mode source: {mode_source}")
        print(f"IRC imaginary-mode norm: {np.linalg.norm(imag_mode_mw):.8f}")
        if evals is not None:
            negative_count = int(np.sum(np.asarray(evals) < 0.0))
            print(f"IRC Hessian negative eigenvalues: {negative_count}")
            print(f"IRC lowest Hessian eigenvalue [mass-weighted a.u.]: {float(np.min(evals)):.8e}")
        print("IRC branch convention:")
        print("  forward  starts along + imaginary mode")
        print("  backward starts along - imaginary mode")

    branches = {}
    for direction, label, nsteps in ((1, "forward", step_forward), (-1, "backward", step_backward)):
        y0 = y_ts + direction * float(initial_displacement) * imag_mode_mw
        branch_trajectory_file = f"{trajectory_prefix}_{label}.xyz" if trajectory_prefix else None
        branches[label] = _follow_branch(
            qcinput,
            atoms,
            y0,
            sqrt_m,
            direction=direction,
            step_size=step_size,
            step_size_min=step_size_min,
            step_size_max=step_size_max,
            max_steps=nsteps,
            predictor=predictor,
            hess_file=hess_file,
            lqa_substeps=lqa_substeps,
            corrector_tol=corrector_tol,
            corrector_maxiter=corrector_maxiter,
            corrector_alpha=corrector_alpha,
            endpoint_gradient_tol=endpoint_gradient_tol,
            adaptive=adaptive,
            optimize_endpoints=optimize_endpoints,
            endpoint_optimizer_kwargs=endpoint_optimizer_kwargs,
            trajectory_file=branch_trajectory_file,
            print_report=print_report,
            reference_energy=ts_point.energy,
            initial_s=float(initial_displacement),
        )

    backward_points = [
        replace(point, s=-abs(point.s))
        for point in reversed(branches["backward"].points)
    ]
    forward_points = [
        replace(point, s=abs(point.s))
        for point in branches["forward"].points
    ]
    trajectory = backward_points + [ts_point] + forward_points

    if trajectory_file is None and trajectory_prefix:
        trajectory_file = f"{trajectory_prefix}_full.xyz"
    _write_trajectory_xyz(trajectory_file, atoms, trajectory, ts_point.energy)
    _write_profile_dat(profile_file, trajectory, ts_point.energy)
    plot_written = _write_profile_plot(plot_file, trajectory, ts_point.energy)

    endpoint_comparison = _compare_irc_endpoints(
        atoms,
        branches["forward"],
        branches["backward"],
        endpoint_reference_initial=endpoint_reference_initial,
        endpoint_reference_final=endpoint_reference_final,
        endpoint_compare_align=endpoint_compare_align,
        endpoint_compare_tol=endpoint_compare_tol,
        print_report=print_report,
    )

    if print_report:
        print("IRC output files:", flush=True)
        print(f"  combined trajectory: {trajectory_file}", flush=True)
        print(f"  energy profile: {profile_file}", flush=True)
        if plot_written is not None:
            print(f"  energy profile plot: {plot_written}", flush=True)

    return IRCResult(
        atoms=atoms,
        q_ts=q_ts,
        ts_point=ts_point,
        imaginary_mode_mw=imag_mode_mw,
        forward=branches["forward"],
        backward=branches["backward"],
        trajectory=trajectory,
        trajectory_file=trajectory_file,
        profile_file=profile_file,
        plot_file=plot_written,
        endpoint_comparison=endpoint_comparison,
    )
