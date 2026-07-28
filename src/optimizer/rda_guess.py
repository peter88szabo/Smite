from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimizer.common import (
    _atoms_q_from_xyz,
    _bfgs_update_hessian,
    _bpg_for_internal_coordinates,
    _energy,
    _gradient,
    _initial_internal_coordinate_state,
    _solve_step,
    step_limit,
)
from optimizer.coordinate_refs import coordinate_label, find_internal_coordinate_indices
from optimizer.internal_coords import (
    InternalCoords,
    best_fit_dq_to_cart,
    compute_internals,
    default_connectivity_model,
    diff_internals,
    initial_internal_hessian_from_model,
    internal_coords_from_bonds,
    internal_gradient,
)
from optimizer.minimum import _line_search_cartesian, _line_search_internal
from optimizer.redundant_internals import initial_redundant_hessian_from_model
from optimizer.reporter import make_reporter
from optimizer.scan import _relax_scan_point_multi
from qchem_interfaces.qchem_validation import validate_qchem_input
from utils.constants import BOHR_TO_ANGSTROM


EV_TO_HARTREE = 0.03674932217565499


@dataclass
class RDAPoint:
    label: str
    q_start: np.ndarray
    q_copt: np.ndarray
    energy: float
    nsteps: int
    direction: str
    delta_d_is: float
    delta_d_fs: float
    beta: float | None = None
    gamma: float | None = None
    message: str = ""


@dataclass
class RDAGuessResult:
    atoms: list[str]
    q: np.ndarray
    reaction_direction: np.ndarray | None
    points: list[RDAPoint]
    direction: str
    bracketed: bool
    message: str
    rda_trajectory_file: str | None = None
    rda_chain_file: str | None = None
    poormans_NEB_file: str | None = None
    poormans_NEB_profile_file: str | None = None
    poormans_NEB_plot_file: str | None = None

    def xyz(self) -> str:
        lines = [str(len(self.atoms)), "RDA quasi-TS guess"]
        q_angstrom = np.asarray(self.q, dtype=float).reshape(-1) * BOHR_TO_ANGSTROM
        for i, atom in enumerate(self.atoms):
            j = 3 * i
            lines.append(
                f"{atom:2s} {q_angstrom[j]:16.10f} {q_angstrom[j + 1]:16.10f} {q_angstrom[j + 2]:16.10f}"
            )
        return "\n".join(lines)


def _format_float(value):
    if value is None:
        return "none"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.10g}"


def _xyz_lines(atoms, q, comment):
    lines = [str(len(atoms)), str(comment)]
    q_angstrom = np.asarray(q, dtype=float).reshape(-1) * BOHR_TO_ANGSTROM
    for i, atom in enumerate(atoms):
        j = 3 * i
        lines.append(
            f"{atom:2s} {q_angstrom[j]:16.10f} {q_angstrom[j + 1]:16.10f} {q_angstrom[j + 2]:16.10f}"
        )
    return lines


def _write_xyz_frame(handle, atoms, q, **metadata):
    comment = " ".join(f"{key}={_format_float(value)}" for key, value in metadata.items())
    handle.write("\n".join(_xyz_lines(atoms, q, comment)))
    handle.write("\n")


def _write_rda_trajectory(filename, atoms, q_is, q_fs, points):
    if not filename:
        return
    with open(filename, "w", encoding="utf-8") as handle:
        _write_xyz_frame(handle, atoms, q_is, label="endpoint_IS")
        for point in points:
            _write_xyz_frame(
                handle,
                atoms,
                point.q_start,
                label=f"{point.label}_start",
                direction=point.direction,
                beta=point.beta,
                gamma=point.gamma,
                delta_d_is=point.delta_d_is,
                delta_d_fs=point.delta_d_fs,
                energy=np.nan,
                nsteps=0,
            )
            if not np.allclose(point.q_start, point.q_copt):
                _write_xyz_frame(
                    handle,
                    atoms,
                    point.q_copt,
                    label=f"{point.label}_copt",
                    direction=point.direction,
                    beta=point.beta,
                    gamma=point.gamma,
                    delta_d_is=point.delta_d_is,
                    delta_d_fs=point.delta_d_fs,
                    energy=point.energy,
                    nsteps=point.nsteps,
                )
        _write_xyz_frame(handle, atoms, q_fs, label="endpoint_FS")


def _write_rda_chain(filename, atoms, q_is, q_fs, points, q_guess):
    if not filename:
        return
    chain = [("endpoint_IS", q_is)]
    for point in points:
        if point.nsteps > 0:
            chain.append((f"{point.label}_copt", point.q_copt))
        elif point.label == "gamma_selected":
            chain.append((point.label, point.q_copt))
    if not chain or not np.allclose(chain[-1][1], q_guess):
        chain.append(("rda_qts", q_guess))
    chain.append(("endpoint_FS", q_fs))
    with open(filename, "w", encoding="utf-8") as handle:
        for index, (label, q) in enumerate(chain):
            _write_xyz_frame(handle, atoms, q, label=label, image=index)


def _write_poormans_neb_chain(filename, atoms, images):
    if not filename:
        return
    with open(filename, "w", encoding="utf-8") as handle:
        for image in images:
            _write_xyz_frame(
                handle,
                atoms,
                image["q"],
                label=image["label"],
                image=image["index"],
                alpha=image["alpha"],
                energy=image["energy"],
                converged=image["converged"],
                opt_steps=image["opt_steps"],
            )


def _write_poormans_neb_profile(filename, images, coord_labels):
    if not filename:
        return
    if not images:
        return
    reference_energy = float(images[0]["energy"])
    header = [
        "image",
        "alpha",
        "energy_Eh",
        "relative_energy_kcal_mol",
        "converged",
        "opt_steps",
    ]
    header.extend(coord_labels)
    hartree_to_kcal_mol = 627.5094740631
    with open(filename, "w", encoding="utf-8") as handle:
        handle.write("# poormans_NEB energetics profile from constrained RDA\n")
        handle.write("# " + " ".join(header) + "\n")
        for image in images:
            rel = (float(image["energy"]) - reference_energy) * hartree_to_kcal_mol
            values = [
                str(int(image["index"])),
                f"{float(image['alpha']):.8f}",
                f"{float(image['energy']):.12f}",
                f"{rel:.8f}",
                str(bool(image["converged"])),
                str(int(image["opt_steps"])),
            ]
            values.extend(f"{float(value):.10f}" for value in image.get("reactive_values", []))
            handle.write(" ".join(values) + "\n")


def _print_poormans_neb_profile(images, coord_labels):
    if not images:
        return
    reference_energy = float(images[0]["energy"])
    hartree_to_kcal_mol = 627.5094740631
    columns = ["image", "alpha", "E[Eh]", "dE[kcal/mol]", "conv", "steps"] + list(coord_labels)
    print("RDA poormans_NEB energetics profile:", flush=True)
    print(" ".join(f"{column:>16s}" for column in columns), flush=True)
    for image in images:
        rel = (float(image["energy"]) - reference_energy) * hartree_to_kcal_mol
        values = [
            f"{int(image['index']):16d}",
            f"{float(image['alpha']):16.6f}",
            f"{float(image['energy']):16.8f}",
            f"{rel:16.6f}",
            f"{str(bool(image['converged'])):>16s}",
            f"{int(image['opt_steps']):16d}",
        ]
        values.extend(f"{float(value):16.6f}" for value in image.get("reactive_values", []))
        print(" ".join(values), flush=True)


def _write_poormans_neb_plot(filename, images, coord_labels):
    if not filename:
        return
    if not images:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("RDA poormans_NEB: matplotlib is not available; skipping PNG profile plot.", flush=True)
        return

    alphas = np.asarray([float(image["alpha"]) for image in images], dtype=float)
    energies = np.asarray([float(image["energy"]) for image in images], dtype=float)
    rel_energies = (energies - energies[0]) * 627.5094740631
    selected_index = int(np.argmax(energies[1:-1]) + 1) if len(images) > 2 else int(np.argmax(energies))

    fig, ax_energy = plt.subplots(figsize=(8.0, 5.0))
    ax_energy.plot(alphas, rel_energies, marker="o", color="tab:red", label="relative energy")
    ax_energy.scatter(
        [alphas[selected_index]],
        [rel_energies[selected_index]],
        color="black",
        zorder=5,
        label="selected qTS image",
    )
    ax_energy.set_xlabel("path coordinate alpha")
    ax_energy.set_ylabel("relative energy [kcal/mol]", color="tab:red")
    ax_energy.tick_params(axis="y", labelcolor="tab:red")
    ax_energy.grid(True, alpha=0.3)

    ax_coord = ax_energy.twinx()
    reactive_matrix = np.asarray(
        [image.get("reactive_values", []) for image in images],
        dtype=float,
    )
    if reactive_matrix.size:
        if reactive_matrix.ndim == 1:
            reactive_matrix = reactive_matrix.reshape(-1, 1)
        for idx in range(reactive_matrix.shape[1]):
            series = reactive_matrix[:, idx]
            span = float(np.max(series) - np.min(series))
            if span > 0.0:
                plotted = (series - np.min(series)) / span
                ylabel = "normalized reactive coordinate"
            else:
                plotted = np.zeros_like(series)
                ylabel = "normalized reactive coordinate"
            label = coord_labels[idx] if idx < len(coord_labels) else f"coord_{idx}"
            ax_coord.plot(alphas, plotted, marker="s", linestyle="--", label=label)
        ax_coord.set_ylabel(ylabel)
        ax_coord.set_ylim(-0.05, 1.05)

    handles1, labels1 = ax_energy.get_legend_handles_labels()
    handles2, labels2 = ax_coord.get_legend_handles_labels()
    ax_energy.legend(handles1 + handles2, labels1 + labels2, loc="best", fontsize=8)
    ax_energy.set_title("poormans_NEB constrained RDA profile")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def _print_rda_table(points):
    print("RDA summary:", flush=True)
    print(
        "%16s %8s %8s %16s %12s %12s %8s"
        % ("label", "beta", "gamma", "E[Eh]", "dIS", "dFS", "dir"),
        flush=True,
    )
    for point in points:
        beta = "-" if point.beta is None else f"{point.beta:.3f}"
        gamma = "-" if point.gamma is None else f"{point.gamma:.3f}"
        print(
            "%16s %8s %8s %16.8f %12.4e %12.4e %8s"
            % (
                point.label,
                beta,
                gamma,
                point.energy,
                point.delta_d_is,
                point.delta_d_fs,
                point.direction,
            ),
            flush=True,
        )


def _as_atom_indices(indices, natoms, label):
    if indices is None:
        return np.arange(natoms, dtype=int)
    values = np.asarray([int(idx) for idx in indices], dtype=int)
    if values.size == 0:
        raise ValueError(f"{label} must contain at least one atom index")
    if np.any(values < 0) or np.any(values >= natoms):
        raise ValueError(f"{label} atom indices must be in the range 0..{natoms - 1}")
    if len(set(values.tolist())) != values.size:
        raise ValueError(f"{label} must not contain duplicate atom indices")
    return values


def _as_index_tuples(items, natoms, size, label):
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


def _reactive_internal_coordinates(natoms, reactive_bonds, reactive_angles, reactive_dihedrals):
    bonds = []
    seen_bonds = set()
    for i, j in reactive_bonds:
        bond = (i, j) if i < j else (j, i)
        if bond not in seen_bonds:
            seen_bonds.add(bond)
            bonds.append(bond)
    angles = list(dict.fromkeys(tuple(angle) for angle in reactive_angles))
    dihedrals = list(dict.fromkeys(tuple(dihedral) for dihedral in reactive_dihedrals))
    if not bonds and not angles and not dihedrals:
        return None
    return InternalCoords(
        nat=natoms,
        bonds=bonds,
        angles=angles,
        linear_bends=[],
        dihedrals=dihedrals,
        impropers=[],
    )


def _kabsch_align(mobile, reference, atoms):
    mobile_xyz = np.asarray(mobile, dtype=float).reshape(-1, 3)
    reference_xyz = np.asarray(reference, dtype=float).reshape(-1, 3)
    idx = np.asarray(atoms, dtype=int)
    mobile_center = mobile_xyz[idx].mean(axis=0)
    reference_center = reference_xyz[idx].mean(axis=0)
    mobile_centered = mobile_xyz[idx] - mobile_center
    reference_centered = reference_xyz[idx] - reference_center
    covariance = mobile_centered.T @ reference_centered
    u, _s, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    aligned = (mobile_xyz - mobile_center) @ rotation + reference_center
    return aligned.reshape(-1)


def _interpolate(q_a, q_b, alpha):
    alpha = float(alpha)
    return (1.0 - alpha) * np.asarray(q_a, dtype=float) + alpha * np.asarray(q_b, dtype=float)


def _normalize(vec):
    vec = np.asarray(vec, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0 or not np.isfinite(norm):
        return None
    return vec / norm


def _cartesian_rmsd(q_a, q_b, active_atoms):
    xyz_a = np.asarray(q_a, dtype=float).reshape(-1, 3)
    xyz_b = np.asarray(q_b, dtype=float).reshape(-1, 3)
    diff = xyz_a[np.asarray(active_atoms, dtype=int)] - xyz_b[np.asarray(active_atoms, dtype=int)]
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _cartesian_all_atom_rmsd(q_a, q_b):
    xyz_a = np.asarray(q_a, dtype=float).reshape(-1, 3)
    xyz_b = np.asarray(q_b, dtype=float).reshape(-1, 3)
    diff = xyz_a - xyz_b
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _union_internal_coordinates(q_is, q_fs, atoms, *, linear_bend_threshold_degrees=None):
    model = default_connectivity_model(atoms)
    ic_is = _initial_internal_coordinate_state(
        q_is,
        atoms,
        model,
        use_redundant_internals=False,
        linear_bends=linear_bend_threshold_degrees is not None,
        linear_bend_threshold_degrees=linear_bend_threshold_degrees,
    )[0]
    ic_fs = _initial_internal_coordinate_state(
        q_fs,
        atoms,
        model,
        use_redundant_internals=False,
        linear_bends=linear_bend_threshold_degrees is not None,
        linear_bend_threshold_degrees=linear_bend_threshold_degrees,
    )[0]
    bonds = sorted(set(ic_is.bonds).union(set(ic_fs.bonds)))
    if not bonds:
        return None
    return internal_coords_from_bonds(
        len(atoms),
        bonds,
        x=q_is,
        linear_bend_threshold_degrees=linear_bend_threshold_degrees,
    )


def _make_distance_function(
    q_is,
    q_fs,
    atoms,
    active_atoms,
    mode,
    linear_bend_threshold_degrees,
    *,
    reactive_bonds=None,
    reactive_angles=None,
    reactive_dihedrals=None,
    reactive_coordinate_weights=None,
):
    mode = str(mode).lower()
    if mode not in {"cartesian", "internal", "hybrid", "reactive_internal"}:
        raise ValueError("distance must be 'cartesian', 'internal', 'hybrid', or 'reactive_internal'")
    active_atoms = np.asarray(active_atoms, dtype=int)
    if mode == "reactive_internal":
        ic = _reactive_internal_coordinates(
            len(atoms),
            reactive_bonds or [],
            reactive_angles or [],
            reactive_dihedrals or [],
        )
        if ic is None:
            raise ValueError(
                "distance='reactive_internal' requires at least one reactive_bonds, "
                "reactive_angles, or reactive_dihedrals entry"
            )
        weights = None
        if reactive_coordinate_weights is not None:
            weights = np.asarray(reactive_coordinate_weights, dtype=float).reshape(-1)
            if weights.size != ic.nint:
                raise ValueError(
                    "reactive_coordinate_weights length must match the number of reactive internal coordinates "
                    f"({ic.nint})"
                )
            if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
                raise ValueError("reactive_coordinate_weights must be finite non-negative values")

        def reactive_internal_distance(qa, qb):
            qia = compute_internals(qa, ic)
            qib = compute_internals(qb, ic)
            dq = diff_internals(qia, qib, ic.ndihedrals, ic.nimpropers)
            if weights is not None:
                return float(np.sqrt(np.sum(weights * dq * dq) / max(1, dq.size)))
            return float(np.linalg.norm(dq) / np.sqrt(max(1, dq.size)))

        return reactive_internal_distance, (
            f"reactive_internal({ic.nbonds} bonds, {ic.nangles} angles, {ic.ndihedrals} dihedrals)"
        )

    ic = None
    if mode in {"internal", "hybrid"}:
        try:
            ic = _union_internal_coordinates(
                q_is,
                q_fs,
                atoms,
                linear_bend_threshold_degrees=linear_bend_threshold_degrees,
            )
        except Exception:
            ic = None
        if mode == "internal" and ic is None:
            raise ValueError("Could not build a stable internal-coordinate set for RDA distance analysis")

    if ic is None:
        return lambda qa, qb: _cartesian_rmsd(qa, qb, active_atoms), "cartesian"

    def internal_distance(qa, qb):
        qia = compute_internals(qa, ic)
        qib = compute_internals(qb, ic)
        dq = diff_internals(qia, qib, ic.ndihedrals, ic.nimpropers)
        return float(np.linalg.norm(dq) / np.sqrt(max(1, dq.size)))

    return internal_distance, "internal"


def _classify_direction(q_start, q_copt, q_is, q_fs, distance_fn, distance_tol):
    delta_is = distance_fn(q_copt, q_is) - distance_fn(q_start, q_is)
    delta_fs = distance_fn(q_copt, q_fs) - distance_fn(q_start, q_fs)
    if abs(delta_is) < distance_tol and abs(delta_fs) < distance_tol:
        return "ND", delta_is, delta_fs
    if delta_is * delta_fs > 0.0:
        return "ND", delta_is, delta_fs
    if delta_is < 0.0 and delta_fs > 0.0:
        return "IS", delta_is, delta_fs
    if delta_is > 0.0 and delta_fs < 0.0:
        return "FS", delta_is, delta_fs
    return "ND", delta_is, delta_fs


def _conditional_cartesian_optimize(
    qcinput,
    atoms,
    q,
    *,
    energy_tol,
    maxstep,
    step_max_component,
    print_report=False,
):
    q = np.asarray(q, dtype=float).reshape(-1).copy()
    energy = _energy(qcinput, q, atoms)
    grad = _gradient(qcinput, q, atoms)
    h_inv = np.eye(q.size)
    previous_energy = energy
    for istep in range(1, int(maxstep) + 1):
        if print_report:
            print(f"    RDA Cartesian conditional step {istep}/{int(maxstep)}", flush=True)
        step = step_limit(-h_inv @ grad, max_component=step_max_component)
        trial_q, trial_energy, trial_grad, accepted_step, _scale = _line_search_cartesian(
            qcinput,
            atoms,
            q,
            energy,
            grad,
            step,
        )
        d_e = trial_energy - previous_energy
        y = trial_grad - grad
        ys = float(np.dot(y, accepted_step))
        if ys > 1.0e-14:
            rho = 1.0 / ys
            ident = np.eye(q.size)
            sy = np.outer(accepted_step, y)
            ys_outer = np.outer(y, accepted_step)
            h_inv = (
                (ident - rho * sy) @ h_inv @ (ident - rho * ys_outer)
                + rho * np.outer(accepted_step, accepted_step)
            )
        q = trial_q
        energy = trial_energy
        grad = trial_grad
        if abs(d_e) < energy_tol:
            return q, energy, istep, "RDA conditional Cartesian optimization reached the energy-change threshold."
        previous_energy = energy
    return q, energy, int(maxstep), "RDA conditional Cartesian optimization reached maxstep."


def _conditional_internal_optimize(
    qcinput,
    atoms,
    q,
    *,
    energy_tol,
    maxstep,
    max_step_internal,
    best_fit_iters,
    best_fit_rms_tol,
    use_redundant_internals,
    connectivity_kcn,
    connectivity_facmin,
    connect_fragments,
    linear_bends,
    linear_bend_threshold_degrees,
    internal_hessian_model,
    print_report=False,
):
    q = np.asarray(q, dtype=float).reshape(-1).copy()
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
        raise ValueError("RDA internal conditional optimization found no internal coordinates")
    energy = _energy(qcinput, q, atoms)
    grad_x = _gradient(qcinput, q, atoms)
    grad_q = internal_gradient(bpg, grad_x)
    if use_redundant_internals and redundant_system is not None:
        hess_q = initial_redundant_hessian_from_model(redundant_system, model=internal_hessian_model)
    else:
        hess_q = initial_internal_hessian_from_model(ic, q_values=qs, model=internal_hessian_model)
    previous_energy = energy

    for istep in range(1, int(maxstep) + 1):
        if print_report:
            print(f"    RDA internal conditional step {istep}/{int(maxstep)}", flush=True)
        dq = _solve_step(hess_q, grad_q)
        step_norm = float(np.linalg.norm(dq))
        if step_norm > max_step_internal and step_norm > 0.0:
            dq *= float(max_step_internal) / step_norm
        trial_q, trial_energy, trial_grad_x, accepted_dq, _cart_step, _scale = _line_search_internal(
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
        trial_bpg = _bpg_for_internal_coordinates(
            trial_q,
            ic,
            use_redundant_internals=use_redundant_internals,
        )
        trial_grad_q = internal_gradient(trial_bpg, trial_grad_x)
        hess_q = _bfgs_update_hessian(hess_q, accepted_dq, trial_grad_q - grad_q)
        d_e = trial_energy - previous_energy
        q = trial_q
        qs = compute_internals(q, ic)
        bpg = trial_bpg
        energy = trial_energy
        grad_x = trial_grad_x
        grad_q = trial_grad_q
        if abs(d_e) < energy_tol:
            return q, energy, istep, "RDA conditional internal optimization reached the energy-change threshold."
        previous_energy = energy
    return q, energy, int(maxstep), "RDA conditional internal optimization reached maxstep."


def _conditional_optimize(qcinput, atoms, q, *, coordinates, fallback_to_cartesian=True, **kwargs):
    coordinates = str(coordinates).lower()
    if coordinates not in {"cartesian", "internal"}:
        raise ValueError("conditional_coordinates must be 'cartesian' or 'internal'")
    if coordinates == "cartesian":
        return _conditional_cartesian_optimize(
            qcinput,
            atoms,
            q,
            energy_tol=kwargs["energy_tol"],
            maxstep=kwargs["maxstep"],
            step_max_component=kwargs["cartesian_step_max_component"],
            print_report=kwargs["print_report"],
        )
    try:
        return _conditional_internal_optimize(
            qcinput,
            atoms,
            q,
            energy_tol=kwargs["energy_tol"],
            maxstep=kwargs["maxstep"],
            max_step_internal=kwargs["max_step_internal"],
            best_fit_iters=kwargs["best_fit_iters"],
            best_fit_rms_tol=kwargs["best_fit_rms_tol"],
            use_redundant_internals=kwargs["use_redundant_internals"],
            connectivity_kcn=kwargs["connectivity_kcn"],
            connectivity_facmin=kwargs["connectivity_facmin"],
            connect_fragments=kwargs["connect_fragments"],
            linear_bends=kwargs["linear_bends"],
            linear_bend_threshold_degrees=kwargs["linear_bend_threshold_degrees"],
            internal_hessian_model=kwargs.get("internal_hessian_model", "simple"),
            print_report=kwargs["print_report"],
        )
    except Exception:
        if not fallback_to_cartesian:
            raise
        if kwargs["print_report"]:
            print("  RDA internal conditional optimization failed; falling back to Cartesian.", flush=True)
        return _conditional_cartesian_optimize(
            qcinput,
            atoms,
            q,
            energy_tol=kwargs["energy_tol"],
            maxstep=kwargs["maxstep"],
            step_max_component=kwargs["cartesian_step_max_component"],
            print_report=kwargs["print_report"],
        )


def _make_point(label, q_start, q_copt, energy, nsteps, q_is, q_fs, distance_fn, distance_tol, *, beta=None, gamma=None, message=""):
    direction, delta_d_is, delta_d_fs = _classify_direction(
        q_start,
        q_copt,
        q_is,
        q_fs,
        distance_fn,
        distance_tol,
    )
    return RDAPoint(
        label=label,
        q_start=np.asarray(q_start, dtype=float).copy(),
        q_copt=np.asarray(q_copt, dtype=float).copy(),
        energy=float(energy),
        nsteps=int(nsteps),
        direction=direction,
        delta_d_is=float(delta_d_is),
        delta_d_fs=float(delta_d_fs),
        beta=beta,
        gamma=gamma,
        message=message,
    )


def _select_bracket_candidate(q_a, q_b, q_is, q_fs, distance_fn, gamma_values):
    best_q = None
    best_gamma = None
    best_score = None
    for gamma in gamma_values:
        q_trial = _interpolate(q_a, q_b, gamma)
        score = abs(distance_fn(q_trial, q_is) - distance_fn(q_trial, q_fs))
        if best_score is None or score < best_score:
            best_q = q_trial
            best_gamma = float(gamma)
            best_score = score
    return best_q, best_gamma


def _find_reactive_coordinate_indices(ic, reactive_bonds, reactive_angles, reactive_dihedrals):
    indices = []
    modes = []
    labels = []
    for reference in reactive_bonds:
        found, canonical = find_internal_coordinate_indices(ic, "bond", reference, label="reactive_bond")
        for idx in found:
            indices.append(idx)
            modes.append("bond")
            labels.append(coordinate_label("bond", canonical))

    for reference in reactive_angles:
        found, canonical = find_internal_coordinate_indices(ic, "angle", reference, label="reactive_angle")
        for idx in found:
            indices.append(idx)
            modes.append("angle")
            labels.append(coordinate_label("angle", canonical))

    for reference in reactive_dihedrals:
        found, canonical = find_internal_coordinate_indices(ic, "dihedral", reference, label="reactive_dihedral")
        for idx in found:
            indices.append(idx)
            modes.append("dihedral")
            labels.append(coordinate_label("dihedral", canonical))
    return indices, modes, labels


def _interpolate_internal_targets(qs_is, qs_fs, coord_indices, modes, alpha, ndihedrals, nimpropers):
    values = []
    for idx, mode in zip(coord_indices, modes):
        if mode == "dihedral":
            dq = diff_internals(
                np.asarray([qs_fs[idx]], dtype=float),
                np.asarray([qs_is[idx]], dtype=float),
                1,
                0,
            )[0]
            values.append(qs_is[idx] + float(alpha) * dq)
        else:
            values.append((1.0 - float(alpha)) * qs_is[idx] + float(alpha) * qs_fs[idx])
    del ndihedrals, nimpropers
    return values


def _run_poormans_neb(
    qcinput,
    atoms,
    q_is,
    q_fs,
    *,
    reactive_bonds,
    reactive_angles,
    reactive_dihedrals,
    nimages,
    maxiter,
    max_step_internal,
    energy_tol,
    max_gradient,
    rms_gradient,
    max_step,
    rms_step,
    best_fit_iters,
    best_fit_rms_tol,
    use_redundant_internals,
    connectivity_kcn,
    connectivity_facmin,
    connect_fragments,
    linear_bends,
    linear_bend_threshold_degrees,
    poormans_NEB_file,
    poormans_NEB_profile_file,
    poormans_NEB_plot_file,
    internal_hessian_model,
    print_report,
):
    if not reactive_bonds and not reactive_angles and not reactive_dihedrals:
        raise ValueError("poormans_NEB requires reactive_bonds, reactive_angles, or reactive_dihedrals")
    nimages = int(nimages)
    if nimages < 3:
        raise ValueError("poormans_NEB_images must be at least 3")

    model = default_connectivity_model(
        atoms,
        kcn=connectivity_kcn,
        facmin=connectivity_facmin,
        connect_fragments=connect_fragments,
    )
    ic, qs_is, _bpg, _redundant_system = _initial_internal_coordinate_state(
        q_is,
        atoms,
        model,
        use_redundant_internals=use_redundant_internals,
        linear_bends=linear_bends,
        linear_bend_threshold_degrees=linear_bend_threshold_degrees,
        extra_bonds=reactive_bonds,
        extra_angles=reactive_angles,
        extra_dihedrals=reactive_dihedrals,
    )
    qs_fs = compute_internals(q_fs, ic)
    coord_indices, modes, labels = _find_reactive_coordinate_indices(
        ic,
        reactive_bonds,
        reactive_angles,
        reactive_dihedrals,
    )

    images = []
    if print_report:
        print("RDA poormans_NEB: constrained relaxation of endpoint-interpolated images", flush=True)
        print(f"RDA poormans_NEB: images={nimages}", flush=True)
        print(f"RDA poormans_NEB: constrained coordinates={labels}", flush=True)

    for image_index in range(nimages):
        alpha = image_index / float(nimages - 1)
        q_start = _interpolate(q_is, q_fs, alpha)
        if image_index == 0:
            q_relaxed = q_is.copy()
            energy = _energy(qcinput, q_relaxed, atoms)
            gradient = _gradient(qcinput, q_relaxed, atoms)
            converged = True
            opt_steps = 0
            message = "Endpoint image."
        elif image_index == nimages - 1:
            q_relaxed = q_fs.copy()
            energy = _energy(qcinput, q_relaxed, atoms)
            gradient = _gradient(qcinput, q_relaxed, atoms)
            converged = True
            opt_steps = 0
            message = "Endpoint image."
        else:
            targets = _interpolate_internal_targets(
                qs_is,
                qs_fs,
                coord_indices,
                modes,
                alpha,
                ic.ndihedrals,
                ic.nimpropers,
            )
            if print_report:
                print(f"RDA poormans_NEB: relaxing image {image_index}/{nimages - 1}, alpha={alpha:.3f}", flush=True)
            q_relaxed, energy, gradient, converged, opt_steps, message = _relax_scan_point_multi(
                qcinput,
                atoms,
                q_start,
                ic,
                coord_indices,
                targets,
                modes,
                use_redundant_internals=use_redundant_internals,
                maxiter=maxiter,
                max_step_internal=max_step_internal,
                max_gradient=max_gradient,
                rms_gradient=rms_gradient,
                max_step=max_step,
                rms_step=rms_step,
                energy_tol=energy_tol,
                best_fit_iters=best_fit_iters,
                best_fit_rms_tol=best_fit_rms_tol,
                print_report=print_report,
                scan_step=image_index,
                reference_label="poormans_NEB",
                target_value=f"alpha={alpha:.3f}",
                target_unit="",
                internal_hessian_model=internal_hessian_model,
            )
        qs_relaxed = compute_internals(q_relaxed, ic)
        images.append(
            {
                "index": image_index,
                "alpha": alpha,
                "q": q_relaxed,
                "energy": energy,
                "gradient": gradient,
                "converged": converged,
                "opt_steps": opt_steps,
                "message": message,
                "label": f"poormans_NEB_{image_index:03d}",
                "reactive_values": [qs_relaxed[idx] for idx in coord_indices],
            }
        )

    _write_poormans_neb_chain(poormans_NEB_file, atoms, images)
    _write_poormans_neb_profile(poormans_NEB_profile_file, images, labels)
    _write_poormans_neb_plot(poormans_NEB_plot_file, images, labels)
    interior = images[1:-1]
    selected = max(interior, key=lambda image: image["energy"]) if interior else max(images, key=lambda image: image["energy"])
    if print_report:
        _print_poormans_neb_profile(images, labels)
        print(
            "RDA poormans_NEB: selected highest-energy relaxed image "
            f"{selected['index']} with E={selected['energy']:.10f} Eh",
            flush=True,
        )
        if poormans_NEB_file:
            print(f"RDA poormans_NEB: wrote relaxed chain to {poormans_NEB_file}", flush=True)
        if poormans_NEB_profile_file:
            print(f"RDA poormans_NEB: wrote energetics profile to {poormans_NEB_profile_file}", flush=True)
        if poormans_NEB_plot_file:
            print(f"RDA poormans_NEB: wrote profile plot to {poormans_NEB_plot_file}", flush=True)
    return images, selected


def _finish_result(
    *,
    atoms,
    q,
    reaction_direction,
    points,
    direction,
    bracketed,
    message,
    q_is,
    q_fs,
    rda_trajectory_file,
    rda_chain_file,
    print_report,
    poormans_NEB_file=None,
    poormans_NEB_profile_file=None,
    poormans_NEB_plot_file=None,
):
    _write_rda_trajectory(rda_trajectory_file, atoms, q_is, q_fs, points)
    _write_rda_chain(rda_chain_file, atoms, q_is, q_fs, points, q)
    if print_report:
        _print_rda_table(points)
        if rda_trajectory_file:
            print(f"RDA: wrote detailed trajectory to {rda_trajectory_file}", flush=True)
        if rda_chain_file:
            print(f"RDA: wrote NEB-like chain to {rda_chain_file}", flush=True)
    return RDAGuessResult(
        atoms=atoms,
        q=q,
        reaction_direction=reaction_direction,
        points=points,
        direction=direction,
        bracketed=bracketed,
        message=message,
        rda_trajectory_file=rda_trajectory_file,
        rda_chain_file=rda_chain_file,
        poormans_NEB_file=poormans_NEB_file,
        poormans_NEB_profile_file=poormans_NEB_profile_file,
        poormans_NEB_plot_file=poormans_NEB_plot_file,
    )


def generate_rda_ts_guess(
    qcinput,
    xyz_initial,
    xyz_final,
    *,
    active_atoms=None,
    align=True,
    align_atoms=None,
    distance="hybrid",
    conditional_coordinates="internal",
    first_energy_tol_ev=0.01,
    later_energy_tol_ev=0.05,
    distance_tol=0.05,
    max_conditional_steps=25,
    beta_start=0.5,
    beta_step=0.1,
    max_bracket_attempts=6,
    gamma_values=None,
    max_step_internal=0.10,
    cartesian_step_max_component=0.10,
    best_fit_iters=8,
    best_fit_rms_tol=1.0e-7,
    connectivity_kcn=16.0,
    connectivity_facmin=0.45,
    connect_fragments=True,
    use_redundant_internals=True,
    linear_bends=False,
    linear_bend_threshold_degrees=None,
    internal_hessian_model="simple",
    fallback_to_cartesian=True,
    print_report=True,
    rda_trajectory_file="rda_trajectory.xyz",
    rda_chain_file="rda_chain.xyz",
    endpoint_same_rmsd_tol=0.05,
    endpoint_active_rmsd_tol=0.05,
    reactive_bonds=None,
    reactive_angles=None,
    reactive_dihedrals=None,
    reactive_coordinate_weights=None,
    poormans_NEB=False,
    poormans_NEB_images=7,
    poormans_NEB_maxiter=40,
    poormans_NEB_file="poormans_NEB_traj_from_constrained_RDA.xyz",
    poormans_NEB_profile_file="poormans_NEB_energetics_profile.dat",
    poormans_NEB_plot_file="poormans_NEB_energetics_profile.png",
    reporter=None,
    verbosity=1,
    log_file=None,
    log_append=False,
    _reporter_active=False,
):
    """Generate a quasi-TS guess from two endpoint structures using RDA.

    This function is only a TS-guess generator.  It does not replace the Baker
    transition-state optimizer.  The intended workflow is:

        reactant/product endpoints
            -> generate_rda_ts_guess(...)
            -> optimize_transition_state(qcinput, rda_result.xyz(), ...)
            -> frequency/IRC validation

    Endpoint assumptions
    --------------------
    xyz_initial and xyz_final must contain the same atoms in the same
    order.  The function does not solve atom mapping.  If the reactant and
    product are atom-scrambled, map them first.  If align=True, the final
    endpoint is Kabsch-aligned to the initial endpoint before interpolation.
    align_atoms can restrict this alignment to a chemically meaningful
    subset, which is usually safer for large or floppy systems.

    Standard RDA mode
    -----------------
    With poormans_NEB=False the function follows the RDA bracketing logic:

    1. Build the Cartesian midpoint between endpoints.
    2. Conditionally optimize that midpoint only until the RDA energy-change
       threshold is met.  This is intentionally not a full minimization.
    3. Determine whether the relaxed midpoint moves toward the initial endpoint,
       toward the final endpoint, or is nondirectional.
    4. If needed, generate beta candidates toward the opposite endpoint until
       opposite directionality is found.
    5. Select a quasi-TS candidate and return it as RDAGuessResult.

    The conditional optimization can be run in Cartesian or internal coordinates
    through conditional_coordinates.  Internal coordinates are generally more
    chemically meaningful, but Cartesian fallback is available with
    fallback_to_cartesian=True.

    Distance-analysis choices
    -------------------------
    distance="cartesian"
        Directionality is based on active-atom Cartesian RMSD.  Set
        active_atoms to keep inactive atoms from dominating the metric.

    distance="hybrid"  (default)
        Cartesian interpolation is still used, but directionality is measured
        with an automatically generated internal-coordinate set when possible.
        If internal-coordinate construction fails, active-atom Cartesian RMSD is
        used as fallback.

    distance="internal"
        Like hybrid, but failure to build the internal-coordinate metric is an
        error instead of falling back to Cartesian RMSD.

    distance="reactive_internal"
        Use only user-provided chemically important coordinates for
        directionality.  This is recommended for reactions such as H transfer,
        proton shuttling, torsions, ring opening/closing, and epoxide formation.
        Provide one or more of:

            reactive_bonds=[(i, j), ...]
            reactive_angles=[(i, j, k), ...]
            reactive_dihedrals=[(i, j, k, l), ...]

        Indices are Python 0-based.  For example, a C2-O15 forming bond in
        1-based chemical notation is reactive_bonds=[(1, 14)].

    Poor-man's NEB mode
    -------------------
    If poormans_NEB=True, the function does not use the normal RDA
    beta-bracketing procedure.  Instead it builds a chain of endpoint-
    interpolated images and performs constrained relaxation of each interior
    image.  The constrained coordinates are exactly the coordinates supplied in
    reactive_bonds, reactive_angles, and reactive_dihedrals.  Their
    values are interpolated between endpoint values, while all remaining
    internal coordinates are relaxed.  This is not a true NEB: there are no
    spring forces and no tangent projection.  It is a cheap string/NEB-like
    preconditioner for generating a better TS guess.

    Poor-man's NEB options:

        poormans_NEB_images
            Number of images including endpoints.  Must be at least 3.

        poormans_NEB_maxiter
            Maximum constrained-relaxation iterations for each interior image.

        poormans_NEB_file
            XYZ trajectory containing the constrained-relaxed image chain.
            The default is poormans_NEB_traj_from_constrained_RDA.xyz.

        poormans_NEB_profile_file
            Text table containing the poor-man's NEB energetics profile:
            image index, alpha, energy, relative energy, convergence, step
            count, and constrained reactive coordinate values.  The default is
            poormans_NEB_energetics_profile.dat.

        poormans_NEB_plot_file
            PNG plot of the same profile.  It is written automatically whenever
            poormans_NEB is run and matplotlib is available.  The default is
            poormans_NEB_energetics_profile.png.

    Output files
    ------------
    rda_trajectory_file writes a detailed RDA trajectory with candidate
    start and conditionally optimized structures.

    rda_chain_file writes a compact endpoint-to-guess-to-endpoint path-like
    chain for visualization.

    poormans_NEB_file writes the constrained-relaxed image chain only when
    poormans_NEB=True.

    Return value
    ------------
    The result contains:

        result.q
            Cartesian coordinates in bohr for the quasi-TS guess.

        result.xyz()
            XYZ string in Angstrom, ready for optimize_transition_state.

        result.reaction_direction
            Approximate Cartesian direction when available.  For internal
            reaction references such as reaction_mode="bond" or
            reaction_mode="angle", it is usually better to pass those simple
            references directly to the Baker optimizer instead of using this
            Cartesian vector.
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
        options.pop("xyz_initial")
        options.pop("xyz_final")
        options["reporter"] = reporter
        options["print_report"] = print_report and reporter.should_report()
        options["_reporter_active"] = True
        with reporter.capture_prints():
            return generate_rda_ts_guess(qcinput, xyz_initial, xyz_final, **options)

    qcinput = validate_qchem_input(qcinput)
    atoms_is, q_is = _atoms_q_from_xyz(xyz_initial)
    atoms_fs, q_fs = _atoms_q_from_xyz(xyz_final)
    if atoms_is != atoms_fs:
        raise ValueError("RDA requires identical atom ordering and atom labels in initial and final geometries")
    atoms = list(atoms_is)
    natoms = len(atoms)
    active_atoms = _as_atom_indices(active_atoms, natoms, "active_atoms")
    align_atoms = _as_atom_indices(align_atoms, natoms, "align_atoms") if align_atoms is not None else active_atoms
    reactive_bonds = _as_index_tuples(reactive_bonds, natoms, 2, "reactive_bonds")
    reactive_angles = _as_index_tuples(reactive_angles, natoms, 3, "reactive_angles")
    reactive_dihedrals = _as_index_tuples(reactive_dihedrals, natoms, 4, "reactive_dihedrals")
    if align:
        if print_report:
            print("RDA: aligning final geometry to initial geometry", flush=True)
        q_fs = _kabsch_align(q_fs, q_is, align_atoms)

    distance_fn, distance_used = _make_distance_function(
        q_is,
        q_fs,
        atoms,
        active_atoms,
        distance,
        linear_bend_threshold_degrees,
        reactive_bonds=reactive_bonds,
        reactive_angles=reactive_angles,
        reactive_dihedrals=reactive_dihedrals,
        reactive_coordinate_weights=reactive_coordinate_weights,
    )
    eps1 = float(first_energy_tol_ev) * EV_TO_HARTREE
    eps2 = float(later_energy_tol_ev) * EV_TO_HARTREE
    if gamma_values is None:
        gamma_values = np.linspace(0.1, 0.9, 9)
    if print_report:
        print("RDA: starting quasi-TS guess generation", flush=True)
        print(f"RDA: atoms={natoms} active_atoms={tuple(int(i) for i in active_atoms)}", flush=True)
        print(f"RDA: distance analysis uses {distance_used} coordinates", flush=True)
        if str(distance).lower() == "reactive_internal":
            print(f"RDA: reactive bonds={reactive_bonds}", flush=True)
            print(f"RDA: reactive angles={reactive_angles}", flush=True)
            print(f"RDA: reactive dihedrals={reactive_dihedrals}", flush=True)
        print(f"RDA: conditional optimization coordinates={conditional_coordinates}", flush=True)
        endpoint_e_is = _energy(qcinput, q_is, atoms)
        endpoint_e_fs = _energy(qcinput, q_fs, atoms)
        endpoint_rmsd = _cartesian_all_atom_rmsd(q_is, q_fs) * BOHR_TO_ANGSTROM
        endpoint_active_rmsd = _cartesian_rmsd(q_is, q_fs, active_atoms) * BOHR_TO_ANGSTROM
        endpoint_distance = distance_fn(q_is, q_fs)
        print("RDA endpoint diagnostics:", flush=True)
        print(f"  E(IS) [Eh] = {endpoint_e_is:.10f}", flush=True)
        print(f"  E(FS) [Eh] = {endpoint_e_fs:.10f}", flush=True)
        print(f"  dE FS-IS [Eh] = {endpoint_e_fs - endpoint_e_is:.10f}", flush=True)
        print(f"  all-atom RMSD [Angstrom] = {endpoint_rmsd:.6f}", flush=True)
        print(f"  active-atom RMSD [Angstrom] = {endpoint_active_rmsd:.6f}", flush=True)
        print(f"  RDA endpoint distance [{distance_used}] = {endpoint_distance:.6e}", flush=True)
        if endpoint_rmsd < endpoint_same_rmsd_tol:
            print(
                "RDA warning: endpoint all-atom RMSD is below "
                f"{endpoint_same_rmsd_tol:.3f} Angstrom; endpoints may be the same minimum.",
                flush=True,
            )
        if endpoint_active_rmsd < endpoint_active_rmsd_tol:
            print(
                "RDA warning: active-atom RMSD is below "
                f"{endpoint_active_rmsd_tol:.3f} Angstrom; reactive coordinates may barely differ.",
                flush=True,
            )

    if poormans_NEB:
        images, selected = _run_poormans_neb(
            qcinput,
            atoms,
            q_is,
            q_fs,
            reactive_bonds=reactive_bonds,
            reactive_angles=reactive_angles,
            reactive_dihedrals=reactive_dihedrals,
            nimages=poormans_NEB_images,
            maxiter=poormans_NEB_maxiter,
            max_step_internal=max_step_internal,
            energy_tol=eps2,
            max_gradient=3.0e-4,
            rms_gradient=1.0e-4,
            max_step=4.0e-3,
            rms_step=2.0e-3,
            best_fit_iters=best_fit_iters,
            best_fit_rms_tol=best_fit_rms_tol,
            use_redundant_internals=use_redundant_internals,
            connectivity_kcn=connectivity_kcn,
            connectivity_facmin=connectivity_facmin,
            connect_fragments=connect_fragments,
            linear_bends=linear_bends,
            linear_bend_threshold_degrees=linear_bend_threshold_degrees,
            poormans_NEB_file=poormans_NEB_file,
            poormans_NEB_profile_file=poormans_NEB_profile_file,
            poormans_NEB_plot_file=poormans_NEB_plot_file,
            internal_hessian_model=internal_hessian_model,
            print_report=print_report,
        )
        points = [
            _make_point(
                image["label"],
                image["q"],
                image["q"],
                image["energy"],
                image["opt_steps"],
                q_is,
                q_fs,
                distance_fn,
                distance_tol,
                message=image["message"],
            )
            for image in images
        ]
        return _finish_result(
            atoms=atoms,
            q=selected["q"],
            reaction_direction=_normalize(q_fs - q_is),
            points=points,
            direction="poormans_NEB",
            bracketed=False,
            message=(
                "RDA poormans_NEB selected the highest-energy constrained-relaxed image "
                "as the quasi-TS guess."
            ),
            q_is=q_is,
            q_fs=q_fs,
            rda_trajectory_file=rda_trajectory_file,
            rda_chain_file=rda_chain_file,
            poormans_NEB_file=poormans_NEB_file,
            poormans_NEB_profile_file=poormans_NEB_profile_file,
            poormans_NEB_plot_file=poormans_NEB_plot_file,
            print_report=print_report,
        )

    common_kwargs = dict(
        maxstep=max_conditional_steps,
        max_step_internal=max_step_internal,
        cartesian_step_max_component=cartesian_step_max_component,
        best_fit_iters=best_fit_iters,
        best_fit_rms_tol=best_fit_rms_tol,
        use_redundant_internals=use_redundant_internals,
        connectivity_kcn=connectivity_kcn,
        connectivity_facmin=connectivity_facmin,
        connect_fragments=connect_fragments,
        linear_bends=linear_bends,
        linear_bend_threshold_degrees=linear_bend_threshold_degrees,
        internal_hessian_model=internal_hessian_model,
        print_report=print_report,
    )

    points: list[RDAPoint] = []
    q_mid = _interpolate(q_is, q_fs, 0.5)
    if print_report:
        print("RDA: conditional optimization of midpoint alpha=0.5", flush=True)
    q_mid_copt, e_mid, n_mid, msg_mid = _conditional_optimize(
        qcinput,
        atoms,
        q_mid,
        coordinates=conditional_coordinates,
        fallback_to_cartesian=fallback_to_cartesian,
        energy_tol=eps1,
        **common_kwargs,
    )
    mid_point = _make_point(
        "midpoint",
        q_mid,
        q_mid_copt,
        e_mid,
        n_mid,
        q_is,
        q_fs,
        distance_fn,
        distance_tol,
        message=f"{msg_mid} distance={distance_used}",
    )
    points.append(mid_point)
    if print_report:
        print(
            "RDA: midpoint finished "
            f"direction={mid_point.direction} dIS={mid_point.delta_d_is:.6e} "
            f"dFS={mid_point.delta_d_fs:.6e} steps={mid_point.nsteps}",
            flush=True,
        )
    if mid_point.direction == "ND":
        if print_report:
            print("RDA: ended, midpoint accepted as quasi-TS guess", flush=True)
        return _finish_result(
            atoms=atoms,
            q=q_mid_copt,
            reaction_direction=_normalize(q_fs - q_is),
            points=points,
            direction="ND",
            bracketed=False,
            message="RDA accepted the conditionally optimized midpoint as a quasi-TS guess.",
            q_is=q_is,
            q_fs=q_fs,
            rda_trajectory_file=rda_trajectory_file,
            rda_chain_file=rda_chain_file,
            print_report=print_report,
        )

    reference = q_fs if mid_point.direction == "IS" else q_is
    previous_direction = mid_point.direction
    previous_copt = q_mid_copt
    bracket_pair = None

    beta = float(beta_start)
    for attempt in range(1, int(max_bracket_attempts) + 1):
        beta_clamped = min(1.0, max(0.0, beta))
        q_beta = _interpolate(q_mid_copt, reference, beta_clamped)
        if print_report:
            print(f"RDA: conditional optimization of beta candidate {attempt}, beta={beta_clamped:.3f}", flush=True)
        q_beta_copt, e_beta, n_beta, msg_beta = _conditional_optimize(
            qcinput,
            atoms,
            q_beta,
            coordinates=conditional_coordinates,
            fallback_to_cartesian=fallback_to_cartesian,
            energy_tol=eps2,
            **common_kwargs,
        )
        beta_point = _make_point(
            f"beta_{attempt}",
            q_beta,
            q_beta_copt,
            e_beta,
            n_beta,
            q_is,
            q_fs,
            distance_fn,
            distance_tol,
            beta=beta_clamped,
            message=f"{msg_beta} distance={distance_used}",
        )
        points.append(beta_point)
        if print_report:
            print(
                "RDA: beta candidate finished "
                f"direction={beta_point.direction} dIS={beta_point.delta_d_is:.6e} "
                f"dFS={beta_point.delta_d_fs:.6e} steps={beta_point.nsteps}",
                flush=True,
            )
        if beta_point.direction == "ND":
            if print_report:
                print("RDA: ended, nondirectional beta candidate accepted as quasi-TS guess", flush=True)
            return _finish_result(
                atoms=atoms,
                q=q_beta_copt,
                reaction_direction=_normalize(q_beta_copt - previous_copt),
                points=points,
                direction="ND",
                bracketed=False,
                message="RDA accepted a nondirectional beta candidate as a quasi-TS guess.",
                q_is=q_is,
                q_fs=q_fs,
                rda_trajectory_file=rda_trajectory_file,
                rda_chain_file=rda_chain_file,
                print_report=print_report,
            )
        if beta_point.direction != previous_direction:
            bracket_pair = (previous_copt, q_beta_copt)
            break
        previous_copt = q_beta_copt
        previous_direction = beta_point.direction
        beta += float(beta_step)

    if bracket_pair is None:
        if print_report:
            print("RDA: ended without bracket; returning last candidate", flush=True)
        return _finish_result(
            atoms=atoms,
            q=points[-1].q_copt,
            reaction_direction=_normalize(points[-1].q_copt - q_mid_copt),
            points=points,
            direction=points[-1].direction,
            bracketed=False,
            message="RDA did not find opposite directionality before max_bracket_attempts; returning the last candidate.",
            q_is=q_is,
            q_fs=q_fs,
            rda_trajectory_file=rda_trajectory_file,
            rda_chain_file=rda_chain_file,
            print_report=print_report,
        )

    q_a, q_b = bracket_pair
    if print_report:
        print("RDA: opposite directionality found; selecting gamma quasi-TS candidate", flush=True)
    q_guess, gamma = _select_bracket_candidate(q_a, q_b, q_is, q_fs, distance_fn, gamma_values)
    gamma_point = _make_point(
        "gamma_selected",
        q_guess,
        q_guess,
        _energy(qcinput, q_guess, atoms),
        0,
        q_is,
        q_fs,
        distance_fn,
        distance_tol,
        gamma=gamma,
        message=f"Selected geometrically between opposite RDA directions; distance={distance_used}",
    )
    points.append(gamma_point)
    if print_report:
        print(f"RDA: ended with bracketed quasi-TS guess, gamma={gamma:.3f}", flush=True)
    return _finish_result(
        atoms=atoms,
        q=q_guess,
        reaction_direction=_normalize(q_b - q_a),
        points=points,
        direction="bracketed",
        bracketed=True,
        message="RDA found opposite directionality and generated a bracketed quasi-TS guess.",
        q_is=q_is,
        q_fs=q_fs,
        rda_trajectory_file=rda_trajectory_file,
        rda_chain_file=rda_chain_file,
        print_report=print_report,
    )
