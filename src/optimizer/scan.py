from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from normalmode.hessian import getHessian
from normalmode.normalmode import getNormalmode
from optimizer.internal_coords import (
    analyze_structure,
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
    build_redundant_internals,
    initial_redundant_hessian_from_model,
)
from optimizer.reporter import make_reporter
from optimizer.common import (
    _atoms_q_from_xyz,
    _bfgs_update_hessian,
    _energy,
    _gradient,
    step_limit,
)
from optimizer.coordinate_refs import (
    coordinate_label,
    find_single_internal_coordinate_index,
    validate_coordinate_reference,
)
from qchem_interfaces.qchem_validation import validate_qchem_input
from utils.atomic_masses import get_mass_vector
from utils.constants import ANGSTROM_TO_BOHR, AU_ANGULAR_FREQUENCY_TO_CM1, BOHR_TO_ANGSTROM, HARTREE_TO_KJMOL
from utils.format_and_print import makeXYZ


def _backtrack_constrained_trial(q, energy, step, project, evaluate, *, min_scale=1.0e-4):
    """Accept only a non-increasing constrained-relaxation step.

    ``project`` reapplies the scan constraint after every trial, so shrinking
    the free-coordinate displacement cannot accidentally release the target.
    """
    scale = 1.0
    q = np.asarray(q, dtype=float)
    step = np.asarray(step, dtype=float)
    while scale >= min_scale:
        trial_q = project(q + scale * step)
        trial_energy = float(evaluate(trial_q))
        if np.isfinite(trial_energy) and trial_energy <= float(energy):
            return trial_q, trial_energy, scale, True
        scale *= 0.5
    return q.copy(), float(energy), 0.0, False


@dataclass
class ScanPoint:
    index: int
    value: float
    value_unit: str
    q: np.ndarray
    energy: float
    gradient: np.ndarray
    converged: bool = False
    optimization_steps: int = 0
    message: str = ""


@dataclass
class CoordinateScanResult:
    atoms: list[str]
    mode: str
    reference: tuple[int, ...]
    points: list[ScanPoint]
    trajectory_file: str | None = None
    energy_profile_file: str | None = None
    energy_profile_plot: str | None = None


@dataclass
class XYZScanPoint:
    index: int
    value: float
    value_unit: str
    q: np.ndarray
    energy: float
    gradient: np.ndarray | None = None
    comment: str = ""


@dataclass
class XYZScanResult:
    atoms: list[str]
    mode: str
    reference: tuple[int, ...]
    points: list[XYZScanPoint]
    xyz_file: str | None = None
    energy_profile_file: str | None = None
    energy_profile_plot: str | None = None


@dataclass
class ScanGridPoint:
    index1: int
    index2: int
    value1: float
    value2: float
    value1_unit: str
    value2_unit: str
    q: np.ndarray
    energy: float
    gradient: np.ndarray
    converged: bool = False
    optimization_steps: int = 0
    message: str = ""


@dataclass
class CoordinateScan2DResult:
    atoms: list[str]
    mode1: str
    reference1: tuple[int, ...]
    mode2: str
    reference2: tuple[int, ...]
    points: list[ScanGridPoint]
    trajectory_file: str | None = None
    energy_profile_file: str | None = None
    energy_profile_plot: str | None = None


@dataclass
class NormalModeScanResult:
    atoms: list[str]
    mode_index: int
    frequency: float | None
    frequency_unit: str
    points: list[ScanPoint]
    trajectory_file: str | None = None
    hessian_file: str | None = None
    energy_profile_file: str | None = None
    energy_profile_plot: str | None = None


def _scan_reference_label(mode, reference):
    return coordinate_label(mode, reference)


def _scan_coordinate_index(ic, mode, reference):
    label = f"scan_{mode}"
    return find_single_internal_coordinate_index(ic, mode, reference, label=label)


def _prepare_scan_mode(scan_mode, scan_bond, scan_angle, scan_dihedral):
    mode = str(scan_mode).lower()
    if mode not in {"bond", "angle", "dihedral"}:
        raise ValueError("scan_mode must be 'bond', 'angle', or 'dihedral'")

    refs = {
        "bond": scan_bond,
        "angle": scan_angle,
        "dihedral": scan_dihedral,
    }
    if refs[mode] is None:
        raise ValueError(f"scan_mode='{mode}' requires scan_{mode}")
    extra = [name for name, value in refs.items() if name != mode and value is not None]
    if extra:
        raise ValueError(f"scan_mode='{mode}' only accepts scan_{mode}")
    return mode, refs[mode]


def _read_xyz_scan_frames(xyz):
    if xyz is None:
        raise ValueError("XYZ scan input cannot be None")
    xyz_file = None
    if hasattr(xyz, "read"):
        text = xyz.read()
    else:
        xyz_text = str(xyz)
        if "\n" not in xyz_text:
            try:
                with open(xyz_text, "r", encoding="utf-8") as handle:
                    text = handle.read()
                xyz_file = xyz_text
            except OSError:
                text = xyz_text
        else:
            text = xyz_text

    raw_lines = text.splitlines()
    frames = []
    index = 0
    while index < len(raw_lines):
        if not raw_lines[index].strip():
            index += 1
            continue
        try:
            natoms = int(raw_lines[index].split()[0])
        except (ValueError, IndexError) as exc:
            raise ValueError(
                "External XYZ scan input must be an XYZ trajectory with an atom-count line before each frame"
            ) from exc
        if natoms <= 0:
            raise ValueError("XYZ frame atom count must be positive")
        if index + 1 + natoms >= len(raw_lines):
            raise ValueError(f"XYZ frame declares {natoms} atoms but is incomplete")
        comment = raw_lines[index + 1].strip()
        coord_lines = raw_lines[index + 2:index + 2 + natoms]
        atoms = []
        q_angstrom = []
        for line in coord_lines:
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Invalid XYZ coordinate line: {line}")
            atoms.append(parts[0])
            q_angstrom.extend([float(parts[1]), float(parts[2]), float(parts[3])])
        q = np.asarray(q_angstrom, dtype=float) * ANGSTROM_TO_BOHR
        frames.append((atoms, q, comment))
        index += natoms + 2
    if not frames:
        raise ValueError("External XYZ scan input did not contain any frames")
    first_atoms = frames[0][0]
    for iframe, (atoms, _q, _comment) in enumerate(frames[1:], start=2):
        if atoms != first_atoms:
            raise ValueError(f"XYZ frame {iframe} has different atom labels/order than frame 1")
    return frames, xyz_file


def _direct_coordinate_value(q, mode, reference, unit):
    q = np.asarray(q, dtype=float).reshape(-1)
    xyz = q.reshape(-1, 3)
    ref = tuple(int(i) for i in reference)
    if mode == "bond":
        i, j = ref
        value = float(np.linalg.norm(xyz[j] - xyz[i]))
    elif mode == "angle":
        i, j, k = ref
        a = xyz[i] - xyz[j]
        b = xyz[k] - xyz[j]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 0.0:
            value = 0.0
        else:
            value = float(np.arccos(np.clip(np.dot(a, b) / denom, -1.0, 1.0)))
    else:
        i, j, k, l = ref
        b0 = xyz[j] - xyz[i]
        b1 = xyz[k] - xyz[j]
        b2 = xyz[l] - xyz[k]
        n0 = np.cross(b0, b1)
        n1 = np.cross(b1, b2)
        n0_norm = float(np.linalg.norm(n0))
        n1_norm = float(np.linalg.norm(n1))
        b1_norm = float(np.linalg.norm(b1))
        if n0_norm <= 0.0 or n1_norm <= 0.0 or b1_norm <= 0.0:
            value = 0.0
        else:
            n0 = n0 / n0_norm
            n1 = n1 / n1_norm
            m1 = np.cross(n0, b1 / b1_norm)
            value = float(np.arctan2(np.dot(m1, n1), np.dot(n0, n1)))
    return _internal_value_to_unit(value, mode, unit)


def _default_unit(mode, unit):
    if unit is not None:
        return str(unit).lower()
    return "angstrom" if mode == "bond" else "degree"


def _values_to_internal(values, mode, unit):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("Scan requires at least one target value")
    if mode == "bond":
        if unit in {"angstrom", "ang", "a"}:
            return values * ANGSTROM_TO_BOHR, "angstrom"
        if unit == "bohr":
            return values, "bohr"
        raise ValueError("Bond scan unit must be 'angstrom' or 'bohr'")

    if unit in {"degree", "degrees", "deg"}:
        return np.deg2rad(values), "degree"
    if unit in {"radian", "radians", "rad"}:
        return values, "radian"
    raise ValueError("Angle and dihedral scan unit must be 'degree' or 'radian'")


def _internal_value_to_unit(value, mode, unit):
    if mode == "bond":
        return float(value * BOHR_TO_ANGSTROM) if unit == "angstrom" else float(value)
    return float(np.rad2deg(value)) if unit == "degree" else float(value)


def _scan_values(mode, unit, values, start, stop, nsteps):
    if values is None:
        if start is None or stop is None or nsteps is None:
            raise ValueError("Provide either values=[...] or start=..., stop=..., nsteps=...")
        if int(nsteps) < 2:
            raise ValueError("nsteps must be at least 2 when start/stop are used")
        values = np.linspace(float(start), float(stop), int(nsteps))
    return _values_to_internal(values, mode, unit)


def _write_scan_frame(handle, atoms, q, point, mode, reference_label):
    handle.write(f"{len(atoms)}\n")
    handle.write(
        "scan_step=%d mode=%s value=%.10f %s E=%.12f Eh GradMax=%.6e relaxed=%s opt_steps=%d\n"
        % (
            point.index,
            reference_label,
            point.value,
            point.value_unit,
            point.energy,
            float(np.max(np.abs(point.gradient))),
            str(point.converged),
            int(point.optimization_steps),
        )
    )
    handle.write(makeXYZ(atoms, np.asarray(q, dtype=float).reshape(-1)))


def _write_scan2d_frame(handle, atoms, q, point, reference_label1, reference_label2):
    handle.write(f"{len(atoms)}\n")
    handle.write(
        "scan_grid=%d,%d mode1=%s value1=%.10f %s mode2=%s value2=%.10f %s "
        "E=%.12f Eh GradMax=%.6e relaxed=%s opt_steps=%d\n"
        % (
            point.index1,
            point.index2,
            reference_label1,
            point.value1,
            point.value1_unit,
            reference_label2,
            point.value2,
            point.value2_unit,
            point.energy,
            float(np.max(np.abs(point.gradient))),
            str(point.converged),
            int(point.optimization_steps),
        )
    )
    handle.write(makeXYZ(atoms, np.asarray(q, dtype=float).reshape(-1)))


def _write_normal_mode_scan_frame(handle, atoms, q, point, mode_label):
    handle.write(f"{len(atoms)}\n")
    handle.write(
        "normal_mode_scan_step=%d mode=%s displacement=%.10f %s "
        "E=%.12f Eh GradMax=%.6e relaxed=%s opt_steps=%d\n"
        % (
            point.index,
            mode_label,
            point.value,
            point.value_unit,
            point.energy,
            float(np.max(np.abs(point.gradient))),
            str(point.converged),
            int(point.optimization_steps),
        )
    )
    handle.write(makeXYZ(atoms, np.asarray(q, dtype=float).reshape(-1)))


def _write_scan_energy_profile_1d(points, *, label, output_file, plot_file=None, print_report=True):
    if not output_file or not points:
        return None, None
    energies = np.asarray([point.energy for point in points], dtype=float)
    first_energy = float(energies[0])
    min_energy = float(np.min(energies))
    with open(output_file, "w", encoding="utf-8") as handle:
        handle.write(
            "# step coordinate_value coordinate_unit energy_Eh "
            "relative_to_first_kcal_mol relative_to_min_kcal_mol converged optimization_steps label\n"
        )
        for point in points:
            rel_first = (point.energy - first_energy) * HARTREE_TO_KJMOL / 4.184
            rel_min = (point.energy - min_energy) * HARTREE_TO_KJMOL / 4.184
            handle.write(
                f"{point.index:8d} {point.value:18.10f} {point.value_unit:18s} "
                f"{point.energy:22.12f} {rel_first:22.10f} {rel_min:22.10f} "
                f"{int(bool(point.converged)):4d} {int(point.optimization_steps):8d} {label}\n"
            )
    made_plot = None
    if plot_file:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            x = np.asarray([point.value for point in points], dtype=float)
            y = (energies - min_energy) * HARTREE_TO_KJMOL / 4.184
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            ax.plot(x, y, marker="o", linewidth=1.5)
            ax.set_xlabel(f"{label} [{points[0].value_unit}]")
            ax.set_ylabel("relative energy [kcal/mol]")
            ax.set_title("Scan energy profile")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_file, dpi=180)
            plt.close(fig)
            made_plot = plot_file
        except Exception as exc:
            if print_report:
                print(f"matplotlib plot skipped for scan profile: {exc}", flush=True)
    if print_report:
        print(f"Scan energy profile written to {output_file}", flush=True)
        if made_plot:
            print(f"Scan energy plot written to {made_plot}", flush=True)
    return output_file, made_plot


def _write_xyz_scan_energy_profile(points, *, label, output_file, plot_file=None, print_report=True):
    if not output_file or not points:
        return None, None
    energies = np.asarray([point.energy for point in points], dtype=float)
    first_energy = float(energies[0])
    min_energy = float(np.min(energies))
    with open(output_file, "w", encoding="utf-8") as handle:
        handle.write(
            "# frame coordinate_value coordinate_unit energy_Eh "
            "relative_to_first_kcal_mol relative_to_min_kcal_mol gradient_max label comment\n"
        )
        for point in points:
            rel_first = (point.energy - first_energy) * HARTREE_TO_KJMOL / 4.184
            rel_min = (point.energy - min_energy) * HARTREE_TO_KJMOL / 4.184
            grad_max = (
                float(np.max(np.abs(point.gradient)))
                if point.gradient is not None
                else np.nan
            )
            handle.write(
                f"{point.index:8d} {point.value:18.10f} {point.value_unit:18s} "
                f"{point.energy:22.12f} {rel_first:22.10f} {rel_min:22.10f} "
                f"{grad_max:16.8e} {label} {point.comment}\n"
            )
    made_plot = None
    if plot_file:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            x = np.asarray([point.value for point in points], dtype=float)
            y = (energies - min_energy) * HARTREE_TO_KJMOL / 4.184
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            ax.plot(x, y, marker="o", linewidth=1.5)
            ax.set_xlabel(f"{label} [{points[0].value_unit}]")
            ax.set_ylabel("relative energy [kcal/mol]")
            ax.set_title("External XYZ single-point energy profile")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_file, dpi=180)
            plt.close(fig)
            made_plot = plot_file
        except Exception as exc:
            if print_report:
                print(f"matplotlib plot skipped for external XYZ scan profile: {exc}", flush=True)
    if print_report:
        print(f"External XYZ scan energy profile written to {output_file}", flush=True)
        if made_plot:
            print(f"External XYZ scan energy plot written to {made_plot}", flush=True)
    return output_file, made_plot


def _write_scan_energy_profile_2d(points, *, label1, label2, output_file, plot_file=None, print_report=True):
    if not output_file or not points:
        return None, None
    energies = np.asarray([point.energy for point in points], dtype=float)
    first_energy = float(energies[0])
    min_energy = float(np.min(energies))
    with open(output_file, "w", encoding="utf-8") as handle:
        handle.write(
            "# index1 index2 value1 value2 unit1 unit2 energy_Eh "
            "relative_to_first_kcal_mol relative_to_min_kcal_mol converged optimization_steps label1 label2\n"
        )
        for point in points:
            rel_first = (point.energy - first_energy) * HARTREE_TO_KJMOL / 4.184
            rel_min = (point.energy - min_energy) * HARTREE_TO_KJMOL / 4.184
            handle.write(
                f"{point.index1:8d} {point.index2:8d} "
                f"{point.value1:18.10f} {point.value2:18.10f} "
                f"{point.value1_unit:18s} {point.value2_unit:18s} "
                f"{point.energy:22.12f} {rel_first:22.10f} {rel_min:22.10f} "
                f"{int(bool(point.converged)):4d} {int(point.optimization_steps):8d} {label1} {label2}\n"
            )
    made_plot = None
    if plot_file:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            x_unique = sorted({point.value1 for point in points})
            y_unique = sorted({point.value2 for point in points})
            rel_min_values = (energies - min_energy) * HARTREE_TO_KJMOL / 4.184
            fig, ax = plt.subplots(figsize=(6.5, 5.2))
            if len(x_unique) > 1 and len(y_unique) > 1 and len(x_unique) * len(y_unique) == len(points):
                x_index = {value: idx for idx, value in enumerate(x_unique)}
                y_index = {value: idx for idx, value in enumerate(y_unique)}
                z = np.full((len(y_unique), len(x_unique)), np.nan, dtype=float)
                for point, rel_min in zip(points, rel_min_values):
                    z[y_index[point.value2], x_index[point.value1]] = rel_min
                x_grid, y_grid = np.meshgrid(x_unique, y_unique)
                contour = ax.contourf(x_grid, y_grid, z, levels=16)
                ax.contour(x_grid, y_grid, z, levels=8, colors="black", linewidths=0.4, alpha=0.45)
                fig.colorbar(contour, ax=ax, label="relative energy [kcal/mol]")
            else:
                scatter = ax.scatter(
                    [point.value1 for point in points],
                    [point.value2 for point in points],
                    c=rel_min_values,
                    s=45,
                )
                fig.colorbar(scatter, ax=ax, label="relative energy [kcal/mol]")
            ax.set_xlabel(f"{label1} [{points[0].value1_unit}]")
            ax.set_ylabel(f"{label2} [{points[0].value2_unit}]")
            ax.set_title("2D scan energy profile")
            fig.tight_layout()
            fig.savefig(plot_file, dpi=180)
            plt.close(fig)
            made_plot = plot_file
        except Exception as exc:
            if print_report:
                print(f"matplotlib plot skipped for 2D scan profile: {exc}", flush=True)
    if print_report:
        print(f"2D scan energy profile written to {output_file}", flush=True)
        if made_plot:
            print(f"2D scan energy plot written to {made_plot}", flush=True)
    return output_file, made_plot


def _mass_weight_vector(mass):
    return np.sqrt(np.repeat(np.asarray(mass, dtype=float).reshape(-1), 3))


def _finite_difference_cartesian_hessian(qcinput, atoms, q, *, step=1.0e-3, print_report=True):
    q = np.asarray(q, dtype=float).reshape(-1)
    hessian = np.zeros((q.size, q.size), dtype=float)
    if print_report:
        print(
            "Computing finite-difference Cartesian Hessian from generic QC gradients "
            f"with step={float(step):.3e} bohr",
            flush=True,
        )
    for icoord in range(q.size):
        dq = np.zeros_like(q)
        dq[icoord] = float(step)
        grad_plus = _gradient(qcinput, q + dq, atoms)
        grad_minus = _gradient(qcinput, q - dq, atoms)
        hessian[:, icoord] = (grad_plus - grad_minus) / (2.0 * float(step))
    return 0.5 * (hessian + hessian.T)


def _normal_mode_scan_values(values, start, stop, nsteps, unit):
    unit = "mass_weighted_bohr" if unit is None else str(unit).lower()
    if values is None:
        if start is None or stop is None or nsteps is None:
            raise ValueError("Provide either values=[...] or start=..., stop=..., nsteps=...")
        if int(nsteps) < 2:
            raise ValueError("nsteps must be at least 2 when start/stop are used")
        values = np.linspace(float(start), float(stop), int(nsteps))
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("Normal-mode scan requires at least one target displacement")
    if unit in {"mass_weighted_bohr", "mw_bohr", "sqrtamu_bohr", "normal"}:
        return values, "mass_weighted_bohr"
    if unit in {"angstrom", "ang", "a"}:
        return values, "angstrom"
    if unit == "bohr":
        return values, "bohr"
    raise ValueError("normal-mode scan unit must be 'mass_weighted_bohr', 'angstrom', or 'bohr'")


def _cartesian_direction_from_normal_coordinate(mode_cart, target, unit):
    mode_cart = np.asarray(mode_cart, dtype=float).reshape(-1)
    if unit == "mass_weighted_bohr":
        return target * mode_cart
    max_component = float(np.max(np.abs(mode_cart))) if mode_cart.size else 0.0
    if max_component <= 0.0:
        raise ValueError("Normal-mode direction has zero Cartesian amplitude")
    scale = target / max_component
    if unit == "angstrom":
        scale *= ANGSTROM_TO_BOHR
    return scale * mode_cart


def _prepare_normal_mode_direction(
    qcinput,
    atoms,
    q,
    *,
    normal_mode_index,
    normal_mode_vector=None,
    normal_mode_vector_is_mass_weighted=False,
    hessian=None,
    hess_file="hessian_normal_mode_scan.hess",
    finite_difference_hessian_step=1.0e-3,
    linear=False,
    print_report=True,
):
    mass_values = get_mass_vector(atoms)
    if any(m is None for m in mass_values):
        raise ValueError("Normal-mode scan could not assign all atomic masses")
    mass = np.asarray(mass_values, dtype=float)
    sqrt_m = _mass_weight_vector(mass)

    if normal_mode_vector is not None:
        vector = np.asarray(normal_mode_vector, dtype=float).reshape(-1)
        if vector.size != q.size:
            raise ValueError("normal_mode_vector must have the same Cartesian size as the geometry")
        if normal_mode_vector_is_mass_weighted:
            mode_mw = vector.copy()
            mode_cart = mode_mw / sqrt_m
        else:
            mode_cart = vector.copy()
            mode_mw = sqrt_m * mode_cart
        norm = float(np.linalg.norm(mode_mw))
        if norm <= 0.0:
            raise ValueError("normal_mode_vector must not be zero")
        mode_mw /= norm
        mode_cart = mode_mw / sqrt_m
        return mode_cart, mode_mw, None, hessian

    if hessian is None:
        xyz_text = makeXYZ(atoms, q)
        try:
            hessian = getHessian(qcinput, hess_file, xyz_text)
        except Exception as exc:
            if print_report:
                print(
                    "Native Hessian call failed for this QC backend; "
                    f"falling back to finite-difference gradients. Reason: {exc}",
                    flush=True,
                )
            hessian = _finite_difference_cartesian_hessian(
                qcinput,
                atoms,
                q,
                step=finite_difference_hessian_step,
                print_report=print_report,
            )
            if hess_file:
                np.savetxt(hess_file, hessian, delimiter=" ", newline="\n")
    ww, _ww_low, L = getNormalmode(mass, hessian, linear=linear, q_eq=q)
    if not ww:
        raise ValueError("No vibrational normal modes were found")
    imode = int(normal_mode_index)
    if imode < 0 or imode >= len(ww):
        raise ValueError(f"normal_mode_index must be in the range 0..{len(ww) - 1}")
    mode_cart = np.asarray(L[:, imode], dtype=float).reshape(-1)
    mode_mw = sqrt_m * mode_cart
    norm = float(np.linalg.norm(mode_mw))
    if norm <= 0.0:
        raise ValueError(f"normal mode {imode} has zero norm")
    mode_mw /= norm
    mode_cart = mode_mw / sqrt_m
    return mode_cart, mode_mw, float(ww[imode]), hessian


def _normal_coordinate_value(q, q_ref, mode_mw, sqrt_m, unit):
    dq = np.asarray(q, dtype=float).reshape(-1) - np.asarray(q_ref, dtype=float).reshape(-1)
    if unit == "mass_weighted_bohr":
        return float(np.dot(mode_mw, sqrt_m * dq))
    max_disp = float(np.max(np.abs(dq))) if dq.size else 0.0
    if unit == "angstrom":
        return max_disp * BOHR_TO_ANGSTROM
    return max_disp


def _set_normal_mode_coordinate(q_ref, q, mode_cart, mode_mw, sqrt_m, target, unit):
    q_ref = np.asarray(q_ref, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float).reshape(-1)
    y = sqrt_m * (q - q_ref)
    y_perp = y - np.dot(mode_mw, y) * mode_mw
    if unit == "mass_weighted_bohr":
        y_new = y_perp + float(target) * mode_mw
        return q_ref + y_new / sqrt_m
    return q_ref + y_perp / sqrt_m + _cartesian_direction_from_normal_coordinate(mode_cart, float(target), unit)


def _relax_normal_mode_scan_point(
    qcinput,
    atoms,
    q,
    q_ref,
    mode_cart,
    mode_mw,
    target,
    unit,
    *,
    maxiter,
    max_step,
    max_gradient,
    rms_gradient,
    energy_tol,
    print_report=False,
    scan_step=None,
):
    mass = np.asarray(get_mass_vector(atoms), dtype=float)
    sqrt_m = _mass_weight_vector(mass)
    q = _set_normal_mode_coordinate(q_ref, q, mode_cart, mode_mw, sqrt_m, target, unit)
    energy = _energy(qcinput, q, atoms)
    message = "Maximum normal-mode constrained scan optimization steps reached."
    hess = np.eye(q.size)

    if print_report:
        print(
            f"  relaxing normal-mode scan step {int(scan_step):4d}: "
            f"target={float(target):10.5f} {unit} E0={energy:18.10f}",
            flush=True,
        )
        print(
            "  %5s %18s %18s %11s %11s %11s"
            % ("step", "E[Eh]", "dE[kJ/mol]", "max_step", "max_grad", "rms_grad"),
            flush=True,
        )

    for iteration in range(1, int(maxiter) + 1):
        grad_x = _gradient(qcinput, q, atoms)
        grad_y = grad_x / sqrt_m
        grad_y = grad_y - np.dot(mode_mw, grad_y) * mode_mw

        try:
            step_y = np.linalg.solve(hess, -grad_y)
        except np.linalg.LinAlgError:
            step_y = np.linalg.lstsq(hess, -grad_y, rcond=None)[0]
        step_y = step_y - np.dot(mode_mw, step_y) * mode_mw
        step_y = step_limit(step_y, max_component=max_step)

        trial_q, trial_energy, _scale, accepted = _backtrack_constrained_trial(
            q,
            energy,
            step_y / sqrt_m,
            lambda candidate: _set_normal_mode_coordinate(
                q_ref, candidate, mode_cart, mode_mw, sqrt_m, target, unit
            ),
            lambda candidate: _energy(qcinput, candidate, atoms),
        )
        if not accepted:
            return q, energy, grad_x, False, iteration - 1, "Normal-mode constrained scan line search failed."
        trial_grad_x = _gradient(qcinput, trial_q, atoms)
        trial_grad_y = trial_grad_x / sqrt_m
        trial_grad_y = trial_grad_y - np.dot(mode_mw, trial_grad_y) * mode_mw

        energy_change = trial_energy - energy
        grad_max = float(np.max(np.abs(trial_grad_y))) if trial_grad_y.size else 0.0
        grad_rms = float(np.linalg.norm(trial_grad_y) / np.sqrt(max(1, trial_grad_y.size)))
        cart_step = trial_q - q
        step_max = float(np.max(np.abs(cart_step))) if cart_step.size else 0.0

        if print_report:
            print(
                "  %5d %18.10f %18.6f %11.3e %11.3e %11.3e"
                % (iteration, trial_energy, energy_change * HARTREE_TO_KJMOL, step_max, grad_max, grad_rms),
                flush=True,
            )

        y_step = sqrt_m * (trial_q - q)
        y_grad = trial_grad_y - grad_y
        hess = _bfgs_update_hessian(hess, y_step, y_grad)
        q = trial_q
        energy = trial_energy

        if abs(energy_change) <= energy_tol and grad_max <= max_gradient and grad_rms <= rms_gradient and step_max <= max_step:
            message = "Normal-mode constrained scan optimization converged."
            return q, energy, trial_grad_x, True, iteration, message

    grad_x = _gradient(qcinput, q, atoms)
    return q, energy, grad_x, False, int(maxiter), message


def _prepare_2d_axis(
    qcinput,
    atoms,
    q_ref,
    ic,
    *,
    axis_number,
    scan_mode,
    scan_bond=None,
    scan_angle=None,
    scan_dihedral=None,
    scan_normal_mode_index=None,
    scan_normal_mode_vector=None,
    scan_normal_mode_vector_is_mass_weighted=False,
    scan_values=None,
    scan_start=None,
    scan_stop=None,
    scan_nsteps=None,
    scan_unit=None,
    hessian=None,
    hess_file="hessian_scan_2d_normal_modes.hess",
    finite_difference_hessian_step=1.0e-3,
    linear=False,
    print_report=True,
):
    mode = str(scan_mode).lower()
    if mode in {"normal_mode", "normalmode", "mode"}:
        imode = 0 if scan_normal_mode_index is None else int(scan_normal_mode_index)
        mode_cart, mode_mw, frequency, hessian = _prepare_normal_mode_direction(
            qcinput,
            atoms,
            q_ref,
            normal_mode_index=imode,
            normal_mode_vector=scan_normal_mode_vector,
            normal_mode_vector_is_mass_weighted=scan_normal_mode_vector_is_mass_weighted,
            hessian=hessian,
            hess_file=hess_file,
            finite_difference_hessian_step=finite_difference_hessian_step,
            linear=linear,
            print_report=print_report,
        )
        targets, value_unit = _normal_mode_scan_values(
            scan_values,
            scan_start,
            scan_stop,
            scan_nsteps,
            scan_unit,
        )
        label = f"normal_mode[{imode}]"
        return {
            "mode": "normal_mode",
            "reference": (imode,),
            "coord_index": None,
            "targets": targets,
            "value_unit": value_unit,
            "label": label,
            "mode_cart": mode_cart,
            "mode_mw": mode_mw,
            "frequency": frequency,
            "hessian": hessian,
        }

    refs = {
        "bond": scan_bond,
        "angle": scan_angle,
        "dihedral": scan_dihedral,
    }
    if scan_normal_mode_index is not None or scan_normal_mode_vector is not None:
        raise ValueError(f"scan{axis_number}_normal_mode inputs require scan{axis_number}_mode='normal_mode'")
    mode, reference = _prepare_scan_mode(mode, scan_bond, scan_angle, scan_dihedral)
    unit = _default_unit(mode, scan_unit)
    coord_index, reference = _scan_coordinate_index(ic, mode, refs[mode])
    targets, value_unit = _scan_values(
        mode,
        unit,
        scan_values,
        scan_start,
        scan_stop,
        scan_nsteps,
    )
    return {
        "mode": mode,
        "reference": reference,
        "coord_index": coord_index,
        "targets": targets,
        "value_unit": value_unit,
        "label": _scan_reference_label(mode, reference),
        "mode_cart": None,
        "mode_mw": None,
        "frequency": None,
        "hessian": hessian,
    }


def _axis_target_display(axis, target):
    if axis["mode"] == "normal_mode":
        return float(target)
    return _internal_value_to_unit(target, axis["mode"], axis["value_unit"])


def _axis_actual_value(q, q_ref, axis, ic, sqrt_m):
    if axis["mode"] == "normal_mode":
        if axis["value_unit"] == "mass_weighted_bohr":
            return _normal_coordinate_value(q, q_ref, axis["mode_mw"], sqrt_m, axis["value_unit"])
        return float(axis["_current_target"])
    qs = compute_internals(q, ic)
    return _internal_value_to_unit(qs[axis["coord_index"]], axis["mode"], axis["value_unit"])


def _apply_mixed_scan_targets(
    q,
    q_ref,
    ic,
    axes,
    targets,
    sqrt_m,
    *,
    best_fit_iters,
    best_fit_rms_tol,
    use_redundant_internals,
):
    q_new = np.asarray(q, dtype=float).reshape(-1)
    for axis, target in zip(axes, targets):
        axis["_current_target"] = float(target)

    for _iteration in range(3):
        for axis, target in zip(axes, targets):
            if axis["mode"] == "normal_mode":
                q_new = _set_normal_mode_coordinate(
                    q_ref,
                    q_new,
                    axis["mode_cart"],
                    axis["mode_mw"],
                    sqrt_m,
                    float(target),
                    axis["value_unit"],
                )
        internal_axes = [axis for axis in axes if axis["mode"] != "normal_mode"]
        if internal_axes:
            q_new = _set_scan_coordinates(
                q_new,
                ic,
                [axis["coord_index"] for axis in internal_axes],
                [axis["_current_target"] for axis in internal_axes],
                [axis["mode"] for axis in internal_axes],
                best_fit_iters=best_fit_iters,
                best_fit_rms_tol=best_fit_rms_tol,
                use_redundant_internals=use_redundant_internals,
            )
    return q_new


def _relax_scan_point_mixed(
    qcinput,
    atoms,
    q,
    q_ref,
    ic,
    axes,
    targets,
    *,
    use_redundant_internals,
    maxiter,
    max_step,
    max_gradient,
    rms_gradient,
    energy_tol,
    best_fit_iters,
    best_fit_rms_tol,
    print_report=False,
    scan_step=None,
):
    mass = np.asarray(get_mass_vector(atoms), dtype=float)
    sqrt_m = _mass_weight_vector(mass)
    q = _apply_mixed_scan_targets(
        q,
        q_ref,
        ic,
        axes,
        targets,
        sqrt_m,
        best_fit_iters=best_fit_iters,
        best_fit_rms_tol=best_fit_rms_tol,
        use_redundant_internals=use_redundant_internals,
    )
    normal_axes = [axis for axis in axes if axis["mode"] == "normal_mode"]
    hess = np.eye(q.size)
    energy = _energy(qcinput, q, atoms)
    message = "Maximum mixed constrained scan optimization steps reached."

    if print_report:
        target_text = "; ".join(
            f"{axis['label']}={_axis_target_display(axis, target):10.5f} {axis['value_unit']}"
            for axis, target in zip(axes, targets)
        )
        print(f"  relaxing mixed scan step {int(scan_step):4d}: {target_text} E0={energy:18.10f}", flush=True)
        print(
            "  %5s %18s %18s %11s %11s %11s"
            % ("step", "E[Eh]", "dE[kJ/mol]", "max_step", "max_grad", "rms_grad"),
            flush=True,
        )

    for iteration in range(1, int(maxiter) + 1):
        grad_x = _gradient(qcinput, q, atoms)
        grad_y = grad_x / sqrt_m
        for axis in normal_axes:
            grad_y = grad_y - np.dot(axis["mode_mw"], grad_y) * axis["mode_mw"]
        try:
            step_y = np.linalg.solve(hess, -grad_y)
        except np.linalg.LinAlgError:
            step_y = np.linalg.lstsq(hess, -grad_y, rcond=None)[0]
        for axis in normal_axes:
            step_y = step_y - np.dot(axis["mode_mw"], step_y) * axis["mode_mw"]
        step_y = step_limit(step_y, max_component=max_step)
        trial_q, trial_energy, _scale, accepted = _backtrack_constrained_trial(
            q,
            energy,
            step_y / sqrt_m,
            lambda candidate: _apply_mixed_scan_targets(
                candidate,
                q_ref,
                ic,
                axes,
                targets,
                sqrt_m,
                best_fit_iters=best_fit_iters,
                best_fit_rms_tol=best_fit_rms_tol,
                use_redundant_internals=use_redundant_internals,
            ),
            lambda candidate: _energy(qcinput, candidate, atoms),
        )
        if not accepted:
            return q, energy, grad_x, False, iteration - 1, "Mixed constrained scan line search failed."
        trial_grad_x = _gradient(qcinput, trial_q, atoms)
        trial_grad_y = trial_grad_x / sqrt_m
        for axis in normal_axes:
            trial_grad_y = trial_grad_y - np.dot(axis["mode_mw"], trial_grad_y) * axis["mode_mw"]

        energy_change = trial_energy - energy
        grad_max = float(np.max(np.abs(trial_grad_y))) if trial_grad_y.size else 0.0
        grad_rms = float(np.linalg.norm(trial_grad_y) / np.sqrt(max(1, trial_grad_y.size)))
        cart_step = trial_q - q
        step_max = float(np.max(np.abs(cart_step))) if cart_step.size else 0.0

        if print_report:
            print(
                "  %5d %18.10f %18.6f %11.3e %11.3e %11.3e"
                % (iteration, trial_energy, energy_change * HARTREE_TO_KJMOL, step_max, grad_max, grad_rms),
                flush=True,
            )

        y_step = sqrt_m * (trial_q - q)
        y_grad = trial_grad_y - grad_y
        hess = _bfgs_update_hessian(hess, y_step, y_grad)
        q = trial_q
        energy = trial_energy
        if abs(energy_change) <= energy_tol and grad_max <= max_gradient and grad_rms <= rms_gradient and step_max <= max_step:
            message = "Mixed constrained scan optimization converged."
            return q, energy, trial_grad_x, True, iteration, message

    grad_x = _gradient(qcinput, q, atoms)
    return q, energy, grad_x, False, int(maxiter), message


def _target_delta(mode, target, current):
    delta = float(target - current)
    if mode == "dihedral":
        if delta > np.pi:
            delta -= 2.0 * np.pi
        elif delta < -np.pi:
            delta += 2.0 * np.pi
    return delta


def _scan_bpg(q, ic, *, use_redundant_internals=False):
    if use_redundant_internals:
        system = build_redundant_internals(q, coordinates=ic)
        return bpg_from_redundant(system), system
    return bpg_matrix(q, ic), None


def _set_scan_coordinate(
    q,
    ic,
    coord_index,
    target,
    mode,
    *,
    best_fit_iters,
    best_fit_rms_tol,
    use_redundant_internals=False,
):
    qs = compute_internals(q, ic)
    bpg, _redundant_system = _scan_bpg(q, ic, use_redundant_internals=use_redundant_internals)
    dq = np.zeros(ic.nint, dtype=float)
    dq[coord_index] = _target_delta(mode, target, qs[coord_index])
    return best_fit_dq_to_cart(
        q,
        qs,
        dq,
        ic,
        bpg,
        n_iter=best_fit_iters,
        rms_tol=best_fit_rms_tol,
    )


def _set_scan_coordinates(
    q,
    ic,
    coord_indices,
    targets,
    modes,
    *,
    best_fit_iters,
    best_fit_rms_tol,
    use_redundant_internals=False,
):
    qs = compute_internals(q, ic)
    bpg, _redundant_system = _scan_bpg(q, ic, use_redundant_internals=use_redundant_internals)
    dq = np.zeros(ic.nint, dtype=float)
    for coord_index, target, mode in zip(coord_indices, targets, modes):
        dq[int(coord_index)] = _target_delta(mode, target, qs[int(coord_index)])
    return best_fit_dq_to_cart(
        q,
        qs,
        dq,
        ic,
        bpg,
        n_iter=best_fit_iters,
        rms_tol=best_fit_rms_tol,
    )


def _relax_scan_point(
    qcinput,
    atoms,
    q,
    ic,
    coord_index,
    target,
    mode,
    *,
    use_redundant_internals,
    maxiter,
    max_step_internal,
    max_gradient,
    rms_gradient,
    max_step,
    rms_step,
    energy_tol,
    best_fit_iters,
    best_fit_rms_tol,
    print_report=False,
    scan_step=None,
    reference_label=None,
    target_value=None,
    target_unit=None,
    internal_hessian_model="simple",
):
    return _relax_scan_point_multi(
        qcinput,
        atoms,
        q,
        ic,
        [coord_index],
        [target],
        [mode],
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
        scan_step=scan_step,
        reference_label=reference_label,
        target_value=target_value,
        target_unit=target_unit,
        internal_hessian_model=internal_hessian_model,
    )


def _relax_scan_point_multi(
    qcinput,
    atoms,
    q,
    ic,
    coord_indices,
    targets,
    modes,
    *,
    use_redundant_internals,
    maxiter,
    max_step_internal,
    max_gradient,
    rms_gradient,
    max_step,
    rms_step,
    energy_tol,
    best_fit_iters,
    best_fit_rms_tol,
    print_report=False,
    scan_step=None,
    reference_label=None,
    target_value=None,
    target_unit=None,
    internal_hessian_model="simple",
):
    coord_indices = [int(idx) for idx in coord_indices]
    free = np.ones(ic.nint, dtype=bool)
    free[coord_indices] = False
    if use_redundant_internals:
        redundant_system = build_redundant_internals(q, coordinates=ic)
        hess_q = initial_redundant_hessian_from_model(redundant_system, model=internal_hessian_model)
    else:
        hess_q = initial_internal_hessian_from_model(
            ic,
            q_values=compute_internals(q, ic),
            model=internal_hessian_model,
        )
    energy = _energy(qcinput, q, atoms)
    message = "Maximum constrained scan optimization steps reached."
    if print_report:
        label = reference_label if reference_label is not None else "scan"
        if target_value is not None and target_unit is not None:
            if isinstance(target_value, str):
                target_text = target_value if not target_unit else f"{target_value} {target_unit}"
            else:
                target_text = f"{target_value:10.5f} {target_unit}"
            print(
                f"  relaxing scan step {int(scan_step):4d}: {label}={target_text} E0={energy:18.10f}",
                flush=True,
            )
        else:
            print(f"  relaxing scan step {int(scan_step):4d}: E0={energy:18.10f}", flush=True)
        print(
            "  %5s %18s %18s %11s %11s %11s %11s"
            % ("step", "E[Eh]", "dE[kJ/mol]", "max_step", "rms_step", "max_grad", "rms_grad"),
            flush=True,
        )

    for iteration in range(1, int(maxiter) + 1):
        qs = compute_internals(q, ic)
        bpg, _redundant_system = _scan_bpg(q, ic, use_redundant_internals=use_redundant_internals)
        grad_x = _gradient(qcinput, q, atoms)
        grad_q = internal_gradient(bpg, grad_x)
        grad_free = grad_q[free]
        hess_free = hess_q[np.ix_(free, free)]

        try:
            dq_free = np.linalg.solve(hess_free, -grad_free)
        except np.linalg.LinAlgError:
            dq_free = np.linalg.lstsq(hess_free, -grad_free, rcond=None)[0]
        dq_free = step_limit(dq_free, max_component=max_step_internal)

        dq = np.zeros(ic.nint, dtype=float)
        dq[free] = dq_free
        for coord_index, target, mode in zip(coord_indices, targets, modes):
            dq[coord_index] = _target_delta(mode, target, qs[coord_index])

        scale = 1.0
        accepted = False
        while scale >= 1.0e-4:
            trial_dq = dq.copy()
            trial_dq[free] = scale * dq_free
            trial_q = best_fit_dq_to_cart(
                q,
                qs,
                trial_dq,
                ic,
                bpg,
                n_iter=best_fit_iters,
                rms_tol=best_fit_rms_tol,
            )
            trial_energy = _energy(qcinput, trial_q, atoms)
            if np.isfinite(trial_energy) and trial_energy <= energy:
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            return q, energy, grad_x, False, iteration - 1, "Constrained scan line search failed."
        trial_grad_x = _gradient(qcinput, trial_q, atoms)
        trial_bpg, _trial_redundant_system = _scan_bpg(
            trial_q,
            ic,
            use_redundant_internals=use_redundant_internals,
        )
        trial_grad_q = internal_gradient(trial_bpg, trial_grad_x)
        cart_step = trial_q - q
        energy_change = trial_energy - energy

        grad_free_new = trial_grad_q[free]
        grad_max = float(np.max(np.abs(grad_free_new))) if grad_free_new.size else 0.0
        grad_rms = float(np.linalg.norm(grad_free_new) / np.sqrt(max(1, grad_free_new.size)))
        step_max = float(np.max(np.abs(cart_step))) if cart_step.size else 0.0
        step_rms = float(np.linalg.norm(cart_step) / np.sqrt(max(1, cart_step.size)))
        if print_report:
            d_e_relax = energy_change * HARTREE_TO_KJMOL
            print(
                "  %5d %18.10f %18.6f %11.3e %11.3e %11.3e %11.3e"
                % (iteration, trial_energy, d_e_relax, step_max, step_rms, grad_max, grad_rms),
                flush=True,
            )

        accepted_dq = diff_internals(compute_internals(trial_q, ic), qs, ic.ndihedrals, ic.nimpropers)
        y = trial_grad_q - grad_q
        hess_q = _bfgs_update_hessian(hess_q, accepted_dq, y)

        q = trial_q
        energy = trial_energy

        if (
            abs(energy_change) <= energy_tol
            and grad_max <= max_gradient
            and grad_rms <= rms_gradient
            and step_max <= max_step
            and step_rms <= rms_step
        ):
            message = "Constrained scan optimization converged."
            return q, energy, trial_grad_x, True, iteration, message

    grad_x = _gradient(qcinput, q, atoms)
    return q, energy, grad_x, False, int(maxiter), message


def scan_xyz_file(
    qcinput,
    xyz,
    *,
    scan_mode="bond",
    scan_bond=None,
    scan_angle=None,
    scan_dihedral=None,
    unit=None,
    calculate_gradient=False,
    energy_profile_file="external_xyz_scan_energy_profile.dat",
    energy_profile_plot="external_xyz_scan_energy_profile.png",
    print_report=True,
    reporter=None,
    verbosity=1,
    log_file=None,
    log_append=False,
    _reporter_active=False,
):
    """Run single-point energies for each frame in an external XYZ trajectory.

    This is an analysis/evaluation scan, not a geometry scan generator.  The
    input ``xyz`` can be a filename, file-like object, or XYZ trajectory string.
    Each frame is read as-is, one single-point energy calculation is run with
    ``qcinput``, and the requested bond/angle/dihedral value is reported beside
    the energy.  No geometry optimization or constrained relaxation is
    performed.  Set ``calculate_gradient=True`` to also evaluate and store the
    gradient for each frame.
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
            return scan_xyz_file(qcinput, xyz, **options)

    qcinput = validate_qchem_input(qcinput)
    frames, xyz_file = _read_xyz_scan_frames(xyz)
    atoms = list(frames[0][0])
    mode, reference = _prepare_scan_mode(scan_mode, scan_bond, scan_angle, scan_dihedral)
    reference = validate_coordinate_reference(mode, reference, len(atoms), f"scan_{mode}")
    value_unit = _default_unit(mode, unit)
    _unused_values, value_unit = _values_to_internal([1.0], mode, value_unit)
    label = _scan_reference_label(mode, reference)

    if print_report:
        source = xyz_file if xyz_file is not None else "XYZ input"
        print("External XYZ single-point scan")
        print(f"Source: {source}")
        print(f"Frames: {len(frames)}")
        print(f"Reported coordinate: {label}")
        print(f"Coordinate unit: {value_unit}")
        print(f"Gradient calculation: {bool(calculate_gradient)}")

    points = []
    reference_energy = None
    for iframe, (_atoms, q, comment) in enumerate(frames, start=1):
        value = _direct_coordinate_value(q, mode, reference, value_unit)
        energy = _energy(qcinput, q, atoms)
        gradient = _gradient(qcinput, q, atoms) if calculate_gradient else None
        point = XYZScanPoint(
            index=iframe,
            value=value,
            value_unit=value_unit,
            q=q.copy(),
            energy=energy,
            gradient=gradient,
            comment=comment,
        )
        points.append(point)

        if print_report:
            if reference_energy is None:
                reference_energy = energy
            d_e_scan = (energy - reference_energy) * HARTREE_TO_KJMOL / 4.184
            grad_text = ""
            if gradient is not None:
                grad_text = f" GradMax={float(np.max(np.abs(gradient))):10.4e}"
            print(
                f"xyz_scan {iframe:5d} {label}={value:12.6f} {value_unit:8s} "
                f"E={energy:18.10f} dE_first={d_e_scan:12.5f} kcal/mol{grad_text}",
                flush=True,
            )

    if print_report and points:
        first_energy = points[0].energy
        min_energy = min(point.energy for point in points)
        print("")
        print("External XYZ single-point scan summary:")
        if calculate_gradient:
            print(
                "%6s %-14s %18s %18s %18s %12s"
                % (
                    "frame",
                    f"{mode}[{value_unit}]",
                    "E[Eh]",
                    "dE_first[kcal/mol]",
                    "dE_min[kcal/mol]",
                    "GradMax",
                )
            )
        else:
            print(
                "%6s %-14s %18s %18s %18s"
                % ("frame", f"{mode}[{value_unit}]", "E[Eh]", "dE_first[kcal/mol]", "dE_min[kcal/mol]")
            )
        for point in points:
            rel_first = (point.energy - first_energy) * HARTREE_TO_KJMOL / 4.184
            rel_min = (point.energy - min_energy) * HARTREE_TO_KJMOL / 4.184
            if calculate_gradient:
                grad_max = (
                    float(np.max(np.abs(point.gradient)))
                    if point.gradient is not None
                    else np.nan
                )
                print(
                    "%6d %14.6f %18.10f %18.6f %18.6f %12.4e"
                    % (point.index, point.value, point.energy, rel_first, rel_min, grad_max)
                )
            else:
                print(
                    "%6d %14.6f %18.10f %18.6f %18.6f"
                    % (point.index, point.value, point.energy, rel_first, rel_min)
                )

    profile_file, profile_plot = _write_xyz_scan_energy_profile(
        points,
        label=label,
        output_file=energy_profile_file,
        plot_file=energy_profile_plot,
        print_report=print_report,
    )

    return XYZScanResult(
        atoms=atoms,
        mode=mode,
        reference=reference,
        points=points,
        xyz_file=xyz_file,
        energy_profile_file=profile_file,
        energy_profile_plot=profile_plot,
    )


def scan_coordinate(
    qcinput,
    xyz,
    *,
    scan_mode="bond",
    scan_bond=None,
    scan_angle=None,
    scan_dihedral=None,
    values=None,
    start=None,
    stop=None,
    nsteps=None,
    unit=None,
    trajectory_file="coordinate_scan.xyz",
    energy_profile_file="scan_energy_profile.dat",
    energy_profile_plot="scan_energy_profile.png",
    relaxed=False,
    relax_maxiter=50,
    relax_max_step_internal=0.10,
    relax_energy_tol=5.0e-6,
    relax_max_gradient=3.0e-4,
    relax_rms_gradient=1.0e-4,
    relax_max_step=4.0e-3,
    relax_rms_step=2.0e-3,
    use_redundant_internals=False,
    print_report=True,
    best_fit_iters=20,
    best_fit_rms_tol=1.0e-7,
    connectivity_kcn=16.0,
    connectivity_facmin=0.45,
    connect_fragments=True,
    internal_hessian_model="simple",
    reporter=None,
    verbosity=1,
    log_file=None,
    log_append=False,
    _reporter_active=False,
):
    """Scan one primitive internal coordinate and optionally save an XYZ trajectory.

    The scan coordinate is specified in the same ORCA-style form used by the TS
    mode API:

    ``scan_mode="bond", scan_bond=(i, j)``
    ``scan_mode="angle", scan_angle=(i, j, k)``
    ``scan_mode="dihedral", scan_dihedral=(i, j, k, l)``

    Bond target values default to Angstrom. Angle and dihedral target values
    default to degrees. Values are absolute target coordinates, not increments.
    If ``relaxed=True``, each target value is treated as a constrained
    optimization: the selected primitive coordinate is held fixed and all other
    primitive internal coordinates are relaxed.  If ``relaxed=False``, each
    geometry is only back-transformed and evaluated.
    Set ``use_redundant_internals=True`` to use Pulay redundant internals for
    the constrained relaxation.
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
            return scan_coordinate(qcinput, xyz, **options)

    qcinput = validate_qchem_input(qcinput)
    atoms, q = _atoms_q_from_xyz(xyz)
    mode, reference = _prepare_scan_mode(scan_mode, scan_bond, scan_angle, scan_dihedral)
    unit = _default_unit(mode, unit)

    model = default_connectivity_model(
        atoms,
        kcn=connectivity_kcn,
        facmin=connectivity_facmin,
        connect_fragments=connect_fragments,
    )
    if use_redundant_internals:
        redundant_system = build_redundant_internals(q, atoms=atoms, model=model)
        ic = redundant_system.coordinates
    else:
        ic = analyze_structure(q, model=model)
    coord_index, reference = _scan_coordinate_index(ic, mode, reference)
    targets_internal, value_unit = _scan_values(mode, unit, values, start, stop, nsteps)
    reference_label = _scan_reference_label(mode, reference)

    if print_report:
        scan_type = "relaxed constrained" if relaxed else "unrelaxed"
        print(f"Coordinate scan: {reference_label}")
        print(f"Scan type: {scan_type}")
        print(f"Target values unit: {value_unit}")

    points = []
    reference_energy = None
    traj_handle = open(trajectory_file, "w", encoding="utf-8") if trajectory_file else None
    try:
        n_targets = len(targets_internal)
        for istep, target in enumerate(targets_internal, start=1):
            target_value = _internal_value_to_unit(target, mode, value_unit)
            if print_report:
                print(
                    f"scan step {istep:4d}/{n_targets:<4d} setting {reference_label}="
                    f"{target_value:10.5f} {value_unit}",
                    flush=True,
                )
            q = _set_scan_coordinate(
                q,
                ic,
                coord_index,
                target,
                mode,
                best_fit_iters=best_fit_iters,
                best_fit_rms_tol=best_fit_rms_tol,
                use_redundant_internals=use_redundant_internals,
            )
            if relaxed:
                q, energy, gradient, converged, opt_steps, message = _relax_scan_point(
                    qcinput,
                    atoms,
                    q,
                    ic,
                    coord_index,
                    target,
                    mode,
                    use_redundant_internals=use_redundant_internals,
                    maxiter=relax_maxiter,
                    max_step_internal=relax_max_step_internal,
                    max_gradient=relax_max_gradient,
                    rms_gradient=relax_rms_gradient,
                    max_step=relax_max_step,
                    rms_step=relax_rms_step,
                    energy_tol=relax_energy_tol,
                    best_fit_iters=best_fit_iters,
                    best_fit_rms_tol=best_fit_rms_tol,
                    print_report=print_report,
                    scan_step=istep,
                    reference_label=reference_label,
                    target_value=target_value,
                    target_unit=value_unit,
                    internal_hessian_model=internal_hessian_model,
                )
            else:
                energy = _energy(qcinput, q, atoms)
                gradient = _gradient(qcinput, q, atoms)
                converged = True
                opt_steps = 0
                message = "Unrelaxed scan point evaluated."
            value = _internal_value_to_unit(compute_internals(q, ic)[coord_index], mode, value_unit)
            point = ScanPoint(
                index=istep,
                value=value,
                value_unit=value_unit,
                q=q.copy(),
                energy=energy,
                gradient=gradient,
                converged=converged,
                optimization_steps=opt_steps,
                message=message,
            )
            points.append(point)

            if traj_handle is not None:
                _write_scan_frame(traj_handle, atoms, q, point, mode, reference_label)

            if print_report:
                if reference_energy is None:
                    reference_energy = energy
                d_e_scan = (energy - reference_energy) * HARTREE_TO_KJMOL / 4.184
                grad_norm = float(np.linalg.norm(gradient))
                print(
                    f"scan     {istep:4d} {reference_label}={value:10.5f} {value_unit:8s} "
                    f"E={energy:18.10f} dE_scan={d_e_scan:12.5f} kcal/mol "
                    f"|g|={grad_norm:10.4e} opt={int(opt_steps):3d} conv={str(converged):5s}"
                )
    finally:
        if traj_handle is not None:
            traj_handle.close()

    if print_report and points:
        value_header = {
            "bond": f"distance[{points[0].value_unit}]",
            "angle": f"angle[{points[0].value_unit}]",
            "dihedral": f"dihedral[{points[0].value_unit}]",
        }[mode]
        reference_energy = points[0].energy
        print("")
        print("Scan summary:")
        print(
            "%5s %-12s %18s %18s"
            % ("step", value_header, "E[Eh]", "dE_scan[kcal/mol]")
        )
        for point in points:
            d_e_scan = (point.energy - reference_energy) * HARTREE_TO_KJMOL / 4.184
            print(
                "%5d %12.6f %18.10f %18.6f"
                % (point.index, point.value, point.energy, d_e_scan)
            )

    profile_file, profile_plot = _write_scan_energy_profile_1d(
        points,
        label=reference_label,
        output_file=energy_profile_file,
        plot_file=energy_profile_plot,
        print_report=print_report,
    )

    return CoordinateScanResult(
        atoms=atoms,
        mode=mode,
        reference=reference,
        points=points,
        trajectory_file=trajectory_file,
        energy_profile_file=profile_file,
        energy_profile_plot=profile_plot,
    )


def scan_normal_mode(
    qcinput,
    xyz,
    *,
    normal_mode_index=0,
    normal_mode_vector=None,
    normal_mode_vector_is_mass_weighted=False,
    hessian=None,
    hess_file="hessian_normal_mode_scan.hess",
    finite_difference_hessian_step=1.0e-3,
    values=None,
    start=None,
    stop=None,
    nsteps=None,
    unit="mass_weighted_bohr",
    trajectory_file="normal_mode_scan.xyz",
    energy_profile_file="scan_energy_profile.dat",
    energy_profile_plot="scan_energy_profile.png",
    relaxed=False,
    relax_maxiter=50,
    relax_max_step=4.0e-3,
    relax_energy_tol=5.0e-6,
    relax_max_gradient=3.0e-4,
    relax_rms_gradient=1.0e-4,
    linear=False,
    print_report=True,
    reporter=None,
    verbosity=1,
    log_file=None,
    log_append=False,
    _reporter_active=False,
):
    """Scan along one Hessian normal mode or a user-provided Cartesian vector.

    The default scan coordinate is the mass-weighted normal coordinate in
    mass_weighted_bohr.  For quick visual scans, unit="angstrom" or unit="bohr"
    treats each value as the signed largest Cartesian atomic displacement along
    the selected mode.  If relaxed=True, each point is optimized only in the
    subspace orthogonal to the selected normal mode; the scan coordinate is
    projected back to the requested value after every relaxation step.
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
            return scan_normal_mode(qcinput, xyz, **options)

    qcinput = validate_qchem_input(qcinput)
    atoms, q_ref = _atoms_q_from_xyz(xyz)
    mode_cart, mode_mw, frequency, hessian = _prepare_normal_mode_direction(
        qcinput,
        atoms,
        q_ref,
        normal_mode_index=normal_mode_index,
        normal_mode_vector=normal_mode_vector,
        normal_mode_vector_is_mass_weighted=normal_mode_vector_is_mass_weighted,
        hessian=hessian,
        hess_file=hess_file,
        finite_difference_hessian_step=finite_difference_hessian_step,
        linear=linear,
        print_report=print_report,
    )
    mass = np.asarray(get_mass_vector(atoms), dtype=float)
    sqrt_m = _mass_weight_vector(mass)
    targets, value_unit = _normal_mode_scan_values(values, start, stop, nsteps, unit)
    imode = int(normal_mode_index)
    mode_label = f"normal_mode[{imode}]"
    freq_cm = None if frequency is None else frequency * AU_ANGULAR_FREQUENCY_TO_CM1

    if print_report:
        scan_type = "relaxed orthogonal" if relaxed else "unrelaxed"
        print(f"Normal-mode scan: {mode_label}")
        if freq_cm is not None:
            print(f"Mode frequency: {freq_cm:.4f} cm-1")
        else:
            print("Mode source: user-provided vector")
        print(f"Scan type: {scan_type}")
        print(f"Target displacement unit: {value_unit}")

    q = q_ref.copy()
    points = []
    reference_energy = None
    traj_handle = open(trajectory_file, "w", encoding="utf-8") if trajectory_file else None
    try:
        n_targets = len(targets)
        for istep, target in enumerate(targets, start=1):
            if print_report:
                print(
                    f"normal-mode scan step {istep:4d}/{n_targets:<4d} "
                    f"setting {mode_label}={float(target):10.5f} {value_unit}",
                    flush=True,
                )
            q = _set_normal_mode_coordinate(q_ref, q, mode_cart, mode_mw, sqrt_m, float(target), value_unit)
            if relaxed:
                q, energy, gradient, converged, opt_steps, message = _relax_normal_mode_scan_point(
                    qcinput,
                    atoms,
                    q,
                    q_ref,
                    mode_cart,
                    mode_mw,
                    float(target),
                    value_unit,
                    maxiter=relax_maxiter,
                    max_step=relax_max_step,
                    max_gradient=relax_max_gradient,
                    rms_gradient=relax_rms_gradient,
                    energy_tol=relax_energy_tol,
                    print_report=print_report,
                    scan_step=istep,
                )
            else:
                energy = _energy(qcinput, q, atoms)
                gradient = _gradient(qcinput, q, atoms)
                converged = True
                opt_steps = 0
                message = "Unrelaxed normal-mode scan point evaluated."

            actual_value = (
                _normal_coordinate_value(q, q_ref, mode_mw, sqrt_m, value_unit)
                if value_unit == "mass_weighted_bohr"
                else float(target)
            )
            point = ScanPoint(
                index=istep,
                value=actual_value,
                value_unit=value_unit,
                q=q.copy(),
                energy=energy,
                gradient=gradient,
                converged=converged,
                optimization_steps=opt_steps,
                message=message,
            )
            points.append(point)

            if traj_handle is not None:
                _write_normal_mode_scan_frame(traj_handle, atoms, q, point, mode_label)

            if print_report:
                if reference_energy is None:
                    reference_energy = energy
                d_e_scan = (energy - reference_energy) * HARTREE_TO_KJMOL / 4.184
                grad_norm = float(np.linalg.norm(gradient))
                print(
                    f"scan_nm  {istep:4d} {mode_label}={actual_value:10.5f} {value_unit:18s} "
                    f"E={energy:18.10f} dE_scan={d_e_scan:12.5f} kcal/mol "
                    f"|g|={grad_norm:10.4e} opt={int(opt_steps):3d} conv={str(converged):5s}"
                )
    finally:
        if traj_handle is not None:
            traj_handle.close()

    if print_report and points:
        reference_energy = points[0].energy
        print("")
        print("Normal-mode scan summary:")
        print(
            "%5s %-22s %18s %18s"
            % ("step", f"displacement[{points[0].value_unit}]", "E[Eh]", "dE_scan[kcal/mol]")
        )
        for point in points:
            d_e_scan = (point.energy - reference_energy) * HARTREE_TO_KJMOL / 4.184
            print(
                "%5d %22.8f %18.10f %18.6f"
                % (point.index, point.value, point.energy, d_e_scan)
            )

    profile_file, profile_plot = _write_scan_energy_profile_1d(
        points,
        label=mode_label,
        output_file=energy_profile_file,
        plot_file=energy_profile_plot,
        print_report=print_report,
    )

    return NormalModeScanResult(
        atoms=atoms,
        mode_index=imode,
        frequency=frequency,
        frequency_unit="hartree_angular_frequency",
        points=points,
        trajectory_file=trajectory_file,
        hessian_file=hess_file if hessian is not None else None,
        energy_profile_file=profile_file,
        energy_profile_plot=profile_plot,
    )


def scan_coordinate_2d(
    qcinput,
    xyz,
    *,
    scan1_mode="bond",
    scan1_bond=None,
    scan1_angle=None,
    scan1_dihedral=None,
    scan1_normal_mode_index=None,
    scan1_normal_mode_vector=None,
    scan1_normal_mode_vector_is_mass_weighted=False,
    scan1_values=None,
    scan1_start=None,
    scan1_stop=None,
    scan1_nsteps=None,
    scan1_unit=None,
    scan2_mode="bond",
    scan2_bond=None,
    scan2_angle=None,
    scan2_dihedral=None,
    scan2_normal_mode_index=None,
    scan2_normal_mode_vector=None,
    scan2_normal_mode_vector_is_mass_weighted=False,
    scan2_values=None,
    scan2_start=None,
    scan2_stop=None,
    scan2_nsteps=None,
    scan2_unit=None,
    trajectory_file="coordinate_scan_2d.xyz",
    energy_profile_file="scan_energy_profile.dat",
    energy_profile_plot="scan_energy_profile.png",
    relaxed=True,
    relax_maxiter=50,
    relax_max_step_internal=0.10,
    relax_energy_tol=5.0e-6,
    relax_max_gradient=3.0e-4,
    relax_rms_gradient=1.0e-4,
    relax_max_step=4.0e-3,
    relax_rms_step=2.0e-3,
    use_redundant_internals=False,
    print_report=True,
    best_fit_iters=20,
    best_fit_rms_tol=1.0e-7,
    connectivity_kcn=16.0,
    connectivity_facmin=0.45,
    connect_fragments=True,
    internal_hessian_model="simple",
    hessian=None,
    hess_file="hessian_scan_2d_normal_modes.hess",
    finite_difference_hessian_step=1.0e-3,
    linear=False,
    reporter=None,
    verbosity=1,
    log_file=None,
    log_append=False,
    _reporter_active=False,
):
    """Scan two coordinates on a grid.

    Each scan axis can be a bond, angle, dihedral, or normal_mode in any
    combination.  For example:

    scan1_mode="bond", scan1_bond=(0, 5)
    scan2_mode="normal_mode", scan2_normal_mode_index=3

    If relaxed=True and both axes are internal coordinates, the original
    internal-coordinate constrained optimizer is used.  If either axis is a
    normal mode, relaxation is done in mass-weighted Cartesian coordinates with
    the requested scan coordinates projected back after each step.
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
            return scan_coordinate_2d(qcinput, xyz, **options)

    qcinput = validate_qchem_input(qcinput)
    atoms, q_ref = _atoms_q_from_xyz(xyz)
    q = q_ref.copy()

    model = default_connectivity_model(
        atoms,
        kcn=connectivity_kcn,
        facmin=connectivity_facmin,
        connect_fragments=connect_fragments,
    )
    if use_redundant_internals:
        redundant_system = build_redundant_internals(q_ref, atoms=atoms, model=model)
        ic = redundant_system.coordinates
    else:
        ic = analyze_structure(q_ref, model=model)

    axis1 = _prepare_2d_axis(
        qcinput,
        atoms,
        q_ref,
        ic,
        axis_number=1,
        scan_mode=scan1_mode,
        scan_bond=scan1_bond,
        scan_angle=scan1_angle,
        scan_dihedral=scan1_dihedral,
        scan_normal_mode_index=scan1_normal_mode_index,
        scan_normal_mode_vector=scan1_normal_mode_vector,
        scan_normal_mode_vector_is_mass_weighted=scan1_normal_mode_vector_is_mass_weighted,
        scan_values=scan1_values,
        scan_start=scan1_start,
        scan_stop=scan1_stop,
        scan_nsteps=scan1_nsteps,
        scan_unit=scan1_unit,
        hessian=hessian,
        hess_file=hess_file,
        finite_difference_hessian_step=finite_difference_hessian_step,
        linear=linear,
        print_report=print_report,
    )
    hessian = axis1.get("hessian", hessian)
    axis2 = _prepare_2d_axis(
        qcinput,
        atoms,
        q_ref,
        ic,
        axis_number=2,
        scan_mode=scan2_mode,
        scan_bond=scan2_bond,
        scan_angle=scan2_angle,
        scan_dihedral=scan2_dihedral,
        scan_normal_mode_index=scan2_normal_mode_index,
        scan_normal_mode_vector=scan2_normal_mode_vector,
        scan_normal_mode_vector_is_mass_weighted=scan2_normal_mode_vector_is_mass_weighted,
        scan_values=scan2_values,
        scan_start=scan2_start,
        scan_stop=scan2_stop,
        scan_nsteps=scan2_nsteps,
        scan_unit=scan2_unit,
        hessian=hessian,
        hess_file=hess_file,
        finite_difference_hessian_step=finite_difference_hessian_step,
        linear=linear,
        print_report=print_report,
    )
    axes = [axis1, axis2]
    mode1, mode2 = axis1["mode"], axis2["mode"]
    reference1, reference2 = axis1["reference"], axis2["reference"]
    coord_index1, coord_index2 = axis1["coord_index"], axis2["coord_index"]
    if mode1 != "normal_mode" and mode2 != "normal_mode" and coord_index1 == coord_index2:
        raise ValueError("The two scan coordinates resolve to the same primitive coordinate")
    targets1_internal, targets2_internal = axis1["targets"], axis2["targets"]
    value1_unit, value2_unit = axis1["value_unit"], axis2["value_unit"]
    reference_label1, reference_label2 = axis1["label"], axis2["label"]
    has_normal_axis = mode1 == "normal_mode" or mode2 == "normal_mode"
    sqrt_m = _mass_weight_vector(np.asarray(get_mass_vector(atoms), dtype=float))

    if print_report:
        scan_type = "relaxed constrained" if relaxed else "unrelaxed"
        print(f"2D coordinate scan: {reference_label1} x {reference_label2}")
        print(f"Scan type: {scan_type}")
        print(f"Axis 1 unit: {value1_unit}")
        print(f"Axis 2 unit: {value2_unit}")

    points = []
    reference_energy = None
    traj_handle = open(trajectory_file, "w", encoding="utf-8") if trajectory_file else None
    try:
        n1 = len(targets1_internal)
        n2 = len(targets2_internal)
        for i1, target1 in enumerate(targets1_internal, start=1):
            value1_target = _axis_target_display(axis1, target1)
            for i2, target2 in enumerate(targets2_internal, start=1):
                value2_target = _axis_target_display(axis2, target2)
                if print_report:
                    print(
                        f"scan grid {i1:4d}/{n1:<4d} {i2:4d}/{n2:<4d} setting "
                        f"{reference_label1}={value1_target:10.5f} {value1_unit}, "
                        f"{reference_label2}={value2_target:10.5f} {value2_unit}",
                        flush=True,
                    )

                if has_normal_axis:
                    q = _apply_mixed_scan_targets(
                        q,
                        q_ref,
                        ic,
                        axes,
                        [target1, target2],
                        sqrt_m,
                        best_fit_iters=best_fit_iters,
                        best_fit_rms_tol=best_fit_rms_tol,
                        use_redundant_internals=use_redundant_internals,
                    )
                else:
                    q = _set_scan_coordinates(
                        q,
                        ic,
                        [coord_index1, coord_index2],
                        [target1, target2],
                        [mode1, mode2],
                        best_fit_iters=best_fit_iters,
                        best_fit_rms_tol=best_fit_rms_tol,
                        use_redundant_internals=use_redundant_internals,
                    )

                if relaxed:
                    if has_normal_axis:
                        q, energy, gradient, converged, opt_steps, message = _relax_scan_point_mixed(
                            qcinput,
                            atoms,
                            q,
                            q_ref,
                            ic,
                            axes,
                            [target1, target2],
                            use_redundant_internals=use_redundant_internals,
                            maxiter=relax_maxiter,
                            max_step=relax_max_step,
                            max_gradient=relax_max_gradient,
                            rms_gradient=relax_rms_gradient,
                            energy_tol=relax_energy_tol,
                            best_fit_iters=best_fit_iters,
                            best_fit_rms_tol=best_fit_rms_tol,
                            print_report=print_report,
                            scan_step=(i1 - 1) * n2 + i2,
                        )
                    else:
                        label = f"{reference_label1}, {reference_label2}"
                        target_text = (
                            f"{value1_target:10.5f} {value1_unit}; "
                            f"{value2_target:10.5f} {value2_unit}"
                        )
                        q, energy, gradient, converged, opt_steps, message = _relax_scan_point_multi(
                            qcinput,
                            atoms,
                            q,
                            ic,
                            [coord_index1, coord_index2],
                            [target1, target2],
                            [mode1, mode2],
                            use_redundant_internals=use_redundant_internals,
                            maxiter=relax_maxiter,
                            max_step_internal=relax_max_step_internal,
                            max_gradient=relax_max_gradient,
                            rms_gradient=relax_rms_gradient,
                            max_step=relax_max_step,
                            rms_step=relax_rms_step,
                            energy_tol=relax_energy_tol,
                            best_fit_iters=best_fit_iters,
                            best_fit_rms_tol=best_fit_rms_tol,
                            print_report=print_report,
                            scan_step=(i1 - 1) * n2 + i2,
                            reference_label=label,
                            target_value=target_text,
                            target_unit="",
                            internal_hessian_model=internal_hessian_model,
                        )
                else:
                    energy = _energy(qcinput, q, atoms)
                    gradient = _gradient(qcinput, q, atoms)
                    converged = True
                    opt_steps = 0
                    message = "Unrelaxed 2D scan point evaluated."

                value1 = _axis_actual_value(q, q_ref, axis1, ic, sqrt_m)
                value2 = _axis_actual_value(q, q_ref, axis2, ic, sqrt_m)
                point = ScanGridPoint(
                    index1=i1,
                    index2=i2,
                    value1=value1,
                    value2=value2,
                    value1_unit=value1_unit,
                    value2_unit=value2_unit,
                    q=q.copy(),
                    energy=energy,
                    gradient=gradient,
                    converged=converged,
                    optimization_steps=opt_steps,
                    message=message,
                )
                points.append(point)

                if traj_handle is not None:
                    _write_scan2d_frame(traj_handle, atoms, q, point, reference_label1, reference_label2)

                if print_report:
                    if reference_energy is None:
                        reference_energy = energy
                    d_e_scan = (energy - reference_energy) * HARTREE_TO_KJMOL / 4.184
                    grad_norm = float(np.linalg.norm(gradient))
                    print(
                        f"scan2d {i1:4d} {i2:4d} "
                        f"{reference_label1}={value1:10.5f} {value1_unit:8s} "
                        f"{reference_label2}={value2:10.5f} {value2_unit:8s} "
                        f"E={energy:18.10f} dE_scan={d_e_scan:12.5f} kcal/mol "
                        f"|g|={grad_norm:10.4e} opt={int(opt_steps):3d} conv={str(converged):5s}"
                    )
    finally:
        if traj_handle is not None:
            traj_handle.close()

    if print_report and points:
        header1 = {
            "bond": f"distance1[{points[0].value1_unit}]",
            "angle": f"angle1[{points[0].value1_unit}]",
            "dihedral": f"dihedral1[{points[0].value1_unit}]",
            "normal_mode": f"mode1[{points[0].value1_unit}]",
        }[mode1]
        header2 = {
            "bond": f"distance2[{points[0].value2_unit}]",
            "angle": f"angle2[{points[0].value2_unit}]",
            "dihedral": f"dihedral2[{points[0].value2_unit}]",
            "normal_mode": f"mode2[{points[0].value2_unit}]",
        }[mode2]
        reference_energy = points[0].energy
        print("")
        print("2D scan summary:")
        print(
            "%5s %5s %-14s %-14s %18s %18s"
            % ("i", "j", header1, header2, "E[Eh]", "dE_scan[kcal/mol]")
        )
        for point in points:
            d_e_scan = (point.energy - reference_energy) * HARTREE_TO_KJMOL / 4.184
            print(
                "%5d %5d %14.6f %14.6f %18.10f %18.6f"
                % (point.index1, point.index2, point.value1, point.value2, point.energy, d_e_scan)
            )

    profile_file, profile_plot = _write_scan_energy_profile_2d(
        points,
        label1=reference_label1,
        label2=reference_label2,
        output_file=energy_profile_file,
        plot_file=energy_profile_plot,
        print_report=print_report,
    )

    return CoordinateScan2DResult(
        atoms=atoms,
        mode1=mode1,
        reference1=reference1,
        mode2=mode2,
        reference2=reference2,
        points=points,
        trajectory_file=trajectory_file,
        energy_profile_file=profile_file,
        energy_profile_plot=profile_plot,
    )


def scan_bond(qcinput, xyz, bond, **kwargs):
    return scan_coordinate(qcinput, xyz, scan_mode="bond", scan_bond=bond, **kwargs)


def scan_angle(qcinput, xyz, angle, **kwargs):
    return scan_coordinate(qcinput, xyz, scan_mode="angle", scan_angle=angle, **kwargs)


def scan_dihedral(qcinput, xyz, dihedral, **kwargs):
    return scan_coordinate(qcinput, xyz, scan_mode="dihedral", scan_dihedral=dihedral, **kwargs)
