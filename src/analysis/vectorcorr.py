import math
from pathlib import Path

import numpy as np

from analysis.collision_vectors import collision_vectors, final_product_collision_vectors
from analysis.fragment_energy import (
    product_energy_partitions,
    product_potential_energies,
    product_total_energy,
    reactant_total_energy,
)
from utils.constants import BOHR_TO_ANGSTROM, HARTREE_TO_KJMOL


ANGLE_LABELS = (
    "theta_vrel_ini_vrel_fin",
    "theta_Lorb_ini_Lorb_fin",
    "theta_vrel_ini_Lorb_ini",
    "theta_vrel_ini_Lorb_fin",
    "theta_vrel_fin_Lorb_ini",
    "theta_vrel_fin_Lorb_fin",
    "theta_Jrot_ini_A_Jrot_fin_A",
    "theta_vrel_ini_Jrot_ini_A",
    "theta_vrel_ini_Jrot_fin_A",
    "theta_vrel_fin_Jrot_ini_A",
    "theta_vrel_fin_Jrot_fin_A",
    "theta_Jrot_ini_B_Jrot_fin_B",
    "theta_vrel_ini_Jrot_ini_B",
    "theta_vrel_ini_Jrot_fin_B",
    "theta_vrel_fin_Jrot_ini_B",
    "theta_vrel_fin_Jrot_fin_B",
    "theta_Jrot_ini_A_Jrot_ini_B",
    "theta_Jrot_fin_A_Jrot_fin_B",
    "theta_Jrot_ini_A_Jrot_fin_B",
    "theta_Jrot_fin_A_Jrot_ini_B",
    "theta_Jrot_ini_A_Lorb_ini",
    "theta_Jrot_fin_A_Lorb_ini",
    "theta_Jrot_ini_A_Lorb_fin",
    "theta_Jrot_fin_A_Lorb_fin",
    "theta_Jrot_ini_B_Lorb_ini",
    "theta_Jrot_fin_B_Lorb_ini",
    "theta_Jrot_ini_B_Lorb_fin",
    "theta_Jrot_fin_B_Lorb_fin",
)


def _as_vector(vector):
    if vector is None:
        return None
    arr = np.asarray(vector, dtype=float).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"Expected a 3-vector, got shape {arr.shape}")
    return arr


def get_angle(vec_A, vec_B, tol=1e-5):
    vec_A = _as_vector(vec_A)
    vec_B = _as_vector(vec_B)
    if vec_A is None or vec_B is None:
        return None

    vec_A_mag = np.linalg.norm(vec_A)
    vec_B_mag = np.linalg.norm(vec_B)
    if vec_A_mag <= tol or vec_B_mag <= tol:
        return None

    arg = np.dot(vec_A, vec_B) / (vec_A_mag * vec_B_mag)
    arg = min(1.0, max(-1.0, float(arg)))
    return math.degrees(math.acos(arg))


def scattering_angle_dict(vrel_ini, vrel_fin, Lorb_ini, Lorb_fin,
                          Jrot_ini_A, Jrot_fin_A, Jrot_ini_B, Jrot_fin_B):
    vectors = (
        (vrel_ini, vrel_fin),
        (Lorb_ini, Lorb_fin),
        (vrel_ini, Lorb_ini),
        (vrel_ini, Lorb_fin),
        (vrel_fin, Lorb_ini),
        (vrel_fin, Lorb_fin),
        (Jrot_ini_A, Jrot_fin_A),
        (vrel_ini, Jrot_ini_A),
        (vrel_ini, Jrot_fin_A),
        (vrel_fin, Jrot_ini_A),
        (vrel_fin, Jrot_fin_A),
        (Jrot_ini_B, Jrot_fin_B),
        (vrel_ini, Jrot_ini_B),
        (vrel_ini, Jrot_fin_B),
        (vrel_fin, Jrot_ini_B),
        (vrel_fin, Jrot_fin_B),
        (Jrot_ini_A, Jrot_ini_B),
        (Jrot_fin_A, Jrot_fin_B),
        (Jrot_ini_A, Jrot_fin_B),
        (Jrot_fin_A, Jrot_ini_B),
        (Jrot_ini_A, Lorb_ini),
        (Jrot_fin_A, Lorb_ini),
        (Jrot_ini_A, Lorb_fin),
        (Jrot_fin_A, Lorb_fin),
        (Jrot_ini_B, Lorb_ini),
        (Jrot_fin_B, Lorb_ini),
        (Jrot_ini_B, Lorb_fin),
        (Jrot_fin_B, Lorb_fin),
    )
    return {label: get_angle(vec_a, vec_b) for label, (vec_a, vec_b) in zip(ANGLE_LABELS, vectors)}


def scattering_angles(vrel_ini, vrel_fin, Lorb_ini, Lorb_fin,
                      Jrot_ini_A, Jrot_fin_A, Jrot_ini_B, Jrot_fin_B):
    angles = scattering_angle_dict(
        vrel_ini, vrel_fin, Lorb_ini, Lorb_fin,
        Jrot_ini_A, Jrot_fin_A, Jrot_ini_B, Jrot_fin_B,
    )
    return [angles[label] for label in ANGLE_LABELS]


def normal_vector_of_collision_plane(vini, vfin, tol=1e-12):
    vini = _as_vector(vini)
    vfin = _as_vector(vfin)
    if vini is None or vfin is None:
        return None

    normal = np.cross(vini, vfin)
    norm = np.linalg.norm(normal)
    if norm <= tol:
        return None
    return normal / norm


def calculate_dihedral_angle(vini, vfin, jrot, tol=1e-12):
    vini = _as_vector(vini)
    vfin = _as_vector(vfin)
    jrot = _as_vector(jrot)
    if vini is None or vfin is None or jrot is None:
        return None

    k_norm = np.linalg.norm(vini)
    if k_norm <= tol:
        return None
    k_unit = vini / k_norm

    n_scatter = normal_vector_of_collision_plane(vini, vfin, tol=tol)
    n_jrot = normal_vector_of_collision_plane(vini, jrot, tol=tol)
    if n_scatter is None or n_jrot is None:
        return None

    x = np.dot(n_scatter, n_jrot)
    y = np.dot(k_unit, np.cross(n_scatter, n_jrot))
    return math.degrees(math.atan2(y, x)) % 360.0


def update_collision_vector_state(collision, stage, fragments=None):
    vectors = collision_vectors(collision, fragments=fragments)
    if stage == "initial":
        collision.vrel_ini = vectors["vrel"]
        collision.Lorb_ini = vectors["orbital_angular_momentum"]
        collision.Jrot_ini_A = vectors["fragment_angular_momentum_A"]
        collision.Jrot_ini_B = vectors["fragment_angular_momentum_B"]
        collision.Erelsq_ini = vectors["relative_energy"]
    elif stage == "final":
        collision.vrel_fin = vectors["vrel"]
        collision.vrelfin_sq = vectors["vrel_sq"]
        collision.Lorb_fin = vectors["orbital_angular_momentum"]
        collision.Jrot_fin_A = vectors["fragment_angular_momentum_A"]
        collision.Jrot_fin_B = vectors["fragment_angular_momentum_B"]
        collision.Erelsq_fin = vectors["relative_energy"]
        collision.bimp_fin = vectors["impact_parameter"]
    else:
        raise ValueError("stage must be 'initial' or 'final'")
    return vectors


def analyze_collision(collision, output_file=None, channel=None, formula=None,
                      step=None, time_fs=None, bond_th_HX=1.5, bond_th_XX=2.0,
                      equilibrium_geometries=None, channel_state=None,
                      isolation_distance=100.0, energy_evaluator=None):
    fragments, final_vectors = final_product_collision_vectors(
        collision,
        bond_th_HX=bond_th_HX,
        bond_th_XX=bond_th_XX,
    )

    if collision.vrel_ini is None or collision.Lorb_ini is None:
        update_collision_vector_state(collision, "initial")
    update_collision_vector_state(collision, "final", fragments=fragments)

    angles = scattering_angle_dict(
        collision.vrel_ini,
        collision.vrel_fin,
        collision.Lorb_ini,
        collision.Lorb_fin,
        collision.Jrot_ini_A,
        collision.Jrot_fin_A,
        collision.Jrot_ini_B,
        collision.Jrot_fin_B,
    )
    dihedrals = {
        "dihedral_vrel_Jrot_A": calculate_dihedral_angle(collision.vrel_ini, collision.vrel_fin, collision.Jrot_fin_A),
        "dihedral_vrel_Jrot_B": calculate_dihedral_angle(collision.vrel_ini, collision.vrel_fin, collision.Jrot_fin_B),
    }

    potential_records = None
    initial_total = None
    final_total = None
    energy_relative = None
    if channel_state is not None:
        potential_records = product_potential_energies(
            collision,
            fragments,
            channel_state,
            equilibrium_geometries=equilibrium_geometries,
            isolation_distance=isolation_distance,
            energy_evaluator=energy_evaluator,
        )
        initial_total = reactant_total_energy(
            collision,
            channel_state,
            isolation_distance=isolation_distance,
            energy_evaluator=energy_evaluator,
        )
        final_total = product_total_energy(
            collision,
            fragments,
            potential_records,
            isolation_distance=isolation_distance,
            energy_evaluator=energy_evaluator,
        )
        energy_relative = final_total["total_energy"] - initial_total["total_energy"]

    fragment_energies = product_energy_partitions(
        collision.q,
        collision.p,
        collision.mass,
        fragments,
        equilibrium_geometries=equilibrium_geometries,
        potential_energies=potential_records,
    )

    result = {
        "channel": channel,
        "formula": formula,
        "step": step,
        "time_fs": time_fs,
        "product_fragments": fragments,
        "redmass": final_vectors["redmass"],
        "Erelsq_ini_hartree": collision.Erelsq_ini,
        "Erelsq_fin_hartree": collision.Erelsq_fin,
        "Erelsq_ini_kjmol": None if collision.Erelsq_ini is None else collision.Erelsq_ini * HARTREE_TO_KJMOL,
        "Erelsq_fin_kjmol": None if collision.Erelsq_fin is None else collision.Erelsq_fin * HARTREE_TO_KJMOL,
        "initial_total_energy_hartree": None if initial_total is None else initial_total["total_energy"],
        "final_total_energy_hartree": None if final_total is None else final_total["total_energy"],
        "energy_relative_to_reactants_hartree": energy_relative,
        "energy_relative_to_reactants_kjmol": None if energy_relative is None else energy_relative * HARTREE_TO_KJMOL,
        "bimp_ini_bohr": collision.bimp,
        "bimp_ini_angstrom": None if collision.bimp is None else collision.bimp * BOHR_TO_ANGSTROM,
        "bimp_fin_bohr": collision.bimp_fin,
        "bimp_fin_angstrom": None if collision.bimp_fin is None else collision.bimp_fin * BOHR_TO_ANGSTROM,
        "angles_deg": angles,
        "dihedrals_deg": dihedrals,
        "fragment_energies": fragment_energies,
    }

    if output_file is not None:
        write_collision_analysis(output_file, result, collision)
    return result


def _format_value(value):
    if value is None:
        return "nan"
    if isinstance(value, str):
        return value
    return f"{float(value):.12e}"


def _format_vector(vector):
    if vector is None:
        return "nan nan nan"
    vector = np.asarray(vector, dtype=float).reshape(3)
    return " ".join(f"{value:.12e}" for value in vector)


def write_collision_analysis(output_file, result, collision):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Smite post-collision vector-correlation analysis\n")
        handle.write("# angles are in degrees; vectors are in internal atomic units\n")
        for key in (
            "channel",
            "formula",
            "step",
            "time_fs",
            "redmass",
            "Erelsq_ini_hartree",
            "Erelsq_fin_hartree",
            "Erelsq_ini_kjmol",
            "Erelsq_fin_kjmol",
            "initial_total_energy_hartree",
            "final_total_energy_hartree",
            "energy_relative_to_reactants_hartree",
            "energy_relative_to_reactants_kjmol",
            "bimp_ini_bohr",
            "bimp_ini_angstrom",
            "bimp_fin_bohr",
            "bimp_fin_angstrom",
        ):
            handle.write(f"{key} {_format_value(result.get(key))}\n")

        handle.write("# product_fragment index formula atom_indices\n")
        for idx, fragment in enumerate(result["product_fragments"]):
            atom_indices = ",".join(str(atom_idx) for atom_idx in fragment["indices"])
            handle.write(f"product_fragment {idx} {fragment['formula']} {atom_indices}\n")
        handle.write(
            "# fragment_energy index formula rotational_reference internal_kinetic_Eh "
            "potential_Eh reference_potential_Eh potential_relative_Eh rotational_Eh vibrational_Eh\n"
        )
        for idx, energy in enumerate(result["fragment_energies"]):
            handle.write(
                f"fragment_energy {idx} {energy['formula']} {energy['rotational_reference']} "
                f"{_format_value(energy['internal_kinetic_energy'])} "
                f"{_format_value(energy['potential_energy'])} "
                f"{_format_value(energy['reference_potential_energy'])} "
                f"{_format_value(energy['potential_energy_relative_to_reference'])} "
                f"{_format_value(energy['rotational_energy'])} "
                f"{_format_value(energy['vibrational_energy'])}\n"
            )

        handle.write(f"vrel_ini {_format_vector(collision.vrel_ini)}\n")
        handle.write(f"vrel_fin {_format_vector(collision.vrel_fin)}\n")
        handle.write(f"Lorb_ini {_format_vector(collision.Lorb_ini)}\n")
        handle.write(f"Lorb_fin {_format_vector(collision.Lorb_fin)}\n")
        handle.write(f"Jrot_ini_A {_format_vector(collision.Jrot_ini_A)}\n")
        handle.write(f"Jrot_fin_A {_format_vector(collision.Jrot_fin_A)}\n")
        handle.write(f"Jrot_ini_B {_format_vector(collision.Jrot_ini_B)}\n")
        handle.write(f"Jrot_fin_B {_format_vector(collision.Jrot_fin_B)}\n")

        handle.write("# angle_name angle_deg\n")
        for label in ANGLE_LABELS:
            handle.write(f"{label} {_format_value(result['angles_deg'][label])}\n")
        for label, value in result["dihedrals_deg"].items():
            handle.write(f"{label} {_format_value(value)}\n")
