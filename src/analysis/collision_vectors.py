import math

import numpy as np

from sampling.polyrotation import angular_momentum
from utils.cenmass import cenmass
from utils.constants import ANGSTROM_TO_BOHR
from utils.graph_cluster import create_adjacency_list, create_chemical_formula


def connected_components(adj_list):
    visited = set()
    components = []

    for start_node in sorted(adj_list):
        if start_node in visited:
            continue
        stack = [start_node]
        component = []
        visited.add(start_node)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def product_fragments(q, atoms, bond_th_HX=1.5, bond_th_XX=2.0):
    adj_list = create_adjacency_list(
        atoms,
        np.asarray(q, dtype=float),
        bond_th_HX * ANGSTROM_TO_BOHR,
        bond_th_XX * ANGSTROM_TO_BOHR,
    )
    fragments = []
    for indices in connected_components(adj_list):
        fragment_atoms = [atoms[i] for i in indices]
        fragments.append(
            {
                "indices": indices,
                "atoms": fragment_atoms,
                "formula": create_chemical_formula(fragment_atoms),
            }
        )
    return fragments


def validate_bimolecular_products(q, atoms, bond_th_HX=1.5, bond_th_XX=2.0):
    fragments = product_fragments(q, atoms, bond_th_HX=bond_th_HX, bond_th_XX=bond_th_XX)
    if len(fragments) != 2:
        formulas = " + ".join(fragment["formula"] for fragment in fragments) or "none"
        raise ValueError(
            "Post-collision vector analysis requires exactly two graph-theoretical "
            f"product fragments; found {len(fragments)} ({formulas})."
        )
    return fragments


def center_of_mass(q, mass):
    q_flat = np.asarray(q, dtype=float).reshape(-1)
    mass = np.asarray(mass, dtype=float)
    q_internal, _ = cenmass(q_flat.copy(), np.zeros_like(q_flat), mass)
    return q_flat.reshape((-1, 3))[0] - q_internal.reshape((-1, 3))[0]


def center_of_mass_velocity(p, mass):
    p = np.asarray(p, dtype=float).reshape((-1, 3))
    mass = np.asarray(mass, dtype=float)
    return np.sum(p, axis=0) / np.sum(mass)


def fragment_angular_momentum(q, p, mass):
    q_internal, p_internal = cenmass(
        np.asarray(q, dtype=float).reshape(-1).copy(),
        np.asarray(p, dtype=float).reshape(-1).copy(),
        np.asarray(mass, dtype=float),
    )
    return np.asarray(angular_momentum(q_internal, p_internal), dtype=float)


def two_fragment_vectors(q, p, mass, fragments):
    if len(fragments) != 2:
        raise ValueError("two_fragment_vectors requires exactly two fragments")

    q_xyz = np.asarray(q, dtype=float).reshape((-1, 3))
    p_xyz = np.asarray(p, dtype=float).reshape((-1, 3))
    mass = np.asarray(mass, dtype=float)

    idx_a = np.asarray(fragments[0]["indices"], dtype=int)
    idx_b = np.asarray(fragments[1]["indices"], dtype=int)

    mass_a = mass[idx_a]
    mass_b = mass[idx_b]
    q_a = q_xyz[idx_a]
    q_b = q_xyz[idx_b]
    p_a = p_xyz[idx_a]
    p_b = p_xyz[idx_b]

    total_mass_a = float(np.sum(mass_a))
    total_mass_b = float(np.sum(mass_b))
    redmass = total_mass_a * total_mass_b / (total_mass_a + total_mass_b)

    com_a = center_of_mass(q_a, mass_a)
    com_b = center_of_mass(q_b, mass_b)
    vel_a = center_of_mass_velocity(p_a, mass_a)
    vel_b = center_of_mass_velocity(p_b, mass_b)

    qrel = com_b - com_a
    vrel = vel_b - vel_a
    orbital_angular_momentum = redmass * np.cross(qrel, vrel)
    fragment_angular_momentum_a = fragment_angular_momentum(q_a, p_a, mass_a)
    fragment_angular_momentum_b = fragment_angular_momentum(q_b, p_b, mass_b)

    vrel_sq = float(np.dot(vrel, vrel))
    vrel_norm = math.sqrt(vrel_sq)
    Lorb_norm = float(np.linalg.norm(orbital_angular_momentum))
    impact_parameter = None
    if redmass > 0.0 and vrel_norm > 0.0:
        impact_parameter = Lorb_norm / (redmass * vrel_norm)

    return {
        "qrel": qrel,
        "vrel": vrel,
        "orbital_angular_momentum": orbital_angular_momentum,
        "fragment_angular_momentum_A": fragment_angular_momentum_a,
        "fragment_angular_momentum_B": fragment_angular_momentum_b,
        "redmass": redmass,
        "vrel_sq": vrel_sq,
        "relative_energy": 0.5 * redmass * vrel_sq,
        "impact_parameter": impact_parameter,
    }


def collision_vectors(collision, fragments=None, bond_th_HX=1.5, bond_th_XX=2.0):
    if fragments is None:
        natom_A = collision.fragment_A.natom
        fragments = (
            {
                "indices": list(range(natom_A)),
                "atoms": collision.atoms[:natom_A],
                "formula": create_chemical_formula(collision.atoms[:natom_A]),
            },
            {
                "indices": list(range(natom_A, collision.natom)),
                "atoms": collision.atoms[natom_A:],
                "formula": create_chemical_formula(collision.atoms[natom_A:]),
            },
        )
    return two_fragment_vectors(collision.q, collision.p, collision.mass, fragments)


def final_product_collision_vectors(collision, bond_th_HX=1.5, bond_th_XX=2.0):
    fragments = validate_bimolecular_products(
        collision.q,
        collision.atoms,
        bond_th_HX=bond_th_HX,
        bond_th_XX=bond_th_XX,
    )
    vectors = two_fragment_vectors(collision.q, collision.p, collision.mass, fragments)
    return fragments, vectors
