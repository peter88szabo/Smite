import copy

import numpy as np

from integrators.gradient import Potential_Energy
from sampling.polyrotation import angular_momentum, calcI
from utils.cenmass import cenmass


def center_of_mass(q, mass):
    q_flat = np.asarray(q, dtype=float).reshape(-1)
    mass = np.asarray(mass, dtype=float)
    q_internal, _ = cenmass(q_flat.copy(), np.zeros_like(q_flat), mass)
    return q_flat.reshape((-1, 3))[0] - q_internal.reshape((-1, 3))[0]


def center_of_mass_velocity(p, mass):
    p = np.asarray(p, dtype=float).reshape((-1, 3))
    mass = np.asarray(mass, dtype=float)
    return np.sum(p, axis=0) / np.sum(mass)


def remove_center_of_mass_motion(q, p, mass):
    q_flat = np.asarray(q, dtype=float).reshape(-1)
    p_flat = np.asarray(p, dtype=float).reshape(-1)
    mass = np.asarray(mass, dtype=float)

    com = center_of_mass(q_flat, mass)
    vcom = center_of_mass_velocity(p_flat, mass)
    q_internal, p_internal = cenmass(q_flat.copy(), p_flat.copy(), mass)
    return q_internal.reshape((-1, 3)), p_internal.reshape((-1, 3)), com, vcom


def kinetic_energy(p, mass):
    p = np.asarray(p, dtype=float).reshape((-1, 3))
    mass = np.asarray(mass, dtype=float)
    return float(np.sum(0.5 * np.sum(p * p, axis=1) / mass))


def inertia_tensor(q_internal, mass):
    _principal_moments, inertia_inverse = calcI(
        np.asarray(q_internal, dtype=float).reshape(-1), mass
    )
    return np.linalg.pinv(inertia_inverse)


def rotational_energy_from_inertia(angmom, inertia, tol=1e-12):
    angmom = np.asarray(angmom, dtype=float).reshape(3)
    inertia = np.asarray(inertia, dtype=float).reshape((3, 3))
    if np.linalg.norm(angmom) <= tol:
        return 0.0

    try:
        omega = np.linalg.pinv(inertia, rcond=tol) @ angmom
    except TypeError:
        omega = np.linalg.pinv(inertia) @ angmom
    return float(0.5 * np.dot(angmom, omega))


def diatomic_rotational_energy(q_internal, mass, angmom, tol=1e-12):
    q_internal = np.asarray(q_internal, dtype=float).reshape((2, 3))
    mass = np.asarray(mass, dtype=float)
    angmom = np.asarray(angmom, dtype=float).reshape(3)

    bond_length = float(np.linalg.norm(q_internal[1] - q_internal[0]))
    reduced_mass = float(mass[0] * mass[1] / (mass[0] + mass[1]))
    moment_of_inertia = reduced_mass * bond_length * bond_length
    if moment_of_inertia <= tol:
        raise ValueError("Cannot compute diatomic rotational energy for zero bond length")
    if np.linalg.norm(angmom) <= tol:
        return 0.0, moment_of_inertia
    return float(np.dot(angmom, angmom) / (2.0 * moment_of_inertia)), moment_of_inertia


def _equilibrium_geometry_for_fragment(fragment, equilibrium_geometries):
    if not equilibrium_geometries:
        return None

    candidates = []
    formula = fragment.get("formula")
    if formula is not None:
        candidates.append(formula)
    candidates.append(tuple(fragment["atoms"]))
    candidates.append("".join(fragment["atoms"]))

    for key in candidates:
        if key in equilibrium_geometries:
            return np.asarray(equilibrium_geometries[key], dtype=float).reshape((-1, 3))
    return None


def fragment_energy_partition(
    q,
    p,
    mass,
    fragment,
    equilibrium_geometries=None,
    potential_energy=None,
    reference_potential_energy=None,
    potential_energy_source=None,
):
    """Partition one separated fragment into internal rotational/vibrational energy.

    A vibrational energy is only reported when both the current and reference
    isolated-fragment potential energies are available.  The reference is
    normally the user-supplied product equilibrium geometry.
    """
    indices = np.asarray(fragment["indices"], dtype=int)
    q_xyz = np.asarray(q, dtype=float).reshape((-1, 3))[indices]
    p_xyz = np.asarray(p, dtype=float).reshape((-1, 3))[indices]
    mass_vec = np.asarray(mass, dtype=float)[indices]

    q_internal, p_internal, com, vcom = remove_center_of_mass_motion(q_xyz, p_xyz, mass_vec)
    translational_kinetic = 0.5 * float(np.sum(mass_vec)) * float(np.dot(vcom, vcom))
    internal_kinetic = kinetic_energy(p_internal, mass_vec)
    fragment_angmom = np.asarray(
        angular_momentum(q_internal.reshape(-1), p_internal.reshape(-1)), dtype=float
    )

    eq_geometry = _equilibrium_geometry_for_fragment(fragment, equilibrium_geometries)
    if eq_geometry is not None and eq_geometry.shape != q_internal.shape:
        raise ValueError(
            f"Equilibrium geometry for {fragment['formula']} has shape {eq_geometry.shape}, "
            f"expected {q_internal.shape}"
        )

    if len(indices) == 1:
        rotational_energy = 0.0
        moment_of_inertia = None
        inertia = None
        rotational_reference = "atom"
    elif len(indices) == 2:
        reference_geometry = q_internal
        rotational_reference = "instantaneous_diatomic"
        if eq_geometry is not None:
            reference_geometry = eq_geometry - center_of_mass(eq_geometry, mass_vec)
            rotational_reference = "equilibrium_diatomic"
        rotational_energy, moment_of_inertia = diatomic_rotational_energy(
            reference_geometry, mass_vec, fragment_angmom
        )
        inertia = None
    else:
        if eq_geometry is not None:
            eq_internal = eq_geometry - center_of_mass(eq_geometry, mass_vec)
            inertia = inertia_tensor(eq_internal, mass_vec)
            rotational_reference = "equilibrium"
        else:
            inertia = inertia_tensor(q_internal, mass_vec)
            rotational_reference = "instantaneous"
        rotational_energy = rotational_energy_from_inertia(fragment_angmom, inertia)
        moment_of_inertia = None

    potential_relative = None
    vibrational_energy = None
    if potential_energy is not None and reference_potential_energy is not None:
        potential_relative = float(potential_energy) - float(reference_potential_energy)
        vibrational_energy = internal_kinetic + potential_relative - rotational_energy
        if abs(vibrational_energy) < 1e-14:
            vibrational_energy = 0.0

    return {
        "formula": fragment["formula"],
        "indices": fragment["indices"],
        "center_of_mass": com,
        "center_of_mass_velocity": vcom,
        "translational_kinetic_energy": translational_kinetic,
        "internal_kinetic_energy": internal_kinetic,
        "potential_energy": potential_energy,
        "reference_potential_energy": reference_potential_energy,
        "potential_energy_relative_to_reference": potential_relative,
        "potential_energy_source": potential_energy_source,
        "rotational_energy": rotational_energy,
        "vibrational_energy": vibrational_energy,
        "rotational_reference": rotational_reference,
        "moment_of_inertia": moment_of_inertia,
        "angular_momentum": fragment_angmom,
        "inertia_tensor": inertia,
    }


def _state_entries(channel_state, key):
    if not isinstance(channel_state, dict):
        return None
    return channel_state.get(key)


def _entry_matches_fragment(entry, fragment):
    if not isinstance(entry, dict):
        return False
    if "indices" in entry and list(entry["indices"]) != list(fragment["indices"]):
        return False
    if "formula" in entry and entry["formula"] != fragment["formula"]:
        return False
    if "atoms" in entry and list(entry["atoms"]) != list(fragment["atoms"]):
        return False
    return any(key in entry for key in ("indices", "formula", "atoms"))


def product_states_for_fragments(fragments, channel_state):
    """Resolve user-supplied product charge/multiplicity records by fragment."""
    entries = _state_entries(channel_state, "products")
    if not isinstance(entries, (list, tuple)) or len(entries) != len(fragments):
        raise ValueError(
            "Post-collision energy analysis requires one 'products' state record "
            "per detected product fragment."
        )

    resolved = []
    unused = list(entries)
    for fragment in fragments:
        matches = [entry for entry in unused if _entry_matches_fragment(entry, fragment)]
        if len(matches) != 1:
            raise ValueError(
                "Each product state must identify exactly one fragment with 'formula', "
                "'atoms', or 'indices'."
            )
        entry = matches[0]
        unused.remove(entry)
        _validate_state(entry, f"product {fragment['formula']}")
        resolved.append(entry)
    return resolved


def reactant_states(channel_state):
    """Return state records for original reactants A and B."""
    entries = _state_entries(channel_state, "reactants")
    if isinstance(entries, dict):
        states = {label: entries.get(label) for label in ("A", "B")}
    elif isinstance(entries, (list, tuple)):
        states = {entry.get("fragment"): entry for entry in entries if isinstance(entry, dict)}
    else:
        states = {}

    for label in ("A", "B"):
        if label not in states or states[label] is None:
            raise ValueError(
                "Post-collision energy analysis requires reactant states for both 'A' and 'B'."
            )
        _validate_state(states[label], f"reactant {label}")
    return states


def _validate_state(state, label):
    if not isinstance(state, dict):
        raise TypeError(f"State specification for {label} must be a dictionary")
    if "charge" not in state or "multiplicity" not in state:
        raise ValueError(f"State specification for {label} requires 'charge' and 'multiplicity'")
    int(state["charge"])
    if int(state["multiplicity"]) < 1:
        raise ValueError(f"State specification for {label} requires multiplicity >= 1")


def _qchem_for_state(base_qchem, state):
    if not isinstance(base_qchem, dict):
        raise ValueError("A collision qchem input is required for fragment-energy evaluation")
    qchem = copy.deepcopy(base_qchem)
    qchem.pop("_last_energy_cache", None)
    qchem.pop("_qchem_validated", None)
    qchem.pop("_unknown_qchem_keys", None)
    overrides = state.get("qchem_overrides", {})
    if overrides:
        if not isinstance(overrides, dict):
            raise TypeError("qchem_overrides must be a dictionary")
        qchem.update(overrides)
    qchem["charge"] = int(state["charge"])
    qchem["multiplicity"] = int(state["multiplicity"])
    qchem["wfu"] = False
    return qchem


def _energy(qchem, q, atoms, energy_evaluator=None):
    if energy_evaluator is not None:
        return float(energy_evaluator(qchem, np.asarray(q, dtype=float), list(atoms)))
    return float(Potential_Energy(qchem, None, np.asarray(q, dtype=float), list(atoms)))


def _coordinates_at_com(q_xyz, mass, target_com):
    q_xyz = np.asarray(q_xyz, dtype=float).reshape((-1, 3))
    source_com = np.average(q_xyz, axis=0, weights=mass)
    return q_xyz + np.asarray(target_com, dtype=float) - source_com


def _pes_isolated_coordinates(collision, focus_indices, focus_q_xyz=None, separation=100.0, base_q=None):
    """Return full PES coordinates with the spectator product moved far away."""
    q_xyz = np.asarray(collision.q if base_q is None else base_q, dtype=float).reshape((-1, 3)).copy()
    mass = np.asarray(collision.mass, dtype=float)
    focus_indices = np.asarray(focus_indices, dtype=int)
    spectator_mask = np.ones(collision.natom, dtype=bool)
    spectator_mask[focus_indices] = False
    spectator_indices = np.flatnonzero(spectator_mask)
    if spectator_indices.size == 0:
        raise ValueError("PES fragment isolation requires a spectator fragment")

    focus_mass = mass[focus_indices]
    focus_com = np.average(q_xyz[focus_indices], axis=0, weights=focus_mass)
    if focus_q_xyz is not None:
        q_xyz[focus_indices] = _coordinates_at_com(focus_q_xyz, focus_mass, focus_com)

    spectator_mass = mass[spectator_indices]
    spectator_com = np.average(q_xyz[spectator_indices], axis=0, weights=spectator_mass)
    direction = spectator_com - focus_com
    norm = np.linalg.norm(direction)
    if norm <= 1.0e-12:
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction /= norm
    destination = focus_com + float(separation) * direction
    q_xyz[spectator_indices] += destination - spectator_com
    return q_xyz.reshape(-1)


def _fragment_potential_energy(
    collision,
    fragment,
    state,
    q_xyz,
    isolation_distance,
    energy_evaluator=None,
):
    qchem = _qchem_for_state(collision.qchem, state)
    if qchem.get("qchem") == "PES":
        q_full = _pes_isolated_coordinates(
            collision,
            fragment["indices"],
            focus_q_xyz=q_xyz,
            separation=isolation_distance,
        )
        return _energy(qchem, q_full, collision.atoms, energy_evaluator), "full_pes_with_spectator_separated"
    return _energy(qchem, np.asarray(q_xyz).reshape(-1), fragment["atoms"], energy_evaluator), "isolated_fragment_qchem"


def product_potential_energies(
    collision,
    fragments,
    channel_state,
    equilibrium_geometries=None,
    isolation_distance=100.0,
    energy_evaluator=None,
):
    """Evaluate current and reference potentials for all separated products.

    For an electronic-structure backend the fragment is evaluated directly.
    A full-dimensional PES cannot generally expose a monomer's absolute energy;
    it is therefore evaluated twice with the spectator far away, so the
    potential difference from the reference geometry is well-defined.
    """
    states = product_states_for_fragments(fragments, channel_state)
    q_xyz = np.asarray(collision.q, dtype=float).reshape((-1, 3))
    result = []
    for fragment, state in zip(fragments, states):
        indices = np.asarray(fragment["indices"], dtype=int)
        current_q = q_xyz[indices]
        potential, source = _fragment_potential_energy(
            collision, fragment, state, current_q, isolation_distance, energy_evaluator
        )
        reference_q = _equilibrium_geometry_for_fragment(fragment, equilibrium_geometries)
        reference_potential = None
        if reference_q is not None:
            reference_potential, _ = _fragment_potential_energy(
                collision, fragment, state, reference_q, isolation_distance, energy_evaluator
            )
        result.append(
            {
                "state": dict(state),
                "potential_energy": potential,
                "reference_potential_energy": reference_potential,
                "potential_energy_source": source,
            }
        )
    return result


def _reactant_fragment(collision, label, q):
    natom_a = collision.fragment_A.natom
    if label == "A":
        indices = list(range(natom_a))
    elif label == "B":
        indices = list(range(natom_a, collision.natom))
    else:
        raise ValueError("Reactant label must be 'A' or 'B'")
    return {
        "indices": indices,
        "atoms": [collision.atoms[index] for index in indices],
        "formula": label,
    }


def _initial_coordinates(collision):
    if not hasattr(collision, "q_collision_initial") or collision.q_collision_initial is None:
        raise ValueError("Initial collision coordinates were not saved before propagation")
    return np.asarray(collision.q_collision_initial, dtype=float)

def reactant_total_energy(collision, channel_state, isolation_distance=100.0, energy_evaluator=None):
    """Return the initial reactant total energy on the chosen electronic surface."""
    states = reactant_states(channel_state)
    q_initial = _initial_coordinates(collision)
    p_initial = np.asarray(collision.p_collision_initial, dtype=float)
    qchem = _qchem_for_state(collision.qchem, states["A"])

    if qchem.get("qchem") == "PES":
        fragment_a = _reactant_fragment(collision, "A", q_initial)
        q_isolated = _pes_isolated_coordinates(
            collision,
            fragment_a["indices"],
            separation=isolation_distance,
            base_q=q_initial,
        )
        potential = _energy(qchem, q_isolated, collision.atoms, energy_evaluator)
    else:
        potential = 0.0
        q_xyz = q_initial.reshape((-1, 3))
        for label in ("A", "B"):
            fragment = _reactant_fragment(collision, label, q_initial)
            indices = np.asarray(fragment["indices"], dtype=int)
            potential += _energy(
                _qchem_for_state(collision.qchem, states[label]),
                q_xyz[indices].reshape(-1),
                fragment["atoms"],
                energy_evaluator,
            )
    kinetic = kinetic_energy(p_initial, collision.mass)
    return {
        "kinetic_energy": kinetic,
        "potential_energy": potential,
        "total_energy": kinetic + potential,
    }


def product_total_energy(
    collision,
    fragments,
    product_potentials,
    isolation_distance=100.0,
    energy_evaluator=None,
):
    """Return final product total energy using isolated fragments/asymptote."""
    qchem = _qchem_for_state(collision.qchem, product_potentials[0]["state"])
    if qchem.get("qchem") == "PES":
        q_full = _pes_isolated_coordinates(
            collision, fragments[0]["indices"], separation=isolation_distance
        )
        potential = _energy(qchem, q_full, collision.atoms, energy_evaluator)
    else:
        potential = sum(entry["potential_energy"] for entry in product_potentials)
    kinetic = kinetic_energy(collision.p, collision.mass)
    return {"kinetic_energy": kinetic, "potential_energy": potential, "total_energy": kinetic + potential}


def product_energy_partitions(
    q,
    p,
    mass,
    fragments,
    equilibrium_geometries=None,
    potential_energies=None,
):
    potential_energies = potential_energies or [None] * len(fragments)
    if len(potential_energies) != len(fragments):
        raise ValueError("Potential-energy records must match the number of product fragments")
    return [
        fragment_energy_partition(
            q,
            p,
            mass,
            fragment,
            equilibrium_geometries=equilibrium_geometries,
            potential_energy=None if energy is None else energy.get("potential_energy"),
            reference_potential_energy=None if energy is None else energy.get("reference_potential_energy"),
            potential_energy_source=None if energy is None else energy.get("potential_energy_source"),
        )
        for fragment, energy in zip(fragments, potential_energies)
    ]
