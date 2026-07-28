from __future__ import annotations

import numpy as np

from optimizer.common import _matches_reversed, _validate_reaction_indices


def coordinate_kind(kind):
    aliases = {
        "b": "bond",
        "bond": "bond",
        "a": "angle",
        "angle": "angle",
        "d": "dihedral",
        "dihedral": "dihedral",
        "torsion": "dihedral",
    }
    key = str(kind).lower()
    if key not in aliases:
        raise ValueError(f"Unsupported coordinate reference type {kind!r}; use bond, angle, or dihedral")
    return aliases[key]


def coordinate_label(kind, reference):
    kind = coordinate_kind(kind)
    tag = {"bond": "B", "angle": "A", "dihedral": "D"}[kind]
    return f"{tag} " + " ".join(str(int(i)) for i in reference)


def validate_coordinate_reference(kind, reference, natoms, label=None):
    kind = coordinate_kind(kind)
    size = {"bond": 2, "angle": 3, "dihedral": 4}[kind]
    return _validate_reaction_indices(reference, natoms, size, label or kind)


def parse_weighted_coordinate_reference(entry, index=None, *, default_weight=0.5):
    entry_label = "coordinate reference" if index is None else f"coordinate reference {int(index)}"
    if isinstance(entry, dict):
        raise ValueError(
            "Dictionary coordinate-reference entries are not supported; "
            "use tuple entries like ('bond', (1, 14), 1.0)"
        )
    try:
        fields = tuple(entry)
    except TypeError as exc:
        raise ValueError(
            f"Each {entry_label} must be a tuple/list: (type, atoms) or (type, atoms, weight)"
        ) from exc
    if len(fields) not in {2, 3}:
        raise ValueError(f"Each {entry_label} must be (type, atoms) or (type, atoms, weight)")

    kind, atoms = fields[:2]
    weight = fields[2] if len(fields) == 3 else None
    kind = coordinate_kind(kind)
    atoms = validate_coordinate_reference(kind, atoms, 10**9, f"{entry_label} atoms")

    if weight is None:
        return kind, atoms, float(default_weight), True
    try:
        weight_value = float(weight)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{entry_label} has a non-numeric weight {weight!r}") from exc
    if not np.isfinite(weight_value):
        raise ValueError(f"{entry_label} has a non-finite weight")
    return kind, atoms, weight_value, False


def find_internal_coordinate_indices(
    ic,
    kind,
    reference,
    *,
    label=None,
    include_linear_bends=True,
    include_impropers=True,
):
    kind = coordinate_kind(kind)
    label = label or kind
    reference = validate_coordinate_reference(kind, reference, ic.nat, label)

    if kind == "bond":
        target = tuple(sorted(reference))
        for idx, bond in enumerate(ic.bonds):
            if tuple(sorted(bond)) == target:
                return [idx], reference
        raise ValueError(f"{label}={reference!r} is not present in the internal coordinate set")

    if kind == "angle":
        target = tuple(reference)
        offset = ic.nbonds
        for idx, angle in enumerate(ic.angles):
            if _matches_reversed(target, angle):
                return [offset + idx], reference
        if include_linear_bends:
            linear_offset = offset + ic.nangles
            linear_matches = [
                linear_offset + idx
                for idx, linear_bend in enumerate(ic.linear_bends)
                if _matches_reversed(target, linear_bend[:3])
            ]
            if linear_matches:
                return linear_matches, reference
        raise ValueError(f"{label}={reference!r} is not present in the internal coordinate set")

    target = tuple(reference)
    offset = ic.nbonds + ic.nangles + ic.nlinear_bends
    for idx, dihedral in enumerate(ic.dihedrals):
        if _matches_reversed(target, dihedral):
            return [offset + idx], reference

    if include_impropers:
        improper_offset = offset + ic.ndihedrals
        for idx, improper in enumerate(ic.impropers):
            if tuple(improper) == target:
                return [improper_offset + idx], reference
    raise ValueError(f"{label}={reference!r} is not present in the internal coordinate set")


def find_single_internal_coordinate_index(ic, kind, reference, *, label=None):
    indices, canonical = find_internal_coordinate_indices(ic, kind, reference, label=label)
    if len(indices) != 1:
        raise ValueError(
            f"{label or kind}={tuple(reference)!r} maps to {len(indices)} internal coordinates; "
            "a single primitive coordinate is required here"
        )
    return indices[0], canonical


def internal_coordinate_direction(ic, kind, reference, *, label=None):
    indices, _canonical = find_internal_coordinate_indices(ic, kind, reference, label=label)
    direction = np.zeros(ic.nint, dtype=float)
    weight = 1.0 / np.sqrt(len(indices))
    for idx in indices:
        direction[idx] = weight
    return direction
