from __future__ import annotations

import inspect
from collections.abc import Mapping


class InputValidationError(ValueError):
    """Raised when a user-facing input dictionary or option set is invalid."""


def validate_mapping(name, value):
    if not isinstance(value, Mapping):
        raise InputValidationError(f"{name} must be a dictionary-like mapping")
    return value


def validate_required_keys(name, mapping, required_keys):
    missing = [key for key in required_keys if key not in mapping]
    if missing:
        raise InputValidationError(f"{name} is missing required key(s): {', '.join(missing)}")


def validate_allowed_keys(name, mapping, allowed_keys, *, strict=False):
    unknown = sorted(set(mapping) - set(allowed_keys))
    if unknown and strict:
        raise InputValidationError(f"{name} has unknown key(s): {', '.join(unknown)}")
    return unknown


def allowed_keyword_arguments(func):
    signature = inspect.signature(func)
    allowed = set()
    has_var_keyword = False
    for key, parameter in signature.parameters.items():
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            allowed.add(key)
        elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
    return allowed, has_var_keyword


def validate_keyword_arguments(function_name, func, options, *, required=(), strict=True):
    validate_mapping(f"{function_name} options", options)
    validate_required_keys(f"{function_name} options", options, required)
    allowed, has_var_keyword = allowed_keyword_arguments(func)
    if has_var_keyword and not strict:
        return []
    return validate_allowed_keys(f"{function_name} options", options, allowed, strict=strict)


def validate_atom_indices(label, indices, natoms, *, size=None, allow_repeats=False):
    try:
        values = tuple(int(index) for index in indices)
    except TypeError as exc:
        raise InputValidationError(f"{label} must be an iterable of atom indices") from exc
    if size is not None and len(values) != int(size):
        raise InputValidationError(f"{label} must contain exactly {size} atom indices")
    if not allow_repeats and len(set(values)) != len(values):
        raise InputValidationError(f"{label} must not contain repeated atom indices")
    bad = [index for index in values if index < 0 or index >= int(natoms)]
    if bad:
        raise InputValidationError(f"{label} atom index out of range 0..{int(natoms) - 1}: {bad}")
    return values


def validate_coordinate_reference(kind, atoms, reference):
    natoms = len(atoms)
    kind = str(kind).lower()
    sizes = {"bond": 2, "angle": 3, "dihedral": 4, "improper": 4}
    if kind not in sizes:
        raise InputValidationError("coordinate reference kind must be bond, angle, dihedral, or improper")
    return validate_atom_indices(f"{kind} reference", reference, natoms, size=sizes[kind])
