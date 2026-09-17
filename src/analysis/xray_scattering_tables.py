"""
Offline X-ray scattering tables for every element the source data covers.

This is the all-element counterpart of :mod:`analysis.xray_scattering_tables_hcno`,
which deliberately hardcodes only H, C, N and O. Conventions, formulae and the
data source are identical; only the coverage differs, so the two agree exactly
where they overlap (pinned by ``tests/test_xray_scattering_tables.py``).

Data read verbatim from the ESRF DABAX library, shipped in ``analysis/data/``:

1) Elastic f0(k): ``f0_InterTables.dat``
2) Inelastic S(x): ``isf_Hubbell.dat``

Conventions:
- DABAX elastic uses k = sin(theta) / lambda  [A^-1]
- DABAX inelastic uses x = sin(theta) / lambda [A^-1]
- Many scattering papers use q = 4*pi*sin(theta)/lambda

Therefore:
    k = x = q / (4*pi)

The neutral atom is the plain ``<El>`` entry of the elastic table, except for
hydrogen, which uses the neutral free-atom entry ``H.`` -- the same choice the
H/C/N/O module makes.

This module exposes:
- atomic elastic form factors from Cromer-Mann coefficients
- atomic inelastic scattering functions from Hubbell tables
- rotationally averaged independent atom model (IAM) scattering
  according to Eq. 25 in Moreno-Carrascosa et al.,
  J. Chem. Theory Comput. 2019, 15, 2836-2846

References for the underlying data:
- International Tables for Crystallography, Vol. C (elastic coefficients; the
  DABAX file follows the Cromer-Mann functional form of
  Cromer and Mann, Acta Cryst. (1968) A24, 321)
- J. H. Hubbell, W. J. Veigele, E. A. Briggs, R. T. Brown, D. T. Cromer and
  R. J. Howerton, J. Phys. Chem. Ref. Data 4, 471 (1975)
"""

from __future__ import annotations

import os
import re
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

Number = Union[float, np.ndarray]


DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
F0_FILE = os.path.join(DATA_DIRECTORY, "f0_InterTables.dat")
ISF_FILE = os.path.join(DATA_DIRECTORY, "isf_Hubbell.dat")

# Hydrogen has both a bonded entry ("H") and a neutral free-atom entry ("H.").
# The free atom is the right one for gas-phase scattering off a trajectory.
_NEUTRAL_ENTRY_OVERRIDES = {"H": "H."}

_SCAN_HEADER = re.compile(r"^#S\s+(\d+)\s+(\S+)\s*$")
_PLAIN_ELEMENT = re.compile(r"^[A-Z][a-z]?$")

# The source file spells ions three ways: mostly "O1-", once "O2-." with the
# trailing dot, and twice with the sign before the digit ("Fe+2", "Ru+4").
# Accepted from callers: "O1-", "O2-", "O-2", "Fe2+", "Fe+2".
_ION_ENTRY = re.compile(r"^([A-Z][a-z]?)(?:(\d+)([+-])|([+-])(\d+))\.?$")

_F0_CACHE: Dict[str, Dict[str, object]] = {}
_ISF_CACHE: Dict[str, np.ndarray] = {}
_ION_INDEX: Dict[Tuple[str, int], Dict[str, object]] = {}
_ATOMIC_NUMBERS: Dict[str, int] = {}
_ISF_APPROXIMATION_WARNED: set = set()
_INCONSISTENT_WARNED: set = set()

# The four-Gaussian fit reproduces the electron count to a few hundredths; a
# larger residual means the tabulated entry itself is wrong, not the fit.
_ELECTRON_COUNT_TOLERANCE = 0.5


def _parse_species(name: str) -> Optional[Tuple[str, int]]:
    """Return ``(element, charge)`` for any accepted spelling, else ``None``."""
    match = _ION_ENTRY.match(name)
    if match is None:
        return None
    element = match.group(1)
    if match.group(2) is not None:
        magnitude, sign = int(match.group(2)), match.group(3)
    else:
        sign, magnitude = match.group(4), int(match.group(5))
    return element, magnitude if sign == "+" else -magnitude


def _iter_scans(path: str):
    """Yield ``(entry_name, data_lines)`` for each ``#S`` block of a DABAX file."""
    with open(path, "r") as handle:
        lines = handle.read().splitlines()

    index = 0
    while index < len(lines):
        match = _SCAN_HEADER.match(lines[index])
        if match is None:
            index += 1
            continue

        name = match.group(2)
        index += 1
        while index < len(lines) and lines[index].startswith("#"):
            index += 1

        rows = []
        while index < len(lines) and lines[index].strip() and not lines[index].startswith("#"):
            rows.append(lines[index])
            index += 1

        yield name, rows


def _coefficients_from(row: str) -> Optional[Dict[str, object]]:
    values = [float(token) for token in row.split()]
    if len(values) != 9:
        return None
    # DABAX column order is a1 a2 a3 a4 c b1 b2 b3 b4.
    return {"a": values[0:4], "c": values[4], "b": values[5:9]}


def _load_f0() -> Dict[str, Dict[str, object]]:
    if _F0_CACHE:
        return _F0_CACHE

    entries = {name: rows for name, rows in _iter_scans(F0_FILE)}

    for name, rows in entries.items():
        if not rows:
            continue

        if _PLAIN_ELEMENT.match(name):
            source = _NEUTRAL_ENTRY_OVERRIDES.get(name, name)
            coefficients = _coefficients_from(entries.get(source, rows)[0])
            if coefficients is not None:
                _F0_CACHE[name] = coefficients
            continue

        species = _parse_species(name)
        if species is None or species[1] == 0:
            continue
        coefficients = _coefficients_from(rows[0])
        if coefficients is not None:
            # Whichever way the file spelled it, index it canonically.
            _ION_INDEX[species] = coefficients

    if not _F0_CACHE:
        raise RuntimeError(f"No elastic coefficients could be parsed from {F0_FILE}")
    return _F0_CACHE


def _load_ions() -> Dict[Tuple[str, int], Dict[str, object]]:
    _load_f0()
    return _ION_INDEX


def atomic_numbers() -> Dict[str, int]:
    """Atomic number of each element, from the ``#S`` index of the elastic file."""
    if not _ATOMIC_NUMBERS:
        with open(F0_FILE) as handle:
            for line in handle:
                match = _SCAN_HEADER.match(line)
                if match and _PLAIN_ELEMENT.match(match.group(2)):
                    _ATOMIC_NUMBERS.setdefault(match.group(2), int(match.group(1)))
    return _ATOMIC_NUMBERS


def electron_count_error(species: str) -> float:
    """``f0(0)`` minus the electron count the species should have.

    ``f0(q=0)`` sums to the number of electrons, so this is a self-consistency
    check on the tabulated coefficients. The four-Gaussian fit leaves residuals
    of a few hundredths; anything larger means the entry itself is wrong.
    """
    element, charge, pars = _resolve(species)
    number = atomic_numbers().get(element)
    if number is None:
        return 0.0
    return (sum(pars["a"]) + pars["c"]) - (number - charge)


def _check_entry_consistency(species: str, element: str, charge: int) -> None:
    """Warn once about a source entry whose electron count does not add up.

    The published actinide block of ``f0_InterTables.dat`` contains two swapped
    pairs -- ``Np3+``/``Np6+``, and ``Np4+``/neutral ``Pu`` -- so those entries
    return another species' coefficients. This check is derived from the data
    rather than from a hardcoded list, so it will flag any similar defect.
    """
    key = (element, charge)
    if key in _INCONSISTENT_WARNED:
        return

    error = electron_count_error(species)
    if abs(error) <= _ELECTRON_COUNT_TOLERANCE:
        return

    _INCONSISTENT_WARNED.add(key)
    warnings.warn(
        f"The tabulated elastic coefficients for {species} sum to "
        f"{sum(_resolve(species)[2]['a']) + _resolve(species)[2]['c']:.3f} "
        f"electrons, but the species has {atomic_numbers()[element] - charge}. "
        f"This entry of {os.path.basename(F0_FILE)} is inconsistent in the "
        "published data and should not be trusted.",
        RuntimeWarning,
        stacklevel=3,
    )


def supported_ions() -> List[str]:
    """Canonical names of every tabulated ionic species, e.g. ``O2-``, ``Fe3+``."""
    return sorted(
        f"{element}{abs(charge)}{'+' if charge > 0 else '-'}"
        for element, charge in _load_ions()
    )


def _load_isf() -> Dict[str, np.ndarray]:
    if _ISF_CACHE:
        return _ISF_CACHE

    for name, rows in _iter_scans(ISF_FILE):
        if not _PLAIN_ELEMENT.match(name) or not rows:
            continue
        _ISF_CACHE[name] = np.array(
            [[float(token) for token in row.split()] for row in rows], dtype=float
        )

    if not _ISF_CACHE:
        raise RuntimeError(f"No inelastic tables could be parsed from {ISF_FILE}")
    return _ISF_CACHE


def supported_elements() -> List[str]:
    """Elements present in both the elastic and the inelastic table."""
    return sorted(set(_load_f0()) & set(_load_isf()))


def _capitalize_symbol(symbol: str) -> str:
    return symbol[:1].upper() + symbol[1:].lower()


def _canonical_case(name: str) -> str:
    """Capitalize the element symbol, leaving any charge suffix alone."""
    match = re.match(r"^([A-Za-z]+)(.*)$", name)
    if match is None:
        return name
    return _capitalize_symbol(match.group(1)) + match.group(2)


def _resolve(species: str) -> Tuple[str, int, Dict[str, object]]:
    """Map a species name onto ``(element, charge, elastic coefficients)``.

    Accepts a neutral symbol (``O``, case-insensitive) or an ion in any of the
    spellings the source file and ordinary usage produce (``O1-``, ``O2-``,
    ``O-2``, ``Fe2+``, ``Fe+2``).
    """
    name = _canonical_case(species.strip())

    parsed = _parse_species(name)
    if parsed is not None and parsed[1] != 0:
        element, charge = _capitalize_symbol(parsed[0]), parsed[1]
        coefficients = _load_ions().get((element, charge))
        if coefficients is None:
            if element in _load_f0():
                raise KeyError(
                    f"No tabulated form factor for ion {species}. "
                    f"Available for {element}: "
                    f"{[i for i in supported_ions() if i.startswith(element)] or 'none'}"
                )
            raise KeyError(f"Unsupported element: {species}")
        if element not in _load_isf():
            raise KeyError(
                f"No inelastic scattering function tabulated for element: {element}"
            )
        return element, charge, coefficients

    element = _capitalize_symbol(name)
    if element not in _load_f0():
        raise KeyError(f"Unsupported element: {species}")
    if element not in _load_isf():
        raise KeyError(
            f"No inelastic scattering function tabulated for element: {species}"
        )
    return element, 0, _load_f0()[element]


def _normalize_element(element: str) -> str:
    """The neutral element a species belongs to."""
    return _resolve(element)[0]


def isf_is_approximated(species: str) -> bool:
    """True when the inelastic term for ``species`` falls back to the neutral atom.

    ``isf_Hubbell.dat`` tabulates neutral atoms only, so any ion's incoherent
    scattering function is taken from its neutral parent.
    """
    return _resolve(species)[1] != 0


def _warn_neutral_isf(element: str, charge: int) -> None:
    """Warn once per ionic species, not once per evaluation."""
    key = (element, charge)
    if key in _ISF_APPROXIMATION_WARNED:
        return
    _ISF_APPROXIMATION_WARNED.add(key)
    sign = "+" if charge > 0 else "-"
    warnings.warn(
        f"No inelastic scattering function is tabulated for "
        f"{element}{abs(charge)}{sign}; using the neutral {element} table. "
        "The elastic term is exact for the ion, the inelastic term is not: it "
        f"corresponds to {element} rather than to an ion with "
        f"{-charge:+d} electrons. Query analysis.xray_scattering_tables."
        "isf_is_approximated() to detect this.",
        RuntimeWarning,
        stacklevel=3,
    )


def _as_array(values: Number) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _maybe_scalar(values: Number, out: np.ndarray) -> Number:
    return float(out) if np.ndim(values) == 0 else out


def _q_to_x(q_ang_inv: Number) -> np.ndarray:
    return _as_array(q_ang_inv) / (4.0 * np.pi)


def _sin_over_x(x: np.ndarray) -> np.ndarray:
    out = np.ones_like(x, dtype=float)
    mask = x != 0.0
    out[mask] = np.sin(x[mask]) / x[mask]
    return out


def f0_cromer_mann(element: str, q_ang_inv: Number) -> Number:
    """Elastic atomic form factor f0(q) from offline Cromer-Mann data.

    ``element`` may be a neutral symbol or an ion such as ``O2-`` or ``Fe3+``.
    """
    el, charge, pars = _resolve(element)
    _check_entry_consistency(element, el, charge)
    a = np.asarray(pars["a"], dtype=float)
    b = np.asarray(pars["b"], dtype=float)
    c = float(pars["c"])

    x = _q_to_x(q_ang_inv)
    out = np.zeros_like(x, dtype=float)
    for ai, bi in zip(a, b):
        out += ai * np.exp(-bi * x * x)
    out += c
    return _maybe_scalar(q_ang_inv, out)


def incoherent_S_hubbell(element: str, q_ang_inv: Number) -> Number:
    """Inelastic scattering function S(q) from offline Hubbell tables.

    Tabulated for neutral atoms only. An ion falls back to its neutral parent
    and warns once; see :func:`isf_is_approximated`.
    """
    el, charge, _ = _resolve(element)
    if charge != 0:
        _warn_neutral_isf(el, charge)
    table = _load_isf()[el]
    x_tab = table[:, 0]
    s_tab = table[:, 1]
    x = _q_to_x(q_ang_inv)
    out = np.interp(x, x_tab, s_tab, left=s_tab[0], right=s_tab[-1])
    return _maybe_scalar(q_ang_inv, out)


def independent_atom_model_scattering(
    elements: Sequence[str],
    coordinates_angstrom: Sequence[Sequence[float]],
    q_ang_inv: Number,
) -> Dict[str, Number]:
    """
    Rotationally averaged IAM scattering from Eq. 25 of Moreno-Carrascosa 2019.

    Parameters
    ----------
    elements
        Element symbols for each atom.
    coordinates_angstrom
        Cartesian coordinates in angstrom with shape (nat, 3).
    q_ang_inv
        Scalar or array of |q| values in A^-1.

    Returns
    -------
    dict
        Keys:
        - ``elastic``: Debye elastic term
        - ``inelastic``: sum of atomic incoherent terms
        - ``total``: elastic + inelastic
    """
    coords = np.asarray(coordinates_angstrom, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coordinates_angstrom must have shape (natoms, 3)")
    if len(elements) != len(coords):
        raise ValueError("elements and coordinates_angstrom must have the same length")

    q = _as_array(q_ang_inv)
    q_flat = q.reshape(-1)

    f_atoms = np.vstack([_as_array(f0_cromer_mann(el, q_flat)) for el in elements])
    s_atoms = np.vstack([_as_array(incoherent_S_hubbell(el, q_flat)) for el in elements])

    rij = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    qr = q_flat[None, None, :] * rij[:, :, None]
    debye = _sin_over_x(qr)

    elastic = np.sum(
        f_atoms[:, None, :] * f_atoms[None, :, :] * debye,
        axis=(0, 1),
    )
    inelastic = np.sum(s_atoms, axis=0)
    total = elastic + inelastic

    if np.ndim(q_ang_inv) == 0:
        return {
            "elastic": float(elastic[0]),
            "inelastic": float(inelastic[0]),
            "total": float(total[0]),
        }

    return {
        "elastic": elastic.reshape(q.shape),
        "inelastic": inelastic.reshape(q.shape),
        "total": total.reshape(q.shape),
    }


class XrayScattering:
    """Offline elastic and inelastic atomic scattering data, all elements."""

    def f0(self, element: str, q_ang_inv: Number) -> Number:
        return f0_cromer_mann(element, q_ang_inv)

    def isf(self, element: str, q_ang_inv: Number) -> Number:
        return incoherent_S_hubbell(element, q_ang_inv)

    def get_isf_table(self, element: str) -> np.ndarray:
        el = _normalize_element(element)
        return _load_isf()[el].copy()

    def get_f0_coefficients(self, element: str) -> Dict[str, List[float]]:
        _, _, pars = _resolve(element)
        return {
            "a": list(pars["a"]),
            "b": list(pars["b"]),
            "c": pars["c"],
        }

    def elements(self) -> List[str]:
        return supported_elements()

    def ions(self) -> List[str]:
        return supported_ions()

    def iam(
        self,
        elements: Sequence[str],
        coordinates_angstrom: Sequence[Sequence[float]],
        q_ang_inv: Number,
    ) -> Dict[str, Number]:
        return independent_atom_model_scattering(elements, coordinates_angstrom, q_ang_inv)


if __name__ == "__main__":
    db = XrayScattering()
    print(f"{len(db.elements())} elements: {' '.join(db.elements())}")

    q = np.linspace(0.0, 25.0, 6)
    for el in ["H", "O", "Kr"]:
        print(f"\nElement: {el}")
        print("Cromer-Mann coefficients:", db.get_f0_coefficients(el))
        print("f0(q):", db.f0(el, q))
        print("isf(q):", db.isf(el, q))
