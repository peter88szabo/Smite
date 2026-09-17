"""All-element X-ray scattering tables.

``analysis.xray_scattering_tables_hcno`` hardcodes H, C, N and O by design.
``analysis.xray_scattering_tables`` reads the same ESRF DABAX source data for
every tabulated element. The load-bearing requirement is that the two agree
*exactly* where they overlap: the wider coverage must not perturb any existing
H/C/N/O result.
"""

import numpy as np
import pytest

from analysis import xray_scattering_tables as tables
from analysis import xray_scattering_tables_hcno as hcno


HCNO = ("H", "C", "N", "O")
WATER = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9270, 0.0]])


@pytest.fixture(scope="module")
def q_grid():
    return np.linspace(0.0, 25.0, 400)


# --------------------------------------------------------------------------
# Agreement with the hardcoded H/C/N/O module
# --------------------------------------------------------------------------

@pytest.mark.parametrize("element", HCNO)
def test_elastic_form_factor_matches_the_hardcoded_module(element, q_grid):
    assert tables.f0_cromer_mann(element, q_grid) == pytest.approx(
        hcno.f0_cromer_mann(element, q_grid), abs=0.0, rel=0.0
    )


@pytest.mark.parametrize("element", HCNO)
def test_inelastic_function_matches_the_hardcoded_module(element, q_grid):
    assert tables.incoherent_S_hubbell(element, q_grid) == pytest.approx(
        hcno.incoherent_S_hubbell(element, q_grid), abs=0.0, rel=0.0
    )


@pytest.mark.parametrize("element", HCNO)
def test_stored_coefficients_match_the_hardcoded_module(element):
    mine = tables.XrayScattering().get_f0_coefficients(element)
    theirs = hcno.XrayScatteringHCNO().get_f0_coefficients(element)
    assert mine["a"] == theirs["a"]
    assert mine["b"] == theirs["b"]
    assert mine["c"] == theirs["c"]


def test_iam_on_water_is_unchanged(q_grid):
    """The path `get_scattering_form_factors` actually takes."""
    mine = tables.independent_atom_model_scattering(["O", "H", "H"], WATER, q_grid)
    theirs = hcno.independent_atom_model_scattering(["O", "H", "H"], WATER, q_grid)

    for key in ("elastic", "inelastic", "total"):
        assert mine[key] == pytest.approx(theirs[key], abs=0.0, rel=0.0)


# --------------------------------------------------------------------------
# The coverage this module exists for
# --------------------------------------------------------------------------

def test_covers_the_whole_pes_library():
    """Every element appearing in peslib must now be supported."""
    supported = set(tables.supported_elements())
    for element in ("H", "C", "N", "O", "F", "Cl", "Ar", "Kr"):
        assert element in supported


def test_coverage_is_the_intersection_of_both_source_tables():
    supported = tables.supported_elements()
    assert len(supported) > 90
    for element in supported:
        assert tables.f0_cromer_mann(element, 1.0) > 0.0
        assert tables.incoherent_S_hubbell(element, 1.0) >= 0.0


@pytest.mark.parametrize(
    "elements, coordinates",
    [
        (["O", "H", "H", "Kr"], np.vstack([WATER, [0.0, 0.0, 2.6]])),
        (["Cl", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.3]])),
        (["F", "H", "H"], WATER),
        (["Ar", "H", "H"], WATER),
    ],
)
def test_systems_that_the_hcno_tables_reject_now_work(elements, coordinates):
    with pytest.raises(KeyError):
        hcno.independent_atom_model_scattering(elements, coordinates, 3.0)

    result = tables.independent_atom_model_scattering(elements, coordinates, 3.0)
    assert result["total"] > 0.0
    assert result["elastic"] > 0.0
    assert result["inelastic"] > 0.0


# --------------------------------------------------------------------------
# Physical sanity of the parsed data
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "element, atomic_number",
    [("H", 1), ("C", 6), ("O", 8), ("F", 9), ("Cl", 17), ("Ar", 18),
     ("Fe", 26), ("Kr", 36), ("I", 53), ("Au", 79)],
)
def test_forward_scattering_recovers_the_atomic_number(element, atomic_number):
    """f0(q=0) is the number of electrons, to the accuracy of the fit.

    The four-Gaussian Cromer-Mann form is a fit, so it does not hit Z exactly.
    Across all 98 supported elements the worst deviation is 0.06 electrons --
    except plutonium, see below.
    """
    assert tables.f0_cromer_mann(element, 0.0) == pytest.approx(atomic_number, abs=0.06)


# The published actinide block of f0_InterTables.dat contains two swapped pairs:
# Np3+/Np6+, and Np4+/neutral Pu. Those four entries therefore carry another
# species' coefficients. They are recorded here so they cannot later be mistaken
# for a regression, and the library warns when one of them is used.
KNOWN_INCONSISTENT = {"Pu", "Np3+", "Np4+", "Np6+"}


def test_only_the_known_actinide_entries_fail_the_electron_count():
    """f0(q=0) must equal the electron count for every other species."""
    offenders = {
        species
        for species in tables.supported_elements() + tables.supported_ions()
        if abs(tables.electron_count_error(species)) > 0.5
    }
    assert offenders == KNOWN_INCONSISTENT

    rest = [
        abs(tables.electron_count_error(species))
        for species in tables.supported_elements() + tables.supported_ions()
        if species not in KNOWN_INCONSISTENT
    ]
    assert max(rest) < 0.06


@pytest.mark.parametrize("species", sorted(KNOWN_INCONSISTENT))
def test_inconsistent_source_entries_warn_when_used(species):
    element, charge = (species, 0)
    parsed = tables._parse_species(species)
    if parsed is not None:
        element, charge = parsed
    tables._INCONSISTENT_WARNED.discard((element, charge))

    with pytest.warns(RuntimeWarning, match="inconsistent in the"):
        tables.f0_cromer_mann(species, 1.0)


def test_well_formed_species_do_not_warn():
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        for species in ("H", "C", "O", "Kr", "Fe", "U", "Pu3+", "U6+"):
            tables.f0_cromer_mann(species, 1.0)


@pytest.mark.parametrize("element", ["H", "O", "Kr"])
def test_elastic_form_factor_decays_with_q(element):
    q = np.array([0.0, 2.0, 5.0, 10.0, 25.0])
    f0 = tables.f0_cromer_mann(element, q)
    assert np.all(np.diff(f0) < 0.0)


@pytest.mark.parametrize("element", ["H", "O", "Kr"])
def test_inelastic_function_rises_towards_the_atomic_number(element):
    """S(q) starts at zero and saturates near Z at large q."""
    assert tables.incoherent_S_hubbell(element, 0.0) == pytest.approx(0.0, abs=1.0e-8)
    assert tables.incoherent_S_hubbell(element, 1.0e3) > 0.0


def test_unknown_element_is_rejected():
    with pytest.raises(KeyError, match="Unsupported element"):
        tables.f0_cromer_mann("Xx", 1.0)


def test_element_symbols_are_case_insensitive():
    assert tables.f0_cromer_mann("kr", 2.0) == tables.f0_cromer_mann("Kr", 2.0)
    assert tables.f0_cromer_mann("KR", 2.0) == tables.f0_cromer_mann("Kr", 2.0)


def test_scalar_and_array_inputs_agree():
    q = 3.7
    assert isinstance(tables.f0_cromer_mann("Kr", q), float)
    assert tables.f0_cromer_mann("Kr", np.array([q]))[0] == pytest.approx(
        tables.f0_cromer_mann("Kr", q)
    )


# --------------------------------------------------------------------------
# Charged fragments
# --------------------------------------------------------------------------

def test_ionic_species_are_exposed():
    ions = tables.supported_ions()
    assert len(ions) > 100
    for ion in ("O1-", "O2-", "Fe2+", "Fe3+", "Cl1-", "Na1+", "Si4+", "H1-"):
        assert ion in ions


@pytest.mark.parametrize(
    "ion, electrons",
    [("O1-", 9), ("O2-", 10), ("Fe2+", 24), ("Fe3+", 23), ("Cl1-", 18),
     ("Na1+", 10), ("Si4+", 10), ("H1-", 2)],
)
def test_ionic_forward_scattering_counts_the_electrons(ion, electrons):
    """f0(q=0) is the electron count, so Z minus the charge."""
    assert tables.f0_cromer_mann(ion, 0.0) == pytest.approx(electrons, abs=0.06)


def test_every_tabulated_ion_recovers_its_electron_count():
    import re

    atomic_numbers = {}
    with open(tables.F0_FILE) as handle:
        for line in handle:
            match = re.match(r"^#S\s+(\d+)\s+([A-Z][a-z]?)\s*$", line)
            if match:
                atomic_numbers.setdefault(match.group(2), int(match.group(1)))

    checked = 0
    for ion in tables.supported_ions():
        element, charge = tables._parse_species(ion)
        if element not in atomic_numbers:
            continue
        if ion in KNOWN_INCONSISTENT:
            continue
        expected = atomic_numbers[element] - charge
        assert tables.f0_cromer_mann(ion, 0.0) == pytest.approx(expected, abs=0.06), ion
        checked += 1

    assert checked > 100


@pytest.mark.parametrize(
    "written, canonical",
    [("Fe+2", "Fe2+"),      # the source file's own inverted spelling
     ("O-2", "O2-"),
     ("o2-", "O2-"),
     ("fe3+", "Fe3+"),
     ("CL1-", "Cl1-")],
)
def test_ion_spellings_are_interchangeable(written, canonical):
    assert tables.f0_cromer_mann(written, 2.0) == tables.f0_cromer_mann(canonical, 2.0)


def test_dotted_source_entry_is_reachable_under_its_plain_name():
    """Oxygen's 2- ion is spelled ``O2-.`` in the file, with a trailing dot."""
    assert tables.f0_cromer_mann("O2-", 0.0) == pytest.approx(10.0, abs=0.06)


def test_an_untabulated_ion_reports_what_is_available():
    with pytest.raises(KeyError, match="No tabulated form factor for ion"):
        tables.f0_cromer_mann("O7+", 1.0)


def test_inelastic_term_for_an_ion_falls_back_to_the_neutral_atom():
    """Hubbell tabulates neutral atoms only; the fallback must be announced."""
    assert tables.isf_is_approximated("O2-") is True
    assert tables.isf_is_approximated("O") is False

    tables._ISF_APPROXIMATION_WARNED.discard(("O", -2))
    with pytest.warns(RuntimeWarning, match="using the neutral O table"):
        approximate = tables.incoherent_S_hubbell("O2-", 3.0)

    assert approximate == tables.incoherent_S_hubbell("O", 3.0)


def test_the_neutral_fallback_warns_once_per_species():
    tables._ISF_APPROXIMATION_WARNED.discard(("Fe", 3))
    with pytest.warns(RuntimeWarning):
        tables.incoherent_S_hubbell("Fe3+", 3.0)

    import warnings as _warnings
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        tables.incoherent_S_hubbell("Fe3+", 4.0)
        tables.incoherent_S_hubbell("Fe3+", 5.0)
    assert caught == []


def test_a_charged_fragment_scatters_more_strongly_than_the_neutral():
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        neutral = tables.independent_atom_model_scattering(["O", "H", "H"], WATER, 3.0)
        anion = tables.independent_atom_model_scattering(["O2-", "H", "H"], WATER, 3.0)

    assert anion["elastic"] > neutral["elastic"]
    # The inelastic term is the neutral one by construction, so it is unchanged.
    assert anion["inelastic"] == pytest.approx(neutral["inelastic"])
