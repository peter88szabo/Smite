import pytest

from utils.atomic_masses import atomic_masses, get_mass_vector
from utils.constants import (
    ATOMIC_MASS_GMOL_TO_AU,
    ATOMIC_MASS_UNIT_TO_ELECTRON_MASS,
)


def test_atomic_mass_unit_uses_u_to_electron_mass_ratio():
    assert ATOMIC_MASS_UNIT_TO_ELECTRON_MASS == pytest.approx(
        1822.88848628, rel=1.0e-12
    )
    assert ATOMIC_MASS_GMOL_TO_AU == ATOMIC_MASS_UNIT_TO_ELECTRON_MASS


def test_tabulated_atomic_weights_are_converted_with_u_not_neutron_mass():
    carbon_mass = get_mass_vector(["C"])[0]

    assert carbon_mass == pytest.approx(
        atomic_masses["C"] * ATOMIC_MASS_UNIT_TO_ELECTRON_MASS
    )
    assert carbon_mass != pytest.approx(
        atomic_masses["C"] * 1838.6836605
    )
