import pytest

from core.fragment import Fragment


def _diatom():
    return Fragment.Diatom_Init(
        fname="H2",
        atoms=["H", "H"],
        req=1.4,
        omega=4400.0,
        diatom="harmonic",
        random_rot=False,
    )


def test_zpe_vibration_is_not_replaced_by_thermal_rotation_temperature():
    fragment = _diatom()
    fragment.Specify_Mode_Sampling(
        init_vib_type="ZPE",
        init_rot_type="Temp",
        temp=300.0,
    )

    assert fragment.vibsampling[0] == (fragment.omega_diat, "Q", 0)
    assert fragment.rotsampling[0] == ("T", 300.0)


def test_diatom_thermal_vibration_requires_and_uses_its_own_mode_request():
    fragment = _diatom()
    fragment.Specify_Mode_Sampling(
        init_vib_type="Temp",
        init_rot_type="Jfix",
        temp=250.0,
        jrot=3,
    )

    assert fragment.vibsampling[0] == (fragment.omega_diat, "T", 250.0)
    assert fragment.rotsampling[0] == ("Q", 3)


def test_diatom_temperature_requirement_is_specific_to_the_requested_mode():
    with pytest.raises(ValueError, match="init_vib_type='Temp'"):
        _diatom().Specify_Mode_Sampling(init_vib_type="Temp", init_rot_type="Jfix")
