import pytest

from normalmode.thermochemistry import J_PER_HARTREE_PER_MOL, eval_thermo


@pytest.mark.parametrize("nu, s_ref", [(100.001, 13.198430), (150.0, 11.028562)])
def test_qrrho_entropy_mixes_every_mode(nu, s_ref):
    # Grimme, Chem. Eur. J. 18, 9955 (2012): S = w S_RRHO + (1-w) S_free-rotor for every
    # mode, w = 1/(1+(nu0/nu)^4), mu' = mu Bav/(mu + Bav). Reference values evaluated
    # independently from these formulas at T = 298.15 K, nu0 = 100 cm-1, Bav = 1e-44 kg m^2.
    # Just above the cutoff the pure RRHO value would be 14.452781 (100.001) and
    # 11.180604 (150) J/mol/K.
    thermo = eval_thermo([nu], [1.0, 0.5, 0.25], 18.0, 1.0, 298.15, 101325.0, 100.0)
    s_vib = thermo.svib * J_PER_HARTREE_PER_MOL  # J/mol/K
    assert abs(s_vib - s_ref) < 2.0e-3, f"nu = {nu}: S_vib = {s_vib:.6f}, reference = {s_ref:.6f}"
