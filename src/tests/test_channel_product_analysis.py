from types import SimpleNamespace

import numpy as np
import pytest

from analysis.vectorcorr import analyze_collision


def _two_product_collision():
    # Two H2 products, each inside the 1.5 A H-X graph threshold and well
    # separated from the other fragment.  Equal and opposite COM momenta make
    # total linear momentum exactly zero while retaining relative translation.
    q = np.array([
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        5.0, 0.0, 0.0,
        6.0, 0.0, 0.0,
    ])
    p = np.array([
        0.0, 0.1, 0.0,
        0.0, 0.1, 0.0,
        0.0, -0.1, 0.0,
        0.0, -0.1, 0.0,
    ])
    return SimpleNamespace(
        q=q.copy(),
        p=p.copy(),
        q_collision_initial=q.copy(),
        p_collision_initial=p.copy(),
        q_ini=q.copy(),
        p_ini=p.copy(),
        mass=np.ones(4),
        natom=4,
        atoms=["H", "H", "H", "H"],
        fragment_A=SimpleNamespace(natom=2),
        fragment_B=SimpleNamespace(natom=2),
        qchem={"qchem": "mock"},
        vrel_ini=None,
        vrel_fin=None,
        Lorb_ini=None,
        Lorb_fin=None,
        Jrot_ini_A=None,
        Jrot_ini_B=None,
        Jrot_fin_A=None,
        Jrot_fin_B=None,
        Erelsq_ini=None,
        Erelsq_fin=None,
        bimp=None,
        bimp_fin=None,
    )


def test_confirmed_two_product_analysis_reports_partitions_angles_and_conservation(tmp_path):
    collision = _two_product_collision()
    output_file = tmp_path / "post_collision.dat"

    result = analyze_collision(
        collision,
        output_file=output_file,
        channel="H2_plus_H2",
        trajectory_initial_energy_hartree=1.0,
        trajectory_final_energy_hartree=0.975,
    )

    assert result["analysis_status"] == "complete"
    assert len(result["product_fragments"]) == 2
    assert result["Erelsq_fin_hartree"] == pytest.approx(0.02)
    assert result["product_com_translational_energy_hartree"] == pytest.approx(0.02)
    assert result["product_rotational_energy_hartree"] == pytest.approx(0.0)
    assert result["product_vibrational_energy_hartree"] is None
    np.testing.assert_allclose(result["orbital_angular_momentum_fin"], [0.0, 0.0, -1.0])
    assert [entry["translational_kinetic_energy"] for entry in result["fragment_energies"]] == pytest.approx([0.01, 0.01])
    assert result["angles_deg"]["theta_vrel_ini_vrel_fin"] == pytest.approx(0.0)
    assert result["conservation"]["linear_momentum_residual_norm"] == pytest.approx(0.0)
    assert result["conservation"]["angular_momentum_residual_norm"] == pytest.approx(0.0)
    assert result["conservation"]["trajectory_energy_residual_hartree"] == pytest.approx(-0.025)

    text = output_file.read_text(encoding="utf-8")
    assert "fragment_energy 0 H2" in text
    assert "linear_momentum_residual_norm" in text
    assert "theta_Jrot_ini_A_Lorb_fin" in text
