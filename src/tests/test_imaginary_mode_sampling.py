import numpy as np
import pytest

from parallel.trajectory_runner import TrajectoryContext, _write_initial_conditions_summary
from sampling.polyvibration import polyatom_vibration_sampling, project_sampled_mode_energies


def _two_atom_stretch_model():
    mass = np.array([1.0, 1.0])
    atoms = ["H", "H"]
    q_eq = np.array([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    # Mass-orthonormal symmetric stretch coordinate.
    normal_modes = np.array([[1.0 / np.sqrt(2.0)], [0.0], [0.0], [-1.0 / np.sqrt(2.0)], [0.0], [0.0]])
    return mass, atoms, q_eq, normal_modes


def test_imaginary_mode_is_excluded_by_default_without_mutating_frequencies():
    mass, atoms, q_eq, normal_modes = _two_atom_stretch_model()
    frequencies = np.array([-0.015])
    records = []

    with pytest.warns(RuntimeWarning, match="excluded from vibrational sampling"):
        q, p = polyatom_vibration_sampling(
            mass,
            atoms,
            q_eq.copy(),
            frequencies,
            normal_modes,
            {0: (frequencies[0], "Q", 4)},
            mode_energy_diagnostics=False,
            sampling_warnings=records,
        )

    np.testing.assert_allclose(q, q_eq)
    np.testing.assert_allclose(p, 0.0)
    np.testing.assert_allclose(frequencies, [-0.015])
    assert records == [
        {
            "type": "excluded_nonpositive_normal_mode",
            "mode": 0,
            "frequency_au": -0.015,
            "frequency_cm1": pytest.approx(-0.015 * 219474.6313705),
            "imaginary_mode_policy": "exclude",
        }
    ]


def test_legacy_abs_policy_remains_available_and_is_explicitly_warned():
    mass, atoms, q_eq, normal_modes = _two_atom_stretch_model()
    frequencies = np.array([-0.015])
    records = []

    with pytest.warns(RuntimeWarning, match="legacy imaginary_mode_policy='abs'"):
        q, p = polyatom_vibration_sampling(
            mass,
            atoms,
            q_eq.copy(),
            frequencies,
            normal_modes,
            {0: (frequencies[0], "Q", 0)},
            imaginary_mode_policy="abs",
            mode_energy_diagnostics=False,
            sampling_warnings=records,
        )

    recovered = project_sampled_mode_energies(
        mass,
        q_eq,
        np.abs(frequencies),
        normal_modes,
        q,
        p,
        print_report=False,
        write_file=False,
    )
    assert recovered["energy"][0] == pytest.approx(0.5 * abs(frequencies[0]))
    assert records[0]["type"] == "absolute_value_imaginary_normal_mode"


def test_legacy_thermal_frequency_cutoff_is_ignored_for_canonical_sampling(monkeypatch):
    mass, atoms, q_eq, normal_modes = _two_atom_stretch_model()
    frequencies = np.array([1.0e-4])
    records = []
    observed = []
    monkeypatch.setattr(
        "sampling.polyvibration.thermal_vibr_mode",
        lambda _rt, effective_frequency: observed.append(effective_frequency) or 0,
    )

    with pytest.warns(RuntimeWarning, match="cutoff was ignored"):
        polyatom_vibration_sampling(
            mass,
            atoms,
            q_eq.copy(),
            frequencies,
            normal_modes,
            {0: (frequencies[0], "T", 300.0)},
            thermal_frequency_cutoff_cm1=50.0,
            mode_energy_diagnostics=False,
            sampling_metadata=records,
        )

    record = records[0]
    assert observed == [pytest.approx(frequencies[0])]
    assert record["type"] == "thermal_vibrational_mode"
    assert record["cutoff_applied"] is False
    assert record["legacy_cutoff_ignored"] is True
    assert record["physical_frequency_cm1"] == pytest.approx(1.0e-4 * 219474.6313705)
    assert record["effective_sampling_frequency_cm1"] == pytest.approx(1.0e-4 * 219474.6313705)


def test_excluded_mode_warning_is_written_to_trajectory_initial_condition_metadata(tmp_path):
    class System:
        atoms = ["H"]
        q = np.zeros(3)
        p = np.zeros(3)
        sampling_metadata = [{
            "type": "thermal_vibrational_mode",
            "mode": 1,
            "effective_sampling_frequency_cm1": 50.0,
        }]
        sampling_warnings = [{"type": "excluded_nonpositive_normal_mode", "mode": 4}]

    initial_conditions_file = tmp_path / "initial_conditions.dat"
    context = TrajectoryContext(
        itraj=2,
        trajectory_dir=str(tmp_path),
        scratch_dir=str(tmp_path / "scratch"),
        traj_file=str(tmp_path / "traj.xyz"),
        backfile=str(tmp_path / "back.xyz"),
        initial_conditions_file=str(initial_conditions_file),
        nproc_per_job=1,
    )
    _write_initial_conditions_summary(System(), context)

    text = initial_conditions_file.read_text(encoding="utf-8")
    assert 'sampling_metadata {"effective_sampling_frequency_cm1": 50.0, "mode": 1, "type": "thermal_vibrational_mode"}' in text
    assert 'sampling_warning {"mode": 4, "type": "excluded_nonpositive_normal_mode"}' in text
