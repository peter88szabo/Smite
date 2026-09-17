from pathlib import Path

import numpy as np
import pytest

from analysis.short_time_spectrum import (
    CYCLES_PER_FS_TO_CM1,
    centered_moving_average,
    mass_weighted_vibrational_velocity,
    project_translation_and_rotation,
    save_dynamic_fragment_tracking,
    save_short_time_spectrum,
    short_time_channel_spectra,
    short_time_fourier_spectrum,
    track_dynamic_fragments,
)
from core.molecule import Molecule


def _diatom_history(signal, dt_fs=0.5, translation=None):
    signal = np.asarray(signal, dtype=float)
    nsamples = signal.size
    positions = np.tile(
        np.array([[-0.6, 0.0, 0.0], [0.6, 0.0, 0.0]]).reshape(1, 6),
        (nsamples, 1),
    )
    velocities = np.zeros_like(positions)
    velocities[:, 0] = -0.5 * signal
    velocities[:, 3] = 0.5 * signal
    if translation is not None:
        velocities += np.tile(np.asarray(translation), 2)[None, :]
    times = np.arange(nsamples) * dt_fs
    return positions, velocities, times


def test_translation_and_rotation_are_projected_frame_by_frame():
    coordinates = np.array(
        [[-1.0, -0.3, 0.0], [1.0, -0.3, 0.0], [0.0, 0.8, 0.0]]
    )
    masses = np.array([2.0, 3.0, 4.0])
    translation = np.array([0.4, -0.2, 0.7])
    angular_velocity = np.array([0.1, -0.3, 0.5])
    velocities = translation + np.cross(angular_velocity, coordinates)

    projected = project_translation_and_rotation(
        np.tile(coordinates.reshape(1, -1), (5, 1)),
        np.tile(velocities.reshape(1, -1), (5, 1)),
        masses,
    )
    np.testing.assert_allclose(projected, 0.0, atol=2.0e-15)


def test_rectangular_spectrum_integrates_to_window_averaged_kinetic_energy():
    rng = np.random.default_rng(1947)
    positions, velocities, _times = _diatom_history(rng.normal(size=256))
    window_samples = 64
    hop_samples = 16

    result = short_time_fourier_spectrum(
        positions,
        velocities,
        [16.0, 16.0],
        dt_fs=0.5,
        window_fs=window_samples * 0.5,
        hop_fs=hop_samples * 0.5,
        window="boxcar",
        nfft=128,
    )
    weighted, _indices = mass_weighted_vibrational_velocity(
        positions, velocities, [16.0, 16.0]
    )
    direct = np.array(
        [
            0.5
            / window_samples
            * np.sum(weighted[start : start + window_samples] ** 2)
            for start in range(
                0,
                len(weighted) - window_samples + 1,
                hop_samples,
            )
        ]
    )
    np.testing.assert_allclose(
        result.kinetic_energy_hartree,
        direct,
        rtol=2.0e-15,
        atol=2.0e-15,
    )


def test_short_time_spectrum_tracks_a_changing_vibrational_frequency():
    dt_fs = 0.5
    nsamples = 2400
    times = np.arange(nsamples) * dt_fs
    signal = np.empty(nsamples)
    midpoint = nsamples // 2
    signal[:midpoint] = np.sin(
        2.0 * np.pi * (800.0 / CYCLES_PER_FS_TO_CM1) * times[:midpoint]
    )
    signal[midpoint:] = np.sin(
        2.0
        * np.pi
        * (1800.0 / CYCLES_PER_FS_TO_CM1)
        * (times[midpoint:] - times[midpoint])
    )
    positions, velocities, times = _diatom_history(signal, dt_fs)

    result = short_time_fourier_spectrum(
        positions,
        velocities,
        [16.0, 16.0],
        times_fs=times,
        window_fs=200.0,
        hop_fs=20.0,
        nfft=1024,
        frequency_bands={"vibration": (400.0, 2200.0)},
        max_frequency_cm1=2500.0,
    )
    early = result.time_fs < 400.0
    late = result.time_fs > 800.0
    assert np.median(result.band_peak_frequency_cm1["vibration"][early]) == pytest.approx(
        800.0, abs=70.0
    )
    assert np.median(result.band_peak_frequency_cm1["vibration"][late]) == pytest.approx(
        1800.0, abs=70.0
    )


def test_atom_selection_excludes_collision_partner_translation():
    dt_fs = 0.5
    times = np.arange(800) * dt_fs
    signal = np.sin(
        2.0 * np.pi * (1600.0 / CYCLES_PER_FS_TO_CM1) * times
    )
    positions, velocities, _ = _diatom_history(signal, dt_fs)
    third_positions = np.column_stack(
        (10.0 - 0.01 * times, np.ones(times.size), np.zeros(times.size))
    )
    third_velocities = np.tile(np.array([-0.01, 0.0, 0.0]), (times.size, 1))
    tri_positions = np.column_stack((positions, third_positions))
    tri_velocities = np.column_stack((velocities, third_velocities))

    selected = short_time_fourier_spectrum(
        tri_positions,
        tri_velocities,
        [16.0, 16.0, 1.0],
        dt_fs=dt_fs,
        atom_indices=(0, 1),
        window_fs=200.0,
        hop_fs=20.0,
        nfft=512,
    )
    isolated = short_time_fourier_spectrum(
        positions,
        velocities,
        [16.0, 16.0],
        dt_fs=dt_fs,
        window_fs=200.0,
        hop_fs=20.0,
        nfft=512,
    )
    np.testing.assert_allclose(
        selected.spectrum_hartree_per_cm1,
        isolated.spectrum_hartree_per_cm1,
        atol=1.0e-15,
    )


def test_frequency_band_integral_and_peak_use_physical_units():
    dt_fs = 0.5
    window_samples = 400
    exact_frequency = (
        10.0 * CYCLES_PER_FS_TO_CM1 / (window_samples * dt_fs)
    )
    times = np.arange(window_samples) * dt_fs
    signal = np.sin(
        2.0 * np.pi * (exact_frequency / CYCLES_PER_FS_TO_CM1) * times
    )
    positions, velocities, _ = _diatom_history(signal, dt_fs)

    result = short_time_fourier_spectrum(
        positions,
        velocities,
        [16.0, 16.0],
        dt_fs=dt_fs,
        window_fs=window_samples * dt_fs,
        hop_fs=20.0,
        frequency_bands={"stretch": (1500.0, 1850.0)},
    )
    np.testing.assert_allclose(
        result.band_energy_hartree["stretch"],
        result.kinetic_energy_hartree,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    assert result.band_peak_frequency_cm1["stretch"][0] == pytest.approx(
        exact_frequency
    )


def test_molecule_convenience_api_and_file_outputs(tmp_path):
    times = np.arange(500) * 0.5
    signal = np.sin(
        2.0 * np.pi * (1200.0 / CYCLES_PER_FS_TO_CM1) * times
    )
    positions, velocities, _ = _diatom_history(signal)
    molecule = Molecule(
        atoms=["O", "O"],
        mass=np.array([16.0, 16.0]),
        q_ini=positions[0],
        p_ini=np.zeros(6),
    )
    molecule.fname = "O2"
    molecule.qsave = list(positions)
    molecule.vsave = list(velocities)
    molecule.tsave = list(times)

    result = molecule.time_resolved_vibrational_spectrum(
        window_fs=100.0,
        hop_fs=10.0,
        max_frequency_cm1=2500.0,
        save=False,
    )
    assert result.time_fs.size > 1

    files = save_short_time_spectrum(
        result,
        tmp_path / "o2_stft",
        write_text=True,
        plot=False,
    )
    assert set(files) == {"npz", "spectrum_data", "band_data"}
    assert all(Path(path).is_file() for path in files.values())


def test_nonuniform_time_grid_and_invalid_selection_fail_cleanly():
    positions = np.zeros((4, 6))
    positions[:, 3] = 1.0
    velocities = np.zeros_like(positions)

    with pytest.raises(ValueError, match="uniform"):
        short_time_fourier_spectrum(
            positions,
            velocities,
            [1.0, 1.0],
            times_fs=[0.0, 0.5, 1.1, 1.5],
            window_fs=1.0,
            hop_fs=0.5,
        )
    with pytest.raises(ValueError, match="duplicates"):
        short_time_fourier_spectrum(
            positions,
            velocities,
            [1.0, 1.0],
            dt_fs=0.5,
            atom_indices=(0, 0),
            window_fs=1.0,
            hop_fs=0.5,
        )


def test_centered_moving_average_preserves_constant_signal():
    np.testing.assert_allclose(
        centered_moving_average(np.full(11, 3.5), 5),
        3.5,
    )
    np.testing.assert_allclose(
        centered_moving_average(np.full(3, 2.5), 20),
        2.5,

    )


def test_virial_ratio_is_one_for_a_resolved_harmonic_mode():
    dt_fs = 0.5
    window_samples = 400
    times = np.arange(window_samples) * dt_fs
    frequency = 10.0 * CYCLES_PER_FS_TO_CM1 / (window_samples * dt_fs)
    phase = 2.0 * np.pi * frequency / CYCLES_PER_FS_TO_CM1 * times
    positions, velocities, _ = _diatom_history(np.sin(phase), dt_fs)
    # For unit masses and the velocities constructed by _diatom_history,
    # T = 0.25 sin^2(phase); the matching harmonic potential is 0.25 cos^2.
    potential = 0.25 * np.cos(phase) ** 2

    result = short_time_fourier_spectrum(
        positions,
        velocities,
        [1.0, 1.0],
        dt_fs=dt_fs,
        window_fs=window_samples * dt_fs,
        hop_fs=20.0,
        potential_energy_hartree=potential,
        reference_potential_hartree=0.0,
    )

    assert result.internal_energy_hartree[0] == pytest.approx(0.25, abs=2.0e-15)
    assert result.virial_ratio[0] == pytest.approx(1.0, abs=2.0e-14)
    assert result.virial_residual_hartree[0] == pytest.approx(0.0, abs=5.0e-15)


def test_fractional_band_bins_centroid_width_and_sensitivity():
    dt_fs = 0.5
    window_samples = 400
    bin_width = CYCLES_PER_FS_TO_CM1 / (window_samples * dt_fs)
    frequency = 10.0 * bin_width
    times = np.arange(window_samples) * dt_fs
    signal = np.sin(
        2.0 * np.pi * frequency / CYCLES_PER_FS_TO_CM1 * times
    )
    positions, velocities, _ = _diatom_history(signal, dt_fs)
    lower, upper = frequency, frequency + 0.5 * bin_width

    result = short_time_fourier_spectrum(
        positions,
        velocities,
        [1.0, 1.0],
        dt_fs=dt_fs,
        window_fs=window_samples * dt_fs,
        hop_fs=20.0,
        frequency_bands={"half_bin": (lower, upper)},
        band_sensitivity_cm1=0.25 * bin_width,
    )

    assert result.band_energy_hartree["half_bin"][0] == pytest.approx(
        0.5 * result.kinetic_energy_hartree[0], rel=2.0e-14, abs=2.0e-14
    )
    assert result.band_centroid_frequency_cm1["half_bin"][0] == pytest.approx(
        frequency + 0.25 * bin_width, abs=2.0e-12
    )
    assert result.band_width_frequency_cm1["half_bin"][0] == pytest.approx(
        0.5 * bin_width / np.sqrt(12.0), abs=2.0e-9
    )
    narrow, expanded = result.band_energy_sensitivity_hartree["half_bin"][0]
    assert narrow <= result.band_energy_hartree["half_bin"][0] <= expanded
    assert result.nominal_resolution_cm1 == pytest.approx(bin_width)


def test_parabolic_peak_refinement_improves_an_off_grid_tone():
    dt_fs = 0.5
    window_samples = 400
    bin_width = CYCLES_PER_FS_TO_CM1 / (window_samples * dt_fs)
    frequency = 10.3 * bin_width
    times = np.arange(window_samples) * dt_fs
    signal = np.sin(
        2.0 * np.pi * frequency / CYCLES_PER_FS_TO_CM1 * times
    )
    positions, velocities, _ = _diatom_history(signal, dt_fs)

    discrete = short_time_fourier_spectrum(
        positions,
        velocities,
        [1.0, 1.0],
        dt_fs=dt_fs,
        window_fs=window_samples * dt_fs,
        hop_fs=20.0,
        window="hann",
        peak_estimator="discrete",
    )
    refined = short_time_fourier_spectrum(
        positions,
        velocities,
        [1.0, 1.0],
        dt_fs=dt_fs,
        window_fs=window_samples * dt_fs,
        hop_fs=20.0,
        window="hann",
        peak_estimator="parabolic",
    )

    discrete_error = abs(discrete.dominant_frequency_cm1[0] - frequency)
    refined_error = abs(refined.dominant_frequency_cm1[0] - frequency)
    assert refined_error < discrete_error


def test_dynamic_fragment_tracking_uses_bond_hysteresis():
    distances = np.array([1.2, 1.85, 1.75, 1.95, 1.85, 1.7])
    positions = np.zeros((distances.size, 9))
    positions[:, 3] = distances
    positions[:, 6] = 5.0

    tracking = track_dynamic_fragments(
        positions,
        dynamic_bonds={"O_O": (0, 1, 1.8, 1.9)},
        dt_fs=1.0,
        position_unit="angstrom",
    )

    np.testing.assert_array_equal(
        tracking.bond_active["O_O"],
        [True, True, True, False, False, True],
    )
    assert tracking.fragments[0] == ((0, 1), (2,))
    assert tracking.fragments[3] == ((0,), (1,), (2,))


def test_simultaneous_pair_and_full_system_channels_report_activity():
    dt_fs = 0.5
    times = np.arange(500) * dt_fs
    signal = np.sin(
        2.0 * np.pi * 1500.0 / CYCLES_PER_FS_TO_CM1 * times
    )
    pair_positions, pair_velocities, _ = _diatom_history(signal, dt_fs)
    hydrogen_position = np.tile([6.0, 0.0, 0.0], (times.size, 1))
    hydrogen_velocity = np.zeros_like(hydrogen_position)
    positions = np.column_stack((pair_positions, hydrogen_position))
    velocities = np.column_stack((pair_velocities, hydrogen_velocity))

    analysis = short_time_channel_spectra(
        positions,
        velocities,
        [16.0, 16.0, 1.0],
        channels={
            "O_O": (0, 1),
            "H_O1": (2, 0),
            "H_O2": (2, 1),
            "HOO_all": (0, 1, 2),
        },
        dynamic_bonds={
            "O_O": (0, 1, 1.8, 2.0),
            "H_O1": (2, 0, 1.35, 1.55),
            "H_O2": (2, 1, 1.35, 1.55),
        },
        channel_frequency_bands={
            "O_O": {"stretch": (1000.0, 2200.0)},
            "H_O1": {"stretch": (2000.0, 4000.0)},
            "H_O2": {"stretch": (2000.0, 4000.0)},
            "HOO_all": {"all_vibration": (0.0, 4000.0)},
        },
        dt_fs=dt_fs,
        window_fs=100.0,
        hop_fs=10.0,
        nfft=256,
        position_unit="angstrom",
    )

    assert set(analysis.spectra) == {"O_O", "H_O1", "H_O2", "HOO_all"}
    np.testing.assert_allclose(analysis.spectra["O_O"].activity_fraction, 1.0)
    np.testing.assert_allclose(analysis.spectra["H_O1"].activity_fraction, 0.0)
    np.testing.assert_allclose(analysis.spectra["HOO_all"].activity_fraction, 0.0)
    # Direct O--O activity must be false in an H-bridged connected HOO complex.
    bridged_positions = np.zeros_like(positions)
    bridged_positions[:, 0] = -1.1
    bridged_positions[:, 3] = 1.1
    bridged_analysis = short_time_channel_spectra(
        bridged_positions,
        np.zeros_like(bridged_positions),
        [16.0, 16.0, 1.0],
        channels={"O_O": (0, 1), "HOO_all": (0, 1, 2)},
        dynamic_bonds={
            "O_O": (0, 1, 1.8, 2.0),
            "H_O1": (2, 0, 1.35, 1.55),
            "H_O2": (2, 1, 1.35, 1.55),
        },
        dt_fs=dt_fs,
        window_fs=100.0,
        hop_fs=10.0,
        nfft=256,
        position_unit="angstrom",
    )
    np.testing.assert_allclose(
        bridged_analysis.spectra["O_O"].activity_fraction, 0.0
    )
    np.testing.assert_allclose(
        bridged_analysis.spectra["HOO_all"].activity_fraction, 1.0
    )





def test_enhanced_numerical_and_plot_outputs(tmp_path):
    dt_fs = 0.5
    window_samples = 400
    times = np.arange(window_samples) * dt_fs
    frequency = 10.0 * CYCLES_PER_FS_TO_CM1 / (window_samples * dt_fs)
    phase = 2.0 * np.pi * frequency / CYCLES_PER_FS_TO_CM1 * times
    positions, velocities, _ = _diatom_history(np.sin(phase), dt_fs)
    potential = 0.25 * np.cos(phase) ** 2

    result = short_time_fourier_spectrum(
        positions,
        velocities,
        [1.0, 1.0],
        dt_fs=dt_fs,
        window_fs=window_samples * dt_fs,
        hop_fs=20.0,
        frequency_bands={"mode": (frequency - 100.0, frequency + 100.0)},
        potential_energy_hartree=potential,
        activity_mask=np.ones(window_samples),
    )
    files = save_short_time_spectrum(
        result,
        tmp_path / "enhanced",
        write_text=True,
        plot=True,
    )
    assert set(files) == {
        "npz",
        "spectrum_data",
        "band_data",
        "spectrogram_plot",
        "band_plot",
        "virial_plot",
        "activity_plot",
    }
    assert all(Path(path).is_file() for path in files.values())

    triatomic_positions = np.column_stack(
        (positions, np.tile([5.0, 0.0, 0.0], (window_samples, 1)))
    )
    tracking = track_dynamic_fragments(
        triatomic_positions,
        dynamic_bonds={"pair": (0, 1, 1.8, 2.0)},
        dt_fs=dt_fs,
        position_unit="angstrom",
    )
    tracking_files = save_dynamic_fragment_tracking(
        tracking,
        tmp_path / "fragments",
        write_text=True,
        plot=True,
    )
    assert set(tracking_files) == {"npz", "fragment_data", "bond_plot"}
    assert all(Path(path).is_file() for path in tracking_files.values())
