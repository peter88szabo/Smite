"""Fourier analysis of a time-resolved scattering signal.

Both transforms are checked against signals whose answer is known in advance:
a modulation injected at a chosen frequency must come back at that frequency,
and a bond that moves between two known lengths must show up at those distances
in the pair distribution.
"""

import numpy as np
import pytest

from analysis.scattering_fourier import (
    atomic_form_factor_squared,
    modified_scattering,
    pair_distribution_function,
    temporal_power_spectrum,
)
from analysis.xray_scattering_tables import independent_atom_model_scattering
from utils.constants import FS_TO_AU_TIME, HARTREE_TO_CM1


def _wavenumber_to_period_fs(wavenumber_cm1):
    """Period in fs of a mode at the given wavenumber."""
    angular_au = wavenumber_cm1 / HARTREE_TO_CM1
    return 2.0 * np.pi / angular_au / FS_TO_AU_TIME


# --------------------------------------------------------------------------
# A: transform along time
# --------------------------------------------------------------------------

@pytest.mark.parametrize("wavenumber", [800.0, 1600.0, 3000.0])
def test_an_injected_modulation_returns_at_its_own_frequency(wavenumber):
    dt_fs = 0.5
    n_frames = 2048
    time_fs = np.arange(n_frames) * dt_fs

    period = _wavenumber_to_period_fs(wavenumber)
    modulation = np.sin(2.0 * np.pi * time_fs / period)
    # A large static offset, which is what the difference signal must remove.
    intensity = 1.0e4 + modulation[:, None] * np.array([1.0, 0.5, 0.25])[None, :]

    frequencies, power = temporal_power_spectrum(time_fs, intensity)

    peak = frequencies[np.argmax(power[:, 0])]
    resolution = frequencies[1] - frequencies[0]
    assert abs(peak - wavenumber) < 2.0 * resolution


def test_the_static_offset_does_not_appear_as_a_zero_frequency_peak():
    time_fs = np.arange(512) * 0.5
    intensity = np.full((512, 3), 1.0e4)
    intensity += np.sin(2.0 * np.pi * time_fs / 20.0)[:, None]

    _, with_reference = temporal_power_spectrum(time_fs, intensity, reference="mean")
    _, without = temporal_power_spectrum(time_fs, intensity, reference="none")

    assert with_reference[0, 0] < 1.0e-6 * without[0, 0]


def test_relative_intensities_of_the_q_channels_are_preserved():
    time_fs = np.arange(1024) * 0.5
    modulation = np.sin(2.0 * np.pi * time_fs / 15.0)
    amplitudes = np.array([1.0, 0.5, 0.25])
    intensity = 1.0e3 + modulation[:, None] * amplitudes[None, :]

    _, power = temporal_power_spectrum(time_fs, intensity)
    peaks = power.max(axis=0)

    assert peaks[0] / peaks[1] == pytest.approx(4.0, rel=0.05)      # (1/0.5)^2
    assert peaks[0] / peaks[2] == pytest.approx(16.0, rel=0.05)     # (1/0.25)^2


def test_windowing_suppresses_truncation_sidelobes():
    """A non-integer number of periods leaks badly without apodization."""
    time_fs = np.arange(256) * 0.5
    intensity = np.sin(2.0 * np.pi * time_fs / 7.3)[:, None] + 100.0

    frequencies, hann = temporal_power_spectrum(time_fs, intensity, window="hann")
    _, boxcar = temporal_power_spectrum(time_fs, intensity, window="none")

    peak = np.argmax(hann[:, 0])
    far = np.abs(np.arange(len(frequencies)) - peak) > 20
    assert hann[far, 0].max() / hann[peak, 0] < boxcar[far, 0].max() / boxcar[peak, 0]


@pytest.mark.parametrize("bad", ["boxcar", "blackman"])
def test_unknown_window_is_rejected(bad):
    with pytest.raises(ValueError, match="Unknown window"):
        temporal_power_spectrum(np.arange(8) * 0.5, np.ones((8, 2)), window=bad)


def test_unknown_reference_is_rejected():
    with pytest.raises(ValueError, match="Unknown reference"):
        temporal_power_spectrum(np.arange(8) * 0.5, np.ones((8, 2)), reference="zero")


def test_a_single_frame_cannot_be_transformed_in_time():
    with pytest.raises(ValueError, match="at least two frames"):
        temporal_power_spectrum([0.0], np.ones((1, 4)))


# --------------------------------------------------------------------------
# B: sine transform along q
# --------------------------------------------------------------------------

def _diatomic_intensity(q, separations, elements=("O", "O")):
    """IAM intensity for a diatomic at each of the given bond lengths."""
    frames = []
    for distance in separations:
        coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, distance]])
        frames.append(
            independent_atom_model_scattering(list(elements), coordinates, q)["elastic"]
        )
    return np.vstack(frames)


def test_a_bond_that_stretches_shows_both_distances():
    """The pair leaves r0 and arrives at r1, so dPDF is negative then positive."""
    q = np.linspace(0.0, 15.0, 600)
    r0, r1 = 1.2, 2.0

    intensity = _diatomic_intensity(q, [r0, r1])
    f2 = atomic_form_factor_squared(["O", "O"], q)

    r, dpdf = pair_distribution_function(
        q, intensity, f2, reference="first", rmin=0.5, rmax=4.0, nr=700
    )

    # Frame 0 is the reference, so it is identically zero.
    assert dpdf[0] == pytest.approx(np.zeros_like(r), abs=1.0e-12)

    difference = dpdf[1]
    depletion = r[np.argmin(difference)]
    accumulation = r[np.argmax(difference)]

    assert depletion == pytest.approx(r0, abs=0.15)
    assert accumulation == pytest.approx(r1, abs=0.15)


def test_no_structural_change_gives_no_signal():
    q = np.linspace(0.0, 15.0, 400)
    intensity = _diatomic_intensity(q, [1.5, 1.5, 1.5])
    f2 = atomic_form_factor_squared(["O", "O"], q)

    _, dpdf = pair_distribution_function(q, intensity, f2, reference="first", nr=200)
    assert np.max(np.abs(dpdf)) == pytest.approx(0.0, abs=1.0e-12)


def test_damping_reduces_truncation_ripple():
    """Truncating the integral at qmax rings; the Gaussian damping suppresses it."""
    q = np.linspace(0.0, 10.0, 400)
    intensity = _diatomic_intensity(q, [1.2, 1.9])
    f2 = atomic_form_factor_squared(["O", "O"], q)

    r, damped = pair_distribution_function(q, intensity, f2, rmin=2.5, rmax=6.0, nr=400)
    _, undamped = pair_distribution_function(
        q, intensity, f2, rmin=2.5, rmax=6.0, nr=400, damping=0.0
    )

    # Well beyond both bond lengths only ripple remains.
    assert np.std(damped[1]) < np.std(undamped[1])


def test_modified_scattering_divides_out_the_atomic_form_factors():
    q = np.linspace(0.5, 10.0, 50)
    f2 = atomic_form_factor_squared(["O", "O"], q)
    difference = np.ones((1, q.size))

    sm = modified_scattering(q, difference, f2)
    assert sm == pytest.approx((q / f2)[None, :])


def test_atomic_form_factor_squared_sums_over_atoms():
    q = np.array([0.0, 3.0])
    single = atomic_form_factor_squared(["O"], q)
    triple = atomic_form_factor_squared(["O", "O", "O"], q)
    assert triple == pytest.approx(3.0 * single)


def test_mismatched_grids_are_rejected():
    q = np.linspace(0.0, 10.0, 20)
    with pytest.raises(ValueError, match="number of q points"):
        pair_distribution_function(q, np.ones((3, 19)), np.ones(20))
    with pytest.raises(ValueError, match="must match the q grid"):
        modified_scattering(q, np.ones((3, 20)), np.ones(19))


# --------------------------------------------------------------------------
# Short-time transform
# --------------------------------------------------------------------------

from analysis.scattering_fourier import (
    read_time_dependent_scattering,
    short_time_power_spectrum,
    short_time_scattering_spectrum,
)


def _switching_signal(first_cm1, second_cm1, n=1024, dt_fs=0.5):
    """A trace whose frequency changes halfway, which a global FFT cannot see."""
    t = np.arange(n) * dt_fs
    half = n // 2
    trace = np.where(
        t < t[half],
        np.sin(2.0 * np.pi * t / _wavenumber_to_period_fs(first_cm1)),
        np.sin(2.0 * np.pi * t / _wavenumber_to_period_fs(second_cm1)),
    )
    return t, 1.0e3 + trace[:, None]


def test_the_spectrogram_follows_a_frequency_that_changes_mid_trajectory():
    t, signal = _switching_signal(1800.0, 900.0)

    centres, frequencies, power = short_time_power_spectrum(
        t, signal, window_fs=80.0, hop_fs=20.0
    )
    band = frequencies > 200.0
    resolution = frequencies[1] - frequencies[0]

    early = frequencies[band][np.argmax(power[1, band, 0])]
    late = frequencies[band][np.argmax(power[-2, band, 0])]

    assert abs(early - 1800.0) < 1.5 * resolution
    assert abs(late - 900.0) < 1.5 * resolution
    assert centres[0] < centres[-1]


def test_a_global_transform_cannot_separate_the_two_regimes():
    """Motivates the short-time version: one spectrum reports only a blend."""
    t, signal = _switching_signal(1800.0, 900.0)
    frequencies, power = temporal_power_spectrum(t, signal)
    peak = frequencies[np.argmax(power[:, 0])]

    # Whichever it picks, it cannot report both.
    assert min(abs(peak - 1800.0), abs(peak - 900.0)) < 200.0


def test_a_one_dimensional_signal_is_accepted():
    t, signal = _switching_signal(1500.0, 1500.0)
    centres, frequencies, power = short_time_power_spectrum(
        t, signal[:, 0], window_fs=100.0, hop_fs=50.0
    )
    assert power.ndim == 3 and power.shape[2] == 1


def test_window_and_hop_default_to_the_trajectory_length():
    t, signal = _switching_signal(1200.0, 1200.0, n=256)
    centres, frequencies, power = short_time_power_spectrum(t, signal)
    assert power.shape[0] > 1
    assert len(centres) == power.shape[0]


def test_a_window_longer_than_the_trajectory_is_rejected():
    t, signal = _switching_signal(1200.0, 1200.0, n=64)
    with pytest.raises(ValueError, match="window_fs needs"):
        short_time_power_spectrum(t, signal, window_fs=10_000.0)


def test_hop_larger_than_the_window_is_rejected():
    t, signal = _switching_signal(1200.0, 1200.0, n=256)
    with pytest.raises(ValueError, match="hop_fs cannot exceed"):
        short_time_power_spectrum(t, signal, window_fs=20.0, hop_fs=100.0)


def test_max_frequency_truncates_the_axis():
    t, signal = _switching_signal(1200.0, 1200.0, n=256)
    _, frequencies, power = short_time_power_spectrum(
        t, signal, window_fs=50.0, hop_fs=25.0, max_frequency_cm1=2000.0
    )
    assert frequencies.max() <= 2000.0
    assert power.shape[1] == frequencies.size


# --------------------------------------------------------------------------
# Re-analysing a saved run
# --------------------------------------------------------------------------

def _write_scattering_file(path, n_times=64, n_q=5):
    times = np.arange(n_times) * 0.5
    q = np.linspace(0.0, 8.0, n_q)
    rows = []
    for i, t in enumerate(times):
        for j, qq in enumerate(q):
            elastic = 100.0 + np.sin(2.0 * np.pi * t / 20.0) * (j + 1)
            inelastic = 0.5 * j
            rows.append((t, qq, elastic, inelastic, elastic + inelastic))
    np.savetxt(path, np.asarray(rows), header='time_fs q_A^-1 elastic inelastic total')
    return times, q


def test_a_saved_run_round_trips(tmp_path):
    path = str(tmp_path / "demo_time_dependent_scattering_form_factors.dat")
    times, q = _write_scattering_file(path)

    data = read_time_dependent_scattering(path)

    assert data["time_fs"] == pytest.approx(times)
    assert data["q_Ainv"] == pytest.approx(q)
    for channel in ("elastic", "inelastic", "total"):
        assert data[channel].shape == (times.size, q.size)
    assert data["total"] == pytest.approx(data["elastic"] + data["inelastic"])


def test_a_file_with_the_wrong_shape_is_rejected(tmp_path):
    path = str(tmp_path / "bad.dat")
    np.savetxt(path, np.zeros((10, 3)))
    with pytest.raises(ValueError, match="does not look like"):
        read_time_dependent_scattering(path)


def test_an_incomplete_grid_is_rejected(tmp_path):
    path = str(tmp_path / "ragged.dat")
    np.savetxt(path, np.array([[0.0, 1.0, 1, 1, 2], [0.0, 2.0, 1, 1, 2],
                               [1.0, 1.0, 1, 1, 2]]))
    with pytest.raises(ValueError, match="not a complete time-by-q grid"):
        read_time_dependent_scattering(path)


def test_short_time_spectrum_reads_straight_from_a_file(tmp_path):
    path = str(tmp_path / "demo_time_dependent_scattering_form_factors.dat")
    _write_scattering_file(path, n_times=128)

    spectrum = short_time_scattering_spectrum(path, window_fs=20.0, hop_fs=5.0)

    for channel in ("elastic", "inelastic", "total"):
        assert spectrum[channel].shape == (
            spectrum["time_fs"].size, spectrum["frequency_cm1"].size
        )
    # The injected 20 fs period is about 1668 cm^-1.
    f = spectrum["frequency_cm1"]
    band = f > 200.0
    peak = f[band][np.argmax(spectrum["elastic"][1][band])]
    assert peak == pytest.approx(1668.0, rel=0.25)


def test_q_range_restricts_the_integration(tmp_path):
    path = str(tmp_path / "demo_time_dependent_scattering_form_factors.dat")
    _write_scattering_file(path, n_times=128, n_q=9)

    everything = short_time_scattering_spectrum(path, window_fs=20.0, hop_fs=5.0)
    narrow = short_time_scattering_spectrum(
        path, window_fs=20.0, hop_fs=5.0, q_range=(0.0, 2.0)
    )

    assert narrow["q_Ainv"].size < everything["q_Ainv"].size
    assert narrow["elastic"].max() < everything["elastic"].max()


def test_an_empty_q_range_is_rejected(tmp_path):
    path = str(tmp_path / "demo_time_dependent_scattering_form_factors.dat")
    _write_scattering_file(path, n_times=64)
    with pytest.raises(ValueError, match="no q points inside"):
        short_time_scattering_spectrum(path, window_fs=10.0, q_range=(50.0, 60.0))
