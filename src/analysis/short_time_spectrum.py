"""Energy-calibrated short-time Fourier analysis for MD trajectories.

Implements Zhang, Xu, and Truhlar, J. Phys. Chem. A 126, 3006-3014
(2022), DOI: 10.1021/acs.jpca.1c09905. This module deliberately contains
no trajectory-propagation logic.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from utils.constants import (
    BOHR_TO_ANGSTROM,
    FS_TO_AU_TIME,
    HARTREE_TO_CM1,
    HARTREE_TO_KJMOL,
)


CYCLES_PER_FS_TO_CM1 = 2.0 * np.pi * HARTREE_TO_CM1 / FS_TO_AU_TIME


@dataclass
class ShortTimeSpectrum:
    """Energy-normalized one-sided short-time spectrum."""

    time_fs: np.ndarray
    frequency_cm1: np.ndarray
    spectrum_hartree_per_cm1: np.ndarray
    kinetic_energy_hartree: np.ndarray
    dominant_frequency_cm1: np.ndarray
    band_energy_hartree: dict[str, np.ndarray]
    band_peak_frequency_cm1: dict[str, np.ndarray]
    band_centroid_frequency_cm1: dict[str, np.ndarray]
    band_width_frequency_cm1: dict[str, np.ndarray]
    band_energy_sensitivity_hartree: dict[str, np.ndarray]
    band_bounds_cm1: dict[str, tuple[float, float]]
    atom_indices: tuple[int, ...]
    dt_fs: float
    window_fs: float
    hop_fs: float
    window: str
    nfft: int
    frequency_bin_width_cm1: float
    nominal_resolution_cm1: float
    band_sensitivity_cm1: float
    peak_estimator: str
    internal_energy_hartree: np.ndarray | None = None
    virial_ratio: np.ndarray | None = None
    virial_residual_hartree: np.ndarray | None = None
    activity_fraction: np.ndarray | None = None
    output_files: dict[str, str] = field(default_factory=dict)


@dataclass
class DynamicFragmentTracking:
    """Frame-by-frame bond connectivity for a reactive trajectory."""

    time_fs: np.ndarray
    fragments: tuple[tuple[tuple[int, ...], ...], ...]
    bond_distance_angstrom: dict[str, np.ndarray]
    bond_active: dict[str, np.ndarray]
    bond_definitions: dict[str, tuple[int, int, float, float]]
    output_files: dict[str, str] = field(default_factory=dict)


@dataclass
class ShortTimeChannelAnalysis:
    """STFT results for several fixed atom channels plus reactive identities."""

    spectra: dict[str, ShortTimeSpectrum]
    fragment_tracking: DynamicFragmentTracking | None = None
    output_files: dict[str, str] = field(default_factory=dict)


def _validate_histories(positions, velocities, masses):
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    masses = np.asarray(masses, dtype=float).reshape(-1)
    if positions.ndim != 2 or velocities.ndim != 2:
        raise ValueError("positions and velocities must have shape (samples, 3N)")
    if positions.shape != velocities.shape:
        raise ValueError("position and velocity histories must have identical shapes")
    if masses.size == 0 or positions.shape[1] != 3 * masses.size:
        raise ValueError("history width must equal three times the atom count")
    if np.any(~np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("masses must be finite and strictly positive")
    if not np.all(np.isfinite(positions)):
        raise ValueError("position history contains non-finite values")
    if not np.all(np.isfinite(velocities)):
        raise ValueError("velocity history contains non-finite values")
    return positions, velocities, masses


def _selected_atoms(atom_indices, natoms):
    if atom_indices is None:
        return np.arange(natoms, dtype=int)
    indices = np.asarray(atom_indices)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("atom_indices must be a non-empty one-dimensional sequence")
    if indices.dtype.kind not in "iu":
        raise ValueError("atom_indices must contain integers")
    if np.any(indices < 0) or np.any(indices >= natoms):
        raise ValueError("atom_indices contains an out-of-range atom index")
    if np.unique(indices).size != indices.size:
        raise ValueError("atom_indices must not contain duplicates")
    return indices.astype(int)


def project_translation_and_rotation(positions, velocities, masses):
    """Remove instantaneous COM translation and best-fit rigid rotation."""

    positions, velocities, masses = _validate_histories(
        positions, velocities, masses
    )
    nsamples = positions.shape[0]
    natoms = masses.size
    q_history = positions.reshape(nsamples, natoms, 3)
    v_history = velocities.reshape(nsamples, natoms, 3)
    projected = np.empty_like(v_history)
    total_mass = float(np.sum(masses))
    identity = np.eye(3)

    for iframe, (coordinates, velocity) in enumerate(zip(q_history, v_history)):
        center = np.sum(masses[:, None] * coordinates, axis=0) / total_mass
        com_velocity = np.sum(masses[:, None] * velocity, axis=0) / total_mass
        relative_coordinates = coordinates - center
        internal_velocity = velocity - com_velocity

        inertia = np.zeros((3, 3))
        angular_momentum = np.zeros(3)
        for mass, coordinate, atom_velocity in zip(
            masses, relative_coordinates, internal_velocity
        ):
            inertia += mass * (
                np.dot(coordinate, coordinate) * identity
                - np.outer(coordinate, coordinate)
            )
            angular_momentum += mass * np.cross(coordinate, atom_velocity)

        angular_velocity = (
            np.linalg.pinv(inertia, rcond=1.0e-12) @ angular_momentum
        )
        projected[iframe] = internal_velocity - np.cross(
            angular_velocity, relative_coordinates
        )
    return projected.reshape(velocities.shape)


def mass_weighted_vibrational_velocity(
    positions, velocities, masses, *, atom_indices=None
):
    """Select atoms, project rigid motion, and mass-weight their velocities."""

    positions, velocities, masses = _validate_histories(
        positions, velocities, masses
    )
    indices = _selected_atoms(atom_indices, masses.size)
    coordinate_indices = (
        3 * indices[:, None] + np.arange(3, dtype=int)[None, :]
    ).reshape(-1)
    selected_masses = masses[indices]
    projected = project_translation_and_rotation(
        positions[:, coordinate_indices],
        velocities[:, coordinate_indices],
        selected_masses,
    )
    return (
        projected * np.sqrt(np.repeat(selected_masses, 3))[None, :],
        tuple(indices),
    )


def _sampling_times(nsamples, dt_fs, times_fs):
    if times_fs is None:
        if dt_fs is None or not np.isfinite(dt_fs) or dt_fs <= 0.0:
            raise ValueError("dt_fs must be finite and positive")
        dt_fs = float(dt_fs)
        return np.arange(nsamples, dtype=float) * dt_fs, dt_fs

    times = np.asarray(times_fs, dtype=float).reshape(-1)
    if times.size != nsamples or not np.all(np.isfinite(times)):
        raise ValueError("times_fs must be finite and match the history length")
    differences = np.diff(times)
    if differences.size == 0 or np.any(differences <= 0.0):
        raise ValueError("at least two strictly increasing times are required")
    inferred_dt = float(np.mean(differences))
    if not np.allclose(
        differences,
        inferred_dt,
        rtol=1.0e-7,
        atol=max(1.0e-12, inferred_dt * 1.0e-10),
    ):
        raise ValueError("short-time Fourier analysis requires uniform sampling")
    if dt_fs is not None and not np.isclose(
        float(dt_fs), inferred_dt, rtol=1.0e-7, atol=1.0e-12
    ):
        raise ValueError("dt_fs is inconsistent with times_fs")
    return times, inferred_dt


def _window_values(window, sample_count):
    if isinstance(window, str):
        name = window.lower()
        if name in {"boxcar", "rectangular", "rectangle"}:
            values, label = np.ones(sample_count), "boxcar"
        elif name in {"hann", "hanning"}:
            values, label = np.hanning(sample_count), "hann"
        elif name == "hamming":
            values, label = np.hamming(sample_count), "hamming"
        elif name == "blackman":
            values, label = np.blackman(sample_count), "blackman"
        else:
            raise ValueError(
                "window must be boxcar, hann, hamming, blackman, or an array"
            )
    else:
        values = np.asarray(window, dtype=float)
        if values.shape != (sample_count,) or not np.all(np.isfinite(values)):
            raise ValueError("a custom window must be finite and match the window")
        label = "custom"

    mean_square = float(np.mean(values * values))
    if mean_square <= np.finfo(float).eps:
        raise ValueError("window must have non-zero squared norm")
    # This is the discrete counterpart of integral g(t)^2 dt = 1.
    return values / np.sqrt(mean_square), label


def _validate_frequency_bands(frequency_bands, nyquist_cm1):
    if frequency_bands is None:
        return {}
    validated = {}
    for name, bounds in frequency_bands.items():
        if not isinstance(name, str) or not name or len(bounds) != 2:
            raise ValueError("frequency bands need a name and two bounds")
        lower, upper = map(float, bounds)
        if (
            not np.isfinite(lower)
            or not np.isfinite(upper)
            or lower < 0.0
            or upper <= lower
        ):
            raise ValueError("frequency bands require 0 <= lower < upper")
        if upper > nyquist_cm1 * (1.0 + 1.0e-12):
            raise ValueError(
                f"frequency band {name!r} exceeds the Nyquist frequency "
                f"of {nyquist_cm1:.6g} cm^-1"
            )
        validated[name] = (lower, min(upper, nyquist_cm1))
    return validated


def _one_sided_weights(nfft):
    weights = np.ones(nfft // 2 + 1)
    weights[1:-1 if nfft % 2 == 0 else None] = 2.0
    return weights


def _peak_trace(spectrum, frequencies, mask=None, *, estimator="parabolic"):
    """Track a spectral maximum, optionally refining it between FFT bins."""

    if estimator not in {"discrete", "parabolic"}:
        raise ValueError("peak_estimator must be 'discrete' or 'parabolic'")
    if mask is None:
        mask = np.ones(frequencies.size, dtype=bool)
        if mask.size > 1:
            mask[0] = False
    selected_indices = np.flatnonzero(mask)
    if selected_indices.size == 0:
        return np.full(spectrum.shape[0], np.nan)

    selected_spectrum = spectrum[:, selected_indices]
    local_peaks = np.argmax(selected_spectrum, axis=1)
    peak_indices = selected_indices[local_peaks]
    result = frequencies[peak_indices].astype(float)
    maxima = spectrum[np.arange(spectrum.shape[0]), peak_indices]

    if estimator == "parabolic" and frequencies.size > 2:
        spacing = float(frequencies[1] - frequencies[0])
        for iframe, index in enumerate(peak_indices):
            if index == 0 or index == frequencies.size - 1:
                continue
            left, center, right = spectrum[iframe, index - 1 : index + 2]
            curvature = left - 2.0 * center + right
            if curvature >= 0.0 or abs(curvature) <= np.finfo(float).eps:
                continue
            displacement = 0.5 * (left - right) / curvature
            result[iframe] += float(np.clip(displacement, -1.0, 1.0)) * spacing

    result[maxima <= 0.0] = np.nan
    return result


def _frequency_cells(frequencies):
    """Return non-overlapping cells represented by each one-sided FFT bin."""

    edges = np.empty(frequencies.size + 1)
    edges[0] = 0.0
    edges[-1] = frequencies[-1]
    edges[1:-1] = 0.5 * (frequencies[:-1] + frequencies[1:])
    return edges


def _band_statistics(spectrum, frequencies, bin_width, lower, upper):
    """Integrate a band with fractional boundary-bin contributions."""

    edges = _frequency_cells(frequencies)
    cell_width = np.diff(edges)
    overlap_lower = np.maximum(edges[:-1], lower)
    overlap_upper = np.minimum(edges[1:], upper)
    overlap = np.maximum(0.0, overlap_upper - overlap_lower)
    density_scale = np.divide(
        bin_width,
        cell_width,
        out=np.zeros_like(cell_width),
        where=cell_width > 0.0,
    )
    energy_weights = overlap * density_scale
    energy = spectrum @ energy_weights

    first_moment_weights = (
        0.5
        * (overlap_upper**2 - overlap_lower**2)
        * density_scale
        * (overlap > 0.0)
    )
    second_moment_weights = (
        (overlap_upper**3 - overlap_lower**3)
        / 3.0
        * density_scale
        * (overlap > 0.0)
    )
    first_moment = spectrum @ first_moment_weights
    second_moment = spectrum @ second_moment_weights
    centroid = np.divide(
        first_moment,
        energy,
        out=np.full_like(energy, np.nan),
        where=energy > 0.0,
    )
    variance = np.divide(
        second_moment,
        energy,
        out=np.full_like(energy, np.nan),
        where=energy > 0.0,
    ) - centroid**2
    width = np.sqrt(np.maximum(variance, 0.0))
    mask = overlap > 0.0
    return energy, centroid, width, mask


def _window_average(values, starts, window_samples, window_values):
    values = np.asarray(values, dtype=float)
    weights = window_values * window_values / window_samples
    return np.asarray(
        [np.dot(values[start : start + window_samples], weights) for start in starts]
    )


def _history_values(values, nsamples, name, *, allow_scalar=True):
    array = np.asarray(values, dtype=float)
    if allow_scalar and array.ndim == 0:
        array = np.full(nsamples, float(array))
    else:
        array = array.reshape(-1)
    if array.shape != (nsamples,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite and contain one value per sample")
    return array


def short_time_fourier_spectrum(
    positions,
    velocities,
    masses,
    *,
    dt_fs=None,
    times_fs=None,
    window_fs=200.0,
    hop_fs=1.0,
    atom_indices=None,
    frequency_bands=None,
    max_frequency_cm1=None,
    window="boxcar",
    nfft=None,
    peak_estimator="parabolic",
    band_sensitivity_cm1=None,
    internal_energy_hartree=None,
    potential_energy_hartree=None,
    reference_potential_hartree=0.0,
    activity_mask=None,
):
    """Compute the paper's energy-normalized short-time spectrum.

    The density is in Hartree per cm^-1 and integrates to the window-averaged
    vibrational kinetic energy. If an internal-energy history, or a potential
    history and its reference zero, is supplied, virial_ratio reports
    2 <T_vib> / <E_internal>. This ratio should approach one only for a
    stationary bound subsystem; it is not expected to do so during a collision.
    """

    weighted_velocity, selected = mass_weighted_vibrational_velocity(
        positions, velocities, masses, atom_indices=atom_indices
    )
    nsamples = weighted_velocity.shape[0]
    times, dt_fs = _sampling_times(nsamples, dt_fs, times_fs)

    for value, name in ((window_fs, "window_fs"), (hop_fs, "hop_fs")):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    window_samples = max(2, int(round(float(window_fs) / dt_fs)))
    hop_samples = max(1, int(round(float(hop_fs) / dt_fs)))
    if window_samples > nsamples:
        raise ValueError(
            f"window_fs requires {window_samples} samples, but only "
            f"{nsamples} are available"
        )
    if hop_samples > window_samples:
        raise ValueError("hop_fs cannot exceed the actual window length")

    if nfft is None:
        nfft = window_samples
    elif (
        isinstance(nfft, (bool, np.bool_))
        or not isinstance(nfft, (int, np.integer))
        or nfft < window_samples
    ):
        raise ValueError("nfft must be an integer at least as large as the window")
    nfft = int(nfft)

    window_values, window_label = _window_values(window, window_samples)
    frequencies = np.fft.rfftfreq(nfft, d=dt_fs) * CYCLES_PER_FS_TO_CM1
    bin_width = CYCLES_PER_FS_TO_CM1 / (nfft * dt_fs)
    nominal_resolution = CYCLES_PER_FS_TO_CM1 / (window_samples * dt_fs)
    if band_sensitivity_cm1 is None:
        band_sensitivity_cm1 = 0.5 * nominal_resolution
    if (
        not np.isfinite(band_sensitivity_cm1)
        or float(band_sensitivity_cm1) < 0.0
    ):
        raise ValueError("band_sensitivity_cm1 must be finite and non-negative")
    band_sensitivity_cm1 = float(band_sensitivity_cm1)

    weights = _one_sided_weights(nfft)
    starts = np.arange(0, nsamples - window_samples + 1, hop_samples)
    spectrum = np.empty((starts.size, frequencies.size))
    normalization = 0.5 / (window_samples * nfft)

    for iwindow, start in enumerate(starts):
        segment = (
            weighted_velocity[start : start + window_samples]
            * window_values[:, None]
        )
        transformed = np.fft.rfft(segment, n=nfft, axis=0)
        energy_per_bin = (
            normalization
            * weights
            * np.sum(np.abs(transformed) ** 2, axis=1)
        )
        spectrum[iwindow] = energy_per_bin / bin_width

    kinetic_energy = np.sum(spectrum, axis=1) * bin_width
    centers = 0.5 * (times[starts] + times[starts + window_samples - 1])
    bands = _validate_frequency_bands(frequency_bands, float(frequencies[-1]))
    band_energy = {}
    band_peaks = {}
    band_centroids = {}
    band_widths = {}
    band_sensitivity = {}
    for name, (lower, upper) in bands.items():
        energy, centroid, width, mask = _band_statistics(
            spectrum, frequencies, bin_width, lower, upper
        )
        peak = _peak_trace(
            spectrum, frequencies, mask, estimator=peak_estimator
        )
        finite_peak = np.isfinite(peak)
        peak[finite_peak] = np.clip(peak[finite_peak], lower, upper)
        band_energy[name] = energy
        band_peaks[name] = peak
        band_centroids[name] = centroid
        band_widths[name] = width

        narrow_lower = min(upper, lower + band_sensitivity_cm1)
        narrow_upper = max(narrow_lower, upper - band_sensitivity_cm1)
        narrow_energy = _band_statistics(
            spectrum,
            frequencies,
            bin_width,
            narrow_lower,
            narrow_upper,
        )[0]
        expanded_energy = _band_statistics(
            spectrum,
            frequencies,
            bin_width,
            max(0.0, lower - band_sensitivity_cm1),
            min(float(frequencies[-1]), upper + band_sensitivity_cm1),
        )[0]
        band_sensitivity[name] = np.column_stack(
            (narrow_energy, expanded_energy)
        )

    dominant_frequency = _peak_trace(
        spectrum, frequencies, estimator=peak_estimator
    )

    averaged_internal_energy = None
    virial_ratio = None
    virial_residual = None
    if internal_energy_hartree is not None and potential_energy_hartree is not None:
        raise ValueError(
            "provide internal_energy_hartree or potential_energy_hartree, not both"
        )
    if internal_energy_hartree is not None:
        internal_history = _history_values(
            internal_energy_hartree, nsamples, "internal_energy_hartree"
        )
    elif potential_energy_hartree is not None:
        potential_history = _history_values(
            potential_energy_hartree, nsamples, "potential_energy_hartree"
        )
        reference_history = _history_values(
            reference_potential_hartree,
            nsamples,
            "reference_potential_hartree",
        )
        instantaneous_kinetic = 0.5 * np.sum(weighted_velocity**2, axis=1)
        internal_history = (
            instantaneous_kinetic + potential_history - reference_history
        )
    else:
        internal_history = None

    if internal_history is not None:
        averaged_internal_energy = _window_average(
            internal_history, starts, window_samples, window_values
        )
        virial_residual = 2.0 * kinetic_energy - averaged_internal_energy
        scale = max(1.0, float(np.max(np.abs(averaged_internal_energy))))
        nonzero = np.abs(averaged_internal_energy) > 64.0 * np.finfo(float).eps * scale
        virial_ratio = np.divide(
            2.0 * kinetic_energy,
            averaged_internal_energy,
            out=np.full_like(kinetic_energy, np.nan),
            where=nonzero,
        )

    activity_fraction = None
    if activity_mask is not None:
        activity_history = _history_values(
            activity_mask, nsamples, "activity_mask"
        )
        if np.any((activity_history < 0.0) | (activity_history > 1.0)):
            raise ValueError("activity_mask values must lie between zero and one")
        activity_fraction = _window_average(
            activity_history, starts, window_samples, window_values
        )

    if max_frequency_cm1 is not None:
        if not np.isfinite(max_frequency_cm1) or max_frequency_cm1 <= 0.0:
            raise ValueError("max_frequency_cm1 must be finite and positive")
        visible = frequencies <= float(max_frequency_cm1)
        frequencies = frequencies[visible]
        spectrum = spectrum[:, visible]

    return ShortTimeSpectrum(
        time_fs=centers,
        frequency_cm1=frequencies,
        spectrum_hartree_per_cm1=spectrum,
        kinetic_energy_hartree=kinetic_energy,
        dominant_frequency_cm1=dominant_frequency,
        band_energy_hartree=band_energy,
        band_peak_frequency_cm1=band_peaks,
        band_centroid_frequency_cm1=band_centroids,
        band_width_frequency_cm1=band_widths,
        band_energy_sensitivity_hartree=band_sensitivity,
        band_bounds_cm1=bands,
        atom_indices=selected,
        dt_fs=dt_fs,
        window_fs=window_samples * dt_fs,
        hop_fs=hop_samples * dt_fs,
        window=window_label,
        nfft=nfft,
        frequency_bin_width_cm1=bin_width,
        nominal_resolution_cm1=nominal_resolution,
        band_sensitivity_cm1=band_sensitivity_cm1,
        peak_estimator=peak_estimator,
        internal_energy_hartree=averaged_internal_energy,
        virial_ratio=virial_ratio,
        virial_residual_hartree=virial_residual,
        activity_fraction=activity_fraction,
    )


def centered_moving_average(values, points):
    """Return a centered moving average with shortened edge windows."""

    values = np.asarray(values, dtype=float)
    if (
        isinstance(points, (bool, np.bool_))
        or not isinstance(points, (int, np.integer))
        or points < 1
    ):
        raise ValueError("points must be a positive integer")
    if values.shape[0] == 0:
        raise ValueError("values must not be empty")
    points = min(int(points), values.shape[0])
    if values.ndim == 2:
        return np.column_stack(
            [centered_moving_average(values[:, i], points)
             for i in range(values.shape[1])]
        )
    if values.ndim != 1:
        raise ValueError("values must be one- or two-dimensional")
    kernel = np.ones(int(points))
    return (
        np.convolve(values, kernel, mode="same")
        / np.convolve(np.ones(values.size), kernel, mode="same")
    )


def save_short_time_spectrum(
    result,
    output_prefix,
    *,
    write_text=True,
    plot=True,
    energy_unit="kJ/mol",
    moving_average_fs=None,
):
    """Save NPZ/text results and spectrogram/band-energy plots."""

    if not isinstance(result, ShortTimeSpectrum):
        raise TypeError("result must be a ShortTimeSpectrum")
    factors = {"kJ/mol": HARTREE_TO_KJMOL, "Hartree": 1.0}
    if energy_unit not in factors:
        raise ValueError("energy_unit must be 'kJ/mol' or 'Hartree'")
    energy_factor = factors[energy_unit]

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    band_names = list(result.band_energy_hartree)
    band_energies = (
        np.column_stack([result.band_energy_hartree[n] for n in band_names])
        if band_names else np.empty((result.time_fs.size, 0))
    )
    band_peaks = (
        np.column_stack(
            [result.band_peak_frequency_cm1[n] for n in band_names]
        )
        if band_names else np.empty((result.time_fs.size, 0))
    )
    band_centroids = (
        np.column_stack(
            [result.band_centroid_frequency_cm1[n] for n in band_names]
        )
        if band_names else np.empty((result.time_fs.size, 0))
    )
    band_widths = (
        np.column_stack(
            [result.band_width_frequency_cm1[n] for n in band_names]
        )
        if band_names else np.empty((result.time_fs.size, 0))
    )
    band_sensitivity = (
        np.stack(
            [result.band_energy_sensitivity_hartree[n] for n in band_names],
            axis=1,
        )
        if band_names else np.empty((result.time_fs.size, 0, 2))
    )
    band_bounds = np.asarray(
        [result.band_bounds_cm1[n] for n in band_names], dtype=float
    ).reshape((-1, 2))
    empty = np.empty(0)

    npz_path = Path(str(prefix) + ".npz")
    np.savez_compressed(
        npz_path,
        time_fs=result.time_fs,
        frequency_cm1=result.frequency_cm1,
        spectrum_hartree_per_cm1=result.spectrum_hartree_per_cm1,
        kinetic_energy_hartree=result.kinetic_energy_hartree,
        dominant_frequency_cm1=result.dominant_frequency_cm1,
        band_names=np.asarray(band_names),
        band_energy_hartree=band_energies,
        band_peak_frequency_cm1=band_peaks,
        band_centroid_frequency_cm1=band_centroids,
        band_width_frequency_cm1=band_widths,
        band_energy_sensitivity_hartree=band_sensitivity,
        band_bounds_cm1=band_bounds,
        atom_indices=np.asarray(result.atom_indices),
        dt_fs=np.asarray(result.dt_fs),
        window_fs=np.asarray(result.window_fs),
        hop_fs=np.asarray(result.hop_fs),
        window=np.asarray(result.window),
        nfft=np.asarray(result.nfft),
        frequency_bin_width_cm1=np.asarray(result.frequency_bin_width_cm1),
        nominal_resolution_cm1=np.asarray(result.nominal_resolution_cm1),
        band_sensitivity_cm1=np.asarray(result.band_sensitivity_cm1),
        peak_estimator=np.asarray(result.peak_estimator),
        internal_energy_hartree=(
            empty if result.internal_energy_hartree is None
            else result.internal_energy_hartree
        ),
        virial_ratio=empty if result.virial_ratio is None else result.virial_ratio,
        virial_residual_hartree=(
            empty if result.virial_residual_hartree is None
            else result.virial_residual_hartree
        ),
        activity_fraction=(
            empty if result.activity_fraction is None else result.activity_fraction
        ),
    )
    files = {"npz": str(npz_path)}

    if write_text:
        spectrum_path = Path(str(prefix) + "_spectrum.dat")
        time_grid, frequency_grid = np.meshgrid(
            result.time_fs, result.frequency_cm1, indexing="ij"
        )
        np.savetxt(
            spectrum_path,
            np.column_stack(
                (
                    time_grid.ravel(),
                    frequency_grid.ravel(),
                    result.spectrum_hartree_per_cm1.ravel(),
                )
            ),
            fmt="%.10e",
            header="time_fs frequency_cm-1 spectrum_Eh_per_cm-1",
        )
        files["spectrum_data"] = str(spectrum_path)

        energy_path = Path(str(prefix) + "_band_energies.dat")
        columns = [
            result.time_fs,
            result.kinetic_energy_hartree,
            result.dominant_frequency_cm1,
        ]
        header = [
            "time_fs",
            "kinetic_energy_Eh",
            "dominant_frequency_cm-1",
        ]
        if result.internal_energy_hartree is not None:
            columns.extend(
                [
                    result.internal_energy_hartree,
                    result.virial_ratio,
                    result.virial_residual_hartree,
                ]
            )
            header.extend(
                ["internal_energy_Eh", "virial_2T_over_E", "virial_residual_Eh"]
            )
        if result.activity_fraction is not None:
            columns.append(result.activity_fraction)
            header.append("activity_fraction")
        for name in band_names:
            columns.extend(
                [
                    result.band_energy_hartree[name],
                    result.band_peak_frequency_cm1[name],
                    result.band_centroid_frequency_cm1[name],
                    result.band_width_frequency_cm1[name],
                    result.band_energy_sensitivity_hartree[name][:, 0],
                    result.band_energy_sensitivity_hartree[name][:, 1],
                ]
            )
            header.extend(
                [
                    f"{name}_energy_Eh",
                    f"{name}_peak_cm-1",
                    f"{name}_centroid_cm-1",
                    f"{name}_width_cm-1",
                    f"{name}_energy_narrow_Eh",
                    f"{name}_energy_expanded_Eh",
                ]
            )
        np.savetxt(
            energy_path,
            np.column_stack(columns),
            fmt="%.10e",
            header=" ".join(header),
        )
        files["band_data"] = str(energy_path)

    if plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "Plotting requires matplotlib; install Smite's analysis extra"
            ) from exc

        heatmap_path = Path(str(prefix) + "_spectrogram.png")
        figure, axis = plt.subplots(figsize=(9, 5))
        image = axis.pcolormesh(
            result.time_fs,
            result.frequency_cm1,
            result.spectrum_hartree_per_cm1.T * energy_factor,
            shading="auto",
        )
        axis.set(
            xlabel="Time (fs)",
            ylabel="Frequency (cm$^{-1}$)",
            title=f"Short-time spectrum ({result.window_fs:g} fs window)",
        )
        figure.colorbar(
            image,
            ax=axis,
            label=f"Kinetic-energy density ({energy_unit} per cm$^{{-1}}$)",
        )
        figure.tight_layout()
        figure.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        files["spectrogram_plot"] = str(heatmap_path)

        if band_names:
            band_plot_path = Path(str(prefix) + "_band_energies.png")
            figure, axis = plt.subplots(figsize=(9, 5))
            smoothing_points = None
            if moving_average_fs is not None:
                if not np.isfinite(moving_average_fs) or moving_average_fs <= 0:
                    raise ValueError("moving_average_fs must be finite and positive")
                smoothing_points = max(
                    1, int(round(moving_average_fs / result.hop_fs))
                )
            band_sum = np.zeros(result.time_fs.size)
            for name in band_names:
                values = result.band_energy_hartree[name] * energy_factor
                band_sum += values
                axis.plot(
                    result.time_fs,
                    values,
                    alpha=0.35 if smoothing_points else 1.0,
                    label=name,
                )
                sensitivity = (
                    result.band_energy_sensitivity_hartree[name] * energy_factor
                )
                axis.fill_between(
                    result.time_fs,
                    sensitivity[:, 0],
                    sensitivity[:, 1],
                    alpha=0.12,
                )
                if smoothing_points:
                    axis.plot(
                        result.time_fs,
                        centered_moving_average(values, smoothing_points),
                        label=f"{name} moving average",
                    )
            axis.plot(
                result.time_fs,
                band_sum,
                "--",
                color="black",
                label="sum of bands",
            )
            axis.plot(
                result.time_fs,
                result.kinetic_energy_hartree * energy_factor,
                ":",
                color="gray",
                label="all frequencies",
            )
            axis.set(
                xlabel="Time (fs)",
                ylabel=f"Short-time averaged kinetic energy ({energy_unit})",
                title="Vibrational energy by frequency interval",
            )
            axis.grid(True)
            axis.legend()
            figure.tight_layout()
            figure.savefig(band_plot_path, dpi=300, bbox_inches="tight")
            plt.close(figure)
            files["band_plot"] = str(band_plot_path)
        if result.virial_ratio is not None:
            virial_path = Path(str(prefix) + "_virial_ratio.png")
            figure, axis = plt.subplots(figsize=(9, 4))
            axis.plot(
                result.time_fs,
                result.virial_ratio,
                label=r"$2\langle T_{vib}\rangle/\langle E_{internal}\rangle$",
            )
            axis.axhline(1.0, color="black", linestyle="--", label="virial limit")
            axis.set(
                xlabel="Time (fs)",
                ylabel="Virial ratio",
                title="Bound-mode virial consistency",
            )
            axis.grid(True)
            axis.legend()
            figure.tight_layout()
            figure.savefig(virial_path, dpi=300, bbox_inches="tight")
            plt.close(figure)
            files["virial_plot"] = str(virial_path)

        if result.activity_fraction is not None:
            activity_path = Path(str(prefix) + "_activity.png")
            figure, axis = plt.subplots(figsize=(9, 3.5))
            axis.plot(result.time_fs, result.activity_fraction)
            axis.set(
                xlabel="Time (fs)",
                ylabel="Window activity fraction",
                title="Dynamic fragment membership in each STFT window",
                ylim=(-0.02, 1.02),
            )
            axis.grid(True)
            figure.tight_layout()
            figure.savefig(activity_path, dpi=300, bbox_inches="tight")
            plt.close(figure)
            files["activity_plot"] = str(activity_path)

    result.output_files.update(files)
    return files


def _validated_dynamic_bonds(dynamic_bonds, natoms):
    if not isinstance(dynamic_bonds, dict) or not dynamic_bonds:
        raise ValueError("dynamic_bonds must be a non-empty mapping")
    validated = {}
    seen_pairs = set()
    for name, definition in dynamic_bonds.items():
        if not isinstance(name, str) or not name or len(definition) not in {3, 4}:
            raise ValueError(
                "each dynamic bond needs a name and (i, j, form[, break])"
            )
        atom_i, atom_j = definition[:2]
        if (
            isinstance(atom_i, (bool, np.bool_))
            or isinstance(atom_j, (bool, np.bool_))
            or not isinstance(atom_i, (int, np.integer))
            or not isinstance(atom_j, (int, np.integer))
        ):
            raise ValueError("dynamic bond atom indices must be integers")
        atom_i, atom_j = int(atom_i), int(atom_j)
        if atom_i == atom_j or min(atom_i, atom_j) < 0 or max(atom_i, atom_j) >= natoms:
            raise ValueError("dynamic bond contains invalid atom indices")
        pair = frozenset((atom_i, atom_j))
        if pair in seen_pairs:
            raise ValueError("dynamic_bonds contains the same atom pair twice")
        seen_pairs.add(pair)
        formation = float(definition[2])
        breaking = formation if len(definition) == 3 else float(definition[3])
        if (
            not np.isfinite(formation)
            or not np.isfinite(breaking)
            or formation <= 0.0
            or breaking < formation
        ):
            raise ValueError(
                "dynamic bond thresholds require 0 < formation <= breaking"
            )
        validated[name] = (atom_i, atom_j, formation, breaking)
    return validated


def track_dynamic_fragments(
    positions,
    *,
    dynamic_bonds,
    dt_fs=None,
    times_fs=None,
    position_unit="bohr",
):
    """Track bonded components with separate formation and breaking thresholds."""

    positions = np.asarray(positions, dtype=float)
    if (
        positions.ndim != 2
        or positions.shape[0] < 2
        or positions.shape[1] == 0
        or positions.shape[1] % 3
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError("positions must be finite with shape (samples, 3N)")
    nsamples, width = positions.shape
    natoms = width // 3
    times, _dt_fs = _sampling_times(nsamples, dt_fs, times_fs)
    if position_unit == "bohr":
        distance_factor = BOHR_TO_ANGSTROM
    elif position_unit == "angstrom":
        distance_factor = 1.0
    else:
        raise ValueError("position_unit must be 'bohr' or 'angstrom'")

    definitions = _validated_dynamic_bonds(dynamic_bonds, natoms)
    coordinates = positions.reshape(nsamples, natoms, 3)
    distances = {}
    active = {}
    for name, (atom_i, atom_j, formation, breaking) in definitions.items():
        separation = np.linalg.norm(
            coordinates[:, atom_i] - coordinates[:, atom_j], axis=1
        ) * distance_factor
        state = False
        state_history = np.empty(nsamples, dtype=bool)
        for iframe, distance in enumerate(separation):
            state = distance <= (breaking if state else formation)
            state_history[iframe] = state
        distances[name] = separation
        active[name] = state_history

    fragments = []
    for iframe in range(nsamples):
        parent = np.arange(natoms)

        def root(atom):
            while parent[atom] != atom:
                parent[atom] = parent[parent[atom]]
                atom = parent[atom]
            return atom

        for name, (atom_i, atom_j, _formation, _breaking) in definitions.items():
            if active[name][iframe]:
                root_i, root_j = root(atom_i), root(atom_j)
                if root_i != root_j:
                    parent[root_j] = root_i
        components = {}
        for atom in range(natoms):
            components.setdefault(root(atom), []).append(atom)
        frame_fragments = tuple(
            sorted((tuple(component) for component in components.values()))
        )
        fragments.append(frame_fragments)

    return DynamicFragmentTracking(
        time_fs=times,
        fragments=tuple(fragments),
        bond_distance_angstrom=distances,
        bond_active=active,
        bond_definitions=definitions,
    )


def _fragment_activity(tracking, atom_indices):
    selected = set(map(int, atom_indices))
    if len(selected) == 2:
        for name, definition in tracking.bond_definitions.items():
            if selected == set(definition[:2]):
                return tracking.bond_active[name].astype(float)
    return np.asarray(
        [
            any(selected.issubset(fragment) for fragment in frame_fragments)
            for frame_fragments in tracking.fragments
        ],
        dtype=float,
    )


def _safe_output_label(name):
    label = "".join(character if character.isalnum() else "_" for character in name)
    return label.strip("_") or "channel"


def save_dynamic_fragment_tracking(
    tracking,
    output_prefix,
    *,
    write_text=True,
    plot=True,
):
    """Save bond distances, hysteretic activities, and fragment identities."""

    if not isinstance(tracking, DynamicFragmentTracking):
        raise TypeError("tracking must be a DynamicFragmentTracking")
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    names = list(tracking.bond_definitions)
    distance_matrix = np.column_stack(
        [tracking.bond_distance_angstrom[name] for name in names]
    )
    active_matrix = np.column_stack(
        [tracking.bond_active[name] for name in names]
    )
    definitions = np.asarray(
        [tracking.bond_definitions[name] for name in names], dtype=float
    )
    fragment_labels = np.asarray(
        [
            "|".join("-".join(map(str, fragment)) for fragment in frame)
            for frame in tracking.fragments
        ]
    )

    npz_path = Path(str(prefix) + ".npz")
    np.savez_compressed(
        npz_path,
        time_fs=tracking.time_fs,
        bond_names=np.asarray(names),
        bond_definitions=definitions,
        bond_distance_angstrom=distance_matrix,
        bond_active=active_matrix,
        fragments=fragment_labels,
    )
    files = {"npz": str(npz_path)}

    if write_text:
        data_path = Path(str(prefix) + "_fragments.dat")
        columns = [tracking.time_fs]
        header = ["time_fs"]
        for index, name in enumerate(names):
            columns.extend([distance_matrix[:, index], active_matrix[:, index]])
            header.extend([f"{name}_distance_A", f"{name}_active"])
        numeric_matrix = np.column_stack(columns)
        with data_path.open("w") as handle:
            handle.write("# " + " ".join(header) + " fragments\n")
            for numeric, fragments in zip(numeric_matrix, fragment_labels):
                handle.write(
                    " ".join(f"{value:.10e}" for value in numeric)
                    + " "
                    + fragments
                    + "\n"
                )
        files["fragment_data"] = str(data_path)

    if plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "Plotting requires matplotlib; install Smite's analysis extra"
            ) from exc
        plot_path = Path(str(prefix) + "_bonds.png")
        figure, axis = plt.subplots(figsize=(9, 5))
        for name in names:
            line = axis.plot(
                tracking.time_fs,
                tracking.bond_distance_angstrom[name],
                label=name,
            )[0]
            formation, breaking = tracking.bond_definitions[name][2:]
            axis.axhline(
                formation,
                color=line.get_color(),
                linestyle=":",
                alpha=0.55,
            )
            if not np.isclose(formation, breaking):
                axis.axhline(
                    breaking,
                    color=line.get_color(),
                    linestyle="--",
                    alpha=0.4,
                )
        axis.set(
            xlabel="Time (fs)",
            ylabel="Bond distance (Angstrom)",
            title="Dynamic fragment tracking",
        )
        axis.grid(True)
        axis.legend()
        figure.tight_layout()
        figure.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        files["bond_plot"] = str(plot_path)

    tracking.output_files.update(files)
    return files


def short_time_channel_spectra(
    positions,
    velocities,
    masses,
    *,
    channels,
    dynamic_bonds=None,
    channel_frequency_bands=None,
    channel_internal_energy_hartree=None,
    channel_potential_energy_hartree=None,
    channel_reference_potential_hartree=None,
    position_unit="bohr",
    **spectrum_kwargs,
):
    """Analyze fixed pair/full-system channels and track their reactive activity."""

    if not isinstance(channels, dict) or not channels:
        raise ValueError("channels must be a non-empty name-to-atom-indices mapping")
    forbidden = {
        "atom_indices",
        "activity_mask",
        "internal_energy_hartree",
        "potential_energy_hartree",
        "reference_potential_hartree",
    }.intersection(spectrum_kwargs)
    if forbidden:
        raise ValueError(
            "channel-specific arguments cannot be supplied as common options: "
            + ", ".join(sorted(forbidden))
        )

    common_bands = spectrum_kwargs.pop("frequency_bands", None)
    positions_array = np.asarray(positions, dtype=float)
    tracking = None
    if dynamic_bonds is not None:
        tracking = track_dynamic_fragments(
            positions_array,
            dynamic_bonds=dynamic_bonds,
            dt_fs=spectrum_kwargs.get("dt_fs"),
            times_fs=spectrum_kwargs.get("times_fs"),
            position_unit=position_unit,
        )

    mappings = (
        ("channel_frequency_bands", channel_frequency_bands),
        ("channel_internal_energy_hartree", channel_internal_energy_hartree),
        ("channel_potential_energy_hartree", channel_potential_energy_hartree),
    )
    for mapping_name, mapping in mappings:
        if mapping is not None and not isinstance(mapping, dict):
            raise ValueError(f"{mapping_name} must be a mapping keyed by channel")

    spectra = {}
    for name, atom_indices in channels.items():
        if not isinstance(name, str) or not name:
            raise ValueError("channel names must be non-empty strings")
        kwargs = dict(spectrum_kwargs)
        kwargs["atom_indices"] = atom_indices
        kwargs["frequency_bands"] = (
            common_bands
            if channel_frequency_bands is None
            else channel_frequency_bands.get(name, common_bands)
        )
        if tracking is not None:
            kwargs["activity_mask"] = _fragment_activity(tracking, atom_indices)
        if (
            channel_internal_energy_hartree is not None
            and name in channel_internal_energy_hartree
        ):
            kwargs["internal_energy_hartree"] = channel_internal_energy_hartree[name]
        if (
            channel_potential_energy_hartree is not None
            and name in channel_potential_energy_hartree
        ):
            kwargs["potential_energy_hartree"] = channel_potential_energy_hartree[name]
            if isinstance(channel_reference_potential_hartree, dict):
                kwargs["reference_potential_hartree"] = (
                    channel_reference_potential_hartree.get(name, 0.0)
                )
            elif channel_reference_potential_hartree is not None:
                kwargs["reference_potential_hartree"] = (
                    channel_reference_potential_hartree
                )
        spectra[name] = short_time_fourier_spectrum(
            positions_array,
            velocities,
            masses,
            **kwargs,
        )

    return ShortTimeChannelAnalysis(
        spectra=spectra,
        fragment_tracking=tracking,
    )


def analyze_molecule_short_time_channels(
    molecule,
    *,
    channels,
    dt_fs=None,
    output_prefix=None,
    save=True,
    write_text=True,
    plot=True,
    energy_unit="kJ/mol",
    moving_average_fs=None,
    **kwargs,
):
    """Analyze and save several reactive vibrational channels together."""

    velocities = np.asarray(getattr(molecule, "vsave", ()), dtype=float)
    positions = np.asarray(getattr(molecule, "qsave", ()), dtype=float)
    times = np.asarray(getattr(molecule, "tsave", ()), dtype=float)
    if velocities.ndim != 2 or velocities.shape[0] < 2:
        raise ValueError(
            "No usable history is available; run dynamics with spectrum=True"
        )
    if positions.shape != velocities.shape:
        raise ValueError("saved position and velocity histories are inconsistent")
    if times.shape != (velocities.shape[0],):
        times = None

    result = short_time_channel_spectra(
        positions,
        velocities,
        molecule.mass,
        channels=channels,
        dt_fs=dt_fs,
        times_fs=times,
        **kwargs,
    )
    if save:
        if output_prefix is None:
            name = getattr(molecule, "fname", None) or "trajectory"
            output_prefix = f"{name}_short_time_channels"
        for channel_name, spectrum in result.spectra.items():
            label = _safe_output_label(channel_name)
            files = save_short_time_spectrum(
                spectrum,
                f"{output_prefix}_{label}",
                write_text=write_text,
                plot=plot,
                energy_unit=energy_unit,
                moving_average_fs=moving_average_fs,
            )
            for kind, path in files.items():
                result.output_files[f"{channel_name}:{kind}"] = path
        if result.fragment_tracking is not None:
            files = save_dynamic_fragment_tracking(
                result.fragment_tracking,
                f"{output_prefix}_dynamic_fragments",
                write_text=write_text,
                plot=plot,
            )
            for kind, path in files.items():
                result.output_files[f"fragments:{kind}"] = path
    return result


def analyze_molecule_short_time_spectrum(
    molecule,
    *,
    dt_fs=None,
    output_prefix=None,
    save=True,
    write_text=True,
    plot=True,
    energy_unit="kJ/mol",
    moving_average_fs=None,
    **kwargs,
):
    """Analyze histories collected with run_trajectory(spectrum=True)."""

    velocities = np.asarray(getattr(molecule, "vsave", ()), dtype=float)
    positions = np.asarray(getattr(molecule, "qsave", ()), dtype=float)
    times = np.asarray(getattr(molecule, "tsave", ()), dtype=float)
    if velocities.ndim != 2 or velocities.shape[0] < 2:
        raise ValueError(
            "No usable history is available; run dynamics with spectrum=True"
        )
    if positions.shape != velocities.shape:
        raise ValueError("saved position and velocity histories are inconsistent")
    if times.shape != (velocities.shape[0],):
        times = None

    result = short_time_fourier_spectrum(
        positions,
        velocities,
        molecule.mass,
        dt_fs=dt_fs,
        times_fs=times,
        **kwargs,
    )
    if save:
        if output_prefix is None:
            name = getattr(molecule, "fname", None) or "trajectory"
            output_prefix = f"{name}_short_time_spectrum"
        save_short_time_spectrum(
            result,
            output_prefix,
            write_text=write_text,
            plot=plot,
            energy_unit=energy_unit,
            moving_average_fs=moving_average_fs,
        )
    return result


__all__ = [
    "CYCLES_PER_FS_TO_CM1",
    "DynamicFragmentTracking",
    "ShortTimeChannelAnalysis",
    "ShortTimeSpectrum",
    "analyze_molecule_short_time_channels",
    "analyze_molecule_short_time_spectrum",
    "centered_moving_average",
    "mass_weighted_vibrational_velocity",
    "project_translation_and_rotation",
    "save_dynamic_fragment_tracking",
    "save_short_time_spectrum",
    "short_time_channel_spectra",
    "short_time_fourier_spectrum",
    "track_dynamic_fragments",
]

