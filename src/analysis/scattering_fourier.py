"""Fourier analysis of a time-resolved scattering signal.

Two transforms of the ``I(t, q)`` map produced by
:func:`analysis.scattering.get_scattering_form_factors`, answering different
questions:

**Along time** -- :func:`temporal_power_spectrum` gives ``P(omega, q)``, the
vibrational frequencies that modulate the scattering at each momentum transfer.
The frequency axis is built with :func:`analysis.spectrum.frequency_axis_cm1`,
so it is directly comparable to ``molecule.vibrational_spectrum``.

**Along q** -- :func:`pair_distribution_function` gives ``dPDF(r, t)``, the
time-resolved real-space pair distribution, which is the conventional way an
ultrafast scattering experiment is read.

**Along time, windowed** -- :func:`short_time_power_spectrum` gives
``P(t, omega)``, for trajectories where the dynamics are not stationary and a
single global spectrum would average distinct regimes together. It is generic
and takes any evenly sampled ``(n_frames, n_channels)`` signal.
:func:`read_time_dependent_scattering` and
:func:`short_time_scattering_spectrum` re-analyse a run already written to disk,
so the window can be changed without recomputing the trajectory.

Both transform a *difference* signal. The static scattering is orders of
magnitude larger than its modulation, so transforming the raw intensity would
bury the dynamics under the ``t``-independent term. ``reference`` selects what
is subtracted:

- ``"mean"``  -- the time average at each q; the right choice for a power
  spectrum, since it removes the zero-frequency component.
- ``"first"`` -- the first frame; matches the experimental convention where
  ``dI = I(t) - I(t < 0)``.

The elastic, inelastic and total channels are all transformed by the callers in
:mod:`analysis.scattering`, but expect the signal in the elastic one: the
inelastic term of the independent atom model depends on the atom list and q and
not on the geometry, so it is constant in time and the difference removes it.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from analysis.spectrum import frequency_axis_cm1


def _difference_signal(intensity: np.ndarray, reference: str) -> np.ndarray:
    """Subtract the static part of ``I(t, q)`` along the time axis."""
    intensity = np.asarray(intensity, dtype=float)
    if intensity.ndim != 2:
        raise ValueError("intensity must have shape (n_frames, n_q)")

    if reference == "mean":
        return intensity - intensity.mean(axis=0, keepdims=True)
    if reference == "first":
        return intensity - intensity[0][None, :]
    if reference == "none":
        return intensity
    raise ValueError(
        f"Unknown reference {reference!r}; choose from 'mean', 'first', 'none'"
    )


def _window(name: Optional[str], sample_count: int) -> np.ndarray:
    if name is None or name == "none":
        return np.ones(sample_count, dtype=float)
    if name == "hann":
        return np.hanning(sample_count)
    if name == "hamming":
        return np.hamming(sample_count)
    raise ValueError(f"Unknown window {name!r}; choose from 'hann', 'hamming', 'none'")


def temporal_power_spectrum(
    time_fs: Sequence[float],
    intensity: np.ndarray,
    *,
    reference: str = "mean",
    window: Optional[str] = "hann",
) -> Tuple[np.ndarray, np.ndarray]:
    """Fourier transform ``I(t, q)`` along time.

    Parameters
    ----------
    time_fs
        Frame times in femtoseconds, assumed evenly spaced.
    intensity
        ``(n_frames, n_q)`` scattering intensity.
    reference
        Static part to remove; see the module docstring.
    window
        Apodization applied along time. A trajectory is short, so the default
        Hann window matters: without it the truncation sidelobes of the strongest
        mode can be mistaken for weaker modes.

    Returns
    -------
    (frequencies_cm1, power)
        ``frequencies_cm1`` has length ``n_frames // 2 + 1`` and ``power`` has
        shape ``(n_frequencies, n_q)``.
    """
    time_fs = np.asarray(time_fs, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    if intensity.ndim != 2:
        raise ValueError("intensity must have shape (n_frames, n_q)")
    if len(time_fs) != intensity.shape[0]:
        raise ValueError("time_fs and intensity disagree on the number of frames")
    if len(time_fs) < 2:
        raise ValueError("at least two frames are needed for a temporal transform")

    dt_fs = float(np.mean(np.diff(time_fs)))
    if not np.isfinite(dt_fs) or dt_fs <= 0.0:
        raise ValueError("frame times must be increasing and evenly spaced")

    signal = _difference_signal(intensity, reference)
    signal = signal * _window(window, signal.shape[0])[:, None]

    power = np.abs(np.fft.rfft(signal, axis=0)) ** 2
    return frequency_axis_cm1(signal.shape[0], dt_fs), power


def modified_scattering(
    q_ang_inv: Sequence[float],
    difference_intensity: np.ndarray,
    atomic_form_factor_squared: Sequence[float],
) -> np.ndarray:
    """The modified scattering intensity ``sM(q) = q dI(q) / sum_i f_i(q)^2``.

    Dividing out the atomic form factors removes the single-atom decay, leaving
    the interference term that carries the interatomic distances.
    """
    q = np.asarray(q_ang_inv, dtype=float)
    denominator = np.asarray(atomic_form_factor_squared, dtype=float)
    if denominator.shape != q.shape:
        raise ValueError("atomic_form_factor_squared must match the q grid")

    safe = np.where(denominator > 0.0, denominator, np.inf)
    return q[None, :] * np.asarray(difference_intensity, dtype=float) / safe[None, :]


def pair_distribution_function(
    q_ang_inv: Sequence[float],
    intensity: np.ndarray,
    atomic_form_factor_squared: Sequence[float],
    *,
    reference: str = "first",
    rmin: float = 0.0,
    rmax: float = 10.0,
    nr: int = 400,
    damping: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sine transform ``dI(q, t)`` into a real-space pair distribution.

    ``dPDF(r, t) = integral sM(q, t) sin(q r) exp(-alpha q^2) dq``

    Parameters
    ----------
    damping
        The ``alpha`` of the Gaussian damping that suppresses the ripple caused
        by truncating the integral at ``qmax``. ``None`` picks
        ``ln(10) / qmax^2``, i.e. the integrand is damped to a tenth of its
        weight at the edge of the measured range.

    Returns
    -------
    (r, dpdf)
        ``r`` in angstrom with length ``nr``; ``dpdf`` has shape
        ``(n_frames, nr)``.
    """
    q = np.asarray(q_ang_inv, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    if intensity.ndim != 2:
        raise ValueError("intensity must have shape (n_frames, n_q)")
    if intensity.shape[1] != q.size:
        raise ValueError("intensity and q_ang_inv disagree on the number of q points")
    if q.size < 2:
        raise ValueError("at least two q points are needed for a sine transform")
    if nr < 2:
        raise ValueError("nr must be at least 2")
    if rmax <= rmin:
        raise ValueError("rmax must be larger than rmin")

    qmax = float(np.max(q))
    if damping is None:
        damping = np.log(10.0) / qmax**2 if qmax > 0.0 else 0.0

    difference = _difference_signal(intensity, reference)
    sm = modified_scattering(q, difference, atomic_form_factor_squared)

    r = np.linspace(rmin, rmax, nr, dtype=float)
    # (nr, nq) kernel: sin(q r) damped towards the truncation edge.
    kernel = np.sin(np.outer(r, q)) * np.exp(-damping * q**2)[None, :]

    # Integrate over q for every frame at once.
    dpdf = np.trapezoid(sm[:, None, :] * kernel[None, :, :], q, axis=2)
    return r, dpdf


def atomic_form_factor_squared(
    elements: Sequence[str], q_ang_inv: Sequence[float]
) -> np.ndarray:
    """``sum_i f_i(q)^2`` for a set of atoms, the sM(q) normalisation."""
    from analysis.xray_scattering_tables import f0_cromer_mann

    q = np.asarray(q_ang_inv, dtype=float)
    total = np.zeros_like(q)
    for element in elements:
        total += np.asarray(f0_cromer_mann(element, q), dtype=float) ** 2
    return total


# --------------------------------------------------------------------------
# Short-time Fourier transform
# --------------------------------------------------------------------------
#
# The transform above assumes the dynamics are stationary over the whole
# trajectory. That holds for a vibrating molecule and fails for a reactive one,
# where modes appear, shift and vanish. A short-time transform resolves that:
# it slides a window along the trajectory and transforms each slice, giving
# P(t, omega) instead of P(omega).
#
# ``short_time_power_spectrum`` is deliberately generic -- it takes any
# ``(n_frames, n_channels)`` signal, so it applies to a scattering map over q
# just as well as to anything else sampled on an even time grid.


def _window_centres(n_samples, window_samples, hop_samples):
    starts = np.arange(0, n_samples - window_samples + 1, hop_samples)
    return starts, starts + window_samples // 2


def short_time_power_spectrum(
    time_fs: Sequence[float],
    signal: np.ndarray,
    *,
    window_fs: Optional[float] = None,
    hop_fs: Optional[float] = None,
    window: Optional[str] = "hann",
    reference: str = "mean",
    max_frequency_cm1: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slide a window along ``signal`` and transform each slice.

    Generic: ``signal`` is ``(n_frames, n_channels)`` and the channels are never
    interpreted, so this works on a scattering map over q, on a single trace, or
    on anything else sampled on an even time grid.

    Parameters
    ----------
    window_fs
        Window length in femtoseconds. ``None`` uses a quarter of the trajectory,
        capped at 200 fs. The window sets the frequency resolution, roughly
        ``33000 / window_fs`` in cm^-1, against the time resolution it costs.
    hop_fs
        Step between consecutive windows. ``None`` uses a quarter of the window.
    reference
        Static part removed **once, globally**, before slicing -- not per window,
        which would also remove any drift the spectrogram is meant to show.

    Returns
    -------
    (centres_fs, frequencies_cm1, power)
        ``power`` has shape ``(n_windows, n_frequencies, n_channels)``.
    """
    time_fs = np.asarray(time_fs, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if signal.ndim == 1:
        signal = signal[:, None]
    if signal.ndim != 2:
        raise ValueError("signal must have shape (n_frames, n_channels)")
    if len(time_fs) != signal.shape[0]:
        raise ValueError("time_fs and signal disagree on the number of frames")
    if len(time_fs) < 4:
        raise ValueError("at least four frames are needed for a short-time transform")

    dt_fs = float(np.mean(np.diff(time_fs)))
    if not np.isfinite(dt_fs) or dt_fs <= 0.0:
        raise ValueError("frame times must be increasing and evenly spaced")

    duration_fs = dt_fs * signal.shape[0]
    if window_fs is None:
        window_fs = min(200.0, 0.25 * duration_fs)
    if hop_fs is None:
        hop_fs = 0.25 * window_fs

    for value, name in ((window_fs, "window_fs"), (hop_fs, "hop_fs")):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    window_samples = max(2, int(round(window_fs / dt_fs)))
    hop_samples = max(1, int(round(hop_fs / dt_fs)))
    if window_samples > signal.shape[0]:
        raise ValueError(
            f"window_fs needs {window_samples} samples but only "
            f"{signal.shape[0]} are available"
        )
    if hop_samples > window_samples:
        raise ValueError("hop_fs cannot exceed the window length")

    detrended = _difference_signal(signal, reference)
    taper = _window(window, window_samples)[:, None]

    starts, centre_indices = _window_centres(
        signal.shape[0], window_samples, hop_samples
    )
    frequencies = frequency_axis_cm1(window_samples, dt_fs)

    power = np.empty((len(starts), frequencies.size, signal.shape[1]), dtype=float)
    for index, start in enumerate(starts):
        slice_ = detrended[start:start + window_samples] * taper
        power[index] = np.abs(np.fft.rfft(slice_, axis=0)) ** 2

    if max_frequency_cm1 is not None:
        keep = frequencies <= float(max_frequency_cm1)
        frequencies, power = frequencies[keep], power[:, keep, :]

    return time_fs[centre_indices], frequencies, power


# --------------------------------------------------------------------------
# Re-analysing a saved run
# --------------------------------------------------------------------------

SCATTERING_CHANNELS = ("elastic", "inelastic", "total")


def read_time_dependent_scattering(path: str) -> dict:
    """Read a ``*_time_dependent_scattering_form_factors.dat`` back into arrays.

    Lets a finished run be re-analysed with different window settings without
    recomputing the trajectory.

    Returns
    -------
    dict
        ``time_fs`` ``(n_frames,)``, ``q_Ainv`` ``(n_q,)``, and ``elastic``,
        ``inelastic``, ``total``, each ``(n_frames, n_q)``.
    """
    table = np.loadtxt(path)
    if table.ndim != 2 or table.shape[1] != 5:
        raise ValueError(
            f"{path} does not look like a time-dependent scattering file: "
            f"expected 5 columns (time_fs q_A^-1 elastic inelastic total), "
            f"got shape {table.shape}"
        )

    time_fs = np.unique(table[:, 0])
    q = np.unique(table[:, 1])
    if time_fs.size * q.size != table.shape[0]:
        raise ValueError(
            f"{path} is not a complete time-by-q grid: {time_fs.size} times and "
            f"{q.size} q points cannot fill {table.shape[0]} rows"
        )

    result = {"time_fs": time_fs, "q_Ainv": q}
    for offset, name in enumerate(SCATTERING_CHANNELS):
        result[name] = table[:, 2 + offset].reshape(time_fs.size, q.size)
    return result


def short_time_scattering_spectrum(
    source,
    *,
    q_range: Optional[Tuple[float, float]] = None,
    integrate_q: bool = True,
    **kwargs,
) -> dict:
    """Short-time spectrum of a scattering run, from a file path or from arrays.

    ``source`` is either the path of a saved
    ``*_time_dependent_scattering_form_factors.dat`` or the dict that
    :func:`read_time_dependent_scattering` returns.

    With ``integrate_q=True``, the default, the spectrogram is summed over q and
    each channel is a ``(n_windows, n_frequencies)`` map -- the usual
    time-versus-frequency view. ``q_range`` restricts that sum, which is how a
    particular distance range is isolated.

    With ``integrate_q=False`` the q axis is kept and each channel is the full
    ``(n_windows, n_frequencies, n_q)`` object, which shows *where* in q each
    modulation lives. Note that this grows as the product of the three axes.

    Remaining keyword arguments go to :func:`short_time_power_spectrum`.

    Returns
    -------
    dict
        ``time_fs``, ``frequency_cm1``, ``q_Ainv``, and one map per channel.
    """
    data = read_time_dependent_scattering(source) if isinstance(source, str) else source

    q = np.asarray(data["q_Ainv"], dtype=float)
    mask = np.ones(q.size, dtype=bool)
    if q_range is not None:
        low, high = float(q_range[0]), float(q_range[1])
        if high <= low:
            raise ValueError("q_range must be increasing")
        mask = (q >= low) & (q <= high)
        if not mask.any():
            raise ValueError(f"no q points inside {q_range}")

    result = {}
    for name in SCATTERING_CHANNELS:
        centres, frequencies, power = short_time_power_spectrum(
            data["time_fs"], np.asarray(data[name], dtype=float)[:, mask], **kwargs
        )
        result[name] = power.sum(axis=2) if integrate_q else power

    result["time_fs"] = centres
    result["frequency_cm1"] = frequencies
    result["q_Ainv"] = q[mask]
    return result
