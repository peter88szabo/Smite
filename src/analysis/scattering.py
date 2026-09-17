import numpy as np

from utils.constants import BOHR_TO_ANGSTROM, FS_TO_AU_TIME


def get_scattering_form_factors(molecule, dt, qmin=0.0, qmax=8.0, nq=600, dpi=600,
                                fourier=True,
                                temporal_reference="mean", temporal_window="hann",
                                pdf_reference="first", rmin=0.0, rmax=10.0, nr=400,
                                pdf_damping=None, max_frequency_cm1=None,
                                short_time=True,
                                short_time_window_fs=None, short_time_hop_fs=None,
                                short_time_q_range=None):
    """Time-resolved scattering, and optionally its two Fourier transforms.

    With ``fourier=True`` two further pairs of files are written alongside the
    time-dependent ones:

    - a power spectrum, the transform along time, showing which vibrational
      frequencies modulate the scattering at each q;
    - a pair distribution function, the sine transform along q, showing the
      interatomic distances and how they evolve.

    Both transform the difference signal rather than the raw intensity, and both
    are produced for the elastic, inelastic and total channels, three columns in
    one ``.dat`` and three panels in one ``.png``, exactly like the direct
    output. See :mod:`analysis.scattering_fourier`.
    """
    import matplotlib.pyplot as plt
    from analysis.xray_scattering_tables import independent_atom_model_scattering

    if len(molecule.qsave) == 0:
        raise ValueError(
            "No coordinate history saved for scattering analysis. "
            "Run the trajectory with spectrum=True first."
        )

    if nq < 2:
        raise ValueError("nq must be at least 2 for scattering analysis")
    if qmax <= qmin:
        raise ValueError("qmax must be larger than qmin")

    if len(molecule.tsave) == len(molecule.qsave) and not np.isnan(np.array(molecule.tsave)).all():
        time_fs = np.array(molecule.tsave, dtype=float)
    else:
        time_fs = np.arange(1, len(molecule.qsave) + 1, dtype=float) * dt / FS_TO_AU_TIME

    q_grid = np.linspace(qmin, qmax, nq, dtype=float)
    elastic = np.zeros((len(molecule.qsave), nq), dtype=float)
    inelastic = np.zeros((len(molecule.qsave), nq), dtype=float)
    total = np.zeros((len(molecule.qsave), nq), dtype=float)

    for iframe, q_snapshot in enumerate(molecule.qsave):
        coords_ang = np.reshape(q_snapshot, (-1, 3)) * BOHR_TO_ANGSTROM
        iam = independent_atom_model_scattering(molecule.atoms, coords_ang, q_grid)
        elastic[iframe, :] = iam["elastic"]
        inelastic[iframe, :] = iam["inelastic"]
        total[iframe, :] = iam["total"]

    rows = []
    for iframe, t_fs in enumerate(time_fs):
        for iq, q_value in enumerate(q_grid):
            rows.append((t_fs, q_value, elastic[iframe, iq], inelastic[iframe, iq], total[iframe, iq]))

    data_filename = molecule.fname + '_time_dependent_scattering_form_factors.dat'
    np.savetxt(
        data_filename,
        np.asarray(rows, dtype=float),
        fmt='%.8e',
        delimiter=' ',
        header='time_fs q_A^-1 elastic inelastic total'
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, constrained_layout=True)
    components = (
        ("Elastic", elastic),
        ("Inelastic", inelastic),
        ("Total", total),
    )

    for ax, (label, matrix) in zip(axes, components):
        mesh = ax.pcolormesh(time_fs, q_grid, matrix.T, shading='auto', cmap='viridis')
        ax.set_ylabel(r"$q$ [$\mathrm{\AA^{-1}}$]")
        ax.set_title(f"{molecule.fname} {label} Scattering")
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("Scattering Form Factor")

    axes[-1].set_xlabel("Time [fs]")

    png_filename = molecule.fname + '_time_dependent_scattering_form_factors.png'
    fig.savefig(png_filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    results = {
        'time_fs': time_fs,
        'q_Ainv': q_grid,
        'elastic': elastic,
        'inelastic': inelastic,
        'total': total,
        'data_file': data_filename,
        'plot_file': png_filename,
    }

    if fourier:
        results.update(
            _save_fourier_transforms(
                molecule, time_fs, q_grid,
                {'elastic': elastic, 'inelastic': inelastic, 'total': total},
                dpi=dpi,
                temporal_reference=temporal_reference,
                temporal_window=temporal_window,
                pdf_reference=pdf_reference,
                rmin=rmin, rmax=rmax, nr=nr, pdf_damping=pdf_damping,
                short_time=short_time,
                short_time_window_fs=short_time_window_fs,
                short_time_hop_fs=short_time_hop_fs,
                short_time_q_range=short_time_q_range,
                max_frequency_cm1=max_frequency_cm1,
            )
        )

    return results


def _save_fourier_transforms(molecule, time_fs, q_grid, components, *, dpi,
                             temporal_reference, temporal_window,
                             pdf_reference, rmin, rmax, nr, pdf_damping,
                             short_time=True, short_time_window_fs=None,
                             short_time_hop_fs=None, short_time_q_range=None,
                             max_frequency_cm1=None):
    """Write both transforms of all three channels, as .dat columns and .png panels.

    The elastic, inelastic and total terms are each transformed and laid out the
    way the direct output is: three columns in one ``.dat``, three stacked panels
    in one ``.png``.

    Expect the signal to sit in the elastic panel. In the independent atom model
    the inelastic term is ``sum_i S_i(q)``, a function of the atom list and q
    alone -- large in the raw intensity, around 70 to 80 per cent of the total at
    high q, but constant along a trajectory. Both transforms subtract the static
    part, so the inelastic panel comes out at the level of rounding error and the
    total panel reproduces the elastic one. They are written so that this is
    visible in the data rather than taken on trust.
    """
    import matplotlib.pyplot as plt
    from analysis.scattering_fourier import (
        atomic_form_factor_squared,
        pair_distribution_function,
        temporal_power_spectrum,
    )

    order = ('elastic', 'inelastic', 'total')
    outputs = {}

    def _save(basename, x_axis, y_axis, maps, *, x_label, y_label, header,
              title, cbar_label, diverging=False):
        rows = [
            (x, y) + tuple(maps[name][i, j] for name in order)
            for i, x in enumerate(x_axis)
            for j, y in enumerate(y_axis)
        ]
        data_file = molecule.fname + basename + '.dat'
        np.savetxt(data_file, np.asarray(rows, dtype=float), fmt='%.8e',
                   delimiter=' ', header=header)

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True,
                                 constrained_layout=True)
        for ax, name in zip(axes, order):
            matrix = maps[name]
            if diverging:
                limit = np.max(np.abs(matrix)) or 1.0
                mesh = ax.pcolormesh(x_axis, y_axis, matrix.T, shading='auto',
                                     cmap='RdBu_r', vmin=-limit, vmax=limit)
            else:
                mesh = ax.pcolormesh(x_axis, y_axis, matrix.T, shading='auto',
                                     cmap='magma')
            ax.set_ylabel(y_label)
            ax.set_title(f"{molecule.fname} {title} ({name.capitalize()})")
            fig.colorbar(mesh, ax=ax).set_label(cbar_label)
        axes[-1].set_xlabel(x_label)

        plot_file = molecule.fname + basename + '.png'
        fig.savefig(plot_file, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return data_file, plot_file

    # ---- A: transform along time -> power spectrum P(omega, q) -------------
    if len(time_fs) >= 2:
        spectra = {}
        for name in order:
            frequencies, spectra[name] = temporal_power_spectrum(
                time_fs, components[name],
                reference=temporal_reference, window=temporal_window,
            )
        if max_frequency_cm1 is not None:
            keep = frequencies <= float(max_frequency_cm1)
            frequencies = frequencies[keep]
            spectra = {name: value[keep] for name, value in spectra.items()}

        data_file, plot_file = _save(
            '_scattering_power_spectrum', frequencies, q_grid, spectra,
            x_label=r"Frequency [cm$^{-1}$]",
            y_label=r"$q$ [$\mathrm{\AA^{-1}}$]",
            header='frequency_cm^-1 q_A^-1 elastic inelastic total',
            title='scattering power spectrum', cbar_label='Power',
        )
        outputs.update({'frequency_cm1': frequencies,
                        'power_spectrum': spectra,
                        'power_spectrum_data_file': data_file,
                        'power_spectrum_plot_file': plot_file})

    # ---- B: sine transform along q -> pair distribution dPDF(r, t) ---------
    f2 = atomic_form_factor_squared(molecule.atoms, q_grid)
    distributions = {}
    for name in order:
        r, distributions[name] = pair_distribution_function(
            q_grid, components[name], f2, reference=pdf_reference,
            rmin=rmin, rmax=rmax, nr=nr, damping=pdf_damping,
        )
    data_file, plot_file = _save(
        '_scattering_pair_distribution', time_fs, r, distributions,
        x_label="Time [fs]",
        y_label=r"$r$ [$\mathrm{\AA}$]",
        header='time_fs r_A elastic inelastic total',
        title=r"$\Delta$PDF", cbar_label=r"$\Delta$PDF", diverging=True,
    )
    outputs.update({'r_Ang': r, 'pair_distribution': distributions,
                    'pair_distribution_data_file': data_file,
                    'pair_distribution_plot_file': plot_file})

    # ---- C: windowed transform along time -> spectrogram P(t, omega) -------
    # Skipped rather than raised when the trajectory is too short to hold a
    # window: the other outputs are still perfectly good.
    if short_time and len(time_fs) >= 8:
        from analysis.scattering_fourier import short_time_scattering_spectrum
        try:
            spectrum = short_time_scattering_spectrum(
                {'time_fs': time_fs, 'q_Ainv': q_grid, **components},
                q_range=short_time_q_range,
                window_fs=short_time_window_fs,
                hop_fs=short_time_hop_fs,
                reference=temporal_reference,
                window=temporal_window,
                max_frequency_cm1=max_frequency_cm1,
            )
        except ValueError:
            spectrum = None
        if spectrum is not None:
            outputs.update(
                save_short_time_scattering_spectrum(spectrum, molecule.fname, dpi=dpi)
            )
            outputs['short_time_spectrum'] = spectrum

            resolved = short_time_scattering_spectrum(
                {'time_fs': time_fs, 'q_Ainv': q_grid, **components},
                q_range=short_time_q_range,
                window_fs=short_time_window_fs,
                hop_fs=short_time_hop_fs,
                reference=temporal_reference,
                window=temporal_window,
                max_frequency_cm1=max_frequency_cm1,
                integrate_q=False,
            )
            outputs.update(
                save_short_time_q_resolved_spectrum(resolved, molecule.fname, dpi=dpi)
            )
            outputs['short_time_spectrum_q_resolved'] = resolved

    return outputs


def save_short_time_scattering_spectrum(spectrum, fname, *, dpi=600, label=''):
    """Write a short-time scattering spectrum as a .dat and a .png.

    ``spectrum`` is what :func:`analysis.scattering_fourier.short_time_scattering_spectrum`
    returns. The layout matches the other scattering outputs: two axis columns
    then the three channels, and three stacked panels.
    """
    import matplotlib.pyplot as plt
    from analysis.scattering_fourier import SCATTERING_CHANNELS

    times = np.asarray(spectrum['time_fs'], dtype=float)
    frequencies = np.asarray(spectrum['frequency_cm1'], dtype=float)

    rows = [
        (t, f) + tuple(float(spectrum[name][i, j]) for name in SCATTERING_CHANNELS)
        for i, t in enumerate(times)
        for j, f in enumerate(frequencies)
    ]
    data_file = f"{fname}_scattering_short_time_spectrum{label}.dat"
    np.savetxt(data_file, np.asarray(rows, dtype=float), fmt='%.8e', delimiter=' ',
               header='time_fs frequency_cm^-1 elastic inelastic total')

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True,
                             constrained_layout=True)
    for ax, name in zip(axes, SCATTERING_CHANNELS):
        mesh = ax.pcolormesh(times, frequencies, np.asarray(spectrum[name]).T,
                             shading='auto', cmap='magma')
        ax.set_ylabel(r"Frequency [cm$^{-1}$]")
        ax.set_title(f"{fname} short-time scattering spectrum ({name.capitalize()})")
        fig.colorbar(mesh, ax=ax).set_label("Power")
    axes[-1].set_xlabel("Time [fs]")

    plot_file = f"{fname}_scattering_short_time_spectrum{label}.png"
    fig.savefig(plot_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return {'short_time_data_file': data_file, 'short_time_plot_file': plot_file}


def reanalyze_scattering_short_time(path, *, fname=None, dpi=600, label='',
                                    q_range=None, **kwargs):
    """Re-run the short-time transform on a scattering file already on disk.

    Reads a ``*_time_dependent_scattering_form_factors.dat`` written by an
    earlier run, applies the short-time transform with whatever window settings
    are given, and writes the usual ``.dat`` and ``.png``. The trajectory does
    not need to be repeated to try a different window.

    Parameters
    ----------
    path
        The saved time-dependent scattering file.
    fname
        Stem for the output files. Defaults to the input file's stem with the
        ``_time_dependent_scattering_form_factors`` suffix stripped.
    q_range
        ``(qmin, qmax)`` to integrate over, isolating a distance range.
    kwargs
        ``window_fs``, ``hop_fs``, ``window``, ``reference``,
        ``max_frequency_cm1``.
    """
    import os
    from analysis.scattering_fourier import short_time_scattering_spectrum

    if fname is None:
        stem = os.path.basename(path)
        for suffix in ('.dat', '_time_dependent_scattering_form_factors'):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
        fname = stem

    spectrum = short_time_scattering_spectrum(path, q_range=q_range, **kwargs)
    spectrum.update(
        save_short_time_scattering_spectrum(spectrum, fname, dpi=dpi, label=label)
    )

    resolved = short_time_scattering_spectrum(
        path, q_range=q_range, integrate_q=False, **kwargs
    )
    spectrum.update(
        save_short_time_q_resolved_spectrum(resolved, fname, dpi=dpi, label=label)
    )
    spectrum['q_resolved'] = resolved
    return spectrum


def save_short_time_q_resolved_spectrum(spectrum, fname, *, dpi=600, label='',
                                        max_panels=4):
    """Write the q-resolved short-time spectrum: P(t, omega, q).

    ``spectrum`` is what
    :func:`analysis.scattering_fourier.short_time_scattering_spectrum` returns
    with ``integrate_q=False``, so each channel is
    ``(n_windows, n_frequencies, n_q)``.

    The ``.dat`` holds the whole object, six columns:
    ``time_fs frequency_cm^-1 q_A^-1 elastic inelastic total``. It has
    ``n_windows * n_frequencies * n_q`` rows, so a fine window on a long
    trajectory makes a large file.

    The ``.png`` cannot show three axes at once, so it lays out frequency
    against q -- keeping the y axis of the other scattering plots -- for up to
    ``max_panels`` evenly spaced time windows, one row each, three channels
    across.
    """
    import matplotlib.pyplot as plt
    from analysis.scattering_fourier import SCATTERING_CHANNELS

    times = np.asarray(spectrum['time_fs'], dtype=float)
    frequencies = np.asarray(spectrum['frequency_cm1'], dtype=float)
    q_grid = np.asarray(spectrum['q_Ainv'], dtype=float)

    rows = [
        (t, f, q) + tuple(float(spectrum[name][i, j, k]) for name in SCATTERING_CHANNELS)
        for i, t in enumerate(times)
        for j, f in enumerate(frequencies)
        for k, q in enumerate(q_grid)
    ]
    data_file = f"{fname}_scattering_short_time_spectrum_q_resolved{label}.dat"
    np.savetxt(data_file, np.asarray(rows, dtype=float), fmt='%.8e', delimiter=' ',
               header='time_fs frequency_cm^-1 q_A^-1 elastic inelastic total')

    selected = np.unique(np.linspace(0, times.size - 1,
                                     min(max_panels, times.size)).astype(int))
    fig, axes = plt.subplots(selected.size, len(SCATTERING_CHANNELS),
                             figsize=(5 * len(SCATTERING_CHANNELS), 4 * selected.size),
                             squeeze=False, constrained_layout=True)
    for row, window in enumerate(selected):
        for column, name in enumerate(SCATTERING_CHANNELS):
            ax = axes[row][column]
            mesh = ax.pcolormesh(frequencies, q_grid,
                                 np.asarray(spectrum[name])[window].T,
                                 shading='auto', cmap='magma')
            ax.set_title(f"{name.capitalize()}, t = {times[window]:.0f} fs")
            if column == 0:
                ax.set_ylabel(r"$q$ [$\mathrm{\AA^{-1}}$]")
            if row == selected.size - 1:
                ax.set_xlabel(r"Frequency [cm$^{-1}$]")
            fig.colorbar(mesh, ax=ax).set_label("Power")
    fig.suptitle(f"{fname} short-time scattering spectrum, resolved in q")

    plot_file = f"{fname}_scattering_short_time_spectrum_q_resolved{label}.png"
    fig.savefig(plot_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return {'short_time_q_resolved_data_file': data_file,
            'short_time_q_resolved_plot_file': plot_file}
