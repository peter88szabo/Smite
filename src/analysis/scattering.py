import numpy as np

from utils.constants import BOHR_TO_ANGSTROM, FS_TO_AU_TIME


def get_scattering_form_factors(molecule, dt, qmin=0.0, qmax=8.0, nq=600, dpi=600):
    import matplotlib.pyplot as plt
    from analysis.xray_scattering_tables_hcno import independent_atom_model_scattering

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

    return {
        'time_fs': time_fs,
        'q_Ainv': q_grid,
        'elastic': elastic,
        'inelastic': inelastic,
        'total': total,
        'data_file': data_filename,
        'plot_file': png_filename,
    }
