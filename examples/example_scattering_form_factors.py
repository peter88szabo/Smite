from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == '__main__':
    import numpy as np

    from smite import Molecule
    from analysis.scattering import reanalyze_scattering_short_time
    from utils.atomic_masses import get_mass_vector
    from utils.constants import ANGSTROM_TO_BOHR, FS_TO_AU_TIME

    '''
    Time-resolved X-ray scattering from an NVE trajectory.

    Formaldehyde on GFN2-xTB, started with the C=O bond stretched by 0.1 A so
    the C=O stretch is excited and there is something for the scattering to
    follow. The run writes, each as a .dat and a .png with elastic, inelastic
    and total side by side:

      *_time_dependent_scattering_form_factors   I(q, t)
      *_scattering_power_spectrum                Fourier transform along time
      *_scattering_pair_distribution             sine transform along q, dPDF(r, t)
      *_scattering_short_time_spectrum           windowed transform, P(t, omega)

    Scattering needs spectrum=True, which saves geometries only for an NVE run.
    '''

    qcinput = {
    'qchem': 'XTB',
    'path': '/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/xtb',
    'nproc': 1,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 1,
    'additional': '--gfn 2 --acc 1.0',
    'wfu': False
    }

    atoms = ['C', 'O', 'H', 'H']

    # Equilibrium formaldehyde with C=O lengthened from 1.208 to 1.308 A.
    xyz_ch2o = np.array([
        0.000,  0.000,  0.000,     # C
        0.000,  0.000,  1.308,     # O
        0.000,  0.943, -0.588,     # H
        0.000, -0.943, -0.588,     # H
    ]) * ANGSTROM_TO_BOHR

    timestep_fs = 0.5
    maxstep = 400

    molecule = Molecule(atoms, get_mass_vector(atoms), xyz_ch2o, np.zeros(12))
    molecule.fname = 'ch2o'
    molecule.qchem = qcinput

    print("\nFormaldehyde on GFN2-xTB, %d steps of %.2f fs\n" % (maxstep, timestep_fs))

    molecule.run_trajectory(integrator='verlet', timestep=timestep_fs,
                            maxstep=maxstep, iprint=50, Rstop=50.0,
                            traj_file='ch2o.xyz',
                            backfile='ch2o_backup.xyz',
                            spectrum=True)

    # qmax sets the real-space resolution of the pair distribution, about
    # pi/qmax; 12 A^-1 resolves features roughly 0.26 A apart.
    scattering = molecule.get_scattering_form_factors(
        timestep_fs * FS_TO_AU_TIME,
        qmin=0.0, qmax=12.0, nq=200,
        rmin=0.5, rmax=4.0, nr=300,
        short_time_window_fs=120.0, short_time_hop_fs=20.0,
        # 0.5 fs sampling puts Nyquist near 33000 cm^-1; cap the plots at the
        # range that actually contains molecular vibrations.
        max_frequency_cm1=4000.0,
        dpi=150,
    )

    print("\n--------- files written ---------")
    for key in sorted(k for k in scattering if k.endswith('_file')):
        print("  %s" % scattering[key])

    frequencies = scattering['frequency_cm1']
    power = scattering['power_spectrum']['elastic'].sum(axis=1)
    band = frequencies > 300.0
    print("\n  strongest scattering modulation: %.0f cm^-1" %
          frequencies[band][np.argmax(power[band])])
    print("  (formaldehyde C=O stretch; ~1746 cm^-1 experimental)")

    # The saved file can be re-analysed with a different window, or over a
    # restricted q range, without repeating the trajectory. `label` keeps
    # several analyses of the same run side by side.
    print("\n--------- re-analysis of the saved run ---------")
    for window_fs, label in ((60.0, '_w60'), (120.0, '_w120')):
        again = reanalyze_scattering_short_time(
            scattering['data_file'], fname=molecule.fname,
            window_fs=window_fs, hop_fs=0.25 * window_fs, dpi=150, label=label,
            max_frequency_cm1=4000.0,
        )
        print("  window %5.0f fs -> %2d windows, %4.0f cm^-1 resolution : %s"
              % (window_fs, again['elastic'].shape[0],
                 again['frequency_cm1'][1] - again['frequency_cm1'][0],
                 again['short_time_plot_file']))
