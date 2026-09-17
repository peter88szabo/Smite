from pathlib import Path
import os
import sys

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE.parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Write every output next to this script, whatever directory it is run from.
os.chdir(HERE)

if __name__ == '__main__':
    import numpy as np

    from smite import Molecule
    from utils.atomic_masses import get_mass_vector
    from utils.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, FS_TO_AU_TIME

    '''
    Time-resolved X-ray scattering from a molecule with several active modes.

    Acetone on GFN2-xTB. Formaldehyde has essentially one heavy-atom mode;
    acetone has a well separated set, so the scattering carries more than a
    single frequency:

        ~420, 500, 512 cm^-1   skeletal bends
        ~827, 854 cm^-1        C-C-C skeletal stretch
       ~1054, 1070, 1183       C-C stretches
             ~1764 cm^-1       C=O stretch
       ~3020-3070 cm^-1        C-H stretches

    Several of them are excited at once by displacing the C=O bond and the two
    C-C bonds by different amounts, which is enough to put energy into the
    symmetric and antisymmetric skeletal modes as well as the C=O stretch.

    X-ray scattering is weighted by electron density, so the C-H stretches
    around 3000 cm^-1 stay weak however much they move: hydrogen contributes one
    electron against six for carbon and eight for oxygen. The heavy-atom modes
    dominate, which is the point worth seeing in the plots.

    The run writes, each as a .dat and a .png with elastic, inelastic and total
    side by side:

      *_time_dependent_scattering_form_factors   I(q, t)
      *_scattering_power_spectrum                Fourier transform along time
      *_scattering_pair_distribution             sine transform along q, dPDF(r, t)
      *_scattering_short_time_spectrum           windowed transform, P(t, omega)
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

    atoms = ['C', 'O', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']

    # GFN2-optimized acetone.
    xyz_acetone = np.array([
         0.00000000,  0.00000000,  0.08463847,   # C carbonyl
         0.00000000,  0.00000000,  1.28951274,   # O
         1.28609984,  0.00000000, -0.70786747,   # C methyl
        -1.28609984,  0.00000000, -0.70786747,   # C methyl
         1.87040631,  0.87527518, -0.42924345,
         1.87040635, -0.87527515, -0.42924341,
         1.11836019,  0.00000000, -1.77922126,
        -1.87040632,  0.87527516, -0.42924341,
        -1.87040634, -0.87527517, -0.42924346,
        -1.11836019,  0.00000000, -1.77922126,
    ])

    # Excite several modes at once. The two C-C bonds are stretched by different
    # amounts on purpose: an equal displacement would drive only the symmetric
    # combination and leave the antisymmetric one silent.
    geometry = xyz_acetone.reshape(-1, 3)
    geometry[1] += [0.0, 0.0,  0.090]     # C=O   longer
    geometry[2] += [0.070, 0.0, 0.0]      # C-C   longer
    geometry[3] -= [0.045, 0.0, 0.0]      # C-C   longer by a different amount

    timestep_fs = 0.5
    maxstep = 1024                        # 512 fs -> about 65 cm^-1 resolution

    molecule = Molecule(atoms, get_mass_vector(atoms),
                        geometry.reshape(-1) * ANGSTROM_TO_BOHR, np.zeros(30))
    molecule.fname = 'acetone'
    molecule.qchem = qcinput

    print("\nAcetone on GFN2-xTB, %d steps of %.2f fs (%.0f fs total)\n"
          % (maxstep, timestep_fs, maxstep * timestep_fs))

    molecule.run_trajectory(integrator='verlet', timestep=timestep_fs,
                            maxstep=maxstep, iprint=128, Rstop=50.0,
                            traj_file='acetone.xyz',
                            backfile='acetone_backup.xyz',
                            spectrum=True)

    scattering = molecule.get_scattering_form_factors(
        timestep_fs * FS_TO_AU_TIME,
        qmin=0.0, qmax=12.0, nq=200,
        rmin=0.5, rmax=6.0, nr=400,
        # 0.5 fs sampling puts Nyquist near 33000 cm^-1. Cap the frequency axis
        # just above the C-H stretches so the plots are readable.
        max_frequency_cm1=3500.0,
        short_time_window_fs=200.0, short_time_hop_fs=40.0,
        dpi=150,
    )

    print("\n--------- files written ---------")
    for key in sorted(k for k in scattering if k.endswith('_file')):
        print("  %s" % scattering[key])

    # Which frequencies actually modulate the scattering.
    frequencies = scattering['frequency_cm1']
    power = scattering['power_spectrum']['elastic'].sum(axis=1)
    band = frequencies > 250.0

    peaks = []
    for index in range(1, power.size - 1):
        if not band[index]:
            continue
        if power[index] > power[index - 1] and power[index] > power[index + 1]:
            peaks.append((power[index], frequencies[index]))
    peaks.sort(reverse=True)

    print("\n--------- strongest scattering modulations ---------")
    strongest = peaks[0][0] if peaks else 1.0
    for weight, frequency in peaks[:6]:
        print("  %7.1f cm^-1   relative weight %6.3f" % (frequency, weight / strongest))

    print("\n  GFN2 harmonic reference: 420, 500, 512, 827, 854, 970, 1054,")
    print("                           1070, 1183, 1364-1468, 1764 (C=O),")
    print("                           3020-3070 (C-H, weak in X-ray scattering)")

    # Which interatomic distances the trajectory actually moved, for comparison
    # with the pair distribution plot.
    coordinates = np.array(molecule.qsave).reshape(-1, len(atoms), 3) * BOHR_TO_ANGSTROM
    distances = np.linalg.norm(
        coordinates[:, :, None, :] - coordinates[:, None, :, :], axis=3
    )
    print("\n--------- heavy-atom distances along the trajectory ---------")
    labels = {(0, 1): 'C=O', (0, 2): 'C-C', (0, 3): 'C-C', (2, 3): 'C...C'}
    for (i, j), label in labels.items():
        trace = distances[:, i, j]
        print("  %-6s mean %.3f A, swing %.3f A" %
              (label, trace.mean(), trace.max() - trace.min()))
