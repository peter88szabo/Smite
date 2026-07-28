import numpy as np

from utils.constants import FS_TO_AU_TIME, HARTREE_TO_CM1


def frequency_axis_cm1(sample_count, dt_fs):
    """Return the one-sided FFT frequency grid for a timestep in femtoseconds."""
    if int(sample_count) != sample_count or sample_count < 2:
        raise ValueError("At least two velocity samples are required for a spectrum")
    if not np.isfinite(dt_fs) or dt_fs <= 0.0:
        raise ValueError("The spectrum timestep must be a finite positive value in fs")

    cycles_per_au = np.fft.rfftfreq(int(sample_count), d=dt_fs * FS_TO_AU_TIME)
    angular_frequency_au = 2.0 * np.pi * cycles_per_au
    return angular_frequency_au * HARTREE_TO_CM1


def vibrational_spectrum(molecule, dt, print_maxfreq=5000.0):
    """Compute the velocity power spectrum; ``dt`` is in femtoseconds."""
    import matplotlib.pyplot as plt

    velocities = np.asarray(molecule.vsave, dtype=float)
    if velocities.ndim != 2 or velocities.shape[1] != len(molecule.p):
        raise ValueError("Saved velocities must have shape (number of samples, 3N)")

    freq_cm1 = frequency_axis_cm1(len(velocities), dt)
    velocity_fft = np.fft.rfft(velocities, axis=0)
    total_velo_power = np.sum(np.abs(velocity_fft) ** 2, axis=1)

    data_to_save = np.column_stack((freq_cm1, total_velo_power))
    data_filename = molecule.fname + '_velocity_power_spectrum.dat'
    np.savetxt(data_filename, data_to_save, fmt='%.8e', delimiter=' ')

    plt.figure(figsize=(8, 5))
    plt.plot(freq_cm1, total_velo_power, label="FFT of Velocity")
    plt.xlim(0, print_maxfreq)
    plt.xlabel("Frequency (cm$^{-1}$)")
    plt.ylabel("Velocity Power Spectrum")
    plt.legend()
    plt.grid()

    file_to_save = molecule.fname + '_velocity_power_spectrum_' + '.png'
    plt.savefig(file_to_save, dpi=300, bbox_inches='tight')
