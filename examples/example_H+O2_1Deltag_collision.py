from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


if __name__ == "__main__":
    import math
    import random

    import numpy as np
    from smite import Collision, Fragment
    from utils.atomic_masses import get_mass_vector
    from utils.constants import (
        BOHR_TO_ANGSTROM,
        CM1_TO_AU_ANGULAR_FREQUENCY,
        KJMOL_TO_HARTREE,
    )

    pes_dir = SRC_DIR.parent / "peslib" / "HO2_1Deltag"
    qcinput_pes = {
        "qchem": "PES",
        "pes_name": "HO2_1Deltag",
        "pes_path": str(pes_dir),
        "wfu": False,
        "hessian_dx": 0.002,
    }

    # Fixed initial state and translational energy: this is not a thermal run.
    nvib_O2 = 0
    jrot_O2 = 0
    collision_energy = 10.0       # kJ/mol
    initial_distance = 10.0       # Angstrom, fragment COM to fragment COM
    bmax = 2.0                    # Angstrom
    timestep = 0.5                # fs

    seed = 5102026
    random.seed(seed)

    # Rotating-Morse O2 parameters.  beta is derived from the equilibrium
    # harmonic curvature so the Morse model reproduces omega near re.
    req_O2 = 1.2075               # Angstrom
    omega_O2 = 1580.19            # cm^-1
    De_O2 = 507.8                 # kJ/mol
    omega_O2_au = omega_O2 * CM1_TO_AU_ANGULAR_FREQUENCY
    De_O2_au = De_O2 * KJMOL_TO_HARTREE
    masses_O2 = get_mass_vector(["O", "O"])
    reduced_mass_O2 = masses_O2[0] * masses_O2[1] / sum(masses_O2)
    beta_O2_au = omega_O2_au / math.sqrt(2.0 * De_O2_au / reduced_mass_O2)
    beta_O2 = beta_O2_au / BOHR_TO_ANGSTROM  # Angstrom^-1

    oxygen = Fragment.Diatom_Init(
        fname="O2",
        atoms=["O", "O"],
        req=req_O2,
        beta=beta_O2,
        De=De_O2,
        random_rot=True,
        diatom="morse",
    )
    oxygen.Specify_Mode_Sampling(
        init_vib_type="Q",
        init_rot_type="Jfix",
        nvib=nvib_O2,
        jrot=jrot_O2,
    )
    hydrogen = Fragment.Atom_Init(fname="H", atoms=["H"])

    print("\nH + O2 collision on the HO2_1Deltag PES")
    print("Rotating-Morse O2 initial state:")
    print(f"  v = {nvib_O2}, J = {jrot_O2}")
    print(f"  re = {req_O2:.4f} Angstrom")
    print(f"  De = {De_O2:.1f} kJ/mol")
    print(f"  beta = {beta_O2:.5f} Angstrom^-1")
    print(f"Fixed collision energy = {collision_energy:.3f} kJ/mol")
    print(f"Initial fragment COM separation = {initial_distance:.3f} Angstrom")
    oxygen.print_mode_sampling()

    # The merged atom order is O(1)=0, O(2)=1, H=2.
    pairs_to_test = {
        "nonreactive_H+O2": [
            ((0, 1), "LT", 1.8),
            ((0, 2), "GT", 12.0),
            ((1, 2), "GT", 12.0),
        ],
        "OH(1)+O(2)": [
            ((0, 2), "LT", 1.35),
            ((0, 1), "GT", 2.5),
            ((1, 2), "GT", 2.5),
        ],
        "OH(2)+O(1)": [
            ((1, 2), "LT", 1.35),
            ((0, 1), "GT", 2.5),
            ((0, 2), "GT", 2.5),
        ],
    }

    reaction = Collision(oxygen, hydrogen, qchem=qcinput_pes)
    reaction.fname = "H_O2_1Deltag_collision"
    reaction.Specify_Collision_Sampling(
        Rini=initial_distance,
        bmax=bmax,
        bsampling=2,
        Ecoll=collision_energy,
        Ecoll_thermal=False,
    )
    reaction.sample_and_run_collision(
        integrator="verlet",
        timestep=timestep,
        maxstep=10000,
        iprint=4,
        pairs_to_stop=pairs_to_test,
        spectrum=True,
    )

    # Simultaneous Zhang-Xu-Truhlar spectra for all three bonds and the
    # complete HOO system. Each pair spectrum projects that pair's translation
    # and rotation, while HOO_all projects the global rigid motion. Hysteretic
    # bond thresholds identify which fixed atom channel is chemically active
    # in every STFT window.
    trajectory_positions = np.asarray(reaction.qsave).reshape((-1, 3, 3))
    roo_bohr = np.linalg.norm(
        trajectory_positions[:, 0] - trajectory_positions[:, 1], axis=1
    )

    # An asymptotic rotating-Morse O2 reference supplies E_internal for the
    # virial diagnostic 2<T_vib>/<E_internal>. It is meaningful for separated
    # bound O2; deviations during close H--O2 interaction are expected because
    # a unique fragment potential-energy partition does not then exist.
    req_O2_bohr = req_O2 / BOHR_TO_ANGSTROM
    o2_morse_potential = De_O2_au * (
        1.0 - np.exp(-beta_O2_au * (roo_bohr - req_O2_bohr))
    ) ** 2

    channel_analysis = reaction.short_time_vibrational_channels(
        dt=timestep,
        channels={
            "O_O": (0, 1),
            "H_O1": (2, 0),
            "H_O2": (2, 1),
            "HOO_all": (0, 1, 2),
        },
        dynamic_bonds={
            # (atom i, atom j, formation distance, breaking distance), Angstrom
            "O_O": (0, 1, 1.80, 2.00),
            "H_O1": (2, 0, 1.35, 1.55),
            "H_O2": (2, 1, 1.35, 1.55),
        },
        channel_frequency_bands={
            "O_O": {
                "low_frequency": (0.0, 800.0),
                "O2_stretch": (800.0, 2500.0),
            },
            "H_O1": {
                "low_frequency": (0.0, 1800.0),
                "OH_stretch": (1800.0, 4000.0),
            },
            "H_O2": {
                "low_frequency": (0.0, 1800.0),
                "OH_stretch": (1800.0, 4000.0),
            },
            "HOO_all": {
                "low_or_bend": (0.0, 1200.0),
                "O2_like": (1200.0, 2200.0),
                "OH_like": (2200.0, 4000.0),
            },
        },
        channel_potential_energy_hartree={
            "O_O": o2_morse_potential,
        },
        window_fs=200.0,
        hop_fs=10.0,
        window="boxcar",
        nfft=512,
        peak_estimator="parabolic",
        max_frequency_cm1=4000.0,
        moving_average_fs=50.0,
        output_prefix="H_O2_1Deltag_STFT",
    )
    print("Short-time channel files:", channel_analysis.output_files)
