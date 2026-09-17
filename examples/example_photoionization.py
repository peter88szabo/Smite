"""Complete photoionization input using previously saved neutral data.

Run from the directory containing neutral.xyz and hessian_neutral.hess, or
set absolute paths below. Set NEUTRAL_TRAJ to a saved SMite q,p trajectory
to use MD frames instead of Hessian-based QCT. No neutral MD/Hessian run is
started by this script. Adapt the methods, spin states, energies and atom
indices to the molecular system and experimental conditions.
"""

from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from smite import Molecule, Photoionization
from utils.atomic_masses import get_mass_vector
from utils.constants import ANGSTROM_TO_BOHR


if __name__ == "__main__":
    NEUTRAL_GEOM = Path("neutral.xyz")
    NEUTRAL_HESSIAN = Path("hessian_neutral.hess")
    NEUTRAL_TRAJ = None  # Or: "neutral_md.xyz"; MD takes precedence if supplied.

    # The existing neutral molecule's ground-state energy interface.
    qcinput_neutral = {
        "qchem": "PySCF",
        "functional": "B3LYP",
        "basis": "pc1",
        "charge": 0,
        "multiplicity": 1,
        "nproc": 4,
        "path": "",
        "additional": "",
        "wfu": False,
    }

    # This is the qchem input passed to Photoionization: ONE ionic PES.
    qcinput_ion = {
        "qchem": "PySCF",
        "functional": "B3LYP",
        "basis": "pc1",
        "charge": 1,
        "multiplicity": 2,
        "nproc": 4,
        "path": "",
        "additional": "",
        "wfu": False,
    }

    # Read the required neutral reference structure (standard XYZ, Angstrom).
    with NEUTRAL_GEOM.open() as handle:
        natoms = int(handle.readline())
        handle.readline()
        rows = [handle.readline().split() for _ in range(natoms)]
    if any(len(row) != 4 for row in rows):
        raise ValueError("neutral.xyz must contain symbol, x, y, z for each atom")
    atoms = [row[0] for row in rows]
    q_reference = np.array([row[1:] for row in rows], dtype=float).ravel() * ANGSTROM_TO_BOHR

    neutral = Molecule(
        atoms=atoms,
        mass=np.asarray(get_mass_vector(atoms), dtype=float),
        q_ini=q_reference,
        p_ini=np.zeros(3 * natoms),
    )
    neutral.fname = "neutral"
    neutral.qchem = qcinput_neutral

    # Your example channel dictionary: the input atom ordering must match it.
    pairs_to_test = {
        "capture_gamma_1": [((2, 17), "LT", 1.5)],
        "capture_gamma_2": [((2, 18), "LT", 1.5)],
        "capture_alpha_1": [((6, 17), "LT", 1.5)],
        "capture_alpha_2": [((6, 18), "LT", 1.5)],
        "reaction_1": [((2, 18), "GT", 8.0), ((2, 17), "GT", 8.0)],
        "reaction_2": [((4, 5), "GT", 8.0)],
    }
    use_channel_stopping = True
    if use_channel_stopping and any(
        not 0 <= index < natoms
        for conditions in pairs_to_test.values()
        for pair, _, _ in conditions
        for index in pair
    ):
        raise ValueError("Update pairs_to_test to the zero-based atom indices in neutral.xyz")

    photoion = Photoionization(
        neutral,
        qchem=qcinput_ion,
        fname="photoion",
        neutra_frankcondon_geom=NEUTRAL_GEOM,
        neutral_hessian=NEUTRAL_HESSIAN,
        neutral_traj=NEUTRAL_TRAJ,
        level=1,
        photon_energy=15.0,       # Example energies, eV; use experimental values.
        electron_energy=4.0,      # A number or (energy_grid, density_grid).
        ionic_energy=11.0,        # Binding energy, also a number or distribution.
        md_start=0,
        md_stride=1,
        sampling_seed=1234,
    )

    if NEUTRAL_TRAJ is None:
        photoion.Specify_Mode_Sampling(
            init_vib_type="Temp",
            init_rot_type="Temp",
            temp=300.0,
            random_rot=True,
            # Optional neutral-mode overrides, e.g. fix_quantum=[(0, 2)].
        )

    # Prepare the ionic launch and actually propagate it on qcinput_ion.
    photoion.sample_and_run_dynamics(
        integrator="leapfrog",
        integrator_order=4,
        timestep=0.5,
        maxstep=10000,
        iprint=1,
        pairs_to_stop=pairs_to_test if use_channel_stopping else None,
        Rstop=None if use_channel_stopping else 20.0,
        reaction_persistence_steps=1,
        thermostat=None,
        traj_file="photoion_trajectory.xyz",
        backfile="photoion_restart.xyz",
    )

    print("Termination reason:", photoion.termination_reason)
    print("Reaction channel:", photoion.termination_channel)
    print("Stopping step:", photoion.termination_step)
    print("Stopping time [fs]:", photoion.termination_time_fs)
    print("Momentum scale:", photoion.initial_state.momentum_scale)
    print("Energy residual [eV]:", photoion.initial_state.energy_residual_ev)
