from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == "__main__":
    USE_MODEL_HESSIAN = True
    INTERNAL_HESSIAN_MODEL = "model" if USE_MODEL_HESSIAN else "simple"
    TS_INITIAL_HESSIAN = "model" if USE_MODEL_HESSIAN else "exact"

    from normalmode import frequency_analysis
    from optimizer import OptimizerConfig, optimize_transition_state
    from utils.constants import BOHR_TO_ANGSTROM

    xyz_methanol_oh_ts_guess = """
    C     0.000000     0.000000     0.000000
    H     0.000000     0.000000     1.093039
    H     1.035956     0.000000    -0.347715
    H    -0.508115     0.897276    -0.359678
    O    -0.670168    -1.005520    -2.609411
    H    -0.166061    -1.826041    -2.718060
    O    -0.678016    -1.182640    -0.386696
    H    -1.017248    -1.076029    -1.419596
    """

    qcinput_xtb_doublet = {
        "qchem": "xTB",
        "path": "/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/xtb",
        "nproc": 4,
        "charge": 0,
        "multiplicity": 2,
        "additional": "",
        "wfu": False,
        "scratch_dir": "scratch/methanol_OH_TS_Orca_PBE0_augccpvdz",
    }


    qcinput_orca_doublet = {
        "qchem": "Orca",
        "path": "/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/orca",
        "nproc": 4,
        "functional": "r2scan-3c",
        "basis": "",
        "charge": 0,
        "multiplicity": 2,
        "additional": "",
        "wfu": False,
        "scratch_dir": "scratch/methanol_OH_TS_Orca_PBE0_augccpvdz",
    }


    qcinp = qcinput_orca_doublet

    result = optimize_transition_state(
        qcinp,
        xyz_methanol_oh_ts_guess,
        settings=OptimizerConfig.from_dict({
            "backend_optimizer": "smite",
            "coordinates": "internal",
            "convergence": {
                "maxstep": 150,
                "energy_tol": 5.0e-6,
                "max_gradient": 3.0e-4,
                "rms_gradient": 1.0e-4,
                "max_step": 4.0e-3,
                "rms_step": 2.0e-3,
            },
            "hessian": {
                "hess_file": "hessian_methanol_OH_TS_Orca_PBE0_augccpvdz.hess",
                "hessian_recalc_interval": 10,
                "final_hessian": True,
                "internal_hessian_correction": False,
                "internal_hessian_model": INTERNAL_HESSIAN_MODEL,
                "ts_initial_hessian": TS_INITIAL_HESSIAN,
                "adaptive_hessian_recalc": False,
                "skip_hessian_recalc_near_convergence": True,
                "hessian_recalc_near_convergence_factor": 3.0,
                "adaptive_hessian_on_negative_mode_change": False,
                "repair_ts_hessian": True,
                "ts_hessian_eigenvalue_floor": 1.0e-4,
            },
            "trust": {
                "trust_radius": 0.06,
            },
            "internal_coordinates": {
                "max_step_internal": 0.10,
                "best_fit_rms_tol": 1.0e-7,
                "use_redundant_internals": True,
            },
            "reaction": {
                "reaction_mode": "transfer",
                "reaction_transfer": (6, 7, 4),
                "mode_tracking_coordinates": "internal",
            },
            "reporting": {
                "trajectory_file": "tsopt_methanol_OH_Orca_PBE0_augccpvdz.xyz",
            },
            "project_eckart": True,
        }),
    )

    print(f"Methanol + OH TS optimization with ORCA {qcinput_orca_doublet['functional']}/{qcinput_orca_doublet['basis']}")
    print("charge:", qcinput_orca_doublet["charge"])
    print("multiplicity:", qcinput_orca_doublet["multiplicity"])
    print("nproc:", qcinput_orca_doublet["nproc"])
    print("coordinates:", result.coordinates)
    print("converged:", result.converged)
    print("steps:", result.nsteps)
    print("energy [hartree]:", result.energy)
    print("imaginary modes:", result.negative_modes)
    print("message:", result.message)
    print("optimized TS candidate geometry [Angstrom]:")

    qopt_angstrom = result.q * BOHR_TO_ANGSTROM
    for i, atom in enumerate(result.atoms):
        j = 3 * i
        print(f"{atom:2s} {qopt_angstrom[j]:16.8f} {qopt_angstrom[j+1]:16.8f} {qopt_angstrom[j+2]:16.8f}")

    frequency_analysis(
        qcinput=qcinp,
        atoms=result.atoms,
        q=result.q,
        fname="methanol_OH_TS_Orca_PBE0_augccpvdz",
        hessFile="hessian_methanol_OH_TS_Orca_PBE0_augccpvdz_final_freq.hess",
        is_eckart=True,
        force_hessian_recalc=True,
        Amp_modeanim=30.0,
        print_nmode=True,
        qrrho_cutoff=50.0,
    )
