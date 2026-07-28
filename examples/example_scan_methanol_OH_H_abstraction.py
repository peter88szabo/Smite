from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == "__main__":
    USE_MODEL_HESSIAN = True
    INTERNAL_HESSIAN_MODEL = "model" if USE_MODEL_HESSIAN else "simple"

    from optimizer import scan_coordinate

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

    qcinput_orca_doublet = {
        "qchem": "Orca",
        "path": "/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/orca",
        "nproc": 4,
        "functional": "b97-3c",
        "basis": "",
        "charge": 0,
        "multiplicity": 2,
        "additional": "",
        "wfu": False,
        "scratch_dir": "scratch/methanol_OH_TS_Orca_PBE0_augccpvdz",
    }


    qcinput_xtb_doublet = {
        "qchem": "XTB",
        "path": "/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/xtb",
        "nproc": 4,
        "charge": 0,
        "multiplicity": 2,
        "additional": "",
        "wfu": False,
        "scratch_dir": "scratch/methanol_OH_TS_Orca_PBE0_augccpvdz",
    }


    qcinp = qcinput_orca_doublet

    print("\nMethanol + OH relaxed H-abstraction scan")
    print("method:", qcinput_orca_doublet["functional"], qcinput_orca_doublet["basis"])

    # Same atom order as test_tsopt_methanol_OH_Orca_PBE0_avdz.py:
    #   methanol O=6, transferred H=7, OH radical O=4.
    # The relaxed scan constrains the forming H---OH distance, H7---O4.
    result = scan_coordinate(
        qcinp,
        xyz_methanol_oh_ts_guess,
        scan_mode="bond",
        scan_bond=(7, 4),
        start=2.20,
        stop=0.98,
        nsteps=7,
        unit="angstrom",
        relaxed=True,
        use_redundant_internals=True,
        internal_hessian_model=INTERNAL_HESSIAN_MODEL,
        relax_maxiter=80,
        relax_max_step_internal=0.10,
        relax_energy_tol=5.0e-6,
        relax_max_gradient=3.0e-4,
        relax_rms_gradient=1.0e-4,
        relax_max_step=4.0e-3,
        relax_rms_step=2.0e-3,
        best_fit_iters=20,
        best_fit_rms_tol=1.0e-7,
        trajectory_file="scan_methanol_OH_H7_O4_relaxed.xyz",
        print_report=True,
    )

    print("scan coordinate: H7---O4, scan_bond=(7, 4)")
    print("trajectory:", result.trajectory_file)
    print("points:", len(result.points))

    if result.points:
        highest = max(result.points, key=lambda point: point.energy)
        print("highest-energy scan point:")
        print("  step:", highest.index)
        print(f"  H7---O4 distance [Angstrom]: {highest.value:.6f}")
        print(f"  energy [hartree]: {highest.energy:.12f}")
        print("  constrained relaxation converged:", highest.converged)
        print("  constrained relaxation steps:", highest.optimization_steps)
