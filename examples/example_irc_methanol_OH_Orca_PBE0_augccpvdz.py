from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == "__main__":
    USE_MODEL_HESSIAN = True
    INTERNAL_HESSIAN_MODEL = "model" if USE_MODEL_HESSIAN else "simple"

    from optimizer import follow_irc

    xyz_methanol_oh_ts = """
    C     0.000000     0.000000     0.000000
    H     0.000000     0.000000     1.093039
    H     1.035956     0.000000    -0.347715
    H    -0.508115     0.897276    -0.359678
    O    -0.670168    -1.005520    -2.609411
    H    -0.166061    -1.826041    -2.718060
    O    -0.678016    -1.182640    -0.386696
    H    -1.017248    -1.076029    -1.419596
    """

    xyz_methanol_oh_reactant = """
    C     0.000000     0.000000     0.000000
    H     0.000000     0.000000     1.093039
    H     1.035956     0.000000    -0.347715
    H    -0.508115     0.897276    -0.359678
    O    -0.670168    -1.005520    -3.050000
    H    -0.166061    -1.826041    -3.158649
    O    -0.678016    -1.182640    -0.386696
    H    -1.017248    -1.076029    -1.419596
    """

    xyz_methanol_oh_product = """
    C     0.000000     0.000000     0.000000
    H     0.000000     0.000000     1.093039
    H     1.035956     0.000000    -0.347715
    H    -0.508115     0.897276    -0.359678
    O    -0.670168    -1.005520    -2.430000
    H    -0.166061    -1.826041    -2.538649
    O    -0.678016    -1.182640    -0.386696
    H    -0.778000    -1.035000    -1.475000
    """
    qcinput_xtb_doublet = {
        "qchem": "xTB",
        "path": "/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/xtb",
        "nproc": 4,
        "charge": 0,
        "multiplicity": 2,
        "wfu": False,
        "scratch_dir": "scratch/methanol_OH_IRC_Orca_PBE0_augccpvdz",
    }

    qcinput_orca_doublet = {
        "qchem": "Orca",
        "path": "/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/orca",
        "nproc": 4,
        "functional": "PBE0",
        "basis": "aug-cc-pvdz",
        "charge": 0,
        "multiplicity": 2,
        "additional": "",
        "wfu": False,
        "scratch_dir": "scratch/methanol_OH_IRC_Orca_PBE0_augccpvdz",
    }

    result = follow_irc(
        qcinput_xtb_doublet,
        xyz_methanol_oh_ts,
        hess_file="hessian_methanol_OH_IRC_TS_Orca_PBE0_augccpvdz.hess",
        initial_displacement=0.02,
        step_size=0.05,
        step_forward=10,
        step_backward=10,
        adaptive=True,
        step_size_min=0.01,
        step_size_max=0.20,
        predictor="euler",
        corrector_tol=1.0e-4,
        corrector_maxiter=20,
        optimize_endpoints=False,
        endpoint_optimizer_kwargs={
            "backend_optimizer": "smite",
            "coordinates": "internal",
            "maxstep": 100,
            "use_redundant_internals": True,
            "internal_hessian_model": INTERNAL_HESSIAN_MODEL,
        },
        endpoint_reference_initial=xyz_methanol_oh_reactant,
        endpoint_reference_final=xyz_methanol_oh_product,
        endpoint_compare_align=True,
        endpoint_compare_tol=0.25,
        trajectory_prefix="irc_methanol_OH_Orca_PBE0_augccpvdz",
        profile_file="irc_methanol_OH_Orca_PBE0_augccpvdz_energy_profile.dat",
        plot_file="irc_methanol_OH_Orca_PBE0_augccpvdz_energy_profile.png",
        print_report=True,
    )

    print("Methanol + OH IRC with ORCA PBE0/aug-cc-pVDZ")
    print("charge:", qcinput_orca_doublet["charge"])
    print("multiplicity:", qcinput_orca_doublet["multiplicity"])
    print("nproc:", qcinput_orca_doublet["nproc"])
    print("forward points:", len(result.forward.points))
    print("backward points:", len(result.backward.points))
    print("combined trajectory points:", len(result.trajectory))
    print("combined trajectory file:", result.trajectory_file)
    print("energy profile file:", result.profile_file)
    print("energy profile plot:", result.plot_file)
    print("endpoint comparison:", result.endpoint_comparison)
    print("forward message:", result.forward.message)
    print("backward message:", result.backward.message)
