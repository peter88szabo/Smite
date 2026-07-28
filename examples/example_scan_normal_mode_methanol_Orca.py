from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == "__main__":
    from optimizer import scan_coordinate_2d, scan_normal_mode

    xyz_methanol = """
    C     0.000000     0.000000     0.000000
    H     0.000000     0.000000     1.093039
    H     1.035956     0.000000    -0.347715
    H    -0.508115     0.897276    -0.359678
    O    -0.678016    -1.182640    -0.386696
    H    -1.017248    -1.076029    -1.419596
    """

    qcinput_orca = {
        "qchem": "Orca",
        "path": "/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/orca",
        "nproc": 4,
        "functional": "r2scan-3c",
        "basis": "",
        "charge": 0,
        "multiplicity": 1,
        "additional": "",
        "wfu": False,
        "scratch_dir": "scratch/methanol_normal_mode_scan_orca",
    }

    qcinput_xtb = {
        "qchem": "XTB",
        "path": "/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/xtb",
        "nproc": 4,
        "charge": 0,
        "multiplicity": 1,
        "additional": "",
        "wfu": False,
        "scratch_dir": "scratch/methanol_normal_mode_scan_xtb",
    }

    qcinp = qcinput_orca

    NORMAL_MODE_INDEX = 0

    print("\nMethanol 1D normal-mode scan")
    print("method:", qcinp.get("functional", qcinp["qchem"]))
    print("normal_mode_index:", NORMAL_MODE_INDEX)

    result_1d = scan_normal_mode(
        qcinp,
        xyz_methanol,
        normal_mode_index=NORMAL_MODE_INDEX,
        start=-0.15,
        stop=0.15,
        nsteps=7,
        unit="angstrom",
        relaxed=True,
        relax_maxiter=40,
        relax_max_step=4.0e-3,
        hess_file="hessian_methanol_normal_mode_scan.hess",
        trajectory_file="scan_methanol_normal_mode_1d.xyz",
        energy_profile_file="scan_methanol_normal_mode_1d_energy_profile.dat",
        energy_profile_plot="scan_methanol_normal_mode_1d_energy_profile.png",
        print_report=True,
    )

    print("1D trajectory:", result_1d.trajectory_file)
    print("1D points:", len(result_1d.points))

    print("\nMethanol 2D scan: C-O bond x normal mode")
    result_2d = scan_coordinate_2d(
        qcinp,
        xyz_methanol,
        scan1_mode="bond",
        scan1_bond=(0, 4),
        scan1_start=1.35,
        scan1_stop=1.55,
        scan1_nsteps=3,
        scan1_unit="angstrom",
        scan2_mode="normal_mode",
        scan2_normal_mode_index=NORMAL_MODE_INDEX,
        scan2_start=-0.10,
        scan2_stop=0.10,
        scan2_nsteps=3,
        scan2_unit="angstrom",
        relaxed=True,
        relax_maxiter=40,
        relax_max_step=4.0e-3,
        use_redundant_internals=True,
        hess_file="hessian_methanol_normal_mode_scan.hess",
        trajectory_file="scan_methanol_CO_bond_x_normal_mode_2d.xyz",
        energy_profile_file="scan_methanol_CO_bond_x_normal_mode_2d_energy_profile.dat",
        energy_profile_plot="scan_methanol_CO_bond_x_normal_mode_2d_energy_profile.png",
        print_report=True,
    )

    print("2D trajectory:", result_2d.trajectory_file)
    print("2D points:", len(result_2d.points))
