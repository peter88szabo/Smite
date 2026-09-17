from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == "__main__":
    USE_MODEL_HESSIAN = True
    INTERNAL_HESSIAN_MODEL = "model" if USE_MODEL_HESSIAN else "simple"

    from optimizer import optimize_spin_crossing
    from utils.constants import BOHR_TO_ANGSTROM

    xyz_ch3oplus = """
    C             0.000000       0.000000       0.000000
    O             0.000000       0.000000       1.380000
    H             1.008807       0.000000      -0.356663
    H            -0.772790      -0.648448      -0.356667
    H            -0.206997       1.173936       0.675224
    """

    qcinput_orca_ch3oplus = {
        "qchem": "xtb",
        "path": "/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/xtb",
        "nproc": 4,
        "functional": "",
        "basis": "",
        "charge": 1,
        "spin_multiplicities": (1, 3),
        "additional": "VeryTightSCF",
        "wfu": False,
        "scratch_dir": "scratch/CH3Oplus_spincross_Orca_PBE0_ma-def2-SVP",
    }

    result = optimize_spin_crossing(
        qcinput_orca_ch3oplus,
        xyz_ch3oplus,
        # Available methods: BFGS, Bofill, SR1, PSB, Broyden, BBGrad1, BBGrad2.
        method="BFGS",
        coordinates="internal",
        use_redundant_internals=True,
        internal_hessian_model=INTERNAL_HESSIAN_MODEL,
        maxstep=40,
        energy_tol=5.0e-5,
        max_step=4.0e-3,
        rms_step=2.5e-3,
        max_gradient=7.0e-4,
        rms_gradient=5.0e-4,
        trajectory_file="spincross_CH3Oplus_Orca_PBE0_ma-def2-SVP.xyz",
        print_report=True,
    )

    print("CH3O+ spin-crossing optimization with ORCA PBE0/ma-def2-SVP")
    print("charge:", qcinput_orca_ch3oplus["charge"])
    print("spin multiplicities:", qcinput_orca_ch3oplus["spin_multiplicities"])
    print("coordinates:", result.coordinates)
    print("redundant internals:", result.use_redundant_internals)
    print("converged:", result.converged)
    print("steps:", result.nsteps)
    print("energy A [hartree]:", result.energy_a)
    print("energy B [hartree]:", result.energy_b)
    print("energy gap [hartree]:", result.energy_gap)
    print("optimized crossing geometry [Angstrom]:")

    qopt_angstrom = result.q * BOHR_TO_ANGSTROM
    for i, atom in enumerate(result.atoms):
        j = 3 * i
        print(f"{atom:2s} {qopt_angstrom[j]:16.8f} {qopt_angstrom[j+1]:16.8f} {qopt_angstrom[j+2]:16.8f}")
