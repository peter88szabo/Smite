from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == "__main__":
    USE_MODEL_HESSIAN = True
    INTERNAL_HESSIAN_MODEL = "model" if USE_MODEL_HESSIAN else "simple"

    from normalmode import frequency_analysis
    from optimizer import OptimizerConfig, optimize_geometry
    from utils.constants import BOHR_TO_ANGSTROM

    USE_REDUNDANT_INTERNALS = True

    xyz_methanol = """
    C       0.0000000000      0.0000000000      0.0000000000
    O       1.5000000000      0.0500000000      0.0000000000
    H      -0.4200000000      1.0700000000      0.0600000000
    H      -0.4200000000     -0.5600000000      0.9200000000
    H      -0.4200000000     -0.5000000000     -0.9200000000
    H       1.8350000000      0.5000000000      0.8350000000
    """

    qcinput_orca_singlet = {
        "qchem": "Orca",
        "path": "/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/orca",
        "nproc": 4,
        "functional": "r2scan-3c",
        "basis": "",
        "charge": 0,
        "multiplicity": 1,
        "additional": "",
        "wfu": False,
        "scratch_dir": "scratch/methanol_model_hessian_geomopt",
    }

    qcinput_xtb_singlet = {
        "qchem": "xtb",
        "path": "/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/xtb",
        "nproc": 4,
        "functional": "",
        "basis": "",
        "charge": 0,
        "multiplicity": 1,
        "additional": "",
        "wfu": False,
        "scratch_dir": "scratch/methanol_model_hessian_geomopt",
    }

    qcinp = qcinput_orca_singlet

    print("Methanol minimum optimization with optional model Hessian")
    print("charge:", qcinp["charge"])
    print("multiplicity:", qcinp["multiplicity"])
    print("method:", qcinp["functional"], qcinp["basis"])
    print("internal_hessian_model:", INTERNAL_HESSIAN_MODEL)
    print("use_redundant_internals:", USE_REDUNDANT_INTERNALS)
    print("Starting methanol geometry optimization", flush=True)

    optimizer_config = OptimizerConfig.from_dict({
        "backend_optimizer": "smite",
        "coordinates": "internal",
        "method": "BFGS",
        "convergence": {
            "maxstep": 100,
            "energy_tol": 5.0e-6,
            "max_gradient": 3.0e-4,
            "rms_gradient": 1.0e-4,
            "max_step": 4.0e-3,
            "rms_step": 2.0e-3,
        },
        "internal_coordinates": {
            "max_step_internal": 0.10,
            "use_redundant_internals": USE_REDUNDANT_INTERNALS,
        },
        "hessian": {
            "internal_hessian_model": INTERNAL_HESSIAN_MODEL,
        },
        "reporting": {
            "trajectory_file": "geomopt_methanol_model_hessian_Orca.xyz",
            "log_file": "geomopt_methanol_model_hessian_Orca.log",
            "print_report": True,
        },
    })

    result = optimize_geometry(
        qcinp,
        xyz_methanol,
        settings=optimizer_config,
    )

    print("Finished methanol geometry optimization", flush=True)
    print("coordinates:", result.coordinates)
    print("converged:", result.converged)
    print("steps:", result.nsteps)
    print("energy [hartree]:", result.energy)
    print("message:", result.message)
    print("optimized geometry [Angstrom]:")

    qopt_angstrom = result.q * BOHR_TO_ANGSTROM
    for i, atom in enumerate(result.atoms):
        j = 3 * i
        print(f"{atom:2s} {qopt_angstrom[j]:16.8f} {qopt_angstrom[j + 1]:16.8f} {qopt_angstrom[j + 2]:16.8f}")

    print("Starting methanol frequency analysis", flush=True)
    frequency_analysis(
        qcinput=qcinp,
        atoms=result.atoms,
        q=result.q,
        fname="methanol_model_hessian_Orca",
        hessFile="hessian_methanol_model_hessian_Orca.hess",
        is_eckart=True,
        force_hessian_recalc=True,
        Amp_modeanim=30.0,
        print_nmode=True,
        qrrho_cutoff=50.0,
    )
    print("Finished methanol frequency analysis", flush=True)
