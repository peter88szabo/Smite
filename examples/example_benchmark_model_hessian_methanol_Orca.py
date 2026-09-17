from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def run_optimization(qcinp, xyz_methanol, model_name):
    from optimizer import OptimizerConfig, optimize_geometry

    print(f"Starting methanol optimization with internal_hessian_model={model_name}", flush=True)
    result = optimize_geometry(
        qcinp,
        xyz_methanol,
        settings=OptimizerConfig.from_dict({
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
                "use_redundant_internals": True,
            },
            "hessian": {
                "internal_hessian_model": model_name,
            },
            "reporting": {
                "trajectory_file": f"benchmark_methanol_{model_name}_hessian_Orca.xyz",
                "log_file": f"benchmark_methanol_{model_name}_hessian_Orca.log",
                "print_report": True,
            },
        }),
    )
    print(f"Finished methanol optimization with internal_hessian_model={model_name}", flush=True)
    return result


def write_summary(results, filename="benchmark_model_hessian_methanol_Orca.dat"):
    converged = [result.energy for _model, result in results if result.converged and result.energy is not None]
    reference_energy = min(converged) if converged else None

    with open(filename, "w", encoding="utf-8") as handle:
        handle.write("# Methanol minimum-search benchmark: simple Hessian vs model Hessian\n")
        handle.write("# Relative energy is measured from the lowest converged final energy.\n")
        handle.write("# model converged steps final_energy_hartree relative_energy_kcal_mol message\n")
        for model_name, result in results:
            if reference_energy is None or result.energy is None:
                rel_kcal = float("nan")
                final_energy = float("nan")
            else:
                final_energy = float(result.energy)
                rel_kcal = (result.energy - reference_energy) * 627.509474
            message = str(result.message).replace("\n", " ")
            handle.write(
                f"{model_name:8s} {int(bool(result.converged)):9d} {int(result.nsteps):5d} "
                f"{final_energy: .12f} {rel_kcal: .6f} {message}\n"
            )
    return filename


if __name__ == "__main__":
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
        "scratch_dir": "scratch/benchmark_model_hessian_methanol_Orca",
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
        "scratch_dir": "scratch/benchmark_model_hessian_methanol_xtb",
    }

    # Use the same quantum chemistry level as the methanol model-Hessian example.
    # Switch to qcinput_xtb_singlet if a fast local benchmark is preferred.
    qcinp = qcinput_orca_singlet

    print("Methanol model-Hessian benchmark")
    print("method:", qcinp["qchem"], qcinp["functional"], qcinp["basis"])
    print("charge:", qcinp["charge"])
    print("multiplicity:", qcinp["multiplicity"])

    HESSIAN_MODELS = ("simple", "model")

    benchmark_results = []
    for hessian_model in HESSIAN_MODELS:
        benchmark_results.append((hessian_model, run_optimization(qcinp, xyz_methanol, hessian_model)))

    summary_file = write_summary(benchmark_results)

    print("")
    print("Model Hessian benchmark summary")
    print("model      converged steps final_energy_hartree       relative_kcal_mol")
    converged_energies = [result.energy for _model, result in benchmark_results if result.converged]
    reference = min(converged_energies) if converged_energies else None
    for model_name, result in benchmark_results:
        final_energy = float("nan") if result.energy is None else float(result.energy)
        relative = float("nan") if reference is None or result.energy is None else (result.energy - reference) * 627.509474
        print(
            f"{model_name:8s} {str(result.converged):>9s} {result.nsteps:5d} "
            f"{final_energy: .12f} {relative: .6f}"
        )
    print("summary file:", summary_file)
