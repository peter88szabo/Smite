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

    xyz_methanol = """
    C       0.0000000000      0.0000000000      0.0000000000
    O       1.4300000000      0.0000000000      0.0000000000
    H      -0.3600000000      1.0200000000      0.0000000000
    H      -0.3600000000     -0.5100000000      0.8833459119
    H      -0.3600000000     -0.5100000000     -0.8833459119
    H       1.7270000000      0.4500000000      0.7940000000
    """

    qcinput_pyscf_singlet = {
        "qchem": "PySCF",
        "path": "",
        "nproc": 4,
        "functional": "PBE0",
        "basis": "pc-1",
        "charge": 0,
        "multiplicity": 1,
        "additional": "",
        "wfu": False,
        "scratch_dir": "scratch/methanol_PySCF_PBE0_pc1",
    }

    result = optimize_geometry(
        qcinput_pyscf_singlet,
        xyz_methanol,
        settings=OptimizerConfig.from_dict({
            "backend_optimizer": "smite",
            "coordinates": "internal",
            "method": "BFGS",
            "convergence": {
                "maxstep": 100,
            },
            "hessian": {
                "internal_hessian_model": INTERNAL_HESSIAN_MODEL,
            },
            "reporting": {
                "trajectory_file": "geomopt_methanol_PySCF_PBE0_pc1.xyz",
            },
        }),
    )

    print("Methanol optimization with PySCF PBE0/pc-1")
    print("charge:", qcinput_pyscf_singlet["charge"])
    print("multiplicity:", qcinput_pyscf_singlet["multiplicity"])
    print("nproc:", qcinput_pyscf_singlet["nproc"])
    print("coordinates:", result.coordinates)
    print("converged:", result.converged)
    print("steps:", result.nsteps)
    print("energy [hartree]:", result.energy)
    print("optimized geometry [Angstrom]:")

    qopt_angstrom = result.q * BOHR_TO_ANGSTROM
    for i, atom in enumerate(result.atoms):
        j = 3 * i
        print(f"{atom:2s} {qopt_angstrom[j]:16.8f} {qopt_angstrom[j+1]:16.8f} {qopt_angstrom[j+2]:16.8f}")

    frequency_analysis(
        qcinput=qcinput_pyscf_singlet,
        atoms=result.atoms,
        q=result.q,
        fname="methanol_PySCF_PBE0_pc1",
        hessFile="hessian_methanol_PySCF_PBE0_pc1.hess",
        is_eckart=True,
        force_hessian_recalc=True,
        Amp_modeanim=30.0,
        print_nmode=True,
        qrrho_cutoff=50.0,
    )
