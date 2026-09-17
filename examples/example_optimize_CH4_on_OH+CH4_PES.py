from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == "__main__":
    USE_MODEL_HESSIAN = True
    INTERNAL_HESSIAN_MODEL = "model" if USE_MODEL_HESSIAN else "simple"

    from optimizer import OptimizerConfig, optimize_geometry
    from normalmode import frequency_analysis
    from utils.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM
    from utils.format_and_print import parseXYZ

    src_dir = SRC_DIR
    pes_dir = src_dir.parent / "peslib" / "OH+CH4"

    qcinput_pes = {
        "qchem": "PES",
        "pes_name": "OH+CH4",
        "pes_path": str(pes_dir),
        "wfu": False,
        "hessian_dx": 0.002,
        # The OH+CH4 PES interface accepts isolated CH4 and internally places
        # a dummy OH fragment this far away.
        "dummy_oh_distance_bohr": 100.0,
    }

    xyz_CH4 = """
    C      0.0000000000      0.0000000000      0.0000000000
    H      0.6275790000      0.6275790000      0.6275790000
    H     -0.6275790000     -0.6275790000      0.6275790000
    H     -0.6275790000      0.6275790000     -0.6275790000
    H      0.6275790000     -0.6275790000     -0.6275790000
    """

    _natoms, atoms, q_angstrom = parseXYZ(xyz_CH4)

    result = optimize_geometry(
        qcinput_pes,
        xyz_CH4,
        settings=OptimizerConfig.from_dict({
            "backend_optimizer": "smite",
            "method": "BFGS",
            "convergence": {
                "maxstep": 100,
            },
            "hessian": {
                "internal_hessian_model": INTERNAL_HESSIAN_MODEL,
            },
            "reporting": {
                "trajectory_file": "geomopt_CH4_on_OH+CH4_PES.xyz",
            },
        }),
    )

    print("CH4 optimization on OH+CH4 PES")
    print("converged:", result.converged)
    print("steps:", result.nsteps)
    print("energy [hartree]:", result.energy)
    print("optimized geometry [Angstrom]:")

    qopt_angstrom = result.q * BOHR_TO_ANGSTROM
    for i, atom in enumerate(result.atoms):
        j = 3 * i
        print(f"{atom:2s} {qopt_angstrom[j]:16.8f} {qopt_angstrom[j+1]:16.8f} {qopt_angstrom[j+2]:16.8f}")

    frequency_analysis(
        qcinput=qcinput_pes,
        atoms=result.atoms,
        q=result.q,
        fname="CH4_on_OH+CH4_PES",
        hessFile="hessian_CH4_on_OH+CH4_PES.hess",
        is_eckart=True,
        force_hessian_recalc=True,
        Amp_modeanim=30.0,
        print_nmode=True,
        qrrho_cutoff=50.0,
    )
