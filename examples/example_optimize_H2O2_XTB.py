from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == "__main__":
    USE_MODEL_HESSIAN = True
    INTERNAL_HESSIAN_MODEL = "model" if USE_MODEL_HESSIAN else "simple"

    import numpy as np

    from normalmode import frequency_analysis
    from optimizer import OptimizerConfig, optimize_geometry
    from utils.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM
    from utils.format_and_print import parseXYZ

    xyz_H2O2 = """
    O            0.05611300699723        0.22647282338165       -0.07718637907274
    O            0.16803213662644        0.16159915200254        1.33740570520030
    H            0.85462287631102       -0.23506870591601       -0.36444208691326
    H           -0.63093201993469        0.62266873053181        1.62422676078569
    """

    qcinput_xtb_singlet = {
        "qchem": "XTB",
        "path": "/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/xtb",
        "nproc": 4,
        "functional": "",
        "basis": "",
        "charge": 0,
        "multiplicity": 1,
        "additional": "--iterations 1000",
        "wfu": False,
        "scratch_dir": "scratch/H2O2_XTB_geomopt",
    }

    _natoms, atoms, q_angstrom = parseXYZ(xyz_H2O2)
    distortion_angstrom = np.array(
        [
            0.000,  0.000,  0.000,
            0.060, -0.035,  0.085,
           -0.045,  0.075, -0.035,
            0.040, -0.060,  0.050,
        ],
        dtype=float,
    )
    q_angstrom = np.asarray(q_angstrom, dtype=float) + distortion_angstrom
    xyz_H2O2_distorted = "\n".join(
        f"{atom:2s} {q_angstrom[3*i]:16.10f} {q_angstrom[3*i+1]:16.10f} {q_angstrom[3*i+2]:16.10f}"
        for i, atom in enumerate(atoms)
    )

    result = optimize_geometry(
        qcinput_xtb_singlet,
        xyz_H2O2_distorted,
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
                "trajectory_file": "geomopt_H2O2_XTB.xyz",
            },
        }),
    )

    print("H2O2 optimization with XTB")
    print("charge:", qcinput_xtb_singlet["charge"])
    print("multiplicity:", qcinput_xtb_singlet["multiplicity"])
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
        qcinput=qcinput_xtb_singlet,
        atoms=result.atoms,
        q=result.q,
        fname="H2O2_XTB",
        hessFile="hessian_H2O2_XTB.hess",
        is_eckart=True,
        Amp_modeanim=30.0,
        print_nmode=True,
        qrrho_cutoff=50.0,
    )
