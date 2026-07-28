from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == "__main__":
    import numpy as np

    from integrators.gradient import Energy, force_calc
    from utils.atomic_masses import get_mass_vector
    from utils.constants import ANGSTROM_TO_BOHR

    qcinput_pes = {
        "qchem": "PES",
        "pes_name": "HO2_3Sigma_negative",
        "path": str(SRC_DIR.parent / "peslib"),
        "wfu": False,
        "hessian_dx": 0.002,
    }

    atoms = ["H", "O", "O"]
    q_angstrom = np.array(
        [
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.95,
            0.0, 1.16, 0.0,
        ],
        dtype=float,
    )
    q = q_angstrom * ANGSTROM_TO_BOHR
    p = np.zeros_like(q)
    mass = get_mass_vector(atoms)
    wmass = np.repeat(mass, 3)

    force = force_calc(qcinput_pes, q, atoms)
    _, potential, total = Energy(qcinput_pes, None, q, p, atoms, wmass)

    print("HO2 PES energy [hartree]:", potential)
    print("HO2 PES total energy [hartree]:", total)
    print("HO2 PES force [hartree/bohr]:", force)
