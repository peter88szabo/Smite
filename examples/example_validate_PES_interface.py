from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == "__main__":
    from qchem_interfaces import validate_pes_interface

    qcinput_pes = {
        "qchem": "PES",
        "pes_name": "OH+CH4",
        "path": "peslib",
        "charge": 0,
        "multiplicity": 2,
        "nproc": 1,
        "wfu": False,
    }

    xyz_ch4 = """
    C   0.0000000000   0.0000000000   0.0000000000
    H   0.0000000000   0.0000000000   1.0890000000
    H   1.0267190000   0.0000000000  -0.3630000000
    H  -0.5133600000  -0.8891650000  -0.3630000000
    H  -0.5133600000   0.8891650000  -0.3630000000
    O   3.0000000000   0.0000000000   0.0000000000
    H   3.9600000000   0.0000000000   0.0000000000
    """

    result = validate_pes_interface(
        qcinput_pes,
        xyz_ch4,
        coordinate_index=0,
        displacement=1.0e-4,
        tolerance=1.0e-4,
        print_report=True,
    )

    if not result.passed:
        raise SystemExit("PES validation failed")
