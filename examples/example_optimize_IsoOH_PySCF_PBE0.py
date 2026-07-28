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

    # trans-IsoOH Case-2 conf-11 from test_IsoOH+O2.py.
    xyz_IsoOH = """
    C          -0.04133087943945      0.07114250078738      1.38999071346232
    C           0.02070690184692      0.01666510282459      0.00788362371526
    C          -1.26856364750845      0.00193604142078     -0.78526109220447
    C           1.25671559933393     -0.01291622892551     -0.64027788592838
    C           1.47848498555495     -0.08846749550178     -2.10972256189703
    H          -0.99051468103380      0.09713273461484      1.91341807774138
    H           0.86502010385865      0.09006595320454      1.98672613761931
    H           2.15887766668238      0.00324193447454     -0.02997803564743
    H           0.52959627377747     -0.17538849005273     -2.65884146455451
    H           1.98306008511518      0.81705697518545     -2.47372434820703
    H          -1.35703924866629      0.88913674900855     -1.42133762484310
    H          -1.32867369902273     -0.87570639962484     -1.43815472591279
    H          -2.13252287462749     -0.01861509393368     -0.11705314031457
    O           2.36925369873426     -1.16389173263393     -2.45883424444901
    H           2.02962571539443     -1.96081955084824     -2.03469142857991
    """

    qcinput_PySCF_doublet = {
        "qchem": "PySCF",
        "path": "",
        "nproc": 4,
        "functional": "PBE0",
        "basis": "sto-3g",
        "charge": 0,
        "multiplicity": 2,
        "additional": "",
        "wfu": False,
        "scratch_dir": "scratch/IsoOH_PySCF_PBE0",
    }

    _natoms, atoms, q_angstrom = parseXYZ(xyz_IsoOH)

    result = optimize_geometry(
        qcinput_PySCF_doublet,
        xyz_IsoOH,
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
                "trajectory_file": "geomopt_IsoOH_PySCF_PBE0.xyz",
            },
        }),
    )

    print("IsoOH radical optimization with PySCF PBE0/STO-3G")
    print("charge:", qcinput_PySCF_doublet["charge"])
    print("multiplicity:", qcinput_PySCF_doublet["multiplicity"])
    print("nproc:", qcinput_PySCF_doublet["nproc"])
    print("converged:", result.converged)
    print("steps:", result.nsteps)
    print("energy [hartree]:", result.energy)
    print("optimized geometry [Angstrom]:")

    qopt_angstrom = result.q * BOHR_TO_ANGSTROM
    for i, atom in enumerate(result.atoms):
        j = 3 * i
        print(f"{atom:2s} {qopt_angstrom[j]:16.8f} {qopt_angstrom[j+1]:16.8f} {qopt_angstrom[j+2]:16.8f}")

    frequency_analysis(
        qcinput=qcinput_PySCF_doublet,
        atoms=result.atoms,
        q=result.q,
        fname="IsoOH_PySCF_PBE0",
        hessFile="hessian_IsoOH_PySCF_PBE0.hess",
        is_eckart=True,
        force_hessian_recalc=True,
        Amp_modeanim=30.0,
        print_nmode=True,
        qrrho_cutoff=50.0,
    )
