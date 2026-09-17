import numpy as np
import os
import re
from utils.constants import BOHR_TO_ANGSTROM
from qchem_interfaces.backend_common import QCBackendError, backend_scratch_dir, parse_error
from qchem_interfaces.energy_cache import store_energy
from qchem_interfaces.numerical_hessian import central_difference_hessian
from qchem_interfaces.wavefunction_output import prepare_wavefunction_output

def makeXYZ(atoms, q):
    natoms = len(atoms)

    xyz = ""

    for i in range(natoms):
        xyz += atoms[i] + " "*6+str(q[i*3]*BOHR_TO_ANGSTROM) \
            + " "*6+str(q[i*3+1]*BOHR_TO_ANGSTROM) \
            + " "*6+str(q[i*3+2]*BOHR_TO_ANGSTROM) \
            + "\n"
    return xyz


def psi4_geometry_specification(charge, multiplicity, xyz):
    """Build a Psi4 molecule while preserving the caller's laboratory frame.

    QCT/collision coordinates are absolute Cartesian coordinates.  Psi4's
    default center-of-mass translation and principal-axis reorientation would
    otherwise make a geometry-dependent lab-frame transformation before each
    energy or gradient evaluation.
    """
    return (
        f"{int(charge)} {int(multiplicity)}\n"
        "units angstrom\n"
        "no_com\n"
        "no_reorient\n"
        f"{xyz}"
    )


def configure_psi4_runtime(psi4, qcinput):
    psi4.core.set_output_file(os.devnull, False)
    psi4.set_options({"PRINT": 0})

    if qcinput.get("scratch_dir"):
        directory = backend_scratch_dir(qcinput, "psi4", "psi4_tmp")
        psi4.core.IOManager.shared_object().set_default_path(os.path.abspath(directory) + os.sep)


def _import_psi4():
    try:
        import psi4
    except ImportError as exc:
        raise QCBackendError(
            "Psi4 is not importable. Install Psi4 in this Python environment "
            "or choose a different qchem backend."
        ) from exc
    return psi4

def Psi4_Energy(filename, q, atoms, qcinput):
    psi4 = _import_psi4()

    configure_psi4_runtime(psi4, qcinput)

    functional   = qcinput["functional"]
    basis        = qcinput["basis"]
    charge       = qcinput["charge"]
    multiplicity = qcinput["multiplicity"]
    nproc        = qcinput["nproc"]
    additional   = qcinput.get("additional", "")
    wfu          = qcinput.get("wfu", False)

    xyz = makeXYZ(atoms, q)
    mol = psi4.geometry(psi4_geometry_specification(charge, multiplicity, xyz))

    psi4.set_num_threads(nproc)

    # -------- functional type: detect meta-GGA --------
    fct = functional.lower()
    # crude but sufficient for now; includes r2scan-3c
    is_mgga = any(tag in fct for tag in ["scan", "r2scan", "m06-l", "tpss", "b97m"])

    # -------- parse additional options --------
    add = additional.lower() if additional else ""
    tight = ("tightscf" in add)

    # user may request SOSCF, but we must ignore it for meta-GGAs
    soscf_requested = ("soscf" in add)
    soscf_allowed   = (soscf_requested and not is_mgga)

    maxiter = 200
    match = re.search(r"maxiter\s*=\s*(\d+)", add)
    if match:
        maxiter = int(match.group(1))

    def _run_psi4(multiref: bool = False):
        # base SCF options (normal mode)
        opts = {
            "basis": basis,
            "reference": "uks" if multiplicity > 1 else "rks",
            "scf_type": "df",                # fast DF-SCF in normal mode
            "maxiter": maxiter,
            "guess": "read",
            "diis": True,
            "soscf": soscf_allowed,
            "e_convergence": 1e-7 if tight else 1e-6,
            "d_convergence": 1e-7 if tight else 1e-6,
        }

        if multiref:
            # more robust SCF for nasty points
            opts["scf_type"] = "pk"                      # more stable than DF
            opts["maxiter"] = max(maxiter * 2, 400)      # give it more room

            # SOSCF only if allowed (not for meta-GGAs)
            if not is_mgga:
                opts["soscf"] = True
            else:
                opts["soscf"] = False

            # robust startup
            opts["guess"] = "sad"
            opts["diis_start"] = 5
            opts["damping_percentage"] = 50
            opts["level_shift"] = 0.5

        psi4.set_options(opts)
        return psi4.energy(functional, return_wfn=True)

    # -------- try normal first, then robust fallback --------
    try:
        ene, wfn = _run_psi4(multiref=False)
    except Exception as e1:
        try:
            ene, wfn = _run_psi4(multiref=True)
        except Exception as e2:
            raise QCBackendError(
                f"Psi4 energy calculation failed. Normal SCF error: {e1}. "
                f"Robust fallback error: {e2}"
            ) from e2
    if not np.isfinite(float(ene)):
        raise QCBackendError(f"Psi4 returned a non-finite energy: {ene}")

    if wfu:
        # The caller already provides the target Molden filename.
        # Do not append a second `.molden` suffix here.
        filename = prepare_wavefunction_output(filename, "Psi4")
        psi4.molden(wfn, filename)

    return ene


# =====================================================================
# GRADIENT → FORCES
# =====================================================================
def Psi4_Force(q, atoms, qcinput):
    psi4 = _import_psi4()

    configure_psi4_runtime(psi4, qcinput)

    functional   = qcinput["functional"]
    basis        = qcinput["basis"]
    charge       = qcinput["charge"]
    multiplicity = qcinput["multiplicity"]
    nproc        = qcinput["nproc"]
    additional   = qcinput.get("additional", "")

    xyz = makeXYZ(atoms, q)
    mol = psi4.geometry(psi4_geometry_specification(charge, multiplicity, xyz))

    psi4.set_num_threads(nproc)

    # -------- functional type: detect meta-GGA --------
    fct = functional.lower()
    is_mgga = any(tag in fct for tag in ["scan", "r2scan", "m06-l", "tpss", "b97m"])

    # -------- parse additional options --------
    add = additional.lower() if additional else ""
    tight = ("tightscf" in add)

    soscf_requested = ("soscf" in add)
    soscf_allowed   = (soscf_requested and not is_mgga)

    maxiter = 200
    match = re.search(r"maxiter\s*=\s*(\d+)", add)
    if match:
        maxiter = int(match.group(1))

    def _run_psi4_grad(multiref: bool = False):
        opts = {
            "basis": basis,
            "reference": "uks" if multiplicity > 1 else "rks",
            "scf_type": "df",
            "maxiter": maxiter,
            "guess": "read",
            "diis": True,
            "soscf": soscf_allowed,
            "e_convergence": 1e-7 if tight else 1e-6,
            "d_convergence": 1e-7 if tight else 1e-6,
        }

        if multiref:
            opts["scf_type"] = "pk"
            opts["maxiter"] = max(maxiter * 2, 400)

            if not is_mgga:
                opts["soscf"] = True
            else:
                opts["soscf"] = False

            opts["guess"] = "sad"
            opts["diis_start"] = 5
            opts["damping_percentage"] = 50
            opts["level_shift"] = 0.5

        psi4.set_options(opts)
        grad = psi4.gradient(functional)
        return grad

    try:
        grad_matrix = _run_psi4_grad(multiref=False)
    except Exception as e1:
        try:
            grad_matrix = _run_psi4_grad(multiref=True)
        except Exception as e2:
            raise QCBackendError(
                f"Psi4 gradient calculation failed. Normal SCF error: {e1}. "
                f"Robust fallback error: {e2}"
            ) from e2

    grad_matrix = np.array(grad_matrix, dtype=float)
    expected = (len(atoms), 3)
    if grad_matrix.shape != expected:
        raise parse_error("Psi4", f"gradient shape is {grad_matrix.shape}, expected {expected}")
    force = -grad_matrix.flatten(order="C")
    try:
        store_energy(qcinput, q, atoms, psi4.variable("CURRENT ENERGY"), force=force)
    except Exception:
        pass
    return force


def Psi4_Hessian(q, atoms, qcinput):
    return central_difference_hessian(
        q,
        lambda coordinates: -Psi4_Force(coordinates, atoms, qcinput),
        dx=qcinput.get("hessian_dx", 0.002),
    )

def parseXYZ(xyz):
    # Split the XYZ string into lines
    lines = xyz.strip().split('\n')
    # Count the number of lines that contain atomic coordinates
    #(excluding any empty or whitespace-only lines)
    Natoms = sum(1 for line in lines if line.strip())

    q = []
    atoms = []
    # Loop through the lines and extract the positions
    for line in lines:
        parts = line.split()
        a, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
        atoms.append(a)
        q.extend([x, y, z])

    q = np.array(q)
    return Natoms, atoms, q

if __name__ == '__main__':
    import psi4

# -------- User inputs --------
    MEM_GB = "1 GB"
    NPROC = 4
    CHARGE = 0          
    MULT   = 2           # spin multiplicity (2S+1): 1=singlet, 2=doublet, 3=triplet, ...

    #trans-IsoOH Case-2 conf-11 (r2SCAN-3c optimized, but the same as with M062X-D3)
    #Coordinates from ORCA-job IsoOH_Case-2_M062Xopt_11_Compound_1 E -271.040327889081
    xyz_IsoOH = '''
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
     '''

    '''
    #============== Start of test call of Psi4: this works perfectly =========================== 
    # Build geometry (first line: charge multiplicity)
    mol = psi4.geometry(f"""{CHARGE} {MULT} {xyz_IsoOH}""")

    psi4.set_memory(MEM_GB)
    psi4.set_num_threads(NPROC)

    tight = False #True  # set False if you need more speed
    soscf = False
    maxiter = 120

    opts = {
        "reference": "uks" if MULT > 1 else "rks",
        "scf_type": "df",
        "maxiter": maxiter,
        "guess": "read",            # use last-step density
        "diis": True,
        "soscf": soscf,             # set True only if DIIS stalls
        "e_convergence": 1e-8 if tight else 1e-7,
        "d_convergence": 1e-8 if tight else 1e-7,

        # fixed fine grid for stable forces
        "dft_radial_points": 175 if tight else 99,
        "dft_spherical_points": 974 if tight else 590,
        "dft_pruning_scheme": "none",    # keep grid fixed across steps
    }

    psi4.set_options(opts)

    #E = psi4.energy("r2scan-3c")
    G = psi4.gradient("r2scan-3c")


    grad, wfn = psi4.gradient("r2scan-3c", return_wfn=True)
    forces = -grad.to_array()
    ene = wfn.energy()

    print()
    print("Energy =", ene)
    print()
    print("Gradient =", grad.to_array())
    print()
    print("Forces =", forces)

    #================== End of test call of Psi4: this works perfectly ========================
    '''


    qcinput = {
    'qchem': 'Psi4',
    'path': '',
    'nproc': 4,
    'functional': 'r2scan',
    'basis': 'sto-3g',
    'charge': 0,
    'multiplicity': 2,
    'additional': '',
    'wfu': False
    }


    natoms, atoms, q = parseXYZ(xyz_IsoOH)

    xyz = makeXYZ(atoms, q)

    print()


    #energy = Psi4_Energy("wavefunc_IsoOH", q, atoms, qcinput)
    force = Psi4_Force(q, atoms, qcinput)

    #print("energy: ", energy)
    print("force: ", force)
