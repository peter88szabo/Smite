import sys
import os
import shutil
import numpy as np

from utils.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM
from qchem_interfaces.backend_common import (
    QCBackendError,
    backend_scratch_dir,
    cleanup_backend_scratch,
    command_error,
    parse_error,
    run_backend_command,
    should_retry_backend,
)
from qchem_interfaces.energy_cache import store_energy
from qchem_interfaces.wavefunction_output import prepare_wavefunction_output, require_wavefunction_file

ORCA_MD_BASENAME = "orca_file_for_abinitioMD"
ORCA_MD_DIRECTORY = "orca_tmp"
ORCA_MD_PREVIOUS_GBW = "orca_previous_step.gbw"

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


def makeXYZ(atoms, q):
    natoms = len(atoms)

    xyz = ""

    for i in range(natoms):
        xyz += atoms[i] + " "*6+str(q[i*3]*BOHR_TO_ANGSTROM) \
            + " "*6+str(q[i*3+1]*BOHR_TO_ANGSTROM) \
            + " "*6+str(q[i*3+2]*BOHR_TO_ANGSTROM) \
            + "\n"
    return xyz


def generateInput(inputfile, nproc, xyz, method, basis, charge, multiplicity, additional,
                  moinp_gbw=None, guess_mode=None):
    additional = additional or ""
    with open(inputfile + ".inp", 'w') as f:
        f.write("%pal nprocs " + str(nproc) + " end \n")
        if moinp_gbw is not None and guess_mode is not None:
            # Only restart from orbitals after the very first MD step, when a
            # previous successful gbw file actually exists.
            f.write("%scf\n")
            f.write("  Guess MORead\n")
            f.write('  MOInp "' + moinp_gbw + '"\n')
            f.write("  GuessMode " + guess_mode + "\n")
            f.write("end\n")
        #f.write("! PAL" + str(nproc) + " \n")
        f.write("! " + method + " " + basis + " engrad  " + additional + "\n")
        f.write("* xyz " + str(charge) + " " + str(multiplicity) + "\n")
        f.write(xyz)
        f.write("*" + "\n")

def generate_Optimizer_Input(what, inputfile, nproc, xyz, method, basis, charge, multiplicity, additional):
    additional = additional or ""
    with open(inputfile + ".inp", 'w') as f:
        f.write("%pal nprocs " + str(nproc) + " end \n")
        #f.write("! PAL" + str(nproc) + " \n")
        # Keep ORCA keywords separated even when `additional` is provided without a leading space.
        f.write("! " + method + " " + basis + " opt freq " + additional + "\n")
        #---------------------------------------------------------------------------------------------------------------
        if what == 'TS':
            f.write("%geom\n")
            f.write("TS_search EF\n")   # Switch on TS search, EF means "eigenvector following", Same as OptTS
            f.write("Calc_Hess true\n")    # Request an exact analytical Hessian in the first optimization step.
            f.write("Recalc_Hess 5\n")     # Recalculate the exact Hessian every 5 steps.
            f.write("MaxIter 100\n")       # max. number of geometry iterations  (default is 3N (N = number of atoms), at least 50 )
            f.write("ProjectTR true\n")    #project out translation and rotation, default is false
            f.write("end\n")
        #---------------------------------------------------------------------------------------------------------------

        f.write("* xyz " + str(charge) + " " + str(multiplicity) + "\n")
        f.write(xyz)
        f.write("*" + "\n")

def callORCA(inputfile, orcapath):
    command = [orcapath, inputfile + ".inp"]
    result = run_backend_command(command, backend_name="ORCA", check=False)
    stdout = result.stdout or ""
    stderr = result.stderr or ""

    with open(inputfile + ".out", 'w') as f:
        f.write(stdout)

    if result.returncode != 0 or "ORCA finished by error termination" in stdout:
        raise command_error("ORCA", result.returncode, stdout, stderr, command=command)

def eatLines(f,num):
    for i in range(num):
        f.readline()

def read_Energy_and_Grad(inputfile):
    engrad_file = inputfile + ".engrad"
    if not os.path.exists(engrad_file):
        raise parse_error("ORCA", "engrad file was not produced", filename=engrad_file)
    try:
        with open(inputfile + ".engrad", "r") as f:
            eatLines(f,3)
            Natoms = int(f.readline())
            eatLines(f,3)
            Energy = float(f.readline())
            eatLines(f,3)
            grad = []
            for i in range(Natoms*3):
                line = f.readline()
                if not line:
                    raise ValueError(f"gradient ended after {len(grad)} values, expected {Natoms * 3}")
                grad.append(float(line))
            return (Energy, np.array(grad))
    except Exception as exc:
        raise parse_error("ORCA", f"could not parse engrad file: {exc}", filename=engrad_file, cause=exc) from exc


def get_orca_md_guess_mode(qcinput, multiplicity):
    guess_mode = qcinput.get("scf_guess_mode", "AUTO")

    if guess_mode is None:
        return None

    if isinstance(guess_mode, str):
        guess_mode = guess_mode.strip()
        if guess_mode.lower() == "none":
            return None
        if guess_mode.upper() == "AUTO":
            # Closed-shell trajectories usually benefit from the cheaper
            # FMatrix projection, while open-shell jobs are safer with the
            # corresponding-orbital projection.
            return "CMatrix" if multiplicity > 1 else "FMatrix"

    allowed_guess_modes = {"FMatrix", "CMatrix"}
    if guess_mode not in allowed_guess_modes:
        raise ValueError(
            "Unsupported ORCA scf_guess_mode. Use None, 'AUTO', 'FMatrix', or 'CMatrix'."
        )
    return guess_mode

def read_Energy_and_Optimized_Geoms(inputfile):
    converged = False
    with open(inputfile + ".out", "r") as f:
        for line in f:
            if 'THE OPTIMIZATION HAS CONVERGED' in line:
                converged = True
                break

   # Even the unconverged final geometry may still be useful to inspect.
   # ORCA writes the optimized structure in XYZ format:
   # line 1 = Natoms, line 2 = comment/energy, then one atom per line.
    with open(inputfile + ".xyz", "r") as f:
        Natoms = int(f.readline())
        comment = f.readline().strip()
        try:
            Energy = float(comment.split()[-1])
        except ValueError:
            raise ValueError("Could not parse ORCA optimized-geometry energy from xyz comment line")
        optxyz = []
        for i in range(Natoms):
            parts = f.readline().split()
            if len(parts) < 4:
                raise ValueError("Malformed ORCA optimized xyz file")
            optxyz.extend([float(parts[1]), float(parts[2]), float(parts[3])])
        return (converged, Energy, optxyz)


def _runOrca_once(Natoms, xyz, path, nproc, method, basis, charge, multiplicity, additional,
                  guess_mode=None, scratch_dir=None):
    fname = ORCA_MD_BASENAME
    directory = backend_scratch_dir({"scratch_dir": scratch_dir}, "orca", ORCA_MD_DIRECTORY)

    inputfile = os.path.join(directory, fname)
    previous_gbw = os.path.join(directory, ORCA_MD_PREVIOUS_GBW)
    moinp_gbw = previous_gbw if guess_mode is not None and os.path.exists(previous_gbw) else None
    engrad_file = inputfile + ".engrad"
    if os.path.exists(engrad_file):
        os.remove(engrad_file)

    generateInput(
        inputfile, nproc, xyz, method, basis, charge, multiplicity, additional,
        moinp_gbw=moinp_gbw, guess_mode=guess_mode if moinp_gbw is not None else None
    )

    try:
        callORCA(inputfile, path)
    except RuntimeError as exc:
        if moinp_gbw is None:
            raise
        text = str(exc)
        guess_failed = "ORCA finished by error termination in GUESS" in text or "No orbitals were found" in text
        if not guess_failed:
            raise
        if os.path.exists(previous_gbw):
            os.remove(previous_gbw)
        if os.path.exists(engrad_file):
            os.remove(engrad_file)
        generateInput(
            inputfile, nproc, xyz, method, basis, charge, multiplicity, additional,
            moinp_gbw=None, guess_mode=None
        )
        callORCA(inputfile, path)

    current_gbw = inputfile + ".gbw"
    if guess_mode is not None and os.path.exists(current_gbw):
        # Keep the restart file under a different basename than the active job
        # so ORCA does not overwrite the MOInp file at the next startup.
        shutil.copyfile(current_gbw, previous_gbw)

    ene, grad = read_Energy_and_Grad(inputfile)
    return ene, grad 


def runOrca(Natoms, xyz, path, nproc, method, basis, charge, multiplicity, additional,
            guess_mode=None, scratch_dir=None, qcinput=None):
    qcinput = qcinput or {"scratch_dir": scratch_dir}
    try:
        return _runOrca_once(
            Natoms, xyz, path, nproc, method, basis, charge, multiplicity, additional,
            guess_mode=guess_mode, scratch_dir=scratch_dir
        )
    except QCBackendError:
        if not should_retry_backend(qcinput):
            raise
        cleanup_backend_scratch({"scratch_dir": scratch_dir}, "orca", ORCA_MD_DIRECTORY)
        return _runOrca_once(
            Natoms, xyz, path, nproc, method, basis, charge, multiplicity, additional,
            guess_mode=None, scratch_dir=scratch_dir
        )


def runOrcaOptimizer(what, Natoms, xyz, path, nproc, method, basis, charge, multiplicity, additional, scratch_dir=None):
    fname = 'orca_file_for_geomopt'
    directory = backend_scratch_dir({"scratch_dir": scratch_dir}, "orca", ORCA_MD_DIRECTORY)

    inputfile = os.path.join(directory, fname)

    generate_Optimizer_Input(what, inputfile, nproc, xyz, method, basis, charge, multiplicity, additional)

    callORCA(inputfile, path)

    return read_Energy_and_Optimized_Geoms(inputfile)


def readOrcaHessian_OLD(inputfile):
    matr = np.matrix([])
    with open(inputfile + ".hess", 'r') as f:
        while f.readline() != "$hessian\n":
            pass
        lines = int(f.readline())
        for k in range(lines//5+1):
            subm = [];
            f.readline()
            for i in range(lines):
                row = [float(y) for y in [x for x in f.readline().split(" ") if x][1:] ];
                subm.append(row);
            if matr.size == 0:
                matr = np.matrix(subm);
            else:
                matr = np.concatenate([matr, np.matrix(subm)], axis=1)
        return matr

def readOrcaHessian(inputfile):
    path = inputfile + ".hess"
    with open(path, "r") as f:
        # jump to $hessian section
        for line in f:
            if line.strip() == "$hessian":
                break
        n = int(next(f).strip())            # number of rows/cols (3N typically)
        nblocks = (n + 4) // 5              # ceil(n/5)

        H = np.zeros((n, n), dtype=float)

        for b in range(nblocks):
            _ = next(f)                     # blank / header line before each block
            ncols = min(5, n - b*5)         # columns in this block
            for i in range(n):
                parts = next(f).split()     # split on any whitespace
                # parts[0] is the row index label; values follow
                vals = [float(x) for x in parts[1:1+ncols]]
                if len(vals) != ncols:
                    raise ValueError(f"Short line while reading block {b}, row {i}: {parts}")
                H[i, b*5:b*5+ncols] = vals

    return H



def getOrcaHessian(
    Natoms,
    xyz,
    path,
    nproc,
    method,
    basis,
    charge,
    multiplicity,
    additional,
    scratch_dir=None,
    hessian_keyword="freq",
):
    fname = 'orca_file_for_abinitioMD'
    directory = backend_scratch_dir({"scratch_dir": scratch_dir}, "orca", ORCA_MD_DIRECTORY)

    inputfile = os.path.join(directory, fname)

   #Create Orca input
    with open(inputfile + ".inp", 'w') as f:
        f.write("%pal nprocs " + str(nproc) + " end \n")
        #f.write("! PAL" + str(nproc) + "\n")
        f.write("! " + method + " " + basis + " " + hessian_keyword + " " + additional + "\n")
        # ORCA requires the coordinate block to start on the next line after
        # `* xyz charge mult`. Without this newline the first atom is appended
        # to the header, producing a malformed Hessian input.
        f.write("* xyz " + str(charge) + " " + str(multiplicity) + "\n")
        f.write(xyz)
        f.write("*" + "\n")

    callORCA(inputfile, path)

    return readOrcaHessian(inputfile)

def Orca_Hessian(Natoms, xyz, qcinput):
    path         =  qcinput['path']
    nproc        =  qcinput['nproc']
    method       =  qcinput['functional']
    basis        =  qcinput['basis']
    charge       =  qcinput['charge']
    multiplicity =  qcinput['multiplicity']
    additional   =  qcinput['additional']
    scratch_dir  =  qcinput.get('scratch_dir')
    hessian_keyword = qcinput.get("orca_hessian_keyword", qcinput.get("hessian_keyword", "freq"))
    return getOrcaHessian(
        Natoms,
        xyz,
        path,
        nproc,
        method,
        basis,
        charge,
        multiplicity,
        additional,
        scratch_dir=scratch_dir,
        hessian_keyword=hessian_keyword,
    )


def Orca_GeomOpt(Natoms, xyz, qcinput):
    path         =  qcinput['path']
    nproc        =  qcinput['nproc']
    method       =  qcinput['functional']
    basis        =  qcinput['basis']
    charge       =  qcinput['charge']
    multiplicity =  qcinput['multiplicity']
    additional   =  qcinput['additional']
    what         =  qcinput['what']
    scratch_dir  =  qcinput.get('scratch_dir')
    return runOrcaOptimizer(what, Natoms, xyz, path, nproc, method, basis, charge, multiplicity, additional, scratch_dir=scratch_dir) 


def Orca_Force(q, atoms, qcinput):
    path         =  qcinput['path']
    nproc        =  qcinput['nproc']
    method       =  qcinput['functional']
    basis        =  qcinput['basis']
    charge       =  qcinput['charge']
    multiplicity =  qcinput['multiplicity']
    additional   =  qcinput['additional']
    scratch_dir  =  qcinput.get('scratch_dir')
    guess_mode   =  get_orca_md_guess_mode(qcinput, multiplicity)

    xyz = makeXYZ(atoms, q)
    Natoms = len(atoms)

    ene, grad = runOrca(
        Natoms, xyz, path, nproc, method, basis, charge, multiplicity, additional,
        guess_mode=guess_mode, scratch_dir=scratch_dir, qcinput=qcinput
    )
    force = -np.array(grad)
    store_energy(qcinput, q, atoms, ene, force=force)

    return force

def Orca_Energy(filename, q, atoms, qcinput):

    xyz = makeXYZ(atoms, q)
    Natoms = len(atoms)

    fname = ORCA_MD_BASENAME
    scratch_dir = qcinput.get('scratch_dir')
    directory = backend_scratch_dir({"scratch_dir": scratch_dir}, "orca", ORCA_MD_DIRECTORY)

    inputfile = os.path.join(directory, fname)

    path         =  qcinput['path']
    nproc        =  qcinput['nproc']
    method       =  qcinput['functional']
    basis        =  qcinput['basis']
    charge       =  qcinput['charge']
    multiplicity =  qcinput['multiplicity']
    additional   =  qcinput['additional']
    wfu          =  qcinput['wfu']
    guess_mode   =  get_orca_md_guess_mode(qcinput, multiplicity)

    ene, grad = runOrca(
        Natoms, xyz, path, nproc, method, basis, charge, multiplicity, additional,
        guess_mode=guess_mode, scratch_dir=scratch_dir, qcinput=qcinput
    )
    store_energy(qcinput, q, atoms, ene, force=-np.array(grad))
   
    if wfu == True:
       filename = prepare_wavefunction_output(filename, "ORCA")
       gbwfile = inputfile
       command_create_molden = [path + "_2mkl", gbwfile, "-molden"]
  
       run_backend_command(command_create_molden, backend_name="ORCA gbw-to-Molden conversion")

       moldenfile = inputfile + ".molden.input"
       require_wavefunction_file(moldenfile, "ORCA")
       shutil.copy(moldenfile, filename)

    return ene


if __name__ == "__main__":


    #=========== Input parameters ====================
    qchem = 'Orca'
    method = 'am1'
    base = ''
    charge = 0
    multiplicity = 1
    path = '/home/peter/Programs/Orca.5.0.4/orca'
    nproc = 1
    additional = ''
    what = 'minimum'

    qcinput = {
    'qchem': qchem,
    'path': path,
    'nproc': nproc,
    'functional': method,
    'basis': base,
    'charge': charge,
    'multiplicity': multiplicity,
    'additional': additional,
    'wfu': False,
    'what': what
    }

    #b3lyp pc-1 optimized structure
    xyz = '''
      C   0.03012969409665      0.00001733670207      0.01195503879116
      C   0.00092680373556      0.00001295536402      1.40500238110811
      C   1.21012210555252      0.00000519141787      2.09645169860924
      C   2.41934864674152      0.00000073033748      1.39735747618202
      C   2.42556896950511      0.00000462477034      0.00064222741084
      C   1.22482516029362      0.00001333122267     -0.70529107890890
      H   -0.95641121721691      0.00001442187302      1.91849769179603
      H   1.20776195081117      0.00000151033234      3.18488546455400
      H   3.36133019449597     -0.00000574367627      1.94367059725995
      H   3.36904118433316      0.00000038988732     -0.54208061229342
      H   1.19482162635062      0.00001495740695     -1.79122865844198
      N   -1.24921099188054      0.00002129911109     -0.73027910494101
      O   -1.19427128638801     -0.00005093501480     -1.95529096603286
      O   -2.28582084043044     -0.00005206973408     -0.07509415509316
     '''
    Natoms, atoms, q_eq = parseXYZ(xyz)

    q = np.array(q_eq) * ANGSTROM_TO_BOHR

#-----------------------------------------------------------------
    fname = "Nitro-Benzene"

    ene, grad = runOrca(Natoms, xyz, path, nproc, method, base, charge, multiplicity, additional)

    print("Ene:", ene)
    print("Grad:", grad)


    #print("Hessian calculation is running")
    #hessian = Orca_Hessian(Natoms, xyz, qcinput)
    #print()
    #print(hessian)

    print() 
    print("geometry optimization has been started")
    ene, ggg = Orca_GeomOpt(Natoms, xyz, qcinput)
    print("geometry optimization is done")
