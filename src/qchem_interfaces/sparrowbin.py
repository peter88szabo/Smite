import subprocess
import numpy as np
import os
from io import StringIO
import shutil
from utils.constants import BOHR_TO_ANGSTROM
from qchem_interfaces.backend_common import backend_scratch_dir, command_error
from qchem_interfaces.energy_cache import store_energy
from qchem_interfaces.wavefunction_output import prepare_wavefunction_output, require_wavefunction_file


def Sparrowbin_Hessian(q, atoms, qcinput):
    '''
    Numerical hessian from Sparrow by calling analytical gradient
    '''

    dx = 0.002
    ndim = len(q)
    hess = np.zeros((ndim, ndim))

    for i in range(ndim):
        q[i] += dx

        gradp1 = -Sparrowbin_Force(q, atoms, qcinput) #negative sign because it's force not gradient

        q[i] -= 2.0*dx

        gradm1 = -Sparrowbin_Force(q, atoms, qcinput) #negative sign because it's force not gradient

        hess[i,:] = 0.5 * (gradp1 - gradm1) / dx

        q[i] += dx #restore partial coordinate

    return hess


def Sparrowbin_Force(q, atoms, qcinput):
    directory = backend_scratch_dir(qcinput, 'sparrowbin', 'sparrowbin_tmp')
    inputfile = os.path.join(directory, "Sparrow_temp_geom_to_run.xyz")
    print_structure(atoms, q, inputfile)

    path         =  qcinput['path']
    method       =  qcinput['functional']
    charge       =  qcinput['charge']
    multiplicity =  qcinput['multiplicity']
    natom        =  len(atoms)
    arg          = ["-G"]

    result = callSparrowbin(inputfile, path, charge, multiplicity, method, arg, natom, cwd=directory)
    ene = result["Energy"]
    force = -np.reshape(result["Gradient"],3*natom)
    store_energy(qcinput, q, atoms, ene, force=force)

    return force


def Sparrowbin_Energy(file_wf, q, atoms, qcinput):
    directory = backend_scratch_dir(qcinput, 'sparrowbin', 'sparrowbin_tmp')

    inputfile = os.path.join(directory, "Sparrow_temp_geom_to_run.xyz")
    print_structure(atoms, q, inputfile)

    path         =  qcinput['path']
    method       =  qcinput['functional']
    charge       =  qcinput['charge']
    multiplicity =  qcinput['multiplicity']
    natom        =  len(atoms)
    wfu          =  qcinput['wfu'] #wfu for wavefunction or dipole (molden format) calculation


    arg = ["-D", "energy calculation"] #-D [ --description ] arg      ||sets a calculation description which will appear in the output
                          #dummy argument (wihtout it problematic)
    if wfu:
       arg = ["-W"]


    result = callSparrowbin(inputfile, path, charge, multiplicity, method, arg, natom, cwd=directory)
    ene = result["Energy"]

    if wfu:
        file_wf = prepare_wavefunction_output(file_wf, "Sparrow_bin")
        sparrow_wf_file = os.path.join(directory, 'wavefunction.molden.input')
        require_wavefunction_file(sparrow_wf_file, "Sparrow_bin")
        shutil.move(sparrow_wf_file, file_wf) 

    return ene 


def callSparrowbin(inputfile, path, charge, multiplicity, method, operation_args, natom, cwd=None):
    if not path:
            raise ValueError("Cannot determine Sparrow PATH")

    input_arg = os.path.basename(inputfile) if cwd is not None else inputfile
    if multiplicity == 1:
        command = [path,
                  "-x", input_arg,
                  "-c", str(charge),
                  "-s", str(multiplicity),
                  "-M", method,
                  *operation_args]
    else:
        # For open-shell calculations Sparrow expects `-u` to receive the
        # number of unpaired electrons, i.e. multiplicity - 1. The operation
        # flag (`arg`) must stay separate; otherwise the command line is malformed.
        command = [path,
                  "-x", input_arg,
                  "-c", str(charge),
                  "-s", str(multiplicity),
                  "-M", method,
                  "-u", str(multiplicity - 1),
                  *operation_args]

    result = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode == 0:
        data = parseSparrowOutput(result.stdout, natom)
    else:
        raise command_error("SPARROW", result.returncode, result.stdout, result.stderr)
    return data

def parseSparrowOutput(s, N):
    try:
        data = s.split("="*60)[1].split("Calculation:")[1]
    except IndexError as exc:
        raise ValueError("Could not parse SPARROW output.") from exc
    result = {}
    if "Energy" in data:
        E = data.split("Energy [hartree]:\n")[1].split("\n")[0]
        result["Energy"] = float(E)
    if "Gradients" in data:
        lines = data.split("Gradients [hartree/bohr]:\n")[1].split("\n")[1:N+1]
        grad = [[float(i) for i in line.split()[1:]] for line in lines]
        result["Gradient"] = np.array(grad)
    return result

def print_structure(atoms, q, filename):
    with open(filename, "w") as file:
        file.write(str(len(atoms)) + "\n")
        file.write("This is a temporary strucutre for Sparrow to calculate energy and gradient\n")
        for i in range(0, len(atoms)):
            jx = 3 * i
            jy = 3 * i + 1
            jz = 3 * i + 2
            file.write("%3s %15.5f  %15.5f %15.5f\n" % (atoms[i], q[jx] * BOHR_TO_ANGSTROM, q[jy] * BOHR_TO_ANGSTROM, q[jz] * BOHR_TO_ANGSTROM))
