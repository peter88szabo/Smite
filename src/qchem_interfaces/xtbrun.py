import numpy as np
import os
import re
import shlex
from io import StringIO

from utils.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM
from qchem_interfaces.backend_common import (
    QCBackendError,
    backend_scratch_dir,
    cleanup_backend_scratch,
    parse_error,
    run_backend_command,
    should_retry_backend,
)
from qchem_interfaces.energy_cache import store_energy
from qchem_interfaces.numerical_hessian import central_difference_hessian
from qchem_interfaces.wavefunction_output import prepare_wavefunction_output, require_wavefunction_file

def parseXTB_energy(s):
    match = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)", s)

    if match:
        energy = float(match.group(1))
    else:
        raise parse_error("xTB", "total energy was not found in stdout", stdout=s)

    return energy

def parseXTB_grad(gradfile, natom):
    if not os.path.exists(gradfile):
        raise parse_error("xTB", "gradient file was not produced", filename=gradfile)
    try:
        with open(gradfile, "r") as f:
             data = f.readlines()

        gradxyz = ''.join(data[(natom+2):(2*natom+2)])

           # Split the XYZ string into lines
        lines = gradxyz.strip().split('\n')
           # Count the number of lines that contain atomic coordinates
           #(excluding any empty or whitespace-only lines)

        grad = []
        # Loop through the lines and extract the gradient components
        for line in lines:
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"short gradient line: {line!r}")
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            grad.extend([x, y, z])
    except Exception as exc:
        raise parse_error("xTB", f"could not read gradient: {exc}", filename=gradfile, cause=exc) from exc

    grad = np.array(grad)
    expected = 3 * int(natom)
    if grad.size != expected:
        raise parse_error("xTB", f"gradient has {grad.size} elements, expected {expected}", filename=gradfile)
    return grad


def print_structure(atoms, q, filename):
    with open(filename, "w") as file:
        file.write(str(len(atoms)) + "\n")
        file.write("This is a temporary strucutre for Sparrow to calculate energy and gradient\n")
        for i in range(0, len(atoms)):
            jx = 3 * i
            jy = 3 * i + 1
            jz = 3 * i + 2
            file.write("%3s %15.5f  %15.5f %15.5f\n" % (atoms[i], q[jx] * BOHR_TO_ANGSTROM, q[jy] * BOHR_TO_ANGSTROM, q[jz] * BOHR_TO_ANGSTROM))


def _call_XTB_once(q, atoms, path, charge, multiplicity, method, arg, additional, natom, scratch_dir=None):
    fname = 'xtb_geom_file_for_abinitioMD.xyz'
    gname = 'gradient'

    directory = backend_scratch_dir({"scratch_dir": scratch_dir}, 'xtb', 'xtb_tmp')

    inputfile = os.path.join(directory, fname)
    gradfile  = os.path.join(directory, gname)

    print_structure(atoms, q, inputfile)

    extra_args = shlex.split(additional) if additional else []
    method_args = ["--gfn", method] if method else []

    if multiplicity == 1:
        # Split user-supplied xTB options into proper CLI tokens.
        # Passing "--acc 10" as one list element makes subprocess deliver it
        # as a single malformed argument, so xTB never sees the intended flag/value pair.
        command = [path, fname, "-c", str(charge), *method_args, arg, *extra_args]
    else:
        command = [path, fname, "-c", str(charge), "-u", str(multiplicity-1), *method_args, arg, *extra_args]

    result = run_backend_command(command, cwd=directory, backend_name="xTB")

    ene = parseXTB_energy(result.stdout)
    if arg == "--grad":
       grad = parseXTB_grad(gradfile, natom)
    else:
       grad = [0.0]*3*natom

    return (ene, grad)


def call_XTB(q, atoms, path, charge, multiplicity, method, arg, additional, natom, scratch_dir=None, qcinput=None):
    qcinput = qcinput or {"scratch_dir": scratch_dir}
    try:
        return _call_XTB_once(q, atoms, path, charge, multiplicity, method, arg, additional, natom, scratch_dir=scratch_dir)
    except QCBackendError:
        if not should_retry_backend(qcinput):
            raise
        cleanup_backend_scratch({"scratch_dir": scratch_dir}, "xtb", "xtb_tmp")
        return _call_XTB_once(q, atoms, path, charge, multiplicity, method, arg, additional, natom, scratch_dir=scratch_dir)



def XTB_Hessian(q, atoms, qcinput):
    return central_difference_hessian(
        q,
        lambda coordinates: -XTB_Force(coordinates, atoms, qcinput),
        dx=qcinput.get("hessian_dx", 0.002),
    )


def XTB_Force(q, atoms, qcinput):

    path         =  qcinput['path']
    method       =  qcinput['functional']
    charge       =  qcinput['charge']
    multiplicity =  qcinput['multiplicity']
    additional   =  qcinput['additional']
    scratch_dir  =  qcinput.get('scratch_dir')
    natom        =  len(atoms)
    arg          = "--grad"

    ene, grad = call_XTB(q, atoms, path, charge, multiplicity, method, arg, additional, natom, scratch_dir=scratch_dir, qcinput=qcinput) 
    force = -np.reshape(grad,3*natom)
    store_energy(qcinput, q, atoms, ene, force=force)

    return force
 

import shutil
def XTB_Energy(file_wf, q, atoms, qcinput):

    path         =  qcinput['path']
    method       =  qcinput['functional']
    charge       =  qcinput['charge']
    multiplicity =  qcinput['multiplicity']
    additional   =  qcinput['additional']
    scratch_dir  =  qcinput.get('scratch_dir')
    natom        =  len(atoms)
    wfu          =  qcinput['wfu'] #wfu for wavefunction or dipole (molden format) calculation

    arg = "--grad"
                      
    if wfu == True:
       arg = "--molden"

    ene, grad = call_XTB(q, atoms, path, charge, multiplicity, method, arg, additional,  natom, scratch_dir=scratch_dir, qcinput=qcinput)

    if not wfu:
       force = -np.reshape(grad,3*natom)
       store_energy(qcinput, q, atoms, ene, force=force)

    if wfu == True:
       file_wf = prepare_wavefunction_output(file_wf, "xTB")
       molden_dir = backend_scratch_dir({"scratch_dir": scratch_dir}, 'xtb', 'xtb_tmp')
       molden_file = os.path.join(molden_dir, "molden.input")
       require_wavefunction_file(molden_file, "xTB")
       shutil.copyfile(molden_file, file_wf)

    return ene



if __name__ == '__main__':

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

    qcinput = {
    'qchem': 'XTB',
    'path': '/home/peter/Programs/xtb-6.6.1/bin/xtb',
    'nproc': 8,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 1,
    'additional': '--acc 10',
    'wfu': False
    }


    natom = 14
    file_wf = ''


    #IRC endpoint of openchain cis-1,3,5 triene
    xyz = '''
      C      -1.185385      1.500364     -0.174799
      C       0.057100      1.525641      0.290486
      C       1.182421      0.671307     -0.090281
      C       1.182420     -0.671306     -0.090282
      C       0.057100     -1.525640      0.290485
      C      -1.185388     -1.500364     -0.174796
      H       0.285347      2.226153      1.090336
      H       2.144030      1.164815     -0.199258
      H       2.144029     -1.164815     -0.199261
      H       0.285350     -2.226154      1.090332
      H      -1.972016     -2.076478      0.294389
      H      -1.462109     -0.924082     -1.042658
      H      -1.972017      2.076476      0.294382
      H      -1.462101      0.924083     -1.042665
     '''
    natom, atoms, q = parseXYZ(xyz)

    q = np.array(q) * ANGSTROM_TO_BOHR

    ene = XTB_Energy(file_wf, q, atoms, qcinput)
    grad = XTB_Force(q, atoms, qcinput)

    print('Ene = ', ene)
    print("Gradient:")
    for i in range(natom):
    	print(grad[3*i], grad[3*i+1], grad[3*i+2] )
