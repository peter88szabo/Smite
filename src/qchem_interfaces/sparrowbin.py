import subprocess
import numpy as np
import os
from io import StringIO


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

    inputfile = "Sparrow_temp_geom_to_run.xyz"
    print_structure(atoms, q, inputfile)

    path         =  qcinput['path']
    method       =  qcinput['functional']
    charge       =  qcinput['charge']
    multiplicity =  qcinput['multiplicity']
    natom        =  len(atoms)
    arg          = "-G"

    result = callSparrowbin(inputfile, path, charge, multiplicity, method, arg, natom)
    ene = result["Energy"]
    force = -np.reshape(result["Gradient"],3*natom)

    return force


def Sparrowbin_Energy(file_wf, q, atoms, qcinput):
    inputfile = "Sparrow_temp_geom_to_run.xyz"
    print_structure(atoms, q, inputfile)

    path         =  qcinput['path']
    method       =  qcinput['functional']
    charge       =  qcinput['charge']
    multiplicity =  qcinput['multiplicity']
    natom        =  len(atoms)
    wfu          =  qcinput['wfu'] #wfu for wavefunction or dipole (molden format) calculation


    arg = "-D energ calc" #-D [ --description ] arg      ||sets a calculation description which will appear in the output
                          #dummy argument (wihtout it problematic)
    if wfu == True:
       arg = "-W"


    result = callSparrowbin(inputfile, path, charge, multiplicity, method, arg, natom)
    ene = result["Energy"]

    return ene 


def callSparrowbin(inputfile, path, charge, multiplicity, method, arg, natom):
    if not path:
            print("Cannot determine Sparrow PATH")
            exit()

    if multiplicity == 1:
        command = [path,
                  "-x", inputfile,
                  "-c", str(charge),
                  "-s", str(multiplicity),arg,
                  "-M", method]
    else:
        command = [path,
                  "-x", inputfile,
                  "-c", str(charge),
                  "-s", str(multiplicity),
                  "-M", method, "-u", arg]

    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)

    if result.returncode == 0:
        data = parseSparrowOutput(result.stdout, natom)
    else:
        print("SPARROW command failed. Exit code:", result.returncode)
    return data

def parseSparrowOutput(s, N):
    data = s.split("="*60)[1].split("Calculation:")[1]
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
    b2a = 0.52917721092

    with open(filename, "w") as file:
        file.write(str(len(atoms)) + "\n")
        file.write("This is a temporary strucutre for Sparrow to calculate energy and gradient\n")
        for i in range(0, len(atoms)):
            jx = 3 * i
            jy = 3 * i + 1
            jz = 3 * i + 2
            file.write("%3s %15.5f  %15.5f %15.5f\n" % (atoms[i], q[jx] * b2a, q[jy] * b2a, q[jz] * b2a))


