import subprocess
import numpy as np
import os
import re
from io import StringIO

def parseXTB_energy(s):
    match = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)", s)

    if match:
        energy = float(match.group(1))
    else:
        print("Total energy not found in the XTB output.")

    return energy

def parseXTB_grad(gradfile, natom):
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
        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        grad.extend([x, y, z])

    return np.array(grad)


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


def call_XTB(q, atoms, path, charge, multiplicity, method, arg, additional, natom):
    if not path:
            print("Cannot determine XTB PATH")
            exit()

    fname = 'xtb_geom_file_for_abinitioMD.xyz'
    gname = 'gradient'

    #directory = 'xtb_tmp'
    #if not os.path.exists(directory):
    #    os.makedirs(directory)

    #inputfile = os.path.join(directory, fname)
    #gradfile  = os.path.join(directory, gname)
    inputfile = fname
    gradfile  = gname

    print_structure(atoms, q, inputfile)

    if multiplicity == 1:
        command = [path, inputfile, "-c", str(charge), arg, additional]
        command_restart = [path, inputfile, "-c", str(charge), arg, additional, "--etemp 1000.0", "--gfnff", "--acc 200"] 
        ##command_restart = [path, inputfile, "-c", str(charge), arg, additional, "--etemp 1000.0", "--acc 200"] 
        #command_restart = [path, inputfile, "-c", str(charge), arg, additional, "--etemp 1000.0", "&&",  
        #                   path, inputfile, "-c", str(charge), arg, additional, "--restart"]
    else:
        command = [path, inputfile, "-c", str(charge), "-u", str(multiplicity-1), arg, additional]
        command_restart = [path, inputfile, "-c", str(charge), "-u", str(multiplicity-1), arg, additional, "--etemp 1000.0", "--gfnff", "--acc 200"]
        ###command_restart = [path, inputfile, "-c", str(charge), "-u", str(multiplicity-1), arg, additional, "--etemp 1000.0", "--acc 200"]
        #command_restart = [path, inputfile, "-c", str(charge), "-u", str(multiplicity-1), arg, additional, "--etemp 1000.0", "&&",
        #                   path, inputfile, "-c", str(charge), "-u", str(multiplicity-1), arg, additional, "--restart"]

    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)

    if result.returncode == 0:
        ene  =  parseXTB_energy(result.stdout)
        if arg == "--grad":
           gradfile = 'gradient'
           grad =  parseXTB_grad(gradfile, natom)
        else:
           grad = [0.0]*3*natom
    else:
        result_restart = subprocess.run(command_restart, stdout=subprocess.PIPE, text=True)

        if result_restart.returncode == 0:
           print("GFNFF calc in progress")
           ene  =  parseXTB_energy(result_restart.stdout)
           if arg == "--grad":
              gradfile = 'gradient'
              grad =  parseXTB_grad(gradfile, natom)
           else:
              grad = [0.0]*3*natom
        else:
           print("SPARROW command failed even after restart. Exit code:", result.returncode)

    return (ene, grad)



def XTB_Hessian(q, atoms, qcinput):
    '''
    Numerical hessian from XTB by calling analytical gradient
    '''

    dx = 0.002
    ndim = len(q)
    hess = np.zeros((ndim, ndim))

    for i in range(ndim):
        q[i] += dx

        gradp1 = -XTB_Force(q, atoms, qcinput) #negative sign because it's force not gradient

        q[i] -= 2.0*dx

        gradm1 = -XTB_Force(q, atoms, qcinput) #negative sign because it's force not gradient

        hess[i,:] = 0.5 * (gradp1 - gradm1) / dx

        q[i] += dx #restore partial coordinate

    return hess


def XTB_Force(q, atoms, qcinput):

    path         =  qcinput['path']
    method       =  qcinput['functional']
    charge       =  qcinput['charge']
    multiplicity =  qcinput['multiplicity']
    additional   =  qcinput['additional']
    natom        =  len(atoms)
    arg          = "--grad"

    ene, grad = call_XTB(q, atoms, path, charge, multiplicity, method, arg, additional, natom) 

    force = -np.reshape(grad,3*natom)

    return force
 

import shutil
def XTB_Energy(file_wf, q, atoms, qcinput):

    path         =  qcinput['path']
    method       =  qcinput['functional']
    charge       =  qcinput['charge']
    multiplicity =  qcinput['multiplicity']
    additional   =  qcinput['additional']
    natom        =  len(atoms)
    wfu          =  qcinput['wfu'] #wfu for wavefunction or dipole (molden format) calculation

    arg = "--dipole"
                      
    if wfu == True:
       arg = "--molden"

    ene, grad = call_XTB(q, atoms, path, charge, multiplicity, method, arg, additional,  natom)


    if wfu == True:
       shutil.copyfile("molden.input", file_wf)

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

    c1 = 0.52917721092  # [bohr]    * c1 = [Ansgtrom] 

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

    q = np.array(q) / c1

    ene = XTB_Energy(file_wf, q, atoms, qcinput)
    grad = XTB_Force(q, atoms, qcinput)

    print('Ene = ', ene)
    print("Gradient:")
    for i in range(natom):
    	print(grad[3*i], grad[3*i+1], grad[3*i+2] )
