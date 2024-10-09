import sys
import os
import subprocess
import numpy as np

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
    b2a = 0.52917721092
    natoms = len(atoms)

    xyz = ""

    for i in range(natoms):
        xyz += atoms[i] + " "*6+str(q[i*3]*b2a) \
            + " "*6+str(q[i*3+1]*b2a) \
            + " "*6+str(q[i*3+2]*b2a) \
            + "\n"
    return xyz


def generateInput(inputfile, nproc, xyz, method, basis, charge, multiplicity, additional):
    with open(inputfile + ".inp", 'w') as f:
        f.write("%pal nprocs " + str(nproc) + " end \n")
        f.write("! " + method + " " + basis + " engrad  " + additional + "\n")
        f.write("* xyz " + str(charge) + " " + str(multiplicity) + "\n")
        f.write(xyz)
        f.write("*" + "\n")

def generate_Optimizer_Input(what, inputfile, nproc, xyz, method, basis, charge, multiplicity, additional):
    with open(inputfile + ".inp", 'w') as f:
        f.write("%pal nprocs " + str(nproc) + " end \n")
        f.write("! " + method + " " + basis + " opt  " + "freq" + additional + "\n")
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
    if not orcapath:
            print("Cannot determine ORCAPATH")
            exit()
    command = [orcapath, inputfile + ".inp"]
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)

    if result.returncode == 0:
        with open(inputfile + ".out", 'w') as f:
            f.write(result.stdout)
    else:
        print("ORCA command failed. Exit code:", result.returncode)

def eatLines(f,num):
    for i in range(num):
        f.readline()

def read_Energy_and_Grad(inputfile):
    with open(inputfile + ".engrad", "r") as f:
        eatLines(f,3)
        Natoms = int(f.readline())
        eatLines(f,3)
        Energy = float(f.readline())
        eatLines(f,3)
        grad = []
        for i in range(Natoms*3):
            grad.append(float(f.readline()))
        return (Energy, np.array(grad))

def read_Energy_and_Optimized_Geoms(inputfile):
    converged = False
    with open(inputfile + ".out", "r") as f:
        if in_text == 'THE OPTIMIZATION HAS CONVERGED':
            converged = True

   #even the unconverged geom will be read
    with open(inputfile + ".xyz", "r") as f:
        Natoms = int(f.readline())
        Energy = float(f.readline())
        optxyz = []
        for i in range(Natoms*3):
            optxyz.append(float(f.readline()))
        return (converged, Energy, optxyz)


def runOrca(Natoms, xyz, path, nproc, method, basis, charge, multiplicity, additional):
    fname = 'orca_file_for_abinitioMD'
    directory = 'orca_tmp'
    if not os.path.exists(directory):
        os.makedirs(directory)

    inputfile = os.path.join(directory, fname)

    generateInput(inputfile, nproc, xyz, method, basis, charge, multiplicity, additional)

    callORCA(inputfile, path)

    ene, grad = read_Energy_and_Grad(inputfile)
    return ene, grad 


def runOrcaOptimizer(what, Natoms, xyz, path, nproc, method, basis, charge, multiplicity, additional):
    fname = 'orca_file_for_geomopt'
    directory = 'orca_tmp'
    if not os.path.exists(directory):
        os.makedirs(directory)

    inputfile = os.path.join(directory, fname)

    generate_Optimizer_Input(what, inputfile, nproc, xyz, method, basis, charge, multiplicity, additional)

    callORCA(inputfile, path)

    converged, ene, optgeom = read_Energy_and_Optimized_Geoms(inputfile)
    return read_Energy_and_Optimized_Geoms(inputfile)


def readOrcaHessian(inputfile):
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


def getOrcaHessian(Natoms, xyz, path, nproc, method, basis, charge, multiplicity, additional):
    fname = 'orca_file_for_abinitioMD'
    directory = 'orca_tmp'

    if not os.path.exists(directory):
        os.makedirs(directory)

    inputfile = os.path.join(directory, fname)

   #Create Orca input
    with open(inputfile + ".inp", 'w') as f:
        f.write("%pal nprocs " + str(nproc) + " end \n")
        f.write("! " + method + " " + basis + " " + "numfreq " + additional + "\n")
        f.write("* xyz " + str(charge) + " " + str(multiplicity))
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
    return getOrcaHessian(Natoms, xyz, path, nproc, method, basis, charge, multiplicity, additional)


def Orca_GeomOpt(Natoms, xyz, qcinput):
    path         =  qcinput['path']
    nproc        =  qcinput['nproc']
    method       =  qcinput['functional']
    basis        =  qcinput['basis']
    charge       =  qcinput['charge']
    multiplicity =  qcinput['multiplicity']
    additional   =  qcinput['additional']
    what         =  qcinput['what']
    return runOrcaOptimizer(what, Natoms, xyz, path, nproc, method, basis, charge, multiplicity, additional) 


def Orca_Force(q, atoms, qcinput):
    path         =  qcinput['path']
    nproc        =  qcinput['nproc']
    method       =  qcinput['functional']
    basis        =  qcinput['basis']
    charge       =  qcinput['charge']
    multiplicity =  qcinput['multiplicity']
    additional   =  qcinput['additional']

    xyz = makeXYZ(atoms, q)
    Natoms = len(atoms)

    ene, grad = runOrca(Natoms, xyz, path, nproc, method, basis, charge, multiplicity, additional)

    return -np.array(grad)

import shutil
import subprocess

def Orca_Energy(filename, q, atoms, qcinput):

    xyz = makeXYZ(atoms, q)
    Natoms = len(atoms)

    fname = 'orca_file_for_abinitioMD'
    directory = 'orca_tmp'

    inputfile = os.path.join(directory, fname)

    if os.path.exists(inputfile):
        ene, grad = read_Energy_and_Grad(inputfile)
        return ene

    path         =  qcinput['path']
    nproc        =  qcinput['nproc']
    method       =  qcinput['functional']
    basis        =  qcinput['basis']
    charge       =  qcinput['charge']
    multiplicity =  qcinput['multiplicity']
    additional   =  qcinput['additional']
    wfu          =  qcinput['wfu']

    ene, grad = runOrca(Natoms, xyz, path, nproc, method, basis, charge, multiplicity, additional)
   
    if wfu == True:
       gbwfile = "orca_tmp/" + fname
       command_create_molden = path + "_2mkl " + gbwfile +  " -molden"
  
       result = subprocess.run(command_create_molden, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

       if result.returncode != 0:
          print("Converting orca.gbw file into molden file format failed:")
          print(result.stderr)

       moldenfile = "orca_tmp/" + fname + ".molden.input"
       shutil.copy(moldenfile, filename)

    return ene


if __name__ == "__main__":


    c1 = 0.52917721092  # [bohr]    * c1 = [Ansgtrom] 
    c3 = 1838.6836605e0 # [g/mol]   * c3 = [electron mass unit]
    c6 = 41.341105      # [fs]      * c6 = [time in au]
    c7 = 2625.5         # [Hartree] * c7 = [kJ/mol] 
    Rgas = 8.3144598/1000.0/c7 #Hartree/K 

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

    q = np.array(q_eq) / c1

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

