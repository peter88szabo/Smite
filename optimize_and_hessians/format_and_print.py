import numpy as np
import os

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

def check_and_create_file(file_name,ending):
    fname = file_name + ending
    # Check if the initial filename exists
    while os.path.exists(fname):
        # If the filename contains a number, extract it and increment
        if fname.endswith(ending):
            base_name = fname[:-(len(ending))]  # Remove the ".xyz" extension
            parts = base_name.split("_")
            if len(parts) == 2 and parts[1].isdigit():
                i = int(parts[1]) + 1
                fname = f"{parts[0]}_{i}{ending}"
            else:
                fname = f"{parts[0]}_1{ending}"
        else:
            # If there is no extension, add "_1.xyz"
            fname = f"{fname}_1{ending}"
    return fname


def print_trajectory(trajfile, atoms, q, p, V, dE, dt, istep):
    b2a = 0.52917721092
    c5=219474.0  #[Hartree]*c5=[cm-1]
    c6=41.341105 #[femto-sec]*c6=[time in au]

    trajfile.write(str(len(atoms)) + "\n")
    trajfile.write("%7s %10d %4s %15.4f %15s %15.8f %10s %12.5f \n" % ("step= ",istep, "    t[fs]= ", dt*istep/c6, "     Vpot[au]= ", V, "      dE[cm-1]= ",dE*c5))
    for i in range(0,len(atoms)): 
        jx = 3*i
        jy = 3*i+1
        jz = 3*i+2
        trajfile.write("%3s %15.5f  %15.5f %15.5f %20.8f  %20.8f %20.8f \n" % (atoms[i],q[jx]*b2a, q[jy]*b2a, q[jz]*b2a, p[jx], p[jy], p[jz]) )

