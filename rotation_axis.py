import numpy as np
from graph_cluster import create_chemical_formula 
from graph_cluster import find_fragments_bfs

def rodrigues_rotation(axis,theta):
    '''
    See the formula:
    https://mathworld.wolfram.com/RodriguesRotationFormula.html

    Rodrigues' rotation formula gives an efficient method for computing
    the rotation matrix R in SO(3) corresponding to a rotation
    by an angle theta about a fixed axis specified by
    the unit vector defined by the axis: omega=(omega_x, omega_y, omega_z) in R^3.
    '''

    axis = axis / np.linalg.norm(axis)

    R = np.zeros((3, 3), dtype=float)

    ct = np.cos(theta)
    st = np.sin(theta)

    omeX = axis[0]
    omeY = axis[1]
    omeZ = axis[2]

    R[0,0] = ct + omeX*omeX*(1.0 - ct)
    R[0,1] = omeX*omeY*(1.0 - ct) - omeZ*st
    R[0,2] = omeX*omeZ*(1.0 - ct) + omeY*st
    
    R[1,0] = omeX*omeY*(1.0 - ct) + omeZ*st
    R[1,1] = ct + omeY*omeY*(1.0 - ct)
    R[1,2] = omeY*omeZ*(1.0 - ct) - omeX*st


    R[2,0] = omeX*omeZ*(1.0 - ct) - omeY*st
    R[2,1] = omeY*omeZ*(1.0 - ct) + omeX*st
    R[2,2] = ct + omeZ*omeZ*(1.0 - ct)

    return R

def parseXYZ(xyz):
    lines = xyz.strip().split('\n')
    Natoms = sum(1 for line in lines if line.strip())

    q = []
    atoms = []
    for line in lines:
        parts = line.split()
        a, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
        atoms.append(a)
        q.extend([x, y, z])

    q = np.array(q)
    return Natoms, atoms, q

def print_trajectory(trajfile, atoms, q, what, angle):
    trajfile.write(str(len(atoms)) + "\n")
    trajfile.write("%7s %20s %10.2f \n" % ("which= ", what, angle))
    for i in range(0, len(atoms)):
        jx = 3 * i
        jy = 3 * i + 1
        jz = 3 * i + 2
        trajfile.write("%3s %15.5f  %15.5f %15.5f \n" % (atoms[i], q[jx], q[jy], q[jz]))

#--------------------------------------------------------------------------------------------------
def locate_frags_to_rotate(atoms, q, atom_ax1, atom_ax2):
#--------------------------------------------------------------------------------------------------

    #############kwarg** here to control these variables*******************************************
    bond_th_HX = 1.5 #bond threshold in Angstrom for H-X, where X = any non H-atom
    bond_th_XX = 2.0 #bond threshold in Angstrom for X-X bonds to be considered as a part of a fragment

    side1_index, side2_index = find_fragments_bfs(atoms, q, atom_ax1, atom_ax2, bond_th_HX, bond_th_XX)

    side1_atoms = [atoms[i] for i in side1_index]
    side2_atoms = [atoms[i] for i in side2_index]

    side1_formula = create_chemical_formula(side1_atoms)
    side2_formula = create_chemical_formula(side2_atoms)

   #here we chose the smaller fragment on the two side of the axis to rotate
    if len(side1_index) <= len(side2_index):
        ind_rot_center = atom_ax1
        rot_frag_index = [i for i in side1_index if i != ind_rot_center]
        spectator_frag_index = [i for i in side2_index if i != ind_rot_center]
    elif len(side1_index) > len(side2_index):
        ind_rot_center = atom_ax2
        rot_frag_index = [i for i in side2_index if i != ind_rot_center]
        spectator_frag_index = [i for i in side1_index if i != ind_rot_center]

    #############kwarg** here to control verbosity*******************************************
    print(f"Atoms on side atom {atom_ax1} of the axis: {side1_index}")
    print("Chemical formula for side 1:", side1_formula)
    print()
    print(f"Atoms on side atom {atom_ax2} of the axis: {side2_index}")
    print("Chemical formula for side 2:", side2_formula)

    print("\nRotation center index:")
    print(ind_rot_center)
    print("\nFragment indicies to rotate (center of axis is excluded from the frag):")
    print(rot_frag_index)
    print("\nSpectator fragment indicies (center of axis is excluded from the frag):")
    print(spectator_frag_index)
    
    return (ind_rot_center, rot_frag_index, spectator_frag_index)
#--------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------
def rotate_fragment(atoms, q, atom_ax1, atom_ax2, auto_frag, rot_angle, atom_rotcenter, frag_index):
#-------------------------------------------------------------------------------------------------------
    '''
    INPUT:
    - rot_angle: angle of rotation given in degree (not radian)
    - q: coordinates of molecule as as single vector q = [x1, y1, z1, x2, y2, z2,..., xN, yN, zN]
    - atoms: atomic symboles of molecule as as single array of strings atoms = ['H','H','O'] for H2O
    - atom_ax1, atom_ax2: indices of atoms to define an axis
    - auto_frag: False or True to switch on the automatic determination of fragment to rotate
    - atom_rotcenter: index of atom to be the center of rotation, it's needed only when auto_frag = False
    - frag_index: indices of atoms in the fragment to rotate, it's needed only when auto_frag = False
 
    OUTPUT:
    - the q-vector of molecule where the fragment has been rotated
    '''

   #automatic location of fragment to be rotated or given by hand:
    if auto_frag == True:
        ind_rot_center, rot_frag_index, spectator_frag_index = locate_frags_to_rotate(atoms, q, atom_ax1, atom_ax2)
    else:
        ########################################here use kwarg** to get the frag_index and atom_rotcenter
        rot_frag_index = frag_index
        ind_rot_center = atom_rotcenter
        if ind_rot_center != atom_ax1 and ind_rot_center != atom_ax2:
            #to avoid errors stemming from human source
            raise ValueError("rot_center must be the same as one of the atoms that defines the rotation axis\n")

    angle = rot_angle * np.pi / 180.0 # degree to radian

    mol_coords = q.reshape(-1, 3)

    axis_start = mol_coords[atom_ax1]
    axis_end = mol_coords[atom_ax2]

    rot_center = mol_coords[ind_rot_center]

   #define axis of rotation
    axis = axis_end - axis_start

   #shift the molecule to the end-point of the axis which is the center of rotation
    shifted_coords = mol_coords - rot_center

   #get the rotation matrix
    rot_mat = rodrigues_rotation(axis, angle)

   #rotate about the end-point of the axis
    rotated_coords = shifted_coords.copy() #the unrotated components preserved
    for i in rot_frag_index:
        rotated_coords[i] = rot_mat @ rotated_coords[i]

   #shift back the rotated molecule to its original center matching the molecule before rotation
    new_mol_coords = rotated_coords + rot_center

    return new_mol_coords.flatten() 
#--------------------------------------------------------------------------------------------------



if __name__ == "__main__":
    xyz1 = '''
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

    xyz2 = '''
    C        -1.75201          1.17793        -2.37478
    C        -1.32154          0.71041        -0.93632
    C         0.12065          0.62961        -0.90775
    C         1.20308          0.03998         0.19387
    O         0.85216          0.55836         1.45331
    O        -0.40304          0.15385         1.85210
    C        -2.00293          0.56549         0.19921
    O        -2.56261          1.32272         1.01919
    H         0.52619          1.38462        -1.57431
    H        -2.75680          0.84532        -2.58282
    H        -1.76159          2.23373        -2.54474
    H        -1.19046          0.74773        -3.11644
    H         2.12334          0.27952        -0.31574
    H         1.22424         -1.06231         0.09006
   '''
    Natoms, atoms, q = parseXYZ(xyz2)

    formula = create_chemical_formula(atoms)
    print("original Molecule: ", formula)
    print()

    # Define the axis using atoms with index 1 and 2 (0-based index)
    atom_ax1 = 2
    atom_ax2 = 3
    print("\nfrag indicies:")
    print(atom_ax1, atom_ax2)


    a0 = 0.0
    dang = 10.0
    auto_frag = True
    rotcenter = 0 
    frag_index = [0, 6, 8, 9]

    # Open the file and print both sets of coordinates
    with open('trajectory.txt', 'w') as trajfile:

        for i in range(36):
            rot_angle = a0 + i*dang

            q_rotated = rotate_fragment(atoms, q, atom_ax1, atom_ax2, auto_frag, rot_angle, rotcenter, frag_index)

            print_trajectory(trajfile, atoms, q_rotated, "After rotation: ", rot_angle)

    print("\nRotation is done.\n")

    #rot_angle = 90.0
    #rot_center, rot_frag_index, spectator_frag_index = locate_frags_to_rotate(atoms, q, atom_ax1, atom_ax2)


    #print("\nRotation center index:")
    #print(rot_center)
    #print("\nFragment indicies to rotate (center of axis is excluded from the frag):")
    #print(rot_frag_index)
    #print("\nSpectator fragment indicies (center of axis is excluded from the frag):")
    #print(spectator_frag_index)

