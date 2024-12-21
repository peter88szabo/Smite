import numpy as np
import random

def surface_normal(ind, q):
    """
    Calculate the normalized surface normal vector given indices of three atoms and their coordinates.
    """

    if len(ind) != 3:
        raise ValueError("Exactly three atomic indices must be given.")
    
    #indices of three atoms needed to define the surface
    i1 = ind[0]
    i2 = ind[1]
    i3 = ind[2]

    p1 = q[3*i1:3*i1+3]
    p2 = q[3*i2:3*i2+3]
    p3 = q[3*i3:3*i3+3]

    # Define the plane by two atomic distance vectors
    v1 = p2 - p1
    v2 = p3 - p1

    normal = np.cross(v1, v2)

    normal_magnitude = np.linalg.norm(normal)
    if normal_magnitude == 0:
        raise ValueError("The three points are collinear and do not define a plane.")
    
    normal_vec = normal / normal_magnitude

    return normal_vec


def rotmat_for_best_overlap(qA, qB):
    #from utils.cemass import cenmassQ
    '''
    Find the optimal overlap between two 3*N dimensional vector
    using Least-Square fitting that results in SVD transformation

    the code assumes that you already shifted the two object into
    their center of mass, then we can calculate the rotation matrix
    '''
    if len(qA) != len(qB):
        raise Exception("Error: qA and qB vector must have the same size in Rigid Rotation transformation")

    #we shif both molecule into their center of mass (now both in the origin)
    #qA = cenmassQ(qA, mass)
    #qB = cenmassQ(qB, mass)

    # Convert q to a 3xN matrix
    coord_A = np.array(qA).reshape(3, -1)
    coord_B = np.array(qB).reshape(3, -1)

    print()
    print("coord_A: ", coord_A)
    print("coord_B: ", coord_B)
    print()

    num_rows, num_cols = coord_A.shape
    if num_rows != 3:
        raise Exception(f"matrix coord_A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = coord_B.shape
    if num_rows != 3:
        raise Exception(f"matrix coord_B is not 3xN, it is {num_rows}x{num_cols}")

    H = coord_A @ np.transpose(coord_B)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    Rmat = Vt.T @ U.T

    # special reflection case (for inversion case)
    if np.linalg.det(Rmat) < 0:
        #multiply 3rd column of V by -1
        Vt[2,:] *= -1
        Rmat = Vt.T @ U.T

    return Rmat 

def normalize(v):
    return v / np.linalg.norm(v)

def rodrigues_rotation_matrix(normal, theta):
    N = normalize(normal)
    N_x, N_y, N_z = N

    K = np.array([
        [0, -N_z, N_y],
        [N_z, 0, -N_x],
        [-N_y, N_x, 0]
    ])

    K2 = K @ K

    # Rodrigues' rotation formula
    Rmat = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * K2

    return Rmat

def rotate_points(points, normal, theta):
    R = rodrigues_rotation_matrix(normal, theta)
    rotated_points = np.dot(points, R.T)  # Apply rotation matrix
    return rotated_points


if __name__ == "__main__":
    from utils.atomic_masses import get_mass_vector

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

    def print_trajectory(trajfile, atoms, q, angle):
        trajfile.write(str(len(atoms)) + "\n")
        trajfile.write("%8s %10.2f \n" % ("angle = ", angle))
        au2Ang=0.5291772e0
        for i in range(0, len(atoms)):
            jx = 3 * i
            jy = 3 * i + 1
            jz = 3 * i + 2
            trajfile.write("%3s %15.5f  %15.5f %15.5f \n" % (atoms[i], q[jx]*au2Ang, q[jy]*au2Ang, q[jz]*au2Ang))


    xyz_C48H18 = '''
C           -2.14340523475483       -3.60229542113849       0.00000000
C           -2.84395339368902       -4.83563379041050       0.00000000
C           -4.25568748932248       -4.85015091978321       0.00000000
C           -4.94520354993805       -3.62410375302998       0.00000000
C           -4.27070654915091       -2.43962961480767       0.00000000
C           -2.86037015595778       -2.38728327778804       0.00000000
C           -2.14267048154587       -1.14424281525927       0.00000000
C           -0.73191129865843       -1.15767830134162       0.00000000
C           -0.01343726076282       -2.38797775017350       0.00000000
C           -0.71869380550000       -3.60942579736532       0.00000000
C           -2.85332001784624       -7.27931256631650       0.00000000
C           -4.26842253206993       -7.26085817434060       0.00000000
C           -4.94737090900842       -6.08494757878369       0.00000000
C           -2.13481991357100       -6.06398061418652       0.00000000
C           -0.77325153963271       -8.49752295128966       0.00000000
C           -2.13631214080366       -8.48945552399652       0.00000000
C           -0.71647706944973       -6.07393892752749       0.00000000
C           -0.02270410372863       -7.30239944754601       0.00000000
C            1.41262870108568       -7.30240501245001       0.00000000
C           -0.01027993639953       -4.83654034524137       0.00000000
C            1.40015134023799       -4.83655035494251       0.00000000
C            2.10637523624709       -6.07393985385079       0.00000000
C            1.40347243656834       -2.38800563492516       0.00000000
C            2.10863210617904       -3.60948541643976       0.00000000
C            3.52621284536659       -8.48950332483980       0.00000000
C            4.24317901714580       -7.27936970061923       0.00000000
C            3.52470904913129       -6.06397708849549       0.00000000
C            2.16313729546929       -8.49754780595329       0.00000000
C            5.65826669079645       -7.26089424371342       0.00000000
C            6.33718928533975       -6.08497067087023       0.00000000
C            5.64566331016939       -4.85022583176435       0.00000000
C            4.23387006368009       -4.83569274398682       0.00000000
C            6.33519991852010       -3.62425251552938       0.00000000
C            5.66065358318471       -2.43976622152291       0.00000000
C            4.25037463944032       -2.38736026328030       0.00000000
C            3.53335347740759       -3.60239296835776       0.00000000
C            3.53274931278632       -1.14430729616017       0.00000000
C            2.12199273175132       -1.15772023842775       0.00000000
C           -2.80239043087603        0.10332603136605       0.00000000
C           -2.11386681399084        1.27976160523444       0.00000000
C           -0.70737757738930        1.29556022954751       0.00000000
C           -0.01408726053307        0.06564034331031       0.00000000
C            1.40421149247463        0.06561465272416       0.00000000
C            0.01617368097651        2.51180894504361       0.00000000
C            1.37402042003713        2.51178459669040       0.00000000
C            2.09755002315958        1.29552155412323       0.00000000
C            3.50404424555929        1.27965684688439       0.00000000
C            4.19253559192592        0.10322894674367       0.00000000
H           -6.02593586348187       -3.63585191151221       0.00000000
H           -4.84262827718479       -1.52621519179806       0.00000000
H           -4.79465281139334       -8.20483727073600       0.00000000
H           -6.02789714966675       -6.06830729970999       0.00000000
H           -0.26826581757656       -9.44959008582719       0.00000000
H           -2.68663792604773       -9.41966923711393       0.00000000
H            4.07652998335570       -9.41972064522941       0.00000000
H            1.65821072137918       -9.44964153094208       0.00000000
H            6.18448808710615       -8.20487174454043       0.00000000
H            7.41780798721020       -6.06871481118622       0.00000000
H            7.41594794842401       -3.63576064192065       0.00000000
H            6.23272801335070       -1.52643455644358       0.00000000
H           -3.87936330827548        0.14220175995579       0.00000000
H           -2.64428321657912        2.22146160836636       0.00000000
H           -0.53818614652534        3.43953838346956       0.00000000
H            1.92843455259711        3.43948256904088       0.00000000
H            4.03448271280181        2.22134651050573       0.00000000
H            5.26954348044517        0.14192510040879       0.00000000
    '''

    seed = 211422
    random.seed(seed)

    c3   = 1838.6836605e0        # [g/mol]    * c3 = [electron mass unit]

    mH = 1.0078*c3
    mC = 12.011*c3
    mN = 14.007*c3
    mO = 15.999*c3

    Natoms, atoms, q_eq = parseXYZ(xyz_C48H18)

    q = q_eq / 0.52917721092

    mass = get_mass_vector(atoms)

    angle = 90.0

    trajfile = open('testfile.xyz', 'w')  
        
    print_trajectory(trajfile, atoms, q, angle)



    ind = [9, 20, 22]
    normalvec = surface_normal(ind, q)

    print("normalvec: ", normalvec)
    print()

    basis = np.array([1.0, 0.0, 0.0]) 

    print("basis: ", basis)


    R = rotmat_for_best_overlap(normalvec, basis)

    print("\nRotmat:")
    print(R)
    print()

    qrot = np.zeros_like(q)  
    for i in range(len(q) // 3):
        qatom = np.array([q[3 * i], q[3 * i + 1], q[3 * i + 2]])
        qatom_rot = R @ qatom  # Rotate the atom's coordinates
        qrot[3 * i:3 * i + 3] = qatom_rot  

    normalvec = surface_normal(ind, qrot)
    print("Rot normalvec: ", normalvec)


    angle = 90.0
    print_trajectory(trajfile, atoms, qrot, angle)

    trajfile.close()

    # Example Usage
    points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Example points
    normal = np.array([0, 0, 1])  # Normal vector
    theta = np.pi / 4  # 45 degrees in radians

    rotated = rotate_points(points, normal, theta)
    print(rotated)




