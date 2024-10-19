import numpy as np
from collections import Counter

'''
The algorithm starts by iterating through each point in the dataset. For each point, it first checks if the point has already been visited. If not, it retrieves all neighbors of the point within the eps distance (using regionQuery). If the number of neighbors is less than minPts, the point is marked as noise; otherwise, it proceeds to expand the cluster from that point (expandCluster). The expandCluster function adds the point to the current cluster and checks all its neighbors. If any neighbor hasn't been visited, it's marked as visited, and its neighbors are retrieved. If this neighbor has enough neighbors itself, they are added to the list of points to be checked for this cluster. This process is recursive and continues until all points in or around the cluster are visited and assigned to a cluster if applicable.


dbscan: The main function that iterates through each point, classifying them into clusters or marking them as noise based on the density criteria.

expandCluster: A helper function that recursively expands a new cluster by adding all densely reachable points to the cluster.

regionQuery: A function to find all points within eps distance of a given point, effectively returning the neighbors of that point.

The example usage section shows how to create a sample dataset (D) and run the DBSCAN algorithm with specified eps and minPts values. The labels array will contain the cluster assignment for each point, with -1 indicating unclassified points and -2 indicating noise points.
'''

# DBSCAN algorithm implementation
def dbscan(D, eps, minPts):
    '''
    D: The dataset containing all data points in XYZ matrix format.
    eps: The radius of the neighborhood around each data point; points within this radius are considered neighbors.
    minPts: The minimum number of points required to form a dense region, which is considered a cluster.
    C: Cluster identifier.
    '''
    labels = np.zeros(len(D), dtype=int) - 1  # Initialize labels to -1 (unclassified)
    C = 0  # Cluster counter

    for P in range(len(D)):
        if labels[P] != -1:  # Skip if already classified
            continue

        NeighborPts = regionQuery(D, P, eps)  # Find neighbors
        if len(NeighborPts) < minPts:  # Mark as noise if not enough neighbors
            labels[P] = -2  # -2 for noise
        else:
            C += 1  # Next cluster label
            expandCluster(D, labels, P, NeighborPts, C, eps, minPts)

    return labels

def expandCluster(D, labels, P, NeighborPts, C, eps, minPts):
    labels[P] = C  # Assign to cluster

    i = 0
    while i < len(NeighborPts):  # Iterate through neighbors
        Pn = NeighborPts[i]
        if labels[Pn] == -2:  # Previously marked as noise
            labels[Pn] = C  # Assign to current cluster
        elif labels[Pn] == -1:  # Not yet classified
            labels[Pn] = C  # Assign to current cluster
            PnNeighborPts = regionQuery(D, Pn, eps)
            if len(PnNeighborPts) >= minPts:  # Add to neighbors if meets minPts
                NeighborPts = NeighborPts + PnNeighborPts
        i += 1

def regionQuery(D, P, eps):
    neighbors = []
    for Pn in range(len(D)):
        if np.linalg.norm(D[P] - D[Pn]) < eps:  # Euclidean distance check
            neighbors.append(Pn)
    return neighbors

def create_chemical_formula(vector):
    # Count the occurrences of each atomic element
    element_counts = Counter(vector)

    # Sort the elements alphabetically, but making sure C and H are always first if they are present
    sorted_elements = sorted(element_counts.keys(), key=lambda x: (x not in ['C', 'H'], x))

    # Format the counts into a chemical formula string without commas
    formula = ''.join([f"{element}{element_counts[element]}" if element_counts[element] > 1 else element for element in sorted_elements])

    return formula


def cluster_chemical_formulas(q, atoms, eps, minPts):
    xyz_mat = q_to_xyz_matrix(q)


    # Perform DBSCAN clustering
    labels = dbscan(xyz_mat, eps, minPts)

    # Find unique clusters, excluding noise
    clusters = np.unique(labels[labels >= 0])

    # Initialize a list to store formulas for each cluster
    cluster_formulas = []

    # Iterate over each cluster to get chemical formula
    for cluster in clusters:
        indices = np.where(labels == cluster)[0]
        cluster_atoms = [atoms[i] for i in indices]
        formula = create_chemical_formula(cluster_atoms)
        cluster_formulas.append(formula)

    # Concatenate all cluster formulas into a single string
    result_formula = ' + '.join(cluster_formulas)

    return result_formula

def q_to_xyz_matrix(vector):
    """
    Transforms a flat vector containing XYZ coordinates of molecules into an XYZ format matrix.

    Parameters:
    vector (numpy array): A flat array containing XYZ coordinates in the format [q1_x, q1_y, q1_z, ..., qN_x, qN_y, qN_z].

    Returns:
    numpy array: A 2D array (matrix) where each row represents a point (atom) in 3D space and each column represents one of the dimensions X, Y, Z.
    """
    # Ensure the vector length is divisible by 3 (for XYZ coordinates)
    assert len(vector) % 3 == 0, "The length of the vector must be divisible by 3."

    # Reshape the vector into a 2D array (matrix) where each row has 3 columns (X, Y, Z)
    xyz_matrix = np.reshape(vector, (-1, 3))

    return xyz_matrix



if __name__ == '__main__':
   from format_and_print import parseXYZ
   from format_and_print import qvec_to_xyz_matrix
   b2a = 0.52917721092

   xyz1 = '''
	C        -1.75201          1.17793        -2.37478 
	C        -1.32154          0.71041        -0.93632 
	C         0.12065          0.62961        -0.90775 
	C         1.20308          0.03998         0.19387 
	O         0.85216          0.55836         1.45331 
	O        -0.40304          0.15385         1.85210 
	C        -2.00293          0.56549         0.19921 
	O        -2.56261          1.32272         1.01919 
	H         0.52619          1.38462        -1.57431 
	H        10.01381         -2.18460        -0.66490 
	H        -2.75680          0.84532        -2.58282 
	H        -1.76159          2.23373        -2.54474 
	H        -1.19046          0.74773        -3.11644 
	H         2.12334          0.27952        -0.31574 
	H         1.22424         -1.06231         0.09006 
	H         9.52233         -1.23171         1.42251 
	H        -0.54978          0.96989         2.14969 
	O         9.02115         -1.72947         0.67795 
	O         9.49421         -2.77697        -0.22554 
   '''

   xyz2 = '''
C         2.61217         -0.70192        -0.27492 
C         1.28313          0.21492        -0.32291 
C         0.86569          0.51923        -1.57718 
C        -0.45253          1.32303        -1.86494 
O        -0.91994          1.28260        -3.16535 
O         0.15600          1.90481        -3.89114 
C         0.98145          0.94067         0.81721 
O         0.47059          2.03890         0.96178 
H         1.38644          0.21014        -2.43607 
H         1.52980          0.84940         1.84754 
H         2.51847         -1.10134         0.62246 
H         2.54758         -1.41764        -1.16914 
H         3.50089         -0.20010        -0.56153 
H        -0.24038          2.39674        -1.52351 
H        -1.32488          0.91328        -1.21912 
H         4.13924         -2.61844         5.81193 
H        -0.29332          2.53059        -4.60120 
O         4.46953         -3.09697         6.69143 
O         3.11843         -3.86618         7.12156 
   '''

   xyz3 = '''
C         0.19163         -1.00208        -1.83975
C         0.37583          0.33008        -2.45669
C         1.10940          0.70495        -3.64599
C         2.01282          0.28796        -4.52353
O         2.56820          0.21758        -5.52629
O        -5.54662         -0.75876        15.89569
C        -0.41426          1.24383        -1.87020
O        -0.34751          2.55641        -1.46844
H         1.05920          1.78918        -3.84525
H        -0.99926          0.99431        -1.02470
H        -0.89352         -1.21443        -1.86096
H         0.65413         -1.96790        -2.06285
H         0.33400         -0.96016        -0.73615
H         5.55020         -1.13264         4.03776
H         8.40448         -1.49609         3.15923
H         0.07546          3.15250        -2.25362
H        -4.67822         -1.07190        15.22827
O         5.94753         -1.47975         3.36258
O         7.23978         -1.84586         3.82761
   '''


   xyz = xyz3

   Natoms, atoms, q = parseXYZ(xyz) 
   xyz_mat = qvec_to_xyz_matrix(q)

   eps = 2.1 #in Angstrom
   minPts = 2 #minum number of points to be a cluster
   labels = dbscan(xyz_mat, eps, minPts)

   
   clusters = np.unique(labels[labels >= 0])  # Exclude noise points
   print(xyz)
  #Print clusters and their atom indices
   for cluster in clusters:
       indices = np.where(labels == cluster)[0]
       print()
       print(f"Cluster {cluster}: Atom Indices {indices}")
       #for i in indices:
       #    print(atoms[i])


   print("Calculate the formula")
   formula = cluster_chemical_formulas(q, atoms, eps, minPts)
   print()
   print("everthing together: ", formula)


   


