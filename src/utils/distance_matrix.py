import numpy as np

def compute_distance_matrix(q):
    N = len(q) // 3 
    coordinates = np.reshape(q, (N, 3))
    dist_matrix = np.sqrt(np.sum((coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]) ** 2, axis=-1))
    return dist_matrix


def distance_matrix_to_adjacency_matrix(distance_matrix, atoms, hydrogen_max_dist, non_h_max_dist):
    N = len(atoms)
    adjacency_matrix = np.zeros((N, N), dtype=int) 

    for i in range(N):
        for j in range(i + 1, N):
            if 'H' in [atoms[i], atoms[j]]:  # If either atom is hydrogen
                max_distance = hydrogen_max_dist
            else:  # Both atoms are non-hydrogen
                max_distance = non_h_max_dist

            # Set to 1 if distance is within the max bond distance, otherwise 0
            if distance_matrix[i, j] <= max_distance:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1  # Symmetric matrix

    return adjacency_matrix


def compute_connectivity(q, atoms, H_max_dist, nonH_max_dist):
    dist_matrix = compute_distance_matrix(q)
    return distance_matrix_to_adjacency_matrix(dist_matrix, atoms, H_max_dist, nonH_max_dist)


def add_matrix_and_get_index(new_matrix, new_q, matrices_list, q_list):
    """
    Adds the new_matrix to matrices_list if it is not already present,
    as well as a new vector to the vector list
    and returns the index of the new_matrix (whether it was just added or already existed).
    and the list of vectors (as a matrix)

    Parameters:
    - new_matrix: A numpy array representing the new matrix to add.
    - new_q: the new coordinate vector
    - matrices_list: A list of numpy arrays representing the matrices.
    - q_list: list of coordiante vectors to be saved
    """
    for index, matrix in enumerate(matrices_list):
        if np.array_equal(new_matrix, matrix):
            # Return the current index if a matching matrix is found
            return index, matrices_list, q_list

    # If no match was found, append the new matrix and return its new index
    matrices_list.append(new_matrix)
    q_list.append(new_q)

    return len(matrices_list) - 1, matrices_list, q_list 







if __name__ == '__main__':
   from clustering import cluster_chemical_formulas
   from format_and_print import parseXYZ
   b2a = 0.52917721092

   xyz0 = '''
    O        -2.56261          1.12272         1.01919 
    H        -2.56261          0.00000         1.01919 
    H         9.52233         -1.23171         1.42251 
    O         9.02115         -1.72947         0.67795 
    O         9.49421         -2.77697        -0.22554 
    '''


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



   Natoms_0, atoms_0, q_0 = parseXYZ(xyz0) 
   Natoms_1, atoms_1, q_1 = parseXYZ(xyz1) 
   Natoms_2, atoms_2, q_2 = parseXYZ(xyz2) 
   Natoms_3, atoms_3, q_3 = parseXYZ(xyz3) 

   eps = 2.1 #in Angstrom
   minPts = 1 #minum number of points to be a cluster

   formula = cluster_chemical_formulas(q_0, atoms_0, eps, minPts)
   print()
   print("Formula 0: ", cluster_chemical_formulas(q_0, atoms_0, eps, minPts))
   print("Formula 1: ", cluster_chemical_formulas(q_1, atoms_1, eps, minPts))
   print("Formula 2: ", cluster_chemical_formulas(q_2, atoms_2, eps, minPts))
   print("Formula 3: ", cluster_chemical_formulas(q_3, atoms_3, eps, minPts))
   print()

###################################################################################################
   
   H_max_dist = 1.5/0.51  # Maximum bond distance for H to any atom (adjust as needed)
   nonH_max_dist = 2.0/0.51    # Maximum bond distance for non-H atom to another non-H atom (adjust as needed)

   connect_0 = compute_connectivity(q_0, atoms_0, H_max_dist, nonH_max_dist)
   connect_1 = compute_connectivity(q_1, atoms_1, H_max_dist, nonH_max_dist)
   connect_2 = compute_connectivity(q_2, atoms_2, H_max_dist, nonH_max_dist)
   connect_3 = compute_connectivity(q_3, atoms_2, H_max_dist, nonH_max_dist)

   # here start the reaction channel 
   prod_con_list = []
   prod_q_list = [] # np.column_stack((vector1, vector2))


   channel, product_connect_list, product_coord_list = add_matrix_and_get_index(connect_0, q_0, prod_con_list, prod_q_list)
   print("product channel: ", channel, "   length: ", len(prod_q_list), "  formula: ", cluster_chemical_formulas(q_0, atoms_0, eps, minPts))

   channel, product_connect_list, product_coord_list = add_matrix_and_get_index(connect_1, q_1, prod_con_list, prod_q_list)
   print("product channel: ", channel, "   length: ", len(prod_q_list), "  formula: ", cluster_chemical_formulas(q_1, atoms_1, eps, minPts))

   channel, product_connect_list, product_coord_list = add_matrix_and_get_index(connect_2, q_2, prod_con_list, prod_q_list)
   print("product channel: ", channel, "   length: ", len(prod_q_list), "  formula: ", cluster_chemical_formulas(q_2, atoms_2, eps, minPts))

   channel, product_connect_list, product_coord_list = add_matrix_and_get_index(connect_3, q_3, prod_con_list, prod_q_list)
   print("product channel: ", channel, "   length: ", len(prod_q_list), "  formula: ", cluster_chemical_formulas(q_3, atoms_3, eps, minPts))

#  try adding a previous element of the list:
   channel, product_connect_list, product_coord_list = add_matrix_and_get_index(connect_0, q_0, prod_con_list, prod_q_list)
   print("product channel: ", channel, "   length: ", len(prod_q_list), "  formula: ", cluster_chemical_formulas(q_0, atoms_0, eps, minPts))




