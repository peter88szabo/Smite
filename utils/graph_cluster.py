import numpy as np
from collections import deque, Counter

def create_adjacency_list(atoms, q, bond_th_HX, bond_th_XX):
    coords = q.reshape(-1, 3)
    adj_list = {i: [] for i in range(len(atoms))}
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):

            #----------------------------------------------------------------
            #Yehh, ugly nested loops...later will be made more aesthetic

            #We do not care about H-H atom pairs only H-X or X-Y pairs
            if atoms[i] != 'H' or atoms[j] != 'H':

                if atoms[i] != 'H' and atoms[j] != 'H':
                    bond_threshold = bond_th_XX 
                else:
                    bond_threshold = bond_th_HX 

                if np.linalg.norm(coords[i] - coords[j]) <= bond_threshold:
                    adj_list[i].append(j)
                    adj_list[j].append(i)
            #----------------------------------------------------------------

    return adj_list

def bfs_traversal(adj_list, start_node, visited, exclude_node):
    queue = deque([start_node])
    cluster = []

    while queue:
        node = queue.popleft()
        if not visited[node] and node != exclude_node:
            visited[node] = True
            cluster.append(node)
            for neighbor in adj_list[node]:
                if not visited[neighbor] and neighbor != exclude_node:
                    queue.append(neighbor)

    return cluster

def find_fragments_bfs(atoms, q, atom1_idx, atom2_idx, bond_th_HX, bond_th_XX):
    adj_list = create_adjacency_list(atoms, q, bond_th_HX, bond_th_XX)
    
    visited = [False] * len(atoms)
    
    cluster1 = bfs_traversal(adj_list, atom1_idx, visited, atom2_idx)
    cluster2 = bfs_traversal(adj_list, atom2_idx, visited, atom1_idx)
    
    return cluster1, cluster2

def create_chemical_formula(atoms):
    element_counts = Counter(atoms)
    sorted_elements = sorted(element_counts.keys(), key=lambda x: (x not in ['C', 'H'], x))
    formula = ''.join([f"{element}{element_counts[element]}" if element_counts[element] > 1 else element for element in sorted_elements])
    return formula

if __name__ == "__main__":

    def parseXYZ(xyz):
        lines = xyz.strip().split('\n')
        q = []
        atoms = []
        for line in lines:
            parts = line.split()
            a, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append(a)
            q.extend([x, y, z])
        q = np.array(q)
        return atoms, q




    xyz0 = '''
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
	H        -2.75680          0.84532        -2.58282
	H        -1.76159          2.23373        -2.54474
	H        -1.19046          0.74773        -3.11644
	H         2.12334          0.27952        -0.31574
	H         1.22424         -1.06231         0.09006
   '''

    
    atoms, q = parseXYZ(xyz0)
    q = q / 0.5291772
    formula = create_chemical_formula(atoms)
    print("original Molecule: ", formula)
    print()
    
    # Define the axis using atoms with indices 1 and 2 (0-based index)
    atom1_idx = 3
    atom2_idx = 4
    bond_th_HX = 1.5/0.5291772
    bond_th_XX = 2.0/0.5291772

    side1_indices, side2_indices = find_fragments_bfs(atoms, q, atom1_idx, atom2_idx, bond_th_HX, bond_th_XX)

    side1_atoms = [atoms[i] for i in side1_indices]
    side2_atoms = [atoms[i] for i in side2_indices]

    side1_formula = create_chemical_formula(side1_atoms)
    side2_formula = create_chemical_formula(side2_atoms)

    print(f"Atoms on side atom {atom1_idx} of the axis: {side1_indices}")
    print("Chemical formula for side 1:", side1_formula)
    print()
    print(f"Atoms on side atom {atom2_idx} of the axis: {side2_indices}")
    print("Chemical formula for side 2:", side2_formula)


