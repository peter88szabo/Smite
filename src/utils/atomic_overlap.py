import numpy as np
from utils.constants import ANGSTROM_TO_BOHR
def check_atomic_overlap(atoms, q, **kwargs):

    # Default threshold values
    H_H_threshold = kwargs.get('H_H_threshold', 0.4 * ANGSTROM_TO_BOHR)
    H_X_threshold = kwargs.get('H_X_threshold', 0.5 * ANGSTROM_TO_BOHR)
    X_X_threshold = kwargs.get('X_X_threshold', 0.8 * ANGSTROM_TO_BOHR)

    N = len(atoms)
    coordinates = np.reshape(q, (N, 3))
    is_overlap = False
    
    for i in range(N):
        for j in range(i + 1, N):
            distance = np.linalg.norm(coordinates[i] - coordinates[j])
            
            if atoms[i] == 'H' and atoms[j] == 'H':
                if distance < H_H_threshold:
                    is_overlap = True
                    return is_overlap
            elif atoms[i] == 'H' or atoms[j] == 'H':
                if distance < H_X_threshold:
                    is_overlap = True
                    return is_overlap
            else:
                if distance < X_X_threshold:
                    is_overlap = True
                    return is_overlap
    return is_overlap
