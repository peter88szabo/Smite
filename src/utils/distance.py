import numpy as np

def distance_matrix(q):
    N = len(q) // 3
    coordinates = np.reshape(q, (N, 3))
    dist_matrix = np.sqrt(np.sum((coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]) ** 2, axis=-1))
    return dist_matrix

def test_to_stop(q, tol=12.0):

    dist = distance_matrix(q)
    
    # Check if any distance in the lower triangle of the matrix exceeds the tolerance
    too_large = np.any(dist[np.tril_indices(dist.shape[0], -1)] > tol)

    return too_large
