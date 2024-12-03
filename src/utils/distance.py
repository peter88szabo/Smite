import numpy as np

def distance_matrix(q):
    N = len(q) // 3
    coordinates = np.reshape(q, (N, 3))
    dist_matrix = np.sqrt(np.sum((coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]) ** 2, axis=-1))
    return dist_matrix

def test_to_stop_general(q, tol=16.0):

    dist = distance_matrix(q)
    
    # Check if any distance in the lower triangle of the matrix exceeds the tolerance
    too_large = np.any(dist[np.tril_indices(dist.shape[0], -1)] > tol)

    return too_large


def test_to_stop_specific(q, pairs_to_test):
    '''
    pairs_to_test = {
    'channel_1': [((2, 17), 'GT', 9.0)],  # Only one pair for this channel
    'channel_2': [((2, 18), 'GT', 9.0), ((2, 4), 'GT', 9.0), ((2, 5), 'GT', 9.0)],  # Multiple pairs for channel_2
    'channel_3': [((2, 7), 'GT', 9.0)],  # Only one pair for this channel
    # add more channels and pairs as needed
    }
    '''

    dist = distance_matrix(q)

    b2a = 0.5291772 #bohr to angstrom

    for channel, conditions in pairs_to_test.items():
        all_conditions_met = True  #Assume all pairs for this channel need to pass

        for (i, j), cond, tolerance in conditions:
            if cond == 'GT':
                if not (dist[i, j] > tolerance / b2a):
                    all_conditions_met = False
                    break  # Stop checking this channel if one condition fails
            elif cond == 'LT':
                if not (dist[i, j] < tolerance / b2a):
                    all_conditions_met = False
                    break  # Stop checking this channel if one condition fails

        if all_conditions_met:
            return (True, channel)  # If all pairs met conditions, return the channel

    return (False, 'default')

