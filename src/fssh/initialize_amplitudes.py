import numpy as np


def initialize_amplitudes(n_states, starting_state):
    c = np.zeros(n_states, dtype=complex)
    c[starting_state] = 1.0
    return c
