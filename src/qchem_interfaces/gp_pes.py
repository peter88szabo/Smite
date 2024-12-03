import numpy as np

from gp_models import get_gpmodel

def PES_Energy(qcoord: np.ndarray, states: int | list | None = None):
    model = get_gpmodel()
    num_states = model.n_states()

    if isinstance(states, int):
        epot = model.mean(qcoord, states)
    elif isinstance(states, list):
        epot = np.zeros(len(states))
        for idx, state in enumerate(states):
            epot[idx] = model.mean(qcoord, states)
    elif num_states == 1:
        epot = model.mean(qcoord)
    else:
        epot = np.zeros(num_states)
        for idx, state in enumerate(range(num_states)):
            epot[idx] = model.mean(qcoord)

    return epot

def PES_Force(qcoord: np.ndarray, states: int | list | None = None):
    model = get_gpmodel()
    num_states = model.n_states()

    if isinstance(states, int):
        force = -model.grad(qcoord, states).reshape(-1, 3)
    elif isinstance(states, list):
        force = np.zeros((len(states), qcoord.shape[0], qcoord.shape[1]))
        for idx, state in enumerate(states):
            force[idx] = -model.grad(qcoord, state).reshape(-1, 3)
    elif num_states == 1:
        force = -model.grad(qcoord).reshape(-1, 3)
    else:
        force = np.zeros((len(num_states), qcoord.shape[0], qcoord.shape[1]))
        for idx, state in enumerate(range(num_states)):
            force[idx] = -model.grad(qcoord).reshape(-1, 3)

    return force

def PES_Hessian(qcoord: np.ndarray, states: int | list | None = None):
    model = get_gpmodel()
    num_states = model.n_states()

    if isinstance(states, int):
        hess = model.hess(qcoord, states)
    elif isinstance(states, list):
        hess = np.zeros((len(states), qcoord.shape[0] * qcoord.shape[1], qcoord.shape[0] * qcoord.shape[1]))
        for idx, state in enumerate(states):
            hess[idx] = model.hess(qcoord, states)
    elif num_states == 1:
        hess = model.hess(qcoord)
    else:
        hess = np.zeros((len(num_states), qcoord.shape[0] * qcoord.shape[1], qcoord.shape[0] * qcoord.shape[1]))
        for idx, state in enumerate(range(num_states)):
            hess[idx] = model.hess(qcoord)

    return hess

