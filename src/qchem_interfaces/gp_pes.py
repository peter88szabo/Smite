import numpy as np
from numpy._typing import NDArray

from qchem_interfaces.gp_models import get_gpmodel


def PES_Energy(qcoord: np.ndarray, states: int | list | None = None) -> NDArray | float:
    model = get_gpmodel()
    num_states = model.n_states()

    if isinstance(states, int):
        epot = model.mean(qcoord, states).item()
    elif isinstance(states, list):
        epot = np.zeros(len(states))
        for idx, state in enumerate(states):
            epot[idx] = model.mean(qcoord, state)
    elif num_states == 1:
        epot = model.mean(qcoord, 0)
    else:
        epot = np.zeros(num_states)
        for idx, state in enumerate(range(num_states)):
            epot[idx] = model.mean(qcoord, idx).item()
    return epot


def PES_Force(qcoord: np.ndarray, states: int | list | None = None) -> NDArray:
    model = get_gpmodel()
    num_states = model.n_states()

    if isinstance(states, int):
        force = -model.grad(qcoord, states)
    elif isinstance(states, list):
        force = np.zeros((len(states), qcoord.shape[0]))
        for idx, state in enumerate(states):
            force[idx] = -model.grad(qcoord, state)
    elif num_states == 1:
        force = -model.grad(qcoord, 0)
    else:
        force = np.zeros((len(range(num_states)), qcoord.shape[0]))
        for idx, state in enumerate(range(num_states)):
            force[idx] = -model.grad(qcoord, idx)

    return force


def PES_Hessian(qcoord: np.ndarray, states: int | list | None = None) -> NDArray:
    model = get_gpmodel()
    num_states = model.n_states()

    if isinstance(states, int):
        hess = model.hess(qcoord, states)
    elif isinstance(states, list):
        hess = np.zeros(
            (
                len(states),
                qcoord.shape[0],
                qcoord.shape[0],
            )
        )
        for idx, state in enumerate(states):
            hess[idx] = model.hess(qcoord, state)
    elif num_states == 1:
        hess = model.hess(qcoord, 0)
    else:
        hess = np.zeros(
            (
                len(num_states),
                qcoord.shape[0],
                qcoord.shape[0],
            )
        )
        for idx, state in enumerate(range(num_states)):
            hess[idx] = model.hess(qcoord, idx)

    return hess
