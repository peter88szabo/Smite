from typing import Optional

import numpy as np
import torch
from torch import ScriptModule
from sklearn.base import TransformerMixin

from qchem_interfaces.gp_modelmanager import get_modelmanager

# [Hartree]*ha2ev=[eV]
ha2ev = 27.2114
# [eV]*ev2ha=[Hartree]
ev2ha = 1 / ha2ev


class GPModel:
    """
    Class to store the Gaussian process models and the transformers for the input data.

    Parameters
    ----------
    models : list[ScriptModule]
        List of GPyTorch models.
    transformers : Optional[list[TransformerMixin]]
        List of transformers for the input data.

    Returns
    -------
    GPModel
        Object containing the Gaussian process models and transformers.

    Raises
    ------
    NotImplementedError
        If the state is not specified and there are multiple states.
    ValueError
        If the specified state is not available.
    """

    def __init__(
        self,
        models: list[ScriptModule],
        transformers: Optional[list[TransformerMixin]] = None,
    ):
        self.models = models
        self.transformers = transformers

    def n_states(self):
        return len(self.models)

    def transform_data(
        self, x: np.ndarray[float], state: Optional[int] = None
    ) -> np.ndarray[float]:
        """
        Transform the input data using the specified transformer.
        """
        state = self._determine_state(state)

        if self.transformers is not None:
            return self.transformers[state].transform(x)
        else:
            return x

    def mean(
        self, x: np.ndarray[float], state: Optional[int] = None
    ) -> np.ndarray[float]:
        """
        Calculate the mean function for the specified coordinates.
        The mean is expected to be energy in eV.
        The energy is converted to Hartree.
        """
        state = self._determine_state(state)

        x = self.transform_data(x, state)
        x = torch.tensor(x, dtype=torch.float64)
        print(x)
        return self.models[state](x)[0].detach().numpy() * ev2ha

    def variance(
        self, x: np.ndarray[float], state: Optional[int] = None
    ) -> np.ndarray[float]:
        """
        Calculate the variance of the mean function for the specified coordinates.
        """
        state = self._determine_state(state)

        x = self.transform_data(x, state)
        x = torch.tensor(x, dtype=torch.float64)
        return self.models[state](x)[1].detach().numpy()

    def grad(
        self, x: np.ndarray[float], state: Optional[int] = None
    ) -> np.ndarray[float]:
        """
        Calculate the gradient of the mean function for the specified coordinates.
        The gradient is expected to have units eV/bohr.
        The gradient is converted to Hartree/bohr.
        """
        state = self._determine_state(state)

        # Reshape the input data to a 2D array
        x = self.transform_data(x, state)

        def mean_f(x):
            return self.models[state](x)[0]

        x = torch.autograd.Variable(
            torch.tensor(x, dtype=torch.float64), requires_grad=True
        )

        grad = torch.autograd.functional.jacobian(mean_f, x).squeeze()

        return grad.detach().numpy() * ev2ha

    def hess(
        self, x: np.ndarray[float], state: Optional[int] = None
    ) -> np.ndarray[float]:
        """
        Calculate the Hessian of the mean function for the specified coordinates.
        The Hessian is expected to have units eV/bohr^2.
        The Hessian is converted to Hartree/bohr^2.
        """
        state = self._determine_state(state)

        x = self.transform_data(x, state)

        def mean_f(x):
            return self.models[state](x)[0]

        x = torch.autograd.Variable(
            torch.tensor(x, dtype=torch.float64), requires_grad=True
        )

        hess = torch.autograd.functional.hessian(mean_f, x).squeeze()

        return hess.detach().numpy() * ev2ha

    def model(self, state: Optional[int] = None) -> ScriptModule:
        """
        Return the GPyTorch model for the specified state.
        """
        state = self._determine_state(state)
        return self.models[state]

    def _determine_state(self, state: Optional[int] = None) -> int:
        """
        Helper function to determine the state if it is not specified.
        """
        if state is None:
            if self.n_states() > 1:
                raise NotImplementedError(
                    "The state must be specified if there are multiple states."
                )
            else:
                state = 0
        elif state not in range(self.n_states()):
            raise ValueError(f"The specified state {state} is not available.")
        return state


_manager = get_modelmanager()
_models = _manager.get_models()
_gpmodel = GPModel(_models)


# Implicit singleton implementation through the Python module system
def get_gpmodel():
    return _gpmodel
