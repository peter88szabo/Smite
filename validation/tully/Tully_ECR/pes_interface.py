"""Smite PES interface to Tully's ECR model.

One atom moving along x; y and z carry no force. ``config["state"]`` picks the
adiabatic surface, 0 for the lower and 1 for the upper, the same convention the
other two-state surfaces in this package use.

Tully's standard reduced mass for these models is 2000 a.u. Set it on the
Molecule; nothing here depends on it.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tully_models import adiabatic_energies          # noqa: E402

MODEL = "ECR"


class PESCalculator:
    name = "Tully_ECR"
    natoms = 1
    required_atoms = ("X",)

    def __init__(self, pes_dir, config=None):
        config = config or {}
        self.state = int(config.get("state", 0))
        if self.state not in (0, 1):
            raise ValueError(f"Tully models have states 0 and 1, got {self.state}")
        self.step = float(config.get("gradient_step", 1.0e-5))

    def _surface(self, x):
        lower, upper = adiabatic_energies(MODEL, np.atleast_1d(x))
        return float((lower if self.state == 0 else upper)[0])

    def energy(self, q, atoms):
        return self._surface(np.asarray(q, dtype=float).reshape(-1)[0])

    def gradient(self, q, atoms):
        q = np.asarray(q, dtype=float).reshape(-1)
        grad = np.zeros_like(q)
        grad[0] = (self._surface(q[0] + self.step)
                   - self._surface(q[0] - self.step)) / (2.0 * self.step)
        return grad

    def force(self, q, atoms):
        return -self.gradient(q, atoms)

    def energy_and_gradient(self, q, atoms):
        return self.energy(q, atoms), self.gradient(q, atoms)

    def hessian(self, q, atoms):
        """Only the xx element is non-zero; y and z carry no potential.

        Supplying it explicitly matters: without a ``hessian`` method pesrun
        falls back to central differences over all three Cartesian directions,
        which costs twelve energy evaluations per state per step for a quantity
        that has one non-trivial entry.
        """
        q = np.asarray(q, dtype=float).reshape(-1)
        hess = np.zeros((q.size, q.size))
        h = max(self.step, 1.0e-4)
        hess[0, 0] = (self._surface(q[0] + h) - 2.0 * self._surface(q[0])
                      + self._surface(q[0] - h)) / h**2
        return hess
