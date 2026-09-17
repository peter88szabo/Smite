"""Smite interface to the H2O + Kr(+) machine-learned potential energy surfaces.

Two TorchScript Gaussian-process models, one per electronic state, trained on
CI calculations including spin-orbit coupling::

    h2okr_ci_soc_dz_s0_*.pt    ground state,  ion-dipole well ~0.89 eV at 2.4 A
    h2okr_ci_soc_dz_s1_*.pt    excited state, repulsive, ~0.97 eV asymptote

``config["state"]`` selects the surface, so one directory serves both states
through two ``qcinput`` dicts -- which is what lets ``fssh`` hop between them::

    states = [{"qchem": "PES", "pes_name": "H2O+Kr+", "state": 0},
              {"qchem": "PES", "pes_name": "H2O+Kr+", "state": 1}]

Units
-----
The models take the twelve Cartesian coordinates **in bohr** and return energies
**in eV**, so this interface feeds Smite's ``q`` straight through and converts
the result to Hartree.

Note that this deliberately bypasses ``qchem_interfaces.gp_models.GPModel``,
whose ``transform_data`` multiplies the incoming coordinates by ``ang2bohr``.
That is correct only for a caller working in Angstrom; Smite works in bohr, so
going through it would hand the models coordinates 1.889x too large. Verified
against the models' own GP variance: fed bohr the variance stays at 1e-5..3e-3
across the whole Kr-O range, fed Angstrom it rises to 1.8..11.7 and the energies
are unphysical.
"""

import os

import numpy as np

# eV -> Hartree
EV_TO_HARTREE = 1.0 / 27.2114


class PESCalculator:
    """Smite interface to the machine-learned H2O + Kr(+) surfaces."""

    name = "H2O+Kr+"
    natoms = 4
    required_atoms = ("O", "H", "H", "Kr")

    def __init__(self, pes_dir, config=None):
        import torch  # imported lazily so the package works without torch

        self.torch = torch
        self.pes_dir = os.path.abspath(pes_dir)
        self.config = config or {}

        self.state = int(self.config.get("state", 0))
        if self.state not in (0, 1):
            raise ValueError(
                f"{self.name} provides states 0 and 1, got state={self.state}"
            )

        model_path = self._resolve_model_path()
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.model_path = model_path

    def _resolve_model_path(self):
        """Find the .pt whose name carries this state, e.g. ``..._s1_....pt``."""
        tag = f"_s{self.state}_"
        candidates = sorted(
            entry for entry in os.listdir(self.pes_dir)
            if entry.endswith(".pt") and tag in entry
        )
        if not candidates:
            raise FileNotFoundError(
                f"No model for state {self.state} in {self.pes_dir}: expected a "
                f"file whose name contains '{tag}' and ends in .pt"
            )
        if len(candidates) > 1:
            raise ValueError(
                f"Ambiguous models for state {self.state} in {self.pes_dir}: "
                f"{candidates}"
            )
        return os.path.join(self.pes_dir, candidates[0])

    # ------------------------------------------------------------------
    # Smite PESCalculator protocol
    # ------------------------------------------------------------------

    def energy(self, q, atoms):
        energy, _ = self.energy_and_gradient(q, atoms)
        return energy

    def gradient(self, q, atoms):
        _, gradient = self.energy_and_gradient(q, atoms)
        return gradient

    def force(self, q, atoms):
        return -self.gradient(q, atoms)

    def energy_and_gradient(self, q, atoms):
        q_ordered, inverse = self._ordered_coordinates(q, atoms)
        x = self.torch.tensor(
            q_ordered.reshape(1, -1), dtype=self.torch.float64, requires_grad=True
        )

        mean = self.model(x)[0].reshape(())
        (grad_ordered,) = self.torch.autograd.grad(mean, x)

        energy = float(mean.detach()) * EV_TO_HARTREE
        gradient = grad_ordered.detach().numpy().reshape(-1) * EV_TO_HARTREE
        return energy, self._restore_order(gradient, inverse)

    def hessian(self, q, atoms):
        """Exact second derivatives by autograd, in Hartree/bohr^2."""
        q_ordered, inverse = self._ordered_coordinates(q, atoms)

        def mean_of(flat):
            return self.model(flat.reshape(1, -1))[0].reshape(())

        x = self.torch.tensor(q_ordered, dtype=self.torch.float64)
        hess = self.torch.autograd.functional.hessian(mean_of, x)
        hess = hess.detach().numpy().reshape(3 * self.natoms, 3 * self.natoms)
        hess = hess * EV_TO_HARTREE

        # Undo the atom permutation on both axes.
        permutation = np.empty(self.natoms, dtype=int)
        for ordered_index, original_index in enumerate(inverse):
            permutation[ordered_index] = original_index
        coordinate_order = np.concatenate(
            [3 * index + np.arange(3) for index in permutation]
        )
        restored = np.zeros_like(hess)
        restored[np.ix_(coordinate_order, coordinate_order)] = hess
        return restored

    def variance(self, q, atoms):
        """GP variance in eV^2 -- large values mean the geometry is extrapolated.

        Not part of the PESCalculator protocol; exposed because a trajectory that
        wanders out of the training region is a silent failure otherwise.
        """
        q_ordered, _ = self._ordered_coordinates(q, atoms)
        with self.torch.no_grad():
            _, var = self.model(
                self.torch.tensor(q_ordered.reshape(1, -1), dtype=self.torch.float64)
            )
        return float(var)

    # ------------------------------------------------------------------
    # Atom ordering: the models expect O, H, H, Kr
    # ------------------------------------------------------------------

    def _ordered_coordinates(self, q, atoms):
        atoms = list(atoms)
        q = np.asarray(q, dtype=float).reshape(-1)
        if len(atoms) != self.natoms or q.size != 3 * self.natoms:
            raise ValueError(
                f"{self.name} requires exactly four atoms and twelve coordinates"
            )
        if atoms.count("O") != 1 or atoms.count("H") != 2 or atoms.count("Kr") != 1:
            raise ValueError(
                f"{self.name} requires atom composition O/H/H/Kr, got {atoms}"
            )

        order = (
            [atoms.index("O")]
            + [i for i, atom in enumerate(atoms) if atom == "H"]
            + [atoms.index("Kr")]
        )

        q_ordered = q.reshape(self.natoms, 3)[order, :].reshape(-1)

        inverse = np.empty(self.natoms, dtype=int)
        for ordered_index, original_index in enumerate(order):
            inverse[ordered_index] = original_index

        return q_ordered, inverse

    def _restore_order(self, vector_ordered, inverse):
        ordered_matrix = np.asarray(vector_ordered, dtype=float).reshape(self.natoms, 3)
        restored = np.zeros((self.natoms, 3), dtype=float)
        for ordered_index, original_index in enumerate(inverse):
            restored[original_index, :] = ordered_matrix[ordered_index, :]
        return restored.reshape(-1)
