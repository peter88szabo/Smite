import ctypes
import os

import numpy as np


class PESCalculator:
    name = "Cl+CH4"
    natoms = 6
    methane_natoms = 5

    def __init__(self, pes_dir, config=None):
        self.pes_dir = os.path.abspath(pes_dir)
        self.config = config or {}
        libname = self.config.get("library", "libclch4_pes.so")
        libpath = libname if os.path.isabs(libname) else os.path.join(self.pes_dir, libname)

        if not os.path.exists(libpath):
            raise FileNotFoundError(f"Missing Cl+CH4 PES shared library: {libpath}. Build it with `make` in {self.pes_dir}.")

        self.lib = ctypes.CDLL(libpath)
        self.lib.clch4_init.argtypes = []
        self.lib.clch4_init.restype = None
        self.lib.clch4_energy_gradient.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.clch4_energy_gradient.restype = None
        self.lib.clch4_init()

    def energy(self, q, atoms):
        energy, _ = self.energy_and_gradient(q, atoms)
        return energy

    def gradient(self, q, atoms):
        _, gradient = self.energy_and_gradient(q, atoms)
        return gradient

    def force(self, q, atoms):
        return -self.gradient(q, atoms)

    def hessian(self, q, atoms):
        q = np.asarray(q, dtype=float).copy()
        atoms = list(atoms)
        dx = float(self.config.get("hessian_dx", 0.002))
        ndim = len(q)
        hess = np.zeros((ndim, ndim), dtype=float)

        for i in range(ndim):
            q[i] += dx
            gradp1 = self.gradient(q, atoms)

            q[i] -= 2.0 * dx
            gradm1 = self.gradient(q, atoms)

            hess[i, :] = 0.5 * (gradp1 - gradm1) / dx
            q[i] += dx

        return hess

    def energy_and_gradient(self, q, atoms):
        q_ordered, inverse = self._ordered_coordinates(q, atoms)
        q_ordered = np.ascontiguousarray(q_ordered, dtype=np.float64)
        gradient_ordered = np.zeros(18, dtype=np.float64)
        energy = ctypes.c_double()
        info = ctypes.c_int(0)

        self.lib.clch4_energy_gradient(
            q_ordered.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(self.natoms),
            ctypes.byref(energy),
            gradient_ordered.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(info),
        )

        if info.value != 0:
            raise RuntimeError(f"Cl+CH4 PES failed with info={info.value}")

        return energy.value, self._restore_order(gradient_ordered, inverse)

    def _ordered_coordinates(self, q, atoms):
        atoms = list(atoms)
        q = np.asarray(q, dtype=float).reshape(-1)
        if atoms.count("C") != 1 or atoms.count("H") != 4:
            raise ValueError("Cl+CH4 requires at least atom composition C/H4")

        if len(atoms) == self.methane_natoms and q.size == 15 and atoms.count("Cl") == 0:
            return self._ordered_isolated_methane(q, atoms)

        if len(atoms) != self.natoms or q.size != 18 or atoms.count("Cl") != 1:
            raise ValueError("Cl+CH4 requires C/H4 for isolated methane or C/H4/Cl for the full PES")

        c_index = atoms.index("C")
        cl_index = atoms.index("Cl")
        h_indices = [i for i, atom in enumerate(atoms) if atom == "H"]

        # Native POTLIB order for this PES is H1, C, H3, H4, H2, Cl.
        # The H labels are equivalent for methane, so keep the user's H order
        # while matching the native element sequence.
        order = [h_indices[0], c_index, h_indices[1], h_indices[2], h_indices[3], cl_index]

        q_matrix = q.reshape(self.natoms, 3)
        q_ordered = q_matrix[order, :].reshape(18)

        inverse = np.empty(self.natoms, dtype=int)
        for ordered_index, original_index in enumerate(order):
            inverse[ordered_index] = original_index

        return q_ordered, inverse

    def _ordered_isolated_methane(self, q, atoms):
        c_index = atoms.index("C")
        h_indices = [i for i, atom in enumerate(atoms) if atom == "H"]

        # Native POTLIB order for this PES is H1, C, H3, H4, H2, Cl.
        order = [h_indices[0], c_index, h_indices[1], h_indices[2], h_indices[3]]
        q_matrix = q.reshape(self.methane_natoms, 3)
        ch4_ordered = q_matrix[order, :]

        dummy_distance = float(self.config.get("dummy_cl_distance_bohr", 100.0))
        dummy_cl = self._center_of_mass_like_center(q_matrix) + np.array([dummy_distance, 0.0, 0.0])
        q_ordered = np.vstack((ch4_ordered, dummy_cl)).reshape(18)

        inverse = np.empty(self.natoms, dtype=int)
        for ordered_index, original_index in enumerate(order):
            inverse[ordered_index] = original_index
        inverse[5] = -1

        return q_ordered, inverse

    def _center_of_mass_like_center(self, q_matrix):
        return np.mean(q_matrix, axis=0)

    def _restore_order(self, vector_ordered, inverse):
        ordered_matrix = np.asarray(vector_ordered, dtype=float).reshape(self.natoms, 3)
        natoms_out = int(np.count_nonzero(inverse >= 0))
        restored = np.zeros((natoms_out, 3), dtype=float)
        for ordered_index, original_index in enumerate(inverse):
            if original_index < 0:
                continue
            restored[original_index, :] = ordered_matrix[ordered_index, :]
        return restored.reshape(3 * natoms_out)
