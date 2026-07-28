import ctypes
import os

import numpy as np


class PESCalculator:
    name = "F+H2"
    natoms = 3

    def __init__(self, pes_dir, config=None):
        self.pes_dir = os.path.abspath(pes_dir)
        self.config = config or {}
        libname = self.config.get("library", "libfh2_pes.so")
        libpath = libname if os.path.isabs(libname) else os.path.join(self.pes_dir, libname)

        if not os.path.exists(libpath):
            raise FileNotFoundError(f"Missing F+H2 PES shared library: {libpath}. Build it with `make` in {self.pes_dir}.")

        self.lib = ctypes.CDLL(libpath)
        self.lib.fh2_init.argtypes = []
        self.lib.fh2_init.restype = None
        self.lib.fh2_energy_gradient.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.fh2_energy_gradient.restype = None
        self.lib.fh2_init()

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
        q_ordered = np.ascontiguousarray(q_ordered, dtype=np.float64)
        gradient_ordered = np.zeros(9, dtype=np.float64)
        energy = ctypes.c_double()
        info = ctypes.c_int(0)

        self.lib.fh2_energy_gradient(
            q_ordered.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(self.natoms),
            ctypes.byref(energy),
            gradient_ordered.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(info),
        )

        if info.value != 0:
            raise RuntimeError(f"F+H2 PES failed with info={info.value}")

        return energy.value, self._restore_order(gradient_ordered, inverse)

    def _ordered_coordinates(self, q, atoms):
        atoms = list(atoms)
        q = np.asarray(q, dtype=float).reshape(-1)
        if len(atoms) != self.natoms or q.size != 9:
            raise ValueError("F+H2 requires exactly three atoms and nine coordinates")
        if atoms.count("F") != 1 or atoms.count("H") != 2:
            raise ValueError("F+H2 requires atom composition F/H/H")

        f_index = atoms.index("F")
        h_indices = [i for i, atom in enumerate(atoms) if atom == "H"]
        order = [f_index] + h_indices

        q_matrix = q.reshape(self.natoms, 3)
        q_ordered = q_matrix[order, :].reshape(9)

        inverse = np.empty(self.natoms, dtype=int)
        for ordered_index, original_index in enumerate(order):
            inverse[ordered_index] = original_index

        return q_ordered, inverse

    def _restore_order(self, vector_ordered, inverse):
        ordered_matrix = np.asarray(vector_ordered, dtype=float).reshape(self.natoms, 3)
        restored = np.zeros((self.natoms, 3), dtype=float)
        for ordered_index, original_index in enumerate(inverse):
            restored[original_index, :] = ordered_matrix[ordered_index, :]
        return restored.reshape(9)
