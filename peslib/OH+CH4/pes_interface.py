import ctypes
import os

import numpy as np


class PESCalculator:
    name = "OH+CH4"
    natoms = 7
    methane_natoms = 5

    def __init__(self, pes_dir, config=None):
        self.pes_dir = os.path.abspath(pes_dir)
        self.config = config or {}
        libname = self.config.get("library", "libohch4_pes.so")
        libpath = libname if os.path.isabs(libname) else os.path.join(self.pes_dir, libname)

        if not os.path.exists(libpath):
            raise FileNotFoundError(f"Missing OH+CH4 PES shared library: {libpath}. Build it with `make` in {self.pes_dir}.")

        self.lib = ctypes.CDLL(libpath)
        self.lib.ohch4_init.argtypes = []
        self.lib.ohch4_init.restype = None
        self.lib.ohch4_energy_gradient.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.ohch4_energy_gradient.restype = None
        self.lib.ohch4_init()

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
        gradient_ordered = np.zeros(21, dtype=np.float64)
        energy = ctypes.c_double()
        info = ctypes.c_int(0)

        self.lib.ohch4_energy_gradient(
            q_ordered.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(self.natoms),
            ctypes.byref(energy),
            gradient_ordered.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(info),
        )

        if info.value != 0:
            raise RuntimeError(f"OH+CH4 PES failed with info={info.value}")

        return energy.value, self._restore_order(gradient_ordered, inverse)

    def _ordered_coordinates(self, q, atoms):
        atoms = list(atoms)
        q = np.asarray(q, dtype=float).reshape(-1)
        if atoms.count("C") != 1 or atoms.count("H") < 4:
            raise ValueError("OH+CH4 requires at least atom composition C/H4")

        if len(atoms) == self.methane_natoms and q.size == 15 and atoms.count("O") == 0:
            return self._ordered_isolated_methane(q, atoms)

        if len(atoms) != self.natoms or q.size != 21 or atoms.count("O") != 1 or atoms.count("H") != 5:
            raise ValueError("OH+CH4 requires C/H4 for isolated methane or C/H5/O for the full PES")

        c_index = atoms.index("C")
        o_index = atoms.index("O")
        h_indices = [i for i, atom in enumerate(atoms) if atom == "H"]

        # The native PES distinguishes the OH hydrogen from the four methane
        # hydrogens. This must be an immutable input-atom identity: choosing
        # the closest H at every PES call switches the coordinate permutation
        # during H abstraction and creates a discontinuous force field.
        oh_h_index = self.config.get("oh_h_index", self.config.get("oh_h_atom_index"))
        if oh_h_index is None:
            raise ValueError(
                "OH+CH4 requires qcinput['oh_h_index'] (the zero-based index "
                "of the OH hydrogen in the supplied atom ordering)."
            )
        if isinstance(oh_h_index, bool) or int(oh_h_index) != oh_h_index:
            raise ValueError("OH+CH4 oh_h_index must be an integer atom index")
        oh_h_index = int(oh_h_index)
        if oh_h_index not in h_indices:
            raise ValueError("OH+CH4 oh_h_index must refer to one of the five H atoms")
        ch_h_indices = [idx for idx in h_indices if idx != oh_h_index]

        # Native POTLIB order for this PES is H1, C, H3, H4, H2, O, H(O).
        order = [ch_h_indices[0], c_index, ch_h_indices[1], ch_h_indices[2], ch_h_indices[3], o_index, oh_h_index]

        q_matrix = q.reshape(self.natoms, 3)
        q_ordered = q_matrix[order, :].reshape(21)

        inverse = np.empty(self.natoms, dtype=int)
        for ordered_index, original_index in enumerate(order):
            inverse[ordered_index] = original_index

        return q_ordered, inverse

    def _ordered_isolated_methane(self, q, atoms):
        c_index = atoms.index("C")
        h_indices = [i for i, atom in enumerate(atoms) if atom == "H"]
        order = [h_indices[0], c_index, h_indices[1], h_indices[2], h_indices[3]]

        q_matrix = q.reshape(self.methane_natoms, 3)
        ch4_ordered = q_matrix[order, :]

        dummy_distance = float(self.config.get("dummy_oh_distance_bohr", 100.0))
        dummy_oh_bond = float(self.config.get("dummy_oh_bond_bohr", 0.9707 / 0.52917721092))
        center = np.mean(q_matrix, axis=0)
        dummy_o = center + np.array([dummy_distance, 0.0, 0.0])
        dummy_h = dummy_o + np.array([dummy_oh_bond, 0.0, 0.0])
        q_ordered = np.vstack((ch4_ordered, dummy_o, dummy_h)).reshape(21)

        inverse = np.empty(self.natoms, dtype=int)
        for ordered_index, original_index in enumerate(order):
            inverse[ordered_index] = original_index
        inverse[5] = -1
        inverse[6] = -1

        return q_ordered, inverse

    def _restore_order(self, vector_ordered, inverse):
        ordered_matrix = np.asarray(vector_ordered, dtype=float).reshape(self.natoms, 3)
        natoms_out = int(np.count_nonzero(inverse >= 0))
        restored = np.zeros((natoms_out, 3), dtype=float)
        for ordered_index, original_index in enumerate(inverse):
            if original_index < 0:
                continue
            restored[original_index, :] = ordered_matrix[ordered_index, :]
        return restored.reshape(3 * natoms_out)
