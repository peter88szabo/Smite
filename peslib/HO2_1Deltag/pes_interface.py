import ctypes
import os
import threading
from contextlib import contextmanager

import numpy as np


_PES_LOCK = threading.RLock()


@contextmanager
def _working_directory(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class PESCalculator:
    """Smite interface to the analytical HO2 1-Delta-g surface."""

    name = "HO2_1Deltag"
    natoms = 3
    required_atoms = ("H", "O", "O")
    required_data_files = ("ho2pes-excited.dat", "para-ex.txt")

    def __init__(self, pes_dir, config=None):
        self.pes_dir = os.path.abspath(pes_dir)
        self.config = config or {}

        missing = [
            filename
            for filename in self.required_data_files
            if not os.path.isfile(os.path.join(self.pes_dir, filename))
        ]
        if missing:
            raise FileNotFoundError(
                "HO2_1Deltag is missing native PES data file(s): "
                + ", ".join(missing)
            )

        libname = self.config.get("library", "libho2_1deltag.so")
        libpath = (
            libname
            if os.path.isabs(libname)
            else os.path.join(self.pes_dir, libname)
        )
        if not os.path.isfile(libpath):
            raise FileNotFoundError(
                f"Missing HO2 1-Delta-g shared library: {libpath}. "
                f"Build it with `make` in {self.pes_dir}."
            )

        self.lib = ctypes.CDLL(libpath)
        self.lib.ho2_1deltag_init.argtypes = []
        self.lib.ho2_1deltag_init.restype = None
        self.lib.ho2_1deltag_energy_gradient.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.ho2_1deltag_energy_gradient.restype = None

        # The legacy Fortran reads data by relative filename and stores spline
        # tables in COMMON blocks, so initialization and calls are serialized.
        with _PES_LOCK, _working_directory(self.pes_dir):
            self.lib.ho2_1deltag_init()

    def energy(self, q, atoms):
        energy, _gradient = self.energy_and_gradient(q, atoms)
        return energy

    def gradient(self, q, atoms):
        _energy, gradient = self.energy_and_gradient(q, atoms)
        return gradient

    def force(self, q, atoms):
        return -self.gradient(q, atoms)

    def energy_and_gradient(self, q, atoms):
        q_ordered, order = self._ordered_coordinates(q, atoms)
        q_ordered = np.ascontiguousarray(q_ordered, dtype=np.float64)
        gradient_ordered = np.zeros(9, dtype=np.float64)
        energy = ctypes.c_double()
        info = ctypes.c_int()

        with _PES_LOCK, _working_directory(self.pes_dir):
            self.lib.ho2_1deltag_energy_gradient(
                q_ordered.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int(self.natoms),
                ctypes.byref(energy),
                gradient_ordered.ctypes.data_as(
                    ctypes.POINTER(ctypes.c_double)
                ),
                ctypes.byref(info),
            )

        if info.value != 0:
            messages = {
                1: "expected exactly three atoms",
                2: "encountered a zero internuclear distance",
                3: "encountered non-finite coordinates or PES output",
            }
            detail = messages.get(info.value, "unknown native-interface error")
            raise RuntimeError(
                f"HO2_1Deltag PES failed with info={info.value}: {detail}"
            )

        gradient = self._restore_order(gradient_ordered, order)
        if not np.isfinite(energy.value) or not np.all(np.isfinite(gradient)):
            raise RuntimeError("HO2_1Deltag returned a non-finite energy or gradient")
        return float(energy.value), gradient

    def _ordered_coordinates(self, q, atoms):
        atoms = list(atoms)
        q = np.asarray(q, dtype=float).reshape(-1)
        if len(atoms) != self.natoms or q.size != 9:
            raise ValueError(
                "HO2_1Deltag requires exactly three atoms and nine coordinates"
            )
        if atoms.count("H") != 1 or atoms.count("O") != 2:
            raise ValueError("HO2_1Deltag requires atom composition H/O/O")
        if not np.all(np.isfinite(q)):
            raise ValueError("HO2_1Deltag coordinates must be finite")

        h_index = atoms.index("H")
        oxygen_indices = [i for i, atom in enumerate(atoms) if atom == "O"]
        order = np.array([h_index, *oxygen_indices], dtype=int)
        q_ordered = q.reshape(self.natoms, 3)[order, :].reshape(9)
        return q_ordered, order

    def _restore_order(self, vector_ordered, order):
        ordered = np.asarray(vector_ordered, dtype=float).reshape(self.natoms, 3)
        restored = np.zeros((self.natoms, 3), dtype=float)
        restored[order, :] = ordered
        return restored.reshape(9)
