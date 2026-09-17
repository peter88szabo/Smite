"""Input adapters for an existing neutral Hessian or saved neutral MD frames.

No neutral electronic-structure Hessian calculation or MD run is started here.
The QCT path calls SMite's existing normal-mode and sampling routines.
"""

from contextlib import contextmanager
from pathlib import Path
from numbers import Integral
import random
import threading

import numpy as np

from normalmode.normalmode import getNormalmode
from photoionization.preparation import validate_phase_point
from sampling.polyvibration import (
    initialize_vibrational_modes, specify_vib_modes, polyatom_vibration_sampling,
)
from sampling.polyrotation import initialize_rotational_modes, polyatom_rotation_sampling
from utils.constants import ANGSTROM_TO_BOHR
from utils.euler import euler_rot


_SAMPLING_LOCK = threading.RLock()


def positive_integer(value, name, *, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if allow_zero else 'positive'} integer")
    return int(value)


@contextmanager
def _legacy_sampling_stream(rng):
    # Existing QCT routines use Python/NumPy global RNGs. Scope their seed and
    # restore the caller's streams, without editing the neutral samplers.
    with _SAMPLING_LOCK:
        python_state, numpy_state = random.getstate(), np.random.get_state()
        seed = int(rng.integers(0, 2**32, dtype=np.uint64))
        try:
            random.seed(seed)
            np.random.seed(seed)
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def _geometry(value, molecule):
    if value is None:
        return np.asarray(molecule.q_ini, dtype=float).copy()
    if isinstance(value, (str, Path)):
        if isinstance(value, Path) or "\n" not in value:
            text = Path(value).read_text()
        else:
            text = value
        lines = text.strip().splitlines()
        if lines and lines[0].strip().isdigit():
            count = int(lines[0])
            if count != molecule.natom or len(lines) != count + 2:
                raise ValueError("Neutral XYZ structure must contain exactly one matching geometry")
            lines = lines[2:]
        rows = [line.split() for line in lines if line.strip()]
        if [row[0] for row in rows] != list(molecule.atoms) or any(len(row) != 4 for row in rows):
            raise ValueError("Neutral geometry atom identities/order must match the molecule")
        return np.array([row[1:] for row in rows], dtype=float).reshape(-1) * ANGSTROM_TO_BOHR
    return np.asarray(value, dtype=float).copy()


class HessianSource:
    def __init__(self, molecule, hessian, geometry=None, options=None):
        self.atoms = list(molecule.atoms)
        q = _geometry(geometry, molecule)
        self.q, _, self.mass = validate_phase_point(q, np.zeros(3 * molecule.natom), molecule.mass)
        if self.mass.size < 2:
            raise ValueError("Hessian QCT preparation requires at least two atoms")
        hess = np.loadtxt(hessian) if isinstance(hessian, (str, Path)) else np.asarray(hessian, dtype=float)
        if hess.shape != (self.q.size, self.q.size) or not np.all(np.isfinite(hess)):
            raise ValueError("Neutral Hessian must be a finite 3N by 3N Cartesian matrix in Hartree/bohr^2")
        if not np.allclose(hess, hess.T, atol=1e-10, rtol=1e-7):
            raise ValueError("Neutral Hessian must be symmetric")
        options = dict(options or {})
        allowed = {"init_vib_type", "init_rot_type", "temp", "jrot", "fix_quantum",
                   "fix_energy", "fix_temp", "fix_wigner", "random_rot"}
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(f"Unknown qct_options: {sorted(unknown)}")
        centered = self.q.reshape(-1, 3) - np.average(self.q.reshape(-1, 3), weights=self.mass, axis=0)
        rank = np.linalg.matrix_rank(centered)
        if rank == 0:
            raise ValueError("Neutral geometry has coincident atoms")
        linear = rank == 1
        self.freq, _, self.modes = getNormalmode(
            self.mass, 0.5 * (hess + hess.T), linear=linear, q_eq=self.q.copy(), is_eckart=True)
        self.freq = np.asarray(self.freq)
        expected = self.q.size - (5 if linear else 6)
        if self.freq.size != expected or np.any(self.freq <= 0):
            raise ValueError("Ground-state QCT requires a minimum with positive vibrational frequencies")
        temp = options.get("temp")
        if str(options.get("init_vib_type", "ZPE")).lower() == "temp" or options.get("init_rot_type") == "Temp":
            if temp is None or not np.isfinite(temp) or temp < 0:
                raise ValueError("Thermal QCT sampling requires a finite nonnegative temp (K)")
        self.vib = initialize_vibrational_modes(self.freq, options.get("init_vib_type", "ZPE"), temp=temp)
        for key in ("fix_quantum", "fix_energy", "fix_temp", "fix_wigner"):
            for entry in options.get(key, []):
                index = entry if key == "fix_wigner" else entry[0]
                if index not in self.vib:
                    raise ValueError(f"{key} contains an invalid normal-mode index: {index}")
                if key != "fix_wigner" and (not np.isfinite(entry[1]) or entry[1] < 0):
                    raise ValueError(f"{key} values must be finite and nonnegative")
                if key == "fix_quantum" and int(entry[1]) != entry[1]:
                    raise ValueError("fix_quantum requires integer vibrational quantum numbers")
        self.vib = specify_vib_modes(self.vib, **options)
        self.rot = initialize_rotational_modes(options.get("init_rot_type", "Jfix"), temp=temp, jrot=options.get("jrot"))
        self.random_rot = options.get("random_rot", False)

    def draw(self, rng, index):
        with _legacy_sampling_stream(rng):
            q, p = polyatom_vibration_sampling(
                self.mass, self.atoms, self.q.copy(), self.freq, self.modes, self.vib,
                mode_energy_diagnostics=False, traj_index=index)
            if self.rot[0][1] is not None:
                p, _, _ = polyatom_rotation_sampling(self.rot, self.mass, q, p)
            if self.random_rot:
                q, p = euler_rot(q, p)
        q, p, _ = validate_phase_point(q, p, self.mass)
        return q, p, index


class MDTrajectorySource:
    """Uniform sampling with replacement from saved SMite extended XYZ frames.

    Coordinates in the file are Angstrom; the last three columns are atomic
    momenta in atomic units. Atom labels/order and complete frames are checked.
    Frame selection never recenters, rotates or modifies the saved momenta.
    """

    def __init__(self, molecule, path, start=0, stride=1):
        start = positive_integer(start, "md_start", allow_zero=True)
        stride = positive_integer(stride, "md_stride")
        self.frames = []
        with open(path, encoding="utf-8") as handle:
            frame = 0
            while True:
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    count = int(line.strip())
                except ValueError as exc:
                    raise ValueError(f"MD frame {frame}: invalid atom count") from exc
                if count != molecule.natom or not handle.readline():
                    raise ValueError(f"MD frame {frame}: atom count mismatch or incomplete header")
                rows = [handle.readline().split() for _ in range(count)]
                if any(len(row) != 7 for row in rows):
                    raise ValueError(f"MD frame {frame}: each atom needs symbol, x y z, px py pz")
                if [row[0] for row in rows] != list(molecule.atoms):
                    raise ValueError(f"MD frame {frame}: atom identities/order differ from molecule")
                values = np.asarray([row[1:] for row in rows], dtype=float)
                q, p, _ = validate_phase_point(values[:, :3] * ANGSTROM_TO_BOHR,
                                               values[:, 3:], molecule.mass)
                if frame >= start and (frame - start) % stride == 0:
                    self.frames.append((q, p, frame))
                frame += 1
        if not self.frames:
            raise ValueError("No MD frames remain after md_start/md_stride selection")

    def draw(self, rng, index):
        del index
        q, p, frame = self.frames[int(rng.integers(len(self.frames)))]
        return q.copy(), p.copy(), frame
