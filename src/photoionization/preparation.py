"""Franck–Condon transfer to one ionic PES; all nuclear arrays use atomic units."""

from dataclasses import dataclass, asdict

import numpy as np

from photoionization.distributions import EnergyConstraints, finite_scalar
from sampling.random_seed import sampling_generator
from utils.constants import HARTREE_TO_EV


@dataclass
class IonicInitialState:
    q: np.ndarray
    p: np.ndarray
    neutral_p: np.ndarray
    level: int
    source_index: int
    photon_energy_ev: float
    electron_energy_ev: float
    ionic_energy_ev: float
    neutral_potential_hartree: float
    ionic_potential_hartree: float
    ion_energy_offset_ev: float
    neutral_kinetic_ev: float
    ionic_kinetic_ev: float
    momentum_scale: float
    energy_residual_ev: float
    selection_weight: float
    neutral_linear_momentum: np.ndarray
    ionic_linear_momentum: np.ndarray
    neutral_angular_momentum: np.ndarray
    ionic_angular_momentum: np.ndarray

    def to_dict(self):
        """JSON-compatible launch record; angular momenta are about the COM."""
        return {key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in asdict(self).items()}


def validate_phase_point(q, p, mass):
    mass = np.asarray(mass, dtype=float)
    if mass.ndim != 1 or not mass.size or not np.all(np.isfinite(mass)) or np.any(mass <= 0):
        raise ValueError("Atomic masses must be a nonempty finite positive 1D array (atomic units)")
    q, p = np.asarray(q, dtype=float), np.asarray(p, dtype=float)
    valid_shapes = {(3 * mass.size,), (mass.size, 3)}
    if q.shape not in valid_shapes or p.shape not in valid_shapes:
        raise ValueError("Coordinates and momenta must each contain 3N Cartesian components")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(p)):
        raise ValueError("Coordinates and momenta must be finite")
    return q.reshape(-1).copy(), p.reshape(-1).copy(), mass.copy()


def prepare_ionic_state(q, p, mass, neutral_potential, ionic_potential, *,
                        photon_energy, level=0, electron_energy=None, ionic_energy=None,
                        ion_energy_offset=0.0, energy_tolerance=1e-8, seed=None,
                        rng=None, source_index=0, constraints=None):
    """Prepare one Cartesian launch without changing the Franck–Condon geometry.

    Potentials are in Hartree; photon/electron/ionic energies and the optional
    additive ionic PES offset are in eV. ``ionic_energy`` is the measured
    binding energy, not ionic kinetic energy or energy above an ionic minimum.
    Level 0 copies p. Level 1 scales every component by one positive factor.
    Neither level models photon/electron recoil or angular momentum transfer.
    """
    if isinstance(level, (bool, np.bool_)) or level not in (0, 1):
        raise ValueError("Only photoionization levels 0 and 1 are implemented")
    q, p, mass = validate_phase_point(q, p, mass)
    vn = finite_scalar(neutral_potential, "neutral_potential")
    vi = finite_scalar(ionic_potential, "ionic_potential")
    offset = finite_scalar(ion_energy_offset, "ion_energy_offset")
    photon = finite_scalar(photon_energy, "photon_energy", nonnegative=True)
    tolerance = finite_scalar(energy_tolerance, "energy_tolerance", nonnegative=True)
    if photon == 0 or tolerance == 0:
        raise ValueError("photon_energy and energy_tolerance must be positive")
    wmass = np.repeat(mass, 3)
    kinetic = float(0.5 * np.sum(p * p / wmass) * HARTREE_TO_EV)
    gap = (vi - vn) * HARTREE_TO_EV + offset
    if not np.isfinite(kinetic) or not np.isfinite(gap):
        raise ValueError("Non-finite kinetic energy or vertical ionization energy")
    rng = sampling_generator(seed) if rng is None else rng

    if level == 0:
        if electron_energy is not None or ionic_energy is not None or constraints is not None:
            raise ValueError("Level 0 keeps momenta unchanged; use Level 1 for experimental energy constraints")
        if gap < -tolerance or gap > photon + tolerance:
            raise ValueError("Level 0 vertical ionization energy is outside the photon energy window")
        binding = float(np.clip(gap, 0.0, photon))
        electron = photon - binding
        scale, weight = 1.0, 1.0
        ion_p = p.copy()
    else:
        if constraints is None:
            constraints = EnergyConstraints(photon, electron_energy, ionic_energy, tolerance)
        binding, electron, weight = constraints.sample(rng, minimum_binding=gap - kinetic)
        target = kinetic + binding - gap
        if target < -tolerance:
            raise ValueError("Experimental energies require negative ionic kinetic energy")
        target = max(0.0, target)
        if kinetic == 0:
            if target > tolerance:
                raise ValueError("Cannot rescale zero atomic momenta to positive kinetic energy; provide a moving neutral sample")
            scale = 1.0
        else:
            scale = float(np.sqrt(target / kinetic))
        ion_p = scale * p

    ionic_kinetic = float(0.5 * np.sum(ion_p * ion_p / wmass) * HARTREE_TO_EV)
    residual = (ionic_kinetic - kinetic) + gap + electron - photon
    if not np.isfinite(scale) or not np.isfinite(residual) or abs(residual) > tolerance:
        raise ValueError(f"Ionic preparation failed energy balance: residual={residual} eV")
    xyz = q.reshape(-1, 3)
    centered = xyz - np.average(xyz, axis=0, weights=mass)
    neutral_p3, ion_p3 = p.reshape(-1, 3), ion_p.reshape(-1, 3)
    return IonicInitialState(
        q=q, p=ion_p, neutral_p=p, level=int(level), source_index=int(source_index),
        photon_energy_ev=photon, electron_energy_ev=electron, ionic_energy_ev=binding,
        neutral_potential_hartree=vn, ionic_potential_hartree=vi,
        ion_energy_offset_ev=offset, neutral_kinetic_ev=kinetic,
        ionic_kinetic_ev=ionic_kinetic, momentum_scale=scale,
        energy_residual_ev=float(residual), selection_weight=float(weight),
        neutral_linear_momentum=neutral_p3.sum(axis=0),
        ionic_linear_momentum=ion_p3.sum(axis=0),
        neutral_angular_momentum=np.cross(centered, neutral_p3).sum(axis=0),
        ionic_angular_momentum=np.cross(centered, ion_p3).sum(axis=0),
    )
