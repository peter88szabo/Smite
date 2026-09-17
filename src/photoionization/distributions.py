"""Nonnegative spline densities and energy-constrained Monte Carlo sampling."""

from numbers import Real

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq


def finite_scalar(value, name, *, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not np.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return value


class EnergyDistribution:
    """Normalized PCHIP cubic spline through (energy_eV, relative_density).

    Supply two one-dimensional arrays, or use ``from_input`` with a tuple
    ``(energies, densities)`` or a two-column table of (energy, density) rows.
    Densities are per unit energy, not probabilities assigned to grid points.
    No extrapolation is performed outside the supplied energy grid.
    """

    def __init__(self, energies, densities):
        x = np.array(energies, dtype=float, copy=True)
        y = np.array(densities, dtype=float, copy=True)
        if x.ndim != 1 or x.size < 2 or y.shape != x.shape:
            raise ValueError("An energy distribution needs matching 1D grids with at least two points")
        if not np.all(np.isfinite(x)) or not np.all(np.diff(x) > 0) or np.any(x < 0):
            raise ValueError("Energy grid must be finite, nonnegative and strictly increasing")
        if not np.all(np.isfinite(y)) or np.any(y < 0) or not np.any(y > 0):
            raise ValueError("Distribution densities must be finite, nonnegative and not all zero")
        y /= np.max(y)
        spline = PchipInterpolator(x, y, extrapolate=False)
        area = float(spline.integrate(x[0], x[-1]))
        if not np.isfinite(area) or area <= 0:
            raise ValueError("Energy distribution must have a finite positive integral")
        self.energies = x
        self.densities = y / area
        self._spline = PchipInterpolator(x, self.densities, extrapolate=False)

    @classmethod
    def from_input(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, tuple) and len(value) == 2:
            return cls(*value)
        table = np.asarray(value, dtype=float)
        if table.ndim != 2 or table.shape[1] != 2:
            raise ValueError("Use (energy_grid, density_grid) or a two-column energy/density table")
        return cls(table[:, 0], table[:, 1])

    def pdf(self, energy):
        values = self._spline(energy)
        return np.maximum(0.0, np.nan_to_num(values, nan=0.0))

    def sample(self, rng, size=1):
        sampler = _SplineSampler(self.energies, self.pdf)
        return np.array([sampler.draw(rng) for _ in range(size)])


class _SplineSampler:
    """Integrate/invert a piecewise polynomial density without a grid bias.

    Four Gauss nodes integrate a product of two cubic splines exactly on
    each interval. Inverting these integrals also handles zero-density gaps.
    """

    _nodes, _weights = np.polynomial.legendre.leggauss(4)

    def __init__(self, knots, density):
        self.knots = np.unique(knots)
        self.density = density
        masses = self.integral(self.knots[:-1], self.knots[1:])
        self.cumulative = np.concatenate(([0.0], np.cumsum(masses)))
        self.normalization = float(self.cumulative[-1])
        if not np.isfinite(self.normalization) or self.normalization <= 0:
            raise ValueError("Experimental energies have no compatible nonzero probability support")

    def integral(self, left, right):
        width = np.asarray(right) - np.asarray(left)
        points = np.asarray(left)[..., None] + width[..., None] * (self._nodes + 1) / 2
        return width / 2 * np.sum(self.density(points) * self._weights, axis=-1)

    def draw(self, rng):
        target = rng.random() * self.normalization
        index = min(np.searchsorted(self.cumulative, target, side="right") - 1,
                    len(self.knots) - 2)
        left, right = self.knots[index:index + 2]
        fraction = (target - self.cumulative[index]) / (
            self.cumulative[index + 1] - self.cumulative[index])
        if fraction <= 0:
            return float(left)
        # Solve on [0,1] to retain accuracy for very narrow energy windows.
        mass = self.cumulative[index + 1] - self.cumulative[index]
        root = brentq(lambda t: self.integral(left, left + t * (right - left)) / mass - fraction,
                      0.0, 1.0, xtol=1e-13)
        return float(left + root * (right - left))


def _energy_input(value, name):
    if value is None:
        return None
    if np.isscalar(value):
        return finite_scalar(value, name, nonnegative=True)
    try:
        return EnergyDistribution.from_input(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid {name}: {exc}") from exc


class EnergyConstraints:
    """Binding energy is photon energy minus electron kinetic energy (eV).

    With two densities, their product is conditioned on this balance and
    the accessible ionic kinetic energy. Constants act as exact constraints.
    """

    def __init__(self, photon_energy, electron_energy=None, ionic_energy=None,
                 tolerance=1e-8):
        self.photon = finite_scalar(photon_energy, "photon_energy", nonnegative=True)
        if self.photon == 0:
            raise ValueError("photon_energy must be positive")
        self.tolerance = finite_scalar(tolerance, "energy_tolerance", nonnegative=True)
        if self.tolerance == 0:
            raise ValueError("energy_tolerance must be positive")
        self.electron = _energy_input(electron_energy, "electron_energy")
        self.ionic = _energy_input(ionic_energy, "ionic_energy")
        if self.electron is None and self.ionic is None:
            raise ValueError("Level 1 requires electron_energy or ionic_energy (or both)")

    def sample(self, rng, *, minimum_binding=0.0):
        low, high = max(0.0, minimum_binding), self.photon
        fixed = None
        if isinstance(self.ionic, float):
            fixed = self.ionic
        if isinstance(self.electron, float):
            inferred = self.photon - self.electron
            if fixed is not None and abs(fixed - inferred) > self.tolerance:
                raise ValueError("Constant ionic_energy + electron_energy must equal photon_energy")
            fixed = inferred if fixed is None else fixed

        def density(binding):
            weight = np.ones_like(binding, dtype=float)
            if isinstance(self.ionic, EnergyDistribution):
                weight *= self.ionic.pdf(binding)
            if isinstance(self.electron, EnergyDistribution):
                weight *= self.electron.pdf(self.photon - binding)
            return weight

        if fixed is not None:
            if fixed < low - self.tolerance or fixed > high + self.tolerance:
                raise ValueError("Experimental energies require negative electron or ionic kinetic energy")
            fixed = float(np.clip(fixed, low, high))
            weight = float(density(fixed))
            if weight <= 0:
                raise ValueError("Constant energy lies outside the other distribution's nonzero support")
            return fixed, self.photon - fixed, weight

        knots = [low, high]
        if isinstance(self.ionic, EnergyDistribution):
            grid = self.ionic.energies
            low, high = max(low, grid[0]), min(high, grid[-1])
            knots.extend(grid)
        if isinstance(self.electron, EnergyDistribution):
            grid = self.photon - self.electron.energies
            low, high = max(low, grid[-1]), min(high, grid[0])
            knots.extend(grid)
        if high <= low:
            raise ValueError("Experimental energies have no accessible overlapping interval")
        knots = np.asarray(knots + [low, high])
        knots = knots[(knots >= low) & (knots <= high)]
        sampler = _SplineSampler(knots, density)
        binding = sampler.draw(rng)
        return binding, self.photon - binding, sampler.normalization
