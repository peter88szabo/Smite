r"""Photon and photoelectron momentum, and how the recoil is shared among atoms.

THE MODEL
=========

Single photoionization is treated as an instantaneous momentum exchange at a
frozen nuclear geometry. A photon of energy :math:`h\nu` is absorbed, an
electron of kinetic energy :math:`\varepsilon` leaves, and the ion must take up
whatever momentum is left over:

.. math::
    \Delta\mathbf{p} \;=\; \mathbf{p}_\gamma \;-\; \mathbf{p}_e .

In atomic units (:math:`m_e = 1`, :math:`c = 137.036`) the two momenta are

.. math::
    \mathbf{p}_e      &= \sqrt{2 m_e \varepsilon}\;\hat{\mathbf{n}}
                       = \sqrt{2\varepsilon}\;\hat{\mathbf{n}},  \\
    \mathbf{p}_\gamma &= \frac{h\nu}{c}\,\hat{\mathbf{k}} ,

with :math:`\hat{\mathbf{n}}` the photoelectron emission direction and
:math:`\hat{\mathbf{k}}` the beam direction.

The photon term is not negligible everywhere. Its size relative to the electron
momentum is :math:`p_\gamma/p_e = h\nu / (c\sqrt{2\varepsilon})`, which is about
0.7% at 21 eV but 4-6% at a few keV, so it is always included rather than
switched on by a threshold.

WHERE THE RECOIL GOES
=====================

The momentum :math:`\Delta\mathbf{p}` is shared among the atoms with weights
:math:`w_i \ge 0`, :math:`\sum_i w_i = 1`:

.. math::
    \mathbf{p}_i \;\longrightarrow\; \mathbf{p}_i + w_i\,\Delta\mathbf{p} .

Total linear momentum then changes by exactly :math:`\Delta\mathbf{p}` for *any*
weights, which is the point. Angular momentum about the centre of mass changes by

.. math::
    \Delta\mathbf{L} \;=\; \sum_i \mathbf{x}_i \times w_i \Delta\mathbf{p}
                     \;=\; \Big(\sum_i w_i \mathbf{x}_i\Big) \times \Delta\mathbf{p} ,

that is, by the displacement of the *recoil centroid* from the centre of mass,
crossed into the recoil. That one expression contains the whole physics of the
three supported modes:

===============  ==========================  ==========================================
``recoil_site``  :math:`w_i`                 :math:`\Delta\mathbf{L}`
===============  ==========================  ==========================================
``"com"``        :math:`m_i / M`             :math:`\mathbf{0}` exactly, because
                                             :math:`\sum_i m_i \mathbf{x}_i = \mathbf{0}`
                                             about the centre of mass
atom index j     :math:`\delta_{ij}`         :math:`\mathbf{x}_j \times \Delta\mathbf{p}`
weight array     user supplied               :math:`(\sum_i w_i\mathbf{x}_i)\times\Delta\mathbf{p}`
===============  ==========================  ==========================================

Physically: ``"com"`` is the delocalised outer-valence limit, where the recoil
becomes pure centre-of-mass translation and excites nothing internally. A single
atom is the localised core-hole limit, where the recoil is applied off the centre
of mass, torques the molecule and excites vibration. An explicit weight array is
the general case -- the physically correct weights for a partially localised hole
are the atomic populations of the Dyson orbital, which the caller supplies.

**That asymmetry is the recoil effect.** It is not a numerical artefact and must
survive the energy rebalancing done in :mod:`photoionization.preparation`.

HOW BIG IS IT
=============

In the delocalised limit the recoil energy has the closed form

.. math::
    E_{\text{rec}} \;=\; \frac{|\mathbf{p}_e|^2}{2M}
                   \;=\; \frac{m_e}{M}\,\varepsilon ,

so for H\ :sub:`2`\ O it runs 0.3 meV at 21 eV, 2.4 meV at 100 eV, 21 meV at
1 keV and 82 meV at 3 keV. Below roughly 100 eV this is far under a vibrational
quantum and Level 3 buys nothing over Level 2; at keV energies it is the size of
a soft mode and is an established, measured photoelectron peak shift.

WHAT THIS MODULE DOES NOT MODEL
===============================

* The photoelectron angular distribution. :math:`\hat{\mathbf{n}}` is sampled
  isotropically. The dipole form
  :math:`\mathrm{d}\sigma/\mathrm{d}\Omega \propto 1 + \beta P_2(\cos\theta)`
  (Cooper and Zare, *J. Chem. Phys.* **48**, 942, 1968) is not implemented; the
  ``beta`` and ``polarization`` arguments exist so that adding it later is not an
  interface change. Isotropic sampling is correct for an angle-integrated
  measurement and averages out over an orientationally random ensemble.
* Electron partial waves, photon spin and post-collision interaction.
* Shake-up and shake-off channels. Single ionization only.
"""

from numbers import Integral

import numpy as np

from utils.constants import AU_SPEED_OF_LIGHT, HARTREE_TO_EV


def sample_electron_momentum(rng, electron_energy_ev, *, beta=None, polarization=None):
    r"""Draw one photoelectron momentum vector, in atomic units.

    The magnitude is fixed by the kinetic energy,

    .. math::
        |\mathbf{p}_e| = \sqrt{2 m_e \varepsilon} = \sqrt{2\varepsilon}

    with :math:`\varepsilon` converted from eV to Hartree and :math:`m_e = 1`.

    The direction is sampled isotropically, by taking :math:`\cos\theta` uniform
    on :math:`[-1, 1]` and :math:`\phi` uniform on :math:`[0, 2\pi)`. Sampling
    :math:`\theta` uniformly instead would bunch the directions at the poles.

    ``beta`` and ``polarization`` are accepted and rejected. They are present so
    that implementing the dipole angular distribution later adds behaviour
    without changing any call site.
    """
    if beta is not None or polarization is not None:
        raise ValueError(
            "Anisotropic photoelectron emission is not implemented; "
            "beta/polarization are reserved for the dipole distribution")
    energy = float(electron_energy_ev)
    if not np.isfinite(energy) or energy < 0:
        raise ValueError("Photoelectron kinetic energy must be finite and nonnegative")

    magnitude = np.sqrt(2.0 * energy / HARTREE_TO_EV)
    cos_theta = rng.uniform(-1.0, 1.0)
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    phi = rng.uniform(0.0, 2.0 * np.pi)
    direction = np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])
    return magnitude * direction


def photon_momentum(photon_energy_ev, direction):
    r"""Momentum carried in by the photon, in atomic units.

    .. math::
        \mathbf{p}_\gamma = \frac{h\nu}{c}\,\hat{\mathbf{k}}

    with :math:`h\nu` converted from eV to Hartree and :math:`c = 137.036` a.u.
    ``direction`` need not be normalised; only its direction is used.
    """
    energy = float(photon_energy_ev)
    if not np.isfinite(energy) or energy <= 0:
        raise ValueError("photon_energy must be finite and positive")
    k = np.asarray(direction, dtype=float).reshape(-1)
    if k.size != 3 or not np.all(np.isfinite(k)):
        raise ValueError("photon_direction must be three finite Cartesian components")
    norm = np.linalg.norm(k)
    if norm == 0:
        raise ValueError("photon_direction must have nonzero length")
    return (energy / HARTREE_TO_EV) / AU_SPEED_OF_LIGHT * (k / norm)


def recoil_weights(recoil_site, mass):
    r"""Resolve ``recoil_site`` into per-atom weights :math:`w_i`.

    Returns a vector with :math:`w_i \ge 0` and :math:`\sum_i w_i = 1`, so that
    applying :math:`\mathbf{p}_i \to \mathbf{p}_i + w_i\Delta\mathbf{p}` changes
    the total linear momentum by exactly :math:`\Delta\mathbf{p}`.

    Accepted forms:

    ``"com"``
        Mass weights :math:`w_i = m_i/M`. The recoil centroid coincides with the
        centre of mass, so :math:`\Delta\mathbf{L} = \mathbf{0}` identically and
        the recoil is pure translation. This is the delocalised valence limit and
        the default, because it is the choice that excites nothing it should not.
    an integer atom index ``j``
        :math:`w_i = \delta_{ij}`. The localised core-hole limit.
    a sequence of ``natom`` weights
        Used as given after normalisation, for example Dyson-orbital atomic
        populations.
    """
    mass = np.asarray(mass, dtype=float).reshape(-1)
    natom = mass.size

    if isinstance(recoil_site, str):
        if recoil_site.lower() != "com":
            raise ValueError(f"Unknown recoil_site {recoil_site!r}; use 'com', an atom index, or weights")
        return mass / mass.sum()

    # bool is a subclass of int, so it is excluded explicitly rather than
    # silently taken as atom 0 or 1.
    if isinstance(recoil_site, Integral) and not isinstance(recoil_site, (bool, np.bool_)):
        index = int(recoil_site)
        if not 0 <= index < natom:
            raise ValueError(f"recoil_site atom index {index} is outside 0..{natom - 1}")
        weights = np.zeros(natom)
        weights[index] = 1.0
        return weights

    weights = np.asarray(recoil_site, dtype=float).reshape(-1)
    if weights.size != natom or not np.all(np.isfinite(weights)):
        raise ValueError(f"recoil_site weights must be {natom} finite values, one per atom")
    if np.any(weights < 0):
        raise ValueError("recoil_site weights must be nonnegative")
    total = weights.sum()
    if total <= 0:
        raise ValueError("recoil_site weights must sum to a positive value")
    return weights / total


def distribute_recoil(delta_p, weights):
    r"""Spread :math:`\Delta\mathbf{p}` over the atoms as a flat 3N vector.

    .. math::
        \big(\mathbf{p}_{\text{rec}}\big)_i = w_i\,\Delta\mathbf{p}
    """
    delta_p = np.asarray(delta_p, dtype=float).reshape(3)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    return np.outer(weights, delta_p).reshape(-1)
