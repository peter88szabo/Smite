r"""Franck-Condon transfer to one ionic PES; all nuclear arrays use atomic units.

THE MODEL
=========

Ionization is treated as instantaneous at a frozen nuclear geometry: the sampled
neutral geometry :math:`\mathbf{q}` is carried over to the ionic surface
unchanged, at every level. Only the momenta can differ, and *how* they differ is
what the four levels select.

Energy balance
--------------

All levels satisfy the same statement, that the photon energy is shared between
the departing electron, the change in electronic potential energy, and the change
in nuclear kinetic energy:

.. math::
    h\nu + T_n + V_n \;=\; \varepsilon + T_i + V_i
    \qquad\Longrightarrow\qquad
    T_i \;=\; T_n + h\nu - \varepsilon - \Delta V ,

where :math:`\Delta V = (V_i - V_n) + \delta_{\text{offset}}` is the vertical gap
at the frozen geometry plus the optional additive ionic PES offset, and
:math:`\varepsilon` is the photoelectron kinetic energy. This relation is
*imposed* during construction and then *verified* independently at the end, to
``energy_tolerance``.

The four levels
---------------

======  =====================================================  ==================  ======
Level   Momenta                                                Conserves P, L      Recoil
======  =====================================================  ==================  ======
0       :math:`\mathbf{p}` unchanged                           yes, trivially      no
1       :math:`s\,\mathbf{p}`, all :math:`3N` components        **no**              no
2       Eckart split; scale the internal part only             **yes, exactly**    no
3       Level 2 plus :math:`\Delta\mathbf{p}`                   by the right amount yes
======  =====================================================  ==================  ======

Level 1 closes the balance with a single factor on every Cartesian component.
That factor multiplies the centre-of-mass translation and the rigid rotation
along with the vibrations, so it changes :math:`\mathbf{P}` and
:math:`\mathbf{L}` even though its own premise is that ionization transfers no
momentum. Levels 2 and 3 exist to fix that, and they are retained here unchanged
so that existing results stay reproducible.

Why the partition is exact (Levels 2 and 3)
-------------------------------------------

Work in mass-weighted momenta

.. math::
    \mathbf{a} = \mathsf{M}^{-1/2}\mathbf{p},
    \qquad a_{3i+\alpha} = \frac{p_{3i+\alpha}}{\sqrt{m_i}} ,

in which the kinetic energy is a plain Euclidean norm,

.. math::
    T = \frac{1}{2}\sum_{i\alpha}\frac{p_{3i+\alpha}^2}{m_i}
      = \frac{1}{2}\,\lvert\mathbf{a}\rvert^2 .

This is the metric that makes everything below exact rather than approximate.
:func:`normalmode.eckart.get_eckart_projector` returns
:math:`\mathsf{R} = \mathsf{B}\mathsf{B}^{+}`, the projector onto the
mass-weighted generators of translation and rotation about the centre of mass.
Because :math:`\mathsf{R}` is an orthogonal projector **in this same metric**,

.. math::
    \mathbf{a}^{\text{ext}} = \mathsf{R}\mathbf{a}, \qquad
    \mathbf{a}^{\text{int}} = (\mathsf{1}-\mathsf{R})\mathbf{a}, \qquad
    \mathbf{a}^{\text{ext}}\!\cdot\mathbf{a}^{\text{int}} = 0 ,

so the kinetic energy splits with **no cross term**,
:math:`T = T^{\text{ext}} + T^{\text{int}}`, and the internal part carries
neither linear nor angular momentum:

.. math::
    \sum_i \sqrt{m_i}\,\mathbf{a}^{\text{int}}_i = \mathbf{0},
    \qquad
    \sum_i \mathbf{x}_i \times \sqrt{m_i}\,\mathbf{a}^{\text{int}}_i = \mathbf{0}.

Scaling :math:`\mathbf{a}^{\text{int}}` therefore changes the vibrational energy
and leaves :math:`\mathbf{P}` and :math:`\mathbf{L}` bit-for-bit unchanged. Doing
the same projection in unweighted Cartesian coordinates would *not* be orthogonal
and would leak energy across the split.

Levels 2 and 3 require experimental energy constraints, exactly as Level 1 does.
Without them the sampled binding energy equals the vertical gap, giving
:math:`T_i = T_n` and :math:`s = 1`, i.e. Level 0.

What no level models
--------------------

Neither the photon nor the electron angular distribution beyond isotropic
emission (see :mod:`photoionization.recoil`), nor electron partial waves, photon
spin, post-collision interaction, or shake-up channels. Levels 0, 1 and 2 model
no recoil at all. A self-consistent photoelectron energy -- solving
:math:`\varepsilon = h\nu - \Delta V - \Delta T(\varepsilon)` so as to *predict*
the recoil peak shift rather than consume a measured one -- is deliberately not
implemented; see the design note in ``report/``.
"""

from dataclasses import dataclass, asdict, field

import numpy as np

from normalmode.eckart import get_eckart_projector
from photoionization import recoil as recoil_module
from photoionization.distributions import EnergyConstraints, finite_scalar
from sampling.random_seed import sampling_generator
from utils.constants import HARTREE_TO_EV

# Below this, a mass-weighted internal momentum is treated as absent rather than
# merely small: there is nothing to rescale and the quadratic degenerates.
MINIMUM_INTERNAL_NORM = 1.0e-20


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
    # ---- appended for Levels 2 and 3; defaults keep older call sites working ----
    # ``momentum_scale`` above is redefined rather than duplicated: it carries the
    # factor applied to whichever component the level rescales -- 1.0 at Level 0,
    # the global 3N factor at Level 1, and the neutral-internal factor s at Levels
    # 2 and 3. ``scaled_subspace`` says which, so the record is self-describing and
    # no consumer has to infer it from ``level``. A NaN sentinel is not usable here
    # because driver.py dumps this record with json.dump(..., allow_nan=False).
    scaled_subspace: str = "all"
    external_kinetic_ev: float | None = None
    internal_kinetic_ev: float | None = None
    recoil_momentum: np.ndarray | None = None
    electron_momentum: np.ndarray | None = None
    photon_momentum: np.ndarray | None = None
    recoil_weights: np.ndarray | None = None
    recoil_kinetic_ev: float | None = None
    recoil_internal_kinetic_ev: float | None = None

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


def eckart_split(q, p, mass):
    r"""Split a momentum into external and internal parts, in mass-weighted form.

    Returns :math:`(\mathbf{a}^{\text{ext}}, \mathbf{a}^{\text{int}})` where

    .. math::
        \mathbf{a} = \mathsf{M}^{-1/2}\mathbf{p}, \qquad
        \mathbf{a}^{\text{ext}} = \mathsf{R}\,\mathbf{a}, \qquad
        \mathbf{a}^{\text{int}} = (\mathsf{1}-\mathsf{R})\,\mathbf{a} .

    The split is orthogonal in the metric that measures kinetic energy, so

    .. math::
        T = \tfrac12\lvert\mathbf{a}\rvert^2
          = \tfrac12\lvert\mathbf{a}^{\text{ext}}\rvert^2
          + \tfrac12\lvert\mathbf{a}^{\text{int}}\rvert^2

    with no cross term, and :math:`\mathbf{a}^{\text{int}}` carries zero linear and
    zero angular momentum. Energies from these vectors are in Hartree.

    ``get_eckart_projector`` centres the geometry itself and uses a pseudoinverse,
    so :math:`\operatorname{rank}\mathsf{R}` comes out as 6 for a nonlinear
    molecule, 5 for a linear one and 3 for a single atom with no special-casing
    here.
    """
    projector = get_eckart_projector(mass, q)
    a = p / np.sqrt(np.repeat(mass, 3))
    a_external = projector @ a
    return a_external, a - a_external


def solve_internal_scale(a_neutral_int, a_recoil_int, t_target, tolerance):
    r"""Scale the neutral internal momentum to hit a target internal energy.

    The ionic internal (vibrational) momentum is built as the recoil
    contribution, which is fixed by momentum conservation and must **not** be
    touched, plus a scaled copy of the neutral internal momentum, which is the
    only free quantity left once the energy balance is imposed:

    .. math::
        \mathbf{a}^{\text{int}}
            = s\,\mathbf{a}^{\text{int}}_{n} + \mathbf{a}^{\text{int}}_{r} .

    Keeping :math:`\mathbf{a}^{\text{int}}_r` out of the scaling is the whole
    point: rescaling the combined internal part would scrub away the
    recoil-induced vibrational excitation that Level 3 exists to capture.

    Requiring :math:`T^{\text{int}} = \tfrac12|\mathbf{a}^{\text{int}}|^2 =
    T^{\text{int}}_{\text{target}}` gives a quadratic in :math:`s`,

    .. math::
        A s^2 + 2 B s + \big(C - 2T^{\text{int}}_{\text{target}}\big) = 0,

    with :math:`A = |\mathbf{a}^{\text{int}}_n|^2`,
    :math:`B = \mathbf{a}^{\text{int}}_n\cdot\mathbf{a}^{\text{int}}_r`,
    :math:`C = |\mathbf{a}^{\text{int}}_r|^2`, solved by

    .. math::
        s = \frac{-B + \sqrt{D}}{A},
        \qquad D = B^2 - A\big(C - 2T^{\text{int}}_{\text{target}}\big).

    The :math:`+` root is taken always: it is the solution that retains the most
    of the neutral internal motion.

    At Level 2 the recoil is absent, :math:`B = C = 0`, and this collapses to
    :math:`s = \sqrt{T^{\text{int}}_{\text{target}}/T^{\text{int}}_n}` -- Level 1's
    factor, but confined to the vibrational subspace.

    Sign of ``s``
    -------------
    For :math:`C < 2T^{\text{int}}_{\text{target}}` the :math:`+` root is positive
    and the sense of the neutral vibrational motion is preserved. When the recoil
    alone overshoots the internal budget the root can come out negative, meaning
    the neutral internal motion has to oppose the recoil to bring the energy down.
    That is permitted rather than rejected: :math:`D \ge 0` means the balance is
    satisfiable, and the sign of an internal momentum has no absolute meaning in a
    Franck-Condon ensemble, since :math:`+\mathbf{a}^{\text{int}}_n` and
    :math:`-\mathbf{a}^{\text{int}}_n` are equally represented in the neutral
    sample. Flipping it introduces no bias. The value is returned so the regime is
    visible in the launch record.

    Raises
    ------
    ValueError
        If there is no internal motion to scale and the target cannot be met by
        the recoil alone; or if :math:`D < 0`, meaning the sampled experimental
        energies leave less internal energy than the recoil by itself deposits.
        Both are physical rejections, not numerical ones: no choice of ``s``
        satisfies them.
    """
    a = float(a_neutral_int @ a_neutral_int)
    b = float(a_neutral_int @ a_recoil_int)
    c = float(a_recoil_int @ a_recoil_int)

    if t_target < -tolerance:
        raise ValueError(
            f"External motion alone exceeds the ionic energy budget by {-t_target * HARTREE_TO_EV:.6g} eV; "
            "the recoil carries more energy than the photon can pay for at this geometry")

    if a <= MINIMUM_INTERNAL_NORM:
        # Nothing to scale: a single atom, or a neutral sample with no vibration.
        # The balance can only be met if the recoil already lands on the target.
        if abs(0.5 * c - t_target) > tolerance:
            raise ValueError(
                "No internal (vibrational) motion available to rescale. A monatomic ion has no "
                "internal subspace, and a neutral sample with none cannot have it manufactured; "
                "use level 0 or 1, or supply a vibrating neutral sample")
        return 1.0

    # By Cauchy-Schwarz B^2 <= A C, so D <= 2 A t_target: a negative target always
    # fails here too. The explicit check above exists only to name the cause.
    discriminant = b * b - a * (c - 2.0 * t_target)
    if discriminant < 0.0:
        raise ValueError(
            "Sampled experimental energies leave less internal energy than the recoil alone "
            f"deposits ({0.5 * c * HARTREE_TO_EV:.6g} eV internal recoil against a "
            f"{t_target * HARTREE_TO_EV:.6g} eV internal budget); no rescaling can satisfy the "
            "energy balance at this geometry")

    return float((-b + np.sqrt(discriminant)) / a)


def prepare_ionic_state(q, p, mass, neutral_potential, ionic_potential, *,
                        photon_energy, level=0, electron_energy=None, ionic_energy=None,
                        ion_energy_offset=0.0, energy_tolerance=1e-8, seed=None,
                        rng=None, source_index=0, constraints=None,
                        recoil_site="com", photon_direction=(0.0, 0.0, 1.0)):
    r"""Prepare one Cartesian launch without changing the Franck-Condon geometry.

    Potentials are in Hartree; photon/electron/ionic energies and the optional
    additive ionic PES offset are in eV. ``ionic_energy`` is the measured binding
    energy, not ionic kinetic energy or energy above an ionic minimum.

    The geometry is never modified at any level. What differs is the momentum:

    * **Level 0** copies :math:`\mathbf{p}` unchanged.
    * **Level 1** scales every one of the :math:`3N` components by one factor
      :math:`s = \sqrt{T_i/T_n}`. This changes :math:`\mathbf{P}` and
      :math:`\mathbf{L}`, which its own no-momentum-transfer premise forbids.
    * **Level 2** splits the momentum with the Eckart projector and scales only
      the internal (vibrational) part, leaving :math:`\mathbf{P}` and
      :math:`\mathbf{L}` exactly unchanged.
    * **Level 3** additionally gives the ion the momentum the photon brought in
      and the electron took away,
      :math:`\Delta\mathbf{p} = \mathbf{p}_\gamma - \mathbf{p}_e`, shared over the
      atoms by ``recoil_site``. :math:`\mathbf{P}` then changes by exactly
      :math:`\Delta\mathbf{p}` and :math:`\mathbf{L}` by
      :math:`(\sum_i w_i\mathbf{x}_i)\times\Delta\mathbf{p}`.

    ``recoil_site`` and ``photon_direction`` apply to Level 3 only. Levels 1, 2
    and 3 all require experimental energy constraints; Level 0 forbids them.
    """
    if isinstance(level, (bool, np.bool_)) or level not in (0, 1, 2, 3):
        raise ValueError("Only photoionization levels 0, 1, 2 and 3 are implemented")
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

    subspace = "all"
    external_kinetic = internal_kinetic = None
    recoil_p = electron_p = photon_p = weights = None
    recoil_kinetic = recoil_internal_kinetic = None

    if level == 0:
        if electron_energy is not None or ionic_energy is not None or constraints is not None:
            raise ValueError("Level 0 keeps momenta unchanged; use Level 1, 2 or 3 for experimental energy constraints")
        if gap < -tolerance or gap > photon + tolerance:
            raise ValueError("Level 0 vertical ionization energy is outside the photon energy window")
        binding = float(np.clip(gap, 0.0, photon))
        electron = photon - binding
        scale, weight = 1.0, 1.0
        ion_p = p.copy()
        subspace = "none"
    elif level == 1:
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
    else:
        # ---- Levels 2 and 3 -------------------------------------------------
        # Same energy balance as Level 1,
        #
        #     T_i = T_n + hv - eps - dV ,
        #
        # but closed in the vibrational subspace instead of by scaling everything.
        if constraints is None:
            constraints = EnergyConstraints(photon, electron_energy, ionic_energy, tolerance)
        binding, electron, weight = constraints.sample(rng, minimum_binding=gap - kinetic)
        t_ion_ev = kinetic + binding - gap
        if t_ion_ev < -tolerance:
            raise ValueError("Experimental energies require negative ionic kinetic energy")
        t_ion_ev = max(0.0, t_ion_ev)

        if level == 3:
            # Momentum the ion must absorb: photon in, electron out.
            #
            #     dp = p_gamma - p_e ,   p_e = sqrt(2 eps) n ,   p_gamma = (hv/c) k
            #
            # spread over the atoms as p_i -> p_i + w_i dp, so that the total
            # linear momentum changes by exactly dp for any weights.
            electron_p = recoil_module.sample_electron_momentum(rng, electron)
            photon_p = recoil_module.photon_momentum(photon, photon_direction)
            weights = recoil_module.recoil_weights(recoil_site, mass)
            recoil_p = recoil_module.distribute_recoil(photon_p - electron_p, weights)
        else:
            recoil_p = np.zeros_like(p)

        # Mass-weighted split. a_ext is FIXED from here on -- it carries all of P
        # and L, both the neutral's and whatever the recoil added -- so leaving it
        # alone is exactly what makes Levels 2 and 3 momentum-correct.
        a_neutral_ext, a_neutral_int = eckart_split(q, p, mass)
        a_recoil_ext, a_recoil_int = eckart_split(q, recoil_p, mass)
        a_external = a_neutral_ext + a_recoil_ext

        t_external = 0.5 * float(a_external @ a_external)
        t_target = t_ion_ev / HARTREE_TO_EV - t_external
        scale = solve_internal_scale(a_neutral_int, a_recoil_int, t_target,
                                     tolerance / HARTREE_TO_EV)

        a_ion = a_external + scale * a_neutral_int + a_recoil_int
        ion_p = a_ion * np.sqrt(wmass)

        subspace = "internal"
        external_kinetic = t_external * HARTREE_TO_EV
        internal_kinetic = 0.5 * float((a_ion - a_external) @ (a_ion - a_external)) * HARTREE_TO_EV
        if level == 3:
            recoil_kinetic = float(0.5 * np.sum(recoil_p * recoil_p / wmass) * HARTREE_TO_EV)
            recoil_internal_kinetic = 0.5 * float(a_recoil_int @ a_recoil_int) * HARTREE_TO_EV

    ionic_kinetic = float(0.5 * np.sum(ion_p * ion_p / wmass) * HARTREE_TO_EV)
    # Independent verification of the balance the construction imposed. At Levels
    # 2 and 3 this is a genuine check rather than a restatement, because the ionic
    # kinetic energy there is assembled from projected pieces.
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
        scaled_subspace=subspace,
        external_kinetic_ev=external_kinetic, internal_kinetic_ev=internal_kinetic,
        recoil_momentum=None if recoil_p is None or level != 3 else recoil_p,
        electron_momentum=electron_p, photon_momentum=photon_p,
        recoil_weights=weights, recoil_kinetic_ev=recoil_kinetic,
        recoil_internal_kinetic_ev=recoil_internal_kinetic,
    )
