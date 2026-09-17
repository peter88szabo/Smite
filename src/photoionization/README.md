# Photoionization: Levels 0 to 3

Pass the ionic interface, neutral structure, Hessian or MD trajectory, and
experimental energies to `Photoionization(...)`. For a neutral Hessian, configure
its QCT modes with `photoion.Specify_Mode_Sampling(...)`. The separate
`sample_and_run_dynamics(...)` call contains propagation and stopping settings.
Each object propagates **one chosen ionic electronic state**; each fresh dynamics
call samples one new ionic launch. A [complete input script](../../examples/example_photoionization.py)
includes molecule setup, interfaces, neutral mode sampling, channel definitions and dynamics.

```python
from smite import Photoionization

# molecule already holds the neutral atoms, masses and ground-state qchem setup.
# qcinput_ion selects the ionic PES, including its charge and multiplicity.
photoion = Photoionization(
    molecule,
    qchem=qcinput_ion,
    neutra_frankcondon_geom="neutral.xyz",
    neutral_traj="neutral_md.xyz",
    md_start=100,
    md_stride=5,
    photon_energy=15.0,             # eV
    level=1,
    electron_energy=4.0,            # constant or distribution
    ionic_energy=11.0,              # constant or distribution; binding energy
    sampling_seed=1234,
)

pairs_to_test = {
    "channel_1": [((0, 1), "GT", 3.0)],
    "channel_2": [((1, 2), "GT", 3.0), ((0, 1), "LT", 2.0)],
}

photoion.sample_and_run_dynamics(
    integrator="leapfrog",
    integrator_order=4,
    timestep=0.5,                  # fs
    maxstep=10000,
    iprint=1,
    pairs_to_stop=pairs_to_test,
    traj_file="photoion_trajectory.xyz",
    backfile="photoion_restart.xyz",
)

print(photoion.termination_channel, photoion.termination_step)
```

For the **QCT source**, construct the object using the neutral geometry and Hessian,
then define the **neutral** mode sampling:

```python
photoion = Photoionization(
    molecule,
    qchem=qcinput_ion,
    neutra_frankcondon_geom="neutral.xyz",
    neutral_hessian="hessian_neutral.hess",
    photon_energy=15.0,
    level=1,
    ionic_energy=([10.0, 11.0, 12.0], [0.0, 1.0, 0.0]),
    sampling_seed=1234,
)

photoion.Specify_Mode_Sampling(
    init_vib_type="Temp",
    init_rot_type="Temp",
    temp=300.0,
    random_rot=True,
)
```

The propagation call is identical for both sources. Electron and ionic inputs
can each be a number or a distribution, in any combination, as described below.
For Level 0, omit the electron/ionic constraints and set `level=0`.
Hessian input requires explicit mode sampling before a fresh launch; otherwise
an error requests `Specify_Mode_Sampling`. Passing explicit `qct_options` to the
constructor is also supported. With MD input, mode resampling is rejected:
its saved q,p frames already define the neutral ensemble.

`neutra_frankcondon_geom` is **required in every case**. It accepts an XYZ file,
XYZ text (Angstrom), or Cartesian coordinates (bohr), with the neutral molecule's
atom order. It defines the reference structure. If `neutral_traj` is supplied,
MD frames provide the sampled coordinates and momenta, even if a Hessian is also
supplied. The Hessian and QCT options are then ignored. Each frame's geometry is
retained through ionization. Without `neutral_traj`, `neutral_hessian` is required;
QCT uses that Hessian and the supplied reference geometry. Missing geometry or a
missing Hessian in QCT mode produces a clear error.

Choose exactly one stopping mode in `sample_and_run_dynamics`:

* `pairs_to_stop`: named channels with the existing `((i, j), "GT"/"LT", distance)`
  conditions. Indices are zero-based and distances are Angstrom. All conditions
  in a channel must hold; the first matching channel in dictionary order wins.
* `Rstop=20.0`: the existing global distance check; stop when **any atom-pair
  distance** exceeds 20 Angstrom. This checks all atom pairs, including pairs
  that were not initially bonded. It does not infer a bond list.

Both modes are checked **at every integration step**, including the terminal
step, independently of `iprint`. `maxstep` remains the time limit.
`reaction_persistence_steps=1` stops at the first channel match; larger values
require that channel on consecutive steps. The global `Rstop` check is immediate.
To use it, replace `pairs_to_stop=pairs_to_test` with `Rstop=20.0` in the run call.

The object is a `Molecule`: `photoion.q` and `photoion.p` contain its current
ionic state; `photoion.initial_state` holds the unchanged launch record.
Stopping diagnostics include `termination_reason`, `termination_channel`,
`termination_step`, and `termination_time_fs`. The run method returns `photoion`.

`sample_initial_conditions()` prepares a launch without propagation. It returns
its energy/momentum record; call the inherited `run_trajectory(...)` to propagate
that already prepared state. `sample_and_run_dynamics(...)` always prepares a
new launch unless `restart=True`.

A configured `sampling_seed` initializes a stream, so repeated fresh calls draw
new, reproducible samples. A `sampling_seed` passed directly to the sampling/run
call reproduces that one launch without advancing the configured stream.
Use distinct `traj_file`/`backfile` names when retaining an ensemble of runs.
Default trajectories are NVE; explicit thermostat settings use the ordinary
unimolecular driver. Restarting uses the ionic checkpoint, without reading or
sampling the neutral source again. On the same object, `restart=True` reuses its
last trajectory/restart paths unless replacements are given.

`output_dir="new_directory"` in a fresh sampling/run call saves the launch's
JSON/NPZ records. Default trajectory, restart and wavefunction paths then go
there too. Explicit trajectory/restart paths are used as supplied.

The existing one-call `molecule.tpepico(...)` interface remains available for
preparing or propagating an ensemble, with `n_samples` ionic launches per call.
`Fragment` inherits that method too. Its original keyword names are documented
in the following section; the constructor above uses the explicit
`neutra_frankcondon_geom`, `neutral_hessian`, and `neutral_traj` names.

## Existing tpepico function interface


Supply **one** neutral source:

* `neutral_hessian`: the previously calculated neutral Cartesian Hessian,
  together with the neutral equilibrium geometry. SMite's existing harmonic
  QCT routines generate neutral coordinates and momenta from these inputs.
* `neutral_md_file`: a trajectory from a neutral MD run you performed beforehand,
  containing saved coordinates **and momenta**. Frames are sampled uniformly
  with replacement. Their coordinates and momenta are used as stored.

Level-selecting arguments:

* `level`: `0` copies the momenta, `1` scales all 3N components, `2` scales only
  the internal part (conserving **P** and **L**), `3` adds photon/photoelectron
  recoil on top of `2`.
* `recoil_site`: `"com"` (default), an atom index, or per-atom weights. Level 3 only.
* `photon_direction`: beam direction, default `(0,0,1)`. Level 3 only.

The module does not run neutral MD, optimize a structure or calculate a Hessian.
It evaluates the neutral and ionic potential energies at each sampled geometry
through the corresponding existing quantum-chemistry/PES interfaces.

## Minimal calls

The molecule must already contain atom identities, masses, a geometry, and its
neutral interface in `molecule.qchem`. Supply the ionic interface separately,
including its charge and multiplicity. It can be a quantum-chemistry interface
or a fitted PES. Charge must increase by one. Selection of the intended
electronic state within a backend remains the responsibility of that backend's
configuration; this module does not implement electronic state tracking.

QCT from a previously computed Hessian:

```python
result = molecule.tpepico(
    ion_qchem,
    neutral_hessian="hessian_neutral.hess",
    neutral_geometry="neutral.xyz",
    qct_options={"init_vib_type": "ZPE"},
    photon_energy=15.0,
    level=1,
    electron_energy=4.0,
    ionic_energy=11.0,
    n_samples=100,
    seed=1234,
)
```

`neutral_geometry` may be an XYZ file, XYZ text, or a Cartesian array. If omitted,
the method uses the molecule's stored **equilibrium** coordinates `q_ini`.
`neutral_hessian` may also be a NumPy matrix. A Hessian file is a plain, square
numeric matrix, as saved by SMite, not a backend-specific Hessian output file.
The geometry, atom order, masses and Hessian must correspond to the same neutral
minimum. No neutral MD is required for this path.

From an already saved neutral MD trajectory:

```python
result = molecule.tpepico(
    ion_qchem,
    neutral_md_file="neutral_md.xyz",
    md_start=100,       # discard the first 100 saved frames, not MD steps
    md_stride=5,        # retain every fifth frame thereafter
    photon_energy=15.0,
    level=1,
    ionic_energy=11.0,
    n_samples=100,
    seed=1234,
)
```

MD input is SMite extended XYZ: atom count, comment line, then one row per atom
with `symbol x y z px py pz`. The last three columns must be **momenta**, not
velocities. Atom identities/order and complete frames are validated. No
recentering, extra rotation, thermostat operation or momentum projection is
applied to a saved frame. Choose equilibration removal and frame spacing to
match the source ensemble you intend to simulate.

## Experimental energies: constants and distributions

**All photon, electron and ionic-energy arguments are in eV.**

Here `ionic_energy` means the experimentally inferred **binding energy**:
photon energy minus electron kinetic energy. It is the energy transferred to
the molecular system in this approximation. It does **not** mean the atoms'
kinetic energy or the ion's internal energy above an ionic minimum. Convert
experimental values using another reference before supplying them here.

Each of `electron_energy` and `ionic_energy` accepts independently:

* A single number, including zero electron energy for an ideal threshold event.
* A tuple `(energy_grid, relative_density_grid)`.
* A two-column NumPy array or list of `(energy, relative_density)` rows.

For example:

```python
electron_pdf = ([3.0, 4.0, 5.0], [0.0, 1.0, 0.0])
ionic_pdf = ([10.0, 11.0, 12.0], [0.0, 1.0, 0.0])

# Each row is an accepted pair of inputs, with photon_energy=15.0:
energy_options = [
    {"electron_energy": 4.0,         "ionic_energy": 11.0},
    {"electron_energy": 4.0,         "ionic_energy": ionic_pdf},
    {"electron_energy": electron_pdf, "ionic_energy": 11.0},
    {"electron_energy": electron_pdf, "ionic_energy": ionic_pdf},
]
```

Either energy can also be omitted. The module samples/uses the supplied one
and obtains the missing one from photon energy balance. At least one is needed
for Level 1. Photon energy is a positive constant for each call.

The grids must be finite, nonnegative and strictly increasing. Densities must
be finite, nonnegative and have a nonzero integral; they need not already be
normalized. They represent **densities per eV**, not bin probabilities.
For nonuniform histogram bins, divide bin probabilities by bin widths first.
Use a tuple for two parallel arrays and an array/list for rows; this avoids
ambiguity for a two-point distribution.

A shape-preserving cubic spline (PCHIP) interpolates the density without
negative overshoot. Monte Carlo draws use its integrated density, with no
extrapolation beyond the supplied grid. Constants are exact constraints.

With both inputs supplied, photon balance couples them:

* Two constants must sum to the photon energy within `energy_tolerance`.
* One constant fixes the other energy. The supplied distribution must have
  nonzero density there; it cannot create independent energy fluctuations.
* Two distributions are treated as independent weighting functions before
  conditioning: their product is sampled along the photon-balance line.

If your ionic distribution was obtained simply by converting the same measured
electron distribution, supply **one** of them. Supplying both multiplies the
same information twice and changes the weighting. Correlated joint experimental
distributions are not inferred from two separate marginal spectra.

The distribution is also conditioned on nonnegative ionic kinetic energy for
the selected neutral frame. An entirely inaccessible frame raises an error
with its source index; it is not silently replaced. `selection_weight` records
the normalization of this conditioning (or a density evaluated at a constant).
It is not an absolute photoionization cross section. Each requested neutral
draw receives one launch; source populations are not automatically reweighted
by ionization cross sections or by this normalization.

## What the levels do

**Level 0** uses exactly the sampled neutral geometry and momenta on the ionic
PES. The vertical PES gap determines the binding energy; photon energy minus
that gap determines the inferred electron energy. A closed photon-energy
window is rejected. No experimental electron/ionic energy constraint is imposed
at this level; omit those arguments.

**Level 1** retains exactly the same sampled geometry, computes the ionic kinetic
energy required by the sampled experimental energies and the neutral/ionic PES
gap, and multiplies **every atomic Cartesian momentum component by one common
nonnegative factor**. There is no decomposition into vibrational momenta for
this correction. Directions are retained (or all momenta become zero if the
target kinetic energy is zero).

**Level 2** does what Level 1 does, but rescales only the **internal
(vibrational)** part of the momentum. The momentum is split with the Eckart
projector in mass-weighted coordinates, where the kinetic energy is a plain
Euclidean norm and the split is therefore exactly orthogonal:
`T = T_ext + T_int` with no cross term, and the internal part carries **zero**
linear and angular momentum. Rescaling it alone hits the same energy target as
Level 1 while leaving **P** and **L** bit-for-bit unchanged. Level 2 is strictly
preferable to Level 1 whenever the neutral sample carries translation or
rotation, which QCT sampling with `init_rot_type` or `random_rot` always does.

**Level 3** adds the momentum the photon brought in and the photoelectron
carried away,

```
dp = p_gamma - p_e ,   p_e = sqrt(2*eps) * n ,   p_gamma = (h*nu/c) * k
```

shared over the atoms as `p_i -> p_i + w_i*dp` with `sum_i w_i = 1`. Total linear
momentum then changes by exactly `dp`. Angular momentum changes by
`(sum_i w_i x_i) x dp`, i.e. by the offset of the **recoil centroid** from the
centre of mass:

| `recoil_site` | weights | change in **L** | physical limit |
|---|---|---|---|
| `"com"` (default) | `m_i/M` | none, exactly | zero internal excitation |
| an atom index `j` | `delta_ij` | `x_j x dp` | localised core hole |
| a weight array | as supplied | `(sum_i w_i x_i) x dp` | e.g. Dyson-orbital populations |

Mass-weighted recoil is pure translation and excites nothing internally; a
localised hole torques the molecule and excites vibration. Note that `"com"` is
the *zero internal excitation* choice, not literally the delocalised-valence
limit: translational recoil is `|dp|^2/2M` for **any** weights, and `m_i/M` is
simply the weighting that minimises `sum_i w_i^2/m_i`. A real delocalised hole
has populations set by electron density, not mass. The physically correct weights
are the Dyson-orbital atomic populations, which you supply as an array; see
`report/2026-09-18_photoionization_levels_2_3_design.md` section 8 for the open
question of deterministic versus stochastic site selection, which is worth a
factor of fifteen in internal excitation and is planned work. That asymmetry is the
recoil effect, and it survives the Level 2 rescale because the rescale is applied
only to the *neutral* internal component, never to the recoil's.

`photon_direction` (default `(0,0,1)`) sets the beam direction. The photoelectron
direction is sampled **isotropically**. Both arguments are rejected below Level 3,
so a caller can never believe recoil was modelled when it was discarded.

Recoil size, for orientation: `E_rec = |p_e|^2/2M = (m_e/M)*eps`, which for water
is 0.3 meV at 21 eV, 2.4 meV at 100 eV, 21 meV at 1 keV and 82 meV at 3 keV.
**Below roughly 100 eV Level 3 buys nothing over Level 2** — the recoil is far
under a vibrational quantum. At keV energies it is the size of a soft mode and is
a measured photoelectron peak shift.

All four levels use a Franck–Condon geometry at the instant of ionization. The
unchanged geometry is the sampled neutral phase point, which need not be the
equilibrium geometry used to build QCT modes.

Levels 1, 2 and 3 all enforce total energy balance within `energy_tolerance`
(default `1e-8` eV), and all three require an experimental electron or ionic
energy constraint; Level 0 takes none. Negative kinetic energy and attempts to
create nonzero momenta by rescaling an all-zero momentum vector are rejected. The
energy includes the **full Cartesian kinetic energy**, including any translation
in the source. It is not automatically an energy in the molecular COM frame.

Levels 2 and 3 add three rejections, all physical rather than numerical:

* A **monatomic** ion has no internal subspace, so there is nothing to rescale.
* The external motion alone exceeding the ionic energy budget.
* The recoil alone depositing more internal energy than the budget allows
  (a negative discriminant in the rescaling quadratic). No choice of scale factor
  can satisfy the balance, so the launch is refused rather than approximated.

`momentum_scale` in the launch record is the factor applied to whichever
component the level rescales: `1.0` at Level 0, the global 3N factor at Level 1,
and the neutral-internal factor at Levels 2 and 3. The companion field
`scaled_subspace` (`"none"`, `"all"`, `"internal"`) says which, so the record is
self-describing. At Levels 2 and 3 this factor may come out **negative** when the
recoil overshoots the internal budget, meaning the neutral vibrational motion has
to oppose the recoil. That is permitted, not an error: the sign of an internal
momentum has no absolute meaning in a Franck–Condon ensemble, where `+a` and
`-a` are equally represented.

No level models photon spin, electron partial waves, post-collision interaction,
or shake-up/shake-off channels. Levels 0, 1 and 2 model no recoil at all. The
photoelectron angular distribution is not modelled at Level 3 either: emission is
isotropic, and the dipole form `1 + beta*P2(cos theta)` is left for later (the
`beta`/`polarization` arguments exist but are refused). Isotropic sampling is
correct for an angle-integrated measurement and averages out over an
orientationally random ensemble.

Momentum behaviour by level: Level 0 preserves nuclear momenta; **Level 1
generally changes both total linear and angular momentum, which is a defect of
that level**; Level 2 conserves both exactly; Level 3 changes them by exactly the
amount the recoil requires. All are recorded before and after in the launch
record, and `recoil_included` in `initial_states.json` states whether recoil was
applied.

A **self-consistent photoelectron energy** — solving
`eps = h*nu - dV - dT(eps)` so as to *predict* the recoil peak shift instead of
consuming a measured one — is deliberately not implemented. Levels 1 to 3 take
the measured energy as input, which is self-consistent with coincidence data
because a measured photoelectron peak is already recoil-shifted. See the design
note in `report/2026-09-18_photoionization_levels_2_3_design.md`.

The two PESs must have a consistent energy reference. If fitted surfaces use
separate zeros, supply `ion_energy_offset` (eV), an additive shift to the ionic
potential used in the preparation balance. Calibrate it from a known gap at a
reference geometry. The offset does not change ionic forces; reported backend
potential energies remain raw, and the offset is recorded separately.

## Choosing a level

The code never picks a level for you and never inspects the photon energy to
decide one. This section is guidance only.

**The one formula.** The recoil energy handed to the nuclei is

```
E_rec  =  |p_e|^2 / (2 M_eff)  =  (m_e / M_eff) * eps
```

where `eps` is the photoelectron kinetic energy and `M_eff` is **the mass that
actually takes the recoil**: the whole molecule for a delocalised outer-valence
hole, but a *single atom* for a localised core hole. That second case is the one
that matters, and it is why photon energy alone is the wrong criterion.

Recoil energy in meV, COM limit / hole localised on the lightest atom:

| molecule | M [u] | 21 eV | 100 eV | 500 eV | 1 keV | 3 keV |
|---|---|---|---|---|---|---|
| H2    |   2.0 | 5.7 / 11.4 | 27.2 / 54.4 | 136 / 272 | 272 / 544 | 816 / 1633 |
| H2O   |  18.0 | 0.6 / 11.4 |  3.0 / 54.4 |  15 / 272 |  30 / 544 |  91 / 1633 |
| CH4   |  16.0 | 0.7 / 11.4 |  3.4 / 54.4 |  17 / 272 |  34 / 544 | 103 / 1633 |
| CO    |  28.0 | 0.4 /  1.0 |  2.0 /  4.6 |  10 /  23 |  20 /  46 |  59 /  137 |
| C6H6  |  78.1 | 0.2 / 11.4 |  0.7 / 54.4 |   4 / 272 |   7 / 544 |  21 / 1633 |
| CF3I  | 195.9 | 0.1 /  0.6 |  0.3 /  2.9 |   1 /  14 |   3 /  29 |   8 /   87 |

Note the columns: on the lightest atom the number depends only on that atom's
mass, not on the molecule. A hole on a hydrogen gives the same 54 meV at 100 eV
in H2 as in benzene.

**Compare `E_rec` against whatever you actually care about** — your analyser
resolution if you are matching a measured peak, or the vibrational energy scale
if you care about the trajectory. Taking a 50 meV resolution as an example, the
photoelectron energy at which recoil becomes visible is:

| | delocalised (COM) | localised on the lightest atom |
|---|---|---|
| H2   |   184 eV |    92 eV |
| H2O  |  1.6 keV |    92 eV |
| CH4  |  1.5 keV |    92 eV |
| CO   |  2.6 keV |   1.1 keV |
| C6H6 |  7.1 keV |    92 eV |
| CF3I |   18 keV |   1.7 keV |

**Suggested choice.**

* **Level 0** — you have no measured electron or ion energy to impose and want a
  pure Franck–Condon transfer. Takes no energy constraints.
* **Level 1** — only to reproduce results generated before Level 2 existed. It is
  superseded: for the same inputs Level 2 gives the same energy with correct
  momentum. There is no case where Level 1 is the better physics.
* **Level 2** — **the default** whenever you impose a measured energy. Threshold
  and VUV work (TPEPICO, He I/II, `hv` up to ~100 eV) with a delocalised valence
  hole sits here: recoil is a few meV at most and Level 3 would change nothing
  you can measure.
* **Level 3** — when `(m_e/M_eff)*eps` reaches the energy you can resolve. In
  practice: any core-level ionization above ~100 eV where the hole sits on a
  light atom, and anything above ~1 keV regardless. Set `recoil_site` to the
  ionized atom for a core hole; leave it `"com"` for a delocalised valence hole,
  where it reduces to pure translation and excites nothing internally.

**Two things that shift the answer.** A hole on hydrogen or a first-row atom puts
`M_eff` one to two orders of magnitude below the molecular mass, so Level 3 earns
its place far below 1 keV. Conversely a heavy atom carrying the hole (iodine
here) pushes the crossover into the many-keV range. The photon momentum term
`p_gamma = hv/c` is under 1% of `p_e` below ~100 eV and reaches 4-6% at a few
keV; it is always included at Level 3 and never matters below soft X-ray.

## Neutral QCT options

`photoion.Specify_Mode_Sampling(...)` uses the existing neutral samplers.
It reads and diagonalizes the supplied neutral Hessian, then configures the
same vibrational/rotational mode choices as polyatomic `Fragment` sampling.
It does not calculate a new Hessian or run neutral MD.
The lowercase alias `specify_mode_sampling(...)` is also available.

Controls (also accepted through explicit `qct_options`) are:
`init_vib_type` (`"ZPE"`, `"Temp"`, `"Wigner"`), `init_rot_type`, `temp` (K),
`jrot`, `fix_quantum`, `fix_energy`, `fix_temp`, `fix_wigner`, and `random_rot`.
Calling the method with no arguments explicitly chooses harmonic QCT with zero
vibrational quantum numbers, no additional rotational excitation, and no random
overall orientation (unless already selected through `qct_options`). Thermal vibrational
sampling includes quantized populations. Wigner sampling is the existing
harmonic ground-state option, rather than fixed-energy QCT.

Mode indices are zero-based. These existing neutral sampling options retain
their native units: in particular, `fix_energy=[(mode, energy), ...]` uses
**Hartree**, while temperatures use K. This differs from the experimental
energy arguments of `tpepico`, which use eV. Only a neutral minimum with positive
vibrational frequencies is accepted. The input Hessian is never recomputed.

## Results and optional ionic dynamics

`result.initial_states` contains the sampled neutral momenta, launch geometry,
ionic momenta, source frame/draw index, potential/kinetic energies, sampled
experimental energies, scale factor, balance residual, and linear/angular
momentum diagnostics. Angular momentum is about the molecular COM.
`result.ions` contains fresh `Molecule` objects configured on the ionic PES.
For a single launch, `result.ion` is a convenience accessor.

Prepare only, then propagate explicitly:

```python
result.ion.run_trajectory(timestep=0.1, maxstep=10000, Rstop=20.0)
```

Or add this to the `tpepico` call to propagate every prepared state:

```python
output_dir="glycine_ionic_launches",   # must be a new directory
dynamics={"timestep": 0.1, "maxstep": 10000, "Rstop": 20.0},
```

Time/stop settings retain `run_trajectory` units (fs and Angstrom).
Supply either `Rstop` or `pairs_to_stop`. The wrapper launches fresh NVE
trajectories with separate trajectory, restart, scratch and wavefunction
locations. It never calls `sample_and_run_trajectory` on the prepared ion.
`output_dir` alone saves initial states without running dynamics. Outputs include
`initial_states.json` and `initial_states.npz`; arrays use atomic units.

This is an initial-state preparation and single-PES dynamics module. Detector
acceptance, coincidence timing, instrument response, branching fractions and
state-resolved photoionization cross sections require subsequent modeling.
Rigid constraints are not supported by this initial implementation.
