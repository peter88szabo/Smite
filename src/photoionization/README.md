# Photoionization: Levels 0 and 1

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

Both levels use a Franck–Condon geometry at the instant of ionization. The
unchanged geometry is the sampled neutral phase point, which need not be the
equilibrium geometry used to build QCT modes.

Level 1 enforces total energy balance within `energy_tolerance` (default
`1e-8` eV). Negative kinetic energy and attempts to create nonzero momenta by
rescaling an all-zero momentum vector are rejected. The energy includes the
**full Cartesian kinetic energy**, including any translation in the source.
It is not automatically an energy in the molecular COM frame.

Neither level includes photon/electron recoil, recoil directions, photon spin,
electron partial waves, or their linear/angular momentum balance. Level 0
preserves nuclear momenta; Level 1 generally changes both total nuclear linear
momentum and angular momentum. These quantities are recorded before and after,
not claimed to be conserved by the ionization model.

The two PESs must have a consistent energy reference. If fitted surfaces use
separate zeros, supply `ion_energy_offset` (eV), an additive shift to the ionic
potential used in the preparation balance. Calibrate it from a known gap at a
reference geometry. The offset does not change ionic forces; reported backend
potential energies remain raw, and the offset is recorded separately.

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
