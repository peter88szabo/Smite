# Smite

Smite is a Python toolkit for quasiclassical trajectory calculations for molecular collisions and unimolecular dynamics. It supports constant-energy NVE trajectories, thermalized NVT trajectories, analytical and machine-learned potential energy surfaces, ab initio molecular dynamics where the forces are evaluated by external quantum chemistry backends, and nonadiabatic dynamics on coupled electronic states. Alongside trajectories it provides geometry optimization, transition-state and IRC following, coordinate scans, and vibrational and thermochemical analysis.

The public import surface is kept intentionally simple:

```python
from smite import Molecule, Fragment, Collision, Photoionization
```

## Main Capabilities

- Molecular collision and unimolecular trajectory simulations.
- NVE dynamics and NVT dynamics through thermostats such as Nose-Hoover and GLE colored-noise thermostat support.
- Analytical and machine-learned PES calculations through `peslib/` and the PES interface.
- Ab initio MD force and energy calls through interfaces such as XTB, ORCA, Psi4, PySCF, SCINE Sparrow, Molpro, and PES-backed calculations.
- Nonadiabatic dynamics by fewest-switches surface hopping on coupled electronic states.
- Photoionization initial conditions for TPEPICO-style experiments.
- Initial condition sampling for translational, rotational, vibrational, thermal, and normal-mode based setups.
- Rigid-fragment propagation with SHAKE/RATTLE constraints and reversible rigid-body drift.
- Geometry optimization, transition-state searches, IRC following, minimum-energy crossing points, and relaxed one- and two-dimensional coordinate scans.
- Vibrational frequency analysis and thermochemistry, including Grimme quasi-RRHO entropies.
- Analysis helpers for vibrational spectra, time-resolved spectra, energy partitioning, and scattering form factors.
- Independent parallel trajectory scheduling, restartable runs, and a `smite-gui` graphical input builder.

## Photoionization initial conditions

Create `photoion = Photoionization(molecule, qchem=qcinput_ion, ...)` with the
neutral source and experimental energies. For Hessian-based QCT, define the
neutral ensemble with `photoion.Specify_Mode_Sampling(...)`. Then call
`photoion.sample_and_run_dynamics(...)` with the usual integration and stopping
options. Use `pairs_to_stop` for named distance-based channels or `Rstop` for the
global atom-pair distance threshold; both are checked every step.

The constructor always requires `neutra_frankcondon_geom`. A supplied
`neutral_traj` selects saved neutral coordinates and momenta; otherwise
`neutral_hessian` is required for QCT. It does not run neutral MD or calculate a
Hessian. The one-call `molecule.tpepico(...)` interface also remains available.

Level 0 retains the sampled momenta. Level 1 scales all atomic Cartesian
momenta to experimental energy constraints. `electron_energy` and `ionic_energy`
each accept a single value or an `(energy_grid, density_grid)` distribution,
in every combination. Energies are eV; `ionic_energy` means binding energy
(photon energy minus electron kinetic energy).

See the [complete photoionization input](examples/example_photoionization.py)
and [usage guide](src/photoionization/README.md) for both input paths, energy
conventions, neutral mode sampling, and ionic dynamics.

## Nonadiabatic dynamics (surface hopping)

Trajectories can propagate on more than one electronic state using Tully's
fewest-switches surface hopping, with Baeck-An nonadiabatic couplings obtained
from the adiabatic energy gap. Pass `q_integrator="fssh"` to `run_trajectory`
together with one `qcinput` per electronic state, lowest state first:

```python
states = [
    {"qchem": "PES", "pes_name": "H2O+Kr+", "state": 0},
    {"qchem": "PES", "pes_name": "H2O+Kr+", "state": 1},
]

molecule.run_trajectory(
    integrator="verlet",
    timestep=0.25,
    q_integrator="fssh",
    num_states=2,
    active_state=1,
    state_qcinput=states,
    dtq=0.025,
    de_cutoff=0.5,
    Rstop=50.0,
)
```

- `state_qcinput` must list exactly `num_states` entries. Its order defines the
  state indices and is never re-sorted by energy, because the hop probabilities
  are indexed by it.
- `dtq` is the quantum timestep, in the same unit as `timestep`, defaulting to
  `timestep / 10`. The electronic amplitudes advance `timestep / dtq` times per
  classical step.
- `de_cutoff` is the adiabatic energy gap in Hartree below which couplings are
  evaluated. Above it the per-state gradients and Hessians are skipped.
- After a hop the surface-selecting keys are merged into `molecule.qchem`, so
  the classical force follows the new state. `molecule.active_state` holds the
  current state throughout the run.

The quantum step runs after the classical step and the thermostat and acts on
the momenta only. With the default `q_integrator=None` the propagation is
identical to an ordinary single-surface trajectory.

## Potential energy surface library

`peslib/` holds ready-to-use surfaces, selected with `{"qchem": "PES",
"pes_name": ...}` or by pointing `pes_path` at any directory:

| Directory | System | Notes |
| --- | --- | --- |
| `ArH2+` | Ar + H2(+) | |
| `Cl+CH4` | Cl + CH4 | analytic Hessian |
| `F+H2` | F + H2 | |
| `HO2_1Deltag` | HO2, O2(1-Delta-g) channel | Fortran, build with `make` |
| `HO2_3Sigma_negative` | HO2, O2(3-Sigma-minus) channel | Fortran, build with `make` |
| `OH+CH4` | OH + CH4 | analytic Hessian |
| `H2O+Kr+` | H2O + Kr(+), two electronic states | machine-learned, needs the `mlpes` extra |

Each directory provides a `pes_interface.py` defining a `PESCalculator`
constructed as `PESCalculator(pes_dir, config)`, where `config` is the `qcinput`
dictionary. It must define `energy(q, atoms)` and either `force(q, atoms)` or
`gradient(q, atoms)`. An optional `hessian(q, atoms)` is used when present and
otherwise obtained by central differences. Coordinates are flat Cartesian arrays
in bohr and energies are in Hartree.

The `H2O+Kr+` surfaces are machine-learned Gaussian-process models, one
TorchScript file per electronic state, and are what the surface-hopping example
above runs on. Their interface also exposes `variance(q, atoms)`, the GP
variance, which grows sharply once a trajectory leaves the training region and
is worth monitoring on long runs.

## Scientific parameter conventions

- For the Andersen thermostat, `thermo_param` is the mean per-atom collision
  time in femtoseconds. The collision probability per MD step is
  `1 - exp(-timestep / thermo_param)`. With COM removal or rigid constraints,
  a single collision clock redraws all Cartesian momenta, then projects the
  constraints. This collective update preserves the constrained Maxwell
  distribution; the mean redraw time of each atom remains `thermo_param`.
- `nfix` is a count of removed degrees of freedom; it is never interpreted as
  a suffix of Cartesian coordinates. Center-of-mass momentum and rigid
  constraints are projected as physical modes.
- Product equilibrium geometries must use the same atom order as the product
  fragment. Their mass-weighted best-fit orientation is used when contracting
  the equilibrium inertia tensor with the instantaneous angular momentum.
- A collision's `sampling_seed` seeds the complete initial-condition draw once;
  its two fragments consume successive draws. Default GLE generators also draw
  their seeds from this stream. An explicit GLE `thermo_param["seed"]` overrides
  that choice for a single trajectory.
- Reaction conditions are checked at every state, including `maxstep`, and the
  terminal frame is written regardless of `iprint`. The molecule records
  `termination_reason` (`"stop_condition"` or `"maxstep"`),
  `termination_channel`, `termination_step`, and `termination_time_fs`.
- Thermochemistry accepts explicit `symmetry_number` and
  `chirality_number` arguments. For example, use `symmetry_number=2` for
  water and `symmetry_number=12` for methane. As in MarXus, the default is
  one because a reliable general symmetry number cannot be inferred from an
  arbitrary noisy geometry.
- Thermal harmonic populations always use each mode's physical frequency.
  The legacy `thermal_frequency_cutoff_cm1` keyword is accepted but ignored
  with a warning because applying it only to populations is noncanonical.
- Morse `energy` sampling specifies total bound rovibrational energy in
  Hartree and therefore requires fixed `J`. Thermal Morse vibration and
  rotation use the coupled bound-state canonical distribution.

## Installation

From the repository root:

```bash
pip install -e .
```

Optional extras:

| Extra | Pulls in | Needed for |
| --- | --- | --- |
| `analysis` | matplotlib | plots produced by the analysis and scan helpers |
| `pyscf` | pyscf | the PySCF force backend |
| `sparrow` | scine-sparrow | the SCINE Sparrow force backend |
| `gui` | PySide6 | the `smite-gui` input builder |
| `mlpes` | torch, scikit-learn | the machine-learned surfaces in `peslib/H2O+Kr+` |
| `dev` | build, pytest | running the test suite and building distributions |
| `all` | matplotlib, pyscf, scine-sparrow, PySide6 | everything except `mlpes` |

```bash
pip install -e ".[analysis]"
pip install -e ".[mlpes]"
pip install -e ".[all]"
```

### Why `mlpes` is not in `all`

`all` deliberately leaves out `mlpes`. PyTorch is a large download, on the order
of a few hundred megabytes, and nothing outside `peslib/H2O+Kr+` uses it: the
whole classical quasiclassical-trajectory side of Smite, every ab initio
backend, the analytic Fortran surfaces, the optimizers, and the thermochemistry
all run without it. Folding it into `all` would impose that cost on everyone who
just wants ordinary QCT runs, so PyTorch is always requested explicitly:

```bash
pip install -e ".[all,mlpes]"
```

Nothing imports PyTorch at module load, so a Smite installed without `mlpes`
behaves normally; only constructing the `H2O+Kr+` calculator raises, and the
tests covering it skip themselves. There is no GPU requirement either -- the
models are small and a CPU-only build is enough:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

`scikit-learn` comes along with `mlpes` because `qchem_interfaces/gp_models.py`
imports it; the `peslib/H2O+Kr+` interface itself needs only PyTorch and NumPy.

The graphical input builder is installed as a console script:

```bash
smite-gui
```

The package name is `smite`.

```bash
python -c "import smite; print(smite.__file__)"
python -c "from smite import Molecule, Fragment, Collision; print(Molecule, Fragment, Collision)"
```

## External Programs

Some force backends require separately installed quantum chemistry programs. Smite does not bundle these executables.

Typical `qchem` dictionaries specify the backend, executable path, method settings, charge, multiplicity, scratch directory, and optional wavefunction output behavior:

```python
qcinput = {
    "qchem": "XTB",
    "path": "/path/to/xtb",
    "charge": 0,
    "multiplicity": 1,
    "scratch_dir": "scratch/my_run",
    "wfu": False,
}
```

For ORCA, Psi4, and other electronic-structure codes, method and basis settings depend on the selected method. Composite methods may not need a separate basis entry.

## Examples

Example input scripts are kept in `src/` as `test*.py` files. They are included in source distributions and installed under the distribution data examples directory.

Examples include collision tests, unimolecular dynamics tests, PES-backed runs, XTB/Psi4/PySCF runs, and thermostat tests such as the Nose-Hoover tert-butyl hydroperoxide example.

Run examples from the `src/` directory unless an example says otherwise:

```bash
cd src
python test_tButylOOH_NoseHoover_unimol.py
```

## Time-resolved vibrational spectra

Run a trajectory with `spectrum=True` to retain coordinates and physical
velocities at every integration step. The short-time spectrum follows the
energy-normalized method of Zhang, Xu, and Truhlar (J. Phys. Chem. A 2022,
126, 3006-3014): translation and rotation are projected out at every frame,
velocities are mass weighted, and integration over a frequency interval gives
its short-time-averaged vibrational kinetic energy.

```python
result = molecule.short_time_vibrational_spectrum(
    dt=0.5,                         # fs
    atom_indices=(0, 1),            # optional molecular subsystem
    window_fs=200.0,
    hop_fs=10.0,
    window="boxcar",                # paper-compatible default
    frequency_bands={
        "low": (0.0, 1000.0),
        "stretch": (1000.0, 2500.0),
    },
    max_frequency_cm1=4000.0,
    output_prefix="trajectory_STFT",
)
```

The analysis writes a compressed numerical archive, text tables, a
time-frequency heatmap, and frequency-band energy curves. Band boundaries use
fractional FFT bins. Each band reports its integrated energy, a parabolically
refined peak, spectral centroid, RMS width, and a narrow/expanded energy
sensitivity envelope. The envelope shifts both boundaries by half the physical
window resolution by default.

For reactive trajectories, analyze all candidate bonds and the full system in
one call while tracking fragment identity with hysteretic formation/breaking
distances:

```python
channels = collision.short_time_vibrational_channels(
    dt=0.5,
    channels={
        "O_O": (0, 1),
        "H_O1": (2, 0),
        "H_O2": (2, 1),
        "HOO_all": (0, 1, 2),
    },
    dynamic_bonds={
        "O_O": (0, 1, 1.80, 2.00),   # indices, form A, break A
        "H_O1": (2, 0, 1.35, 1.55),
        "H_O2": (2, 1, 1.35, 1.55),
    },
    channel_frequency_bands={...},
    window_fs=200.0,
    hop_fs=10.0,
)
```

Supplying `potential_energy_hartree` plus its reference zero (or an
`internal_energy_hartree` history) also produces the bound-mode virial
diagnostic `2<T_vib>/<E_internal>` and its plot. It should approach one for a
stationary bound subsystem, not during close reactive interaction where a
fragment energy partition is ambiguous.

Shorter windows improve time localization while longer windows improve true
frequency resolution. The Rayleigh resolution is approximately
`33356 / window_fs` cm-1; increasing `nfft` only makes the plotted frequency
grid finer and does not improve that physical resolution.

## Rigid-fragment propagation

Fragments constructed with `rigid=True` automatically retain all of their
reference intramolecular distances during dynamics.  Constrained
velocity-Verlet/leapfrog uses SHAKE for the position drift and, by default,
RATTLE for the final momentum projection:

```python
rigid_diatom = Fragment.Diatom_Init(
    "H2",
    ["H", "H"],
    req=0.741,
    rigid=True,
)

rigid_diatom.run_trajectory(
    integrator="verlet",
    constraint_algorithm="rattle",
    constraint_tolerance=1.0e-10,
    constraint_velocity_tolerance=1.0e-10,
    constraint_max_iterations=200,
    # ... normal trajectory arguments ...
)
```

Set `constraint_algorithm="shake"` for position-only SHAKE on regular distance
constraint groups. RATTLE is recommended whenever momenta, temperatures, or
thermostats matter. Rigid-body split groups always retain tangent momenta after
force kicks; Andersen updates project rigid momenta for either setting.
Rigid groups belonging to separate collision partners remain independent.

Rigid constraints currently support the `verlet` and `leapfrog` integrators.
Other integrators fail explicitly instead of silently allowing the fragment to
deform. Linear and planar groups with a singular distance-constraint Jacobian,
and groups with more than 32 atoms, use rigid-body Hamiltonian splitting.
The free nonlinear rotation is split symmetrically about the principal axes;
linear rotation is exact. This preserves shape and free angular momentum,
is reversible, and has second-order time accuracy with bounded energy error
for stable timesteps. The method follows
[Dullweber, Leimkuhler and McLachlan (1997)](https://doi.org/10.1063/1.474310).
Large groups retain an O(N) diagnostic constraint set. Mass-weighted fitting is
used to project the starting geometry, rather than to approximate free drift.

## Geometry optimization, transition states, IRC, and scans

`molecule.optimize_geometry(...)` relaxes the current structure in place. The
`optimizer` package drives the same `qcinput` backends for standalone work:

- `optimize_geometry` minimum searches, in internal coordinates by default or
  in Cartesians with `coordinates="cartesian"`, with trust-radius control,
  optional Hessian recalculation, and an optional closing frequency analysis.
- `optimize_transition_state` first-order saddle point searches.
- `follow_irc` intrinsic reaction coordinate following from a transition state
  in mass-weighted coordinates, writing the path and energy profile and
  optionally relaxing and comparing both endpoints.
- `optimize_spin_crossing` minimum-energy crossing points between two spin
  multiplicities.
- `scan_bond`, `scan_angle`, `scan_dihedral`, `scan_coordinate`,
  `scan_coordinate_2d`, `scan_normal_mode`, and `scan_xyz_file` for relaxed and
  rigid scans, including two-dimensional grids and displacement along a normal
  mode.

## Vibrational analysis and thermochemistry

`frequency_analysis(...)` builds or reads a Hessian, projects out translation
and rotation in the Eckart frame, reports harmonic frequencies, and can write
per-mode normal-mode animations. `thermochemistry_analysis(...)` turns those
frequencies into partition functions and thermodynamic quantities, using either
rigid-rotor harmonic-oscillator vibrational entropies or Grimme's quasi-RRHO
treatment controlled by a frequency cutoff, with configurable symmetry and
chirality numbers.

## Parallel Trajectory Runs

Use `base_seed` to reproduce an ensemble independently of worker scheduling.
An ensemble-level `sampling_seed` is accepted as an alias when `base_seed` is
absent; `base_seed` takes precedence when both are supplied. The runner derives
one seed per trajectory and does not forward a shared seed into each sampler.
Explicit GLE seeds are also derived per trajectory. Seeded results differ from
older versions that reused the same seed for both collision fragments.

Parallel execution is handled by `parallel.trajectory_runner`; the single-trajectory APIs remain unchanged. The runner starts up to `nparallel_jobs` independent trajectories at once, and when one trajectory finishes it immediately starts the next pending trajectory until `total_trajectories` is reached.

```python
from parallel.trajectory_runner import run_parallel_collisions

results = run_parallel_collisions(
    collision_template=reaction,
    total_trajectories=100,
    nparallel_jobs=8,
    nproc_per_job=4,
    run_name="acetone_OH",
    integrator="leapfrog",
    timestep=0.7,
    maxstep=10000,
    iprint=1,
    pairs_to_stop=pairs_to_test,
)
```

Each trajectory gets an isolated output directory such as `acetone_OH/traj_000003/`, with its own trajectory, backup, initial-condition, suppressed-output, and `status.json` files. Temporary quantum chemistry files go separately under `scratch/acetone_OH/traj_000003/` by default. PES trajectories are forced to one core by default; external quantum chemistry backends receive `qchem["nproc"] = nproc_per_job` and `qchem["scratch_dir"]` for that trajectory-specific scratch directory.

During parallel runs, per-trajectory sampling/initial-condition text is written to files named like `initial_conditions_for_traj_000003.dat`. The usual trajectory propagation lines are not printed directly by workers; the parent runner displays only the currently active trajectories in a compact progress table with step, time in fs, energy drift, temperature, and center-of-mass distance when available.

Progress display is controlled with `progress_mode`. Use `progress_mode="table"` for a live table that overwrites the same terminal lines, `progress_mode="log"` for compact line updates in redirected output, `progress_mode="none"` to silence it, or keep the default `progress_mode="auto"` to choose table mode only when stdout is an interactive terminal.

## Repository Layout

- `src/smite.py`: public compatibility module exposing `Molecule`, `Fragment`, `Collision`, and `Photoionization`.
- `src/core/`: main molecular, fragment, and collision classes.
- `src/dynamics/`: trajectory scratch, wavefunction, constraint, rigid-body, integrator, thermostat, and quantum driver helpers.
- `src/integrators/`: MD, predictor-corrector, and constrained RATTLE integrators.
- `src/fssh/`: fewest-switches surface hopping, Baeck-An couplings, and the adapter serving them from `pesrun`.
- `src/photoionization/`: photoionization initial conditions and ionic dynamics.
- `src/optimizer/`: minima, transition states, IRC, spin crossings, and coordinate scans.
- `src/normalmode/`: Hessians, Eckart projection, frequencies, and thermochemistry.
- `src/analysis/`: spectra, energy partitioning, conservation checks, and scattering form factors.
- `src/parallel/`: independent parallel trajectory scheduling helpers.
- `src/thermostats/`: NVT thermostat implementations.
- `src/qchem_interfaces/`: external quantum chemistry and PES force interfaces.
- `src/smite_gui/`: the `smite-gui` graphical input builder.
- `peslib/`: predefined potential energy surfaces and PES interface examples.
- `src/sampling/`, `src/utils/`: supporting simulation tools.
- `src/tests/`: the pytest suite.
- `src/test*.py`: example input scripts.
- `papers/`: local reference material, not packaged.

Generated scratch directories, output files, compiled PES binaries, build products, and the `papers/` reference directory are intentionally excluded from package distributions. PES source/interface files are kept in source distributions so they can be rebuilt locally when needed.

## Development Checks

Run the test suite from the repository root:

```bash
python -m pytest
```

Tests that need an optional dependency skip themselves: the `H2O+Kr+` surfaces
are skipped without PyTorch or without the model weights, and the Fortran
surfaces are skipped without `gfortran`.

Basic syntax check:

```bash
cd src
python -m py_compile smite.py core/*.py dynamics/*.py parallel/*.py analysis/*.py qchem_interfaces/*.py integrators/*.py thermostats/*.py fssh/*.py photoionization/*.py optimizer/*.py normalmode/*.py
```

Build a source and wheel distribution from the repository root:

```bash
python -m build
```

Install locally in editable mode while developing:

```bash
pip install -e .
```
