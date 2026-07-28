# Smite

Smite is a Python toolkit for quasiclassical trajectory calculations for molecular collisions and unimolecular dynamics. It supports constant-energy NVE trajectories, thermalized NVT trajectories, analytical potential energy surfaces, and ab initio molecular dynamics where the forces are evaluated by external quantum chemistry backends.

The public import surface is kept intentionally simple:

```python
from smite import Molecule, Fragment, Collision
```

## Main Capabilities

- Molecular collision and unimolecular trajectory simulations.
- NVE dynamics and NVT dynamics through thermostats such as Nose-Hoover and GLE colored-noise thermostat support.
- Analytical PES calculations through `peslib/` and the PES interface.
- Ab initio MD force and energy calls through interfaces such as XTB, ORCA, Psi4, PySCF, SCINE Sparrow, Molpro, and PES-backed calculations.
- Initial condition sampling for translational, rotational, vibrational, thermal, and normal-mode based setups.
- Analysis helpers for vibrational spectra and scattering form factors.

## Installation

From the repository root:

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[analysis]"
pip install -e ".[pyscf]"
pip install -e ".[sparrow]"
pip install -e ".[all]"
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

## Parallel Trajectory Runs

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

- `src/smite.py`: public compatibility module exposing `Molecule`, `Fragment`, and `Collision`.
- `src/core/`: main molecular, fragment, and collision classes.
- `src/dynamics/`: trajectory scratch, wavefunction, integrator, and thermostat driver helpers.
- `src/integrators/`: MD and predictor-corrector integrators.
- `src/parallel/`: independent parallel trajectory scheduling helpers.
- `src/thermostats/`: NVT thermostat implementations.
- `src/qchem_interfaces/`: external quantum chemistry and PES force interfaces.
- `peslib/`: predefined potential energy surfaces and PES interface examples.
- `src/sampling/`, `src/normalmode/`, `src/optimizer/`, `src/utils/`: supporting simulation tools.
- `src/test*.py`: example input scripts.
- `papers/`: local reference material, not packaged.

Generated scratch directories, output files, compiled PES binaries, build products, and the `papers/` reference directory are intentionally excluded from package distributions. PES source/interface files are kept in source distributions so they can be rebuilt locally when needed.

## Development Checks

Basic syntax check:

```bash
cd src
python -m py_compile smite.py core/*.py dynamics/*.py parallel/*.py analysis/*.py qchem_interfaces/*.py integrators/*.py thermostats/*.py
```

Build a source and wheel distribution from the repository root:

```bash
python -m build
```

Install locally in editable mode while developing:

```bash
pip install -e .
```
