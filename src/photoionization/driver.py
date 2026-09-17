"""Prepare and optionally propagate a single chosen ionic electronic state."""

from copy import deepcopy
from dataclasses import dataclass
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np

from integrators.gradient import Potential_Energy
from photoionization.distributions import EnergyConstraints, EnergyDistribution, finite_scalar
from photoionization.preparation import prepare_ionic_state
from photoionization.sources import HessianSource, MDTrajectorySource, positive_integer
from qchem_interfaces.qchem_validation import validate_qchem_input
from sampling.random_seed import sampling_generator
from utils.constants import HARTREE_TO_EV


def _configuration(config):
    if not isinstance(config, dict):
        raise ValueError("Neutral and ionic PES interfaces must be qchem dictionaries")
    ignored = {"_last_energy_cache", "_qchem_validated", "_unknown_qchem_keys", "scratch_dir"}
    return validate_qchem_input(deepcopy({key: value for key, value in config.items() if key not in ignored}))


def _json_input(value):
    if isinstance(value, EnergyDistribution):
        return {"energies_ev": value.energies.tolist(), "density_per_ev": value.densities.tolist()}
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_input(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_input(item) for item in value]
    return value


@dataclass
class TPEPICOResult:
    ions: list
    initial_states: list
    source: str
    output_dir: str | None = None

    @property
    def ion(self):
        if len(self.ions) != 1:
            raise ValueError("Use result.ions for an ensemble; result.ion requires one sample")
        return self.ions[0]


def tpepico(molecule, ion_qchem, *, photon_energy, neutral_hessian=None,
            neutral_geometry=None, neutral_md_file=None, qct_options=None,
            level=0, n_samples=1, electron_energy=None, ionic_energy=None,
            md_start=0, md_stride=1, ion_energy_offset=0.0, seed=None,
            energy_tolerance=1e-8, output_dir=None, dynamics=None):
    """Prepare ions from a supplied neutral Hessian/geometry OR saved neutral MD.

    Exactly one of ``neutral_hessian`` and ``neutral_md_file`` is required.
    The Hessian is a Cartesian matrix or text matrix file (Hartree/bohr^2).
    Geometry defaults to molecule.q_ini; arrays are bohr, XYZ text/files are
    Angstrom. MD files use SMite's extended XYZ q,p format. Neither source
    initiates neutral MD or a Hessian calculation.

    Photon, electron, ionic (binding) energies and ion_energy_offset are eV.
    Electron/ionic inputs independently accept constants or (grid, density)
    tuples or two-column tables. Provide at least one for Level 1; the other
    follows from photon energy balance. Level 0 accepts neither constraint.
    ``n_samples`` is the number of neutral draws / ionic launches.

    The ionic interface must explicitly identify its charge and multiplicity.
    Both PES energies need a common reference. An additive ion_energy_offset
    can align fitted surfaces; it affects balance, not the forces.

    By default only prepare states. ``dynamics`` optionally supplies keyword
    arguments for each ion's ordinary run_trajectory, on its selected PES.
    This starts a fresh NVE trajectory and never re-samples the prepared ion.
    """
    from core.molecule import Molecule

    n_samples = positive_integer(n_samples, "n_samples")
    if isinstance(level, (bool, np.bool_)) or level not in (0, 1):
        raise ValueError("Only photoionization levels 0 and 1 are implemented")
    photon = finite_scalar(photon_energy, "photon_energy", nonnegative=True)
    tolerance = finite_scalar(energy_tolerance, "energy_tolerance", nonnegative=True)
    if photon == 0 or tolerance == 0:
        raise ValueError("photon_energy and energy_tolerance must be positive")
    offset = finite_scalar(ion_energy_offset, "ion_energy_offset")
    if level == 0 and (electron_energy is not None or ionic_energy is not None):
        raise ValueError("Use Level 1 to impose experimental electron/ionic energy constraints")
    constraints = EnergyConstraints(photon, electron_energy, ionic_energy, tolerance) if level == 1 else None
    if (neutral_hessian is None) == (neutral_md_file is None):
        raise ValueError("Provide exactly one of neutral_hessian or neutral_md_file")
    if neutral_md_file is not None and (neutral_geometry is not None or qct_options is not None):
        raise ValueError("neutral_geometry and qct_options apply only to the neutral_hessian source")
    if neutral_hessian is not None and (md_start != 0 or md_stride != 1):
        raise ValueError("md_start and md_stride apply only to neutral_md_file")
    if not isinstance(ion_qchem, dict) or not {"charge", "multiplicity"}.issubset(ion_qchem):
        raise ValueError("ion_qchem must explicitly specify the chosen ionic charge and multiplicity")
    neutral_config, ion_config = _configuration(molecule.qchem), _configuration(ion_qchem)
    if ion_config["charge"] != neutral_config["charge"] + 1:
        raise ValueError("Single photoionization requires ionic charge = neutral charge + 1")
    if getattr(molecule, "has_rigid_constraints", False):
        raise ValueError("Photoionization currently requires an unconstrained molecular source")

    run_options = None if dynamics is None else dict(dynamics)
    if run_options is not None:
        forbidden = {"traj_file", "backfile", "wavefunction_dir", "startstep", "collision", "restart", "thermostat", "thermo_param", "thermo_temp"}
        if forbidden.intersection(run_options):
            raise ValueError("dynamics starts a fresh ionic NVE trajectory; output paths are managed by tpepico")
        if (run_options.get("Rstop") is None) == (run_options.get("pairs_to_stop") is None):
            raise ValueError("dynamics requires exactly one of Rstop or pairs_to_stop")
        inspect.signature(Molecule.run_trajectory).bind_partial(None, **run_options)
    if output_dir is not None and Path(output_dir).exists():
        raise FileExistsError(f"Choose a new output_dir to preserve existing results: {output_dir}")

    source = (MDTrajectorySource(molecule, neutral_md_file, md_start, md_stride)
              if neutral_md_file is not None else
              HessianSource(molecule, neutral_hessian, neutral_geometry, qct_options))
    source_name = "neutral_md" if neutral_md_file is not None else "neutral_hessian_qct"
    rng = sampling_generator(seed)
    rng_initial_state = deepcopy(rng.bit_generator.state)
    states = []
    # Separate scratch trees prevent neutral/ion guesses from contaminating
    # each other or the user's preceding neutral calculations.
    with TemporaryDirectory(prefix="smite_tpepico_") as scratch:
        neutral_eval, ion_eval = _configuration(neutral_config), _configuration(ion_config)
        for name, config in (("neutral", neutral_eval), ("ion", ion_eval)):
            config["scratch_dir"] = str(Path(scratch) / name)
            Path(config["scratch_dir"]).mkdir()
            config["wfu"] = False
        for index in range(n_samples):
            q, p, source_index = source.draw(rng, index)
            vn = Potential_Energy(neutral_eval, None, q.copy(), list(molecule.atoms))
            vi = Potential_Energy(ion_eval, None, q.copy(), list(molecule.atoms))
            try:
                state = prepare_ionic_state(
                    q, p, molecule.mass, vn, vi, photon_energy=photon, level=level,
                    constraints=constraints, ion_energy_offset=offset,
                    energy_tolerance=tolerance, rng=rng, source_index=source_index)
            except ValueError as exc:
                raise ValueError(f"Launch {index}, {source_name} sample {source_index}: {exc}") from exc
            states.append(state)

    token = uuid4().hex[:12]
    ions = []
    for index, state in enumerate(states):
        ion = Molecule(atoms=list(molecule.atoms), mass=np.array(molecule.mass, copy=True),
                       q_ini=state.q.copy(), p_ini=state.p.copy())
        ion.qchem = _configuration(ion_config)
        ion.fname = f"tpepico_{token}_{index:06d}"
        ion.vini = state.ionic_potential_hartree
        ion.tini = state.ionic_kinetic_ev / HARTREE_TO_EV
        ion.photoionization_initial_state = state
        ions.append(ion)

    result = TPEPICOResult(ions=ions, initial_states=states, source=source_name)
    if output_dir is not None or run_options is not None:
        destination = Path(output_dir) if output_dir is not None else Path(f"tpepico_{token}")
        destination.mkdir(parents=True, exist_ok=False)
        result.output_dir = str(destination.resolve())
        metadata = {
            "source": source_name, "neutral_md_file": str(neutral_md_file) if neutral_md_file is not None else None,
            "neutral_hessian_file": str(neutral_hessian) if isinstance(neutral_hessian, (str, Path)) else None,
            "qct_options": _json_input(qct_options),
            "md_start": int(md_start), "md_stride": int(md_stride),
            "rng_initial_state": rng_initial_state,
            "energy_inputs": {"photon_ev": photon,
                              "electron": _json_input(constraints.electron) if constraints else None,
                              "ionic": _json_input(constraints.ionic) if constraints else None},
            "atoms": list(molecule.atoms), "mass_au": np.asarray(molecule.mass).tolist(),
            "ionic_energy_definition": "binding energy = photon energy - electron kinetic energy",
            "units": {"q": "bohr", "p": "atomic units", "angular_momentum": "hbar"},
            "recoil_included": False, "initial_states": [state.to_dict() for state in states],
        }
        with (destination / "initial_states.json").open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, allow_nan=False)
        np.savez(destination / "initial_states.npz", q=np.array([s.q for s in states]),
                 p=np.array([s.p for s in states]), neutral_p=np.array([s.neutral_p for s in states]),
                 mass=molecule.mass, atoms=molecule.atoms)
        if run_options is not None:
            for ion in ions:
                ion.run_trajectory(
                    **run_options, traj_file=str(destination / f"{ion.fname}.xyz"),
                    backfile=str(destination / f"{ion.fname}_restart.xyz"),
                    wavefunction_dir=str(destination / f"{ion.fname}_wavefunctions"))
    return result
