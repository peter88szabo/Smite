"""Define photoionization initial conditions, then propagate one ionic trajectory."""

from copy import deepcopy
from pathlib import Path

import numpy as np

from core.molecule import Molecule
from photoionization.distributions import EnergyConstraints, finite_scalar
from photoionization.driver import _configuration, tpepico
from photoionization.preparation import validate_phase_point
from photoionization.sources import HessianSource, _geometry
from sampling.random_seed import sampling_generator


class Photoionization(Molecule):
    """Define the ionic system and source; set neutral QCT modes when needed.

    ``neutral`` supplies atom identities, masses and the ground-state PES.
    ``qchem`` selects one ionic PES, including its charge and multiplicity.
    ``neutra_frankcondon_geom`` is always required (XYZ text/file or bohr array).
    ``neutral_traj`` selects saved MD q,p frames; otherwise ``neutral_hessian``
    is required for QCT. Construction does not run MD or calculate a Hessian.
    The neutral molecule is never propagated or modified by this object.
    Each fresh ``sample_and_run_dynamics`` call prepares one new ionic launch.
    """

    def __init__(
        self, neutral, qchem, fname=None, *, neutra_frankcondon_geom=None,
        neutral_hessian=None, neutral_traj=None, photon_energy=None,
        qct_options=None, level=0, electron_energy=None, ionic_energy=None,
        md_start=0, md_stride=1, ion_energy_offset=0.0, energy_tolerance=1e-8,
        sampling_seed=None,
    ):
        if not isinstance(neutral, Molecule):
            raise TypeError("neutral must be a Molecule or Fragment")
        if neutra_frankcondon_geom is None:
            raise ValueError("neutra_frankcondon_geom must be given for both MD and QCT preparation")
        if neutral_traj is None and neutral_hessian is None:
            raise ValueError("neutral_hessian must be given when neutral_traj is not supplied")
        if photon_energy is None:
            raise ValueError("Photoionization preparation requires photon_energy")
        if not isinstance(qchem, dict) or not {"charge", "multiplicity"}.issubset(qchem):
            raise ValueError("qchem must specify the chosen ionic charge and multiplicity")
        neutral_config, ion_config = _configuration(neutral.qchem), _configuration(qchem)
        if ion_config["charge"] != neutral_config["charge"] + 1:
            raise ValueError("Single photoionization requires ionic charge = neutral charge + 1")
        if neutral.has_rigid_constraints:
            raise ValueError("Photoionization currently requires an unconstrained molecular source")
        geometry, _, _ = validate_phase_point(
            _geometry(neutra_frankcondon_geom, neutral),
            np.zeros(3 * neutral.natom), neutral.mass)
        super().__init__(list(neutral.atoms), neutral.mass.copy(),
                         geometry.copy(), np.zeros_like(geometry))
        self.neutral = neutral
        self.qchem = ion_config
        self.fname = str(fname) if fname is not None else "photoion_" + (neutral.fname or "molecule")
        self.neutra_frankcondon_geom = geometry.copy()
        self.neutral_traj = neutral_traj
        self.neutral_hessian = neutral_hessian if neutral_traj is None else None
        self.sampling_set = False
        self._photoionization_sampling = None
        self._photoionization_rng = None
        self.initial_state = None
        self.preparation_output_dir = None
        self.traj_file = None
        self.backfile = None
        self.wavefunction_dir = None

        # A supplied trajectory selects MD, even if a Hessian was also given.
        # In that case only its saved q,p frames are sampled; the required
        # geometry establishes the molecular structure and atom ordering.
        from_md = neutral_traj is not None
        self.Specify_Photoionization_Sampling(
            photon_energy=photon_energy,
            neutral_hessian=None if from_md else neutral_hessian,
            neutral_geometry=None if from_md else geometry,
            neutral_md_file=neutral_traj,
            qct_options=None if from_md else qct_options,
            level=level, electron_energy=electron_energy, ionic_energy=ionic_energy,
            md_start=md_start, md_stride=md_stride, ion_energy_offset=ion_energy_offset,
            energy_tolerance=energy_tolerance, sampling_seed=sampling_seed)

    def Specify_Photoionization_Sampling(
        self, *, photon_energy, neutral_hessian=None, neutral_geometry=None,
        neutral_md_file=None, qct_options=None, level=0, electron_energy=None,
        ionic_energy=None, md_start=0, md_stride=1, ion_energy_offset=0.0,
        energy_tolerance=1e-8, sampling_seed=None,
    ):
        """Configure neutral input and Level 0/1 experimental energy sampling.

        Supply a neutral Hessian/geometry OR a saved neutral MD q,p file.
        Energies are eV; ionic_energy is binding energy. Electron and ionic
        inputs independently accept constants or (energy_grid, density_grid).
        sampling_seed initializes a stream: successive runs draw new states.
        No neutral dynamics, Hessian calculation or energy evaluation runs here.
        """
        if isinstance(level, (bool, np.bool_)) or level not in (0, 1):
            raise ValueError("Only photoionization levels 0 and 1 are implemented")
        photon = finite_scalar(photon_energy, "photon_energy", nonnegative=True)
        tolerance = finite_scalar(energy_tolerance, "energy_tolerance", nonnegative=True)
        if photon == 0 or tolerance == 0:
            raise ValueError("photon_energy and energy_tolerance must be positive")
        offset = finite_scalar(ion_energy_offset, "ion_energy_offset")
        if (neutral_hessian is None) == (neutral_md_file is None):
            raise ValueError("Provide exactly one of neutral_hessian or neutral_md_file")
        if neutral_md_file is not None and (neutral_geometry is not None or qct_options is not None):
            raise ValueError("neutral_geometry and qct_options apply only to neutral_hessian")
        if neutral_hessian is not None and (md_start != 0 or md_stride != 1):
            raise ValueError("md_start and md_stride apply only to neutral_md_file")
        if level == 0 and (electron_energy is not None or ionic_energy is not None):
            raise ValueError("Use Level 1 to impose experimental electron/ionic energy constraints")
        if level == 1:
            EnergyConstraints(photon, electron_energy, ionic_energy, tolerance)
        options = deepcopy(dict(
            photon_energy=photon, neutral_hessian=neutral_hessian,
            neutral_geometry=neutral_geometry, neutral_md_file=neutral_md_file,
            qct_options=qct_options, level=level, electron_energy=electron_energy,
            ionic_energy=ionic_energy, md_start=md_start, md_stride=md_stride,
            ion_energy_offset=offset, energy_tolerance=tolerance))
        rng = sampling_generator(sampling_seed)
        self._photoionization_sampling = options
        self._photoionization_rng = rng
        self._neutral_mode_sampling_set = neutral_md_file is not None or qct_options is not None
        self.sampling_set = True
        return self

    specify_photoionization_sampling = Specify_Photoionization_Sampling

    def Specify_Mode_Sampling(
        self, init_vib_type="ZPE", init_rot_type="Jfix", *, temp=None, jrot=None,
        fix_quantum=None, fix_energy=None, fix_temp=None, fix_wigner=None,
        random_rot=None, sampling_seed=None,
    ):
        """Configure the NEUTRAL Hessian-based ensemble, using SMite QCT modes.

        The vibrational/rotational choices and mode overrides follow Fragment's
        Specify_Mode_Sampling conventions: temperature in K, fix_energy in
        Hartree, and zero-based mode indices. The supplied neutral Hessian is
        read and diagonalized; no Hessian calculation or neutral MD is run.
        MD input already supplies q,p and does not use mode sampling.
        """
        if self._photoionization_sampling["neutral_md_file"] is not None:
            raise ValueError("Specify_Mode_Sampling applies to neutral Hessian/QCT input; "
                             "neutral_traj already supplies sampled coordinates and momenta")
        previous = self._photoionization_sampling.get("qct_options") or {}
        options = dict(init_vib_type=init_vib_type, init_rot_type=init_rot_type,
                       temp=temp, jrot=jrot,
                       random_rot=previous.get("random_rot", False) if random_rot is None else random_rot)
        for key, value in (("fix_quantum", fix_quantum), ("fix_energy", fix_energy),
                           ("fix_temp", fix_temp), ("fix_wigner", fix_wigner)):
            if value is not None:
                options[key] = deepcopy(value)
        source = HessianSource(
            self.neutral, self._photoionization_sampling["neutral_hessian"],
            self._photoionization_sampling["neutral_geometry"], options)
        rng = sampling_generator(sampling_seed) if sampling_seed is not None else self._photoionization_rng
        self._photoionization_sampling["qct_options"] = deepcopy(options)
        self.neutral_frequencies = source.freq.copy()
        self.neutral_vibsampling = deepcopy(source.vib)
        self.neutral_rotsampling = deepcopy(source.rot)
        self._photoionization_rng = rng
        self._neutral_mode_sampling_set = True
        return self

    specify_mode_sampling = Specify_Mode_Sampling

    def sample_initial_conditions(self, *, sampling_seed=None, output_dir=None):
        """Prepare one ion on this object, optionally saving its launch record.

        An explicit sampling_seed reproduces one particular launch without
        advancing the stream configured when this object was created.
        """
        if not self.sampling_set:
            raise ValueError("Provide photon_energy and the neutral source in Photoionization(...), "
                             "or call Specify_Photoionization_Sampling before running dynamics")
        if not self._neutral_mode_sampling_set:
            raise ValueError("Call photoion.Specify_Mode_Sampling(...) to define the neutral QCT ensemble "
                             "before sampling or running dynamics")
        seed = sampling_seed
        if seed is None:
            seed = self._photoionization_rng.integers(0, 2**32, size=4, dtype=np.uint32)
        prepared = tpepico(self.neutral, self.qchem, **self._photoionization_sampling,
                          seed=seed, output_dir=output_dir)
        ion = prepared.ion
        fname = self.fname
        # Reinitialize the molecular dynamics state for this fresh launch;
        # keep this object's neutral reference and sampling configuration.
        super().__init__(list(ion.atoms), ion.mass.copy(), ion.q_ini.copy(), ion.p_ini.copy())
        self.qchem = ion.qchem
        self.fname = fname
        self.vini, self.tini = ion.vini, ion.tini
        self.initial_state = prepared.initial_states[0]
        self.photoionization_initial_state = self.initial_state
        self.preparation_output_dir = prepared.output_dir
        self.termination_reason = self.termination_channel = None
        self.termination_step = self.termination_time_fs = None
        return self.initial_state

    def sample_and_run_dynamics(
        self, integrator="verlet", integrator_order=4, timestep=1.0,
        startstep=0, maxstep=100, iprint=2, traj_file=None, backfile=None,
        restart=False, pairs_to_stop=None, Rstop=None, reaction_persistence_steps=1,
        thermostat=None, thermo_param=None, thermo_temp=None, spectrum=False,
        constraint_algorithm="rattle", constraint_tolerance=1e-10,
        constraint_velocity_tolerance=1e-10, constraint_max_iterations=200,
        sampling_seed=None, output_dir=None, wavefunction_dir=None,
    ):
        """Sample one ionic launch and use the ordinary unimolecular driver.

        Choose pairs_to_stop (named channels) OR Rstop (global pair-distance
        threshold), in Angstrom. Conditions are checked every integration step,
        including the terminal step, independently of iprint. maxstep is also
        enforced. restart=True resumes a checkpoint without resampling.
        Default dynamics are NVE. Time step is fs, as for Collision.
        """
        if (pairs_to_stop is None) == (Rstop is None):
            raise ValueError("Give exactly one of Rstop or pairs_to_stop")
        if restart:
            if sampling_seed is not None:
                raise ValueError("restart=True resumes ionic momenta without sampling_seed")
        else:
            self.sample_initial_conditions(sampling_seed=sampling_seed, output_dir=output_dir)

        directory = Path(output_dir or self.preparation_output_dir or ".")
        if traj_file is None:
            traj_file = self.traj_file if restart and self.traj_file else directory / f"traj_{self.fname}.xyz"
        if backfile is None:
            backfile = self.backfile if restart and self.backfile else directory / f"backup_{self.fname}.xyz"
        if wavefunction_dir is None:
            wavefunction_dir = (self.wavefunction_dir if restart and self.wavefunction_dir else
                                Path(traj_file).parent / f"{Path(traj_file).stem}_wavefunctions")
        self.traj_file, self.backfile = str(traj_file), str(backfile)
        self.wavefunction_dir = str(wavefunction_dir)
        self.run_trajectory(
            integrator=integrator, integrator_order=integrator_order, timestep=timestep,
            startstep=startstep, maxstep=maxstep, iprint=iprint,
            traj_file=self.traj_file, backfile=self.backfile, restart=restart,
            collision=False, pairs_to_stop=pairs_to_stop, Rstop=Rstop,
            reaction_persistence_steps=reaction_persistence_steps,
            thermostat=thermostat, thermo_param=thermo_param, thermo_temp=thermo_temp,
            spectrum=spectrum, constraint_algorithm=constraint_algorithm,
            constraint_tolerance=constraint_tolerance,
            constraint_velocity_tolerance=constraint_velocity_tolerance,
            constraint_max_iterations=constraint_max_iterations,
            wavefunction_dir=self.wavefunction_dir)
        return self
