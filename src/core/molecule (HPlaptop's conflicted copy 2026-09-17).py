import numpy as np
import os
import math
import shutil
import random
import json
import hashlib

from utils.cenmass                import cenmass
from utils.euler                  import euler_rot          
from utils.format_and_print       import parse_MDtraj_as_sampling 
from utils.format_and_print       import parseCheckPoint
from utils.format_and_print       import parseXYZ
from utils.format_and_print       import print_trajectory
from utils.atomic_overlap         import check_atomic_overlap
from utils.atomic_masses          import get_mass_vector 
from utils.clustering             import cluster_chemical_formulas
from utils.constants              import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM
from utils.constants              import CM1_TO_AU_ANGULAR_FREQUENCY, FS_TO_AU_TIME
from utils.constants              import HARTREE_TO_KJMOL
from utils.constants              import KJMOL_TO_HARTREE, R_GAS_HARTREE_PER_K
from utils.distance               import test_to_stop_general
from utils.distance               import test_to_stop_specific
from utils.distance               import ReactionChannelHysteresis

from normalmode.hessian           import getHessian
from normalmode.eckart            import eckart_transform
from normalmode.nmodeprint        import print_normalmode 
from normalmode.normalmode        import print_frequencies 
from normalmode.normalmode        import getNormalmode  

from sampling.polyvibration       import initialize_vibrational_modes
from sampling.polyrotation        import initialize_rotational_modes
from sampling.polyvibration       import polyatom_vibration_sampling
from sampling.polyvibration       import specify_vib_modes 
from sampling.polyrotation        import polyatom_rotation_sampling
from sampling.thermal             import thermal_collision_energy
from sampling.diatom              import diatom_rotation_rigidrot_sampling
from sampling.diatom              import diatom_vibration_harmonic_sampling 
from sampling.morse_diatom        import diatom_vibration_morse_sampling
from sampling.surface             import orient_and_rotate_surface
from sampling.surface             import rotate_about_axis 


from integrators.gradient         import Energy
from qchem_interfaces.qchem_validation import validate_qchem_input

from thermostats.randmomentum     import random_initialize_momenta
from analysis.spectrum            import vibrational_spectrum as compute_vibrational_spectrum
from analysis.short_time_spectrum import (
    analyze_molecule_short_time_channels,
    analyze_molecule_short_time_spectrum,
)
from analysis.scattering          import get_scattering_form_factors as compute_scattering_form_factors
from dynamics.integrator_driver   import apply_integrator
from dynamics.integrator_driver   import initialize_integrator
from dynamics.scratch             import set_trajectory_scratch_dir
from dynamics.scratch             import trajectory_scratch_dir
from dynamics.thermostat_driver   import apply_thermostat
from dynamics.thermostat_driver   import initialize_gle_state
from dynamics.thermostat_driver   import prepare_thermostat
from dynamics.wavefunction        import prepare_wavefunction_directory
from dynamics.wavefunction        import wavefunction_file
from dynamics.constraints         import RigidConstraintSolver


class Molecule:
    def __init__(self, atoms=None, mass=None, q_ini=None, p_ini=None, nfix=0, restart=False, xyz_file_path=None):
        if restart:
            if not xyz_file_path:
                raise ValueError("Backup file must be provided when restart is True.")
            try:
                # parseCheckPoint is imported as a module-level helper, not a Molecule method.
                last_step, atoms, q_ini, p_ini = parseCheckPoint(xyz_file_path)
                mass = get_mass_vector(atoms)  
            except FileNotFoundError:
                raise ValueError("!!!!!!!!!!!!!!! Backup file does not exist !!!!!!!!!!!!!!!!!!!")
        else:
            last_step = 0
            if len(mass) != len(atoms) or len(q_ini) != 3 * len(atoms) or len(p_ini) != 3 * len(atoms):
                raise ValueError("Lengths of inputs are not consistent.")  

        self.nfix       = nfix #fix number of degree of freedom 
        self.remove_com = bool(int(nfix) >= 3)
        self._nfix_includes_com = self.remove_com
        self.atoms      = atoms
        self.natom      = len(atoms)
        self.q_ini      = np.array(q_ini)  
        self.p_ini      = np.array(p_ini)  
        self.q          = np.array(q_ini)  
        self.p          = np.array(p_ini)  
        self.mass       = np.array(mass)
        self.wmass      = np.repeat(mass, 3)
        self.totmass    = np.sum(mass) 
        self.fromMD     = False
        self.qchem      = None 
        self.last_step  = last_step
        self.Rstop      = None 
        self.pairstop   = None
        self._reaction_channel_hysteresis = None
        self.reaction_channel_candidate = None
        self.reaction_channel_persistence = 0
        self.reaction_persistence_steps = 1
        self.fname      = None
        self.vref       = 0.0  #the equilibrium pot energy of fragment 
        self.vini       = None #the initial (sampled) pot energy which is likely out of equilibrium
        self.tini       = None
        self.vsave      = []
        self.qsave      = []
        self.tsave      = []
        self._nosehoover_state = None
        self._gle_state = None
        self._rigid_constraint_groups = []
        self._rigid_constraint_signature_cache = None
        self._constraint_solver = None
        self._constraint_algorithm = "rattle"
        self._constraint_position_tolerance = 1.0e-10
        self._constraint_velocity_tolerance = 1.0e-10
        self._constraint_max_iterations = 200


    @property
    def has_rigid_constraints(self):
        return bool(self._rigid_constraint_groups)

    def add_rigid_constraint_group(
        self,
        atom_indices,
        reference_q=None,
        degrees_of_freedom_removed=None,
    ):
        """Register one independently rigid set of atoms.

        Atom indices refer to this molecule's global atom ordering.  The
        reference geometry supplies the distances retained by SHAKE/RATTLE.
        """
        indices = np.asarray(atom_indices, dtype=int)
        if indices.ndim != 1 or len(indices) < 2:
            raise ValueError("A rigid constraint group must contain at least two atoms")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("Rigid constraint group atom indices must be unique")
        if np.any(indices < 0) or np.any(indices >= self.natom):
            raise ValueError("Rigid constraint group contains an out-of-range atom index")
        for group in self._rigid_constraint_groups:
            if np.intersect1d(indices, group["atom_indices"]).size:
                raise ValueError("Rigid constraint groups may not overlap")

        if reference_q is None:
            full_reference = np.asarray(self.q, dtype=float).reshape((-1, 3))
            reference_positions = full_reference[indices]
        else:
            reference = np.asarray(reference_q, dtype=float)
            if reference.shape == (3 * self.natom,):
                reference_positions = reference.reshape((-1, 3))[indices]
            elif reference.shape == (len(indices), 3):
                reference_positions = reference
            elif reference.shape == (3 * len(indices),):
                reference_positions = reference.reshape((-1, 3))
            else:
                raise ValueError(
                    "Rigid reference coordinates must describe either the full molecule "
                    "or exactly the selected atom group"
                )

        if np.any(~np.isfinite(reference_positions)):
            raise ValueError("Rigid reference coordinates contain NaN or infinity")

        if degrees_of_freedom_removed is None:
            if len(indices) == 2:
                degrees_of_freedom_removed = 1
            else:
                centered = reference_positions - np.mean(reference_positions, axis=0)
                linear = np.linalg.matrix_rank(centered, tol=1.0e-10) <= 1
                degrees_of_freedom_removed = 3 * len(indices) - (5 if linear else 6)

        degrees_of_freedom_removed = int(degrees_of_freedom_removed)
        if degrees_of_freedom_removed < 1:
            raise ValueError("A rigid group must remove at least one degree of freedom")
        maximum_removed = 1 if len(indices) == 2 else 3 * len(indices) - 5
        if degrees_of_freedom_removed > maximum_removed:
            raise ValueError("Rigid-group degrees of freedom exceed the physical maximum")

        self._rigid_constraint_groups.append(
            {
                "atom_indices": np.array(indices, copy=True),
                "reference_positions": np.array(reference_positions, copy=True),
                "degrees_of_freedom_removed": degrees_of_freedom_removed,
            }
        )
        self._constraint_solver = None
        self._rigid_constraint_signature_cache = None

    def rigid_constraint_signature(self):
        """Return a stable fingerprint of rigid-group topology and distances."""
        if self._rigid_constraint_signature_cache is not None:
            return self._rigid_constraint_signature_cache
        digest = hashlib.sha256()
        for group in self._rigid_constraint_groups:
            indices = np.asarray(group["atom_indices"], dtype=int)
            reference = np.asarray(group["reference_positions"], dtype=float)
            header = {
                "atom_indices": indices.tolist(),
                "degrees_of_freedom_removed": int(
                    group["degrees_of_freedom_removed"]
                ),
            }
            digest.update(
                json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
            for local_i in range(len(indices)):
                for local_j in range(local_i + 1, len(indices)):
                    displacement = reference[local_i] - reference[local_j]
                    distance_hex = float(np.dot(displacement, displacement)).hex()
                    digest.update(distance_hex.encode("ascii"))
                    digest.update(b";")
        self._rigid_constraint_signature_cache = digest.hexdigest()
        return self._rigid_constraint_signature_cache

    def copy_rigid_constraint_groups_from(self, other, atom_offset=0):
        """Copy rigid groups from a fragment into a combined molecule."""
        atom_offset = int(atom_offset)
        for group in getattr(other, "_rigid_constraint_groups", ()):
            self.add_rigid_constraint_group(
                np.asarray(group["atom_indices"], dtype=int) + atom_offset,
                reference_q=np.asarray(group["reference_positions"], dtype=float),
                degrees_of_freedom_removed=group["degrees_of_freedom_removed"],
            )

    def prepare_rigid_constraints(
        self,
        algorithm="rattle",
        position_tolerance=1.0e-10,
        velocity_tolerance=1.0e-10,
        max_iterations=200,
    ):
        """Initialize constraint topology and project the starting state."""
        algorithm = str(algorithm).lower()
        if algorithm not in {"shake", "rattle"}:
            raise ValueError("constraint_algorithm must be 'shake' or 'rattle'")

        self._constraint_algorithm = algorithm
        self._constraint_position_tolerance = float(position_tolerance)
        self._constraint_velocity_tolerance = float(velocity_tolerance)
        self._constraint_max_iterations = int(max_iterations)

        if not self.has_rigid_constraints:
            self._constraint_solver = None
            return None

        solver = RigidConstraintSolver.from_rigid_groups(
            self._rigid_constraint_groups,
            position_tolerance=self._constraint_position_tolerance,
            velocity_tolerance=self._constraint_velocity_tolerance,
            max_iterations=self._constraint_max_iterations,
        )
        if self.nfix < solver.degrees_of_freedom_removed:
            raise ValueError(
                "nfix is smaller than the number of degrees of freedom removed "
                "by the registered rigid groups"
            )

        self.q = solver.project_positions(self.q, self.wmass)
        self.p = solver.rattle(self.q, self.p, self.wmass)
        self._constraint_solver = solver
        return solver

    def project_rigid_momenta(self):
        """Restore the RATTLE velocity constraints after a momentum operation."""
        if not self.has_rigid_constraints:
            return self.p
        if self._constraint_solver is None:
            self.prepare_rigid_constraints(
                algorithm=self._constraint_algorithm,
                position_tolerance=self._constraint_position_tolerance,
                velocity_tolerance=self._constraint_velocity_tolerance,
                max_iterations=self._constraint_max_iterations,
            )
        self.p = self._constraint_solver.rattle(self.q, self.p, self.wmass)
        return self.p

    def thermostat_removed_dof(self):
        """Return the statistical DOF count removed from thermostatting."""
        removed = int(self.nfix)
        if removed < 0:
            raise ValueError("nfix must be non-negative")
        if self.remove_com and not self._nfix_includes_com:
            removed += min(3, len(self.p))
        if removed >= len(self.p):
            raise ValueError(
                "The number of removed degrees of freedom must be smaller than 3N"
            )
        return removed

    def project_center_of_mass_momentum(self):
        """Project momenta onto the zero-total-momentum subspace."""
        if not self.remove_com:
            return self.p
        momentum = np.asarray(self.p, dtype=float).reshape((-1, 3))
        masses = np.asarray(self.mass, dtype=float)
        total_mass = float(np.sum(masses))
        if total_mass <= 0.0 or not np.isfinite(total_mass):
            raise ValueError("Center-of-mass projection requires positive finite masses")
        com_velocity = np.sum(momentum, axis=0) / total_mass
        momentum -= masses[:, None] * com_velocity[None, :]
        self.p = momentum.ravel()
        return self.p

    def non_com_removed_dof(self):
        """Return removed DOFs other than an explicitly counted COM triplet."""
        removed = int(self.nfix)
        if self._nfix_includes_com:
            removed -= min(3, len(self.p))
        return max(0, removed)


    def save_velocity_and_distance_matrix(self, time_fs=None):
        # Store the analysis history needed for post-processed spectra and
        # time-dependent scattering. Coordinates stay in Bohr internally and are
        # converted only in the analysis routines that need Angstrom.
        # Canonical momenta satisfy p_i = m_i v_i; store physical Cartesian
        # velocities, not mass-weighted coordinates p_i/sqrt(m_i).
        self.vsave.append(np.asarray(self.p, dtype=float) / self.wmass)
        self.qsave.append(np.array(self.q, copy=True))
        self.tsave.append(float(time_fs) if time_fs is not None else np.nan)

        #qxyz = np.reshape(self.q, (-1, 3))

        #dist_matrix = np.linalg.norm(qxyz[:, np.newaxis, :] - qxyz[np.newaxis, :, :], axis=-1)

        # Extract the upper triangle of dist_matrix without the diagonal entries
        #upper_triangle_indices = np.triu_indices_from(dist_matrix, k=1)
        #dist_vector = dist_matrix[upper_triangle_indices]

        #each element of the list is the distance matrix corresponding to a timestep 
        #self.dsave.append(dist_vector)


    def vibrational_spectrum(self, dt, print_maxfreq=5000.0):
        return compute_vibrational_spectrum(self, dt, print_maxfreq=print_maxfreq)

    def short_time_vibrational_spectrum(self, dt=None, **kwargs):
        """Analyze histories collected with spectrum=True using STFT."""
        return analyze_molecule_short_time_spectrum(
            self,
            dt_fs=dt,
            **kwargs,
        )

    def short_time_vibrational_channels(self, channels, dt=None, **kwargs):
        """Analyze several fixed atom channels with dynamic fragment tracking."""
        return analyze_molecule_short_time_channels(
            self,
            channels=channels,
            dt_fs=dt,
            **kwargs,
        )

    def time_resolved_vibrational_spectrum(self, dt=None, **kwargs):
        """Alias for short_time_vibrational_spectrum."""
        return self.short_time_vibrational_spectrum(dt=dt, **kwargs)

    def get_scattering_form_factors(self, dt, qmin=0.0, qmax=8.0, nq=600, dpi=600):
        return compute_scattering_form_factors(self, dt, qmin=qmin, qmax=qmax, nq=nq, dpi=dpi)

    def optimize_geometry(self, update=True, **kwargs):
        from optimizer import optimize_geometry

        result = optimize_geometry(self.qchem, self.atoms, self.q, **kwargs)
        if update:
            self.q = np.asarray(result.q, dtype=float)
            self.q_ini = np.asarray(result.q, dtype=float)
        return result

    def center_of_mass(self, q, mass):
        """
        Calculate the center of mass of the fragment
        this is different then cenmass() from utils
        cenmass() shift the fragmentnt into the COM
        """
        xyz = np.reshape(q, (-1, 3))
        return np.average(xyz, axis=0, weights=mass)

    def rotate_random(self):
        """
        Rotate randmoly the fragment molecule
        about its center of mass w.r.t Euler angles
        """
        qq, pp = cenmass(self.q, self.p, self.mass)

        self.q, self.p = euler_rot(qq, pp)

    def rotate_about_bond(self, atom1, atom2, theta, **kwargs):
        """
        Rotate a fragment with theta angle about a bond
        """
        verbosity  = kwargs.get('verbosity', False)
        traj_index = kwargs.get('traj_index', -999)
        bond_th_HX = kwargs.get('bond_th_HX', 1.4 * ANGSTROM_TO_BOHR) # bond threshold in Angstrom for H-X, where X = any non H-atom
        bond_th_XX = kwargs.get('bond_th_XX', 2.0 * ANGSTROM_TO_BOHR) # bond threshold in Angstrom for X-X bonds to be considered as a part of a fragment

        self.q = rotate_fragment(self.atoms, self.q, atom_ax1=atom1, atom_ax2=atom2, auto_frag=True, rot_angle=theta,
                                 bond_th_HX = bond_th_HX, bond_th_XX = bond_th_XX)

    ##wave function file should be generated automatically based on the name of molecule
    #as it given in the qcinput dictionary
    def get_energy(self, file_wf=None):
        check = False
        if file_wf is None and self.qchem['wfu']:
            file_wf = 'garbage_wavefunc_file.txt'
            check = True

        #Everything else here (other than this single line)
        #is to prevent unnecessary wave function printing when wfu is switched on
        T, V, E = Energy(self.qchem, file_wf, self.q, self.p, self.atoms, self.wmass)

        if check and os.path.exists(file_wf) and self.qchem['wfu']:
            os.remove(file_wf)
            
        return T, V, E 

    def _trajectory_scratch_dir(self, traj_file):
        return trajectory_scratch_dir(self.fname, traj_file)

    def _set_trajectory_scratch_dir(self, traj_file, restart=False):
        return set_trajectory_scratch_dir(self, traj_file, restart=restart)

    @staticmethod
    def _restart_state_path(backfile):
        return str(backfile) + ".state.json"

    @staticmethod
    def _json_compatible(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, os.PathLike):
            return os.fspath(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): Molecule._json_compatible(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [Molecule._json_compatible(item) for item in value]
        return value

    def _save_restart_state(self, backfile, completed_step, initial_energy, thermostat,
                            thermo_param=None, thermo_temp=None, dt=0.0,
                            integrator="verlet", integrator_order=4, propagator=None):
        py_state = random.getstate()
        np_state = np.random.get_state()
        state = {
            "version": 2,
            "completed_step": int(completed_step),
            "initial_energy": float(initial_energy),
            "thermostat": thermostat,
            "thermo_param": self._json_compatible(thermo_param),
            "thermo_temp": thermo_temp,
            "timestep_au": float(dt),
            "integrator": {
                "name": integrator,
                "order": int(integrator_order),
            },
            "python_random_state": {
                "version": py_state[0],
                "internal": list(py_state[1]),
                "gauss": py_state[2],
            },
            "numpy_random_state": {
                "name": np_state[0],
                "keys": np_state[1].tolist(),
                "position": np_state[2],
                "has_gauss": np_state[3],
                "cached_gaussian": np_state[4],
            },
        }

        if integrator == "predcorr":
            if propagator is None:
                raise ValueError("Predictor-corrector restart requires an initialized propagator")
            state["integrator"]["predcorr"] = {
                "step": int(propagator.step),
                "save_veloc": propagator.save_veloc.tolist(),
                "save_force": propagator.save_force.tolist(),
            }

        if self.has_rigid_constraints:
            constrained_dof = sum(
                int(group["degrees_of_freedom_removed"])
                for group in self._rigid_constraint_groups
            )
            state["constraints"] = {
                "algorithm": self._constraint_algorithm,
                "position_tolerance": self._constraint_position_tolerance,
                "velocity_tolerance": self._constraint_velocity_tolerance,
                "max_iterations": self._constraint_max_iterations,
                "degrees_of_freedom_removed": constrained_dof,
                "signature": self.rigid_constraint_signature(),
            }

        if self._nosehoover_state is not None:
            state["nosehoover_xi"] = float(self._nosehoover_state.xi)
        if self._gle_state is not None:
            state["gle"] = {
                "gp": self._gle_state.gp.tolist(),
                "rng_state": self._json_compatible(self._gle_state.rng.bit_generator.state),
            }

        collision_state = {}
        for name in (
            "q_collision_initial", "p_collision_initial", "vrel_ini", "Lorb_ini",
            "Jrot_ini_A", "Jrot_ini_B", "Erelsq_ini", "bimp",
        ):
            value = getattr(self, name, None)
            if value is not None:
                collision_state[name] = self._json_compatible(value)
        if collision_state:
            state["collision_state"] = collision_state

        reaction_hysteresis = getattr(self, "_reaction_channel_hysteresis", None)
        if reaction_hysteresis is not None:
            state["reaction_hysteresis"] = reaction_hysteresis.restart_state()

        state_file = self._restart_state_path(backfile)
        temporary_file = state_file + ".tmp"
        with open(temporary_file, "w", encoding="utf-8") as handle:
            json.dump(state, handle, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_file, state_file)

    def _load_restart_state(self, backfile, completed_step):
        state_file = self._restart_state_path(backfile)
        try:
            with open(state_file, "r", encoding="utf-8") as handle:
                state = json.load(handle)
        except FileNotFoundError:
            print(f"Warning: restart state file {state_file} is absent; RNG and thermostat state cannot be restored")
            return None

        if state.get("version") != 2:
            raise ValueError(f"Unsupported restart-state version in {state_file}")
        if int(state.get("completed_step", -1)) != int(completed_step):
            raise ValueError("Coordinate checkpoint and restart-state sidecar refer to different steps")
        return state

    def _restore_restart_state(self, state, thermostat, thermo_param, thermo_temp, dt):
        if state is None:
            return None
        if state.get("thermostat") != thermostat:
            raise ValueError(
                "Restart thermostat does not match the thermostat stored in the checkpoint"
            )

        current_thermo_param = self._json_compatible(thermo_param)
        if state.get("thermo_param") != current_thermo_param or state.get("thermo_temp") != thermo_temp:
            raise ValueError("Restart thermostat parameters do not match the checkpoint")
        if not np.isclose(float(state.get("timestep_au")), float(dt), rtol=0.0, atol=1.0e-14):
            raise ValueError("Restart timestep does not match the checkpoint")

        saved_constraints = state.get("constraints")
        if self.has_rigid_constraints and saved_constraints is None:
            raise ValueError(
                "Rigid restart checkpoint does not contain SHAKE/RATTLE metadata"
            )
        if saved_constraints is not None:
            if not self.has_rigid_constraints:
                raise ValueError(
                    "Restart checkpoint contains rigid constraints but the current molecule does not"
                )
            if saved_constraints.get("algorithm") != self._constraint_algorithm:
                raise ValueError("Restart constraint algorithm does not match the checkpoint")
            constraint_values = (
                ("position_tolerance", self._constraint_position_tolerance),
                ("velocity_tolerance", self._constraint_velocity_tolerance),
            )
            for name, current_value in constraint_values:
                if not np.isclose(
                    float(saved_constraints.get(name)),
                    float(current_value),
                    rtol=0.0,
                    atol=0.0,
                ):
                    raise ValueError(f"Restart constraint {name} does not match the checkpoint")
            if int(saved_constraints.get("max_iterations", -1)) != self._constraint_max_iterations:
                raise ValueError("Restart constraint max_iterations does not match the checkpoint")
            if int(saved_constraints.get("degrees_of_freedom_removed", -1)) != int(
                self._constraint_solver.degrees_of_freedom_removed
            ):
                raise ValueError("Restart rigid degrees of freedom do not match the checkpoint")
            if saved_constraints.get("signature") != self.rigid_constraint_signature():
                raise ValueError("Restart rigid constraint topology does not match the checkpoint")

        if thermostat == "nosehoover" and "nosehoover_xi" in state:
            from thermostats.nosehoover import NoseHoover

            self._nosehoover_state = NoseHoover(
                nfix=self.thermostat_removed_dof(),
                wmass=self.wmass,
                tau=float(thermo_param) * FS_TO_AU_TIME,
                target_temp=thermo_temp,
                scale_all=self.has_rigid_constraints,
            )
            self._nosehoover_state.xi = float(state["nosehoover_xi"])

        if thermostat == "gle" and "gle" in state:
            if self._gle_state is None:
                raise ValueError("GLE state could not be initialized for restart")
            gp = np.asarray(state["gle"]["gp"], dtype=float)
            if gp.shape != self._gle_state.gp.shape:
                raise ValueError("Stored GLE auxiliary state has an incompatible shape")
            self._gle_state.gp = gp
            self._gle_state.rng.bit_generator.state = state["gle"]["rng_state"]

        py_state = state["python_random_state"]
        random.setstate((
            int(py_state["version"]),
            tuple(py_state["internal"]),
            py_state["gauss"],
        ))
        np_state = state["numpy_random_state"]
        np.random.set_state((
            np_state["name"],
            np.asarray(np_state["keys"], dtype=np.uint32),
            int(np_state["position"]),
            int(np_state["has_gauss"]),
            float(np_state["cached_gaussian"]),
        ))

        for name, value in state.get("collision_state", {}).items():
            if isinstance(value, list):
                value = np.asarray(value, dtype=float)
            setattr(self, name, value)

        reaction_hysteresis = getattr(self, "_reaction_channel_hysteresis", None)
        saved_hysteresis = state.get("reaction_hysteresis")
        if reaction_hysteresis is not None and saved_hysteresis is not None:
            reaction_hysteresis.restore_restart_state(saved_hysteresis)
            self.reaction_channel_candidate = reaction_hysteresis.candidate_channel
            self.reaction_channel_persistence = reaction_hysteresis.consecutive_steps
        return float(state["initial_energy"])

    @staticmethod
    def _restore_integrator_restart_state(state, propagator, integrator, integrator_order):
        if state is None:
            return
        saved = state.get("integrator", {})
        if saved.get("name") != integrator or int(saved.get("order", -1)) != int(integrator_order):
            raise ValueError("Restart integrator or order does not match the checkpoint")
        if integrator != "predcorr":
            return
        if propagator is None or "predcorr" not in saved:
            raise ValueError("Predictor-corrector history is absent from the checkpoint")

        history = saved["predcorr"]
        save_veloc = np.asarray(history["save_veloc"], dtype=float)
        save_force = np.asarray(history["save_force"], dtype=float)
        if save_veloc.shape != propagator.save_veloc.shape or save_force.shape != propagator.save_force.shape:
            raise ValueError("Predictor-corrector restart history has an incompatible shape")
        propagator.step = int(history["step"])
        propagator.save_veloc = save_veloc
        propagator.save_force = save_force

    def stormer_single_step(self, dt):
        apply_integrator(self, 'stormer', dt)

    def leapfrog_single_step(self, dt):
        apply_integrator(self, 'leapfrog', dt)

    def verlet_single_step(self, dt):
        apply_integrator(self, 'verlet', dt)

    def rk4_single_step(self, dt):
        apply_integrator(self, 'rk4', dt)

    def symplectic_single_step(self, dt, this_class):
        apply_integrator(self, 'symplectic', dt, this_class)

    def sprk_single_step(self, dt, this_class):
        apply_integrator(self, 'sprk', dt, this_class)

    def predcorr_single_step(self, dt, this_class):
        apply_integrator(self, 'predcorr', dt, this_class)

    def thermo_berendsen(self, tau, dt, Ttarg):
        self.p = apply_thermostat(self, 'berendsen', tau / FS_TO_AU_TIME, Ttarg, dt)

    def thermo_andersen(self, collision_time, dt, Ttarg):
        """Apply Andersen collisions; ``collision_time`` and ``dt`` are in a.u."""
        self.p = apply_thermostat(
            self, 'andersen', collision_time / FS_TO_AU_TIME, Ttarg, dt
        )

    def thermo_nosehoover(self, tau, dt, Ttarg):
        self.p = apply_thermostat(self, 'nosehoover', tau / FS_TO_AU_TIME, Ttarg, dt)

    def thermo_gle(self):
        self.p = apply_thermostat(self, 'gle', None, None, None)

    def _init_gle_thermostat(self, dt, thermo_param, thermo_temp):
        initialize_gle_state(self, dt, thermo_param, thermo_temp)


    def tpepico(self, ion_qchem, *, photon_energy, neutral_hessian=None,
                neutral_geometry=None, neutral_md_file=None, qct_options=None,
                level=0, n_samples=1, electron_energy=None, ionic_energy=None,
                md_start=0, md_stride=1, ion_energy_offset=0.0, seed=None,
                energy_tolerance=1e-8, output_dir=None, dynamics=None):
        """Prepare one ionic PES from a neutral Hessian/geometry or saved MD q,p.

        Level 0 preserves momenta; Level 1 scales all Cartesian momenta to
        experimental energy constraints. Energies are eV; ionic_energy means
        binding energy (photon minus electron energy). Each energy constraint
        accepts a number or (energy_grid, density_grid). See photoionization/README.md.
        Neutral sampling routines are reused, and no neutral MD is run here.
        """
        from photoionization.driver import tpepico
        return tpepico(
            self, ion_qchem, photon_energy=photon_energy, neutral_hessian=neutral_hessian,
            neutral_geometry=neutral_geometry, neutral_md_file=neutral_md_file,
            qct_options=qct_options, level=level, n_samples=n_samples,
            electron_energy=electron_energy, ionic_energy=ionic_energy,
            md_start=md_start, md_stride=md_stride, ion_energy_offset=ion_energy_offset,
            seed=seed, energy_tolerance=energy_tolerance, output_dir=output_dir, dynamics=dynamics)

    def run_trajectory(self, integrator='verlet',
                             integrator_order=4,
                             timestep=1.0,
                             startstep=0,
                             maxstep=100,
                             iprint=2,
                             traj_file=None,
                             backfile=None,
                             restart=False,
                             collision=False,
                             pairs_to_stop=None,
                             reaction_persistence_steps=1,
                             Rstop=None,
                             thermostat=None,
                             thermo_param=None,
                             thermo_temp=None,
                             constraint_algorithm="rattle",
                             constraint_tolerance=1.0e-10,
                             constraint_velocity_tolerance=1.0e-10,
                             constraint_max_iterations=200,
                             spectrum=False,
                             post_collision_analysis=True,
                             post_collision_analysis_file=None,
                             post_collision_bond_th_HX=1.5,
                             post_collision_bond_th_XX=2.0,
                             post_collision_equilibrium_geometries=None,
                             post_collision_channel_states=None,
                             post_collision_isolation_distance=100.0,
                             wavefunction_dir="wavefunction_along_trajectory"):

        if traj_file is None:
            traj_file = 'traj_' + self.fname + '.xyz' 

        if backfile is None:
            backfile = 'backup_for_restart_' + self.fname + '.xyz'

        restart_state = None
        if restart:
            startstep = self.restart_init(fname=self.fname, backfile=backfile, restart=True)
            restart_state = self._load_restart_state(backfile, startstep)

        self.post_collision_analysis_result = None
        self.termination_reason = None
        self.termination_channel = None
        self.termination_step = None
        self.termination_time_fs = None
        self.reaction_channel_candidate = None
        self.reaction_channel_persistence = 0
        if pairs_to_stop is not None and Rstop is None:
            self.Rstop = None
            self.pairstop = pairs_to_stop
            self._reaction_channel_hysteresis = ReactionChannelHysteresis(
                reaction_persistence_steps
            )
            self.reaction_persistence_steps = self._reaction_channel_hysteresis.required_steps
        elif pairs_to_stop is None and Rstop is not None:
            self.pairstop = None
            self.Rstop = Rstop * ANGSTROM_TO_BOHR #reactive event condition for trajectory 
            self._reaction_channel_hysteresis = None
        else:
            raise ValueError("\nEither Rstop or pairs_to_stop must be given in the input")

        if self.has_rigid_constraints and integrator not in {"verlet", "leapfrog"}:
            raise ValueError(
                "Rigid SHAKE/RATTLE propagation currently supports only the "
                "verlet and leapfrog integrators"
            )

        self._set_trajectory_scratch_dir(traj_file, restart=restart)

        #--------------------------------------------------------------------------------------
        if not restart:
            self.vsave = []
            self.qsave = []
            self.tsave = []
            print("\n********************************************************************************************")
            if os.path.exists(traj_file):
                os.remove(traj_file)
                print(f"{traj_file} already exisits, it has been deleted to create a new one")

            wf_dir = prepare_wavefunction_directory(self.qchem, restart=restart, wf_dir=wavefunction_dir)


            print("********************************************************************************************\n")
        #--------------------------------------------------------------------------------------
        if restart:
            wf_dir = prepare_wavefunction_directory(self.qchem, restart=restart, wf_dir=wavefunction_dir)


        dt = timestep * FS_TO_AU_TIME
        self.prepare_rigid_constraints(
            algorithm=constraint_algorithm,
            position_tolerance=constraint_tolerance,
            velocity_tolerance=constraint_velocity_tolerance,
            max_iterations=constraint_max_iterations,
        )
        prepare_thermostat(self, thermostat, thermo_param, thermo_temp, dt, restart=restart)
        restart_initial_energy = self._restore_restart_state(
            restart_state, thermostat, thermo_param, thermo_temp, dt
        )

        # Initial energy
        T0, V0, E0 = self.get_energy() 
        if restart_initial_energy is not None:
            E0 = restart_initial_energy

        tstop = False
        Rcom_min = 1000000.0

        if collision and self.Rstop is not None:
            # For a collision Rstop is an outer reactant COM-separation
            # threshold, so the initial reactant separation must lie inside it.
            # Therefore a bimolecular run must start inside that bound, i.e. Rini < Rstop.
            if self.Rstop <= self.Rini:
                raise ValueError("Rstop must be larger than Rini")

        propag = initialize_integrator(integrator, integrator_order, len(self.q))
        self._restore_integrator_restart_state(
            restart_state, propag, integrator, integrator_order
        )

        with open(traj_file, "a") as file_trj:
            # Inspect the terminal state too, without an extra integration.
            for istep in range(startstep, maxstep + 1):

                # Reaction tests are dynamics conditions, not output conditions:
                # evaluate them at every stored state regardless of iprint.
                channel = 'default'
                Rcom_actual = None
                if collision:
                    Rcom_actual = self.reactants_cenmass_distance()
                    Rcom_min = min(Rcom_min, Rcom_actual)

                if self.pairstop is not None and self.Rstop is None:
                    tstop, channel = self._reaction_channel_hysteresis.update(
                        q=self.q, pairs_to_test=self.pairstop
                    )
                    self.reaction_channel_candidate = (
                        self._reaction_channel_hysteresis.candidate_channel
                    )
                    self.reaction_channel_persistence = (
                        self._reaction_channel_hysteresis.consecutive_steps
                    )
                elif self.pairstop is None and self.Rstop is not None:
                    if collision:
                        tstop = Rcom_actual > self.Rstop
                    else:
                        tstop = test_to_stop_general(q=self.q, tol=self.Rstop)
                    channel = 'Not Specified'

                #-----------------------------------------------------------------------------
                if (iprint > 0 and istep % iprint == 0 and istep > startstep) or istep in {startstep, maxstep} or tstop:

                    if self.qchem['wfu']:
                        file_wf = wavefunction_file(wf_dir, self.fname, istep)
                        T, V, E = self.get_energy(file_wf=file_wf) 
                    else:
                        T, V, E = self.get_energy() 

                    act_temp = self.traj_temperature()

                    dE = E - E0


                    self.print_trajectory(file_trj, istep, dt, T, V, dE, act_temp)
                    #print_trajectory(file_trj, self.atoms, self.q, self.p, V, dE, dt, istep)



                    if collision:
                        print(f"step: {istep:<10d} t[fs]: {istep*dt/FS_TO_AU_TIME:>12.1f} V[Eh]: {(V-self.vref):<16.6f} E[Eh]: {(E-self.vref):<16.6f} dE[kJ]: {dE*HARTREE_TO_KJMOL:16.3f}    T[K]: {act_temp:10.1f}   Rcom[A]: {Rcom_actual*BOHR_TO_ANGSTROM:8.2f}", flush=True)
                        progress_callback = getattr(self, "_parallel_progress_callback", None)
                        if callable(progress_callback):
                            progress_callback(
                                step=istep,
                                time_fs=istep * dt / FS_TO_AU_TIME,
                                energy_drift_kjmol=dE * HARTREE_TO_KJMOL,
                                temperature_K=act_temp,
                                rcom_angstrom=Rcom_actual * BOHR_TO_ANGSTROM,
                            )

                    else: #if not collision (just unimolecular dynamics) then we can ran the test anytime
                        print(f"step: {istep:<10d} t[fs]: {istep*dt/FS_TO_AU_TIME:<12.2f}  V[Eh]: {V:<16.6f} E[Eh]: {E:<16.6f} dE[kJ]: {dE*HARTREE_TO_KJMOL:16.3f}     T[K]: {act_temp:<10.1f}", flush=True)
                        progress_callback = getattr(self, "_parallel_progress_callback", None)
                        if callable(progress_callback):
                            progress_callback(
                                step=istep,
                                time_fs=istep * dt / FS_TO_AU_TIME,
                                energy_drift_kjmol=dE * HARTREE_TO_KJMOL,
                                temperature_K=act_temp,
                                rcom_angstrom=None,
                            )

                    if tstop:
                        self.termination_reason = "stop_condition"
                        self.termination_channel = channel
                        self.termination_step = istep
                        self.termination_time_fs = istep * timestep
                        #minPts: minum number of points to be a cluster
                        #eps: in Angstrom the tolerance within can be considered something as cluster

                        formula = cluster_chemical_formulas(self.q, self.atoms, eps=4.2, minPts=2)

                        ###here we need analysis function that calculate based on the number and type of the fragments:
                        #  - final Evib, Erot, Erel, Lorb_fin, Jrot_fin, vrel_fin 
                        #  - scattering angles
                        #  - or spectrum if is requested
                        #  - quantum number
                        #  - lifetime

                        if collision and post_collision_analysis:
                            analysis_file = post_collision_analysis_file
                            if analysis_file is None:
                                traj_dir = os.path.dirname(traj_file)
                                analysis_file = os.path.join(traj_dir, "post_collision_vectorcorr.dat") if traj_dir else "post_collision_vectorcorr.dat"

                            channel_state = None
                            if isinstance(post_collision_channel_states, dict):
                                if "products" in post_collision_channel_states:
                                    channel_state = post_collision_channel_states
                                else:
                                    channel_state = post_collision_channel_states.get(channel)
                            if channel_state is None and isinstance(self.pairstop, dict):
                                embedded_state = self.pairstop.get(channel)
                                if isinstance(embedded_state, dict) and "products" in embedded_state:
                                    channel_state = embedded_state

                            self.post_collision_analysis(
                                output_file=analysis_file,
                                channel=channel,
                                formula=formula,
                                step=istep,
                                time_fs=istep * dt / FS_TO_AU_TIME,
                                bond_th_HX=post_collision_bond_th_HX,
                                bond_th_XX=post_collision_bond_th_XX,
                                equilibrium_geometries=post_collision_equilibrium_geometries,
                                channel_state=channel_state,
                                isolation_distance=post_collision_isolation_distance,
                                trajectory_initial_energy_hartree=E0,
                                trajectory_final_energy_hartree=E,
                            )

                        print("\n Reactive event found: ", formula, "    Reaction channel: ", channel)
                        break
                #-----------------------------------------------------------------------------

                if istep == maxstep:
                    self.termination_reason = "maxstep"
                    self.termination_step = istep
                    self.termination_time_fs = istep * timestep
                    print("\n Propagation time has reached the maximum number of steps")
                    break

                apply_integrator(self, integrator, dt, propag)
                apply_thermostat(self, thermostat, thermo_param, thermo_temp, dt)

                if spectrum and thermostat is None:
                    self.save_velocity_and_distance_matrix(time_fs=(istep + 1) * dt / FS_TO_AU_TIME)


                #--------------------------------------------------------------------------
                #Backup:
                # The restart parser only consumes the step number and the per-atom q/p data.
                # Do not reuse stale V/T metadata from the last printed step when iprint > 1.
                backup_T = sum(0.5*np.array(self.p)*np.array(self.p)/np.array(self.wmass))
                backup_temp = self.traj_temperature()
                backup_file=open(backfile,'w')
                completed_step = istep + 1
                self.print_trajectory(backup_file, completed_step, dt, backup_T, 0.0, 0.0, backup_temp)
                backup_file.close()
                self._save_restart_state(
                    backfile, completed_step, E0, thermostat,
                    thermo_param, thermo_temp, dt, integrator,
                    integrator_order, propag,
                )
                #--------------------------------------------------------------------------



    def restart_init(self, fname, integrator='verlet', timestep=1.0, startstep=0, maxstep=100, iprint=2, traj_file=None, backfile=None, restart=False):
        self.fname = fname

        if backfile is None:
            raise ValueError('backup file name must be given in the input')

        if restart:
            try:
                last_step, self.atoms, self.q, self.p = parseCheckPoint(backfile)
                # Rebuild masses from the atoms loaded out of the checkpoint.
                self.mass = get_mass_vector(self.atoms)
                self.wmass = np.repeat(self.mass, 3)
                self.natom = len(self.atoms)
                self.totmass = np.sum(self.mass)
                self.last_step = last_step
                return self.last_step
            except FileNotFoundError as exc:
                raise ValueError("Backup file does not exist") from exc
        raise ValueError("restart_init requires restart=True")

    def traj_temperature(self):
        '''
        Actual temperature of the system

        N*R*T/2 = Ekin

        where N is the number of degrees of freedom of the system

        nfix is used for the number of the fixeddegrees of freedom
        (e.g. nfix=3, if translational modes are frozen)
        '''
       #kinetic energy of the system
        Ekin=sum(self.p*self.p/self.wmass)*0.5

        ndof = len(self.p) - self.thermostat_removed_dof()
        if ndof <= 0:
            raise ValueError("Number of active degrees of freedom must be positive in traj_temperature()")

        return 2.0*Ekin/float(ndof)/R_GAS_HARTREE_PER_K

    def merge_with(self, other_molecule):
        if not isinstance(other_molecule, Molecule):
            raise ValueError("other_fragment must be an instance of Molecule or Fragment")
        from core.fragment import Fragment
        
        combined_atoms = np.concatenate([self.atoms, other_molecule.atoms])
        combined_mass = np.concatenate([self.mass, other_molecule.mass])
        combined_coord_eq = np.concatenate([self.q, other_molecule.q])
        combined_mom = np.concatenate([self.p, other_molecule.p])
        
        # Create a new instance with combined attributes
        new_fragment = Fragment(combined_atoms, combined_mass, combined_coord_eq, combined_mom)
        new_fragment.nfix = (
            self.non_com_removed_dof() + other_molecule.non_com_removed_dof()
        )
        new_fragment.remove_com = self.remove_com or other_molecule.remove_com
        new_fragment._nfix_includes_com = False
        new_fragment.copy_rigid_constraint_groups_from(self, atom_offset=0)
        new_fragment.copy_rigid_constraint_groups_from(
            other_molecule, atom_offset=self.natom
        )

        return new_fragment

    def print_structure(self, trajfile, igeom):
        trajfile.write(str(self.natom) + "\n")
        trajfile.write("%8s %10d \n" % ("geom = ", igeom ))
        for i in range(self.natom):
            jx = 3*i
            jy = 3*i+1
            jz = 3*i+2
            trajfile.write("%3s %15.5f  %15.5f %15.5f \n" % (self.atoms[i], self.q[jx]*BOHR_TO_ANGSTROM, self.q[jy]*BOHR_TO_ANGSTROM, self.q[jz]*BOHR_TO_ANGSTROM))


    def print_trajectory(self, trajfile, istep, dt, T, V, dE, Temp):
        trajfile.write(str(self.natom) + "\n")
        trajfile.write("%6s %10d %8s %10.2f %12s %16.8f %12s %12.8f %10s %15.4f %8s %7.1f\n" %
                       ("step= ",istep, " t[fs]= ", dt*istep/FS_TO_AU_TIME, " Vpot[Eh]= ", V, " Tkin[Eh]= ", T, " dE[kJ]= ",dE*HARTREE_TO_KJMOL, " T[K]= ", Temp ))
        for i in range(self.natom):
            jx = 3*i
            jy = 3*i+1
            jz = 3*i+2
            trajfile.write("%3s %20.12f  %20.12f %20.12f  %20.12f  %20.12f %20.12f\n" % (self.atoms[i], self.q[jx]*BOHR_TO_ANGSTROM, self.q[jy]*BOHR_TO_ANGSTROM, self.q[jz]*BOHR_TO_ANGSTROM, self.p[jx], self.p[jy], self.p[jz]))
        trajfile.flush()

    def delete_orca_tmp(self):
        import os
        import shutil

        folder_name = 'orca_tmp'

        if os.path.exists(folder_name) and os.path.isdir(folder_name):
            shutil.rmtree(folder_name)

    def init_qchem_interface(self):
        self.qchem = validate_qchem_input(self.qchem)
        if self.qchem['qchem'] in {'PySCF', 'XTB', 'Sparrow_bin', 'Sparrow_Py'}:
            nproc = self.qchem['nproc']
            os.environ['OMP_NUM_THREADS'] = str(nproc)
