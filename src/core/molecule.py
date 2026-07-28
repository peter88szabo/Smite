import numpy as np
import os
import math
import shutil
import random
import json

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
        self.fname      = None
        self.vref       = 0.0  #the equilibrium pot energy of fragment 
        self.vini       = None #the initial (sampled) pot energy which is likely out of equilibrium
        self.tini       = None
        self.vsave      = []
        self.qsave      = []
        self.tsave      = []
        self._nosehoover_state = None
        self._gle_state = None


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
        if thermostat == "nosehoover" and "nosehoover_xi" in state:
            from thermostats.nosehoover import NoseHoover

            self._nosehoover_state = NoseHoover(
                nfix=self.nfix,
                wmass=self.wmass,
                tau=float(thermo_param) * FS_TO_AU_TIME,
                target_temp=thermo_temp,
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

    def thermo_andersen(self, prob, dt, Ttarg):
        self.p = apply_thermostat(self, 'andersen', prob, Ttarg, dt)

    def thermo_nosehoover(self, tau, dt, Ttarg):
        self.p = apply_thermostat(self, 'nosehoover', tau / FS_TO_AU_TIME, Ttarg, dt)

    def thermo_gle(self):
        self.p = apply_thermostat(self, 'gle', None, None, None)

    def _init_gle_thermostat(self, dt, thermo_param, thermo_temp):
        initialize_gle_state(self, dt, thermo_param, thermo_temp)


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
                             Rstop=None,
                             thermostat=None,
                             thermo_param=None,
                             thermo_temp=None,
                             spectrum=False,
                             post_collision_analysis=False,
                             post_collision_analysis_file=None,
                             post_collision_bond_th_HX=1.5,
                             post_collision_bond_th_XX=2.0,
                             post_collision_equilibrium_geometries=None,
                             post_collision_channel_states=None,
                             post_collision_isolation_distance=100.0):

        if traj_file is None:
            traj_file = 'traj_' + self.fname + '.xyz' 

        if backfile is None:
            backfile = 'backup_for_restart_' + self.fname + '.xyz'

        restart_state = None
        if restart:
            startstep = self.restart_init(fname=self.fname, backfile=backfile, restart=True)
            restart_state = self._load_restart_state(backfile, startstep)

        if pairs_to_stop is not None and Rstop is None:
            self.pairstop = pairs_to_stop
        elif pairs_to_stop is None and Rstop is not None:
            self.Rstop = Rstop * ANGSTROM_TO_BOHR #reactive event condition for trajectory 
        else:
            raise ValueError("\nEither Rstop or pairs_to_stop must be given in the input")

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

            wf_dir = prepare_wavefunction_directory(self.qchem, restart=restart)


            print("********************************************************************************************\n")
        #--------------------------------------------------------------------------------------
        if restart:
            wf_dir = prepare_wavefunction_directory(self.qchem, restart=restart)


        dt = timestep * FS_TO_AU_TIME
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
            for istep in range(startstep, maxstep):

                # Reaction tests are dynamics conditions, not output conditions:
                # evaluate them at every stored state regardless of iprint.
                channel = 'default'
                Rcom_actual = None
                if collision:
                    Rcom_actual = self.reactants_cenmass_distance()
                    Rcom_min = min(Rcom_min, Rcom_actual)

                if self.pairstop is not None and self.Rstop is None:
                    tstop, channel = test_to_stop_specific(q=self.q, pairs_to_test=self.pairstop)
                elif self.pairstop is None and self.Rstop is not None:
                    if collision:
                        tstop = Rcom_actual > self.Rstop
                    else:
                        tstop = test_to_stop_general(q=self.q, tol=self.Rstop)
                    channel = 'Not Specified'

                #-----------------------------------------------------------------------------
                if (iprint > 0 and istep % iprint == 0 and istep > startstep) or istep == startstep or tstop:

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
                            )

                        print("\n Reactive event found: ", formula, "    Reaction channel: ", channel)
                        break
                #-----------------------------------------------------------------------------

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

                if istep == (maxstep-1):
                    print("\n Propgation time has reached the maximum number of steps")


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

        ndof = len(self.p) - self.nfix
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
