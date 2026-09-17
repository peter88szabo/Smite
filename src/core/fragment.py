import numpy as np
import os
import math
import shutil
import random

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
from sampling.random_seed         import set_sampling_seed


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

from core.molecule import Molecule

class Fragment(Molecule):
    def __init__(self, atoms, mass, q_ini, p_ini):

        super().__init__(atoms=atoms, mass=mass, q_ini=q_ini, p_ini=p_ini)
        self.hessian      = None
        self.Lmat         = None #Normal mode to Cartesian transformator (eigvec of Hessian)
        self.freq         = None
        self.hessFile     = None
        self.vibsampling  = None
        self.rotsampling  = None
        self.linear       = None
        self.req_diat     = None
        self.omega_diat   = None
        self.overlap      = None
        self.rigid        = False
        self.surface      = False
        self.random_rot   = False
        self.MDsamp_qp    = None
        self.MDsamp_index = None
        

    @classmethod
    def Atom_Init(cls, fname, atoms):
        if len(atoms) != 1:
            raise ValueError("ERROR: Atom_Init requires only a single atom. You must provide an array with a single element, for instance: ['Cl']")

        q_ini = np.array([0.0, 0.0 , 0.0])
        p_ini = np.array([0.0, 0.0 , 0.0])

        mass = get_mass_vector(atoms)

        this = cls(atoms=atoms, mass=mass, q_ini=q_ini, p_ini=p_ini)

        this.fname = fname

        return this 

    @classmethod
    def atom_init(cls, fname, atoms):
        return cls.Atom_Init(fname=fname, atoms=atoms)

    @classmethod
    def Diatom_Init(cls, fname, atoms, req=None, omega=None, alpha=None, beta=None, De=None, rigid=False, random_rot=True, diatom='harmonic', nfix=0, sampling_seed=None, seed_metadata_file="sampling_seed_metadata.jsonl"):
        set_sampling_seed(sampling_seed, label=f"{fname}:Diatom_Init", metadata_file=seed_metadata_file)
        if len(atoms) != 2:
            raise ValueError("ERROR: Diatom_Init accept only a diatomic molecule.")

        if req is None: 
            raise ValueError("ERROR: req must be given in Diatom_Init()")
        elif (omega is None) and not rigid and diatom == 'harmonic': 
            raise ValueError("ERROR: For non-rigid harmonic diatom req and omega must be given in Diatom_Init()")
        elif ((beta is None and alpha is None) or De is None) and not rigid and diatom == 'morse': 
            raise ValueError("ERROR: For non-rigid Morse diatom req, beta (or alpha) and De must be given in Diatom_Init()")

        req = req * ANGSTROM_TO_BOHR

        if rigid:
            nfix = 1

        mass = get_mass_vector(atoms)

        q1 = np.array([req, 0.0, 0.0])
        q2 = np.zeros(3)
        p1 = np.zeros(3)
        p2 = np.zeros(3)

        q_ini = np.append(q1, q2)
        p_ini = np.append(p1, p2)

        q_ini, p_ini  = cenmass(q_ini, p_ini, mass)

        this = cls(atoms=atoms, mass=mass, q_ini=q_ini, p_ini=p_ini)

        this.fname      = fname
        this.req_diat = req
        this.omega_diat = omega * CM1_TO_AU_ANGULAR_FREQUENCY if omega is not None else None #from cm-1 to atomic unit
        if diatom == 'morse':
            beta_input = beta if beta is not None else alpha
            # User-facing Morse parameters follow the molecular-input convention:
            # req in Angstrom, beta in Angstrom^-1, De in kJ/mol.
            this.beta_diat = beta_input * BOHR_TO_ANGSTROM
            this.alpha_diat = this.beta_diat
            this.De_diat = De * KJMOL_TO_HARTREE
            redmass = mass[0] * mass[1] / (mass[0] + mass[1])
            this.omega_diat = this.beta_diat * math.sqrt(2.0 * this.De_diat / redmass)
        else:
            this.beta_diat = None
            this.alpha_diat = alpha
            this.De_diat = De
        this.freq = [this.omega_diat] if this.omega_diat is not None else None
        this.rigid  = rigid
        # Keep the physical model name separate from the rigid/non-rigid flag.
        # Overwriting self.rigid with a string breaks the downstream sampling logic.
        this.diatom_model = diatom
        this.random_rot = random_rot
        this.nvib = 0
        this.jvib = 0
        # Initialize the diatomic vibrational sampler immediately so that
        # Specify_Mode_Sampling can safely overwrite the default excitation.
        if not rigid and diatom in {'harmonic', 'morse'}:
            this.vibsampling = {0: (this.omega_diat, 'Q', 0)}
        else:
            this.vibsampling = None
        this.rotsampling = None
        this.nfix = nfix
        this.remove_com = True
        this._nfix_includes_com = bool(not rigid and int(nfix) >= 3)
        if rigid:
            this.add_rigid_constraint_group(
                np.arange(this.natom),
                reference_q=this.q_ini,
                degrees_of_freedom_removed=nfix,
            )

        return this 

    @classmethod
    def diatom_init(cls, fname, atoms, req=None, omega=None, alpha=None, beta=None, De=None, rigid=False, random_rot=True, diatom='harmonic', nfix=0, sampling_seed=None, seed_metadata_file="sampling_seed_metadata.jsonl"):
        return cls.Diatom_Init(
            fname=fname,
            atoms=atoms,
            req=req,
            omega=omega,
            alpha=alpha,
            beta=beta,
            De=De,
            rigid=rigid,
            random_rot=random_rot,
            diatom=diatom,
            nfix=nfix,
            sampling_seed=sampling_seed,
            seed_metadata_file=seed_metadata_file,
        )

    @classmethod
    def Polyatom_Init(cls, fname, qchem, xyz, nfix=0, linear=False, is_eckart=True, random_rot=True,
            rigid=False, fromMD=False, surface=False, surf_3atom=None, Amp_modeanim=30.0, print_nmode=True,
            sampling_seed=None, seed_metadata_file="sampling_seed_metadata.jsonl"):
        set_sampling_seed(sampling_seed, label=f"{fname}:Polyatom_Init", metadata_file=seed_metadata_file)

        natom, atoms, q_eq = parseXYZ(xyz)
        q_eq = np.array(q_eq) * ANGSTROM_TO_BOHR  #Angstrom to Bohr
        p_ini = np.zeros(len(q_eq))

        if len(atoms) <= 2:
            raise ValueError("ERROR: PolyatomInit requires more than two atoms.")

        if rigid:
            nfix = 3 * natom - 6 + linear

        mass = get_mass_vector(atoms)

        this = cls(atoms=atoms, mass=mass, q_ini=q_eq, p_ini=p_ini)

        this.fname      = fname
        this.qchem      = validate_qchem_input(qchem)
        this.rigid      = rigid
        this.surface    = surface
        this.surf_3atom = surf_3atom 
        this.random_rot = random_rot
        this.linear     = linear
        this.nfix       = nfix
        this.remove_com = True
        this._nfix_includes_com = bool(not rigid and int(nfix) >= 3)
        this.fromMD     = fromMD
        if rigid:
            this.add_rigid_constraint_group(
                np.arange(this.natom),
                reference_q=this.q_ini,
                degrees_of_freedom_removed=nfix,
            )

        #set the number of processors if not Orca used as qchem
        #this.init_qchem_interface()


        if surface:
            this.random_rot = False
            this.linear     = False 
       #--------------------------------------------------------------------
        if not rigid and not fromMD:
            hessFile = 'hessian_' + fname + '.hess'
            hessian = getHessian(qcinput=this.qchem, hessFile=hessFile, xyz=xyz)

            if print_nmode:
                freq, freq_low, Lmat = print_normalmode(fname=fname, atoms=atoms, mass=mass, q_eq=q_eq, hessian=hessian,
                                                    give_freq_and_Lmat=True, Amp=Amp_modeanim, is_eckart=is_eckart, linear=linear)
            else:
                freq, freq_low, Lmat = getNormalmode(mass=mass, hessian=hessian, linear=linear, q_eq=q_eq, is_eckart=is_eckart)

            freq_all = np.append(freq_low, freq)

            print_frequencies(fname, freq_all, linear=linear)

            this.hessFile   = hessFile
            this.hessian    = hessian
            this.freq       = freq
            this.Lmat       = Lmat

            if this.qchem['qchem'] == 'Orca':
                this.delete_orca_tmp()

        return this

    @classmethod
    def polyatom_init(cls, fname, qchem, xyz, nfix=0, linear=False, is_eckart=True, random_rot=True,
            rigid=False, fromMD=False, surface=False, surf_3atom=None, Amp_modeanim=30.0, print_nmode=True,
            sampling_seed=None, seed_metadata_file="sampling_seed_metadata.jsonl"):
        return cls.Polyatom_Init(
            fname=fname,
            qchem=qchem,
            xyz=xyz,
            nfix=nfix,
            linear=linear,
            is_eckart=is_eckart,
            random_rot=random_rot,
            rigid=rigid,
            fromMD=fromMD,
            surface=surface,
            surf_3atom=surf_3atom,
            Amp_modeanim=Amp_modeanim,
            print_nmode=print_nmode,
            sampling_seed=sampling_seed,
            seed_metadata_file=seed_metadata_file,
        )


    def Atom_Sampling(self):
        if len(self.atoms) != 1:
            raise ValueError("ERROR: Atom_Sampling requires only a single atom.")

        self.q = np.array([0.0, 0.0 , 0.0])
        self.p = np.array([0.0, 0.0 , 0.0])
        return

    def atom_sampling(self):
        return self.Atom_Sampling()

    def Diatom_Sampling(self, **kwargs):
        set_sampling_seed(
            kwargs.get("sampling_seed"),
            label=f"{getattr(self, 'fname', 'diatom')}:Diatom_Sampling",
            metadata_file=kwargs.get("seed_metadata_file", "sampling_seed_metadata.jsonl"),
        )
        if len(self.atoms) != 2:
            raise ValueError("ERROR: DiatomHaromicInit accept only a diatomic molecule.")

        verbosity = kwargs.get('verbosity', False)
        traj_index = kwargs.get('traj_index', -999)

        redmass = self.mass[0]*self.mass[1] / (self.mass[0] + self.mass[1])

        if self.rigid == False:
            if self.diatom_model == 'harmonic':
                self.q, self.p = diatom_vibration_harmonic_sampling(
                    vib_modes=self.vibsampling,
                    req=self.req_diat,
                    omega=self.omega_diat,
                    mass=self.mass
                )
            elif self.diatom_model == 'morse':
                self.q, self.p, angmom, inertia, jrot, nvib, Enj = diatom_vibration_morse_sampling(
                    vib_modes=self.vibsampling,
                    rot_modes=self.rotsampling,
                    req=self.req_diat,
                    beta=self.beta_diat,
                    De=self.De_diat,
                    mass=self.mass
                )
                self.inertia = inertia
                self.angmom = angmom
                self.jvib = jrot
                self.nvib = nvib
            else:
                raise NotImplementedError(
                    f"Diatomic vibrational sampling for model '{self.diatom_model}' is not implemented"
                )

        self.q, self.p = cenmass(self.q, self.p, self.mass)

        # Skip rotational dressing when no rotational excitation was requested.
        # This matches the polyatomic path and avoids sending `None` into the
        # rigid-rotor sampler for diatomics.
        if (self.rigid or self.diatom_model == 'harmonic') and self.rotsampling[0][1] is not None:
            self.p, angmom, inertia = diatom_rotation_rigidrot_sampling(rot_modes=self.rotsampling, mass=self.mass, q=self.q, p=self.p)
            self.inertia  = inertia
            self.angmom   = angmom

        if self.random_rot == True:
            self.q, self.p = euler_rot(self.q, self.p)
        return

    def diatom_sampling(self, **kwargs):
        return self.Diatom_Sampling(**kwargs)

    def Specify_Mode_Sampling(self, init_vib_type='ZPE', init_rot_type='Jfix', MDfile=None, MDprimitive=False, **kwargs):
        set_sampling_seed(
            kwargs.get("sampling_seed"),
            label=f"{getattr(self, 'fname', 'fragment')}:Specify_Mode_Sampling",
            metadata_file=kwargs.get("seed_metadata_file", "sampling_seed_metadata.jsonl"),
        )
        temp = kwargs.get('temp', None)
        nvib = kwargs.get('nvib', None)
        energy = kwargs.get('energy', None)
        jrot = kwargs.get('jrot', None)

        init_vib_type_norm = str(init_vib_type).lower()
        if init_vib_type_norm == 'temp' and temp is None:
            raise ValueError("temp must be given when init_vib_type='Temp'")
        if init_rot_type == 'Temp' and temp is None:
            raise ValueError("temp must be given when init_rot_type='Temp'")

        #-----------------------------------------------------------------------------------
        #Non-rigid Diatom case:
        if not self.rigid and len(self.atoms) == 2:
            if self.vibsampling is None:
                raise ValueError(f"Diatomic vibrational sampling is not initialized for diatom model '{self.diatom_model}'")
            if init_vib_type_norm == 'wigner':
                self.vibsampling[0] = (self.omega_diat, 'W', 0)
            elif 'nvib' in kwargs and nvib is not None:
                self.vibsampling[0] = (self.omega_diat, 'Q', nvib)
            elif 'energy' in kwargs and energy is not None:
                self.vibsampling[0] = (self.omega_diat, 'E', energy)
            elif init_vib_type_norm == 'temp':
                self.vibsampling[0] = (self.omega_diat, 'T', temp)
            elif init_vib_type_norm == 'zpe':
                self.vibsampling[0] = (self.omega_diat, 'Q', 0)
            else:
                raise ValueError("init_vib_type must be 'ZPE', 'Temp', 'Wigner', 'Q', or 'E'")

        #-----------------------------------------------------------------------------------
        #Non-rigid Polyatom case Harmonic Sampling:
        if not self.rigid and not self.fromMD and len(self.atoms) > 2:
           #uniformly initialize all modes (w.r.t to ZPE or Temperature):
            self.vibsampling = initialize_vibrational_modes(freq=self.freq, init_vib_type=init_vib_type, temp=temp)

           #if necessary then we may change certain modes sampling
            if {'fix_quantum', 'fix_energy', 'fix_temp', 'fix_wigner'}.intersection(kwargs):
                self.vibsampling = specify_vib_modes(vib_modes=self.vibsampling, **kwargs)

        #-----------------------------------------------------------------------------------
        #Non-rigid Polyatom case Sampling MD file:
        if not self.rigid and self.fromMD and not MDprimitive and len(self.atoms) > 2:
            if MDfile is None:
                raise ValueError("If fromMD = True then MDfile must be given in input")
            if not os.path.exists(MDfile):
                raise ValueError("MDfile is not found")

            self.MDsamp_index, self.MDsamp_qp = parse_MDtraj_as_sampling(MDfile)
        #-----------------------------------------------------------------------------------

        self.rotsampling = initialize_rotational_modes(init_rot_type=init_rot_type, temp=temp, jrot=jrot)

        self.MDprimitive = MDprimitive
        self.MDtemp      = temp

        return

    def specify_mode_sampling(self, init_vib_type='ZPE', init_rot_type='Jfix', MDfile=None, MDprimitive=False, **kwargs):
        return self.Specify_Mode_Sampling(
            init_vib_type=init_vib_type,
            init_rot_type=init_rot_type,
            MDfile=MDfile,
            MDprimitive=MDprimitive,
            **kwargs,
        )


    def sample_and_run_trajectory(self, integrator='verlet', integrator_order=4, timestep=1.0, startstep=0, maxstep=100, iprint=1,
                                  traj_file=None, backfile=None, Rstop=None, pairs_to_stop=None,
                                  reaction_persistence_steps=1,
                                  thermostat=None, thermo_param=None, thermo_temp=None, spectrum=False,
                                  restart=False, constraint_algorithm="rattle",
                                  constraint_tolerance=1.0e-10,
                                  constraint_velocity_tolerance=1.0e-10,
                                  constraint_max_iterations=200, **sampling_kwargs):

        if traj_file is None:
            traj_file = 'traj_' + self.fname + '.xyz' 
        if backfile is None:
            backfile = 'backup_for_restart_' + self.fname + '.xyz'

        #---------------------------------------------
        # Sample the internal motions of a fragment:
        #---------------------------------------------
        if not restart:
            if self.natom == 1:
                self.atom_sampling()
            elif self.natom == 2:
                self.diatom_sampling(**sampling_kwargs)
            elif self.natom > 2:
                self.polyatom_sampling(**sampling_kwargs)
            else:
                raise ValueError("Wrong sampling option in sample_and_run_trajectory() function")
        #---------------------------------------------

        if pairs_to_stop is None and Rstop is None:
            raise ValueError("\nEither Rstop or pairs_to_stop must be given in the input")


        self.run_trajectory(
            integrator=integrator,
            integrator_order=integrator_order,
            timestep=timestep,
            startstep=startstep,
            maxstep=maxstep,
            iprint=iprint,
            traj_file=traj_file,
            backfile=backfile,
            restart=restart,
            Rstop=Rstop,
            pairs_to_stop=pairs_to_stop,
            reaction_persistence_steps=reaction_persistence_steps,
            thermostat=thermostat,
            thermo_param=thermo_param,
            thermo_temp=thermo_temp,
            constraint_algorithm=constraint_algorithm,
            constraint_tolerance=constraint_tolerance,
            constraint_velocity_tolerance=constraint_velocity_tolerance,
            constraint_max_iterations=constraint_max_iterations,
            spectrum=spectrum,
        )



    def print_mode_sampling(self):
        print(f"\n{self.fname}")
        print("Vibrational mode sampling (Q: fixed quantum, E: fixed energy, T: thermal)")
        print("mode freq   sampl  quantum")
        for i in self.vibsampling:
            freq, sampl, quantum = self.vibsampling[i]
            if isinstance(freq, np.generic):
                freq = float(freq)
            if isinstance(quantum, np.generic):
                quantum = float(quantum)
            freq_str = f"{freq:.1f}" if isinstance(freq, (int, float)) else str(freq)
            quantum_str = f"{quantum:.1f}" if isinstance(quantum, float) else str(quantum)
            print(f"{i} : ({freq_str}, '{sampl}', {quantum_str})")
        print(f"\nRotational mode sampling (Q: fixed quantum, T: thermal)")
        print("mode  sampl  quantum")
        for i in self.rotsampling:
            sampl, quantum = self.rotsampling[i]
            if isinstance(quantum, np.generic):
                quantum = float(quantum)
            quantum_str = f"{quantum:.1f}" if isinstance(quantum, float) else str(quantum)
            print(f"{i} : ('{sampl}', {quantum_str})")
        print()

    def Polyatom_Sampling(self, **kwargs):
        set_sampling_seed(
            kwargs.get("sampling_seed"),
            label=f"{getattr(self, 'fname', 'polyatom')}:Polyatom_Sampling",
            metadata_file=kwargs.get("seed_metadata_file", "sampling_seed_metadata.jsonl"),
        )
        verbosity = kwargs.get('verbosity', False)
        traj_index = kwargs.get('traj_index', -999)

        if len(self.atoms) <= 2:
            raise ValueError("ERROR: Polyatom_Sample requires more than two atoms.")

        if not self.rigid and not self.fromMD and not self.MDprimitive:
            self.sampling_metadata = []
            # Backward-compatible public attribute; warnings and sampling
            # decisions are now recorded in one trajectory metadata stream.
            self.sampling_warnings = self.sampling_metadata
            self.q, self.p = polyatom_vibration_sampling(mass=self.mass, atoms=self.atoms, q_eq=self.q_ini, ww=self.freq, L=self.Lmat,
                                                   vib_modes=self.vibsampling, verbosity=verbosity, traj_index=traj_index,
                                                   mode_energy_diagnostics=kwargs.get("mode_energy_diagnostics", True),
                                                   mode_energy_file=kwargs.get("mode_energy_file", "sampled_mode_energy_diagnostics.dat"),
                                                   imaginary_mode_policy=kwargs.get("imaginary_mode_policy", "exclude"),
                                                   thermal_frequency_cutoff_cm1=kwargs.get("thermal_frequency_cutoff_cm1", 0.0),
                                                   sampling_metadata=self.sampling_metadata)

        if not self.rigid and self.fromMD and not self.MDprimitive:
            self.sampling_polyatom_from_MDfile() 

        #just primitve MD momenta generation, no sophisticated method sampling
        if not self.rigid and self.fromMD and self.MDprimitive:
            self.p = random_initialize_momenta(self.wmass, self.MDtemp)

        self.q, self.p = cenmass(self.q, self.p, self.mass)

        #coordinate (self.q) does not change when we dress up the molecule with an angular momentum to rotate
        #if the excitation (jrot or temp in init_rot_sampling is not defined (is NONE)
        #then we won't do rotation sampling
        if not self.MDprimitive and not self.surface and self.rotsampling[0][1] is not None:
            self.p, angmom, inertia = polyatom_rotation_sampling(rot_modes=self.rotsampling, mass=self.mass, q=self.q, p=self.p)

            # A proper vibrational energy decomposition is not implemented here yet.
            # Do not store a physically wrong Evib=0.0 after vibrational sampling.
            evib = None
            erot = sum([angmom[i]**2 / inertia[i]/2.0 for i in range(len(angmom))])

            self.erot     = erot
            self.vib      = evib
            self.inertia  = inertia
            self.angmom   = angmom

        #randomly rotate the molecule about its center of mass (surface is not rotated)
        if self.random_rot and not self.surface:
            self.q, self.p = euler_rot(self.q, self.p)

        self.p = np.array(self.p)
        self.q = np.array(self.q)

        self.overlap = check_atomic_overlap(self.atoms, self.q)

        if self.overlap:
            raise ValueError("Overlap detected when the polyatomic fragment is sampled")

        return

    def polyatom_sampling(self, **kwargs):
        return self.Polyatom_Sampling(**kwargs)

    def sampling_polyatom_from_MDfile(self):
        '''
        we already saved the q, p coords (as dictionary self.MDsamp_qp)
        originally created from a molecular dynamics run (likely NVT)

        then we choose a random trajectory index (random time)
        that will define self.q and self.p

        since those q and p coord are distorted
        first we need remove the COM motion
        and probably Eckart rotation too...or not???
        '''

        if not self.fromMD:
            raise ValueError("First you must initialize Polyatom_Init() as fromMD=True")

        #the index of the first time-step
        #and the index of last time-step in the MD file
        first, last = self.MDsamp_index

        #just in case the index does not exsist in the list
        while True:
            rnd = random.randint(first, last)
            if rnd in self.MDsamp_qp:
                self.q, self.p = self.MDsamp_qp[rnd]
                #later after calling this function
                #we purify the COM motion
                break
