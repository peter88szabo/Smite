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
from analysis.vectorcorr          import analyze_collision as compute_collision_vector_correlations
from analysis.vectorcorr          import update_collision_vector_state
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


def surface_impact_geometry(Rini, bimp, theta, normal_direction, in_plane_direction):
    """Return displacement and incoming direction for a surface collision.

    The projectile COM is placed a distance ``Rini`` from the surface COM while
    its perpendicular distance from the incoming straight-line trajectory is
    exactly ``bimp``.
    """
    if Rini < bimp:
        raise ValueError("Rini must be at least as large as the impact parameter")

    normal = np.asarray(normal_direction, dtype=float)
    normal /= np.linalg.norm(normal)
    tangent = np.asarray(in_plane_direction, dtype=float)
    tangent -= np.dot(tangent, normal) * normal
    tangent /= np.linalg.norm(tangent)
    transverse = np.cross(normal, tangent)

    approach = math.cos(theta) * normal + math.sin(theta) * tangent
    longitudinal = math.sqrt(Rini * Rini - bimp * bimp)
    displacement = longitudinal * approach + bimp * transverse
    return displacement, approach


class Collision(Molecule):
    def __init__(self, fragment_A, fragment_B, qchem):
        atoms = fragment_A.atoms + fragment_B.atoms
        mass = np.append(fragment_A.mass, fragment_B.mass)
        q_ini = np.append(fragment_A.q_ini, fragment_B.q_ini)
        p_ini = np.append(fragment_A.p_ini, fragment_B.p_ini)

        super().__init__(atoms=atoms, mass=mass, q_ini=q_ini, p_ini=p_ini)
        self.qchem = validate_qchem_input(qchem)
        self.Rini = None
        self.bmax = None
        self.bsampling = None
        self.bimp = None
        self.Ecoll = None
        self.Ecoll_thermal = None
        self.tempcoll = None
        self.redmass = 0.0
        self.fragment_A = fragment_A
        self.fragment_B = fragment_B
        self.nfix = (
            fragment_A.non_com_removed_dof() + fragment_B.non_com_removed_dof()
        )
        self.remove_com = True
        self._nfix_includes_com = False
        self.copy_rigid_constraint_groups_from(fragment_A, atom_offset=0)
        self.copy_rigid_constraint_groups_from(fragment_B, atom_offset=fragment_A.natom)
        self.sampling_set = False
        self.Rcom = None

        self.vrel_ini    = None
        self.vrel_fin    = None
        self.vrelfin_sq  = None
        self.Lorb_ini    = None
        self.Lorb_fin    = None
        self.Jrot_ini_A  = None
        self.Jrot_fin_A  = None
        self.Jrot_ini_B  = None
        self.Jrot_fin_B  = None
        self.lifetime    = None
        self.Erelsq_ini  = None
        self.Erelsq_fin  = None
        self.Evib_ini    = None
        self.Erot_ini_eq = None
        self.Erot_ini    = None
        self.Evib_fin    = None
        self.Erot_fin    = None
        self.Erot_fin_eq = None
        self.bimp_fin    = None
        self.post_collision_analysis_result = None
        self.q_collision_initial = None
        self.p_collision_initial = None


    def Specify_Collision_Sampling(self, Rini=None, bmax=None, bsampling=2, Ecoll=None, Ecoll_thermal=False, temp=None,
                                   surf_skew_max=90.0, surf_skew_fix=None, surf_target_atom=None, surf_side=1,
                                   sampling_seed=None, seed_metadata_file="sampling_seed_metadata.jsonl"):
        set_sampling_seed(sampling_seed, label="Collision:Specify_Collision_Sampling", metadata_file=seed_metadata_file)

        if Rini == None or bmax == None or (Ecoll == None and Ecoll_thermal==False) or (Ecoll_thermal==True and temp==None):
            raise ValueError("Rini, bmax and Ecoll (or Ecoll_thermal) must be give in the input of Specify_Collision_Sampling()")

        # Backward-compatible mapping for the older boolean interface:
        # False -> fixed impact parameter, True -> area-uniform sampling b = bmax*sqrt(rand).
        # Without this normalization Python treats True == 1, which silently switches
        # old inputs onto the linear sampler.
        if isinstance(bsampling, bool):
            bsampling = 2 if bsampling else 0

        if Rini != None:
            Rini = Rini * ANGSTROM_TO_BOHR
        if bmax != None:
            bmax = bmax * ANGSTROM_TO_BOHR
        if Ecoll != None:
            Ecoll = Ecoll * KJMOL_TO_HARTREE
        

        self.Rini = Rini
        self.bmax = bmax
        self.Ecoll = Ecoll
        self.bsampling = bsampling
        self.Ecoll_thermal = Ecoll_thermal
        self.tempcoll = temp
        self.sampling_set = True
        self.surf_skew_max = surf_skew_max * math.pi / 180.0
        self.surf_skew_fix = surf_skew_fix 
        if surf_target_atom is not None:
            raise NotImplementedError("surf_target_atom is accepted by the interface but is not implemented in the current surface-collision setup")
        self.surf_target_atom = surf_target_atom
        self.surf_side = surf_side

    def specify_collision_sampling(self, Rini=None, bmax=None, bsampling=2, Ecoll=None, Ecoll_thermal=False, temp=None,
                                   surf_skew_max=90.0, surf_skew_fix=None, surf_target_atom=None, surf_side=1,
                                   sampling_seed=None, seed_metadata_file="sampling_seed_metadata.jsonl"):
        return self.Specify_Collision_Sampling(
            Rini=Rini,
            bmax=bmax,
            bsampling=bsampling,
            Ecoll=Ecoll,
            Ecoll_thermal=Ecoll_thermal,
            temp=temp,
            surf_skew_max=surf_skew_max,
            surf_skew_fix=surf_skew_fix,
            surf_target_atom=surf_target_atom,
            surf_side=surf_side,
            sampling_seed=sampling_seed,
            seed_metadata_file=seed_metadata_file,
        )

    def Set_Relative_Init_Coords_Molecule(self):
        r'''
        The potato is aimed by the projectile
        First, its center of mass will be put into the origin
        then we set the relative position and velocity of the projectiele

                   
                Z |
                  |
                  |
                  |************ <---P (projectile)
                  |                 |
              /\\\\\\\\             | bimp (b - impact param)
             /    |    \            |
            |     O-----|---------------------->
             \  /      /           sepx       X
              \\\\\\\\/
              /          sepx = sqrt(R^2 - b^2)
             /           R: distance between O-P
            /
           /
          /
         Y

        '''

        if self.sampling_set == False:
            raise ValueError("Error: First you must call Specify_Collision_Sampling() after you initialized the Collision() class")

        #shif the A and B molecule to their center of mass:
        qA, pA = cenmass(self.fragment_A.q, self.fragment_A.p, self.fragment_A.mass)
        qB, pB = cenmass(self.fragment_B.q, self.fragment_B.p, self.fragment_B.mass)

        if self.Ecoll_thermal:
            RT = R_GAS_HARTREE_PER_K * self.tempcoll
            self.Ecoll = thermal_collision_energy(RT) 
            print(f"temp[K]: {self.tempcoll:<12.2f}     Ecoll[kJ/mol]: {self.Ecoll*HARTREE_TO_KJMOL:<12.3f}")

        if self.Ecoll == None or self.bmax == None or self.Rini == None:
            raise ValueError("Ecoll (unless it's thermall sampled), bmax and Rini must be given in the input of Set_Relative_Init_Coords()")

        if self.bsampling == 0:
            print(f"Impact parameter is kept fix (not sampled): b = bmax")
            self.bimp = self.bmax #fix impact parameter for opacity function P(b) calculations
        elif self.bsampling == 1:
            self.bimp = self.bmax * random.uniform(0.0,1.0)
            print(f"Impact parameter sampled linearly: b = bmax * random[0,1]")
            print(f"Maximum of Impact parameter: bmax = {self.bmax*BOHR_TO_ANGSTROM:<12.2f} Angstrom")
            print(f"\nImpact parameter has been randomly sampled: b = {self.bimp*BOHR_TO_ANGSTROM:<12.2f} Angstrom")
        elif self.bsampling == 2:
            self.bimp = self.bmax * math.sqrt(random.uniform(0.0,1.0))
            print(f"Impact parameter sampled square: b = bmax * sqrt(random[0,1])")
            print(f"Maximum of Impact parameter: bmax = {self.bmax*BOHR_TO_ANGSTROM:<12.2f} Angstrom")
            print(f"\nImpact parameter has been randomly sampled: b = {self.bimp*BOHR_TO_ANGSTROM:<12.2f} Angstrom")
        else:
            raise ValueError("Wrong value chosen for bsampling. It can be bsampling = 0, 1 or 2")

        #Rini is the initial separation of center of masses
        #while sepx is the separation along the X-axis where the attack happens
        sepx = math.sqrt(self.Rini * self.Rini - self.bimp*self.bimp)

        #shift fragment B along the x-axis with sepx (separation along x-axis)
        #and shifted along the z-axis with bimp(impact paramter)
        for i in range(len(self.fragment_B.mass)):
            jx = 3 * i
            jy = 3 * i + 1
            jz = 3 * i + 2
            qB[jx] += sepx
            qB[jz] += self.bimp

        wA = self.fragment_A.totmass
        wB = self.fragment_B.totmass

        redmass = wA*wB/(wA+wB)

        #velocity is distributed in a center of mass system
        velRel = math.sqrt(2.0*self.Ecoll/redmass)
        velA = velRel*wB / (wA+wB)
        velB = velA - velRel

        #velocity measured along the X-axis
        for i in range(len(self.fragment_A.mass)):
            jx = 3 * i
            pA[jx] += velA * self.fragment_A.mass[i]

        for i in range(len(self.fragment_B.mass)):
            jx = 3 * i
            pB[jx] += velB * self.fragment_B.mass[i]

        self.q = np.append(qA, qB)
        self.p = np.append(pA, pB)
        self.redmass = redmass

        return 

    def set_relative_init_coords_molecule(self):
        return self.Set_Relative_Init_Coords_Molecule()


    def Set_Relative_Init_Coords_Surface(self):
        r'''
        The potato (molecule or surface) is aimed by the projectile
        First its center of mass will be put into the origin

        If the potato is a surface then it's needed to be
        oriented in a proper way before collision

        By convention, the normal vector of the surface plane
        is oriented to points toward the X-axis

                Z |
                  |
                  |
                  |************ <---P (projectile)
                  |                 |
              /\\\\\\\\             | bimp (b - impact param)
             /    |    \            |
            |     O-----|---------------------->
             \  /      /           sepx       X
              \\\\\\\\/
              /          sepx = sqrt(R^2 - b^2)
             /           R: distance between O-P
            /
           /
          /
         Y

        '''

        if self.sampling_set == False:
            raise ValueError("Error: First you must call Specify_Collision_Sampling() after you initialized the Collision() class")

        #shif the A and B molecule to their center of mass:
        qA, pA = cenmass(self.fragment_A.q, self.fragment_A.p, self.fragment_A.mass)
        qB, pB = cenmass(self.fragment_B.q, self.fragment_B.p, self.fragment_B.mass)


        if self.Ecoll_thermal:
            RT = R_GAS_HARTREE_PER_K * self.tempcoll
            self.Ecoll = thermal_collision_energy(RT) 
            print(f"temp[K]: {self.tempcoll:<12.2f}     Ecoll[kJ/mol]: {self.Ecoll*HARTREE_TO_KJMOL:<12.3f}")

        if self.Ecoll == None or self.bmax == None or self.Rini == None:
            raise ValueError("Ecoll (unless it's thermall sampled), bmax and Rini must be given in the input of Set_Relative_Init_Coords()")

        if self.Rini < self.bmax:
            raise ValueError("Error: Rini must be larger than b (impact paremter) for surface collisions")


        if self.bsampling == 0:
            print(f"Impact parameter is kept fix (not sampled): b = bmax")
            self.bimp = self.bmax #fix impact parameter for opacity function P(b) calculations
        elif self.bsampling == 1:
            self.bimp = self.bmax * random.uniform(0.0,1.0)
            print(f"Impact parameter sampled linearly: b = bmax * random[0,1]")
            print(f"Maximum of Impact parameter: bmax = {self.bmax*BOHR_TO_ANGSTROM:<12.2f} Angstrom")
            print(f"\nImpact parameter has been randomly sampled: b = {self.bimp*BOHR_TO_ANGSTROM:<12.2f} Angstrom")
        elif self.bsampling == 2:
            self.bimp = self.bmax * math.sqrt(random.uniform(0.0,1.0))
            print(f"Impact parameter sampled square: b = bmax * sqrt(random[0,1])")
            print(f"Maximum of Impact parameter: bmax = {self.bmax*BOHR_TO_ANGSTROM:<12.2f} Angstrom")
            print(f"\nImpact parameter has been randomly sampled: b = {self.bimp*BOHR_TO_ANGSTROM:<12.2f} Angstrom")
        else:
            raise ValueError("Wrong value chosen for bsampling. It can be bsampling = 0, 1 or 2")

        #**************************************************************************************
        #Random sampling of the skew angle for the projectile
        #-------------------------------------------------------
        if self.surf_skew_fix is None:
            theta_max = self.surf_skew_max

            ctmax = 1.0 - np.cos(theta_max)

            # Sample u uniformly in [0, 1] and invert the capped solid-angle CDF:
            # P(theta' < theta) = (1 - cos(theta)) / (1 - cos(theta_max)).
            # Using a variable drawn directly in [0, theta_max] distorts the incidence-angle distribution.
            rnd_skew = random.uniform(0.0, 1.0)
            theta_skew = np.arccos(1.0 - rnd_skew * ctmax)

        else: 
        # Fix skew angle for the projectile (given in degrees by the user).
            theta_skew = self.surf_skew_fix * math.pi / 180.0
        # Randomize the surface-frame azimuth and orient the surface normal.
        phi_yz_surf = random.uniform(0, 2 * math.pi)
        phi_yz_proj = random.uniform(0, 2 * math.pi)
        print("surface rotation angle: ", phi_yz_surf * 180.0 / math.pi)
        print("projectile rotation angle: ", phi_yz_proj * 180.0 / math.pi)

        lab_axis = np.array([1.0, 0.0, 0.0]) * self.surf_side
        if self.fragment_A.surface:
            qA, pA = orient_and_rotate_surface(
                self.fragment_A.surf_3atom, lab_axis, phi_yz_surf, qA, pA, self.fragment_A.mass
            )
        if self.fragment_B.surface:
            qB, pB = orient_and_rotate_surface(
                self.fragment_B.surf_3atom, lab_axis, phi_yz_surf, qB, pB, self.fragment_B.mass
            )

        # Build an orthogonal collision frame. ``approach`` is the direction
        # from the surface COM to the projectile COM; the transverse component
        # is perpendicular to it. Thus |r| = Rini and the impact parameter is
        # exactly bimp for normal and oblique incidence alike.
        yz_dir = np.array([0.0, np.cos(phi_yz_proj), np.sin(phi_yz_proj)])
        displacement, approach = surface_impact_geometry(
            self.Rini, self.bimp, theta_skew, lab_axis, yz_dir
        )

        if self.fragment_B.surface and not self.fragment_A.surface:
            print("Fragment B is the surface")
            print("theta_skew: ", theta_skew * 180.0 / math.pi)
            projectile_q, projectile_p = qA, pA
            projectile_mass = self.fragment_A.mass
            vel = math.sqrt(2.0 * self.Ecoll / self.fragment_A.totmass)
        elif self.fragment_A.surface and not self.fragment_B.surface:
            print("Fragment A is the surface")
            print("theta_skew: ", theta_skew * 180.0 / math.pi)
            projectile_q, projectile_p = qB, pB
            projectile_mass = self.fragment_B.mass
            vel = math.sqrt(2.0 * self.Ecoll / self.fragment_B.totmass)
        else:
            raise ValueError("Error: Fragment A or B must be a surface!!!")

        for i, atom_mass in enumerate(projectile_mass):
            component = slice(3 * i, 3 * i + 3)
            projectile_q[component] += displacement
            projectile_p[component] -= vel * approach * atom_mass


        self.q = np.append(qA, qB)
        self.p = np.append(pA, pB)

        return 

    def set_relative_init_coords_surface(self):
        return self.Set_Relative_Init_Coords_Surface()


    def reactants_actual_coordinate(self):
        natom_A = self.fragment_A.natom
        natom_B = self.fragment_B.natom

        qA = self.q[:3*natom_A]
        qB = self.q[3*natom_A:]

        return (qA, qB)

    def reactants_cenmass_distance(self):
        qA, qB = self.reactants_actual_coordinate()

        comA = self.center_of_mass(qA, self.fragment_A.mass)
        comB = self.center_of_mass(qB, self.fragment_B.mass)

        com_dist = np.linalg.norm(np.array(comA) - np.array(comB))
        return com_dist


    def Sample_Bimolecular_Reactants(self, **kwargs):
        #---------------------------------------------
        # Sample the internal motions of a fragment:
        #---------------------------------------------
        def sample_fragment(fragment):
            if fragment.natom == 1:
                fragment.atom_sampling()
            elif fragment.natom == 2:
                fragment.diatom_sampling(**kwargs)
            else:
                fragment.polyatom_sampling(**kwargs)
        #---------------------------------------------

        sample_fragment(self.fragment_A)
        sample_fragment(self.fragment_B)

        # Preserve per-fragment sampling decisions in the collision-level
        # metadata written by parallel trajectory runs.  This includes, for
        # example, the effective frequency used for a thermally sampled soft
        # vibrational mode.
        self.sampling_metadata = []
        for fragment_label, fragment in (("A", self.fragment_A), ("B", self.fragment_B)):
            for record in getattr(fragment, "sampling_metadata", ()):
                enriched_record = dict(record)
                enriched_record["fragment"] = fragment_label
                self.sampling_metadata.append(enriched_record)
        self.sampling_warnings = self.sampling_metadata


        if self.fragment_A.surface or self.fragment_B.surface:
            self.set_relative_init_coords_surface()
        else:
            self.set_relative_init_coords_molecule()

        self.overlap = check_atomic_overlap(self.atoms, self.q)

        if self.overlap:
            raise ValueError("Error: Overlap detected during the initialization of the bimolecular reaction!")

        self.q_collision_initial = np.array(self.q, copy=True)
        self.p_collision_initial = np.array(self.p, copy=True)
        update_collision_vector_state(self, "initial")

        return

    def sample_bimolecular_reactants(self, **kwargs):
        return self.Sample_Bimolecular_Reactants(**kwargs)

    def sample_and_run_collision(self, integrator='verlet', integrator_order=4, timestep=1.0, startstep=0, maxstep=100, iprint=2,
                                      traj_file=None, backfile=None, restart=False, pairs_to_stop=None, Rstop=None,
                                      reaction_persistence_steps=1,
                                      thermostat=None, thermo_param=None, thermo_temp=None, spectrum=False,
                                      constraint_algorithm="rattle", constraint_tolerance=1.0e-10,
                                      constraint_velocity_tolerance=1.0e-10,
                                      constraint_max_iterations=200, **kwargs):

        if traj_file is None:
            traj_file = 'traj_of_reaction_' + self.fragment_A.fname + '_+_' + self.fragment_B.fname + '.xyz'

        if backfile is None:
            backfile = 'backup_for_restart_of_reaction_' + self.fragment_A.fname + '_+_' + self.fragment_B.fname + '.xyz'

        if pairs_to_stop is None and Rstop is None:
            raise ValueError("\nEither Rstop or pairs_to_stop must be given in the input")

        post_collision_analysis = kwargs.pop("post_collision_analysis", True)
        post_collision_analysis_file = kwargs.pop("post_collision_analysis_file", None)
        post_collision_bond_th_HX = kwargs.pop("post_collision_bond_th_HX", 1.5)
        post_collision_bond_th_XX = kwargs.pop("post_collision_bond_th_XX", 2.0)
        post_collision_equilibrium_geometries = kwargs.pop("post_collision_equilibrium_geometries", None)
        post_collision_channel_states = kwargs.pop("post_collision_channel_states", None)
        post_collision_isolation_distance = kwargs.pop("post_collision_isolation_distance", 100.0)

        if not restart:
            self.sample_bimolecular_reactants(**kwargs)

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
            collision=True,
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
            post_collision_analysis=post_collision_analysis,
            post_collision_analysis_file=post_collision_analysis_file,
            post_collision_bond_th_HX=post_collision_bond_th_HX,
            post_collision_bond_th_XX=post_collision_bond_th_XX,
            post_collision_equilibrium_geometries=post_collision_equilibrium_geometries,
            post_collision_channel_states=post_collision_channel_states,
            post_collision_isolation_distance=post_collision_isolation_distance,
        )

    def post_collision_analysis(self, output_file=None, channel=None, formula=None, step=None, time_fs=None,
                                bond_th_HX=1.5, bond_th_XX=2.0, equilibrium_geometries=None,
                                channel_state=None, isolation_distance=100.0,
                                trajectory_initial_energy_hartree=None,
                                trajectory_final_energy_hartree=None):
        result = compute_collision_vector_correlations(
            self,
            output_file=output_file,
            channel=channel,
            formula=formula,
            step=step,
            time_fs=time_fs,
            bond_th_HX=bond_th_HX,
            bond_th_XX=bond_th_XX,
            equilibrium_geometries=equilibrium_geometries,
            channel_state=channel_state,
            isolation_distance=isolation_distance,
            trajectory_initial_energy_hartree=trajectory_initial_energy_hartree,
            trajectory_final_energy_hartree=trajectory_final_energy_hartree,
        )
        self.post_collision_analysis_result = result
        return result

    def parallel_traj_sample_and_run_collision(self, ntraj=None, total_trajectories=None, nparallel_jobs=None,
                                               max_parallel_jobs=None, slurm=False, nproc_per_job=None,
                                               cores_per_traj=None,
                                               integrator='verlet', integrator_order=4, timestep=1.0,
                                               startstep=0, maxstep=100, iprint=2, traj_file=None,
                                               backfile=None, restart=False, pairs_to_stop=None, Rstop=None,
                                               reaction_persistence_steps=1,
                                               thermostat=None, thermo_param=None, thermo_temp=None,
                                               spectrum=False, base_seed=None, **kwargs):
        from parallel.trajectory_runner import run_parallel_collisions

        if total_trajectories is None:
            total_trajectories = ntraj
        if nproc_per_job is None:
            nproc_per_job = 1 if cores_per_traj is None else cores_per_traj
        run_name = kwargs.pop("run_name", f"{self.fragment_A.fname}_+_{self.fragment_B.fname}")
        base_output_dir = kwargs.pop("base_output_dir", ".")
        scratch_base_dir = kwargs.pop("scratch_base_dir", "scratch")
        stop_on_error = kwargs.pop("stop_on_error", False)
        progress_report = kwargs.pop("progress_report", True)
        progress_mode = kwargs.pop("progress_mode", "auto")
        progress_interval = kwargs.pop("progress_interval", 2.0)

        if total_trajectories is None:
            total_cores = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count() or 1)) if slurm else (os.cpu_count() or 1)
            total_trajectories = max(1, total_cores // nproc_per_job)

        run_kwargs = dict(
            integrator=integrator,
            integrator_order=integrator_order,
            timestep=timestep,
            startstep=startstep,
            maxstep=maxstep,
            iprint=iprint,
            restart=restart,
            pairs_to_stop=pairs_to_stop,
            reaction_persistence_steps=reaction_persistence_steps,
            Rstop=Rstop,
            thermostat=thermostat,
            thermo_param=thermo_param,
            thermo_temp=thermo_temp,
            spectrum=spectrum,
            post_collision_analysis=kwargs.pop("post_collision_analysis", True),
            post_collision_analysis_file=kwargs.pop("post_collision_analysis_file", None),
            post_collision_bond_th_HX=kwargs.pop("post_collision_bond_th_HX", 1.5),
            post_collision_bond_th_XX=kwargs.pop("post_collision_bond_th_XX", 2.0),
            post_collision_equilibrium_geometries=kwargs.pop("post_collision_equilibrium_geometries", None),
            **kwargs,
        )
        if traj_file is not None:
            run_kwargs["traj_file"] = traj_file
        if backfile is not None:
            run_kwargs["backfile"] = backfile

        return run_parallel_collisions(
            collision_template=self,
            total_trajectories=total_trajectories,
            nparallel_jobs=nparallel_jobs,
            max_parallel_jobs=max_parallel_jobs,
            nproc_per_job=nproc_per_job,
            run_name=run_name,
            base_output_dir=base_output_dir,
            scratch_base_dir=scratch_base_dir,
            slurm=slurm,
            stop_on_error=stop_on_error,
            base_seed=base_seed,
            progress_report=progress_report,
            progress_mode=progress_mode,
            progress_interval=progress_interval,
            **run_kwargs,
        )

    def multi_parallel_traj_sample_and_run_collision(self, *args, **kwargs):
        return self.parallel_traj_sample_and_run_collision(*args, **kwargs)

    def paralell_traj_sample_and_run_collision(self, *args, **kwargs):
        return self.parallel_traj_sample_and_run_collision(*args, **kwargs)

    def multi_paralell_traj_sample_and_run_collision(self, *args, **kwargs):
        return self.parallel_traj_sample_and_run_collision(*args, **kwargs)
