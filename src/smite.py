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


from integrators.stormerverlet    import stormer_verlet
from integrators.leapfrog         import leapfrog
from integrators.verlet           import velverlet
from integrators.rungekutta       import rk4
from integrators.symplectic       import Symplectic
from integrators.sprk             import SPRK
from integrators.predcorr         import PredCorr 
from integrators.gradient         import Energy

from thermostats.randmomentum     import random_initialize_momenta
from thermostats.berendsen        import thermo_berendsen 
from thermostats.andersen         import thermo_andersen 
#from thermostats.nosehoover       import NoseHoover

class Molecule:
    def __init__(self, atoms=None, mass=None, q_ini=None, p_ini=None, nfix=0, restart=False, xyz_file_path=None):
        if restart:
            if not xyz_file_path:
                raise ValueError("Backup file must be provided when restart is True.")
            try:
                last_step, atoms, q_ini, p_ini = self.parseCheckPoint(xyz_file_path)
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

    def save_velocity_and_distance_matrix(self):
        #mass weighted velcity u = M^1/2 * v
        #each element of the list is mass weigthed velocity vector corresponding to a timestep
        self.vsave.append(self.p / np.sqrt(self.wmass)) 

        #qxyz = np.reshape(self.q, (-1, 3))

        #dist_matrix = np.linalg.norm(qxyz[:, np.newaxis, :] - qxyz[np.newaxis, :, :], axis=-1)

        # Extract the upper triangle of dist_matrix without the diagonal entries
        #upper_triangle_indices = np.triu_indices_from(dist_matrix, k=1)
        #dist_vector = dist_matrix[upper_triangle_indices]

        #each element of the list is the distance matrix corresponding to a timestep 
        #self.dsave.append(dist_vector)


    def vibrational_spectrum(self, dt, print_maxfreq=5000.0):
        import matplotlib.pyplot as plt
        import math

        c5=219474.0 #[Hartree]*c5=[cm-1]
        two_pi = 2.0 * math.pi

        freq_cm1 = None
        total_velo_power = None

        for i in range(len(self.p)):
            vthis = [v[i] for v in self.vsave]

            actual_length = len(vthis)

            velocity_fft = np.fft.rfft(vthis)
            velo_power = np.abs(velocity_fft)**2

        # Compute the corresponding frequencies (same for all components)
            if freq_cm1 is None:  # Compute only once
                freq_hz = np.fft.rfftfreq(actual_length, d=dt)
                freq_cm1 = freq_hz * c5 / two_pi

        # Accumulate the power spectrum
            if total_velo_power is None:
                total_velo_power = velo_power
            else:
                total_velo_power += velo_power

        plt.figure(figsize=(8, 5))
        plt.plot(freq_cm1, total_velo_power, label="FFT of Velocity")
        plt.xlim(0, print_maxfreq)  # Set x-axis limit
        plt.xlabel("Frequency (cm$^{-1}$)")
        plt.ylabel("Velocity Power Spectrum")
        plt.legend()
        plt.grid()

        file_to_save = self.fname + '_velocity_power_spectrum_' + '.png'
        plt.savefig(file_to_save, dpi=300, bbox_inches='tight')

        return

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
        bond_th_HX = kwargs.get('bond_th_HX', 1.4/0.5291772) # bond threshold in Angstrom for H-X, where X = any non H-atom
        bond_th_XX = kwargs.get('bond_th_XX', 2.0/0.5291772) # bond threshold in Angstrom for X-X bonds to be considered as a part of a fragment

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

    def stormer_single_step(self, dt):
        self.q, self.p = stormer_verlet(self.qchem, dt, self.wmass, self.q, self.p, self.atoms)

    def leapfrog_single_step(self, dt):
        self.q, self.p = leapfrog(self.qchem, dt, self.wmass, self.q, self.p, self.atoms)

    def verlet_single_step(self, dt):
        self.q, self.p = velverlet(self.qchem, dt, self.wmass, self.q, self.p, self.atoms)

    def rk4_single_step(self, dt):
        self.q, self.p = rk4(self.qchem, dt, self.wmass, self.q, self.p, self.atoms)

    def symplectic_single_step(self, dt, this_class):
        self.q, self.p = this_class.symplectic(self.qchem, dt, self.wmass, self.q, self.p, self.atoms)

    def sprk_single_step(self, dt, this_class):
        self.q, self.p = this_class.sprk(self.qchem, dt, self.wmass, self.q, self.p, self.atoms)

    def predcorr_single_step(self, dt, this_class):
        self.q, self.p = this_class.predcorr(self.qchem, dt, self.wmass, self.q, self.p, self.atoms)

    def thermo_berendsen(self, tau, dt, Ttarg):
        self.p = thermo_berendsen(self.nfix, self.p, self.wmass, dt, tau, Ttarg)

    def thermo_andersen(self, prob, dt, Ttarg):
        self.p = thermo_andersen(self.nfix, self.p, self.wmass, dt, prob, Ttarg)


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
                             spectrum=False):

        wf_dir = "wavefunction_along_trajectory"

        if traj_file is None:
            traj_file = 'traj_' + self.fname + '.xyz' 

        if backfile is None:
            backfile = 'backup_for_restart_' + self.fname + '.xyz'

        b2a = 0.52917721092

        if pairs_to_stop is not None and Rstop is None:
            self.pairstop = pairs_to_stop
        elif pairs_to_stop is None and Rstop is not None:
            self.Rstop = Rstop/b2a #Angstrom to Bohr #reactive event condition for trajectory 
        else:
            raise ValueError("\nEither Rstop or pairs_to_stop must be given in the input")

        #--------------------------------------------------------------------------------------
        if not restart:
            self.vsave = []
            print("\n********************************************************************************************")
            if os.path.exists(traj_file):
                os.remove(traj_file)
                print(f"{traj_file} already exisits, it has been deleted to create a new one")

            if self.qchem['wfu'] and os.path.exists(wf_dir) and os.path.isdir(wf_dir):
                shutil.rmtree(wf_dir)
                os.makedirs(wf_dir)
                print(f"\n{wf_dir} already exisits, it has been deleted to create a new one")
            elif self.qchem['wfu'] and not os.path.exists(wf_dir):
                os.makedirs(wf_dir)
                print(f"\n{wf_dir} created")


            print("********************************************************************************************\n")
        #--------------------------------------------------------------------------------------


        # Initial energy
        T0, V0, E0 = self.get_energy() 

        c5   = 219474.e0             # [Hartree]  * c5 = [cm-1]
        c6   = 41.341105             # [fs]       * c6 = [time in au]
        c7   = 2625.5                # [Hartree]  * c7 = [kJ/mol]

        dt = timestep * c6
        tstop = False
        Rcom_min = 1000000.0

        if collision and self.Rstop is not None:
            if self.Rstop < self.Rini:
                raise ValueError("Rstop must be larger than Rini")

        if integrator == 'predcorr':
            propag = PredCorr(integrator_order, len(self.q))
        if integrator == 'symplectic':
            propag = Symplectic(integrator_order)
        if integrator == 'sprk':
            propag = SPRK(integrator_order)

        with open(traj_file, "a") as file_trj:
            for istep in range(startstep, maxstep):

                #-----------------------------------------------------------------------------
                if (iprint > 0 and istep % iprint == 0 and istep > startstep) or istep == 0:

                    if self.qchem['wfu']:
                        file_wf = os.path.join(wf_dir, 'wavefunc_' + self.fname + '_step_' + str(istep) + '.molden')
                        T, V, E = self.get_energy(file_wf=file_wf) 
                    else:
                        T, V, E = self.get_energy() 

                    act_temp = self.traj_temperature()

                    dE = E - E0


                    self.print_trajectory(file_trj, istep, dt, T, V, dE, act_temp)
                    #print_trajectory(file_trj, self.atoms, self.q, self.p, V, dE, dt, istep)



                    if collision:
                        Rcom_actual = self.reactants_cenmass_distance()

                        if Rcom_actual < Rcom_min:
                            Rcom_min = Rcom_actual

                        if self.pairstop is not None and self.Rstop is None:
                            tstop, channel = test_to_stop_specific(q=self.q, pairs_to_test=self.pairstop)
                        elif self.pairstop is None and self.Rstop is not None:
                            tstop = test_to_stop_general(q=self.q, tol=self.Rstop)
                            channel = 'Not Specified'

                        print(f"step: {istep:<10d} t[fs]: {istep*dt/c6:<12.1f} V[Eh]: {(V-self.vref):<16.6f} E[Eh]: {(E-self.vref):<16.6f} dE[kJ]: {dE*c7:16.3f}    T[K]: {act_temp:10.1f} Rcom[A]: {Rcom_actual*b2a:8.2f}")

                    else: #if not collision (just unimolecular dynamics) then we can ran the test anytime
                        print(f"step: {istep:<10d} t[fs]: {istep*dt/c6:<12.2f}  V[Eh]: {V:<16.6f} E[Eh]: {E:<16.6f} dE[kJ]: {dE*c7:16.3f}     T[K]: {act_temp:<10.1f}")
                        if self.pairstop is not None and self.Rstop is None:
                            tstop, channel = test_to_stop_specific(q=self.q, pairs_to_test=self.pairstop)
                        elif self.pairstop is None and self.Rstop is not None:
                            tstop = test_to_stop_general(q=self.q, tol=self.Rstop)
                            channel = 'Not Specified'

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

                        print("\n Reactive event found: ", formula, "    Reaction channel: ", channel)
                        break
                #-----------------------------------------------------------------------------

                if integrator == 'stormer':
                    self.stormer_single_step(dt)
                elif integrator == 'verlet':
                    self.verlet_single_step(dt)
                elif integrator == 'leapfrog':
                    self.leapfrog_single_step(dt)
                elif integrator == 'rk4':
                    self.rk4_single_step(dt)
                elif integrator == 'predcorr':
                    self.predcorr_single_step(dt, propag)
                elif integrator == 'symplectic':
                    self.symplectic_single_step(dt, propag)
                elif integrator == 'sprk':
                    self.sprk_single_step(dt, propag)
                else:
                    raise ValueError("Non existing integrator. You can choose from: leapfrog, verlet, rk4, symplectic(4,6,8) and predcorr(order)")

                if thermostat is not None and (thermo_param or thermo_temp) is None:
                    raise ValueError("Since thermostate switched on the parameter and temperature must be given")
                if thermostat == 'berendsen':
                    tau = thermo_param * 41.341105 #from fs to atomic time unit
                    self.thermo_berendsen(tau, dt, thermo_temp)
                if thermostat == 'andersen':
                    self.thermo_andersen(thermo_param, dt, thermo_temp)

                if spectrum and thermostat is None:
                    self.save_velocity_and_distance_matrix()


                #--------------------------------------------------------------------------
                #Backup:
                backup_file=open(backfile,'w')
                self.print_trajectory(backup_file, istep, dt, T, V, dE, act_temp)
                #print_trajectory(backup_file, self.atoms, self.q, self.p, V, dE, dt, istep)
                backup_file.close()
                #--------------------------------------------------------------------------

                if istep == (maxstep-1):
                    print("\n Propgation time has reached the maximum number of steps")


    def restart_init(self, fname, integrator='verlet', timestep=1.0, startstep=0, maxstep=100, iprint=2, traj_file=None, backfile=None, restart=False):
        self.fname = fname

        if backfile is None:
            raise ValueError('backup file name must be given in the input')

        if traj_file is None:
            traj_file = 'traj_' + self.fname + '.xyz'


        if restart:
            try:
                last_step, self.atoms, self.q, self.p = parseCheckPoint(backfile)
                self.mass = get_mass_vector(atoms)
                startstep = last_step
            except FileNotFoundError:
                raise ValueError("!!!!!!!!!!!!!!! Backup file does not exist !!!!!!!!!!!!!!!!!!!")
        else:
            raise ValueError("Wrong sampling option in sample_and_run_trajectory() function")

    def traj_temperature(self):
        '''
        Actual temperature of the system

        N*R*T/2 = Ekin

        where N is the number of degrees of freedom of the system

        nfix is used for the number of the fixeddegrees of freedom
        (e.g. nfix=3, if translational modes are frozen)
        '''
        Rgas = (8.3144598/1000.0/2625.5) #in Hartree/K

       #kinetic energy of the system
        Ekin=sum(self.p*self.p/self.wmass)*0.5

        return 2.0*Ekin/float(len(self.p)-self.nfix)/Rgas

    def merge_with(self, other_molecule):
        if not isinstance(other_molecule, Molecule):
            raise ValueError("other_fragment must be an instance of Molecule or Fragment")
        
        combined_atoms = np.concatenate([self.atoms, other_molecule.atoms])
        combined_mass = np.concatenate([self.mass, other_molecule.mass])
        combined_coord_eq = np.concatenate([self.q, other_molecule.q])
        combined_mom = np.concatenate([self.p, other_molecule.p])
        
        # Create a new instance with combined attributes
        new_fragment = Fragment(combined_atoms, combined_mass, combined_coord_eq, combined_mom)

        return new_fragment

    def print_structure(self, trajfile, igeom):
        b2a = 0.52917721092
        trajfile.write(str(self.natom) + "\n")
        trajfile.write("%8s %10d \n" % ("geom = ", igeom ))
        for i in range(self.natom):
            jx = 3*i
            jy = 3*i+1
            jz = 3*i+2
            trajfile.write("%3s %15.5f  %15.5f %15.5f \n" % (self.atoms[i], self.q[jx]*b2a, self.q[jy]*b2a, self.q[jz]*b2a))


    def print_trajectory(self, trajfile, istep, dt, T, V, dE, Temp):
        b2a = 0.52917721092
        c7 = 2625.5  #[Hartree]*c7=[kJ/mol] 
        c5 = 219474.0  #[Hartree]*c5=[cm-1]
        c6 = 41.341105 #[femto-sec]*c6=[time in au]
        trajfile.write(str(self.natom) + "\n")
        trajfile.write("%6s %10d %8s %10.2f %12s %16.8f %12s %12.8f %10s %15.4f %8s %7.1f\n" %
                       ("step= ",istep, " t[fs]= ", dt*istep/c6, " Vpot[Eh]= ", V, " Tkin[Eh]= ", T, " dE[kJ]= ",dE*c7, " T[K]= ", Temp ))
        for i in range(self.natom):
            jx = 3*i
            jy = 3*i+1
            jz = 3*i+2
            trajfile.write("%3s %15.5f  %15.5f %15.5f  %16.6f  %16.6f %16.6f\n" % (self.atoms[i], self.q[jx]*b2a, self.q[jy]*b2a, self.q[jz]*b2a, self.p[jx], self.p[jy], self.p[jz]))

    def delete_orca_tmp(self):
        import os
        import shutil

        folder_name = 'orca_tmp'

        if os.path.exists(folder_name) and os.path.isdir(folder_name):
            shutil.rmtree(folder_name)



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
    def Diatom_Init(cls, fname, atoms, req=None, omega=None, alpha=None, De=None, rigid=False, random_rot=True, diatom='harmonic', nfix=0):
        if len(atoms) != 2:
            raise ValueError("ERROR: Diatom_Init accept only a diatomic molecule.")

        req = req/0.52917721092 #Angstrom to Bohr
  
        if req is None: 
            raise ValueError("ERROR: req must be given in Diatom_Init()")
        elif (req is None and omega is None) and not rigid and diatom == 'harmonic': 
            raise ValueError("ERROR: For non-rigid harmonic diatom req and omega must be given in Diatom_Init()")
        elif (req is None and alpha is None and De is None) and not rigid and diatom == 'morse': 
            raise ValueError("ERROR: For non-rigid Morse diatom req, alpha, De must be given in Diatom_Init()")

        if rigid:
            nfix = 1

        c9=1.0e8/0.5291772e0
        c10=137.035999074

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
        this.omega_diat = omega*c10/c9*(math.pi * 2) #from cm-1 to atomic unit
        this.alpha_diat = alpha
        this.De_diat = De
        this.freq = [this.omega_diat]
        this.rigid  = rigid
        this.rigid  = diatom
        this.random_rot = random_rot
        this.nvib = 0
        this.jvib = 0
        this.vibsampling = None
        this.rotsampling = None
        this.nfix = nfix

        return this 

    @classmethod
    def Polyatom_Init(cls, fname, qchem, xyz, nfix=0, linear=False, is_eckart=True, random_rot=True,
            rigid=False, fromMD=False, surface=False, surf_3atom=None, Amp_modeanim=30.0, print_nmode=True):

        natom, atoms, q_eq = parseXYZ(xyz)
        q_eq = np.array(q_eq) / 0.52917721092  #Angstrom to Bohr
        p_ini = np.zeros(len(q_eq))

        if len(atoms) <= 2:
            raise ValueError("ERROR: PolyatomInit requires more than two atoms.")

        nfix = 0
        if rigid:
            nfix = 3 * natom - 6 + linear

        mass = get_mass_vector(atoms)

        this = cls(atoms=atoms, mass=mass, q_ini=q_eq, p_ini=p_ini)

        this.fname      = fname
        this.qchem      = qchem
        this.rigid      = rigid
        this.surface    = surface
        this.surf_3atom = surf_3atom 
        this.random_rot = random_rot
        this.linear     = linear
        this.nfix       = nfix
        this.fromMD     = fromMD
        if surface:
            this.random_rot = False
            this.linear     = False 
       #--------------------------------------------------------------------
        if not rigid and not fromMD:
            hessFile = 'hessian_' + fname + '.hess'
            hessian = getHessian(qcinput=qchem, hessFile=hessFile, xyz=xyz)

            if print_nmode:
                freq, freq_low, Lmat = print_normalmode(fname=fname, atoms=atoms, mass=mass, q_eq=q_eq, hessian=hessian,
                                                    give_freq_and_Lmat=True, Amp=Amp_modeanim, is_eckart=is_eckart, linear=linear)
            else:
                freq, freq_low, Lmat = getNormalmode(mass=mass, hessian=hessian, linear=linear, q_eq=q_eq, is_eckart=is_eckart)

            freq_all = np.append(freq_low, freq)

            print_frequencies(fname,freq_all)

            this.hessFile   = hessFile
            this.hessian    = hessian
            this.freq       = freq
            this.Lmat       = Lmat

            if qchem['qchem'] == 'Orca':
                this.delete_orca_tmp()

        return this


    def Atom_Sampling(self):
        if len(self.atoms) != 1:
            raise ValueError("ERROR: Atom_Sampling requires only a single atom.")

        self.q = np.array([0.0, 0.0 , 0.0])
        self.p = np.array([0.0, 0.0 , 0.0])
        return

    def Diatom_Sampling(self, **kwargs):
        if len(self.atoms) != 2:
            raise ValueError("ERROR: DiatomHaromicInit accept only a diatomic molecule.")

        verbosity = kwargs.get('verbosity', False)
        traj_index = kwargs.get('traj_index', -999)

        redmass = self.mass[0]*self.mass[1] / (self.mass[0] + self.mass[1])

        if self.rigid == False:
            self.q, self.p = diatom_vibration_harmonic_sampling(vib_modes=self.vibsampling, req=self.req_diat, omega=self.omega_diat, mass=self.mass) 

        self.q, self.p = cenmass(self.q, self.p, self.mass)

        self.p, angmom, inertia = diatom_rotation_rigidrot_sampling(rot_modes=self.rotsampling, mass=self.mass, q=self.q, p=self.p)

        self.inertia  = inertia
        self.angmom   = angmom

        if self.random_rot == True:
            self.q, self.p = euler_rot(self.q, self.p)
        return


    def Specify_Mode_Sampling(self, init_vib_type='ZPE', init_rot_type='Jfix', MDfile=None, MDprimitive=False, **kwargs):
        temp = kwargs.get('temp', None)
        nvib = kwargs.get('nvib', None)
        energy = kwargs.get('energy', None)
        jrot = kwargs.get('jrot', None)

        #-----------------------------------------------------------------------------------
        #Non-rigid Diatom case:
        if not self.rigid and len(self.atoms) == 2:
            if 'nvib' in kwargs:
                self.vibsampling[0] = (self.omega_diat, 'Q', nvib) 
            elif 'energy' in kwargs:
                self.vibsampling[0] = (self.omega_diat, 'E', energy) 
            elif 'temp' in kwargs:
                self.vibsampling[0] = (self.omega_diat, 'T', temp) 

        #-----------------------------------------------------------------------------------
        #Non-rigid Polyatom case Harmonic Sampling:
        if not self.rigid and not self.fromMD and len(self.atoms) > 2:
           #uniformly initialize all modes (w.r.t to ZPE or Temperature):
            self.vibsampling = initialize_vibrational_modes(freq=self.freq, init_vib_type=init_vib_type, temp=temp)

           #if necessary then we may change certain modes sampling
            if {'fix_quantum', 'fix_energy', 'fix_temp'}.intersection(kwargs):
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


    def sample_and_run_trajectory(self, integrator='verlet', integrator_order=4, timestep=1.0, startstep=0, maxstep=100, iprint=1,
                                  traj_file=None, backfile=None, Rstop=None, pairs_to_stop=None,
                                  thermostat=None, thermo_param=None, thermo_temp=None, spectrum=False):

        if traj_file is None:
            traj_file = 'traj_' + self.fname + '.xyz' 
        if backfile is None:
            backfile = 'backup_for_restart_' + self.fname + '.xyz'

        #---------------------------------------------
        # Sample the internal motions of a fragment:
        #---------------------------------------------
        if self.natom == 1:
            self.Atom_Sampling()
        elif self.natom == 2:
            self.Diatom_Sampling()
        elif self.natom > 2:
            self.Polyatom_Sampling()
        else:
            raise ValueError("Wrong sampling option in sample_and_run_trajectory() function")
        #---------------------------------------------

        if pairs_to_stop is None and Rstop is None:
            raise ValueError("\nEither Rstop or pairs_to_stop must be given in the input")


        self.run_trajectory(integrator=integrator, integrator_order=integrator_order, timestep=timestep, startstep=startstep, maxstep=maxstep,
                            iprint=iprint, traj_file=traj_file,  backfile=backfile, restart=False,
                            Rstop=Rstop, pairs_to_stop=pairs_to_stop, thermostat=thermostat, thermo_param=thermo_param, thermo_temp=thermo_temp, spectrum=spectrum)


    def print_mode_sampling(self):
        print(f"\n{self.fname}")
        print("Vibrational mode sampling (Q: fixed quantum, E: fixed energy, T: thermal)")
        print("mode freq   sampl  quantum")
        for i in self.vibsampling:
            print(i,":",self.vibsampling[i])
        print(f"\nRotational mode sampling (Q: fixed quantum, T: thermal)")
        print("mode  sampl  quantum")
        for i in self.rotsampling:
            print(i,":",self.rotsampling[i])
        print()

    def Polyatom_Sampling(self, **kwargs):
        verbosity = kwargs.get('verbosity', False)
        traj_index = kwargs.get('traj_index', -999)

        if len(self.atoms) <= 2:
            raise ValueError("ERROR: Polyatom_Sample requires more than two atoms.")

        if not self.rigid and not self.fromMD and not self.MDprimitive:
            self.q, self.p = polyatom_vibration_sampling(mass=self.mass, atoms=self.atoms, q_eq=self.q_ini, ww=self.freq, L=self.Lmat,
                                                   vib_modes=self.vibsampling, verbosity=verbosity, traj_index=traj_index)

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

            #here we should add the vibration too
            evib = 0.0
            erot = sum([angmom[i]**2 / inertia[i]/2.0 for i in range(len(angmom))])

            self.erot     = erot
            self.vib      = evib #the vibrational that Evib = Etot - Erot
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

        if self.fromMD is None:
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



class Collision(Molecule):
    def __init__(self, fragment_A, fragment_B, qchem):
        atoms = fragment_A.atoms + fragment_B.atoms
        mass = np.append(fragment_A.mass, fragment_B.mass)
        q_ini = np.append(fragment_A.q_ini, fragment_B.q_ini)
        p_ini = np.append(fragment_A.p_ini, fragment_B.p_ini)

        super().__init__(atoms=atoms, mass=mass, q_ini=q_ini, p_ini=p_ini)
        self.qchem = qchem
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


    def Specify_Collision_Sampling(self, Rini=None, bmax=None, bsampling=False, Ecoll=None, Ecoll_thermal=False, temp=None,
                                   surf_skew_max=180.0, surf_skew_fix=None, surf_target_atom=None, surf_side=1):

        if Rini == None or bmax == None or (Ecoll == None and Ecoll_thermal==False) or (Ecoll_thermal==True and temp==None):
            raise ValueError("Rini, bmax and Ecoll (or Ecoll_thermal) must be give in the input of Specify_Collision_Sampling()")

        if Rini != None:
            Rini = Rini/0.52917721092 #from Angstrom to Bohr
        if bmax != None:
            bmax = bmax/0.52917721092
        if Ecoll != None:
            Ecoll = Ecoll/2625.5 #from kJ/mol to Hartree
        

        self.Rini = Rini
        self.bmax = bmax
        self.Ecoll = Ecoll
        self.bsampling = bsampling
        self.Ecoll_thermal = Ecoll_thermal
        self.tempcoll = temp
        self.sampling_set = True
        self.surf_skew_fix = surf_skew_fix * math.pi / 180.0
        self.surf_skew_max = surf_skew_max * math.pi / 180.0
        self.surf_target_atom = surf_target_atom
        self.surf_side = surf_side

    def Set_Relative_Init_Coords(self):
        '''
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
                 /
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
            Rgas = (8.3144598/1000.0/2625.5) #in Hartree/K
            RT = Rgas * self.tempcoll
            self.Ecoll = thermal_collision_energy(RT) 
            print(f"temp[K]: {self.tempcoll:<12.2f}     Ecoll[kJ/mol]: {self.Ecoll*2625.5:<12.3f}")

        if self.Ecoll == None or self.bmax == None or self.Rini == None:
            raise ValueError("Ecoll (unless it's thermall sampled), bmax and Rini must be given in the input of Set_Relative_Init_Coords()")

        if self.bsampling:
            self.bimp = self.bmax * math.sqrt(random.uniform(0.0,1.0))
        else:
            self.bimp = self.bmax #fix impact parameter for opacity function P(b) calculations

        #**************************************************************************************
        #Either of the fragments is a surface:
        #**************************************************************************************
        if self.fragment_A.surface or self.fragment_B.surface:
            #-------------------------------------------------------
            if self.surf_skew_fix is None:
            #Random sampling of the skew angle for the projectile
                theta_max = self.surf_skew_max

                tmax = 1.0 - np.cos(theta_max)

                rnd_skew = random.uniform(0, theta_max)

                theta_skew = np.arccos(1.0 - rnd_skew * tmax)
            else: 
            #Fix skew angle for the projectile
                theta_skew = self.surf_skew_fix
           #-------------------------------------------------------
            #rotate with random angle in the yz-plane
            phi_yz = random.uniform(0, 2 * math.pi)
           #-------------------------------------------------------

            #rotate the surface into the Y-Z plane (the normalvector of surface points toward X)
            #self.surf_side to chose which side of the surface (by default self.surf_side = 1)
            lab_axis = np.array([1.0, 0.0, 0.0]) * self.surf_side # X-axis

            if self.fragment_A.surface:
                surf_3atom = self.fragment_A.surf_3atom  #3 atoms that define the surface

                qA, pA = orient_and_rotate_surface(surf_3atom, lab_axis, phi_yz, self.fragment_A.q, self.fragment_A.p, self.fragment_A.mass)

            if self.fragment_B.surface:
                surf_3atom = self.fragment_B.surf_3atom #3 atoms that define the surface

                qB, pB = orient_and_rotate_surface(surf_3atom, lab_axis, phi_yz, self.fragment_B.q, self.fragment_B.p, self.fragment_B.mass)
        #**************************************************************************************

        #Rini is the initial separation of center of masses
        #while sepx is the separation along the X-axis where the attack happens
        sepx = math.sqrt(self.Rini * self.Rini - self.bimp*self.bimp)

        wA = self.fragment_A.totmass
        wB = self.fragment_B.totmass

        redmass = wA*wB/(wA+wB)

        #velocity is distributed in a center of mass system
        velRel = math.sqrt(2.0*self.Ecoll/redmass)
        velA = velRel*wB / (wA+wB)
        velB = velA - velRel

        #shift fragment B along the x-axis with sepx (separation along x-axis)
        #and shifted along the z-axis with bimp(impact paramter)
        if self.fragment_B.surface: #in case of fragment-B is the surface then the smaller projectile is shifted 
            for i in range(len(self.fragment_B.mass)):
                jx = 3 * i
                jy = 3 * i + 1
                jz = 3 * i + 2
                qA[jx] += sepx
                qA[jz] += self.bimp
        else: #in case of a normal molecule-molecule scattering or when fragment-A is the surface:
            for i in range(len(self.fragment_B.mass)):
                jx = 3 * i
                jy = 3 * i + 1
                jz = 3 * i + 2
                qB[jx] += sepx
                qB[jz] += self.bimp


        #it has only velocity along the X-axis
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
                fragment.Atom_Sampling()
            elif fragment.natom == 2:
                fragment.Diatom_Sampling(**kwargs)
            else:
                fragment.Polyatom_Sampling(**kwargs)
        #---------------------------------------------

        sample_fragment(self.fragment_A)
        sample_fragment(self.fragment_B)


        self.Set_Relative_Init_Coords()

        self.overlap = check_atomic_overlap(self.atoms, self.q)

        if self.overlap:
            raise ValueError("Error: Overlap detected during the initialization of the bimolecular reaction!")

        return

    def sample_and_run_collision(self, integrator='verlet', integrator_order=4, timestep=1.0, startstep=0, maxstep=100, iprint=2,
                                      traj_file=None, backfile=None, restart=False, pairs_to_stop=None, Rstop=None, spectrum=False, **kwargs):

        if traj_file is None:
            traj_file = 'traj_of_reaction_' + self.fragment_A.fname + '_+_' + self.fragment_B.fname + '.xyz'

        if backfile is None:
            backfile = 'backup_for_restart_of_reaction_' + self.fragment_A.fname + '_+_' + self.fragment_B.fname + '.xyz'

        if pairs_to_stop is None and Rstop is None:
            raise ValueError("\nEither Rstop or pairs_to_stop must be given in the input")


        self.Sample_Bimolecular_Reactants(**kwargs)

        self.run_trajectory(integrator=integrator, integrator_order=integrator_order, timestep=timestep, startstep=startstep, maxstep=maxstep,
                            iprint=iprint, traj_file=traj_file,  backfile=backfile, restart=restart, collision=True, Rstop=Rstop, pairs_to_stop=pairs_to_stop, spectrum=spectrum)

