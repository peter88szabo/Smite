import numpy as np
import os
import math

from utils.cenmass                import cenmass
from utils.euler                  import euler_rot          
from utils.format_and_print       import parseCheckPoint
from utils.format_and_print       import parseXYZ
from utils.format_and_print       import print_trajectory
from utils.atomic_overlap         import check_atomic_overlap
from utils.atomic_masses          import get_mass_vector 
from utils.clustering             import cluster_chemical_formulas

from normalmode.hessian           import getHessian
from normalmode.eckart            import eckart_transform
from normalmode.nmodeprint        import print_normalmode 
from normalmode.normalmode        import print_frequencies 

from sampling.polyvibration       import initialize_vibrational_modes
from sampling.polyrotation        import initialize_rotational_modes
from sampling.polyvibration       import polyatom_vibration_sampling
from sampling.polyvibration       import specify_vib_modes 
from sampling.polyrotation        import polyatom_rotation_sampling
from sampling.thermal             import thermal_collision_energy
from sampling.diatom              import diatom_rotation_rigidrot_sampling
from sampling.diatom              import diatom_vibration_harmonic_sampling 


from integrators.integrators      import velverlet
from integrators.gradient         import Energy

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
        self.qchem      = None 
        self.last_step  = last_step
        self.Rstop      = None 


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
    def get_energy(self):
        file_wf = "wavevfuntion" #or it should be given as class variable from self.fname
        return Energy(self.qchem, file_wf, self.q, self.p, self.atoms, self.wmass)

    def verlet_single_step(self, dt):
        self.q, self.p = velverlet(self.qchem, dt, self.wmass, self.q, self.p, self.atoms)

    def run_trajectory(self, integrator='verlet', timestep=1.0, startstep=0, maxstep=100,
                       iprint=2, traj_file='trajectory.xyz', backfile="checkpoint.xyz", restart=False, collision=False, Rstop=10.0):
        file_wf = "wavevfuntion" #or it should be given as class variable from self.fname

        b2a = 0.52917721092
        
        self.Rstop = Rstop/b2a #Angstrom to Bohr #reactive event condition for trajectory 

        #--------------------------------------------------------------------------------------
        if not restart:
            print("\n*************************************************************************")
            if os.path.exists(traj_file):
                os.remove(traj_file)
                print(f"{traj_file} already exisits, it has been deleted to create a new one.")
            print("***************************************************************************")
        #--------------------------------------------------------------------------------------


        # Initial energy
        T0, V0, E0 = self.get_energy() 

        c5   = 219474.e0             # [Hartree]  * c5 = [cm-1]
        c6   = 41.341105             # [fs]       * c6 = [time in au]
        c7   = 2625.5                # [Hartree]  * c7 = [kJ/mol]

        dt = timestep * c6
        tstop = False
        Rcom_min = 1000000.0

        with open(traj_file, "a") as file_trj:
            for istep in range(startstep, maxstep):

                #-----------------------------------------------------------------------------
                if (iprint > 0 and istep % iprint == 0 and istep > startstep) or istep == 0:
                    T, V, E = self.get_energy() 

                    act_temp = self.traj_temperature()

                    dE = E - E0


                    print_trajectory(file_trj, self.atoms, self.q, self.p, V, dE, dt, istep)



                    if collision:
                        Rcom_actual = self.reactants_cenmass_distance()

                        print(f"step: {istep:<10d} t[fs]: {istep*dt/c6:<12.2f}  V[Eh]: {V:<15.5f} E[Eh]: {E:<15.5f} dE[cm-1]: {dE*c5:17.3f} T[K]: {act_temp:<10.2f} Rcom[A]: {Rcom_actual*b2a:8.2f}")

                        if Rcom_actual < Rcom_min:
                            Rcom_min = Rcom_actual

                        if Rcom_actual < 0.8*self.Rini: #preventing the intiail detection of the two reactants as reactive event
                            tstop = self.test_to_stop(tol=self.Rstop)
                    else: #if not collision (just unimolecular dynamics) then we can ran the test anytime
                        print(f"step: {step:<10d} t[fs]: {istep*dt/c6:<12.2f}  V[Eh]: {V:<15.5f} E[Eh]: {E:<15.5f} dE[cm-1]: {dE*c5:<17.3f} T[K]: {act_temp:<10.2f}")
                        tstop = self.test_to_stop(tol=self.Rstop)


                    if tstop:
                        #minPts: minum number of points to be a cluster
                        #eps: in Angstrom the tolerance within can be considered something as cluster

                        formula = cluster_chemical_formulas(self.q, self.atoms, eps=2.1, minPts=2)

                        print("\n Reactive event found: ", formula)
                        break
                #-----------------------------------------------------------------------------

                if integrator == 'verlet':
                    self.verlet_single_step(dt)
                else:
                    raise ValueError("Only Verlet integrator is avaiable at the moment")

                #--------------------------------------------------------------------------
                #Backup:
                backup_file=open(backfile,'w')
                print_trajectory(backup_file, self.atoms, self.q, self.p, V, dE, dt, istep)
                backup_file.close()
                #--------------------------------------------------------------------------

                if istep == (maxstep-1):
                    print("\n Propgation time has reached the maximum number of steps")


    def restart_init(self, integrator='verlet', timestep=1.0, startstep=0, maxstep=100, iprint=2, traj_file='trajectory.xyz', backfile="checkpoint.xyz", restart=False):
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

    def test_to_stop(self, tol=16.0):
        coord = np.reshape(self.q, (-1, 3))

        #distance matrix:
        dist_mat = np.sqrt(np.sum((coord[:, np.newaxis, :] - coord[np.newaxis, :, :]) ** 2, axis=-1))

        too_large = np.any(dist_mat[np.tril_indices(dist_mat.shape[0], -1)] > tol)

        return too_large



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


    def print_trajectory(self, trajfile, igeom):
        b2a = 0.52917721092
        trajfile.write(str(self.natom) + "\n")
        trajfile.write("%8s %10d \n" % ("geom = ", igeom ))
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
        self.fname        = None
        self.hessFile     = None
        self.vibsampling  = None
        self.rotsampling  = None
        self.linear       = None
        self.req_diat     = None
        self.omega_diat   = None
        self.overlap      = None
        self.rigid        = False
        self.random_rot   = False
        

    @classmethod
    def Atom_Init(cls, atoms):
        if len(atoms) != 1:
            raise ValueError("ERROR: Atom_Init requires only a single atom. You must provide an array with a single element, for instance: ['Cl']")

        q_ini = np.array([0.0, 0.0 , 0.0])
        p_ini = np.array([0.0, 0.0 , 0.0])

        mass = get_mass_vector(atoms)

        return cls(atoms=atoms, mass=mass, q_ini=q_ini, p_ini=p_ini)

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
    def Polyatom_Init(cls, fname, qchem, xyz, nfix=0, linear=False, is_eckart=True, random_rot=True, rigid=False, Amp_modeanim=30.0):

        natom, atoms, q_eq = parseXYZ(xyz)
        q_eq = np.array(q_eq) / 0.52917721092  #Angstrom to Bohr
        p_ini = np.zeros(len(q_eq))

        if len(atoms) <= 2:
            raise ValueError("ERROR: PolyatomInit requires more than two atoms.")

        nfix = 0
        if rigid:
            nfix = 3 * natom - 6 + linear


        mass = get_mass_vector(atoms)

        hessFile = 'hessian_' + fname + '.hess'

        hessian = getHessian(qcinput=qchem, hessFile=hessFile, xyz=xyz)

        freq, freq_low, Lmat = print_normalmode(fname=fname, atoms=atoms, mass=mass, q_eq=q_eq, hessian=hessian,
                                                give_freq_and_Lmat=True, Amp=Amp_modeanim,
                                                is_eckart=is_eckart, linear=linear)

        freq_all = np.append(freq_low, freq)

        print_frequencies(fname,freq_all)

        this = cls(atoms=atoms, mass=mass, q_ini=q_eq, p_ini=p_ini)

        this.fname      = fname
        this.hessFile   = hessFile
        this.hessian    = hessian
        this.freq       = freq
        this.Lmat       = Lmat
        this.qchem      = qchem
        this.linear     = linear
        this.rigid      = rigid
        this.random_rot = random_rot
        this.nfix       = nfix
        
        if qchem['qchem'] == 'Orca':
            self.delete_orca_tmp()

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


    def Specify_Mode_Sampling(self, init_vib_type='ZPE', init_rot_type='Jfix', **kwargs):
        temp = kwargs.get('temp', None)
        jrot = kwargs.get('jrot', None)
        nvib = kwargs.get('nvib', None)
        energy = kwargs.get('energy', None)

        #Non-rigid Diatom case:
        if not self.rigid and len(self.atoms) == 2:
            if 'nvib' in kwargs:
                self.vibsampling[0] = (self.omega_diat, 'Q', nvib) 
            elif 'energy' in kwargs:
                self.vibsampling[0] = (self.omega_diat, 'E', energy) 
            elif 'temp' in kwargs:
                self.vibsampling[0] = (self.omega_diat, 'T', temp) 

        #Non-rigid Polyatom case:
        elif not self.rigid and len(self.atoms) > 2:
           #uniformly initialize all modes (w.r.t to ZPE or Temperature):
            self.vibsampling = initialize_vibrational_modes(freq=self.freq, init_vib_type=init_vib_type, temp=temp)

           #if necessary then we may change certain modes sampling
            if {'fix_quantum', 'fix_energy', 'fix_temp'}.intersection(kwargs):
                self.vibsampling = specify_vib_modes(vib_modes=self.vibsampling, **kwargs)

        self.rotsampling = initialize_rotational_modes(init_rot_type=init_rot_type, temp=temp, jrot=jrot)
        return


    def sample_and_run_trajectory(self, integrator='verlet', timestep=1.0, startstep=0, maxstep=100, iprint=1,
                                  traj_file='trajectory.xyz', backfile="checkpoint.xyz", Rstop=10.0):
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

        self.run_trajectory(integrator=integrator, timestep=timestep, startstep=startstep, maxstep=maxstep,
                            iprint=iprint, traj_file=traj_file,  backfile=backfile, restart=False, Rstop=Rstop)


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

        if self.rigid == False:
            self.q, self.p = polyatom_vibration_sampling(mass=self.mass, atoms=self.atoms, q_eq=self.q_ini, ww=self.freq, L=self.Lmat,
                                                   vib_modes=self.vibsampling, verbosity=verbosity, traj_index=traj_index)


        self.q, self.p = cenmass(self.q, self.p, self.mass)

        #coordinate (self.q) does not change when we dress up the molecule with an angular momentum to rotate
        self.p, angmom, inertia = polyatom_rotation_sampling(rot_modes=self.rotsampling, mass=self.mass, q=self.q, p=self.p)

       #here we should add the vibration too
        evib = 0.0
        erot = sum([angmom[i]**2 / inertia[i]/2.0 for i in range(len(angmom))])

        self.erot     = erot
        self.vib      = evib #the vibrational that Evib = Etot - Erot
        self.inertia  = inertia
        self.angmom   = angmom


       #randomly rotate the molecule about its center of mass
        if self.random_rot == True:
            self.q, self.p = euler_rot(self.q, self.p)

        self.p = np.array(self.p)
        self.q = np.array(self.q)

        self.overlap = check_atomic_overlap(self.atoms, self.q)

        if self.overlap:
            raise ValueError("Overlap detected when the polyatomic fragment is sampled")

        return


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

    def Specify_Collision_Sampling(self, Rini=None, bmax=None, bsampling=False, Ecoll=None, Ecoll_thermal=False, temp=None):

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
    
    def Set_Relative_Init_Coords(self):

        if self.sampling_set == False:
            raise ValueError("Error: First you must call Specify_Collision_Sampling() after you initialized the Collision() class")


        #shif the A and B molecule to their center of mass:
        qA, pA = cenmass(self.fragment_A.q, self.fragment_A.p, self.fragment_A.mass)
        qB, pB = cenmass(self.fragment_B.q, self.fragment_B.p, self.fragment_B.mass)


        if self.Ecoll_thermal:
            Rgas = (8.3144598/1000.0/2625.5) #in Hartree/K
            RT = Rgas * self.tempcoll
            self.Ecoll = thermal_collision_energy(RT) 
            print("temp[K], Ecoll[kJ/mol]: ", self.tempcoll, self.Ecoll*2625.5)

        if self.Ecoll == None or self.bmax == None or self.Rini == None:
            raise ValueError("Ecoll (unless it's thermall sampled), bmax and Rini must be given in the input of Set_Relative_Init_Coords()")

        if self.bsampling:
            self.bimp = self.bmax * math.sqrt(random.uniform(0.0,1.0))
        else:
            self.bimp = self.bmax #fix impact parameter for opacity function P(b) calculations
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

        velRel = math.sqrt(2.0*self.Ecoll/redmass)
        velA = velRel*wB / (wA+wB)
        velB = velA - velRel

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

    def sample_and_run_collision(self, integrator='verlet', timestep=1.0, startstep=0, maxstep=100, iprint=2,
                                      traj_file='trajectory.xyz', backfile="checkpoint.xyz", restart=False, Rstop=10.0, **kwargs):

        self.Sample_Bimolecular_Reactants(**kwargs)

        self.run_trajectory(integrator=integrator, timestep=timestep, startstep=startstep, maxstep=maxstep,
                            iprint=iprint, traj_file=traj_file,  backfile=backfile, restart=restart, collision=True, Rstop=Rstop)


        

if __name__ == '__main__':
    import random

    c1   = 0.52917721092         # [bohr]     * c1 = [Ansgtrom]
    c3   = 1838.6836605e0        # [g/mol]    * c3 = [electron mass unit]
    c5   = 219474.e0            # [Hartree]  * c5 = [cm-1]
    c6   = 41.341105             # [fs]       * c6 = [time in au]
    c7   = 2625.5                # [Hartree]  * c7 = [kJ/mol]
    c9   = 1.0e8/c1             # [freqcm-1] *c9=[freq(bohr^(-1))]
    c10  = 137.035999074        # [speed of light in atomic unit]
    Rgas = 8.3144598/1000.0/c7 #Hartree/K

    xyz_water = '''
      O    -0.011100  0.0000  -0.00788
      H     0.007500  0.0000   0.95111
      H     0.899200  0.0000  -0.30990
       '''

    #IRC endpoint of openchain cis-1,3,5 triene
    xyz_diene = '''
      C      -1.185385      1.500364     -0.174799
      C       0.057100      1.525641      0.290486
      C       1.182421      0.671307     -0.090281
      C       1.182420     -0.671306     -0.090282
      C       0.057100     -1.525640      0.290485
      C      -1.185388     -1.500364     -0.174796
      H       0.285347      2.226153      1.090336
      H       2.144030      1.164815     -0.199258
      H       2.144029     -1.164815     -0.199261
      H       0.285350     -2.226154      1.090332
      H      -1.972016     -2.076478      0.294389
      H      -1.462109     -0.924082     -1.042658
      H      -1.972017      2.076476      0.294382
      H      -1.462101      0.924083     -1.042665
     '''

     #ZZAllyl-peroxy+O2_Case1 M062X avtz optim Con 21 (11 csak a futtatos jelolesben 21)
    xyz_zzallyl = '''
    C   -0.19595538763398      0.07192231509414      0.00714857985445
    C   -0.01703579534250     -0.10344222305327      1.49976587011423
    C   -1.00831293517504     -0.74304790940588      2.21484017141187
    C   -1.09565313995228     -0.96572164058650      3.69139746432409
    O   -2.44905662107855     -1.09778426038872      4.10589699809888
    O   -3.03738052053937      0.19479040763344      4.06761535894273
    C   1.14919758827461      0.41511797958774      2.03274034295381
    O   1.44878216770210      0.30476609400103      3.35160340084751
    H   -1.82826168270363     -1.17741133496109      1.65145938615843
    H   1.87669555170293      0.91631236085033      1.40830140902315
    H   0.76154446389898      0.10310849679981     -0.51109606581980
    H   -0.77393280198763     -0.75201110449544     -0.40695658995199
    H   -0.72775230388759      0.99760660592287     -0.21471742835545
    H   -0.61505142152353     -0.17733195303782      4.26524823267036
    H   -0.65062609284075     -1.92286967316247      3.98246076165767
    H   2.29766286152308      0.71228566068994      3.53361144883480
    H   -3.35772393043684      0.25312017851188      3.15850065923520
     '''


    seed = 220022
    random.seed(seed)

    qcinput = {
    'qchem': 'XTB',
    'path': '/home/peter/orca_6_0_0/xtb',
    'nproc': 8,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 2,
    'additional': '--gfnff --acc 50 --iterations 1000 --spinpol --tblite',
    'wfu': False
    }

    qcinput_Orca = {
    'qchem': 'Orca',
    'path': '/home/peter/orca_6_0_0/orca',
    'nproc': 4,
    'functional': 'HF-3c',
    'basis': '',
    'charge': 0,
    'multiplicity': 2,
    'additional': '',
    'wfu': False
    }

    qcinput_PySCF = {
    'qchem': 'PySCF',
    'path': '',
    'nproc': 4,
    'functional': 'PBE',
    'basis': 'sto-3g',
    'charge': 0,
    'multiplicity': 1,
    'additional': '',
    'wfu': False
    }


    rigid = False
    random_rot = False #True
    ngeom = 100

    fname_water = "water"
    traj_file_water = "traj_" + fname_water + ".xyz"

    fname_diene = "diene"
    traj_file_diene = "traj_" + fname_diene + ".xyz"


    fix_quantum = [(14, 1),
                   (15, 2)]

    fix_energy  = [(0, 0.0),
                   (1, 0.0),
                   (2, 0.0)]

    fix_temp    = [(5, 330.0),
                   (6, 430.0)]

    fix_quantum_water = [(0, 6)]

    #water  = Fragment.Polyatom_Init(fname=fname_water, qchem=qcinput, xyz=xyz_water, random_rot=random_rot)
    #water.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=10, fix_quantum=fix_quantum_water)

    #diene  = Fragment.Polyatom_Init(fname=fname_diene, qchem=qcinput, xyz=xyz_diene, random_rot=True)
    #diene.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=10, fix_quantum=fix_quantum, fix_temp=fix_temp)
    #diene.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=0, fix_quantum=fix_quantum)

    fname_zzallyl = "ZZAllyl"
    traj_file_diene = "traj_" + fname_zzallyl + ".xyz"
    zzallyl  = Fragment.Polyatom_Init(fname=fname_zzallyl, qchem=qcinput, xyz=xyz_zzallyl, random_rot=True)
    zzallyl.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=0, fix_quantum=fix_quantum)

    clorine = Fragment.Atom_Init(atoms=['Cl'])
    fname_atom = 'Cl'


    req_O2 = 1.2
    omega_O2 = 1580.0
    fname_oxygen = 'O2'

    oxygen = Fragment.Diatom_Init(fname=fname_oxygen, atoms=['O','O'], req=req_O2, omega=omega_O2, random_rot=True, diatom='harmonic')
    oxygen.Specify_Mode_Sampling(init_rot_type='Jfix', nvib=3, jrot=0)

    print("--------- ZZ-OH-Allyl Isoprenyl radical--------")
    zzallyl.print_mode_sampling()
    print("--------- ZZ-OH-Allyl Isoprenyl radical DONE--------")


    print("\nReaction of ZZ-OH-allyl + O2")
    print()


    reaction =  Collision(zzallyl, oxygen, qchem=qcinput) 
    #reaction.Specify_Collision_Sampling(Rini=7.0, bmax=4.0, bsampling=True, Ecoll=None, Ecoll_thermal=True, temp=300.0)
    reaction.Specify_Collision_Sampling(Rini=5.5, bmax=4.0, bsampling=True, Ecoll_thermal=True, temp=300.0)


    backfile = "reaction_backup.xyz"
    traj_file_reaction = "traj_" + fname_zzallyl + "+" + fname_oxygen + ".xyz"


    reaction.sample_and_run_collision(integrator='verlet', timestep=0.5, maxstep=2000, iprint=4, Rstop=12.0, traj_file=traj_file_reaction, backfile=backfile)



   




