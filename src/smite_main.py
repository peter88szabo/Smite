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

from normalmode.hessian           import getHessian
from normalmode.eckart            import eckart_transform
from normalmode.nmodeprint        import print_normalmode 
from normalmode.normalmode        import print_frequencies 

from sampling.polyvibration       import polyatom_vibration_sampling
from sampling.polyrotation        import polyatom_rotation_sampling
from sampling.init_vib_rot_modes  import init_vib_rot_modes
from integrators.integrators      import velverlet
from integrators.gradient         import Energy

class Molecule:
    def __init__(self, atoms=None, mass=None, q_ini=None, p_ini=None, nfix=0, restart=False, xyz_file_path=None):
        if restart:
            if not xyz_file_path:
                raise ValueError("xyz_file_path must be provided when restart is True.")
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


    def center_of_mass(self):
        """
        Calculate the center of mass of the fragment and shift the molecule
        to its center of mass.
        """
        self.q, self.p = cenmass(self.q, self.p, self.mass)

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

    def run_trajectory(self, integrator='verlet', dt=42.0, startstep=0, maxstep=100, iprint=2, traj_file='trajectory.xyz', backfile="checkpoint.xyz", restart=False, ):
        file_wf = "wavevfuntion" #or it should be given as class variable from self.fname

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

        with open(traj_file, "a") as file_trj:
            for istep in range(startstep, maxstep):

                if (iprint > 0 and istep % iprint == 0 and istep > startstep) or istep == 0:
                    T, V, E = self.get_energy() 

                    act_temp = self.traj_temperature()

                    dE = E - E0
                    print(f"step: {istep:<10d} time[fs]: {istep*dt/c6:<12.2f}  V[au]: {V:<13.5f} E[au]: {E:<13.5f} dE[cm-1]: {dE*c5:<12.3f} Temp[K]: {act_temp:<10.2f}")

                    print_trajectory(file_trj, self.atoms, self.q, self.p, V, dE, dt, istep)
                    #--------------------------------------------------------------------------
                    #Backup:
                    backup_file=open(backfile,'w')
                    print_trajectory(backup_file, self.atoms, self.q, self.p, V, dE, dt, istep)
                    backup_file.close()
                    #--------------------------------------------------------------------------

                if integrator == 'verlet':
                    self.verlet_single_step(dt)
                else:
                    raise ValueError("Only Verlet integrator is avaiable at the moment")

                tstop = False#test_to_stop(i, q, stopcond['atomA'], stopcond['atomB'], stopcond['rdist'])

                if tstop == True:
                    print("\n Reactive event found")
                    break


    def restart_init(self, integrator='verlet', dt=42.0, startstep=0, maxstep=100, iprint=2, traj_file='trajectory.xyz', backfile="checkpoint.xyz", restart=False):
        if restart:
            try:
                last_step, self.atoms, self.q, self.p = parseCheckPoint(backfile)
                self.mass = get_mass_vector(atoms)
                startstep = last_step
            except FileNotFoundError:
                raise ValueError("!!!!!!!!!!!!!!! Backup file does not exist !!!!!!!!!!!!!!!!!!!")
        else:
            raise ValueError("Wrong sampling option in sample_and_run_trajectory() function")


    def sample_and_run_trajectory(self, integrator='verlet', dt=42.0, startstep=0, maxstep=100, iprint=2, traj_file='trajectory.xyz', backfile="checkpoint.xyz", restart=False):
        #---------------------------------------------
        # Sample the internal motions of a fragment:
        #---------------------------------------------
        if self.natom == 1 and not restart:
            self.Atom_Sampling()
        elif self.natom == 2 and not restart:
            self.Diatom_Sampling()
        elif self.natom > 2 and not restart:
            self.Polyatom_Sampling()
        elif restart:
            startstep = self.last_step
        else:
            raise ValueError("Wrong sampling option in sample_and_run_trajectory() function")
        #---------------------------------------------

        self.run_trajectory(integrator=integrator, dt=dt, startstep=startstep, maxstep=maxstep,
                            iprint=iprint, traj_file=traj_file,  backfile=backfile, restart=restart) 


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


    def print_trajectory(self, trajfile, igeom):
        b2a = 0.52917721092
        trajfile.write(str(self.natom) + "\n")
        trajfile.write("%8s %10d \n" % ("geom = ", igeom ))
        for i in range(self.natom):
            jx = 3*i
            jy = 3*i+1
            jz = 3*i+2
            trajfile.write("%3s %15.5f  %15.5f %15.5f  %16.6f  %16.6f %16.6f\n" % (self.atoms[i], self.q[jx]*b2a, self.q[jy]*b2a, self.q[jz]*b2a, self.p[jx], self.p[jy], self.p[jz]))


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
            raise ValueError("ERROR: Atom_Init requires only a single atom.")

        q_ini = np.array([0.0, 0.0 , 0.0])
        p_ini = np.array([0.0, 0.0 , 0.0])

        mass = get_mass_vector(atoms)

        return cls(atoms=atoms, mass=mass, q_ini=q_ini, p_ini=p_ini)

    @classmethod
    def Diatom_Init(cls, atoms, **kwargs):
        if len(atoms) != 2:
            raise ValueError("ERROR: Diatom_Init accept only a diatomic molecule.")

        req = kwargs.get('req', None)
        omega = kwargs.get('omega', None)
        diat = kwargs.get('diatom', None)
        random_rot = kwargs.get('random_rot', True)
        rigid = kwargs.get('rigid', False)


        c9=1.0e8/0.5291772e0
        c10=137.035999074

        if req in kwargs and omega in kwargs:
            print("Parameters of diatom are defined by the user: req[Angstrom]={req} and omega[cm-1]={omega}")
        else:
            raise ValueError("ERROR: Diatom_Init needs (req[Angstrom] and omega[cm-1]) or diatom='XY' as input")

        mass = get_mass_vector(atoms)

        q1 = [req, 0.0, 0.0]
        q2 = [0.0, 0.0, 0.0]
        p1 = [0.0, 0.0, 0.0]
        p2 = [0.0, 0.0, 0.0]

        q_ini, p_ini  = cenmass(q1+q2, p1+p2, mass)

        this = cls(atoms=atoms, mass=mass, q_ini=q_ini, p_ini=p_ini)

        this.req_diat = req
        this.omega_diat = omega*c10/c9*(math.pi * 2)
        this.freq = [this.omega_diat]
        this.rigid  = rigid
        this.random_rot = random_rot

        return this 

    @classmethod
    def Polyatom_Init(cls, fname, qchem, xyz, **kwargs):
        linear = kwargs.get('linear', False)
        Amp_modeanim = kwargs.get('Amp_modeanim', 30.0)
        is_eckart = kwargs.get('is_eckart', True)
        random_rot = kwargs.get('random_rot', True)
        rigid = kwargs.get('rigid', False)


        natom, atoms, q_eq = parseXYZ(xyz)
        q_eq = np.array(q_eq) / 0.52917721092  #Angstrom to Bohr
        p_ini = np.zeros(len(q_eq))

        if len(atoms) <= 2:
            raise ValueError("ERROR: PolyatomInit requires more than two atoms.")

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

        return this

    def Atom_Sampling(self):
        if len(atoms) != 1:
            raise ValueError("ERROR: Atom_Sampling requires only a single atom.")

        self.q = np.array([0.0, 0.0 , 0.0])
        self.p = np.array([0.0, 0.0 , 0.0])
        return

    def Diatom_Sampling(self, **kwargs):
        if len(atoms) != 2:
            raise ValueError("ERROR: DiatomHaromicInit accept only a diatomic molecule.")

        verbosity = kwargs.get('verbosity', False)
        traj_index = kwargs.get('traj_index', -999)

        redmass = self.mass[0]*self.mass[1] / (self.mass[0] + self.mass[1])

        if self.rigid == False:

            if self.vibsampling['Q']:
                energy = self.omega_diat*(nvib + 0.5)
            elif self.vibsampling['T']:
                nvib = thermal_vibr_mode(temp, self.omega_diat)
                energy = self.omega_diat*(nvib + 0.5)
            elif self.vibsampling['E']:
                nvib = energy/self.omega_diat - 0.5 #non-integer quantum number
            else:
                raise ValueError("Either nvib=xxx or temp=xxx or energy=xxx must be given as input in Diatom_Sampling")

            jrot = 10
            ####Rotation and vibration should be separated as in case of polyatom
            self.q, self.p = diatom_init_harm(self.req_diat, self.omega_diat, self.mass, jrot, nvib)

        else:
            self.q = self.q_ini
            self.p = 0.0 #here we should add the option for rigid rotation. Everthing is built in above, we should separate it

        if self.random_rot == True:
            self.q, self.p = euler_rot(self.q, self.p)

        return

    def Specify_Mode_Sampling(self, init_vib_type, init_rot_type, **kwargs):
        self.vibsampling, self.rotsampling = init_vib_rot_modes(self.freq, init_vib_type=init_vib_type, init_rot_type=init_rot_type, **kwargs)
        return

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
        jrot = kwargs.get('jrot', 0)

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

        return


class Reaction(Molecule):
    def __init__(self, fragment_A, fragment_B, Rdist):
        super().__init__(atoms, mass, q_ini, p_ini)
        self.fragment_A = fragment_A
        self.fragment_B = fragment_B
        self.atoms = fragment_A.atoms + fragment_B.atoms
        self.mass = np.append(fragment_A.mass, fragment_B.mass)
        self.q_ini = np.append(fragment_A.q_ini, fragment_B.q_ini)
        self.p_ini = np.append(fragment_A.p_ini, fragment_B.p_ini)
    
    def Set_Relative_Init_Coords(self, Rdist):

        #shif the A and B molecule to their center of mass:
        qA = cenmassQ(self.fragment_A.q, self.fragment_A.mass)
        qB = cenmassQ(self.fragment_B.q, self.fragment_B.mass)

        #shift fragment B along the z-axis
        for i in range(len(massB)):
            jx = 3 * i
            jy = 3 * i + 1
            jz = 3 * i + 2
            fragment_B.q[jz] += Rdist 

        #wA = self.fragment_A.totmass
        #wB = self.fragment_B.totmass

        self.q = np.append(fragment_A.q, fragment_B.q)
        atoms = fragment_A.atom + fragment_B.atom  #this is just simple array, not numpy

        return 

    def Sample_Bimolecular_Reactants(self, Rdist, bmax, bsampling):
        #---------------------------------------------
        # Sample the internal motions of a fragment:
        #---------------------------------------------
        def sample_fragment(fragment):
            if fragment.natom == 1:
                fragment.Atom_Sampling()
            elif fragment.natom == 2:
                fragment.Diatom_Sampling()
            else:
                fragment.Polyatom_Sampling()
        #---------------------------------------------

        sample_fragment(self.fragment_A)
        sample_fragment(self.fragment_B)

        self.Set_Relative_Init_Coords(Rdist)

        self.overlap = check_atomic_overlap(self.atoms, self.q)

        return


        

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

    seed = 211422
    random.seed(seed)

    qcinput = {
    'qchem': 'XTB',
    'path': '/home/peter/orca_6_0_0/xtb',
    'nproc': 8,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 1,
    'additional': '--acc 10',
    'wfu': False
    }

    qcinput_Orca = {
    'qchem': 'Orca',
    'path': '/home/peter/orca_6_0_0/orca',
    'nproc': 4,
    'functional': 'PBE',
    'basis': 'pc-0',
    'charge': 0,
    'multiplicity': 1,
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


    fix_quantum = [(34, 0),
                   (35, 0)]

    fix_energy  = [(0, 0.0),
                   (1, 0.0),
                   (2, 0.0)]

    fix_temp    = [(5, 330.0),
                   (6, 430.0)]

    fix_quantum_water = [(0, 6)]

    #water  = Fragment.Polyatom_Init(fname=fname_water, qchem=qcinput, xyz=xyz_water, random_rot=random_rot)
    #water.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=10, fix_quantum=fix_quantum_water)

    diene  = Fragment.Polyatom_Init(fname=fname_diene, qchem=qcinput, xyz=xyz_diene, random_rot=random_rot)
    #diene.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=10, fix_quantum=fix_quantum, fix_temp=fix_temp)
    diene.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=520, fix_quantum=fix_quantum)

    '''
    print("----------------------------------------")
    print()
    print("water atoms: ", water.atoms)
    print("water mass:  ", water.mass)
    print("water q:     ", water.q)
    print("water p:     ", water.q)
    print()
    water.print_mode_sampling()

    if os.path.exists(traj_file_water):
        os.remove(traj_file_water)
        print(f"\n{traj_file_water} already exists, it has been deleted to create a new one.\n")

    with open(traj_file_water, "a") as file_trj:
        for igeom in range(ngeom):
            water.Polyatom_Sampling() #in kwargs can be given random_rot=False, rigid=True...
            water.print_structure(file_trj, igeom) 
    print("Water done")

   # if os.path.exists(traj_file_diene):
   #     os.remove(traj_file_diene)
   #     print(f"\n{traj_file_diene} already exists, it has been deleted to create a new one.\n")

    if os.path.exists(traj_file_diene):
        os.remove(traj_file_diene)
        print(f"\n{traj_file_diene} already exists, it has been deleted to create a new one.\n")
   
    '''

    print()
    print("--------- Diene--------------------------------")
    diene.print_mode_sampling()
    #with open(traj_file_diene, "a") as file_trj:
    #    for igeom in range(ngeom):
    #        diene.Polyatom_Sampling() #in kwargs can be given random_rot=False, rigid=True...
    #        diene.print_structure(file_trj, igeom)


    nstep = 2000 
    dt = 0.5*c6
    diene.Polyatom_Sampling() #in kwargs can be given random_rot=False, rigid=True...
    '''
    with open(traj_file_diene, "a") as file_trj:
        for istep in range(nstep):
            print(f"trajectory step: {istep}")
            diene.verlet_single_step(dt)
            diene.print_structure(file_trj, istep)
    '''

    integrator = 'verlet'
    maxstep = 100 
    iprint = 2
    restart = False
    backfile = "diene_backup.xyz"
    startstep = 0
    diene.sample_and_run_trajectory(integrator=integrator, dt=dt, maxstep=maxstep, iprint=iprint, traj_file=traj_file_diene, backfile=backfile, restart=restart)





    print("Diene done")





