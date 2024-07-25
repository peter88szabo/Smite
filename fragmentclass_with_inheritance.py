import numpy as np

from cenmass            import cenmass
from euler              import euler_rot          
from polyrotation       import calcI, poly_rotation_init 
from format_and_print   import parseXYZ
from rel_init_coords    import setRelativeInitCoords 
from integrators        import velverlet
from hessian            import getHessian
from polyvibrationinit  import polyatom_vibration_init
from gradient           import Energy
from format_and_print   import print_trajectory

class Molecule:
    def __init__(self, atoms, mass, q_ini, p_ini):
        if len(mass) != len(atoms) or len(q_ini) != 3*len(atoms) or len(p_ini) != 3*len(atoms):
           raise ValueError("Lengths of inputs are not consistent.")

        self.atoms    = atoms
        self.natom    = len(atoms)
        self.q_ini    = np.array(q_ini)  
        self.p_ini    = np.array(p_ini) 
        self.q        = np.array(q_ini)  
        self.p        = np.array(p_ini) 
        self.mass     = np.array(mass)
        self.wmass    = np.zeros(3*len(atoms))  # Auxiliary mass vector
        self.totmass  = np.sum(mass) 
        self.angmom   = np.zeros(3)  
        self.inertia  = np.zeros(3)
        
        for i in range(len(atoms)):
            self.wmass[3*i:3*i+3] = np.sqrt(mass[i])

    def CalcHessian(self, qcinput, xyz, hessFile):
        return getHessian(self, qcinput, hessFile, xyz)

    def EckartProjectedHessian(mass, qeq, hessian):
        return eckart_transform(mass, qeq, hessian)

    def centermass_reduction(self):
        """
        Calculate the center of mass of the fragment.
        
        Returns:
        - A numpy array representing the center of mass coordinates (x, y, z).
        """
        total_mass = np.sum(self.mass)
        # Reshape q to separate x, y, z for each atom
        positions = self.q.reshape(-1, 3)
        weighted_positions = positions * self.mass[:, np.newaxis]  # Multiply each position by its atom's mass
        center_of_mass = np.sum(weighted_positions, axis=0) / self.totmass
        return center_of_mass

    def rotate_random(self):
        """
        Rotate randmoly the fragment molecule and its momenta
        about its center of mass wrt Euler angles
        """
        qq, pp = cenmass(self.q, self.p, self.mass)

        self.q, self.p = euler_rot(qq, pp)

    def verlet_single_step(self, dt, qcinput):
        self.q, self.p = velverlet(qcinput, dt, self.wmass, self.q, self.p, self.atoms)

    #def rattle_single_step(self, fixed_internals, constrained_bonds, dt, tolerance)
    #    self.q, self.p = rattle(self.q, self.p, self.mass, fixed_internals, constrained_bonds, dt, tolerance)


    def integrate_multi_step(self, integrator, dt, qcinput, file_wf, maxstep, iprint, traj_file):
        # Initial energy
        T0, V0, E0 = Energy(qcinput, file_wf, self.q, self.p, self.atoms, self.wmass)

        with open(traj_file, "a") as file_trj:
            for istep in range(maxstep):

                if integrator == 'verlet':
                    self.q, self.p = velverlet(qcinput, dt, self.wmass, self.q, self.p, self.atoms) #self.verlet_single_step(dt, qcinput)
                else:
                    raise ValueError("Only Verlet integrator is avaiable")

                T, V, E = Energy(qcinput, file_wf, self.q, self.p, self.atoms, self.wmass)

                act_temp = self.traj_temperature(0)

                dE = E - E0

                if iprint  > 0:
                    print(istep, T, V, E, dE)
                    print_trajectory(file_trj, self.atoms, self.q, self.p, V, dE, dt, istep)

    def initialize_momenta(self, Temp):
        """
        Initialize atomic momenta to fulfill
        the Maxwell-Boltzmann distribution

        The formula for velocities:
        v(i)=sqrt(RT/m(i))*N

        where N is a random number with normal distirbution

        But we calculate here momenta instead of velocities
        """

        RT = (8.3144598/1000.0/2625.5) * Temp  #Rgas in Hartree/K

        for i in range(len(self.p)):
            self.p[i] = np.sqrt(self.wmass[i] * RT) * np.random.normal()


    def traj_temperature(self,nfix):
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

        return 2.0*Ekin/float(len(self.p)-nfix)/Rgas


    def thermo_berendsen(self, nfix, dt, tau, Ttarg):
        '''
        Berendsen thermostate
        Ttarg = target temperature
        tau = time constant,scaling (coupling) parameter
        tau has to be: tau > dt
        '''

        if(tau < dt):
            raise ValueError("Wrong tau paramter in berendsen thermostate")

        Tact = self.traj_temperature(nfix)

        dum = dt * (Ttarg-Tact)/Tact/tau

        self.p = self.p * np.sqrt(1.0 + dum)


    def merge_with(self, other_molecule):
        if not isinstance(other_molecule, Molecule):
            raise ValueError("other_fragment must be an instance of Molecule or Fragment")
        
        combined_atoms = np.concatenate([self.atoms, other_molecule.atoms])
        combined_mass = np.concatenate([self.mass, other_molecule.mass])
        combined_coord_eq = np.concatenate([self.q, other_molecule.q])
        combined_momenta = np.concatenate([self.p, other_molecule.p])
        
        # Create a new instance with combined attributes
        new_fragment = Fragment(combined_atoms, combined_mass, combined_coord_eq)
        new_fragment.p = combined_momenta  # Set the combined momenta
        # Initialize wmass for the new fragment based on its mass
        new_fragment.wmass = np.array([np.sqrt(m) for m in combined_mass for _ in range(3)])

        return new_fragment



class Fragment(Molecule):
    def __init__(self, atoms, mass, q_ini, p_ini):

        super().__init__(atoms, mass, q_ini, p_ini)
        
        self.nvib     = np.zeros(3*len(atoms)) 
        self.jrot     = 0.0  
        self.krot     = 0.0  
        self.erot     = 0.0 
        self.evib     = 0.0 
        self.etot     = 0.0

        
        for i in range(len(atoms)):
            self.wmass[3*i:3*i+3] = np.sqrt(mass[i])


    def CalcHessian(qcinput, xyz, hessFile):
        return getHessian(qcinput, hessFile, xyz)

    def EckartProjectedHessian(mass, qeq, hessian):
        return eckart_transform(mass, qeq, hessian)


    @classmethod
    def AtomInit(cls, atoms, mass):
        if len(atoms) != 1:
            raise ValueError("ERROR: AtomInit requires only a single atom.")
        q_ini = np.array([0.0, 0.0 , 0.0])
        p_ini = np.array([0.0, 0.0 , 0.0])
        return cls(atoms, mass, q_ini, p_ini)


    @classmethod
    def DiatomHaromicInit(cls, atoms, mass, req, omega, nvib, jrot):
        if len(atoms) != 2:
            raise ValueError("ERROR: DiatomHaromicInit accept only a diatomic molecule.")

        q_ini, p_ini = diatom_init_harm(req, omega, mass, jrot, nvib)

        return cls(atoms, mass, q_ini, p_ini)

    @classmethod
    def PolyatomInit(cls, vibfile, xyz, mass, hessian, fixvib, nvib, fixrot, jrot, Temp):
        natom, atoms, q_eq = parseXYZ(xyz)

        if len(atoms) <= 2:
            raise ValueError("ERROR: PolyatomInit requires more than two atoms.")


        q_eq = np.array(q_eq) / 0.52917721092  #Angstrom to Bohr

        p = np.zeros(3*len(atoms))

        RT = (8.3144598/1000.0/2625.5) * Temp  #Rgas in Hartree/K

        q, p, ww_all = polyatom_vibration_init(vibfile, RT, fixvib, mass, q_eq, hessian)


#################################################################################EZ nem lesz igy joo#################################
        if fixrot != 0:
	    #inertia, Iinv = calcI(q_eq, mass)
            #jrot = thermal_J(temp)
            print("Thermal sampling of rotors not avaiable yet")
#################################################################################EZ nem lesz igy joo#################################

       #set rotational Cartesian coords (q,p) based on jrot quantum number
        q, p, inertia, angmom = poly_rotation_init(jrot, q_eq, mass, q, p)

       #randomly rotate the molecule about its center of mass
        q, p = cenmass(q, p, mass)
        q, p = euler_rot(q, p)
       #-----------------------------------------------------------------


        this = cls(atoms, mass, q, p)

        this.natom = natom
        this.atoms = atoms

        evib = 0.0
        erot = sum([angmom[i]**2/inertia[i]/2.0 for i in range(len(angmom))])

        this.erot     = erot
        this.etot     = erot + evib
        this.inertia  = inertia
        this.angmom   = angmom

        return this


    @classmethod
    def RigidPolyatomInit(cls, xyz, mass, fixrot, jrot, temp):
        natom, atoms, q_eq = parseXYZ(xyz)

        if len(atoms) <= 2:
            raise ValueError("ERROR: PolyatomInit requires more than two atoms.")


        q_eq = np.array(q_eq) / 0.52917721092  #Angstrom to Bohr

        p = np.zeros(3*len(atoms))

       #-----------------------------------------------------------------
       # Setting the initial rotational state of the polyatomic fragment
       #-----------------------------------------------------------------
       #shif the molecule to its center of mass
        q, p = cenmass(q_eq, p, mass)

#################################################################################EZ nem lesz igy joo#################################
        if fixrot != 0:
	    #inertia, Iinv = calcI(q_eq, mass)
            #jrot = thermal_J(temp)
            print("Thermal sampling of rotors not avaiable yet")
#################################################################################EZ nem lesz igy joo#################################

       #set rotational Cartesian coords (q,p) based on jrot quantum number
        q, p, inertia, angmom = poly_rotation_init(jrot, q_eq, mass, q, p)

       #randomly rotate the molecule about its center of mass
        q, p = cenmass(q, p, mass)
        q, p = euler_rot(q, p)
       #-----------------------------------------------------------------


        this = cls(atoms, mass, q, p)

        evib = 0.0
        erot = sum([angmom[i]**2/inertia[i]/2.0 for i in range(len(angmom))])

        this.erot     = erot
        this.etot     = erot + evib
        this.inertia  = inertia
        this.angmom   = angmom

        return this



class CollisionSystem:
    def __init__(self, fragment_A, fragment_B, temp, Ecoll, fixb, bmax, Rini):

        self.natomA           =  fragment_A.natom
        self.natomB           =  fragment_B.natom

        self.atomsA           =  fragment_A.atoms
        self.atomsB           =  fragment_B.atoms

        self.qA_ini           =  fragment_A.q
        self.qB_ini           =  fragment_B.q

        self.pA_ini           =  fragment_A.p
        self.pB_ini           =  fragment_B.p

        self.inertiaA_ini     =  fragment_A.inertia
        self.inertiaB_ini     =  fragment_B.inertia

        self.angmomA_ini      =  fragment_A.angmom
        self.angmomB_ini      =  fragment_B.angmom

        self.jrotA_ini        =  fragment_A.jrot
        self.jrotB_ini        =  fragment_B.jrot

        self.erotA_ini        =  fragment_A.erot
        self.erotB_ini        =  fragment_B.erot

        self.nvibA_ini        =  fragment_A.nvib
        self.nvibB_ini        =  fragment_B.nvib

        self.evibA_ini        =  fragment_A.evib
        self.evibB_ini        =  fragment_B.evib

        self.etotA_ini        =  fragment_A.etot
        self.etotB_ini        =  fragment_B.etot

        self.totmassA_ini     =  fragment_A.totmass
        self.totmassB_ini     =  fragment_B.totmass

        self.temp             =  temp
        self.bmax             =  bmax   
        self.Rini             =  Rini
        self.wmass            =  np.concatenate([fragment_A.wmass, fragment_B.wmass])

        # Combine the attributes of both fragments

        self.angmom_tot_abs_ini   =  0.0 
        self.angmom_tot_ini       =  np.zeros(3)
        self.angmom_orb_ini       =  np.zeros(3)
        self.angmom_rot_ini       =  fragment_A.angmom + fragment_B.angmom

        q, p, atoms, mass, bimp, Ecoll = setRelativeInitCoords(temp, Ecoll, fixb, bmax, Rini,
                                                               fragment_A.atoms, fragment_B.atoms,
                                                               fragment_A.mass,  fragment_B.mass,
                                                               fragment_A.q,     fragment_B.q,
                                                               fragment_A.p,     fragment_B.p)

        self.atoms        =  atoms
        self.mass         =  mass
        self.bimp         =  bimp
        self.Ecoll        =  Ecoll
        self.q            =  q
        self.p            =  p
        

if __name__ == '__main__':
    import random
    from matrix_print import print_matrix_pretty 

    c1   = 0.52917721092         # [bohr]     * c1 = [Ansgtrom]
    c3   = 1838.6836605e0        # [g/mol]    * c3 = [electron mass unit]
    c5   = 219474.e0            # [Hartree]  * c5 = [cm-1]
    c6   = 41.341105             # [fs]       * c6 = [time in au]
    c7   = 2625.5                # [Hartree]  * c7 = [kJ/mol]
    c9   = 1.0e8/c1             # [freqcm-1] *c9=[freq(bohr^(-1))]
    c10  = 137.035999074        # [speed of light in atomic unit]
    Rgas = 8.3144598/1000.0/c7 #Hartree/K

    mH  = 1.00782503223*c3
    mC  = 12.011*c3
    mN = 14.007*c3
    mO  = 15.999*c3
    mKr = 83.798*c3

    xyz_water = '''
      O    -0.011100  0.0000  -0.00788
      H     0.007500  0.0000   0.95111
      H     0.899200  0.0000  -0.30990
       '''

    #IRC endpoint of openchain cis-1,3,5 triene
    xyz = '''
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

    mass =  [mC]*6 + [mH]*8


    mass_water =  [mO]*1 + [mH]*2

    fixvib = [999,0]
    nvib = [0.0]
    fixrot = 1
    jrot = 1.0
    temp = 300.0
    dt = 1.0*c6
    maxstep = 1000
    iprint = 1
    filename = 'diene'

    seed = 221422
    random.seed(seed)


    qchem = 'XTB'
    functional = ''
    base = ''
    charge = 0
    multiplicity = 1 
    path = '/home/peter/Programs/xtb-6.6.1/bin/xtb'
    nproc = 8
    wfu = False
    additional = '--acc 10'
    #additional = '--acc 20 --gfn 1'
    #additional = '--gfnff'


    qcinput = [0]*9
    qcinput[0] = qchem
    qcinput[1] = path
    qcinput[2] = nproc
    qcinput[3] = functional
    qcinput[4] = base
    qcinput[5] = charge
    qcinput[6] = multiplicity
    qcinput[7] = additional
    qcinput[8] = wfu


    hessFile = 'hessian_' + filename + '.hess'
    vibfile = 'vibration_' + filename + '.dat'
    file_wf = 'wavefunc_' + filename + '.dat'
    traj_file = 'traj_' + filename + '.xyz'


    hessian = Fragment.CalcHessian(qcinput, xyz, hessFile)
    mol = Fragment.PolyatomInit(vibfile,xyz, mass, hessian, fixvib, nvib, fixrot, jrot, temp)
    mol.integrate_multi_step('verlet', dt, qcinput, file_wf, maxstep, iprint, traj_file)

'''
    kripton = Fragment.AtomInit(['Kr'], [mKr])
    kripton.initialize_momenta(300.0)

    print()
    print("========================================================")
    print("Kripton fragment properties:")
    print("========================================================")
    print("natom: ", kripton.natom)
    print("total mass: ", kripton.totmass/c3)
    print("mass vec: ", kripton.mass/c3)
    print("jrot : ", kripton.jrot)
    print("angmom : ", kripton.angmom)
    print("inertia :", kripton.inertia)
    print("erot : ", kripton.erot)
    print("evib : ", kripton.evib)
    print("etot : ", kripton.etot)
    print("q: ", kripton.q)
    print("p: ", kripton.p)



    temp = 0.0
    Ecoll = 0.18
    fixb = True #False
    bmax = 6.0
    Rini = 12.0
    bigone = CollisionSystem(water, kripton, temp, Ecoll, fixb, bmax, Rini)

'''
