import numpy as np
import os

from cenmass            import cenmass, cenmassQ
from euler              import euler_rotQ          
from polyrotation       import calcI, poly_rotation_init 
from format_and_print   import parseXYZ
from rel_init_coords    import setRelativeInitCoords 
from hessian            import getHessian
from PES_Scan_polyvib   import polyatom_vibration_init
from gradient           import Energy
from format_and_print   import print_trajectory
from eckart             import eckart_transform

class Molecule:
    def __init__(self, atoms, mass, q_ini):
        if len(mass) != len(atoms) or len(q_ini) != 3*len(atoms):
           raise ValueError("Lengths of inputs are not consistent.")

        self.atoms    = atoms
        self.natom    = len(atoms)
        self.q_ini    = np.array(q_ini)  
        self.q        = np.array(q_ini)  
        self.mass     = np.array(mass)
        self.wmass    = np.zeros(3*len(atoms))  # Auxiliary mass vector
        self.totmass  = np.sum(mass) 
        self.inertia  = np.zeros(3)
        
        for i in range(len(atoms)):
            self.wmass[3*i:3*i+3] = np.sqrt(mass[i])

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
        qq = cenmassQ(self.q, self.mass)

        self.q = euler_rotQ(qq)


    def merge_with(self, other_molecule):
        if not isinstance(other_molecule, Molecule):
            raise ValueError("other_fragment must be an instance of Molecule or Fragment")
        
        combined_atoms = np.concatenate([self.atoms, other_molecule.atoms])
        combined_mass = np.concatenate([self.mass, other_molecule.mass])
        combined_coord_eq = np.concatenate([self.q, other_molecule.q])
        
        # Create a new instance with combined attributes
        new_fragment = Fragment(combined_atoms, combined_mass, combined_coord_eq)

        return new_fragment

    def print_structure(self,trajfile,igeom):
        b2a = 0.52917721092
        trajfile.write(str(self.natom) + "\n")
        trajfile.write("%8s %10d \n" % ("geom = ", igeom ))
        for i in range(self.natom):
            jx = 3*i
            jy = 3*i+1
            jz = 3*i+2
            trajfile.write("%3s %15.5f  %15.5f %15.5f \n" % (self.atoms[i], self.q[jx]*b2a, self.q[jy]*b2a, self.q[jz]*b2a))


class Fragment(Molecule):
    def __init__(self, atoms, mass, q_ini):

        super().__init__(atoms, mass, q_ini)
        
        self.nvib     = np.zeros(3*len(atoms)) 


    def CalcHessian(qcinput, xyz, hessFile):
        return getHessian(qcinput, hessFile, xyz)

    def EckartProjectedHessian(mass, xyz, hessian):
        natom, atoms, qeq = parseXYZ(xyz)
        qeq = np.array(qeq) / 0.52917721092  #Angstrom to Bohr
        return eckart_transform(mass, qeq, hessian)


    @classmethod
    def AtomInit(cls, atoms, mass):
        if len(atoms) != 1:
            raise ValueError("ERROR: AtomInit requires only a single atom.")
        q_ini = np.array([0.0, 0.0 , 0.0])
        return cls(atoms, mass, q_ini)


    @classmethod
    def DiatomHaromicInit(cls, phase_sampling, eulerrot, atoms, mass, req, omega, nvib, temp):
        if len(atoms) != 2:
            raise ValueError("ERROR: DiatomHaromicInit accept only a diatomic molecule.")

        redmass = mass[0]*mass[1] / (mass[0] + mass[1])

        if phase_sampling == 'cosine':
            dr = math.sqrt( 2 * ( nvib + 0.5 ) / ( redmass * omega)) * math.cos(random.uniform(0, 2 * math.pi))
        elif phase_sampling == 'linear':
            dr = math.sqrt( 2 * ( nvib + 0.5 ) / ( redmass * omega)) * random.uniform(-1, 1)
        else:
            raise ValueError("{sampling} method is not available. Options to choose: cosine, linear ")

        r = req + dr

        q1 = [r, 0.0, 0.0]
        q2 = [0.0, 0.0, 0.0]
        q_ini  = cenmassQ(q1+q2, mass)

        if eulerrot == True:
            q_ini = euler_rotQ(q_ini)

        return cls(atoms, mass, q_ini)

    @classmethod
    def PolyatomInit(cls, vibfile, rigid, eulerrot, phase_sampling, xyz, mass, hessian, fixvib, nvib, fixrot, jrot, Temp):

        natom, atoms, q_eq = parseXYZ(xyz)
        q_eq = np.array(q_eq) / 0.52917721092  #Angstrom to Bohr

        if len(atoms) <= 2:
            raise ValueError("ERROR: PolyatomInit requires more than two atoms.")

        ###################################################################nvib missing!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #### to control the amplitudes
        # polyatom_vibration_init() should read nvib too, in order control the amplitudes
        ##
        #
        # Plus because of the hindered Rotors, when we treat it as a free rotor
        # then maybe two atom overlaps. These should be discarded from sampling
        # for these discarded tyrings we need a collector, and also a flag with True or False

        if rigid == False:
            q = polyatom_vibration_init(vibfile, phase_sampling, Temp, fixvib, mass, q_eq, hessian)

        q = np.array(q)
        mass = np.array(mass)

        q = cenmassQ(q, mass)
       #randomly rotate the molecule about its center of mass
        if eulerrot == True:
            qini = euler_rotQ(q)
       #-----------------------------------------------------------------


        this = cls(atoms, mass, q)

        this.natom = natom
        this.atoms = atoms

        return this


class CollisionSystem(Molecule):
    def __init__(self, fragment_A, fragment_B, Rini):
        super().__init__(atoms, mass, q_ini)
    
    @classmethod
    def Set_Relative_Init_Coords(cls,fragment_A, fragment_B, Rini):
        Rdist = Rini / 0.52917721092

        #shif the A and B molecule to their center of mass:
        qA = cenmassQ(qA, pA, massA)
        qB = cenmassQ(qB, pB, massB)

        #shift fragment B along the z-axis
        for i in range(len(massB)):
            jx = 3 * i
            jy = 3 * i + 1
            jz = 3 * i + 2
            fragment_B.q[jz] += Rdist 

        wA = sum(massA)
        wB = sum(massB)

        q = np.append(fragment_A.q, fragment_B.q)
        atoms = fragment_A.atom + fragment_B.atom  #this is just simple array, not numpy


        #When any atomic distance is smaller then X-Y: 0.4 Ansgtrom 

        return cls(fragment_A, fragment_B, Rini)


        

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
    temp = 2000.0
    dt = 1.0*c6
    maxstep = 1000
    iprint = 1
    filename = 'diene'

    seed = 211422
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
    traj_file = 'traj_' + filename + '.xyz'

    rigid = False
    eulerrot = True
    phase_sampling = 'linear'
    ngeom = 100


    if os.path.exists(traj_file):
        os.remove(traj_file)
        print()
        print(f"{traj_file} already exisits, it has been deleted to create a new one.")
        print()



    hessian = Fragment.CalcHessian(qcinput, xyz, hessFile)
    #hessian = Fragment.EckartProjectedHessian(mass, xyz, hessian)
    print("\nCalculation of Hessian is done \n")




    with open(traj_file, "a") as file_trj:
        for igeom in range(ngeom):

            mol = Fragment.PolyatomInit(vibfile, rigid, eulerrot, phase_sampling, xyz, mass, hessian, fixvib, nvib, fixrot, jrot, temp)

            mol.print_structure(file_trj, igeom) 

            print("geom: ", igeom, " is done")



