from pyscf import gto, dft
from pyscf.tools import molden
import scine_utilities as su
import scine_sparrow

import numpy as np
from format_and_print import makeXYZ, parseXYZ

def PySCF_Force(q, atoms, charge, multiplicity, functional, base):
    xyz = makeXYZ(atoms, q)
    mol = gto.M(
            atom = xyz,
            basis = base,
            charge = charge,
            spin = multiplicity -1,
            verbose=0)

    if multiplicity != 1:
        mf = dft.UKS(mol).run(xc = functional)
    else:
        mf = dft.RKS(mol).run(xc = functional)

    #E = mf.kernel()
    g = mf.nuc_grad_method()
    grad = sum(g.kernel().tolist(), [])
    return -np.array(grad) 

def PySCF_Energy(filename, q, atoms, charge, multiplicity, functional, base):
    xyz = makeXYZ(atoms, q)
    mol = gto.M(
            atom = xyz,
            basis = base,
            charge = charge,
            spin = multiplicity -1,
            verbose=0)

    if multiplicity != 1:
        mf = dft.UKS(mol).run(xc = functional)
    else:
        mf = dft.RKS(mol).run(xc = functional)

    E = mf.kernel()

    with open(filename, "w") as file_wf:
        molden.header(mol, file_wf)
        molden.orbital_coeff(mol, file_wf, mf.mo_coeff, ene=mf.mo_energy, occ=mf.mo_occ)

    return E



def print_structure(atoms, q, filename):
    b2a = 0.52917721092

    with open(filename, "w") as file:
        file.write(str(len(atoms)) + "\n")
        file.write("This is a temporary strucutre for Sparrow to calculate energy and gradient\n")
        for i in range(0, len(atoms)):
            jx = 3 * i
            jy = 3 * i + 1
            jz = 3 * i + 2
            file.write("%3s %15.5f  %15.5f %15.5f\n" % (atoms[i], q[jx] * b2a, q[jy] * b2a, q[jz] * b2a))


def Sparrow_Energy(filename, q, atoms, charge, multiplicity, method, wfu):
    geomfile = 'geomfile_to_read_by_Sparrow.xyz'

    print_structure(atoms, q, geomfile)

    manager = su.core.ModuleManager.get_instance()
    calculator = manager.get('calculator', method)
    calculator.structure = su.io.read(geomfile)[0]
    calculator.set_required_properties([su.Property.Energy])

    if multiplicity != 1:
        calculator.settings['spin_mode'] = 'unrestricted'

    results = calculator.calculate()

    E = results.energy

   #------------------------------------------------------------------------------
   #print wavefunction:
    if wfu == True:
    	wf_gen = su.core.to_wf_generator(calculator)
    	#print(wf_gen.wavefunction)
    	wf_gen.wavefunction2file(filename)
   #------------------------------------------------------------------------------

    return E


def Sparrow_Force(q, atoms, charge, multiplicity, method):
    geomfile = 'geomfile_to_read_by_Sparrow.xyz'

    print_structure(atoms, q, geomfile)

    manager = su.core.ModuleManager.get_instance()
    calculator = manager.get('calculator', method)
    calculator.structure = su.io.read(geomfile)[0]
    #calculator.set_required_properties([su.Property.Energy,su.Property.Gradients])
    calculator.set_required_properties([su.Property.Gradients])

    if multiplicity != 1:
        calculator.settings['spin_mode'] = 'unrestricted'

    results = calculator.calculate()

    #E = results.energy
    force = -np.array(results.gradients.flatten())

    return force


def Energy(file_wf, qchem, q, p, atoms, wmass, charge, multiplicity, functional, base, wfu):

    if qchem == 'PySCF':
        V = PySCF_Energy(file_wf, q, atoms, charge, multiplicity, functional, base)
    elif qchem == 'Sparrow':
        method = functional
        V = Sparrow_Energy(file_wf, q, atoms, charge, multiplicity, method, wfu) 
    else:
        raise ValueError("Non-Existing Quantum Chemical method is input. Avaiable packages: PySCF, Sparrow")

    T = sum(0.5*p*p/wmass)
    Etot = T+V
    return(T, V, Etot)

def force_calc(qchem, q, atoms, charge, multiplicity, functional, base):
    if qchem == 'PySCF':
        force = PySCF_Force(q, atoms, charge, multiplicity, functional, base) 
    elif qchem == 'Sparrow':
        force = Sparrow_Force(q, atoms, charge, multiplicity, functional)
    else:
        raise ValueError("Non-Existing Quantum Chemical method is input. Avaiable packages: PySCF, Sparrow")

    return force




#xyz ='''
#  O  -0.06783047125742      0.00000000000000     -0.04795183080185
#  H   0.03988406002555      0.00000000000000      0.96552726825997
#  H   0.92361641123187      0.00000000000000     -0.28423843745812
#'''
#c1 = 0.52917721092

#Natoms, atoms, q = parseXYZ(xyz)
#q = np.array(q) / c1


#charge = 0
#multiplicity = 1
#functional = 'b3lyp'
#base = 'pc-0'

#E_PySCF    = PySCF_Energy(q, atoms, charge, multiplicity, functional, base)
#grad_PySCF = PySCF_Force(q, atoms, charge, multiplicity, functional, base)


#method = 'DFTB3'
#method = 'PM6'
#E_Sparrow = Sparrow_Energy(q, atoms, charge, multiplicity, method)
#grad_Sparrow = Sparrow_Force(q, atoms, charge, multiplicity, method) 



#print("Energy = ", E_PySCF, E_Sparrow)
#print()

#for i in range(3*Natoms):
#    print("%4d %18.5e %18.5e" % (i, grad_PySCF[i], grad_Sparrow[i]))

#print()

