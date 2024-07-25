
import numpy as np
from format_and_print import makeXYZ

def PySCF_Force(q, atoms, qcinput):
    from pyscf       import gto, dft
    functional   =   qcinput[3]
    base         =   qcinput[4]
    charge       =   qcinput[5]
    multiplicity =   qcinput[6]

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

def PySCF_Energy(filename, q, atoms, qcinput):
    from pyscf       import gto, dft
    from pyscf.tools import molden

    functional   =   qcinput[3]
    base         =   qcinput[4]
    charge       =   qcinput[5]
    multiplicity =   qcinput[6]
    wfu          =   qcinput[8]

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

    if wfu == True:
    	with open(filename, "w") as file_wf:
        	molden.header(mol, file_wf)
        	molden.orbital_coeff(mol, file_wf, mf.mo_coeff, ene=mf.mo_energy, occ=mf.mo_occ)

    return E

def PySCF_Hessian(Natoms, xyz, qcinput):
    from pyscf       import gto, dft
    functional   =   qcinput[3]
    base         =   qcinput[4]
    charge       =   qcinput[5]
    multiplicity =   qcinput[6]

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

    h = mf.Hessian().kernel()
    hess = h.transpose(0,2,1,3).reshape(Natoms*3,Natoms*3)
    return hess

