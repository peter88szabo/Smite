import numpy as np

def makeXYZ(atoms, q):
    b2a = 0.52917721092
    natoms = len(atoms)

    xyz = ""

    for i in range(natoms):
        xyz += atoms[i] + " "*6+str(q[i*3]*b2a) \
            + " "*6+str(q[i*3+1]*b2a) \
            + " "*6+str(q[i*3+2]*b2a) \
            + "\n"
    return xyz

def PySCF_Force(q, atoms, qcinput):
    from pyscf       import gto, dft

    functional   =   qcinput['functional']
    basis        =   qcinput['basis']
    charge       =   qcinput['charge']
    multiplicity =   qcinput['multiplicity']

    xyz = makeXYZ(atoms, q)
    mol = gto.M(
            atom = xyz,
            basis = basis,
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

    functional   =   qcinput['functional']
    basis        =   qcinput['basis']
    charge       =   qcinput['charge']
    multiplicity =   qcinput['multiplicity']
    wfu          =   qcinput['wfu']

    xyz = makeXYZ(atoms, q)
    mol = gto.M(
            atom = xyz,
            basis = basis,
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

    functional   =   qcinput['functional']
    basis        =   qcinput['basis']
    charge       =   qcinput['charge']
    multiplicity =   qcinput['multiplicity']

    mol = gto.M(
            atom = xyz,
            basis = basis,
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

