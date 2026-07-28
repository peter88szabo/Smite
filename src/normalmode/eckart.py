import numpy as np

def center_coordinates(mass, q):
    q = np.asarray(q, dtype=float).copy()
    mass = np.asarray(mass, dtype=float)
    xyz = q.reshape(-1, 3)
    center = np.sum(xyz * mass[:, None], axis=0) / np.sum(mass)
    return (xyz - center).reshape(-1)


def build_bmat(mass, q):
    ndim = len(q)
    natom = ndim // 3

    bmat = np.zeros((ndim, 6))

    # b[1-6] vectors merged as a matrix
    # from J. W. McIver J. Chern. Phys., Vol. 88, No. 2, 15 January 1988
    for i in range(natom):
        bmat[3*i,   0] =  np.sqrt(mass[i])             # b1
        bmat[3*i+1, 1] =  np.sqrt(mass[i])             # b2
        bmat[3*i+2, 2] =  np.sqrt(mass[i])             # b3

        bmat[3*i+1, 3] =  np.sqrt(mass[i]) * q[3*i+2]  # b4
        bmat[3*i+2, 3] = -np.sqrt(mass[i]) * q[3*i+1]  # b4

        bmat[3*i,   4] = -np.sqrt(mass[i]) * q[3*i+2]  # b5
        bmat[3*i+2, 4] =  np.sqrt(mass[i]) * q[3*i]    # b5

        bmat[3*i,   5] =  np.sqrt(mass[i]) * q[3*i+1]  # b6
        bmat[3*i+1, 5] = -np.sqrt(mass[i]) * q[3*i]    # b6

    return bmat

def get_eckart_projector(mass, q):
    # Ref: J. W. McIver J. Chern. Phys., Vol. 88, No. 2, 15 January 1988

    q = center_coordinates(mass, q)
    bmat = build_bmat(mass, q)

    # B @ pinv(B) projects onto the independent external vectors.  Its rank is
    # six for a nonlinear molecule and five for an exactly linear molecule.
    # Using the pseudoinverse avoids squaring B's condition number through
    # inv(B.T @ B) and handles the null rotation about a linear molecular axis.
    rmat = bmat @ np.linalg.pinv(bmat, rcond=1.0e-12)
    return 0.5 * (rmat + rmat.T)

def eckart_transform(mass, q, hess):
    #Eckart transformation:
    #projection out the 3 translations
    # and 3 rotations from the hessian
    #Ref: J. W. McIver J. Chern. Phys., Vol. 88, No. 2, 15 January 1988

    print()
    print("!!!!!              Eckart transformation switched on            !!!!!")
    print("!!!!! Translations and Rotations projected out from the Hessian !!!!!")
    print()

    ndim = len(q)
   # R = 3 translation + 3 rotation projector
   # see Eq D2 and D9 in the Appendix of Ref
    rmat = get_eckart_projector(mass, q)

   # P: projector to remove translation and rotation
   # P = 1 - R, see Eq 35 in Ref
    proj = np.identity(ndim) - rmat

    hess = 0.5 * (np.asarray(hess, dtype=float) + np.asarray(hess, dtype=float).T)
    proj = 0.5 * (proj + proj.T)
    hess_proj = np.matmul(hess, proj)

    hess_eckart = np.matmul(proj, hess_proj)

    return 0.5 * (hess_eckart + hess_eckart.T)


def eckart_reactionpath_transform(mass, q, grad, hess):
   #besides eckart transformation, the reaction path gradient
   #is also projected out. See Eq 2 in main text and D1 in Appendix
   #Ref: J. W. McIver J. Chern. Phys., Vol. 88, No. 2, 15 January 1988

    ndim = len(q)
 
   #normalized gradient along the reaction path
    grp = grad/np.linalg.norm(grad) 

   # R = 3 translation + 3 rotation projector
    rmat = get_eckart_projector(mass, q)

   # P = 1 - R - proj(grad)
    proj_grad = np.matmul(grp,np.transpose(grp))
    proj = np.identity(ndim) - rmat - proj_grad 

    hess_proj = np.matmul(hess, proj)

    hess_rph = np.matmul(proj, hess_proj)

    return hess_rph



#from gradient         import PySCF_Force
#from hessian          import getHessian
#from format_and_print import parseXYZ
#from polyatom         import getNormalmode
#from cenmass import cenmass



#-----------------------------------------------------------------
#charge = 0
#multiplicity = 1
#functional = 'b3lyp'
#base = 'sto-3g'

#xyz = '''
# C   0.041504   0.000119   0.018112
# C   0.007547   0.000107   1.424817
# C   1.229617   0.000023   2.125206
# C   2.450070  -0.000018   1.415311
# C   2.460308   0.000066   0.003460
# C   1.245690   0.000041  -0.709808
# H  -0.968265   0.000141   1.932043
# H   1.231453  -0.000043   3.223826
# H   3.400466  -0.000138   1.966557
# H   3.414912  -0.000010  -0.540322
# H   1.201740   0.000054  -1.808689
# N  -1.304719  -0.000072  -0.762361
# O  -1.235931  -0.000065  -2.082955
# O  -2.416230  -0.000207  -0.045999
# '''
#hessFile = 'hessian_NO2-benzene_B3LYP_STO3G.hess'

#Natoms, atoms, q = parseXYZ(xyz)
#-----------------------------------------------------------------


#hess = getHessian(hessFile, Natoms, xyz, charge, multiplicity, functional, base)
#ww, L = getNormalmode(mass, hess)
#print()
#hess_eckart = eckart_transform(mass, q, hess)
#ww_eckart, L_eckart = getNormalmode(mass, hess_eckart)

#===========================================================================
#xyz = '''
#  O   -0.06783047125742      0.00000000000000     -0.04795183080185
#  H   0.03988406002555      0.00000000000000      0.96552726825997
#  H   0.92361641123187      0.00000000000000     -0.28423843745812
# '''
#hessFile = 'Water_B3LYP_STO3G.hess'

#Natoms, atoms, q = parseXYZ(xyz)
#-----------------------------------------------------------------

#hess = getHessian(hessFile, Natoms, xyz, charge, multiplicity, functional, base)
#ww, L = getNormalmode(mass, hess)
#print()
#hess_eckart = eckart_transform(mass, q, hess)
#ww_eckart, L_eckart = getNormalmode(mass, hess_eckart)

#grad = -PySCF_Force(q, atoms, charge, multiplicity, functional, base) 

#hess_rph = eckart_reactionpath_transform(mass, q, grad, hess)
#ww_rph, L_rph = getNormalmode(mass, hess_rph)
