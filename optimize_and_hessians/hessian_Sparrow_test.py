import numpy as np
import time
from hessian import getHessian
from normalmode import getNormalmode, print_frequencies
from gradient import PySCF_Force, PySCF_Energy
from eckart import eckart_transform
from format_and_print import parseXYZ, makeXYZ #, print_trajectory
from nmodeprint  import print_normalmode
import scine_utilities as su
import scine_sparrow
from gradient import Sparrow_Force

def Sparrow_Hessian(q, atoms, charge, multiplicity, method):
    dx = 0.002
    ndim = len(q)
    hess = np.zeros((ndim, ndim))

    for i in range(ndim):
        q[i] += dx
        gradp1 = -Sparrow_Force(q, atoms, charge, multiplicity, method)

        q[i] -= 2*dx
        gradm1 = -Sparrow_Force(q, atoms, charge, multiplicity, method)

        #for j in range(ndim): 
        #    hess[i,j] = 0.5 * (gradp1[j] - gradm1[j]) / dx

        hess[i,:] = 0.5 * (gradp1 - gradm1) / dx

        q[i] += dx #restore partial coordinate
            
    return hess


c1 = 0.52917721092  # [bohr]    * c1 = [Ansgtrom] 
c3 = 1838.6836605e0 # [g/mol]   * c3 = [electron mass unit]
c6 = 41.341105      # [fs]      * c6 = [time in au]
c7 = 2625.5         # [Hartree] * c7 = [kJ/mol] 
Rgas = 8.3144598/1000.0/c7 #Hartree/K 

mH = 1.00782503223*c3
mC = 12.011*c3
mN = 14.007*c3
mO = 15.999*c3

#=========== Input parameters ====================
charge = 0
multiplicity = 1
#functional = 'b3lyp'
#functional = 'wb97x'
functional = 'pbe0'
base = 'pc-0'
#base = 'sto-3g'


#xyz ='''
#  O   -0.06783047125742      0.00000000000000     -0.04795183080185
#  H   0.03988406002555      0.00000000000000      0.96552726825997
#  H   0.92361641123187      0.00000000000000     -0.28423843745812
#'''

xyz = '''
C         0.02901          0.00005         0.01089
C         0.00105          0.00007         1.40408
C         1.21034          0.00002         2.09545
C         2.41945         -0.00002         1.39752
C         2.42491          0.00002         0.00148
C         1.22447          0.00005        -0.70516
H        -0.95361          0.00011         1.92181
H         1.20866         -0.00001         3.18342
H         3.36132         -0.00010         1.94381
H         3.36853          0.00000        -0.54009
H         1.20006          0.00005        -1.79088
N        -1.25105         -0.00005        -0.73145
O        -1.19735         -0.00004        -1.95628
O        -2.28739         -0.00016        -0.07615
 '''

Natoms, atoms, q_eq = parseXYZ(xyz)

q = np.array(q_eq) / c1

#-----------------------------------------------------------------
#mass = [mC, mC, mC, mC, mC, mC, mH, mH, mH, mH, mH, mN, mO, mO]
mass =  [mC]*6 + [mH]*5 + [mN] + [mO]*2
#mass =  [mO] + [mH]*2

#hessFile = 'hessian_NO2-benzene_B3LYP_pc1.hess'
#hessFile = 'hessian_NO2-benzene_PBE0_pc1.hess'
#hessFile = 'hessian_NO2-benzene_CAM-B3LYP_pc1.hess'
#hessFile = 'hessian_NO2-benzene_wb97x_pc1.hess'
#hessFile = 'hessian_NO2-benzene_b3lyp_sto3g.hess'

#hessFile = 'hessian_water_pbe0_pc0.hess'

hessFile = 'hessian_NO2-benzene_pbe0_pc0.hess'
qchem = 'PySCF'

hess = getHessian(qchem, hessFile, xyz, charge, multiplicity, functional, base)

ww, ww_low, L = getNormalmode(mass, hess)
ww_all = np.append(ww_low, ww)
#print_frequencies(ww_all)


Natoms, atoms, q_eq = parseXYZ(xyz)
q_eq = np.array(q_eq) / c1

#hess_eckart = eckart_transform(mass, q_eq, hess)

#print()
#print("After Eckart correction")
#ww_eckart, ww_low_eckart, L = getNormalmode(mass, hess_eckart)
#ww_all_eckart = np.append(ww_low_eckart, ww_eckart)
#print_frequencies(ww_all_eckart)

#print_normalmode("mode_23.xyz",23, 20.0, atoms, mass, q_eq, hess_eckart)
#print_normalmode("mode_24.xyz",24, 20.0, atoms, mass, q_eq, hess_eckart)
#print_normalmode("mode_25.xyz",25, 20.0, atoms, mass, q_eq, hess_eckart)

method = 'DFTB3'

hess_sp = Sparrow_Hessian(q, atoms, charge, multiplicity, method)


print("PySCF Hessian:")
print(hess)
print()
print("Sparrow Hessian:")
print(hess_sp)


ww_sp, ww_low_sp, L = getNormalmode(mass, hess_sp)
ww_all_sp = np.append(ww_low_sp, ww_sp)

print("PySCF Freqs:")
print_frequencies(ww_all)
print("Sparrow Freqs:")
print_frequencies(ww_all_sp)



#sys.exit()


