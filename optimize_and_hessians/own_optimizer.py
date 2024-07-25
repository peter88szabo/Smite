import numpy as np

def Step_Limit(dX):
    """
    Scale back step length (dX) if it's too big.
    """
    STPMX = 0.1
    stpmax = len(dX) * STPMX

    stpl = np.linalg.norm(dX)

    if stpl > stpmax:
        dX = dX / stpl * stpmax

    lgstst = np.max(np.abs(dX))

    if lgstst > STPMX:
        dX = dX / lgstst * STPMX

    return dX

def Simple_SteepestDescent(Xold, Grad):
    """
    Simple steepest descent.
    """
    beta = 0.7
    ChgeX = -beta * Grad

    ChgeX = Step_Limit(ChgeX)

    Xnew = Xold + ChgeX

    return Xnew


def BB_TwoPointGrad(Eq, StpLim, X_1, X_2, G_1, G_2):
    """
    Barzilai-Borwein two-point step size gradient method
    (IMA Journal of Numerical Analysis, 1988, 8, 141-149)

    Parameters:
        Eq: int         --> Equation selection (1 or 2).
        StpLim: str     --> Step limit option ("StepLim_ON" or any other string).
        X_1: np.ndarray --> Previous coordinate.
        X_2: np.ndarray --> Current coordinate.
        G_1: np.ndarray --> Previous effective gradient.
        G_2: np.ndarray --> Current effective gradient.

    Returns:
        X_3: np.ndarray --> The new updated coordinate towards minium
    """

    dG = G_2 - G_1
    dX = X_2 - X_1

    Ovrlp_XG = np.dot(dG, dX)  # <dX|dG>
    Xsq = np.dot(dX, dX)       # <dX|dX>
    Gsq = np.dot(dG, dG)       # <dG|dG>

    if Eq == 1:
        alpha = Ovrlp_XG / Gsq  # Eq(5) in Ref paper
    elif Eq == 2:
        alpha = Xsq / Ovrlp_XG  # Eq(6) in Ref paper
    else:
        raise ValueError("Problem in BB_TwoPointGrad routine")

    ChgeX = -alpha * G_2

    if StpLim == "StepLim_ON":
        ChgeX = Step_Limit(ChgeX)

    X_3 = X_2 + ChgeX

    return X_3

def BFGS_UpdateX(StpLim, X_1, X_2, G_1, G_2, Hi_1):
    """
    Updating the approximate inverse-Hessian

    Inputs:
    StpLim: str - "StepLim_ON" if step limit is enabled
    X_1, X_2: numpy.ndarray - previous two coordinates
    G_1, G_2: numpy.ndarray - previous two effective gradients
    Hi_1: numpy.ndarray - previous inverse Hessian

    Outputs:
    Hi_2: numpy.ndarray - updated approximate inverse Hessian
    X_3: numpy.ndarray - the new updated coordinate towards the MECP
    """

 
    ndim = len(X_2)

    DelG = G_2 - G_1
    DelX = X_2 - X_1
    HDelG = np.matmul(Hi_1, DelG)
    fac = np.dot(DelG, DelX)
    fae = np.dot(DelG, HDelG)

    fac = 1.0 / fac
    fad = 1.0 / fae
    w = fac * DelX - fad * HDelG

    # Update Hi_2
    Hi_2 = np.zeros((ndim, ndim), dtype=float)
    for i in range(ndim):
        for j in range(ndim):
            dum1 = fac * DelX[i] * DelX[j]
            dum2 = fad * HDelG[i] * HDelG[j] + fae * w[i] * w[j]
            Hi_2[i, j] = Hi_1[i, j] + dum1 - dum2

    ChgeX = -np.matmul(Hi_2, G_2)

    if StpLim == "StepLim_ON":
        ChgeX = Step_Limit(ChgeX)

    X_3 = X_2 + ChgeX

    return Hi_2, X_3



def TestConv_and_Print(istep, X_3, X_2, G_2, E_1, E_2, TGMax, TGRMS, TDXMax, TDXRMS, TDE):
    """
    Checks convergence, and updates report.
    """

    DE = E_2 - E_1

    DeltaX = X_3 - X_2

    DXMax = np.max(np.abs(DeltaX))
    Gmax = np.max(np.abs(G_2))

    GRMS = np.linalg.norm(G_2) / np.sqrt(float(len(G_2)))
    DXRMS = np.linalg.norm(DeltaX) / np.sqrt(float(len(G_2)))

    Conv = 0

    if (Gmax <= TGMax) and (GRMS <= TGRMS) and (DXMax <= TDXMax) and (DXRMS <= TDXRMS) and (abs(DE) <= TDE):
        Conv = 1

    if istep == 1:
        Print_Header_for_Report(TDE, TDXMax, TDXRMS, TGMax, TGRMS)

    form2 = "%4d %20.8f %20.8f %18.5f %15.5f  %15.5f  %15.5f  %15.5f"
    print(form2  %(istep, E_1, E_2, DE, DXMax, DXRMS, Gmax, GRMS))

    return Conv

def Print_Header_for_Report(TDE, TDXMax, TDXRMS, TGMax, TGRMS):
    """
    Prints the header for the report.
    """
    tokJmol=2625.5

    form0 = "%45s %18.5f %15.5f %15.5f  %15.5f  %15.5f"
    form1 = "%5s %20s %20s %18s %15s %15s %15s %15s"
    form00 = "%45s %18s %15s %15s %15s %15s"

    print(form0  % ("Convergence Threshold Values:", TDE * tokJmol, TDXMax, TDXRMS, TGMax, TGRMS))
    print(form00 % ("                              ", "  |  ", "  |  ", "  |  ", "  |  ", "  |  "))
    print(form00 % ("                              ", "  V  ", "  V  ", "  V  ", "  V  ", "  V  "))
    print(form1  % ("Step", "       E1[au]", "        E2[au]", "dE[kJ/mol]", "Max[dX]", "RMS[dX]", "  Max[Grad]", "   RMS[Grad]"))

def print_opt_traj(atoms, q, istep, ene, GradMax, output_file):
    b2a = 0.52917721092
    output_file.write(str(len(atoms)) + "\n")
    output_file.write("%7s %10d %6s %20.8f %10s %20.8f \n" % ("step= ", istep, "Ene = ", ene, "GradMax = ", GradMax))
    for i in range(0, len(atoms)):
        jx = 3 * i
        jy = 3 * i + 1
        jz = 3 * i + 2
        output_file.write("%3s %15.5f  %15.5f %15.5f\n" % (atoms[i], q[jx] * b2a, q[jy] * b2a, q[jz] * b2a))

from pyscf import gto, dft, hessian
from hessian import Sparrow_Hessian
import os
def get_initial_InverseHessian(qchem, inverse_hessFile, Natoms, xyz, charge, multiplicity, functional, base):

    print("qchem = ", qchem)
    if os.path.exists(inverse_hessFile):
        print()
        print("Loading Inverse-Hessian from file")
        print()
        hess = np.loadtxt(inverse_hessFile, delimiter=' ')
        if hess.shape == (Natoms*3,Natoms*3):
            return hess

    print("Recalculating Inverse-Hessian from scratch because the program could not find it in the file: ", inverse_hessFile)
    print("or the size of Inverse-Hessian matrix is wrong in the given file. Its size must be (Natoms*3,Natoms*3)")


    if qchem == 'PySCF':
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

        hessian = h.transpose(0,2,1,3).reshape(Natoms*3,Natoms*3)

    elif qchem == 'Sparrow':
        hessian = Sparrow_Hessian(q, atoms, charge, multiplicity, functional)



    Hinv = np.linalg.inv(hessian)
    np.savetxt(inverse_hessFile, Hinv, delimiter=' ', newline='\n')

    return Hinv


import sys
import time
from hessian import getHessian
from normalmode import getNormalmode, print_frequencies
from gradient import Sparrow_Energy, Sparrow_Force

from eckart import eckart_transform
from format_and_print import parseXYZ, makeXYZ #, print_trajectory
from nmodeprint  import print_normalmode

#time-stamp start
t_start = time.time()

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
qchem = 'Sparrow'
#qchem = 'PySCF'
charge = 0
multiplicity = 1
functional = 'DFTB3'
base = ''
#base = ''
#base = 'sto-3g'

xyz = '''
  C         0.00517          0.00004        -0.00282
  C        -0.00058          0.00006         1.40800
  C         1.20616          0.00002         2.09874
  C         2.41753         -0.00002         1.39640
  C         2.42567          0.00002        -0.00384
  C         1.22692          0.00005        -0.70835
  H        -0.94667          0.00011         1.94082
  H         1.21183          0.00001         3.18510
  H         3.35780         -0.00009         1.94177
  H         3.37149          0.00001        -0.53829
  H         1.21970          0.00006        -1.79415
  N        -1.08085         -0.00004        -0.69057
  O        -1.22049         -0.00003        -1.99832
  O        -2.43553         -0.00016        -0.07529
 '''
Natoms, atoms, q_eq = parseXYZ(xyz)

q = np.array(q_eq) / c1

#-----------------------------------------------------------------
#mass = [mC, mC, mC, mC, mC, mC, mH, mH, mH, mH, mH, mN, mO, mO]
mass =  [mC]*6 + [mH]*5 + [mN] + [mO]*2

#wmass = []   # auxiliary mass vector, same length as q and p
#for ww in mass:
#    wmass += [ww]*3
#wmass = np.array(wmass)

TDE=5.e-5     # energy difference
TDXMax=4.e-3  # max norm of step (old X --> new X)
TDXRMS=2.5e-3 # rms of step (old X --> new X)
TGMax=7.e-4   # max norm of gradient
TGRMS=5.e-4   # rms of gradient
maxstep = 100


X_1 = q

ndim = len(X_1)

print()
print("Computing Initial Hessian (Let's go for a coffee; it takes some time...)")
invhessFile = 'Inverse_Hessian_to_Initialize_QuasiNewton_Optimizer.hess'

iH_1 = get_initial_InverseHessian(qchem, invhessFile, Natoms, xyz, charge, multiplicity, functional, base)


print("Initial Inverse Hessian is ready")
print()
print("Starting Optimization")
print()


file_wf = "nemkell.molden"

E_1 = Sparrow_Energy(file_wf, X_1, atoms, charge, multiplicity, functional)
G_1 = Sparrow_Force(X_1, atoms, charge, multiplicity, functional) 

X_2 = Simple_SteepestDescent(X_1, G_1)

iH_2 = iH_1

#---------------------------------------------------------------------
fname = "geomopt_traj.xyz"
with open(fname, "a") as file:
    for istep in range(1,maxstep+1):
        G_2 = Sparrow_Force(X_2, atoms, charge, multiplicity, functional) 
        E_2 = Sparrow_Energy(file_wf, X_2, atoms, charge, multiplicity, functional)

        X_3 = BB_TwoPointGrad(1, "StepLim_ON", X_1, X_2, G_1, G_2)
        #iH_2, X_3 = BFGS_UpdateX("StepLim_ON", X_1, X_2, G_1, G_2, iH_1)

        Conv = TestConv_and_Print(istep, X_3, X_2, G_2, E_1, E_2, TGMax, TGRMS, TDXMax, TDXRMS, TDE)

        #Saving old coords:
        X_1 = X_2
        X_2 = X_3

        E_1 = E_2
        G_1 = G_2
        iH_1 = iH_2

        GradMax = np.max(np.abs(G_2))
        print_opt_traj(atoms, X_2, istep, E_2, GradMax, file)
 
        if Conv == 1:
            print("Convergence reached.")
            break
#---------------------------------------------------------------------


if istep < maxstep:
    print("!!!!!Convergence!!!!!")
    print("End of Optimization")
else:
    print("Sorry, NO Convergence :(")


t_end = time.time()
print((t_end - t_start), "s")




xyz = makeXYZ(atoms, X_2)
#hessFile = 'hessian_NO2-benzene_B3LYP_pc1.hess'
#hessFile = 'hessian_NO2-benzene_PBE0_pc1.hess'
#hessFile = 'hessian_NO2-benzene_CAM-B3LYP_pc1.hess'
#hessFile = 'hessian_NO2-benzene_wb97x_pc1.hess'
#hessFile = 'hessian_NO2-benzene_b3lyp_sto3g.hess'
hessFile = 'hessian_NO2-benzene_' + functional + '_' + base + '.hess'


hess = getHessian(qchem, hessFile, xyz, charge, multiplicity, functional, base)

#ww, ww_low, L = getNormalmode(mass, hess)
#ww_all = np.append(ww_low, ww)
#print_frequencies(ww_all)


Natoms, atoms, q_eq = parseXYZ(xyz)
q_eq = np.array(q_eq) / c1

hess_eckart = eckart_transform(mass, q_eq, hess)

print()
print("After Eckart correction")
ww_eckart, ww_low_eckart, L = getNormalmode(mass, hess_eckart)
ww_all_eckart = np.append(ww_low_eckart, ww_eckart)
print_frequencies(ww_all_eckart)

print_normalmode("mode_20.xyz",23, 20.0, atoms, mass, q_eq, hess_eckart)
print_normalmode("mode_21.xyz",23, 20.0, atoms, mass, q_eq, hess_eckart)
print_normalmode("mode_22.xyz",23, 20.0, atoms, mass, q_eq, hess_eckart)
print_normalmode("mode_23.xyz",23, 20.0, atoms, mass, q_eq, hess_eckart)
print_normalmode("mode_24.xyz",24, 20.0, atoms, mass, q_eq, hess_eckart)
print_normalmode("mode_25.xyz",25, 20.0, atoms, mass, q_eq, hess_eckart)
print_normalmode("mode_26.xyz",26, 20.0, atoms, mass, q_eq, hess_eckart)
print_normalmode("mode_27.xyz",27, 20.0, atoms, mass, q_eq, hess_eckart)
print_normalmode("mode_28.xyz",28, 20.0, atoms, mass, q_eq, hess_eckart)
print_normalmode("mode_29.xyz",29, 20.0, atoms, mass, q_eq, hess_eckart)
print_normalmode("mode_30.xyz",30, 20.0, atoms, mass, q_eq, hess_eckart)
print_normalmode("mode_31.xyz",30, 20.0, atoms, mass, q_eq, hess_eckart)
print_normalmode("mode_32.xyz",30, 20.0, atoms, mass, q_eq, hess_eckart)
print_normalmode("mode_33.xyz",30, 20.0, atoms, mass, q_eq, hess_eckart)

#sys.exit()


