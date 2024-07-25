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

def BFGS_Update(StpLim, X_1, X_2, G_1, G_2, Hi_1):
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


import os
from hessian import getHessian

def get_initial_InverseHessian(qcinput, hessFile, inverse_hessFile, xyz):

    if os.path.exists(inverse_hessFile):
        print()
        print("Loading Inverse-Hessian from file")
        print()
        Hinv = np.loadtxt(inverse_hessFile, delimiter=' ')
        if Hinv.shape == (Natoms*3,Natoms*3):
            return Hinv
        else:
            print("The size of Inverse-Hessian matrix is wrong in the given file. Its size must be (Natoms*3,Natoms*3)")
            print()

    print("Recalculating Inverse-Hessian from scratch because the program could not find it in the file: ", inverse_hessFile)
    print()
    print("Let's go for a coffee; it takes some time...")

    hessian = getHessian(qcinput, hessFile, xyz)

    Hinv = np.linalg.inv(hessian)

    np.savetxt(inverse_hessFile, Hinv, delimiter=' ', newline='\n')

    return Hinv

import sys
import time
from gradient         import force_calc
from gradient         import Potential_Energy
from format_and_print import parseXYZ, makeXYZ
def geom_optimzer(optinput, qcinput, hessFile, invhessFile, xyz, atoms, mass):
    c1 = 0.52917721092  # [bohr]    * c1 = [Ansgtrom]
   
    method     = optinput[0] 
    TDE        = optinput[1] #5.e-5    
    TDXMax     = optinput[2] #4.e-3 
    TDXRMS     = optinput[3] #2.5e-3
    TGMax      = optinput[3] #7.e-4  
    TGRMS      = optinput[4] #5.e-4  
    maxstep    = optinput[5] #100
    hessrecalc = optinput[6]

    Natoms, atoms, q = parseXYZ(xyz)
    q = np.array(q) / c1

    t_start = time.time()
    X_1 = q

    iH_1 = 0.0

    if method == "BFGS":
       iH_1 = get_initial_InverseHessian(qcinput, hessFile, inverse_hessFile, xyz)
       print("Initial Inverse Hessian is ready")

    print()
    print("Starting Optimization")
    print()

    file_wf = 'garbage.txt'

    E_1 =  Potential_Energy(qcinput, file_wf, q, atoms)
    G_1 = -force_calc(qcinput, q, atoms) #minus sign is need because we need gradient not force
    X_2 =  Simple_SteepestDescent(X_1, G_1)

    iH_2 = iH_1

    fname = "geomopt_traj.xyz"

    with open(fname, "a") as file:
         for istep in range(1,maxstep+1):
             G_2 = -force_calc(qcinput, q, atoms) #minus sign is need because we need gradient not forc
             E_2 = Potential_Energy(qcinput, file_wf, q, atoms)
            
             if method == "TwoPointGrad_1":
                X_3 = BB_TwoPointGrad(1, "StepLim_ON", X_1, X_2, G_1, G_2)
                iH_2 = 0.0
             if method == "TwoPointGrad_2":
                X_3 = BB_TwoPointGrad(2, "StepLim_ON", X_1, X_2, G_1, G_2)
                iH_2 = 0.0
             if method == "BFGS":
                iH_2, X_3 = BFGS_UpdateX("StepLim_ON", X_1, X_2, G_1, G_2, iH_1)
                #if istep % hessrecalc == 0:
             else:
                raise ValueError("Non-Existing optimzing method. Avaiable method: TwoPointGrad_1, TwoPointGrad_2, BFGS")
 
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

    if istep < maxstep:
       print("!!!!!Convergence!!!!!")
       #make_normal_analysis(qcinput, hessFile_end, X_2, atoms, mass)
    else:
       print("Sorry, NO Convergence :(")

    print()
    print("End of Optimization")

    t_end = time.time()
    print((t_end - t_start), "s")



if __name__ == '__main__':

   c1 = 0.52917721092  # [bohr]    * c1 = [Ansgtrom] 
   c3 = 1838.6836605e0 # [g/mol]   * c3 = [electron mass unit]
   c6 = 41.341105      # [fs]      * c6 = [time in au]
   c7 = 2625.5         # [Hartree] * c7 = [kJ/mol] 

   mH = 1.00782503223*c3
   mC = 12.011*c3
   mN = 14.007*c3
   mO = 15.999*c3

   TDE=5.e-5     # energy difference
   TDXMax=4.e-3  # max norm of step (old X --> new X)
   TDXRMS=2.5e-3 # rms of step (old X --> new X)
   TGMax=7.e-4   # max norm of gradient
   TGRMS=5.e-4   # rms of gradient
   maxstep = 100
   hessrecalc = 5 

   optinput[0] = method  
   optinput[1] = TDE     
   optinput[2] = TDXMax  
   optinput[3] = TDXRMS  
   optinput[3] = TGMax   
   optinput[4] = TGRMS   
   optinput[5] = maxstep 
   optinput[6] = hessrecalc
   
   geom_optimzer(optinput, qcinput, hessFile, invhessFile, xyz, atoms, mass)

