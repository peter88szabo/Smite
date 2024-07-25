import numpy as np
import math
import os
from polyatom import getNormalmode

def print_nmode_traj(atoms, q, dt, istep, output_file):
    b2a = 0.52917721092
    output_file.write(str(len(atoms)) + "\n")
    output_file.write("%7s %10d %4s %15.4f\n" % ("step= ", istep, "    theta[rad]= ", dt * istep))
    for i in range(0, len(atoms)):
        jx = 3 * i
        jy = 3 * i + 1
        jz = 3 * i + 2
        output_file.write("%3s %15.5f  %15.5f %15.5f\n" % (atoms[i], q[jx] * b2a, q[jy] * b2a, q[jz] * b2a))

def print_normalmode(fname, imode, A, atoms, mass, q_eq, hessian):

    ww, ww_low, L = getNormalmode(mass, hessian)

    #energy = [ww[i]*(nvib[i] + 0.5) for i in range(len(ww))]
    #ampl = [math.sqrt(2.0 * energy[i])/ww[i] for i in range(len(ww))]

    L_tr = np.transpose(L)
    q_eq_tr = np.transpose(q_eq)

    nmode = len(q_eq) - 6
    ampl = [0.0] * nmode
    ampl[imode] = A

    q_norm = [0.0] * nmode
    ntheta = 600
    dt = 2.0 * math.pi / ntheta
    theta = 0.0

    if os.path.exists(fname):
        os.remove(fname)
        print(f"{fname} already exisits, it has been deleted to create a new one.")
    
    with open(fname, "a") as file:
        for i in range(ntheta+1):
            theta += i * dt
            q_norm[imode] = ampl[imode]* math.cos(theta)
            q_cart = q_eq_tr + np.matmul(L, np.transpose(np.array(q_norm)))
            print_nmode_traj(atoms, q_cart, dt, i, file)


