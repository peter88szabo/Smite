import numpy as np
import math
import os
from normalmode import getNormalmode

def print_nmode_traj(atoms, q, dt, istep, output_file):
    b2a = 0.52917721092
    output_file.write(str(len(atoms)) + "\n")
    output_file.write("%7s %10d %4s %15.4f\n" % ("step= ", istep, "    theta[rad]= ", dt * istep))
    for i in range(0, len(atoms)):
        jx = 3 * i
        jy = 3 * i + 1
        jz = 3 * i + 2
        output_file.write("%3s %15.5f  %15.5f %15.5f\n" % (atoms[i], q[jx] * b2a, q[jy] * b2a, q[jz] * b2a))

def print_single_mode(filename, atoms, nmode, q_eq, L, Amp, imode, **kwargs):
    ntheta = kwargs.get('ntheta', 600)

    q_eq_tr = np.transpose(q_eq)
    ampl = [0.0] * nmode
    ampl[imode] = Amp

    q_norm = [0.0] * nmode
    dt = 2.0 * math.pi / ntheta
    theta = 0.0

    directory = "normal_mode_animation_" + filename   # replace with your desired directory path

    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} created.")

    fnm = os.path.join(directory, "mode_" + str(imode) + "_" + filename + ".xyz")

    if os.path.exists(fnm):
        os.remove(fnm)
        #print(f"{fnm} already exists, it has been deleted to create a new one.")

    #energy = [ww[i]*(nvib[i] + 0.5) for i in range(len(ww))]
    #ampl = [math.sqrt(2.0 * energy[i])/ww[i] for i in range(len(ww))]

    with open(fnm, "a") as file:
        for i in range(ntheta+1):
            theta += i * dt
            q_norm[imode] = ampl[imode] * math.cos(theta)
            q_cart = q_eq_tr + np.matmul(L, np.transpose(np.array(q_norm)))
            print_nmode_traj(atoms, q_cart, dt, i, file)



def print_normalmode(fname, atoms, mass, q_eq, hessian, **kwargs):

    give_freq_and_Lmat = kwargs.get('give_freq_and_Lmat', False)
    linear = kwargs.get('linear', False)
    imode = kwargs.get('imode', None)
    Amp = kwargs.get('Amp', 20.0)
    is_eckart = kwargs.get('is_eckart', False)

    nmode = len(q_eq) - 6 + int(linear)

    ww, ww_low, L = getNormalmode(mass, hessian, linear=linear, q_eq=q_eq, is_eckart=is_eckart)

    if imode != None:
        print_single_mode(fname, atoms, nmode, q_eq, L, Amp, imode)
    else:
        for im in range(nmode):
            print_single_mode(fname, atoms, nmode, q_eq, L, Amp, im)

    if give_freq_and_Lmat == True:
        return (ww, ww_low, L)

    return

