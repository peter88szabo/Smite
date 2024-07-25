import scine_utilities as su
import scine_sparrow
from sparrowbin import Sparrowbin_Force

import numpy as np


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


def SparrowPy_Energy(filename, q, atoms, qcinput):
    method       =   qcinput[3]
    base         =   qcinput[4]
    charge       =   qcinput[5]
    multiplicity =   qcinput[6]
    wfu          =   qcinput[8]

    geomfile = 'geomfile_to_read_by_Sparrow.xyz'

    print_structure(atoms, q, geomfile)

    manager = su.core.ModuleManager.get_instance()
    calculator = manager.get('calculator', method)
    calculator.structure = su.io.read(geomfile)[0]
    calculator.set_required_properties([su.Property.Energy])

    calculator.settings['molecular_charge'] = charge
    calculator.settings['spin_multiplicity'] = multiplicity


    if multiplicity != 1:
        calculator.settings['spin_mode'] = 'unrestricted'

    results = calculator.calculate()

    E = results.energy

    if wfu == True:
       wf_gen = su.core.to_wf_generator(calculator)
       wf_gen.wavefunction2file(filename)
   #------------------------------------------------------------------------------

    return E


def SparrowPy_Force(q, atoms, qcinput):

    method       =   qcinput[3]
    base         =   qcinput[4]
    charge       =   qcinput[5]
    multiplicity =   qcinput[6]

    geomfile = 'geomfile_to_read_by_Sparrow.xyz'

    print_structure(atoms, q, geomfile)

    manager = su.core.ModuleManager.get_instance()
    calculator = manager.get('calculator', method)
    calculator.structure = su.io.read(geomfile)[0]
    #calculator.set_required_properties([su.Property.Energy,su.Property.Gradients])
    calculator.set_required_properties([su.Property.Gradients])

    calculator.settings['molecular_charge'] = charge
    calculator.settings['spin_multiplicity'] = multiplicity


    if multiplicity != 1:
        calculator.settings['spin_mode'] = 'unrestricted'

    results = calculator.calculate()

    #E = results.energy
    force = -np.array(results.gradients.flatten())

    return force


def SparrowPy_Hessian(q, atoms, qcinput):
    '''
    Numerical hessian from Sparrow by calling analytical gradient
    '''

    dx = 0.002
    ndim = len(q)
    hess = np.zeros((ndim, ndim))

    for i in range(ndim):
        q[i] += dx

        gradp1 = -SparrowPy_Force(q, atoms, qcinput) #negative sign because it's force not gradient

        q[i] -= 2.0*dx

        gradm1 = -SparrowPy_Force(q, atoms, qcinput) #negative sign because it's force not gradient

        hess[i,:] = 0.5 * (gradp1 - gradm1) / dx

        q[i] += dx #restore partial coordinate

    return hess

