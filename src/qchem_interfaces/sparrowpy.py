import os
import numpy as np
from utils.constants import BOHR_TO_ANGSTROM
from qchem_interfaces.backend_common import backend_scratch_dir
from qchem_interfaces.energy_cache import store_energy
from qchem_interfaces.numerical_hessian import central_difference_hessian
from qchem_interfaces.wavefunction_output import prepare_wavefunction_output


def print_structure(atoms, q, filename):
    with open(filename, "w") as file:
        file.write(str(len(atoms)) + "\n")
        file.write("This is a temporary strucutre for Sparrow to calculate energy and gradient\n")
        for i in range(0, len(atoms)):
            jx = 3 * i
            jy = 3 * i + 1
            jz = 3 * i + 2
            file.write("%3s %15.5f  %15.5f %15.5f\n" % (atoms[i], q[jx] * BOHR_TO_ANGSTROM, q[jy] * BOHR_TO_ANGSTROM, q[jz] * BOHR_TO_ANGSTROM))


def SparrowPy_Energy(filename, q, atoms, qcinput):
    import scine_utilities as su
    import scine_sparrow

    method       =   qcinput['functional']
    basis        =   qcinput['basis']
    charge       =   qcinput['charge']
    multiplicity =   qcinput['multiplicity']
    wfu          =   qcinput['wfu']
    directory    =   backend_scratch_dir(qcinput, 'sparrowpy', 'sparrowpy_tmp')

    geomfile = os.path.join(directory, 'geomfile_to_read_by_Sparrow.xyz')

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
        filename = prepare_wavefunction_output(filename, "Sparrow_Py")
        wf_gen = su.core.to_wf_generator(calculator)
        wf_gen.wavefunction2file(filename)
   #------------------------------------------------------------------------------

    return E


def SparrowPy_Force(q, atoms, qcinput):
    import scine_utilities as su
    import scine_sparrow


    method       =   qcinput['functional']
    basis        =   qcinput['basis']
    charge       =   qcinput['charge']
    multiplicity =   qcinput['multiplicity']
    directory    =   backend_scratch_dir(qcinput, 'sparrowpy', 'sparrowpy_tmp')

    geomfile = os.path.join(directory, 'geomfile_to_read_by_Sparrow.xyz')

    print_structure(atoms, q, geomfile)

    manager = su.core.ModuleManager.get_instance()
    calculator = manager.get('calculator', method)
    calculator.structure = su.io.read(geomfile)[0]
    calculator.set_required_properties([su.Property.Energy, su.Property.Gradients])

    calculator.settings['molecular_charge'] = charge
    calculator.settings['spin_multiplicity'] = multiplicity


    if multiplicity != 1:
        calculator.settings['spin_mode'] = 'unrestricted'

    results = calculator.calculate()

    force = -np.array(results.gradients.flatten())
    store_energy(qcinput, q, atoms, results.energy, force=force)

    return force


def SparrowPy_Hessian(q, atoms, qcinput):
    return central_difference_hessian(
        q,
        lambda coordinates: -SparrowPy_Force(coordinates, atoms, qcinput),
        dx=qcinput.get("hessian_dx", 0.002),
    )
