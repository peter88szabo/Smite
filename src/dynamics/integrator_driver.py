from integrators.stormerverlet import stormer_verlet
from integrators.leapfrog import leapfrog
from integrators.verlet import velverlet
from integrators.rungekutta import rk4
from integrators.symplectic import Symplectic
from integrators.sprk import SPRK
from integrators.predcorr import PredCorr
from integrators.rattle import constrained_velocity_verlet


def initialize_integrator(name, order, ndim):
    if name == "predcorr":
        return PredCorr(order, ndim)
    if name == "symplectic":
        return Symplectic(order)
    if name == "sprk":
        return SPRK(order)
    return None


def apply_integrator(molecule, name, dt, propagator=None):
    if getattr(molecule, "has_rigid_constraints", False):
        if molecule._constraint_solver is None:
            molecule.prepare_rigid_constraints(
                algorithm=molecule._constraint_algorithm,
                position_tolerance=molecule._constraint_position_tolerance,
                velocity_tolerance=molecule._constraint_velocity_tolerance,
                max_iterations=molecule._constraint_max_iterations,
            )
        if name not in {"verlet", "leapfrog"}:
            raise ValueError(
                "Rigid SHAKE/RATTLE propagation currently supports only the "
                "verlet and leapfrog integrators"
            )
        molecule.q, molecule.p = constrained_velocity_verlet(
            molecule.qchem,
            dt,
            molecule.wmass,
            molecule.q,
            molecule.p,
            molecule.atoms,
            molecule._constraint_solver,
            apply_rattle=molecule._constraint_algorithm == "rattle",
        )
    elif name == "stormer":
        molecule.q, molecule.p = stormer_verlet(molecule.qchem, dt, molecule.wmass, molecule.q, molecule.p, molecule.atoms)
    elif name == "verlet":
        molecule.q, molecule.p = velverlet(molecule.qchem, dt, molecule.wmass, molecule.q, molecule.p, molecule.atoms)
    elif name == "leapfrog":
        molecule.q, molecule.p = leapfrog(molecule.qchem, dt, molecule.wmass, molecule.q, molecule.p, molecule.atoms)
    elif name == "rk4":
        molecule.q, molecule.p = rk4(molecule.qchem, dt, molecule.wmass, molecule.q, molecule.p, molecule.atoms)
    elif name == "predcorr":
        molecule.q, molecule.p = propagator.predcorr(molecule.qchem, dt, molecule.wmass, molecule.q, molecule.p, molecule.atoms)
    elif name == "symplectic":
        molecule.q, molecule.p = propagator.symplectic(molecule.qchem, dt, molecule.wmass, molecule.q, molecule.p, molecule.atoms)
    elif name == "sprk":
        molecule.q, molecule.p = propagator.sprk(molecule.qchem, dt, molecule.wmass, molecule.q, molecule.p, molecule.atoms)
    else:
        raise ValueError("Non existing integrator. You can choose from: leapfrog, verlet, rk4, symplectic(4,6,8) and predcorr(order)")
