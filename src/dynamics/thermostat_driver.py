import numpy as np

from thermostats.berendsen import thermo_berendsen
from thermostats.andersen import thermo_andersen
from thermostats.nosehoover import thermo_nosehoover
from thermostats.gle import make_gle_thermostat, thermo_gle
from utils.constants import FS_TO_AU_TIME, R_GAS_HARTREE_PER_K


SUPPORTED_THERMOSTATS = {"berendsen", "andersen", "nosehoover", "gle"}


def validate_thermostat_input(thermostat, thermo_param, thermo_temp):
    if thermostat is not None and thermostat not in SUPPORTED_THERMOSTATS:
        raise ValueError("Non existing thermostat. Choose from: berendsen, andersen, nosehoover, gle")

    if thermostat is not None and (thermo_param is None or thermo_temp is None):
        raise ValueError("Since thermostat is switched on the parameter and temperature must be given")


def initialize_gle_state(molecule, dt, thermo_param, thermo_temp):
    if thermo_temp is None:
        raise ValueError("GLE thermostat requires thermo_temp in Kelvin")

    if isinstance(thermo_param, dict):
        if "wopt" not in thermo_param:
            raise ValueError("GLE thermo_param dictionary must contain key 'wopt'")
        wopt = float(thermo_param["wopt"])
        a_file = thermo_param.get("a_file", thermo_param.get("A_file", "GLE-A"))
        c_file = thermo_param.get("c_file", thermo_param.get("C_file", "GLE-C"))
        seed = thermo_param.get("seed")
    else:
        if thermo_param is None:
            raise ValueError("GLE thermostat requires thermo_param=wopt or thermo_param={'wopt': ...}")
        wopt = float(thermo_param)
        a_file = "GLE-A"
        c_file = "GLE-C"
        seed = None

    kt = R_GAS_HARTREE_PER_K * thermo_temp
    rng = np.random.default_rng(seed) if seed is not None else None
    molecule._gle_state = make_gle_thermostat(
        dt=dt,
        wopt=wopt,
        kt=kt,
        ndim=len(molecule.p),
        a_file=a_file,
        c_file=c_file,
        rng=rng,
    )


def prepare_thermostat(molecule, thermostat, thermo_param, thermo_temp, dt, restart=False):
    validate_thermostat_input(thermostat, thermo_param, thermo_temp)

    if thermostat == "nosehoover" and not restart:
        molecule._nosehoover_state = None
    elif thermostat == "gle" and (not restart or molecule._gle_state is None):
        initialize_gle_state(molecule, dt, thermo_param, thermo_temp)


def apply_thermostat(molecule, thermostat, thermo_param, thermo_temp, dt):
    has_constraints = getattr(molecule, "has_rigid_constraints", False)
    removed_dof = (
        molecule.thermostat_removed_dof() if thermostat is not None else 0
    )
    if thermostat == "berendsen":
        tau = thermo_param * FS_TO_AU_TIME
        molecule.p = thermo_berendsen(removed_dof, molecule.p, molecule.wmass, dt, tau, thermo_temp)
    elif thermostat == "andersen":
        collision_time = thermo_param * FS_TO_AU_TIME
        molecule.p = thermo_andersen(removed_dof, molecule.p, molecule.wmass, dt, collision_time, thermo_temp)
    elif thermostat == "nosehoover":
        tau = thermo_param * FS_TO_AU_TIME
        molecule.p, molecule._nosehoover_state = thermo_nosehoover(
            removed_dof,
            molecule.p,
            molecule.wmass,
            dt,
            tau,
            thermo_temp,
            molecule._nosehoover_state,
            scale_all=has_constraints,
        )
    elif thermostat == "gle":
        molecule.p = thermo_gle(removed_dof, molecule.p, molecule.wmass, molecule._gle_state)

    if thermostat is not None and getattr(molecule, "remove_com", False):
        molecule.project_center_of_mass_momentum()

    if (
        thermostat is not None
        and getattr(molecule, "has_rigid_constraints", False)
        and molecule._constraint_algorithm == "rattle"
    ):
        molecule.project_rigid_momenta()

    return molecule.p
