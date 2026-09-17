"""Pure-Python script generation helpers for the Smite GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pprint import pformat


SUPPORTED_FRAGMENT_TYPES = ("Atom", "Diatom", "Polyatom")
SUPPORTED_QCHEM = ("XTB", "Orca", "PySCF", "Psi4", "Sparrow_bin", "Sparrow_Py", "PES")
INTEGRATORS = ("leapfrog", "verlet", "stormer", "rk4", "symplectic", "sprk", "predcorr")
THERMOSTATS = ("None", "berendsen", "andersen", "nosehoover", "gle")


@dataclass(frozen=True)
class QChemConfig:
    qchem: str = "XTB"
    path: str = ""
    nproc: int = 1
    functional: str = ""
    basis: str = ""
    charge: int = 0
    multiplicity: int = 1
    additional: str = ""
    wfu: bool = False
    scratch_dir: str = ""
    pes_path: str = ""

    def as_dict(self) -> dict:
        data = {
            "qchem": self.qchem,
            "path": self.path,
            "nproc": self.nproc,
            "functional": self.functional,
            "basis": self.basis,
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "additional": self.additional,
            "wfu": self.wfu,
        }
        if self.scratch_dir:
            data["scratch_dir"] = self.scratch_dir
        if self.pes_path:
            data["pes_path"] = self.pes_path
        return data


@dataclass(frozen=True)
class FragmentConfig:
    variable: str
    fname: str
    fragment_type: str = "Polyatom"
    atoms: str = "H,O"
    xyz: str = ""
    qchem: QChemConfig = field(default_factory=QChemConfig)
    req: float = 1.0
    omega: float = 1000.0
    diatom_model: str = "harmonic"
    beta: float = 1.0
    de: float = 100.0
    rigid: bool = False
    random_rot: bool = True
    linear: bool = False
    init_vib_type: str = "ZPE"
    init_rot_type: str = "Jfix"
    temp: float = 300.0
    nvib: int = 0
    jrot: int = 0


@dataclass(frozen=True)
class RunConfig:
    integrator: str = "leapfrog"
    integrator_order: int = 4
    timestep: float = 1.0
    maxstep: int = 1000
    iprint: int = 1
    rstop: float = 10.0
    spectrum: bool = False
    thermostat: str = "None"
    thermo_param: float = 100.0
    thermo_temp: float = 300.0
    traj_file: str = ""
    backfile: str = ""


@dataclass(frozen=True)
class CollisionConfig:
    rini: float = 5.0
    bmax: float = 5.0
    bsampling: int = 2
    ecoll: float = 10.0
    ecoll_thermal: bool = False
    temp: float = 300.0
    post_collision_analysis: bool = False


@dataclass(frozen=True)
class OptimizerConfig:
    atoms: str = "C,H,H,H,H"
    xyz: str = ""
    qchem: QChemConfig = field(default_factory=QChemConfig)
    method: str = "BFGS"
    backend_optimizer: str = "auto"
    maxstep: int = 100
    energy_tol: float = 5.0e-5
    max_step: float = 4.0e-3
    rms_step: float = 2.5e-3
    max_gradient: float = 7.0e-4
    rms_gradient: float = 5.0e-4
    trajectory_file: str = "geomopt_traj.xyz"
    print_report: bool = True
    orca_what: str = "minimum"


def _py_string(value: str) -> str:
    return repr(value)


def _atoms_literal(text: str) -> str:
    atoms = [item.strip() for item in text.replace("\n", ",").split(",") if item.strip()]
    return repr(atoms or ["H"])


def _qchem_block(name: str, config: QChemConfig) -> str:
    return f"{name} = {pformat(config.as_dict(), width=88)}"


def _fragment_block(config: FragmentConfig, qchem_name: str) -> str:
    var = config.variable
    fname = _py_string(config.fname)
    if config.fragment_type == "Atom":
        return (
            f"{var} = Fragment.Atom_Init(fname={fname}, atoms={_atoms_literal(config.atoms)})\n"
            f"{var}.Specify_Mode_Sampling(init_vib_type={_py_string(config.init_vib_type)}, "
            f"init_rot_type={_py_string(config.init_rot_type)}, jrot={config.jrot})"
        )
    if config.fragment_type == "Diatom":
        morse_args = ""
        if config.diatom_model == "morse":
            morse_args = f", beta={config.beta}, De={config.de}"
        return (
            f"{var} = Fragment.Diatom_Init(\n"
            f"    fname={fname},\n"
            f"    atoms={_atoms_literal(config.atoms)},\n"
            f"    req={config.req},\n"
            f"    omega={config.omega},\n"
            f"    rigid={config.rigid},\n"
            f"    random_rot={config.random_rot},\n"
            f"    diatom={_py_string(config.diatom_model)}{morse_args},\n"
            f")\n"
            f"{_sampling_call(var, config)}"
        )
    return (
        f"xyz_{var} = '''\n{config.xyz.rstrip()}\n'''\n\n"
        f"{var} = Fragment.Polyatom_Init(\n"
        f"    fname={fname},\n"
        f"    qchem={qchem_name},\n"
        f"    xyz=xyz_{var},\n"
        f"    rigid={config.rigid},\n"
        f"    random_rot={config.random_rot},\n"
        f"    linear={config.linear},\n"
        f")\n"
        f"{_sampling_call(var, config)}"
    )


def _sampling_call(var: str, config: FragmentConfig) -> str:
    args = [
        f"init_vib_type={_py_string(config.init_vib_type)}",
        f"init_rot_type={_py_string(config.init_rot_type)}",
    ]
    if config.init_vib_type == "Temp" or config.init_rot_type == "Temp":
        args.append(f"temp={config.temp}")
    if config.init_vib_type != "Temp" and config.fragment_type == "Diatom":
        args.append(f"nvib={config.nvib}")
    if config.init_rot_type == "Jfix":
        args.append(f"jrot={config.jrot}")
    return f"{var}.Specify_Mode_Sampling({', '.join(args)})"


def _run_kwargs(run: RunConfig) -> list[str]:
    kwargs = [
        f"integrator={_py_string(run.integrator)}",
        f"integrator_order={run.integrator_order}",
        f"timestep={run.timestep}",
        f"maxstep={run.maxstep}",
        f"iprint={run.iprint}",
        f"Rstop={run.rstop}",
        f"spectrum={run.spectrum}",
    ]
    if run.traj_file:
        kwargs.append(f"traj_file={_py_string(run.traj_file)}")
    if run.backfile:
        kwargs.append(f"backfile={_py_string(run.backfile)}")
    if run.thermostat != "None":
        kwargs.extend(
            [
                f"thermostat={_py_string(run.thermostat)}",
                f"thermo_param={run.thermo_param}",
                f"thermo_temp={run.thermo_temp}",
            ]
        )
    return kwargs


def _format_call(target: str, method: str, kwargs: list[str]) -> str:
    body = ",\n        ".join(kwargs)
    return f"{target}.{method}(\n        {body},\n    )"


def generate_unimolecular_script(fragment: FragmentConfig, run: RunConfig, seed: int | None = None) -> str:
    qchem_name = f"qcinput_{fragment.variable}"
    seed_block = f"    random.seed({seed})\n\n" if seed is not None else ""
    return (
        "if __name__ == '__main__':\n"
        "    import random\n"
        "    from smite import Fragment\n\n"
        f"{_indent(_qchem_block(qchem_name, fragment.qchem))}\n\n"
        f"{seed_block}"
        f"{_indent(_fragment_block(fragment, qchem_name))}\n\n"
        f"{_indent(_format_call(fragment.variable, 'sample_and_run_trajectory', _run_kwargs(run)))}\n"
    )


def generate_collision_script(
    fragment_a: FragmentConfig,
    fragment_b: FragmentConfig,
    qchem: QChemConfig,
    collision: CollisionConfig,
    run: RunConfig,
    seed: int | None = None,
) -> str:
    qchem_a = f"qcinput_{fragment_a.variable}"
    qchem_b = f"qcinput_{fragment_b.variable}"
    qchem_reaction = "qcinput_reaction"
    collision_args = [
        f"Rini={collision.rini}",
        f"bmax={collision.bmax}",
        f"bsampling={collision.bsampling}",
    ]
    if collision.ecoll_thermal:
        collision_args.extend(["Ecoll_thermal=True", f"temp={collision.temp}"])
    else:
        collision_args.append(f"Ecoll={collision.ecoll}")

    run_kwargs = _run_kwargs(run)
    if collision.post_collision_analysis:
        run_kwargs.append("post_collision_analysis=True")

    seed_block = f"    random.seed({seed})\n\n" if seed is not None else ""
    return (
        "if __name__ == '__main__':\n"
        "    import random\n"
        "    from smite import Fragment, Collision\n\n"
        f"{_indent(_qchem_block(qchem_a, fragment_a.qchem))}\n\n"
        f"{_indent(_qchem_block(qchem_b, fragment_b.qchem))}\n\n"
        f"{_indent(_qchem_block(qchem_reaction, qchem))}\n\n"
        f"{seed_block}"
        f"{_indent(_fragment_block(fragment_a, qchem_a))}\n\n"
        f"{_indent(_fragment_block(fragment_b, qchem_b))}\n\n"
        f"    reaction = Collision({fragment_a.variable}, {fragment_b.variable}, qchem={qchem_reaction})\n"
        f"    reaction.Specify_Collision_Sampling({', '.join(collision_args)})\n\n"
        "    # For channel-specific stopping, replace Rstop with pairs_to_stop={...} below.\n"
        f"{_indent(_format_call('reaction', 'sample_and_run_collision', run_kwargs))}\n"
    )


def generate_optimizer_script(config: OptimizerConfig) -> str:
    qchem_name = "qcinput"
    atoms = _atoms_literal(config.atoms)
    kwargs = [
        f"backend_optimizer={_py_string(config.backend_optimizer)}",
        f"method={_py_string(config.method)}",
        f"maxstep={config.maxstep}",
        f"energy_tol={config.energy_tol}",
        f"max_step={config.max_step}",
        f"rms_step={config.rms_step}",
        f"max_gradient={config.max_gradient}",
        f"rms_gradient={config.rms_gradient}",
        f"trajectory_file={_py_string(config.trajectory_file)}",
        f"print_report={config.print_report}",
        f"orca_what={_py_string(config.orca_what)}",
    ]
    optimizer_kwargs = ",\n        ".join(kwargs)
    return (
        "if __name__ == '__main__':\n"
        "    import numpy as np\n\n"
        "    from optimizer import optimize_geometry\n"
        "    from utils.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM\n"
        "    from utils.format_and_print import parseXYZ\n\n"
        f"{_indent(_qchem_block(qchem_name, config.qchem))}\n\n"
        f"    xyz = '''\n{_indent(config.xyz.rstrip())}\n    '''\n\n"
        f"    expected_atoms = {atoms}\n"
        "    _natoms, atoms, q_angstrom = parseXYZ(xyz)\n"
        "    if expected_atoms and atoms != expected_atoms:\n"
        "        print('Warning: atoms parsed from XYZ differ from the Atoms field.')\n"
        "    q = np.asarray(q_angstrom, dtype=float) * ANGSTROM_TO_BOHR\n\n"
        "    result = optimize_geometry(\n"
        f"        {qchem_name},\n"
        "        atoms,\n"
        "        q,\n"
        f"        {optimizer_kwargs},\n"
        "    )\n\n"
        "    print('converged:', result.converged)\n"
        "    print('steps:', result.nsteps)\n"
        "    print('energy [hartree]:', result.energy)\n"
        "    print('optimized geometry [Angstrom]:')\n"
        "    qopt = result.q * BOHR_TO_ANGSTROM\n"
        "    for i, atom in enumerate(result.atoms):\n"
        "        j = 3 * i\n"
        "        print(f'{atom:2s} {qopt[j]:16.8f} {qopt[j+1]:16.8f} {qopt[j+2]:16.8f}')\n"
    )


def _indent(text: str, spaces: int = 4) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line else "" for line in text.splitlines())
