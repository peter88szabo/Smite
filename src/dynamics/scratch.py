import os
import re
import shutil

from qchem_interfaces.qchem_validation import validate_qchem_input


def trajectory_scratch_dir(fname, traj_file):
    traj_name = os.path.splitext(os.path.basename(traj_file))[0]
    safe_name = re.sub(r"[^A-Za-z0-9_.+-]+", "_", traj_name).strip("_")
    if not safe_name:
        safe_name = fname or "trajectory"
    return os.path.join("scratch", safe_name)


def set_trajectory_scratch_dir(molecule, traj_file, restart=False):
    if not isinstance(molecule.qchem, dict):
        return

    scratch_root = trajectory_scratch_dir(molecule.fname, traj_file)
    os.makedirs(scratch_root, exist_ok=True)
    molecule.qchem["scratch_dir"] = scratch_root
    molecule.qchem = validate_qchem_input(molecule.qchem)

    if not restart and molecule.qchem["qchem"] in {"XTB", "Orca"}:
        backend_dir = os.path.join(scratch_root, molecule.qchem["qchem"].lower())
        if os.path.isdir(backend_dir):
            shutil.rmtree(backend_dir)
