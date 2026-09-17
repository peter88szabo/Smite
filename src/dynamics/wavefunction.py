import os
import shutil


WAVEFUNCTION_DIR = "wavefunction_along_trajectory"


def prepare_wavefunction_directory(qchem, restart=False, wf_dir=WAVEFUNCTION_DIR):
    if restart or not qchem.get("wfu", False):
        return wf_dir

    if os.path.isdir(wf_dir):
        shutil.rmtree(wf_dir)
        os.makedirs(wf_dir)
        print(f"\n{wf_dir} already exisits, it has been deleted to create a new one")
    elif not os.path.exists(wf_dir):
        os.makedirs(wf_dir)
        print(f"\n{wf_dir} created")

    return wf_dir


def wavefunction_file(wf_dir, fname, istep):
    return os.path.join(wf_dir, "wavefunc_" + fname + "_step_" + str(istep) + ".molden")
