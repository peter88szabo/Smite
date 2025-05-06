import random
from smite import Molecule, Fragment, Collision
from qchem_interfaces.gp_modelmanager import get_modelmanager
from pathlib import Path
import os

def main():
    manager = get_modelmanager()
    manager.set_model_directory("/home/jenne/repos/Smite/peslib/H2O+Kr+")
    manager.load_models_from_directory()

    xyz_water = """
    O      0.000000    0.000000    0.000000
    H      0.586193    0.757215    0.000000
    H      0.586193   -0.757215    0.000000
    """

    qcinput = {"qchem": "PES", "wfu": ""}

    seed = 12949102
    random.seed(seed)

    constrained_bonds = [[0, 1], [0, 2], [1, 2]]

    water = Fragment.Polyatom_Init(
        fname="h2o", qchem=qcinput, xyz=xyz_water, random_rot=True, rigid=True, constrained_bonds=constrained_bonds, num_states=2, active_state=1
    )
    water.Specify_Mode_Sampling(init_rot_type="Jfix", jrot=3, rigid=True)

    krypton = Fragment.Atom_Init(fname="Kr", atoms=["Kr"], num_states=2, active_state=1)

    pairs_to_test = {
        "Habstr_1": [((0, 1), "GT", 8.0)],
        "Habstr_2": [((0, 2), "GT", 8.0)],
        "Nonreact": [((0, 1), "LT", 2.5), ((0, 2), "LT", 2.5), ((0, 3), "GT", 8.0)],
        # add more channels and pairs as needed
    }

    print("\nReaction of H2O + Cl\n")

    reaction = Collision(water, krypton, qchem=qcinput)
    reaction.Specify_Collision_Sampling(
        Rini=4., bmax=1.0, bsampling=True, Ecoll=20.0, temp=100.0
    )
    reaction.sample_and_run_collision(
        integrator="sprk",
        tolerance=1e-8,
        maxstep=4,
        timestep=1,
        iprint=1,
        pairs_to_stop=pairs_to_test,
        q_integrator="fssh",
        de_cutoff=0.1
    )

if __name__ == "__main__":
    main()

