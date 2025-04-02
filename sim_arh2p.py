import random
from smite import Molecule, Fragment, Collision
from qchem_interfaces.gp_modelmanager import get_modelmanager

def main():
    manager = get_modelmanager()
    manager.set_model_directory("/home/jenne/repos/Smite/peslib/ArH2p_gp")
    manager.load_models_from_directory()

    qcinput = {"qchem": "PES", "wfu": ""}

    seed = 12949103
    random.seed(seed)

    #constrained_bonds = [[0, 1]]

    hh = Fragment.Diatom_Init(
        fname="hh", atoms=["H", "H"], req=0.751, omega=4161, random_rot=True
    )
    hh.Specify_Mode_Sampling(init_rot_type="Temp", temp=300.0)

    ar = Fragment.Atom_Init(fname="ar", atoms=["Ar"])

    pairs_to_test = {
        "form arh": [((0, 1), "GT", 15.0), ((0,2), "LT", 1.3)],
        "form arh'": [((0, 1), "GT", 15.0), ((1,2), "LT", 1.3)],
        "complete dissociation": [((0, 1), "GT", 10.0), ((0, 2), "GT", 10.0), ((1, 2), "GT", 10.0)],
        "Nonreact": [((0, 1), "LT", 2.5), ((0, 2), "GT", 15.0), ((1, 2), "GT", 15.0)],
        # add more channels and pairs as needed
    }
    reaction = Collision(hh, ar, qchem=qcinput)
    reaction.Specify_Collision_Sampling(
        Rini=4., bmax=4.0, bsampling=True, Ecoll_thermal=True, temp=300.0
    )
    reaction.multi_paralell_traj_sample_and_run_collision(
        ntraj=100,
        cores_per_traj=4,
        integrator="symplectic",
        maxstep=100,
        timestep=0.1,
        iprint=1,
        pairs_to_stop=pairs_to_test,
    )

if __name__ == "__main__":
    main()

