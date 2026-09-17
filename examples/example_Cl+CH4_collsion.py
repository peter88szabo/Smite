from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == "__main__":
    import random
    from pathlib import Path

    from smite import Fragment, Collision

    src_dir = SRC_DIR
    pes_dir = src_dir.parent / "peslib" / "Cl+CH4"

    qcinput_pes = {
        "qchem": "PES",
        "pes_name": "Cl+CH4",
        "pes_path": str(pes_dir),
        "wfu": False,
        "hessian_dx": 0.002,
    }

    seed = 2222028
    random.seed(seed)

    # Tetrahedral methane near its gas-phase equilibrium C-H distance.
    # Atom indices after merging are C=0, H1=1, H2=2, H3=3, H4=4, Cl=5.
    xyz_CH4 = """
    C      0.0000000000      0.0000000000      0.0000000000
    H      0.6275790000      0.6275790000      0.6275790000
    H     -0.6275790000     -0.6275790000      0.6275790000
    H     -0.6275790000      0.6275790000     -0.6275790000
    H      0.6275790000     -0.6275790000     -0.6275790000
    """

    temp_rot = 300.0      # K, for CH4 rotational sampling
    Ecoll = 50.0          # kJ/mol
    Rini = 10.0           # Angstrom
    bmax = 3.0            # Angstrom

    methane = Fragment.Polyatom_Init(
        fname="CH4",
        qchem=qcinput_pes,
        xyz=xyz_CH4,
        random_rot=True,
    )
    methane.Specify_Mode_Sampling(init_vib_type="ZPE", init_rot_type="Temp", temp=temp_rot)

    chlorine = Fragment.Atom_Init(fname="Cl", atoms=["Cl"])

    print("------------------------ CH4 --------------------")
    print("Non-rigid methane initial geometry:")
    print("  C-H = 1.0870 Angstrom")
    methane.print_mode_sampling()
    print("-------------------------------------------------\n")

    pairs_to_test = {
        "nonreactive_Cl+CH4": [
            ((1, 5), "GT", 12.0),
            ((2, 5), "GT", 12.0),
            ((3, 5), "GT", 12.0),
            ((4, 5), "GT", 12.0),
            ((0, 5), "GT", 12.0),
        ],
        "HCl(1)+CH3": [
            ((1, 5), "LT", 1.6),
            ((0, 1), "GT", 1.8),
        ],
        "HCl(2)+CH3": [
            ((2, 5), "LT", 1.6),
            ((0, 2), "GT", 1.8),
        ],
        "HCl(3)+CH3": [
            ((3, 5), "LT", 1.6),
            ((0, 3), "GT", 1.8),
        ],
        "HCl(4)+CH3": [
            ((4, 5), "LT", 1.6),
            ((0, 4), "GT", 1.8),
        ],
    }

    print("\nReaction of Cl + CH4 on Cl+CH4 PES\n")

    reaction = Collision(methane, chlorine, qchem=qcinput_pes)
    reaction.Specify_Collision_Sampling(
        Rini=Rini,
        bmax=bmax,
        bsampling=1,  # linear impact-parameter sampling: b = bmax * random[0, 1]
        Ecoll=Ecoll,
    )

    reaction.sample_and_run_collision(
        integrator="verlet",
        timestep=0.2,
        maxstep=10000,
        iprint=4,
        pairs_to_stop=pairs_to_test,
    )
