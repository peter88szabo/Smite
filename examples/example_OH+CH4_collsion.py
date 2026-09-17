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
    pes_dir = src_dir.parent / "peslib" / "OH+CH4"

    qcinput_pes = {
        "qchem": "PES",
        "pes_name": "OH+CH4",
        "pes_path": str(pes_dir),
        "wfu": False,
        "hessian_dx": 0.002,
        # The atom ordering after Collision(CH4, OH) is C/H4/O/H(OH).
        # Keep this identity fixed throughout H abstraction.
        "oh_h_index": 6,
    }

    seed = 2222029
    random.seed(seed)

    # Tetrahedral methane near its gas-phase equilibrium C-H distance.
    # Atom indices after merging are C=0, H1=1, H2=2, H3=3, H4=4, O=5, H(OH)=6.
    xyz_CH4 = """
    C      0.0000000000      0.0000000000      0.0000000000
    H      0.6275790000      0.6275790000      0.6275790000
    H     -0.6275790000     -0.6275790000      0.6275790000
    H     -0.6275790000      0.6275790000     -0.6275790000
    H      0.6275790000     -0.6275790000     -0.6275790000
    """

    req_OH = 0.9707       # Angstrom
    omega_OH = 3737.76    # cm^-1

    temp_rot = 300.0      # K, for CH4 and OH rotational sampling
    Ecoll = 2.0          # kJ/mol
    Rini = 10.0           # Angstrom
    bmax = 4.0            # Angstrom

    methane = Fragment.Polyatom_Init(
        fname="CH4",
        qchem=qcinput_pes,
        xyz=xyz_CH4,
        random_rot=True,
    )
    methane.Specify_Mode_Sampling(init_vib_type="ZPE", init_rot_type="Temp", temp=temp_rot)

    hydroxyl = Fragment.Diatom_Init(
        fname="OH",
        atoms=["O", "H"],
        req=req_OH,
        omega=omega_OH,
        random_rot=True,
        diatom="harmonic",
    )
    hydroxyl.Specify_Mode_Sampling(init_rot_type="Temp", nvib=0, temp=temp_rot)

    print("------------------------ CH4 --------------------")
    print("Non-rigid methane initial geometry:")
    print("  C-H = 1.0870 Angstrom")
    methane.print_mode_sampling()
    print("------------------------ OH ---------------------")
    print("Harmonic oscillator:")
    print(f"  req   = {req_OH:.4f} Angstrom")
    print(f"  omega = {omega_OH:.2f} cm^-1")
    print("-------------------------------------------------\n")

    pairs_to_test = {
        "nonreactive_OH+CH4": [
            ((0, 5), "GT", 12.0),
            ((1, 5), "GT", 12.0),
            ((2, 5), "GT", 12.0),
            ((3, 5), "GT", 12.0),
            ((4, 5), "GT", 12.0),
            ((0, 6), "GT", 12.0),
        ],
        "H2O_from_H1+CH3": [
            ((1, 5), "LT", 1.25),
            ((5, 6), "LT", 1.30),
            ((0, 1), "GT", 1.8),
        ],
        "H2O_from_H2+CH3": [
            ((2, 5), "LT", 1.25),
            ((5, 6), "LT", 1.30),
            ((0, 2), "GT", 1.8),
        ],
        "H2O_from_H3+CH3": [
            ((3, 5), "LT", 1.25),
            ((5, 6), "LT", 1.30),
            ((0, 3), "GT", 1.8),
        ],
        "H2O_from_H4+CH3": [
            ((4, 5), "LT", 1.25),
            ((5, 6), "LT", 1.30),
            ((0, 4), "GT", 1.8),
        ],
    }

    print("\nReaction of OH + CH4 on OH+CH4 PES\n")

    reaction = Collision(methane, hydroxyl, qchem=qcinput_pes)
    reaction.Specify_Collision_Sampling(
        Rini=Rini,
        bmax=bmax,
        bsampling=1,  # linear impact-parameter sampling: b = bmax * random[0, 1]
        Ecoll=Ecoll,
    )

    reaction.sample_and_run_collision(
        integrator="verlet",
        timestep=0.5,
        maxstep=40000,
        iprint=4,
        pairs_to_stop=pairs_to_test,
    )
