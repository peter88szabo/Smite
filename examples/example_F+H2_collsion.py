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
    pes_dir = src_dir.parent / "peslib" / "F+H2"

    qcinput_pes = {
        "qchem": "PES",
        "pes_name": "F+H2",
        "pes_path": str(pes_dir),
        "wfu": False,
        "hessian_dx": 0.002,
    }

    seed = 2122227
    random.seed(seed)

    # Literature gas-phase H2 constants, used here as a harmonic rotating
    # oscillator.
    req_H2 = 0.7414       # Angstrom
    omega_H2 = 4401.21    # cm^-1

    temp_rot = 300.0      # K, for H2 rotational sampling
    Ecoll = 1.0          # kJ/mol
    Rini = 10.0           # Angstrom
    bmax = 2.0            # Angstrom

    hydrogen = Fragment.Diatom_Init(
        fname="H2",
        atoms=["H", "H"],
        req=req_H2,
        omega=omega_H2,
        random_rot=True,
        diatom="harmonic",
    )
    hydrogen.Specify_Mode_Sampling(init_rot_type="Temp", nvib=0, temp=temp_rot)

    fluorine = Fragment.Atom_Init(fname="F", atoms=["F"])

    print("------------------------ H2 ---------------------")
    print("Harmonic oscillator:")
    print(f"  req   = {req_H2:.4f} Angstrom")
    print(f"  omega = {omega_H2:.2f} cm^-1")
    hydrogen.print_mode_sampling()
    print("-------------------------------------------------\n")

    # Atom indices in the merged Collision object are H(1)=0, H(2)=1, F=2.
    # Nonreactive is checked only after outgoing F is farther than Rini from
    # both H atoms; otherwise this channel would be true at the initial geometry.
    # Do not constrain the H-H distance here: a flyby with vibrational/rotational
    # energy transfer is still nonreactive if no FH bond is formed.
    pairs_to_test = {
        "nonreactive_F+H2": [
            ((0, 2), "GT", 12.0),
            ((1, 2), "GT", 12.0),
        ],
        "FH(1)+H(2)": [
            ((0, 2), "LT", 1.35),
            ((0, 1), "GT", 2.5),
            ((1, 2), "GT", 2.5),
        ],
        "FH(2)+H(1)": [
            ((1, 2), "LT", 1.35),
            ((0, 1), "GT", 2.5),
            ((0, 2), "GT", 2.5),
        ],
    }

    print("\nReaction of F + H2 on F+H2 PES\n")

    reaction = Collision(hydrogen, fluorine, qchem=qcinput_pes)
    reaction.Specify_Collision_Sampling(
        Rini=Rini,
        bmax=bmax,
        bsampling=1,  # linear impact-parameter sampling: b = bmax * random[0, 1]
        Ecoll=Ecoll,
    )

    reaction.sample_and_run_collision(
        integrator="verlet",
        timestep=0.5,
        maxstep=10000,
        iprint=2,
        pairs_to_stop=pairs_to_test,
    )
