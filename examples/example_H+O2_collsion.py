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
    pes_dir = src_dir.parent / "peslib" / "HO2_3Sigma_negative"

    # H + O2 on the HO2 3Sigma- PES. The PES plugin lives in repo-level peslib and
    # expects one H atom and two O atoms; it reorders atoms internally if needed.
    qcinput_pes = {
        "qchem": "PES",
        "pes_name": "HO2_3Sigma_negative",
        "pes_path": str(pes_dir),
        "wfu": False,
        "hessian_dx": 0.002,
    }

    seed = 5102026
    random.seed(seed)

    # Literature gas-phase O2(X 3Sigma_g-) constants, used here as a harmonic
    # rotating oscillator.
    req_O2 = 1.2075       # Angstrom
    omega_O2 = 1580.19    # cm^-1

    temp_rot = 300.0      # K, for O2 rotational sampling
    Ecoll = 10.0          # kJ/mol
    Rini = 10.0           # Angstrom
    bmax = 2.0            # Angstrom

    oxygen = Fragment.Diatom_Init(
        fname="O2",
        atoms=["O", "O"],
        req=req_O2,
        omega=omega_O2,
        random_rot=True,
        diatom="harmonic",
    )
    oxygen.Specify_Mode_Sampling(init_rot_type="Temp", nvib=0, temp=temp_rot)

    hydrogen = Fragment.Atom_Init(fname="H", atoms=["H"])

    print("------------------------ O2 ---------------------")
    print("Harmonic oscillator:")
    print(f"  req   = {req_O2:.4f} Angstrom")
    print(f"  omega = {omega_O2:.2f} cm^-1")
    oxygen.print_mode_sampling()
    print("-------------------------------------------------\n")

    # Atom indices in the merged Collision object are O(1)=0, O(2)=1, H=2.
    # Nonreactive is checked only after the outgoing H is farther than Rini,
    # otherwise this channel would be true at the initial geometry.
    pairs_to_test = {
        "nonreactive_H+O2": [
            ((0, 1), "LT", 1.8),
            ((0, 2), "GT", 12.0),
            ((1, 2), "GT", 12.0),
        ],
        "HO(1)+O(2)": [
            ((0, 2), "LT", 1.35),
            ((0, 1), "GT", 2.5),
            ((1, 2), "GT", 2.5),
        ],
        "HO(2)+O(1)": [
            ((1, 2), "LT", 1.35),
            ((0, 1), "GT", 2.5),
            ((0, 2), "GT", 2.5),
        ],
    }

    print("\nReaction of H + O2 on HO2_3Sigma_negative PES\n")

    reaction = Collision(oxygen, hydrogen, qchem=qcinput_pes)
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
        iprint=4,
        pairs_to_stop=pairs_to_test,
    )
