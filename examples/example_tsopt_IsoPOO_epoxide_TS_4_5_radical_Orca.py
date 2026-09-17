from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == "__main__":
    USE_MODEL_HESSIAN = True
    INTERNAL_HESSIAN_MODEL = "model" if USE_MODEL_HESSIAN else "simple"
    TS_INITIAL_HESSIAN = "model" if USE_MODEL_HESSIAN else "exact"

    import math
    import numpy as np

    from normalmode import frequency_analysis
    from optimizer import OptimizerConfig, optimize_transition_state
    from utils.constants import BOHR_TO_ANGSTROM

    xyz_isopoo_epoxide_ts_guess = """
19
Conf: 12
  C   -1.17638568927240      1.11146838847380      3.04881808656764
  C   -0.42936203566709      1.89347064444358      4.06879940114204
  C   0.95166513785303      1.65517459672131      4.46972877372279
  H   1.47078181813630      0.79934234747642      4.01696065930384
  C   -2.49382814212771      1.71850923744815      2.62137737662779
  C   -0.38127171425958      0.52961473436911      1.88405075460219
  O   -1.10146141882968     -0.45771198780338      1.18833834712985
  H   -0.96961284309354      2.64623130635128      4.62660598854411
  O   1.50143554435219      2.36943169693685      5.27699198177717
  H   0.57041939471571      0.11178645692719      2.20890844094949
  H   -0.20015919350069      1.32830842752521      1.16097522933892
  H   -3.06833765986171      0.98164586935324      2.06535847054733
  H   -3.06377387106345      2.02845254900760      3.49514649334792
  H   -2.31175706495938      2.58117677734643      1.98098674469085
  O   -1.28973678483642      0.26956309533845      4.17319887831493
  O   -2.49029135087853     -0.89245222722169      3.90247584869018
  H   -2.48694023797212     -1.22490394071386      4.80983139297920
  O   -0.87906046708132     -1.70460240314428      1.82924758811902
  H   -1.52641642165359     -1.67824256883544      2.55882854360463
    """

    qcinput_orca_doublet = {
        "qchem": "Orca",
        "path": "/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/orca",
        "nproc": 4,
        "functional": "r2scan-3c",
        "basis": "",
        "charge": 0,
        "multiplicity": 2,
        "additional": "",
        "wfu": False,
        "scratch_dir": "scratch/IsoPOO_epoxide_TS_4_5_radical",
    }

    qcinput_xtb_doublet = {
        "qchem": "xtb",
        "path": "/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/xtb",
        "nproc": 4,
        "functional": "",
        "basis": "",
        "charge": 0,
        "multiplicity": 2,
        "additional": "",
        "wfu": False,
        "scratch_dir": "scratch/IsoPOO_epoxide_TS_4_5_radical",
    }

    qcinp = qcinput_xtb_doublet

    # User notation is 1-based:
    #   C2-O15 forming/shrinking bond -> Python indices (1, 14)
    #   C1-C2-O15 bending angle -> Python indices (0, 1, 14)
    # REACTION_MODE can be:
    #   combined: Baker follows one internal vector built from REACTION_COORDINATES.
    #   angle: Baker follows only REACTION_ANGLE.
    #   bond: Baker follows only REACTION_BOND.
    # In REACTION_COORDINATES, use tuple entries: (type, atoms) or
    # (type, atoms, weight).  If weight is omitted, the default is 0.5 before
    # normalizing the final combined vector.
    REACTION_MODE = "combined"  # choose "combined", "angle", or "bond"
    EXTRA_BONDS = [(1, 14)]
    EXTRA_ANGLES = [(0, 1, 14)]
    REACTION_BOND = (1, 14)
    REACTION_ANGLE = (0, 1, 14)
    REACTION_COORDINATES = [
        ("bond", REACTION_BOND),
        ("angle", REACTION_ANGLE, 0.5),
    ]

    def xyz_to_atoms_q_angstrom(xyz_text):
        lines = [line.strip() for line in xyz_text.splitlines() if line.strip()]
        try:
            natoms = int(lines[0].split()[0])
            coord_lines = lines[2:2 + natoms]
        except (ValueError, IndexError):
            coord_lines = lines
        atoms = []
        q_angstrom = []
        for line in coord_lines:
            fields = line.split()
            atoms.append(fields[0])
            q_angstrom.extend(float(value) for value in fields[1:4])
        return atoms, np.asarray(q_angstrom, dtype=float)

    def angle_degrees(coords, i, j, k):
        v1 = coords[i] - coords[j]
        v2 = coords[k] - coords[j]
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom <= 0.0:
            return float("nan")
        cosine = float(np.dot(v1, v2) / denom)
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    def print_reactive_coordinates(label, atoms, q_angstrom):
        del atoms
        coords = np.asarray(q_angstrom, dtype=float).reshape(-1, 3)
        c2_o15 = float(np.linalg.norm(coords[REACTION_BOND[0]] - coords[REACTION_BOND[1]]))
        c1_c2_o15 = angle_degrees(coords, *REACTION_ANGLE)
        print(f"{label} reactive coordinates:", flush=True)
        print(f"  C2-O15 distance [Angstrom]: {c2_o15:.6f}", flush=True)
        print(f"  C1-C2-O15 angle [degree]: {c1_c2_o15:.6f}", flush=True)

    print("IsoPOO epoxide TS-4-5 radical Baker TS optimization")
    print("charge:", qcinput_orca_doublet["charge"])
    print("multiplicity:", qcinput_orca_doublet["multiplicity"])
    print("method:", qcinput_orca_doublet["functional"], qcinput_orca_doublet["basis"])
    print("reaction_mode:", REACTION_MODE)
    print("forming/shrinking bond: C2-O15 in user notation, Python indices (1, 14)")
    print("forced extra bond: C2-O15, Python indices (1, 14)")
    print("forced extra angle: C1-C2-O15, Python indices (0, 1, 14)")
    print("OOH group: O15-O16-H17 in user notation, Python indices (14, 15, 16)")
    input_atoms, input_q_angstrom = xyz_to_atoms_q_angstrom(xyz_isopoo_epoxide_ts_guess)
    print_reactive_coordinates("Input TS guess", input_atoms, input_q_angstrom)
    print("Starting Baker TS optimization", flush=True)

    if REACTION_MODE == "combined":
        reaction_kwargs = {
            "reaction_mode": "combined",
            "reaction_coordinates": REACTION_COORDINATES,
            "extra_bonds": EXTRA_BONDS,
            "extra_angles": EXTRA_ANGLES,
        }
    elif REACTION_MODE == "angle":
        reaction_kwargs = {
            "reaction_mode": "angle",
            "reaction_angle": REACTION_ANGLE,
            "extra_bonds": EXTRA_BONDS,
            "extra_angles": EXTRA_ANGLES,
        }
    elif REACTION_MODE == "bond":
        reaction_kwargs = {
            "reaction_mode": "bond",
            "reaction_bond": REACTION_BOND,
            "extra_bonds": EXTRA_BONDS,
            "extra_angles": EXTRA_ANGLES,
        }
    else:
        raise ValueError("REACTION_MODE must be 'combined', 'angle', or 'bond'")

    result = optimize_transition_state(
        qcinp,
        xyz_isopoo_epoxide_ts_guess,
        settings=OptimizerConfig.from_dict({
            "backend_optimizer": "smite",
            "coordinates": "internal",
            "convergence": {
                "maxstep": 150,
                "energy_tol": 5.0e-6,
                "max_gradient": 3.0e-4,
                "rms_gradient": 1.0e-4,
                "max_step": 4.0e-3,
                "rms_step": 2.0e-3,
            },
            "hessian": {
                "hess_file": "hessian_IsoPOO_epoxide_TS_4_5_radical.hess",
                "hessian_recalc_interval": 10,
                "final_hessian": True,
                "internal_hessian_correction": False,
                "internal_hessian_model": INTERNAL_HESSIAN_MODEL,
                "ts_initial_hessian": TS_INITIAL_HESSIAN,
                "adaptive_hessian_recalc": False,
                "skip_hessian_recalc_near_convergence": True,
                "hessian_recalc_near_convergence_factor": 3.0,
                "adaptive_hessian_on_negative_mode_change": False,
                "repair_ts_hessian": True,
                "ts_hessian_eigenvalue_floor": 1.0e-4,
            },
            "trust": {
                "trust_radius": 0.06,
            },
            "internal_coordinates": {
                "max_step_internal": 0.10,
                "best_fit_rms_tol": 1.0e-7,
                "use_redundant_internals": True,
                "extra_bonds": reaction_kwargs.get("extra_bonds"),
                "extra_angles": reaction_kwargs.get("extra_angles"),
            },
            "reaction": {
                key: value
                for key, value in reaction_kwargs.items()
                if key.startswith("reaction_")
            } | {
                "mode_tracking_coordinates": "internal",
            },
            "reporting": {
                "trajectory_file": "tsopt_IsoPOO_epoxide_TS_4_5_radical_Orca.xyz",
            },
            "project_eckart": True,
        }),
    )

    print("Finished Baker TS optimization", flush=True)
    print("coordinates:", result.coordinates)
    print("converged:", result.converged)
    print("steps:", result.nsteps)
    print("energy [hartree]:", result.energy)
    print("imaginary modes:", result.negative_modes)
    print("message:", result.message)
    print("optimized TS candidate geometry [Angstrom]:")

    qopt_angstrom = result.q * BOHR_TO_ANGSTROM
    for i, atom in enumerate(result.atoms):
        j = 3 * i
        print(f"{atom:2s} {qopt_angstrom[j]:16.8f} {qopt_angstrom[j + 1]:16.8f} {qopt_angstrom[j + 2]:16.8f}")
    print_reactive_coordinates("Optimized TS candidate", result.atoms, qopt_angstrom)

    print("Starting frequency analysis for IsoPOO epoxide TS-4-5 radical", flush=True)
    frequency_analysis(
        qcinput=qcinp,
        atoms=result.atoms,
        q=result.q,
        fname="IsoPOO_epoxide_TS_4_5_radical",
        hessFile="hessian_IsoPOO_epoxide_TS_4_5_radical_final_freq.hess",
        is_eckart=True,
        force_hessian_recalc=True,
        Amp_modeanim=30.0,
        print_nmode=True,
        qrrho_cutoff=50.0,
    )
    print("Finished frequency analysis for IsoPOO epoxide TS-4-5 radical", flush=True)
