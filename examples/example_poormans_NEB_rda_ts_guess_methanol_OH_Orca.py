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
    from optimizer import OptimizerConfig, generate_rda_ts_guess, optimize_geometry, optimize_transition_state
    from utils.constants import BOHR_TO_ANGSTROM

    PREOPTIMIZE_ENDPOINTS = False #True
    USE_POORMANS_NEB = True
    ENDPOINT_PREOPT_MAXSTEPS = 150
    ENDPOINT_SAME_RMSD_TOL_ANGSTROM = 0.05

    # Atom order:
    #   0 C, 1-3 methyl H, 4 incoming OH oxygen, 5 OH hydrogen,
    #   6 methanol oxygen, 7 transferred hydrogen.
    # User notation is 1-based:
    #   breaking O7-H8 bond -> Python indices (6, 7)
    #   forming O5-H8 bond -> Python indices (4, 7)
    #   transfer angle O7-H8-O5 -> Python indices (6, 7, 4)
    # REACTION_MODE controls only the Baker TS search after the poormans_NEB
    # guess.  The poormans_NEB constrained relaxation always uses the reactive
    # bonds and angle below as path coordinates.
    #   combined: follow a weighted internal vector from REACTION_COORDINATES.
    #   transfer: follow the Cartesian atom-transfer vector.
    #   bond: follow only the forming O5-H8 bond.
    # In REACTION_COORDINATES, use tuple entries: (type, atoms) or
    # (type, atoms, weight).  Missing weight means 0.5 before final
    # normalization.
    REACTION_MODE = "combined"  # choose "combined", "transfer", or "bond"
    BREAKING_BOND = (6, 7)
    FORMING_BOND = (4, 7)
    TRANSFER_ANGLE = (6, 7, 4)
    REACTION_TRANSFER = (6, 7, 4)
    EXTRA_BONDS = [FORMING_BOND]
    EXTRA_ANGLES = [TRANSFER_ANGLE]
    REACTION_COORDINATES = [
        ("bond", BREAKING_BOND),
        ("bond", FORMING_BOND),
        ("angle", TRANSFER_ANGLE, 0.5),
    ]

    xyz_reactant = """
    C     0.000000     0.000000     0.000000
    H     0.000000     0.000000     1.093039
    H     1.035956     0.000000    -0.347715
    H    -0.508115     0.897276    -0.359678
    O    -0.670168    -1.005520    -3.050000
    H    -0.166061    -1.826041    -3.158649
    O    -0.678016    -1.182640    -0.386696
    H    -1.017248    -1.076029    -1.419596
    """

    xyz_product = """
    C     0.000000     0.000000     0.000000
    H     0.000000     0.000000     1.093039
    H     1.035956     0.000000    -0.347715
    H    -0.508115     0.897276    -0.359678
    O    -0.670168    -1.005520    -2.430000
    H    -0.166061    -1.826041    -2.538649
    O    -0.678016    -1.182640    -0.386696
    H    -0.778000    -1.035000    -1.475000
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
        "scratch_dir": "scratch/methanol_OH_poormans_NEB_RDA_TS",
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
        "scratch_dir": "scratch/methanol_OH_poormans_NEB_RDA_TS",
    }

    qcinp = qcinput_orca_doublet

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

    def atoms_q_to_xyz(atoms, q_angstrom, comment):
        lines = [str(len(atoms)), comment]
        for i, atom in enumerate(atoms):
            j = 3 * i
            lines.append(
                f"{atom:2s} {q_angstrom[j]:16.10f} {q_angstrom[j + 1]:16.10f} {q_angstrom[j + 2]:16.10f}"
            )
        return "\n".join(lines)

    def result_to_xyz(result, comment):
        return atoms_q_to_xyz(result.atoms, result.q * BOHR_TO_ANGSTROM, comment)

    def endpoint_rmsd_angstrom(result_a, result_b):
        if result_a.atoms != result_b.atoms:
            raise ValueError("Cannot compare endpoints with different atom labels/order")
        diff = ((result_a.q - result_b.q) * BOHR_TO_ANGSTROM).reshape(-1, 3)
        return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))

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
        breaking = float(np.linalg.norm(coords[BREAKING_BOND[0]] - coords[BREAKING_BOND[1]]))
        forming = float(np.linalg.norm(coords[FORMING_BOND[0]] - coords[FORMING_BOND[1]]))
        angle = angle_degrees(coords, *TRANSFER_ANGLE)
        print(f"{label} reactive coordinates:", flush=True)
        print(f"  O7-H8 breaking distance [Angstrom]: {breaking:.6f}", flush=True)
        print(f"  O5-H8 forming distance [Angstrom]: {forming:.6f}", flush=True)
        print(f"  O7-H8-O5 angle [degree]: {angle:.6f}", flush=True)

    reactant_atoms, reactant_q_angstrom = xyz_to_atoms_q_angstrom(xyz_reactant)
    product_atoms, product_q_angstrom = xyz_to_atoms_q_angstrom(xyz_product)

    print("Methanol + OH poormans_NEB RDA + Baker TS optimization")
    print("charge:", qcinp["charge"])
    print("multiplicity:", qcinp["multiplicity"])
    print("method:", qcinp["functional"], qcinp["basis"])
    print("RDA reactive bonds: O7-H8 and O5-H8 in user notation, Python indices (6, 7) and (4, 7)")
    print("RDA/TS transfer angle: O7-H8-O5 in user notation, Python indices (6, 7, 4)")
    print("Baker TS reaction_mode:", REACTION_MODE)
    print("forced extra bonds:", EXTRA_BONDS)
    print("forced extra angles:", EXTRA_ANGLES)
    print_reactive_coordinates("Initial reactant endpoint guess", reactant_atoms, reactant_q_angstrom)
    print_reactive_coordinates("Initial product endpoint guess", product_atoms, product_q_angstrom)

    if PREOPTIMIZE_ENDPOINTS:
        print("Starting full reactant pre-optimization", flush=True)
        reactant_preopt = optimize_geometry(
            qcinp,
            xyz_reactant,
            settings=OptimizerConfig.from_dict({
                "backend_optimizer": "smite",
                "coordinates": "internal",
                "method": "BFGS",
                "convergence": {"maxstep": ENDPOINT_PREOPT_MAXSTEPS},
                "internal_coordinates": {
                    "use_redundant_internals": True,
                    "max_step_internal": 0.08,
                    "extra_bonds": EXTRA_BONDS,
                    "extra_angles": EXTRA_ANGLES,
                },
                "hessian": {"internal_hessian_model": INTERNAL_HESSIAN_MODEL},
                "reporting": {
                    "trajectory_file": "preopt_methanol_OH_reactant_poormans_NEB.xyz",
                    "print_report": True,
                },
            }),
        )
        print("Finished reactant pre-optimization", flush=True)
        print("reactant pre-optimization converged:", reactant_preopt.converged, flush=True)
        print("reactant pre-optimization steps:", reactant_preopt.nsteps, flush=True)
        print_reactive_coordinates(
            "Pre-optimized reactant endpoint",
            reactant_preopt.atoms,
            reactant_preopt.q * BOHR_TO_ANGSTROM,
        )

        print("Starting full product pre-optimization", flush=True)
        product_preopt = optimize_geometry(
            qcinp,
            xyz_product,
            settings=OptimizerConfig.from_dict({
                "backend_optimizer": "smite",
                "coordinates": "internal",
                "method": "BFGS",
                "convergence": {"maxstep": ENDPOINT_PREOPT_MAXSTEPS},
                "internal_coordinates": {
                    "use_redundant_internals": True,
                    "max_step_internal": 0.08,
                    "extra_bonds": EXTRA_BONDS,
                    "extra_angles": EXTRA_ANGLES,
                },
                "hessian": {"internal_hessian_model": INTERNAL_HESSIAN_MODEL},
                "reporting": {
                    "trajectory_file": "preopt_methanol_OH_product_poormans_NEB.xyz",
                    "print_report": True,
                },
            }),
        )
        print("Finished product pre-optimization", flush=True)
        print("product pre-optimization converged:", product_preopt.converged, flush=True)
        print("product pre-optimization steps:", product_preopt.nsteps, flush=True)
        print_reactive_coordinates(
            "Pre-optimized product endpoint",
            product_preopt.atoms,
            product_preopt.q * BOHR_TO_ANGSTROM,
        )

        endpoint_rmsd = endpoint_rmsd_angstrom(reactant_preopt, product_preopt)
        print(f"pre-optimized endpoint RMSD [Angstrom]: {endpoint_rmsd:.6f}", flush=True)
        if endpoint_rmsd < ENDPOINT_SAME_RMSD_TOL_ANGSTROM:
            raise RuntimeError(
                "Reactant and product pre-optimizations converged to the same structure "
                f"within {ENDPOINT_SAME_RMSD_TOL_ANGSTROM:.3f} Angstrom RMSD. "
                "RDA needs two distinct endpoint minima."
            )

        xyz_reactant_for_rda = result_to_xyz(reactant_preopt, "pre-optimized RDA reactant endpoint")
        xyz_product_for_rda = result_to_xyz(product_preopt, "pre-optimized RDA product endpoint")
    else:
        print("Endpoint pre-optimization is switched off", flush=True)
        xyz_reactant_for_rda = xyz_reactant
        xyz_product_for_rda = xyz_product

    print("Starting RDA quasi-TS guess generation with poormans_NEB constrained relaxation", flush=True)
    rda_guess = generate_rda_ts_guess(
        qcinp,
        xyz_reactant_for_rda,
        xyz_product_for_rda,
        active_atoms=(4, 6, 7),
        align_atoms=(0, 1, 2, 3, 6),
        distance="reactive_internal",
        reactive_bonds=[BREAKING_BOND, FORMING_BOND],
        reactive_angles=[TRANSFER_ANGLE],
        reactive_dihedrals=[],
        conditional_coordinates="internal",
        use_redundant_internals=True,
        internal_hessian_model=INTERNAL_HESSIAN_MODEL,
        max_conditional_steps=20,
        max_step_internal=0.08,
        first_energy_tol_ev=0.01,
        later_energy_tol_ev=0.05,
        distance_tol=0.05,
        print_report=True,
        rda_trajectory_file="rda_methanol_OH_poormans_NEB_detailed.xyz",
        rda_chain_file="rda_methanol_OH_poormans_NEB_chain.xyz",
        endpoint_same_rmsd_tol=ENDPOINT_SAME_RMSD_TOL_ANGSTROM,
        endpoint_active_rmsd_tol=ENDPOINT_SAME_RMSD_TOL_ANGSTROM,
        poormans_NEB=USE_POORMANS_NEB,
        poormans_NEB_images=7,
        poormans_NEB_maxiter=100,
        poormans_NEB_file="poormans_NEB_methanol_OH_traj_from_constrained_RDA.xyz",
        poormans_NEB_profile_file="poormans_NEB_methanol_OH_energetics_profile.dat",
        poormans_NEB_plot_file="poormans_NEB_methanol_OH_energetics_profile.png",
    )
    print("Finished RDA quasi-TS guess generation", flush=True)
    print("RDA message:", rda_guess.message)
    print("RDA detailed trajectory:", rda_guess.rda_trajectory_file)
    print("RDA NEB-like chain:", rda_guess.rda_chain_file)
    print("poormans_NEB constrained trajectory:", rda_guess.poormans_NEB_file)
    print("poormans_NEB energetics profile:", rda_guess.poormans_NEB_profile_file)
    print("poormans_NEB energetics plot:", rda_guess.poormans_NEB_plot_file)
    print_reactive_coordinates("RDA quasi-TS guess", rda_guess.atoms, rda_guess.q * BOHR_TO_ANGSTROM)

    if REACTION_MODE == "combined":
        reaction_kwargs = {
            "reaction_mode": "combined",
            "reaction_coordinates": REACTION_COORDINATES,
            "extra_bonds": EXTRA_BONDS,
            "extra_angles": EXTRA_ANGLES,
        }
    elif REACTION_MODE == "transfer":
        reaction_kwargs = {
            "reaction_mode": "transfer",
            "reaction_transfer": REACTION_TRANSFER,
            "extra_bonds": EXTRA_BONDS,
            "extra_angles": EXTRA_ANGLES,
        }
    elif REACTION_MODE == "bond":
        reaction_kwargs = {
            "reaction_mode": "bond",
            "reaction_bond": FORMING_BOND,
            "extra_bonds": EXTRA_BONDS,
            "extra_angles": EXTRA_ANGLES,
        }
    else:
        raise ValueError("REACTION_MODE must be 'combined', 'transfer', or 'bond'")

    print("Starting Baker TS optimization from poormans_NEB RDA quasi-TS guess", flush=True)
    result = optimize_transition_state(
        qcinp,
        rda_guess.xyz(),
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
                "hess_file": "hessian_methanol_OH_poormans_NEB_RDA_TS.hess",
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
            "trust": {"trust_radius": 0.06},
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
            "reporting": {"trajectory_file": "tsopt_methanol_OH_poormans_NEB_RDA.xyz"},
            "project_eckart": True,
        }),
    )
    print("Finished Baker TS optimization", flush=True)

    print("Methanol + OH TS optimization from poormans_NEB RDA guess")
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

    print("Starting frequency analysis for methanol + OH poormans_NEB RDA TS", flush=True)
    frequency_analysis(
        qcinput=qcinp,
        atoms=result.atoms,
        q=result.q,
        fname="methanol_OH_poormans_NEB_RDA_TS_Orca_b97_3c",
        hessFile="hessian_methanol_OH_poormans_NEB_RDA_TS_Orca_b97_3c_final_freq.hess",
        is_eckart=True,
        force_hessian_recalc=True,
        Amp_modeanim=30.0,
        print_nmode=True,
        qrrho_cutoff=50.0,
    )
    print("Finished frequency analysis for methanol + OH poormans_NEB RDA TS", flush=True)
