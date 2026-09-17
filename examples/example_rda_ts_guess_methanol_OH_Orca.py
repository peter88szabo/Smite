from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == "__main__":
    USE_MODEL_HESSIAN = True
    INTERNAL_HESSIAN_MODEL = "model" if USE_MODEL_HESSIAN else "simple"
    TS_INITIAL_HESSIAN = "model" if USE_MODEL_HESSIAN else "exact"

    from normalmode import frequency_analysis
    from optimizer import OptimizerConfig, generate_rda_ts_guess, optimize_geometry, optimize_transition_state
    from utils.constants import BOHR_TO_ANGSTROM

    PREOPTIMIZE_ENDPOINTS = False #True
    ENDPOINT_PREOPT_MAXSTEPS = 150
    ENDPOINT_SAME_RMSD_TOL_ANGSTROM = 0.05

    def result_to_xyz(result, comment):
        lines = [str(len(result.atoms)), comment]
        q_angstrom = result.q * BOHR_TO_ANGSTROM
        for i, atom in enumerate(result.atoms):
            j = 3 * i
            lines.append(
                f"{atom:2s} {q_angstrom[j]:16.10f} {q_angstrom[j + 1]:16.10f} {q_angstrom[j + 2]:16.10f}"
            )
        return "\n".join(lines)

    def endpoint_rmsd_angstrom(result_a, result_b):
        if result_a.atoms != result_b.atoms:
            raise ValueError("Cannot compare endpoints with different atom labels/order")
        qa = result_a.q * BOHR_TO_ANGSTROM
        qb = result_b.q * BOHR_TO_ANGSTROM
        diff = (qa - qb).reshape(-1, 3)
        return float((diff * diff).sum(axis=1).mean() ** 0.5)

    # Atom order must be identical in reactant and product:
    #   0 C, 1-3 methyl H, 4 incoming OH oxygen, 5 OH hydrogen,
    #   6 methanol oxygen, 7 transferred hydrogen.
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
        "scratch_dir": "scratch/methanol_OH_RDA_TS",
    }

    qcinp = qcinput_orca_doublet

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
                },
                "hessian": {"internal_hessian_model": INTERNAL_HESSIAN_MODEL},
                "reporting": {
                    "trajectory_file": "preopt_methanol_OH_reactant.xyz",
                    "print_report": True,
                },
            }),
        )
        xyz_reactant_for_rda = result_to_xyz(reactant_preopt, "pre-optimized RDA reactant endpoint")
        print("Finished reactant pre-optimization", flush=True)
        print("reactant pre-optimization converged:", reactant_preopt.converged, flush=True)
        print("reactant pre-optimization steps:", reactant_preopt.nsteps, flush=True)

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
                },
                "hessian": {"internal_hessian_model": INTERNAL_HESSIAN_MODEL},
                "reporting": {
                    "trajectory_file": "preopt_methanol_OH_product.xyz",
                    "print_report": True,
                },
            }),
        )
        xyz_product_for_rda = result_to_xyz(product_preopt, "pre-optimized RDA product endpoint")
        print("Finished product pre-optimization", flush=True)
        print("product pre-optimization converged:", product_preopt.converged, flush=True)
        print("product pre-optimization steps:", product_preopt.nsteps, flush=True)

        endpoint_rmsd = endpoint_rmsd_angstrom(reactant_preopt, product_preopt)
        print(f"pre-optimized endpoint RMSD [Angstrom]: {endpoint_rmsd:.6f}", flush=True)
        if endpoint_rmsd < ENDPOINT_SAME_RMSD_TOL_ANGSTROM:
            raise RuntimeError(
                "Reactant and product pre-optimizations converged to the same structure "
                f"within {ENDPOINT_SAME_RMSD_TOL_ANGSTROM:.3f} Angstrom RMSD. "
                "RDA needs two distinct endpoint minima."
            )
    else:
        print("Endpoint pre-optimization is switched off", flush=True)
        xyz_reactant_for_rda = xyz_reactant
        xyz_product_for_rda = xyz_product

    print("Starting RDA quasi-TS guess generation", flush=True)
    rda_guess = generate_rda_ts_guess(
        qcinp,
        xyz_reactant_for_rda,
        xyz_product_for_rda,
        active_atoms=(4, 6, 7),
        align_atoms=(0, 1, 2, 3, 6),
        distance="reactive_internal",
        reactive_bonds=[(6, 7), (4, 7)],
        reactive_angles=[(6, 7, 4)],
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
        rda_trajectory_file="rda_methanol_OH_detailed.xyz",
        rda_chain_file="rda_methanol_OH_chain.xyz",
        endpoint_same_rmsd_tol=ENDPOINT_SAME_RMSD_TOL_ANGSTROM,
        endpoint_active_rmsd_tol=ENDPOINT_SAME_RMSD_TOL_ANGSTROM,
    )
    print("Finished RDA quasi-TS guess generation", flush=True)

    print("RDA quasi-TS generation")
    print("message:", rda_guess.message)
    print("direction:", rda_guess.direction)
    print("bracketed:", rda_guess.bracketed)
    print("RDA points:")
    for point in rda_guess.points:
        print(
            f"  {point.label:14s} dir={point.direction:2s} "
            f"dIS={point.delta_d_is: .6e} dFS={point.delta_d_fs: .6e} "
            f"steps={point.nsteps:3d}"
        )

    ts_reaction_settings = {}
    if rda_guess.reaction_direction is not None:
        ts_reaction_settings["reaction_mode"] = "direction"
        ts_reaction_settings["reaction_direction"] = rda_guess.reaction_direction
    else:
        ts_reaction_settings["reaction_mode"] = "lowest"

    print("Starting Baker TS optimization from RDA quasi-TS guess", flush=True)
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
                "hess_file": "hessian_methanol_OH_RDA_TS.hess",
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
            },
            "reaction": ts_reaction_settings | {"mode_tracking_coordinates": "internal"},
            "reporting": {"trajectory_file": "tsopt_methanol_OH_RDA.xyz"},
            "project_eckart": True,
        }),
    )
    print("Finished Baker TS optimization", flush=True)

    print("Methanol + OH TS optimization from RDA guess")
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

    print("Starting frequency analysis for the RDA-started TS candidate", flush=True)
    frequency_analysis(
        qcinput=qcinp,
        atoms=result.atoms,
        q=result.q,
        fname="methanol_OH_RDA_TS_Orca_b97_3c",
        hessFile="hessian_methanol_OH_RDA_TS_Orca_b97_3c_final_freq.hess",
        is_eckart=True,
        force_hessian_recalc=True,
        Amp_modeanim=30.0,
        print_nmode=True,
        qrrho_cutoff=50.0,
    )
    print("Finished frequency analysis for the RDA-started TS candidate", flush=True)
