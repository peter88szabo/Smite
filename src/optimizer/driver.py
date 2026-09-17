from __future__ import annotations

from dataclasses import replace

from optimizer.common import (
    _atoms_q_from_xyz,
)
from optimizer.minimum import (
    _orca_native_optimize_geometry,
    _smite_cartesian_optimize_geometry,
    _smite_internal_optimize_geometry,
)
from optimizer.transition_state import (
    ORCA_TS_ENERGY_TOL,
    ORCA_TS_MAX_GRADIENT,
    ORCA_TS_MAX_STEP,
    ORCA_TS_RMS_GRADIENT,
    ORCA_TS_RMS_STEP,
    _cartesian_reaction_direction_from_bond,
    _cartesian_reaction_direction_from_transfer,
    _reaction_reference_label,
    _smite_cartesian_ts_optimize_geometry,
    _smite_internal_ts_optimize_geometry,
)
from optimizer.reporter import make_reporter
from optimizer.settings import OptimizerSettings
from qchem_interfaces.qchem_validation import validate_qchem_input


def _maybe_run_frequency_analysis(
    result,
    qcinput,
    *,
    run_frequency_analysis=False,
    frequency_analysis_name=None,
    frequency_analysis_hess_file=None,
    frequency_analysis_linear=False,
    frequency_analysis_eckart=True,
    qrrho_cutoff=50.0,
    thermo_temp=298.15,
    thermo_pressure=101325.0,
    thermo_multiplicity=None,
    thermo_symmetry_number=1.0,
    thermo_chirality_number=1.0,
):
    if not run_frequency_analysis:
        return result
    from normalmode.normalmode import frequency_analysis

    fname = frequency_analysis_name
    if fname is None:
        fname = "tsopt" if result.target == "transition_state" else "geomopt"
    frequency_analysis(
        qcinput=qcinput,
        atoms=result.atoms,
        q=result.q,
        fname=fname,
        hessFile=frequency_analysis_hess_file,
        linear=frequency_analysis_linear,
        is_eckart=frequency_analysis_eckart,
        qrrho_cutoff=qrrho_cutoff,
        temp=thermo_temp,
        pressure=thermo_pressure,
        multiplicity=thermo_multiplicity,
        symmetry_number=thermo_symmetry_number,
        chirality_number=thermo_chirality_number,
        electronic_energy=result.energy,
    )
    return result


def optimize_geometry(
    qcinput,
    xyz,
    *,
    method="BFGS",
    backend_optimizer="auto",
    coordinates="internal",
    target="minimum",
    maxstep=100,
    energy_tol=5.0e-5,
    max_step=4.0e-3,
    rms_step=2.5e-3,
    max_gradient=7.0e-4,
    rms_gradient=5.0e-4,
    trajectory_file="geomopt_traj.xyz",
    print_report=True,
    orca_what="minimum",
    max_step_internal=0.2,
    best_fit_iters=5,
    best_fit_rms_tol=1.0e-7,
    connectivity_kcn=16.0,
    connectivity_facmin=0.45,
    connect_fragments=True,
    hess_file=None,
    hessian_recalc_interval=0,
    trust_radius=0.1,
    trust_radius_min=1.0e-4,
    trust_radius_max=None,
    reaction_mode="lowest",
    reaction_direction=None,
    reaction_bond=None,
    reaction_angle=None,
    reaction_dihedral=None,
    reaction_transfer=None,
    reaction_coordinates=None,
    max_rejected_steps=8,
    project_eckart=True,
    final_hessian=True,
    min_gradient_improvement=0.0,
    internal_hessian_correction=True,
    mode_tracking_coordinates="internal",
    adaptive_hessian_recalc=False,
    adaptive_hessian_overlap_min=0.55,
    adaptive_hessian_rho_min=0.05,
    adaptive_hessian_rho_max=5.0,
    adaptive_hessian_trust_fraction=0.15,
    adaptive_hessian_on_negative_mode_change=True,
    adaptive_hessian_on_trust_collapse=False,
    adaptive_hessian_retry_recalc=False,
    adaptive_hessian_recalc_cooldown=3,
    use_redundant_internals=False,
    skip_hessian_recalc_near_convergence=True,
    hessian_recalc_near_convergence_factor=3.0,
    repair_ts_hessian=True,
    ts_hessian_eigenvalue_floor=1.0e-4,
    linear_bends=False,
    linear_bend_threshold_degrees=None,
    extra_bonds=None,
    extra_angles=None,
    extra_dihedrals=None,
    internal_hessian_model="simple",
    ts_initial_hessian="exact",
    ts_model_negative_curvature=0.05,
    run_frequency_analysis=False,
    frequency_analysis_name=None,
    frequency_analysis_hess_file=None,
    frequency_analysis_linear=False,
    frequency_analysis_eckart=True,
    qrrho_cutoff=50.0,
    thermo_temp=298.15,
    thermo_pressure=101325.0,
    thermo_multiplicity=None,
    thermo_symmetry_number=1.0,
    thermo_chirality_number=1.0,
    settings=None,
    reporter=None,
    verbosity=1,
    log_file=None,
    log_append=False,
    _reporter_active=False,
):
    if settings is not None:
        if isinstance(settings, dict):
            settings = OptimizerSettings.from_dict(settings)
        if not isinstance(settings, OptimizerSettings):
            raise TypeError("settings must be an OptimizerSettings instance or dict")
        method = settings.method
        backend_optimizer = settings.backend_optimizer
        coordinates = settings.coordinates
        target = settings.target
        orca_what = settings.orca_what
        project_eckart = settings.project_eckart

        maxstep = settings.convergence.maxstep
        energy_tol = settings.convergence.energy_tol
        max_step = settings.convergence.max_step
        rms_step = settings.convergence.rms_step
        max_gradient = settings.convergence.max_gradient
        rms_gradient = settings.convergence.rms_gradient

        max_step_internal = settings.internal_coordinates.max_step_internal
        best_fit_iters = settings.internal_coordinates.best_fit_iters
        best_fit_rms_tol = settings.internal_coordinates.best_fit_rms_tol
        connectivity_kcn = settings.internal_coordinates.connectivity_kcn
        connectivity_facmin = settings.internal_coordinates.connectivity_facmin
        connect_fragments = settings.internal_coordinates.connect_fragments
        use_redundant_internals = settings.internal_coordinates.use_redundant_internals
        linear_bends = settings.internal_coordinates.linear_bends
        linear_bend_threshold_degrees = settings.internal_coordinates.linear_bend_threshold_degrees
        extra_bonds = settings.internal_coordinates.extra_bonds
        extra_angles = settings.internal_coordinates.extra_angles
        extra_dihedrals = settings.internal_coordinates.extra_dihedrals

        hess_file = settings.hessian.hess_file
        hessian_recalc_interval = settings.hessian.hessian_recalc_interval
        final_hessian = settings.hessian.final_hessian
        internal_hessian_correction = settings.hessian.internal_hessian_correction
        adaptive_hessian_recalc = settings.hessian.adaptive_hessian_recalc
        adaptive_hessian_overlap_min = settings.hessian.adaptive_hessian_overlap_min
        adaptive_hessian_rho_min = settings.hessian.adaptive_hessian_rho_min
        adaptive_hessian_rho_max = settings.hessian.adaptive_hessian_rho_max
        adaptive_hessian_trust_fraction = settings.hessian.adaptive_hessian_trust_fraction
        adaptive_hessian_on_negative_mode_change = settings.hessian.adaptive_hessian_on_negative_mode_change
        adaptive_hessian_on_trust_collapse = settings.hessian.adaptive_hessian_on_trust_collapse
        adaptive_hessian_retry_recalc = settings.hessian.adaptive_hessian_retry_recalc
        adaptive_hessian_recalc_cooldown = settings.hessian.adaptive_hessian_recalc_cooldown
        skip_hessian_recalc_near_convergence = settings.hessian.skip_hessian_recalc_near_convergence
        hessian_recalc_near_convergence_factor = settings.hessian.hessian_recalc_near_convergence_factor
        repair_ts_hessian = settings.hessian.repair_ts_hessian
        ts_hessian_eigenvalue_floor = settings.hessian.ts_hessian_eigenvalue_floor
        internal_hessian_model = settings.hessian.internal_hessian_model
        ts_initial_hessian = settings.hessian.ts_initial_hessian
        ts_model_negative_curvature = settings.hessian.ts_model_negative_curvature

        trust_radius = settings.trust.trust_radius
        trust_radius_min = settings.trust.trust_radius_min
        trust_radius_max = settings.trust.trust_radius_max
        max_rejected_steps = settings.trust.max_rejected_steps
        min_gradient_improvement = settings.trust.min_gradient_improvement

        reaction_mode = settings.reaction.reaction_mode
        reaction_direction = settings.reaction.reaction_direction
        reaction_bond = settings.reaction.reaction_bond
        reaction_angle = settings.reaction.reaction_angle
        reaction_dihedral = settings.reaction.reaction_dihedral
        reaction_transfer = settings.reaction.reaction_transfer
        reaction_coordinates = settings.reaction.reaction_coordinates
        mode_tracking_coordinates = settings.reaction.mode_tracking_coordinates

        trajectory_file = settings.reporting.trajectory_file
        print_report = settings.reporting.print_report
        verbosity = settings.reporting.verbosity
        log_file = settings.reporting.log_file
        log_append = settings.reporting.log_append
    else:
        settings = OptimizerSettings.from_options(
            method=method,
            backend_optimizer=backend_optimizer,
            coordinates=coordinates,
            target=target,
            maxstep=maxstep,
            energy_tol=energy_tol,
            max_step=max_step,
            rms_step=rms_step,
            max_gradient=max_gradient,
            rms_gradient=rms_gradient,
            trajectory_file=trajectory_file,
            print_report=print_report,
            verbosity=verbosity,
            log_file=log_file,
            log_append=log_append,
            orca_what=orca_what,
            max_step_internal=max_step_internal,
            best_fit_iters=best_fit_iters,
            best_fit_rms_tol=best_fit_rms_tol,
            connectivity_kcn=connectivity_kcn,
            connectivity_facmin=connectivity_facmin,
            connect_fragments=connect_fragments,
            hess_file=hess_file,
            hessian_recalc_interval=hessian_recalc_interval,
            trust_radius=trust_radius,
            trust_radius_min=trust_radius_min,
            trust_radius_max=trust_radius_max,
            reaction_mode=reaction_mode,
            reaction_direction=reaction_direction,
            reaction_bond=reaction_bond,
            reaction_angle=reaction_angle,
            reaction_dihedral=reaction_dihedral,
            reaction_transfer=reaction_transfer,
            reaction_coordinates=reaction_coordinates,
            max_rejected_steps=max_rejected_steps,
            project_eckart=project_eckart,
            final_hessian=final_hessian,
            min_gradient_improvement=min_gradient_improvement,
            internal_hessian_correction=internal_hessian_correction,
            mode_tracking_coordinates=mode_tracking_coordinates,
            adaptive_hessian_recalc=adaptive_hessian_recalc,
            adaptive_hessian_overlap_min=adaptive_hessian_overlap_min,
            adaptive_hessian_rho_min=adaptive_hessian_rho_min,
            adaptive_hessian_rho_max=adaptive_hessian_rho_max,
            adaptive_hessian_trust_fraction=adaptive_hessian_trust_fraction,
            adaptive_hessian_on_negative_mode_change=adaptive_hessian_on_negative_mode_change,
            adaptive_hessian_on_trust_collapse=adaptive_hessian_on_trust_collapse,
            adaptive_hessian_retry_recalc=adaptive_hessian_retry_recalc,
            adaptive_hessian_recalc_cooldown=adaptive_hessian_recalc_cooldown,
            use_redundant_internals=use_redundant_internals,
            skip_hessian_recalc_near_convergence=skip_hessian_recalc_near_convergence,
            hessian_recalc_near_convergence_factor=hessian_recalc_near_convergence_factor,
            repair_ts_hessian=repair_ts_hessian,
            ts_hessian_eigenvalue_floor=ts_hessian_eigenvalue_floor,
            linear_bends=linear_bends,
            linear_bend_threshold_degrees=linear_bend_threshold_degrees,
            extra_bonds=extra_bonds,
            extra_angles=extra_angles,
            extra_dihedrals=extra_dihedrals,
            internal_hessian_model=internal_hessian_model,
            ts_initial_hessian=ts_initial_hessian,
            ts_model_negative_curvature=ts_model_negative_curvature,
        )

    reporter = make_reporter(
        print_report=print_report,
        reporter=reporter,
        verbosity=verbosity,
        log_file=log_file,
        append=log_append,
    )
    if not _reporter_active:
        with reporter.capture_prints():
            return optimize_geometry(
                qcinput,
                xyz,
                method=method,
                backend_optimizer=backend_optimizer,
                coordinates=coordinates,
                target=target,
                maxstep=maxstep,
                energy_tol=energy_tol,
                max_step=max_step,
                rms_step=rms_step,
                max_gradient=max_gradient,
                rms_gradient=rms_gradient,
                trajectory_file=trajectory_file,
                print_report=print_report and reporter.should_report(),
                orca_what=orca_what,
                max_step_internal=max_step_internal,
                best_fit_iters=best_fit_iters,
                best_fit_rms_tol=best_fit_rms_tol,
                connectivity_kcn=connectivity_kcn,
                connectivity_facmin=connectivity_facmin,
                connect_fragments=connect_fragments,
                hess_file=hess_file,
                hessian_recalc_interval=hessian_recalc_interval,
                trust_radius=trust_radius,
                trust_radius_min=trust_radius_min,
                trust_radius_max=trust_radius_max,
                reaction_mode=reaction_mode,
                reaction_direction=reaction_direction,
                reaction_bond=reaction_bond,
                reaction_angle=reaction_angle,
                reaction_dihedral=reaction_dihedral,
                reaction_transfer=reaction_transfer,
                reaction_coordinates=reaction_coordinates,
                max_rejected_steps=max_rejected_steps,
                project_eckart=project_eckart,
                final_hessian=final_hessian,
                min_gradient_improvement=min_gradient_improvement,
                internal_hessian_correction=internal_hessian_correction,
                mode_tracking_coordinates=mode_tracking_coordinates,
                adaptive_hessian_recalc=adaptive_hessian_recalc,
                adaptive_hessian_overlap_min=adaptive_hessian_overlap_min,
                adaptive_hessian_rho_min=adaptive_hessian_rho_min,
                adaptive_hessian_rho_max=adaptive_hessian_rho_max,
                adaptive_hessian_trust_fraction=adaptive_hessian_trust_fraction,
                adaptive_hessian_on_negative_mode_change=adaptive_hessian_on_negative_mode_change,
                adaptive_hessian_on_trust_collapse=adaptive_hessian_on_trust_collapse,
                adaptive_hessian_retry_recalc=adaptive_hessian_retry_recalc,
                adaptive_hessian_recalc_cooldown=adaptive_hessian_recalc_cooldown,
                use_redundant_internals=use_redundant_internals,
                skip_hessian_recalc_near_convergence=skip_hessian_recalc_near_convergence,
                hessian_recalc_near_convergence_factor=hessian_recalc_near_convergence_factor,
                repair_ts_hessian=repair_ts_hessian,
                ts_hessian_eigenvalue_floor=ts_hessian_eigenvalue_floor,
                linear_bends=linear_bends,
                linear_bend_threshold_degrees=linear_bend_threshold_degrees,
                extra_bonds=extra_bonds,
                extra_angles=extra_angles,
                extra_dihedrals=extra_dihedrals,
                internal_hessian_model=internal_hessian_model,
                ts_initial_hessian=ts_initial_hessian,
                ts_model_negative_curvature=ts_model_negative_curvature,
                run_frequency_analysis=run_frequency_analysis,
                frequency_analysis_name=frequency_analysis_name,
                frequency_analysis_hess_file=frequency_analysis_hess_file,
                frequency_analysis_linear=frequency_analysis_linear,
                frequency_analysis_eckart=frequency_analysis_eckart,
                qrrho_cutoff=qrrho_cutoff,
                thermo_temp=thermo_temp,
                thermo_pressure=thermo_pressure,
                thermo_multiplicity=thermo_multiplicity,
                reporter=reporter,
                thermo_symmetry_number=thermo_symmetry_number,
                thermo_chirality_number=thermo_chirality_number,
                _reporter_active=True,
            )

    qcinput = validate_qchem_input(qcinput)
    qchem = qcinput["qchem"]
    atoms, q = _atoms_q_from_xyz(xyz)

    def finish(result):
        return _maybe_run_frequency_analysis(
            result,
            qcinput,
            run_frequency_analysis=run_frequency_analysis,
            frequency_analysis_name=frequency_analysis_name,
            frequency_analysis_hess_file=frequency_analysis_hess_file,
            frequency_analysis_linear=frequency_analysis_linear,
            frequency_analysis_eckart=frequency_analysis_eckart,
            qrrho_cutoff=qrrho_cutoff,
            thermo_temp=thermo_temp,
            thermo_pressure=thermo_pressure,
            thermo_multiplicity=thermo_multiplicity,
            thermo_symmetry_number=thermo_symmetry_number,
            thermo_chirality_number=thermo_chirality_number,
        )
    if backend_optimizer not in {"auto", "backend", "smite"}:
        raise ValueError("backend_optimizer must be 'auto', 'backend', or 'smite'")
    if coordinates not in {"internal", "cartesian"}:
        raise ValueError("coordinates must be 'internal' or 'cartesian'")
    if target not in {"minimum", "transition_state"}:
        raise ValueError("target must be 'minimum' or 'transition_state'")

    if target == "transition_state":
        if backend_optimizer == "backend":
            raise ValueError("backend-native transition-state optimization is not implemented")
        if energy_tol == 5.0e-5:
            energy_tol = ORCA_TS_ENERGY_TOL
        if max_step == 4.0e-3:
            max_step = ORCA_TS_MAX_STEP
        if rms_step == 2.5e-3:
            rms_step = ORCA_TS_RMS_STEP
        if max_gradient == 7.0e-4:
            max_gradient = ORCA_TS_MAX_GRADIENT
        if rms_gradient == 5.0e-4:
            rms_gradient = ORCA_TS_RMS_GRADIENT
        reaction_mode = str(reaction_mode).lower()
        if reaction_mode == "combined":
            reaction_mode = "coordinates"
        if reaction_coordinates is not None and reaction_mode == "lowest":
            reaction_mode = "coordinates"
        allowed_reaction_modes = {"lowest", "direction", "bond", "angle", "dihedral", "transfer", "coordinates"}
        if reaction_mode not in allowed_reaction_modes:
            raise ValueError(
                "reaction_mode must be 'lowest', 'direction', 'bond', 'angle', 'dihedral', "
                "'transfer', 'coordinates', or 'combined'"
            )
        if hess_file is None:
            hess_file = "hessian_ts.hess"
        if trust_radius_max is None:
            trust_radius_max = 0.4 if coordinates == "internal" else 0.3
        reaction_references = {
            "direction": reaction_direction,
            "bond": reaction_bond,
            "angle": reaction_angle,
            "dihedral": reaction_dihedral,
            "transfer": reaction_transfer,
            "coordinates": reaction_coordinates,
        }
        reference_inputs = sum(item is not None for item in reaction_references.values())
        if reference_inputs > 1:
            raise ValueError(
                "Specify only one of reaction_direction, reaction_bond, reaction_angle, "
                "reaction_dihedral, reaction_transfer, or reaction_coordinates"
            )
        if reaction_mode == "lowest" and reference_inputs:
            raise ValueError("reaction_mode='lowest' cannot be combined with a reaction reference input")
        elif reaction_mode != "lowest":
            if reaction_references[reaction_mode] is None:
                if reaction_mode == "direction":
                    label = "reaction_direction"
                elif reaction_mode == "coordinates":
                    label = "reaction_coordinates"
                else:
                    label = f"reaction_{reaction_mode}"
                raise ValueError(f"reaction_mode='{reaction_mode}' requires {label}")
            invalid = [
                key for key, value in reaction_references.items()
                if key != reaction_mode and value is not None
            ]
            if invalid:
                label = "reaction_coordinates" if reaction_mode == "coordinates" else f"reaction_{reaction_mode}"
                raise ValueError(f"reaction_mode='{reaction_mode}' only accepts {label}")
        if coordinates == "cartesian" and reaction_mode in {"angle", "dihedral", "coordinates"}:
            raise ValueError(
                "reaction_mode='angle', reaction_mode='dihedral', and reaction_mode='coordinates' "
                "require coordinates='internal'"
            )
        if reaction_bond is not None and coordinates == "cartesian":
            reaction_direction = _cartesian_reaction_direction_from_bond(q, atoms, reaction_bond)
        elif reaction_transfer is not None:
            reaction_direction = _cartesian_reaction_direction_from_transfer(q, atoms, reaction_transfer)
        reaction_label = _reaction_reference_label(
            reaction_bond=reaction_bond,
            reaction_angle=reaction_angle,
            reaction_dihedral=reaction_dihedral,
            reaction_transfer=reaction_transfer,
            reaction_coordinates=reaction_coordinates,
        )

        if coordinates == "cartesian":
            return finish(_smite_cartesian_ts_optimize_geometry(
                qcinput,
                atoms,
                q,
                maxstep=maxstep,
                energy_tol=energy_tol,
                max_step=max_step,
                rms_step=rms_step,
                max_gradient=max_gradient,
                rms_gradient=rms_gradient,
                trajectory_file=trajectory_file,
                print_report=print_report,
                hess_file=hess_file,
                hessian_recalc_interval=hessian_recalc_interval,
                trust_radius=trust_radius,
                trust_radius_min=trust_radius_min,
                trust_radius_max=trust_radius_max,
                reaction_direction=reaction_direction,
                reaction_label=reaction_label,
                max_rejected_steps=max_rejected_steps,
                project_eckart=project_eckart,
                final_hessian=final_hessian,
                min_gradient_improvement=min_gradient_improvement,
                adaptive_hessian_recalc=adaptive_hessian_recalc,
                adaptive_hessian_overlap_min=adaptive_hessian_overlap_min,
                adaptive_hessian_rho_min=adaptive_hessian_rho_min,
                adaptive_hessian_rho_max=adaptive_hessian_rho_max,
                adaptive_hessian_trust_fraction=adaptive_hessian_trust_fraction,
                adaptive_hessian_on_negative_mode_change=adaptive_hessian_on_negative_mode_change,
                adaptive_hessian_on_trust_collapse=adaptive_hessian_on_trust_collapse,
                adaptive_hessian_retry_recalc=adaptive_hessian_retry_recalc,
                adaptive_hessian_recalc_cooldown=adaptive_hessian_recalc_cooldown,
            ))

        return finish(_smite_internal_ts_optimize_geometry(
            qcinput,
            atoms,
            q,
            maxstep=maxstep,
            energy_tol=energy_tol,
            max_step=max_step,
            rms_step=rms_step,
            max_gradient=max_gradient,
            rms_gradient=rms_gradient,
            max_step_internal=max_step_internal,
            best_fit_iters=best_fit_iters,
            best_fit_rms_tol=best_fit_rms_tol,
            trajectory_file=trajectory_file,
            print_report=print_report,
            hess_file=hess_file,
            hessian_recalc_interval=hessian_recalc_interval,
            trust_radius=trust_radius,
            trust_radius_min=trust_radius_min,
            trust_radius_max=trust_radius_max,
            reaction_direction=reaction_direction,
            reaction_bond=reaction_bond,
            reaction_angle=reaction_angle,
            reaction_dihedral=reaction_dihedral,
            reaction_transfer=reaction_transfer,
            reaction_coordinates=reaction_coordinates,
            max_rejected_steps=max_rejected_steps,
            connectivity_kcn=connectivity_kcn,
            connectivity_facmin=connectivity_facmin,
            connect_fragments=connect_fragments,
            project_eckart=project_eckart,
            final_hessian=final_hessian,
            min_gradient_improvement=min_gradient_improvement,
            internal_hessian_correction=internal_hessian_correction,
            mode_tracking_coordinates=mode_tracking_coordinates,
            adaptive_hessian_recalc=adaptive_hessian_recalc,
            adaptive_hessian_overlap_min=adaptive_hessian_overlap_min,
            adaptive_hessian_rho_min=adaptive_hessian_rho_min,
            adaptive_hessian_rho_max=adaptive_hessian_rho_max,
            adaptive_hessian_trust_fraction=adaptive_hessian_trust_fraction,
            adaptive_hessian_on_negative_mode_change=adaptive_hessian_on_negative_mode_change,
            adaptive_hessian_on_trust_collapse=adaptive_hessian_on_trust_collapse,
            adaptive_hessian_retry_recalc=adaptive_hessian_retry_recalc,
            adaptive_hessian_recalc_cooldown=adaptive_hessian_recalc_cooldown,
            use_redundant_internals=use_redundant_internals,
            skip_hessian_recalc_near_convergence=skip_hessian_recalc_near_convergence,
            hessian_recalc_near_convergence_factor=hessian_recalc_near_convergence_factor,
            repair_ts_hessian=repair_ts_hessian,
            ts_hessian_eigenvalue_floor=ts_hessian_eigenvalue_floor,
            linear_bends=linear_bends,
            linear_bend_threshold_degrees=linear_bend_threshold_degrees,
            extra_bonds=extra_bonds,
            extra_angles=extra_angles,
            extra_dihedrals=extra_dihedrals,
            internal_hessian_model=internal_hessian_model,
            ts_initial_hessian=ts_initial_hessian,
            ts_model_negative_curvature=ts_model_negative_curvature,
        ))

    if backend_optimizer == "backend" and qchem == "Orca":
        return finish(_orca_native_optimize_geometry(qcinput, atoms, q, what=orca_what))

    if backend_optimizer == "backend":
        raise ValueError(f"No backend-native optimizer is implemented for {qchem}")

    if coordinates == "cartesian":
        return finish(_smite_cartesian_optimize_geometry(
            qcinput,
            atoms,
            q,
            method=method,
            maxstep=maxstep,
            energy_tol=energy_tol,
            max_step=max_step,
            rms_step=rms_step,
            max_gradient=max_gradient,
            rms_gradient=rms_gradient,
            trajectory_file=trajectory_file,
            print_report=print_report,
        ))

    return finish(_smite_internal_optimize_geometry(
        qcinput,
        atoms,
        q,
        method=method,
        maxstep=maxstep,
        energy_tol=energy_tol,
        max_step=max_step,
        rms_step=rms_step,
        max_gradient=max_gradient,
        rms_gradient=rms_gradient,
        trajectory_file=trajectory_file,
        print_report=print_report,
        max_step_internal=max_step_internal,
        best_fit_iters=best_fit_iters,
        best_fit_rms_tol=best_fit_rms_tol,
        connectivity_kcn=connectivity_kcn,
        connectivity_facmin=connectivity_facmin,
        connect_fragments=connect_fragments,
        use_redundant_internals=use_redundant_internals,
        linear_bends=linear_bends,
        linear_bend_threshold_degrees=linear_bend_threshold_degrees,
            extra_bonds=extra_bonds,
            extra_angles=extra_angles,
            extra_dihedrals=extra_dihedrals,
            internal_hessian_model=internal_hessian_model,
        ))


def geom_optimizer(optinput, qcinput, hessFile, invhessFile, xyz, atoms=None, mass=None):
    del hessFile, invhessFile, mass
    method = optinput[0]
    energy_tol = optinput[1]
    max_step = optinput[2]
    rms_step = optinput[3]
    max_gradient = optinput[4]
    rms_gradient = optinput[5]
    maxstep = optinput[6]

    del atoms
    return optimize_geometry(
        qcinput,
        xyz,
        method=method,
        backend_optimizer="smite",
        maxstep=maxstep,
        energy_tol=energy_tol,
        max_step=max_step,
        rms_step=rms_step,
        max_gradient=max_gradient,
        rms_gradient=rms_gradient,
    )


def geom_optimzer(*args, **kwargs):
    return geom_optimizer(*args, **kwargs)


def Optimize(qcinput, file_wf, xyz):
    del file_wf
    return optimize_geometry(qcinput, xyz)


def optimize_transition_state(qcinput, xyz, **kwargs):
    if "settings" in kwargs and kwargs["settings"] is not None:
        if isinstance(kwargs["settings"], dict):
            settings_dict = dict(kwargs["settings"])
            settings_dict["target"] = "transition_state"
            kwargs["settings"] = settings_dict
        else:
            kwargs["settings"] = replace(kwargs["settings"], target="transition_state")
    kwargs["target"] = "transition_state"
    return optimize_geometry(qcinput, xyz, **kwargs)
