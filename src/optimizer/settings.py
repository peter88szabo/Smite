from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from collections.abc import Mapping


def _field_names(dataclass_type):
    return {item.name for item in fields(dataclass_type)}


def _coerce_dataclass_section(section_type, value, section_name):
    if value is None:
        return section_type()
    if isinstance(value, section_type):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"optimizer config section '{section_name}' must be a dict or {section_type.__name__}")
    allowed = _field_names(section_type)
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Unknown optimizer config keys in '{section_name}': {', '.join(unknown)}")
    return section_type(**dict(value))


@dataclass(frozen=True)
class ConvergenceSettings:
    maxstep: int = 100
    energy_tol: float = 5.0e-5
    max_step: float = 4.0e-3
    rms_step: float = 2.5e-3
    max_gradient: float = 7.0e-4
    rms_gradient: float = 5.0e-4


@dataclass(frozen=True)
class InternalCoordinateSettings:
    max_step_internal: float = 0.2
    best_fit_iters: int = 5
    best_fit_rms_tol: float = 1.0e-7
    connectivity_kcn: float = 16.0
    connectivity_facmin: float = 0.45
    connect_fragments: bool = True
    use_redundant_internals: bool = False
    linear_bends: bool = False
    linear_bend_threshold_degrees: float | None = None
    extra_bonds: object = None
    extra_angles: object = None
    extra_dihedrals: object = None


@dataclass(frozen=True)
class HessianSettings:
    hess_file: str | None = None
    hessian_recalc_interval: int = 0
    final_hessian: bool = True
    internal_hessian_correction: bool = True
    adaptive_hessian_recalc: bool = False
    adaptive_hessian_overlap_min: float = 0.55
    adaptive_hessian_rho_min: float = 0.05
    adaptive_hessian_rho_max: float = 5.0
    adaptive_hessian_trust_fraction: float = 0.15
    adaptive_hessian_on_negative_mode_change: bool = True
    adaptive_hessian_on_trust_collapse: bool = False
    adaptive_hessian_retry_recalc: bool = False
    adaptive_hessian_recalc_cooldown: int = 3
    skip_hessian_recalc_near_convergence: bool = True
    hessian_recalc_near_convergence_factor: float = 3.0
    repair_ts_hessian: bool = True
    ts_hessian_eigenvalue_floor: float = 1.0e-4
    internal_hessian_model: str = "simple"
    ts_initial_hessian: str = "exact"
    ts_model_negative_curvature: float = 0.05


@dataclass(frozen=True)
class TrustSettings:
    trust_radius: float = 0.1
    trust_radius_min: float = 1.0e-4
    trust_radius_max: float | None = None
    max_rejected_steps: int = 8
    min_gradient_improvement: float = 0.0


@dataclass(frozen=True)
class ReactionReferenceSettings:
    reaction_mode: str = "lowest"
    reaction_direction: object = None
    reaction_bond: object = None
    reaction_angle: object = None
    reaction_dihedral: object = None
    reaction_transfer: object = None
    reaction_coordinates: object = None
    mode_tracking_coordinates: str = "internal"


@dataclass(frozen=True)
class ReportingSettings:
    trajectory_file: str | None = "geomopt_traj.xyz"
    print_report: bool = True
    verbosity: int = 1
    log_file: str | None = None
    log_append: bool = False


@dataclass(frozen=True)
class OptimizerSettings:
    method: str = "BFGS"
    backend_optimizer: str = "auto"
    coordinates: str = "internal"
    target: str = "minimum"
    orca_what: str = "minimum"
    project_eckart: bool = True
    convergence: ConvergenceSettings = field(default_factory=ConvergenceSettings)
    internal_coordinates: InternalCoordinateSettings = field(default_factory=InternalCoordinateSettings)
    hessian: HessianSettings = field(default_factory=HessianSettings)
    trust: TrustSettings = field(default_factory=TrustSettings)
    reaction: ReactionReferenceSettings = field(default_factory=ReactionReferenceSettings)
    reporting: ReportingSettings = field(default_factory=ReportingSettings)

    @classmethod
    def from_dict(cls, config):
        if isinstance(config, cls):
            return config
        if not isinstance(config, Mapping):
            raise TypeError("OptimizerConfig must be built from a dict-like object")

        section_types = {
            "convergence": ConvergenceSettings,
            "internal_coordinates": InternalCoordinateSettings,
            "hessian": HessianSettings,
            "trust": TrustSettings,
            "reaction": ReactionReferenceSettings,
            "reporting": ReportingSettings,
        }
        top_level = _field_names(cls)
        nested_keys = set().union(*(_field_names(section_type) for section_type in section_types.values()))
        allowed = top_level | nested_keys
        unknown = sorted(set(config) - allowed)
        if unknown:
            raise ValueError(f"Unknown optimizer config keys: {', '.join(unknown)}")

        top_options = {
            key: config[key]
            for key in ("method", "backend_optimizer", "coordinates", "target", "orca_what", "project_eckart")
            if key in config
        }
        for section_name, section_type in section_types.items():
            section_data = {}
            if section_name in config:
                section_value = config[section_name]
                if isinstance(section_value, section_type):
                    top_options[section_name] = section_value
                    continue
                if not isinstance(section_value, Mapping):
                    raise TypeError(
                        f"optimizer config section '{section_name}' must be a dict or {section_type.__name__}"
                    )
                section_data.update(section_value)
            section_keys = _field_names(section_type)
            for key in section_keys:
                if key in config:
                    section_data[key] = config[key]
            top_options[section_name] = _coerce_dataclass_section(section_type, section_data, section_name)

        return cls(**top_options)

    def to_dict(self):
        return asdict(self)

    def as_flat_dict(self):
        flat = {
            "method": self.method,
            "backend_optimizer": self.backend_optimizer,
            "coordinates": self.coordinates,
            "target": self.target,
            "orca_what": self.orca_what,
            "project_eckart": self.project_eckart,
        }
        for section_name in ("convergence", "internal_coordinates", "hessian", "trust", "reaction", "reporting"):
            section = getattr(self, section_name)
            if is_dataclass(section):
                flat.update(asdict(section))
        return flat

    @classmethod
    def from_options(cls, **options):
        default_convergence = ConvergenceSettings()
        default_internal = InternalCoordinateSettings()
        default_hessian = HessianSettings()
        default_trust = TrustSettings()
        default_reaction = ReactionReferenceSettings()
        default_reporting = ReportingSettings()
        default_optimizer = cls()
        convergence = ConvergenceSettings(
            maxstep=options.get("maxstep", default_convergence.maxstep),
            energy_tol=options.get("energy_tol", default_convergence.energy_tol),
            max_step=options.get("max_step", default_convergence.max_step),
            rms_step=options.get("rms_step", default_convergence.rms_step),
            max_gradient=options.get("max_gradient", default_convergence.max_gradient),
            rms_gradient=options.get("rms_gradient", default_convergence.rms_gradient),
        )
        internal_coordinates = InternalCoordinateSettings(
            max_step_internal=options.get("max_step_internal", default_internal.max_step_internal),
            best_fit_iters=options.get("best_fit_iters", default_internal.best_fit_iters),
            best_fit_rms_tol=options.get("best_fit_rms_tol", default_internal.best_fit_rms_tol),
            connectivity_kcn=options.get("connectivity_kcn", default_internal.connectivity_kcn),
            connectivity_facmin=options.get("connectivity_facmin", default_internal.connectivity_facmin),
            connect_fragments=options.get("connect_fragments", default_internal.connect_fragments),
            use_redundant_internals=options.get(
                "use_redundant_internals",
                default_internal.use_redundant_internals,
            ),
            linear_bends=options.get("linear_bends", default_internal.linear_bends),
            linear_bend_threshold_degrees=options.get(
                "linear_bend_threshold_degrees",
                default_internal.linear_bend_threshold_degrees,
            ),
            extra_bonds=options.get("extra_bonds", default_internal.extra_bonds),
            extra_angles=options.get("extra_angles", default_internal.extra_angles),
            extra_dihedrals=options.get("extra_dihedrals", default_internal.extra_dihedrals),
        )
        hessian = HessianSettings(
            hess_file=options.get("hess_file", default_hessian.hess_file),
            hessian_recalc_interval=options.get(
                "hessian_recalc_interval",
                default_hessian.hessian_recalc_interval,
            ),
            final_hessian=options.get("final_hessian", default_hessian.final_hessian),
            internal_hessian_correction=options.get(
                "internal_hessian_correction",
                default_hessian.internal_hessian_correction,
            ),
            adaptive_hessian_recalc=options.get(
                "adaptive_hessian_recalc",
                default_hessian.adaptive_hessian_recalc,
            ),
            adaptive_hessian_overlap_min=options.get(
                "adaptive_hessian_overlap_min",
                default_hessian.adaptive_hessian_overlap_min,
            ),
            adaptive_hessian_rho_min=options.get(
                "adaptive_hessian_rho_min",
                default_hessian.adaptive_hessian_rho_min,
            ),
            adaptive_hessian_rho_max=options.get(
                "adaptive_hessian_rho_max",
                default_hessian.adaptive_hessian_rho_max,
            ),
            adaptive_hessian_trust_fraction=options.get(
                "adaptive_hessian_trust_fraction",
                default_hessian.adaptive_hessian_trust_fraction,
            ),
            adaptive_hessian_on_negative_mode_change=options.get(
                "adaptive_hessian_on_negative_mode_change",
                default_hessian.adaptive_hessian_on_negative_mode_change,
            ),
            adaptive_hessian_on_trust_collapse=options.get(
                "adaptive_hessian_on_trust_collapse",
                default_hessian.adaptive_hessian_on_trust_collapse,
            ),
            adaptive_hessian_retry_recalc=options.get(
                "adaptive_hessian_retry_recalc",
                default_hessian.adaptive_hessian_retry_recalc,
            ),
            adaptive_hessian_recalc_cooldown=options.get(
                "adaptive_hessian_recalc_cooldown",
                default_hessian.adaptive_hessian_recalc_cooldown,
            ),
            skip_hessian_recalc_near_convergence=options.get(
                "skip_hessian_recalc_near_convergence",
                default_hessian.skip_hessian_recalc_near_convergence,
            ),
            hessian_recalc_near_convergence_factor=options.get(
                "hessian_recalc_near_convergence_factor",
                default_hessian.hessian_recalc_near_convergence_factor,
            ),
            repair_ts_hessian=options.get("repair_ts_hessian", default_hessian.repair_ts_hessian),
            ts_hessian_eigenvalue_floor=options.get(
                "ts_hessian_eigenvalue_floor",
                default_hessian.ts_hessian_eigenvalue_floor,
            ),
            internal_hessian_model=options.get(
                "internal_hessian_model",
                default_hessian.internal_hessian_model,
            ),
            ts_initial_hessian=options.get(
                "ts_initial_hessian",
                default_hessian.ts_initial_hessian,
            ),
            ts_model_negative_curvature=options.get(
                "ts_model_negative_curvature",
                default_hessian.ts_model_negative_curvature,
            ),
        )
        trust = TrustSettings(
            trust_radius=options.get("trust_radius", default_trust.trust_radius),
            trust_radius_min=options.get("trust_radius_min", default_trust.trust_radius_min),
            trust_radius_max=options.get("trust_radius_max", default_trust.trust_radius_max),
            max_rejected_steps=options.get("max_rejected_steps", default_trust.max_rejected_steps),
            min_gradient_improvement=options.get(
                "min_gradient_improvement",
                default_trust.min_gradient_improvement,
            ),
        )
        reaction = ReactionReferenceSettings(
            reaction_mode=options.get("reaction_mode", default_reaction.reaction_mode),
            reaction_direction=options.get("reaction_direction", default_reaction.reaction_direction),
            reaction_bond=options.get("reaction_bond", default_reaction.reaction_bond),
            reaction_angle=options.get("reaction_angle", default_reaction.reaction_angle),
            reaction_dihedral=options.get("reaction_dihedral", default_reaction.reaction_dihedral),
            reaction_transfer=options.get("reaction_transfer", default_reaction.reaction_transfer),
            reaction_coordinates=options.get(
                "reaction_coordinates",
                default_reaction.reaction_coordinates,
            ),
            mode_tracking_coordinates=options.get(
                "mode_tracking_coordinates",
                default_reaction.mode_tracking_coordinates,
            ),
        )
        reporting = ReportingSettings(
            trajectory_file=options.get("trajectory_file", default_reporting.trajectory_file),
            print_report=options.get("print_report", default_reporting.print_report),
            verbosity=options.get("verbosity", default_reporting.verbosity),
            log_file=options.get("log_file", default_reporting.log_file),
            log_append=options.get("log_append", default_reporting.log_append),
        )
        return cls(
            method=options.get("method", default_optimizer.method),
            backend_optimizer=options.get("backend_optimizer", default_optimizer.backend_optimizer),
            coordinates=options.get("coordinates", default_optimizer.coordinates),
            target=options.get("target", default_optimizer.target),
            orca_what=options.get("orca_what", default_optimizer.orca_what),
            project_eckart=options.get("project_eckart", default_optimizer.project_eckart),
            convergence=convergence,
            internal_coordinates=internal_coordinates,
            hessian=hessian,
            trust=trust,
            reaction=reaction,
            reporting=reporting,
        )


OptimizerConfig = OptimizerSettings
