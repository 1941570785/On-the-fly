from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Thresholds:
    tau_R_low: float = 0.40
    tau_R_high: float = 0.75
    tau_V: float = 0.55
    tau_V_min: float = 0.45
    tau_B: float = 0.12
    tau_Q: float = 0.10


COUPLED_INNOVATION_MODE = "on_the_fly_innovation_v1"
COUPLED_RUNTIME_MODE = "paper_aligned_semantic_v1"


@dataclass(frozen=True)
class CoupledInnovationConfig:
    enabled: bool
    requested_mode: str
    runtime_mode: str
    model_name: str = COUPLED_INNOVATION_MODE
    thresholds: Thresholds = field(default_factory=Thresholds)
    recovery_delay_frames: int = 8
    recovery_max_attempts: int = 3
    recovery_attempts_per_tick: int = 3
    recovery_commit_bridge: str = "true_source_commit"
    defer_recovery_support_bridge: str = "v1"
    recovery_commit_control: str = "off"
    direct_density_control: str = "off"
    direct_update_prev_desc_on_hold: str = "off"
    direct_density_upper_per_100: float = 75.0
    direct_density_hard_upper_per_100: float = 90.0
    online_quality_metric_fields: tuple[str, ...] = ()
    offline_stage_metric_fields: tuple[str, ...] = (
        "PSNR",
        "SSIM",
        "LPIPS",
        "absolute_relative_translation_error",
        "absolute_relative_rotation_error",
        "keyframe_density_per_100",
        "keyframe_gap_p50",
        "keyframe_gap_p90",
        "keyframe_gap_p99",
        "direct_admit_count",
        "defer_recoverable_count",
        "discard_count",
        "recovery_attempt_count",
        "recovery_success_count",
        "true_source_materialized_count",
        "valid_2d3d_correspondence_count",
        "pnp_inlier_count",
        "miniba_inlier_count",
        "reference_support_effective_count",
        "surrogate_commit_count",
        "duplicate_commit_count",
        "contamination_count",
    )

    def module_switches(self) -> dict[str, str]:
        return {
            "risk_admission": self.runtime_mode,
            "recovery_commit_bridge": self.recovery_commit_bridge,
            "defer_recovery_support_bridge": self.defer_recovery_support_bridge,
            "recovery_commit_control": self.recovery_commit_control,
            "direct_density_control": self.direct_density_control,
            "direct_update_prev_desc_on_hold": self.direct_update_prev_desc_on_hold,
        }

    def stage_metric_contract(self) -> dict[str, list[str]]:
        return {
            "online_quality_metric_fields": list(self.online_quality_metric_fields),
            "offline_stage_metric_fields": list(self.offline_stage_metric_fields),
        }

    def to_trace_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["thresholds"] = asdict(self.thresholds)
        data["module_switches"] = self.module_switches()
        data["stage_metric_contract"] = self.stage_metric_contract()
        return data


def _value(args: Any, name: str, default: Any) -> Any:
    value = getattr(args, name, None)
    if value is None or value == "":
        return default
    return value


def _float(args: Any, name: str, default: float) -> float:
    return float(_value(args, name, default))


def _float_with_legacy_default(args: Any, name: str, default: float, legacy_default: float) -> float:
    value = getattr(args, name, None)
    if value is None or value == "":
        return float(default)
    parsed = float(value)
    if parsed == float(legacy_default):
        return float(default)
    return parsed


def _int(args: Any, name: str, default: int) -> int:
    return int(_value(args, name, default))


def _str(args: Any, name: str, default: str) -> str:
    return str(_value(args, name, default))


def resolve_coupled_innovation_config(args: Any) -> CoupledInnovationConfig:
    requested = str(getattr(args, "risk_admission_mode", "off") or "off")
    enabled = requested == COUPLED_INNOVATION_MODE
    base_thresholds = Thresholds()
    tau_R_low_default = 0.10 if enabled else base_thresholds.tau_R_low
    thresholds = Thresholds(
        tau_R_low=_float(args, "paper_aligned_tau_R_low", tau_R_low_default),
        tau_R_high=_float(args, "paper_aligned_tau_R_high", base_thresholds.tau_R_high),
        tau_V=_float(args, "paper_aligned_tau_V", base_thresholds.tau_V),
        tau_V_min=_float(args, "paper_aligned_tau_V_min", base_thresholds.tau_V_min),
        tau_B=_float(args, "paper_aligned_tau_B", base_thresholds.tau_B),
        tau_Q=_float(args, "paper_aligned_tau_Q", base_thresholds.tau_Q),
    )
    runtime_mode = COUPLED_RUNTIME_MODE if enabled else requested
    return CoupledInnovationConfig(
        enabled=enabled,
        requested_mode=requested,
        runtime_mode=runtime_mode,
        thresholds=thresholds,
        recovery_delay_frames=_int(args, "paper_aligned_recovery_delay_frames", 8),
        recovery_max_attempts=_int(args, "paper_aligned_semantic_recovery_max_attempts", 3),
        recovery_attempts_per_tick=_int(args, "paper_aligned_semantic_recovery_attempts_per_tick", 3),
        recovery_commit_bridge=_str(args, "paper_aligned_recovery_commit_bridge", "true_source_commit"),
        defer_recovery_support_bridge=_str(
            args,
            "paper_aligned_defer_recovery_support_bridge",
            "v1" if enabled else "off",
        ),
        recovery_commit_control=_str(
            args,
            "paper_aligned_recovery_commit_control",
            "off",
        ),
        direct_density_control=_str(
            args,
            "paper_aligned_direct_density_control",
            "off",
        ),
        direct_update_prev_desc_on_hold=_str(args, "paper_aligned_direct_update_prev_desc_on_hold", "off"),
        direct_density_upper_per_100=(
            _float_with_legacy_default(
                args,
                "paper_aligned_direct_density_upper_per_100",
                75.0,
                45.0,
            )
            if enabled
            else _float(args, "paper_aligned_direct_density_upper_per_100", 45.0)
        ),
        direct_density_hard_upper_per_100=(
            _float_with_legacy_default(
                args,
                "paper_aligned_direct_density_hard_upper_per_100",
                90.0,
                50.0,
            )
            if enabled
            else _float(args, "paper_aligned_direct_density_hard_upper_per_100", 50.0)
        ),
    )


def apply_coupled_innovation_defaults(args: Any) -> CoupledInnovationConfig:
    cfg = resolve_coupled_innovation_config(args)
    if not cfg.enabled:
        return cfg
    setattr(args, "risk_admission_mode", cfg.runtime_mode)
    setattr(args, "paper_aligned_recovery_commit_bridge", cfg.recovery_commit_bridge)
    setattr(args, "paper_aligned_defer_recovery_support_bridge", cfg.defer_recovery_support_bridge)
    setattr(args, "paper_aligned_recovery_commit_control", cfg.recovery_commit_control)
    setattr(args, "paper_aligned_direct_density_control", cfg.direct_density_control)
    setattr(args, "paper_aligned_direct_update_prev_desc_on_hold", cfg.direct_update_prev_desc_on_hold)
    setattr(args, "paper_aligned_direct_density_upper_per_100", cfg.direct_density_upper_per_100)
    setattr(args, "paper_aligned_direct_density_hard_upper_per_100", cfg.direct_density_hard_upper_per_100)
    return cfg
