from __future__ import annotations

from copy import copy
from dataclasses import asdict, dataclass, field, replace
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
RENDER_FRAME_ASSIMILATION_PROFILE = "render_frame_assimilation_v1"
POSE_ONLY_RENDER_SKELETON_PROFILE = "pose_only_render_skeleton_v1"
BASELINE_RENDER_LOCK_PROFILE = "pose_only_baseline_render_lock_v1"
BASELINE_RENDER_LOCK_INTRA_FRAME_PROFILE = "baseline_render_lock_intra_frame_v1"
BASELINE_RENDER_LOCK_INTRA_FRAME_V2_PROFILE = "baseline_render_lock_intra_frame_v2"
BASELINE_RENDER_LOCK_INTRA_FRAME_V3_PROFILE = "baseline_render_lock_intra_frame_v3"
BASELINE_RENDER_LOCK_INTRA_FRAME_V4_PROFILE = "baseline_render_lock_intra_frame_v4"
BASELINE_RENDER_LOCK_INTRA_FRAME_V5_PROFILE = "baseline_render_lock_intra_frame_v5"
BASELINE_RENDER_LOCK_INTRA_FRAME_V6_PROFILE = "baseline_render_lock_intra_frame_v6"
BASELINE_RENDER_LOCK_INTRA_FRAME_V7_PROFILE = "baseline_render_lock_intra_frame_v7"
BASELINE_RENDER_LOCK_INTRA_FRAME_V8_PROFILE = "baseline_render_lock_intra_frame_v8"
BASELINE_RENDER_LOCK_INTRA_FRAME_V9_PROFILE = "baseline_render_lock_intra_frame_v9"
BASELINE_RENDER_LOCK_INTRA_FRAME_V10_PROFILE = "baseline_render_lock_intra_frame_v10"
BASELINE_RENDER_LOCK_INTRA_FRAME_V11_PROFILE = "baseline_render_lock_intra_frame_v11"
BASELINE_RENDER_LOCK_INTRA_FRAME_V12_PROFILE = "baseline_render_lock_intra_frame_v12"
BASELINE_RENDER_LOCK_INTRA_FRAME_V13_PROFILE = "baseline_render_lock_intra_frame_v13"
BASELINE_RENDER_LOCK_INTRA_FRAME_V14_PROFILE = "baseline_render_lock_intra_frame_v14"
BASELINE_RENDER_LOCK_INTRA_FRAME_V15_PROFILE = "baseline_render_lock_intra_frame_v15"
BASELINE_RENDER_LOCK_INTRA_FRAME_V16_PROFILE = "baseline_render_lock_intra_frame_v16"
BASELINE_RENDER_LOCK_INTRA_FRAME_V17_PROFILE = "baseline_render_lock_intra_frame_v17"
BASELINE_RENDER_LOCK_INTRA_FRAME_V18_PROFILE = "baseline_render_lock_intra_frame_v18"
BASELINE_RENDER_LOCK_INTRA_FRAME_V19_PROFILE = "baseline_render_lock_intra_frame_v19"
BASELINE_RENDER_LOCK_INTRA_FRAME_V20_PROFILE = "baseline_render_lock_intra_frame_v20"
BASELINE_RENDER_LOCK_INTRA_FRAME_V21_PROFILE = "baseline_render_lock_intra_frame_v21"
BASELINE_RENDER_LOCK_INTRA_FRAME_V22_PROFILE = "baseline_render_lock_intra_frame_v22"
BASELINE_RENDER_LOCK_INTRA_FRAME_V23_PROFILE = "baseline_render_lock_intra_frame_v23"
BASELINE_RENDER_LOCK_INTRA_FRAME_V24_PROFILE = "baseline_render_lock_intra_frame_v24"
BASELINE_RENDER_LOCK_INTRA_FRAME_V25_PROFILE = "baseline_render_lock_intra_frame_v25"
BASELINE_RENDER_LOCK_INTRA_FRAME_V26_PROFILE = "baseline_render_lock_intra_frame_v26"
BASELINE_RENDER_LOCK_INTRA_FRAME_V27_PROFILE = "baseline_render_lock_intra_frame_v27"
BASELINE_RENDER_LOCK_INTRA_FRAME_V28_PROFILE = "baseline_render_lock_intra_frame_v28"
BASELINE_RENDER_LOCK_INTRA_FRAME_V29_PROFILE = "baseline_render_lock_intra_frame_v29"
BASELINE_RENDER_LOCK_INTRA_FRAME_V30_PROFILE = "baseline_render_lock_intra_frame_v30"
BASELINE_RENDER_LOCK_INTRA_FRAME_V31_PROFILE = "baseline_render_lock_intra_frame_v31"
BASELINE_RENDER_LOCK_INTRA_FRAME_V32_PROFILE = "baseline_render_lock_intra_frame_v32"
BASELINE_RENDER_LOCK_INTRA_FRAME_V33_PROFILE = "baseline_render_lock_intra_frame_v33"
BASELINE_RENDER_LOCK_INTRA_FRAME_V34_PROFILE = "baseline_render_lock_intra_frame_v34"
BASELINE_RENDER_LOCK_INTRA_FRAME_V35_PROFILE = "baseline_render_lock_intra_frame_v35"
BASELINE_RENDER_LOCK_INTRA_FRAME_V36_PROFILE = "baseline_render_lock_intra_frame_v36"
BASELINE_RENDER_LOCK_INTRA_FRAME_V37_PROFILE = "baseline_render_lock_intra_frame_v37"
BASELINE_RENDER_LOCK_INTRA_FRAME_V38_PROFILE = "baseline_render_lock_intra_frame_v38"
BASELINE_RENDER_LOCK_INTRA_FRAME_V39_PROFILE = "baseline_render_lock_intra_frame_v39"
BASELINE_RENDER_LOCK_INTRA_FRAME_V40_PROFILE = "baseline_render_lock_intra_frame_v40"
BASELINE_RENDER_LOCK_INTRA_FRAME_V41_PROFILE = "baseline_render_lock_intra_frame_v41"
BASELINE_RENDER_LOCK_INTRA_FRAME_V42_PROFILE = "baseline_render_lock_intra_frame_v42"
BASELINE_RENDER_LOCK_INTRA_FRAME_V43_PROFILE = "baseline_render_lock_intra_frame_v43"
BASELINE_RENDER_LOCK_INTRA_FRAME_V44_PROFILE = "baseline_render_lock_intra_frame_v44"
BASELINE_RENDER_LOCK_INTRA_FRAME_V45_PROFILE = "baseline_render_lock_intra_frame_v45"
BASELINE_RENDER_LOCK_INTRA_FRAME_V46_PROFILE = "baseline_render_lock_intra_frame_v46"
BASELINE_RENDER_LOCK_INTRA_FRAME_V47_PROFILE = "baseline_render_lock_intra_frame_v47"
BASELINE_RENDER_LOCK_INTRA_FRAME_V48_PROFILE = "baseline_render_lock_intra_frame_v48"
BASELINE_RENDER_LOCK_INTRA_FRAME_V49_PROFILE = "baseline_render_lock_intra_frame_v49"
BASELINE_RENDER_LOCK_INTRA_FRAME_V50_PROFILE = "baseline_render_lock_intra_frame_v50"
BASELINE_RENDER_LOCK_INTRA_FRAME_V51_PROFILE = "baseline_render_lock_intra_frame_v51"
BASELINE_RENDER_LOCK_INTRA_FRAME_V52_PROFILE = "baseline_render_lock_intra_frame_v52"
BASELINE_RENDER_LOCK_INTRA_FRAME_V53_PROFILE = "baseline_render_lock_intra_frame_v53"
BASELINE_RENDER_LOCK_INTRA_FRAME_V54_PROFILE = "baseline_render_lock_intra_frame_v54"
BASELINE_RENDER_LOCK_INTRA_FRAME_V55_PROFILE = "baseline_render_lock_intra_frame_v55"
BASELINE_RENDER_LOCK_INTRA_FRAME_V56_PROFILE = "baseline_render_lock_intra_frame_v56"
BASELINE_RENDER_LOCK_INTRA_FRAME_V57_PROFILE = "baseline_render_lock_intra_frame_v57"
BASELINE_RENDER_LOCK_INTRA_FRAME_V58_PROFILE = "baseline_render_lock_intra_frame_v58"
BASELINE_RENDER_LOCK_INTRA_FRAME_V59_PROFILE = "baseline_render_lock_intra_frame_v59"
BASELINE_RENDER_LOCK_INTRA_FRAME_V60_PROFILE = "baseline_render_lock_intra_frame_v60"
BASELINE_RENDER_LOCK_INTRA_FRAME_V61_PROFILE = "baseline_render_lock_intra_frame_v61"
BASELINE_RENDER_LOCK_INTRA_FRAME_V62_PROFILE = "baseline_render_lock_intra_frame_v62"
BASELINE_RENDER_LOCK_INTRA_FRAME_V63_PROFILE = "baseline_render_lock_intra_frame_v63"
BASELINE_RENDER_LOCK_INTRA_FRAME_V64_PROFILE = "baseline_render_lock_intra_frame_v64"


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
    recovery_commit_materialization: str = "off"
    direct_density_control: str = "off"
    direct_update_prev_desc_on_hold: str = "off"
    direct_density_upper_per_100: float = 75.0
    direct_density_hard_upper_per_100: float = 90.0
    pose_render_assimilation_profile: str = "off"
    pose_render_texture_sampling: str = "off"
    pose_render_texture_sampling_alpha: float = 0.12
    pose_render_texture_sampling_min_selectivity: float = 0.18
    pose_render_keyframe_sampling: str = "off"
    pose_render_edge_loss: str = "off"
    pose_render_psnr_loss: str = "off"
    pose_render_psnr_loss_target_mask: str = "off"
    pose_render_psnr_loss_support_weight: str = "off"
    pose_render_psnr_loss_context_weight: str = "off"
    pose_render_psnr_loss_max_applied_ratio: float = 1.0
    pose_render_psnr_loss_ambiguity_low: float = 0.0
    pose_render_psnr_loss_ambiguity_high: float = 0.0
    pose_render_psnr_loss_ambiguity_min_applied_ratio: float = 1.0
    pose_render_psnr_loss_scene_guard: str = "off"
    pose_render_psnr_loss_scene_guard_raw_low: float = 0.0
    pose_render_psnr_loss_scene_guard_raw_high: float = 0.0
    pose_render_psnr_loss_scene_guard_min_ratio: float = 1.0
    pose_render_psnr_loss_scene_guard_min_events: int = 0
    pose_render_psnr_loss_scene_guard_min_applied: int = 0
    pose_render_psnr_loss_scene_guard_max_events: int = 0
    pose_render_psnr_loss_structure_gate: str = "off"
    pose_render_psnr_loss_structure_min_score: float = 0.0
    pose_render_psnr_loss_structure_min_raw_mean: float = 0.0
    pose_render_psnr_loss_late_raw_stop: str = "off"
    pose_render_psnr_loss_late_raw_stop_min_mean: float = 0.0
    pose_render_psnr_loss_health_gate: str = "off"
    pose_render_psnr_loss_health_min_score: float = 0.0
    pose_render_psnr_loss_health_min_raw_loss: float = 0.0
    pose_render_extra_optimization: str = "off"
    pose_render_update_gate: str = "off"
    pose_render_update_gate_psnr_health_min_score: float = 0.0
    pose_render_update_gate_psnr_health_min_raw_loss: float = 0.0
    pose_render_update_gate_psnr_health_scale: float = 1.0
    pose_render_pre_refine: str = "off"
    pose_render_init_weighting: str = "off"
    render_frame_policy: str = "off"
    training_background_mode: str = "random_v1"
    test_exposure_harmonization: str = "neighbor_average_v1"
    test_exposure_guard_max_delta: float = 0.0
    test_render_calibration: str = "off"
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
            "recovery_commit_materialization": self.recovery_commit_materialization,
            "direct_density_control": self.direct_density_control,
            "direct_update_prev_desc_on_hold": self.direct_update_prev_desc_on_hold,
            "pose_render_assimilation_profile": self.pose_render_assimilation_profile,
            "pose_render_texture_sampling": self.pose_render_texture_sampling,
            "pose_render_texture_sampling_alpha": str(
                self.pose_render_texture_sampling_alpha
            ),
            "pose_render_texture_sampling_min_selectivity": str(
                self.pose_render_texture_sampling_min_selectivity
            ),
            "pose_render_keyframe_sampling": self.pose_render_keyframe_sampling,
            "pose_render_edge_loss": self.pose_render_edge_loss,
            "pose_render_psnr_loss": self.pose_render_psnr_loss,
            "pose_render_psnr_loss_target_mask": self.pose_render_psnr_loss_target_mask,
            "pose_render_psnr_loss_support_weight": self.pose_render_psnr_loss_support_weight,
            "pose_render_psnr_loss_context_weight": self.pose_render_psnr_loss_context_weight,
            "pose_render_psnr_loss_max_applied_ratio": str(
                self.pose_render_psnr_loss_max_applied_ratio
            ),
            "pose_render_psnr_loss_ambiguity_low": str(
                self.pose_render_psnr_loss_ambiguity_low
            ),
            "pose_render_psnr_loss_ambiguity_high": str(
                self.pose_render_psnr_loss_ambiguity_high
            ),
            "pose_render_psnr_loss_ambiguity_min_applied_ratio": str(
                self.pose_render_psnr_loss_ambiguity_min_applied_ratio
            ),
            "pose_render_psnr_loss_scene_guard": self.pose_render_psnr_loss_scene_guard,
            "pose_render_psnr_loss_scene_guard_raw_low": str(
                self.pose_render_psnr_loss_scene_guard_raw_low
            ),
            "pose_render_psnr_loss_scene_guard_raw_high": str(
                self.pose_render_psnr_loss_scene_guard_raw_high
            ),
            "pose_render_psnr_loss_scene_guard_min_ratio": str(
                self.pose_render_psnr_loss_scene_guard_min_ratio
            ),
            "pose_render_psnr_loss_scene_guard_min_events": str(
                self.pose_render_psnr_loss_scene_guard_min_events
            ),
            "pose_render_psnr_loss_scene_guard_min_applied": str(
                self.pose_render_psnr_loss_scene_guard_min_applied
            ),
            "pose_render_psnr_loss_scene_guard_max_events": str(
                self.pose_render_psnr_loss_scene_guard_max_events
            ),
            "pose_render_psnr_loss_structure_gate": self.pose_render_psnr_loss_structure_gate,
            "pose_render_psnr_loss_structure_min_score": str(
                self.pose_render_psnr_loss_structure_min_score
            ),
            "pose_render_psnr_loss_structure_min_raw_mean": str(
                self.pose_render_psnr_loss_structure_min_raw_mean
            ),
            "pose_render_psnr_loss_late_raw_stop": self.pose_render_psnr_loss_late_raw_stop,
            "pose_render_psnr_loss_late_raw_stop_min_mean": str(
                self.pose_render_psnr_loss_late_raw_stop_min_mean
            ),
            "pose_render_psnr_loss_health_gate": self.pose_render_psnr_loss_health_gate,
            "pose_render_psnr_loss_health_min_score": str(
                self.pose_render_psnr_loss_health_min_score
            ),
            "pose_render_psnr_loss_health_min_raw_loss": str(
                self.pose_render_psnr_loss_health_min_raw_loss
            ),
            "pose_render_extra_optimization": self.pose_render_extra_optimization,
            "pose_render_update_gate": self.pose_render_update_gate,
            "pose_render_update_gate_psnr_health_min_score": str(
                self.pose_render_update_gate_psnr_health_min_score
            ),
            "pose_render_update_gate_psnr_health_min_raw_loss": str(
                self.pose_render_update_gate_psnr_health_min_raw_loss
            ),
            "pose_render_update_gate_psnr_health_scale": str(
                self.pose_render_update_gate_psnr_health_scale
            ),
            "pose_render_pre_refine": self.pose_render_pre_refine,
            "pose_render_init_weighting": self.pose_render_init_weighting,
            "render_frame_policy": self.render_frame_policy,
            "training_background_mode": self.training_background_mode,
            "test_exposure_harmonization": self.test_exposure_harmonization,
            "test_exposure_guard_max_delta": str(self.test_exposure_guard_max_delta),
            "test_render_calibration": self.test_render_calibration,
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
    assimilation_profile = _str(args, "paper_aligned_pose_render_assimilation_profile", "off")
    if assimilation_profile == BASELINE_RENDER_LOCK_INTRA_FRAME_V58_PROFILE:
        inherited_args = copy(args)
        setattr(
            inherited_args,
            "paper_aligned_pose_render_assimilation_profile",
            BASELINE_RENDER_LOCK_INTRA_FRAME_V57_PROFILE,
        )
        inherited_cfg = resolve_coupled_innovation_config(inherited_args)
        return replace(
            inherited_cfg,
            pose_render_assimilation_profile=BASELINE_RENDER_LOCK_INTRA_FRAME_V58_PROFILE,
            pose_render_psnr_loss_target_mask="mask_aware_random_nonzero_gt_v1",
            training_background_mode="mask_aware_dark_scene_deterministic_random_v1",
        )
    if assimilation_profile == BASELINE_RENDER_LOCK_INTRA_FRAME_V59_PROFILE:
        inherited_args = copy(args)
        setattr(
            inherited_args,
            "paper_aligned_pose_render_assimilation_profile",
            BASELINE_RENDER_LOCK_INTRA_FRAME_V56_PROFILE,
        )
        inherited_cfg = resolve_coupled_innovation_config(inherited_args)
        return replace(
            inherited_cfg,
            pose_render_assimilation_profile=BASELINE_RENDER_LOCK_INTRA_FRAME_V59_PROFILE,
            pose_render_psnr_loss_target_mask="off",
            pose_render_psnr_loss_context_weight="mask_aware_no_mask_boost_v1",
        )
    if assimilation_profile == BASELINE_RENDER_LOCK_INTRA_FRAME_V60_PROFILE:
        inherited_args = copy(args)
        setattr(
            inherited_args,
            "paper_aligned_pose_render_assimilation_profile",
            BASELINE_RENDER_LOCK_INTRA_FRAME_V56_PROFILE,
        )
        inherited_cfg = resolve_coupled_innovation_config(inherited_args)
        return replace(
            inherited_cfg,
            pose_render_assimilation_profile=BASELINE_RENDER_LOCK_INTRA_FRAME_V60_PROFILE,
            pose_render_psnr_loss_target_mask="off",
            pose_render_psnr_loss_context_weight="mask_aware_no_mask_raw_response_boost_v1",
        )
    if assimilation_profile == BASELINE_RENDER_LOCK_INTRA_FRAME_V61_PROFILE:
        inherited_args = copy(args)
        setattr(
            inherited_args,
            "paper_aligned_pose_render_assimilation_profile",
            BASELINE_RENDER_LOCK_INTRA_FRAME_V56_PROFILE,
        )
        inherited_cfg = resolve_coupled_innovation_config(inherited_args)
        return replace(
            inherited_cfg,
            pose_render_assimilation_profile=BASELINE_RENDER_LOCK_INTRA_FRAME_V61_PROFILE,
            pose_render_psnr_loss_target_mask="off",
            pose_render_psnr_loss_context_weight="mask_aware_no_mask_raw_response_gate_boost_v1",
        )
    if assimilation_profile == BASELINE_RENDER_LOCK_INTRA_FRAME_V62_PROFILE:
        inherited_args = copy(args)
        setattr(
            inherited_args,
            "paper_aligned_pose_render_assimilation_profile",
            BASELINE_RENDER_LOCK_INTRA_FRAME_V56_PROFILE,
        )
        inherited_cfg = resolve_coupled_innovation_config(inherited_args)
        return replace(
            inherited_cfg,
            pose_render_assimilation_profile=BASELINE_RENDER_LOCK_INTRA_FRAME_V62_PROFILE,
            pose_render_psnr_loss_target_mask="off",
            pose_render_psnr_loss_context_weight="mask_aware_no_mask_scene_low_gate_boost_v1",
        )
    if assimilation_profile == BASELINE_RENDER_LOCK_INTRA_FRAME_V63_PROFILE:
        inherited_args = copy(args)
        setattr(
            inherited_args,
            "paper_aligned_pose_render_assimilation_profile",
            BASELINE_RENDER_LOCK_INTRA_FRAME_V56_PROFILE,
        )
        inherited_cfg = resolve_coupled_innovation_config(inherited_args)
        return replace(
            inherited_cfg,
            pose_render_assimilation_profile=BASELINE_RENDER_LOCK_INTRA_FRAME_V63_PROFILE,
            pose_render_psnr_loss_target_mask="off",
            pose_render_psnr_loss_context_weight="mask_aware_no_mask_scene_low_fast_gate_boost_v1",
        )
    if assimilation_profile == BASELINE_RENDER_LOCK_INTRA_FRAME_V64_PROFILE:
        inherited_args = copy(args)
        setattr(
            inherited_args,
            "paper_aligned_pose_render_assimilation_profile",
            BASELINE_RENDER_LOCK_INTRA_FRAME_V31_PROFILE,
        )
        inherited_cfg = resolve_coupled_innovation_config(inherited_args)
        return replace(
            inherited_cfg,
            pose_render_assimilation_profile=BASELINE_RENDER_LOCK_INTRA_FRAME_V64_PROFILE,
            training_background_mode="mask_aware_dark_scene_fixed_black_v1",
            test_exposure_harmonization="dark_scene_off_guarded_v1",
        )
    runtime_off_clean_baseline = bool(
        assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V23_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V24_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V29_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V30_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V31_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V32_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V33_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V34_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V35_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V36_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V37_PROFILE,
        }
    )
    runtime_off_profile = bool(
        assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V22_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V23_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V24_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V29_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V30_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V31_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V32_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V33_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V34_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V35_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V36_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V37_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V52_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V53_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V54_PROFILE,
        }
    )
    runtime_mode = (
        "off"
        if enabled and runtime_off_profile
        else "paper_aligned_baseline_passthrough"
        if enabled and assimilation_profile == BASELINE_RENDER_LOCK_INTRA_FRAME_V21_PROFILE
        else COUPLED_RUNTIME_MODE
        if enabled
        else requested
    )
    render_frame_assimilation = bool(
        enabled and assimilation_profile == RENDER_FRAME_ASSIMILATION_PROFILE
    )
    pose_only_render_skeleton = bool(
        enabled and assimilation_profile == POSE_ONLY_RENDER_SKELETON_PROFILE
    )
    baseline_render_lock = bool(
        enabled
        and assimilation_profile
        in {
            BASELINE_RENDER_LOCK_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V2_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V3_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V4_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V5_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V6_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V7_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V8_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V9_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V10_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V11_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V12_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V13_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V14_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V15_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V16_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V17_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V18_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V19_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V20_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V21_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V22_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V25_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V26_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V27_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V28_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V29_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V30_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V31_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V33_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V34_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V35_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V36_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V37_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V38_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V39_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V40_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V41_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V42_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V43_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V44_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V45_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V46_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V47_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V48_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V49_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V50_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V51_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V52_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V53_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V54_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V55_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V56_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V57_PROFILE,
        }
    )
    intra_frame_refine = bool(
        enabled
        and assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V2_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V3_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V4_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V5_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V6_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V7_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V9_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V10_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V11_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V12_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V13_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V14_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V15_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V16_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V17_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V18_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V19_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V20_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V25_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V37_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V38_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V39_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V40_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V41_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V42_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V43_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V44_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V45_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V46_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V47_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V48_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V49_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V50_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V51_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V52_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V55_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V56_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V57_PROFILE,
        }
    )
    pose_safe_render_skeleton = bool(render_frame_assimilation or pose_only_render_skeleton)
    pose_side_recovery_only = bool(pose_safe_render_skeleton or baseline_render_lock)
    direct_density_default = "pose_safe_streaming_memory_v1" if pose_safe_render_skeleton else "off"
    render_frame_policy = "baseline_keyframe_lock_v1" if baseline_render_lock else "off"
    recovery_materialization_default = "off"
    recovery_control_default = "off"
    pose_render_texture_sampling = (
        "residual_edge_response_guard_mask_conservative_v6"
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V49_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V50_PROFILE,
        }
        else
        "residual_edge_response_guard_non_dark_v5"
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V47_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V48_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V51_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V52_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V55_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V56_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V57_PROFILE,
        }
        else
        "residual_edge_response_guard_v4"
        if assimilation_profile == BASELINE_RENDER_LOCK_INTRA_FRAME_V34_PROFILE
        else
        "residual_edge_response_guard_v2"
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V28_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V29_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V30_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V31_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V32_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V33_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V34_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V35_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V36_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V37_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V39_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V40_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V41_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V42_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V43_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V46_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V48_PROFILE,
        }
        else "residual_edge_v1"
        if (
            render_frame_assimilation
            or assimilation_profile == BASELINE_RENDER_LOCK_INTRA_FRAME_V8_PROFILE
            or assimilation_profile == BASELINE_RENDER_LOCK_INTRA_FRAME_V26_PROFILE
        )
        else "off"
    )
    pose_render_texture_sampling_alpha = (
        _float_with_legacy_default(
            args,
            "paper_aligned_pose_render_texture_sampling_alpha",
            0.03,
            0.12,
        )
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V28_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V29_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V30_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V31_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V32_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V33_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V34_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V35_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V36_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V37_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V39_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V40_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V41_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V42_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V43_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V46_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V47_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V48_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V49_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V50_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V51_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V52_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V55_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V56_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V57_PROFILE,
        }
        else _float(args, "paper_aligned_pose_render_texture_sampling_alpha", 0.12)
    )
    pose_render_texture_sampling_min_selectivity = (
        _float_with_legacy_default(
            args,
            "paper_aligned_pose_render_texture_sampling_min_selectivity",
            2.0
            if assimilation_profile
            in {
                BASELINE_RENDER_LOCK_INTRA_FRAME_V33_PROFILE,
                BASELINE_RENDER_LOCK_INTRA_FRAME_V34_PROFILE,
                BASELINE_RENDER_LOCK_INTRA_FRAME_V35_PROFILE,
                BASELINE_RENDER_LOCK_INTRA_FRAME_V36_PROFILE,
                BASELINE_RENDER_LOCK_INTRA_FRAME_V37_PROFILE,
                BASELINE_RENDER_LOCK_INTRA_FRAME_V39_PROFILE,
                BASELINE_RENDER_LOCK_INTRA_FRAME_V40_PROFILE,
                BASELINE_RENDER_LOCK_INTRA_FRAME_V41_PROFILE,
                BASELINE_RENDER_LOCK_INTRA_FRAME_V42_PROFILE,
                BASELINE_RENDER_LOCK_INTRA_FRAME_V43_PROFILE,
                BASELINE_RENDER_LOCK_INTRA_FRAME_V46_PROFILE,
                BASELINE_RENDER_LOCK_INTRA_FRAME_V47_PROFILE,
                BASELINE_RENDER_LOCK_INTRA_FRAME_V48_PROFILE,
                BASELINE_RENDER_LOCK_INTRA_FRAME_V49_PROFILE,
                BASELINE_RENDER_LOCK_INTRA_FRAME_V50_PROFILE,
                BASELINE_RENDER_LOCK_INTRA_FRAME_V51_PROFILE,
                BASELINE_RENDER_LOCK_INTRA_FRAME_V52_PROFILE,
                BASELINE_RENDER_LOCK_INTRA_FRAME_V55_PROFILE,
                BASELINE_RENDER_LOCK_INTRA_FRAME_V56_PROFILE,
                BASELINE_RENDER_LOCK_INTRA_FRAME_V57_PROFILE,
            }
            else 1.8,
            0.18,
        )
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V28_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V29_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V30_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V31_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V32_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V33_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V34_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V35_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V36_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V37_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V39_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V40_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V41_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V42_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V43_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V46_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V47_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V48_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V49_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V50_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V51_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V52_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V55_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V56_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V57_PROFILE,
        }
        else _float(args, "paper_aligned_pose_render_texture_sampling_min_selectivity", 0.18)
    )
    pose_render_keyframe_sampling = "pose_confidence_temporal_v1" if render_frame_assimilation else "off"
    pose_render_edge_loss = "gradient_pose_adaptive_v1" if render_frame_assimilation else "off"
    pose_render_psnr_loss = (
        "robust_mse_pose_risk_v1"
        if assimilation_profile == BASELINE_RENDER_LOCK_INTRA_FRAME_V6_PROFILE
        else "freq_mse_pose_risk_v1"
        if assimilation_profile == BASELINE_RENDER_LOCK_INTRA_FRAME_V7_PROFILE
        else "mse_pose_safe_v1"
        if (render_frame_assimilation or intra_frame_refine)
        else "off"
    )
    pose_render_psnr_loss_target_mask = (
        "nonzero_gt_v1"
        if assimilation_profile == BASELINE_RENDER_LOCK_INTRA_FRAME_V3_PROFILE
        else "mask_aware_random_nonzero_gt_v1"
        if assimilation_profile == BASELINE_RENDER_LOCK_INTRA_FRAME_V57_PROFILE
        else _str(args, "paper_aligned_pose_render_psnr_loss_target_mask", "off")
    )
    pose_render_psnr_loss_support_weight = (
        "raw_pose_support_v1"
        if assimilation_profile == BASELINE_RENDER_LOCK_INTRA_FRAME_V4_PROFILE
        else (
            "raw_pose_support_gate_v1"
            if assimilation_profile == BASELINE_RENDER_LOCK_INTRA_FRAME_V5_PROFILE
            else _str(args, "paper_aligned_pose_render_psnr_loss_support_weight", "off")
        )
    )
    pose_render_psnr_loss_context_weight = _str(
        args,
        "paper_aligned_pose_render_psnr_loss_context_weight",
        "off",
    )
    pose_render_psnr_loss_max_applied_ratio = (
        0.08
        if assimilation_profile == BASELINE_RENDER_LOCK_INTRA_FRAME_V9_PROFILE
        else _float(args, "paper_aligned_pose_render_psnr_loss_max_applied_ratio", 1.0)
    )
    pose_render_psnr_loss_ambiguity_low = (
        0.003
        if assimilation_profile == BASELINE_RENDER_LOCK_INTRA_FRAME_V10_PROFILE
        else _float(args, "paper_aligned_pose_render_psnr_loss_ambiguity_low", 0.0)
    )
    pose_render_psnr_loss_ambiguity_high = (
        0.005
        if assimilation_profile == BASELINE_RENDER_LOCK_INTRA_FRAME_V10_PROFILE
        else _float(args, "paper_aligned_pose_render_psnr_loss_ambiguity_high", 0.0)
    )
    pose_render_psnr_loss_ambiguity_min_applied_ratio = (
        0.08
        if assimilation_profile == BASELINE_RENDER_LOCK_INTRA_FRAME_V10_PROFILE
        else _float(
            args,
            "paper_aligned_pose_render_psnr_loss_ambiguity_min_applied_ratio",
            1.0,
        )
    )
    pose_render_psnr_loss_scene_guard = (
        "raw_loss_ratio_precommit_non_dark_mask_guard_v3"
        if assimilation_profile == BASELINE_RENDER_LOCK_INTRA_FRAME_V48_PROFILE
        else
        "mask_aware_dark_background_guard_v4"
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V56_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V57_PROFILE,
        }
        else
        "raw_loss_ratio_precommit_non_dark_guard_v2"
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V47_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V50_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V51_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V52_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V55_PROFILE,
        }
        else
        "raw_loss_ratio_precommit_coverage_guard_v1"
        if assimilation_profile == BASELINE_RENDER_LOCK_INTRA_FRAME_V43_PROFILE
        else "raw_loss_ratio_precommit_render_gap_guard_v1"
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V40_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V41_PROFILE,
        }
        else "raw_loss_ratio_precommit_guard_v1"
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V39_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V42_PROFILE,
        }
        else "raw_loss_ratio_guard_v1"
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V11_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V12_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V13_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V14_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V15_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V16_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V17_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V18_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V19_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V20_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V25_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V37_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V38_PROFILE,
        }
        else _str(args, "paper_aligned_pose_render_psnr_loss_scene_guard", "off")
    )
    pose_render_psnr_loss_scene_guard_raw_low = (
        0.0028
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V11_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V12_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V13_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V14_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V15_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V16_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V17_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V18_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V19_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V20_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V25_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V37_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V38_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V39_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V40_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V41_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V42_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V43_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V47_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V48_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V50_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V51_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V52_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V55_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V56_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V57_PROFILE,
        }
        else _float(args, "paper_aligned_pose_render_psnr_loss_scene_guard_raw_low", 0.0)
    )
    pose_render_psnr_loss_scene_guard_raw_high = (
        0.0055
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V11_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V12_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V13_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V14_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V15_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V16_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V17_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V18_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V19_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V20_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V25_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V37_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V38_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V39_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V40_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V41_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V42_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V43_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V47_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V48_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V50_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V51_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V52_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V55_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V56_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V57_PROFILE,
        }
        else _float(args, "paper_aligned_pose_render_psnr_loss_scene_guard_raw_high", 0.0)
    )
    pose_render_psnr_loss_scene_guard_min_ratio = (
        0.10
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V11_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V12_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V13_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V14_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V15_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V16_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V17_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V18_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V19_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V20_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V25_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V37_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V38_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V39_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V40_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V41_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V42_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V43_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V47_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V48_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V50_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V51_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V52_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V55_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V56_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V57_PROFILE,
        }
        else _float(args, "paper_aligned_pose_render_psnr_loss_scene_guard_min_ratio", 1.0)
    )
    pose_render_psnr_loss_scene_guard_min_events = (
        384
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V11_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V12_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V13_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V14_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V15_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V16_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V17_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V18_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V19_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V20_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V25_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V37_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V38_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V39_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V40_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V41_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V42_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V43_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V47_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V48_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V50_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V51_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V52_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V55_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V56_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V57_PROFILE,
        }
        else _int(args, "paper_aligned_pose_render_psnr_loss_scene_guard_min_events", 0)
    )
    pose_render_psnr_loss_scene_guard_min_applied = (
        48
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V11_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V12_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V13_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V14_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V15_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V16_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V17_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V18_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V19_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V20_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V25_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V37_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V38_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V39_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V40_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V41_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V42_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V43_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V47_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V48_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V50_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V51_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V52_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V55_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V56_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V57_PROFILE,
        }
        else _int(args, "paper_aligned_pose_render_psnr_loss_scene_guard_min_applied", 0)
    )
    pose_render_psnr_loss_scene_guard_max_events = (
        1000
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V12_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V13_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V14_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V15_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V16_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V17_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V18_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V19_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V20_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V25_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V37_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V38_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V39_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V40_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V41_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V42_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V43_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V47_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V48_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V50_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V51_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V52_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V55_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V56_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V57_PROFILE,
        }
        else _int(args, "paper_aligned_pose_render_psnr_loss_scene_guard_max_events", 0)
    )
    pose_render_psnr_loss_structure_gate = (
        "gradient_correlation_v1"
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V13_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V37_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V38_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V39_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V40_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V41_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V42_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V43_PROFILE,
        }
        else _str(args, "paper_aligned_pose_render_psnr_loss_structure_gate", "off")
    )
    pose_render_psnr_loss_structure_min_score = (
        0.55
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V13_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V37_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V38_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V39_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V40_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V41_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V42_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V43_PROFILE,
        }
        else _float(
            args,
            "paper_aligned_pose_render_psnr_loss_structure_min_score",
            0.0,
        )
    )
    pose_render_psnr_loss_structure_min_raw_mean = (
        0.0065
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V13_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V37_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V38_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V39_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V40_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V41_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V42_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V43_PROFILE,
        }
        else _float(
            args,
            "paper_aligned_pose_render_psnr_loss_structure_min_raw_mean",
            0.0,
        )
    )
    pose_render_psnr_loss_late_raw_stop = (
        "raw_mean_stop_v1"
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V14_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V15_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V16_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V17_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V18_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V19_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V20_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V25_PROFILE,
        }
        else _str(args, "paper_aligned_pose_render_psnr_loss_late_raw_stop", "off")
    )
    pose_render_psnr_loss_late_raw_stop_min_mean = (
        0.0065
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V14_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V15_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V16_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V17_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V18_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V19_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V20_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V25_PROFILE,
        }
        else _float(
            args,
            "paper_aligned_pose_render_psnr_loss_late_raw_stop_min_mean",
            0.0,
        )
    )
    pose_render_psnr_loss_health_gate = (
        "gradient_correlation_v1"
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V17_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V18_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V19_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V20_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V25_PROFILE,
        }
        else _str(args, "paper_aligned_pose_render_psnr_loss_health_gate", "off")
    )
    pose_render_psnr_loss_health_min_score = (
        0.62
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V17_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V18_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V19_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V20_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V25_PROFILE,
        }
        else _float(args, "paper_aligned_pose_render_psnr_loss_health_min_score", 0.0)
    )
    pose_render_psnr_loss_health_min_raw_loss = (
        0.0045
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V17_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V18_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V19_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V20_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V25_PROFILE,
        }
        else _float(args, "paper_aligned_pose_render_psnr_loss_health_min_raw_loss", 0.0)
    )
    pose_render_update_gate_psnr_health_min_score = (
        0.84
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V18_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V19_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V20_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V25_PROFILE,
        }
        else _float(
            args,
            "paper_aligned_pose_render_update_gate_psnr_health_min_score",
            0.0,
        )
    )
    pose_render_update_gate_psnr_health_min_raw_loss = (
        0.0045
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V18_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V19_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V20_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V25_PROFILE,
        }
        else _float(
            args,
            "paper_aligned_pose_render_update_gate_psnr_health_min_raw_loss",
            0.0,
        )
    )
    pose_render_update_gate_psnr_health_scale = (
        0.55
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V18_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V19_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V20_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V25_PROFILE,
        }
        else _float(
            args,
            "paper_aligned_pose_render_update_gate_psnr_health_scale",
            1.0,
        )
    )
    training_background_mode = (
        "dark_scene_fixed_black_v1"
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V44_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V45_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V46_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V47_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V48_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V49_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V50_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V51_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V52_PROFILE,
        }
        else "mask_aware_dark_scene_fixed_black_v1"
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V56_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V57_PROFILE,
        }
        else "mask_aware_fixed_black_v1"
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V54_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V55_PROFILE,
        }
        else "target_mean_v1"
        if assimilation_profile == BASELINE_RENDER_LOCK_INTRA_FRAME_V15_PROFILE
        else (
            "deterministic_random_v1"
            if assimilation_profile == BASELINE_RENDER_LOCK_INTRA_FRAME_V16_PROFILE
            else _str(args, "paper_aligned_training_background_mode", "random_v1")
        )
    )
    test_exposure_harmonization = (
        "off"
        if assimilation_profile == BASELINE_RENDER_LOCK_INTRA_FRAME_V44_PROFILE
        else "dark_scene_off_guarded_v1"
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V45_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V46_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V47_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V48_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V49_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V50_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V51_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V52_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V55_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V56_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V57_PROFILE,
        }
        else "source_time_adaptive_v2"
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V35_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V36_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V37_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V38_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V39_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V40_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V41_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V42_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V43_PROFILE,
        }
        else "source_time_guarded_v1"
        if assimilation_profile == BASELINE_RENDER_LOCK_INTRA_FRAME_V20_PROFILE
        else "source_time_interp_v1"
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V19_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V25_PROFILE,
        }
        else _str(
            args,
            "paper_aligned_test_exposure_harmonization",
            "neighbor_average_v1",
        )
    )
    test_exposure_guard_max_delta = (
        0.35
        if assimilation_profile == BASELINE_RENDER_LOCK_INTRA_FRAME_V20_PROFILE
        else _float(args, "paper_aligned_test_exposure_guard_max_delta", 0.0)
    )
    test_render_calibration = (
        "diag_affine_v1"
        if assimilation_profile == BASELINE_RENDER_LOCK_INTRA_FRAME_V20_PROFILE
        else _str(args, "paper_aligned_test_render_calibration", "off")
    )
    pose_render_extra_optimization = (
        "pose_confidence_render_response_v2"
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V25_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V26_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V27_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V28_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V30_PROFILE,
        }
        else "pose_confidence_v1"
        if render_frame_assimilation
        else "off"
    )
    if assimilation_profile == BASELINE_RENDER_LOCK_INTRA_FRAME_V29_PROFILE:
        pose_render_extra_optimization = "off"
    if assimilation_profile in {
        BASELINE_RENDER_LOCK_INTRA_FRAME_V31_PROFILE,
        BASELINE_RENDER_LOCK_INTRA_FRAME_V32_PROFILE,
        BASELINE_RENDER_LOCK_INTRA_FRAME_V33_PROFILE,
        BASELINE_RENDER_LOCK_INTRA_FRAME_V34_PROFILE,
        BASELINE_RENDER_LOCK_INTRA_FRAME_V35_PROFILE,
    }:
        pose_render_extra_optimization = "render_response_v3"
    if assimilation_profile in {
        BASELINE_RENDER_LOCK_INTRA_FRAME_V36_PROFILE,
        BASELINE_RENDER_LOCK_INTRA_FRAME_V37_PROFILE,
        BASELINE_RENDER_LOCK_INTRA_FRAME_V39_PROFILE,
        BASELINE_RENDER_LOCK_INTRA_FRAME_V40_PROFILE,
    }:
        pose_render_extra_optimization = "render_response_v4"
    if assimilation_profile in {
        BASELINE_RENDER_LOCK_INTRA_FRAME_V41_PROFILE,
        BASELINE_RENDER_LOCK_INTRA_FRAME_V42_PROFILE,
        BASELINE_RENDER_LOCK_INTRA_FRAME_V43_PROFILE,
        BASELINE_RENDER_LOCK_INTRA_FRAME_V46_PROFILE,
        BASELINE_RENDER_LOCK_INTRA_FRAME_V47_PROFILE,
        BASELINE_RENDER_LOCK_INTRA_FRAME_V48_PROFILE,
        BASELINE_RENDER_LOCK_INTRA_FRAME_V51_PROFILE,
        BASELINE_RENDER_LOCK_INTRA_FRAME_V52_PROFILE,
        BASELINE_RENDER_LOCK_INTRA_FRAME_V55_PROFILE,
        BASELINE_RENDER_LOCK_INTRA_FRAME_V56_PROFILE,
        BASELINE_RENDER_LOCK_INTRA_FRAME_V57_PROFILE,
    }:
        pose_render_extra_optimization = "render_response_v5"
    if assimilation_profile in {
        BASELINE_RENDER_LOCK_INTRA_FRAME_V49_PROFILE,
        BASELINE_RENDER_LOCK_INTRA_FRAME_V50_PROFILE,
    }:
        pose_render_extra_optimization = "render_response_mask_conservative_v6"
    pose_render_update_gate = (
        "pose_confidence_soft_psnr_health_v1"
        if assimilation_profile
        in {
            BASELINE_RENDER_LOCK_INTRA_FRAME_V18_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V19_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V20_PROFILE,
            BASELINE_RENDER_LOCK_INTRA_FRAME_V25_PROFILE,
        }
        else "pose_confidence_soft_v1"
        if (render_frame_assimilation or intra_frame_refine)
        else "off"
    )
    if assimilation_profile in {
        BASELINE_RENDER_LOCK_INTRA_FRAME_V39_PROFILE,
        BASELINE_RENDER_LOCK_INTRA_FRAME_V40_PROFILE,
        BASELINE_RENDER_LOCK_INTRA_FRAME_V41_PROFILE,
        BASELINE_RENDER_LOCK_INTRA_FRAME_V42_PROFILE,
        BASELINE_RENDER_LOCK_INTRA_FRAME_V43_PROFILE,
        BASELINE_RENDER_LOCK_INTRA_FRAME_V51_PROFILE,
    }:
        pose_render_update_gate = "off"
    pose_render_pre_refine = "pose_only_v1" if render_frame_assimilation else "off"
    pose_render_init_weighting = "risk_opacity_adaptive_v2" if render_frame_assimilation else "off"
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
            "off"
            if runtime_off_clean_baseline
            else "v1"
            if enabled
            else "off",
        ),
        recovery_commit_control=(
            recovery_control_default
            if pose_side_recovery_only
            else _str(args, "paper_aligned_recovery_commit_control", "off")
        ),
        recovery_commit_materialization=(
            recovery_materialization_default
            if pose_side_recovery_only
            else _str(args, "paper_aligned_recovery_commit_materialization", "off")
        ),
        direct_density_control=(
            direct_density_default
            if pose_safe_render_skeleton
            else _str(args, "paper_aligned_direct_density_control", "off")
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
        pose_render_assimilation_profile=assimilation_profile,
        pose_render_texture_sampling=pose_render_texture_sampling,
        pose_render_texture_sampling_alpha=pose_render_texture_sampling_alpha,
        pose_render_texture_sampling_min_selectivity=(
            pose_render_texture_sampling_min_selectivity
        ),
        pose_render_keyframe_sampling=pose_render_keyframe_sampling,
        pose_render_edge_loss=pose_render_edge_loss,
        pose_render_psnr_loss=pose_render_psnr_loss,
        pose_render_psnr_loss_target_mask=pose_render_psnr_loss_target_mask,
        pose_render_psnr_loss_support_weight=pose_render_psnr_loss_support_weight,
        pose_render_psnr_loss_context_weight=pose_render_psnr_loss_context_weight,
        pose_render_psnr_loss_max_applied_ratio=pose_render_psnr_loss_max_applied_ratio,
        pose_render_psnr_loss_ambiguity_low=pose_render_psnr_loss_ambiguity_low,
        pose_render_psnr_loss_ambiguity_high=pose_render_psnr_loss_ambiguity_high,
        pose_render_psnr_loss_ambiguity_min_applied_ratio=(
            pose_render_psnr_loss_ambiguity_min_applied_ratio
        ),
        pose_render_psnr_loss_scene_guard=pose_render_psnr_loss_scene_guard,
        pose_render_psnr_loss_scene_guard_raw_low=(
            pose_render_psnr_loss_scene_guard_raw_low
        ),
        pose_render_psnr_loss_scene_guard_raw_high=(
            pose_render_psnr_loss_scene_guard_raw_high
        ),
        pose_render_psnr_loss_scene_guard_min_ratio=(
            pose_render_psnr_loss_scene_guard_min_ratio
        ),
        pose_render_psnr_loss_scene_guard_min_events=(
            pose_render_psnr_loss_scene_guard_min_events
        ),
        pose_render_psnr_loss_scene_guard_min_applied=(
            pose_render_psnr_loss_scene_guard_min_applied
        ),
        pose_render_psnr_loss_scene_guard_max_events=(
            pose_render_psnr_loss_scene_guard_max_events
        ),
        pose_render_psnr_loss_structure_gate=pose_render_psnr_loss_structure_gate,
        pose_render_psnr_loss_structure_min_score=(
            pose_render_psnr_loss_structure_min_score
        ),
        pose_render_psnr_loss_structure_min_raw_mean=(
            pose_render_psnr_loss_structure_min_raw_mean
        ),
        pose_render_psnr_loss_late_raw_stop=pose_render_psnr_loss_late_raw_stop,
        pose_render_psnr_loss_late_raw_stop_min_mean=(
            pose_render_psnr_loss_late_raw_stop_min_mean
        ),
        pose_render_psnr_loss_health_gate=pose_render_psnr_loss_health_gate,
        pose_render_psnr_loss_health_min_score=pose_render_psnr_loss_health_min_score,
        pose_render_psnr_loss_health_min_raw_loss=(
            pose_render_psnr_loss_health_min_raw_loss
        ),
        pose_render_extra_optimization=pose_render_extra_optimization,
        pose_render_update_gate=pose_render_update_gate,
        pose_render_update_gate_psnr_health_min_score=(
            pose_render_update_gate_psnr_health_min_score
        ),
        pose_render_update_gate_psnr_health_min_raw_loss=(
            pose_render_update_gate_psnr_health_min_raw_loss
        ),
        pose_render_update_gate_psnr_health_scale=(
            pose_render_update_gate_psnr_health_scale
        ),
        pose_render_pre_refine=pose_render_pre_refine,
        pose_render_init_weighting=pose_render_init_weighting,
        render_frame_policy=render_frame_policy,
        training_background_mode=training_background_mode,
        test_exposure_harmonization=test_exposure_harmonization,
        test_exposure_guard_max_delta=test_exposure_guard_max_delta,
        test_render_calibration=test_render_calibration,
    )


def apply_coupled_innovation_defaults(args: Any) -> CoupledInnovationConfig:
    cfg = resolve_coupled_innovation_config(args)
    if not cfg.enabled:
        return cfg
    setattr(args, "risk_admission_mode", cfg.runtime_mode)
    setattr(args, "paper_aligned_recovery_commit_bridge", cfg.recovery_commit_bridge)
    setattr(args, "paper_aligned_defer_recovery_support_bridge", cfg.defer_recovery_support_bridge)
    setattr(args, "paper_aligned_recovery_commit_control", cfg.recovery_commit_control)
    setattr(args, "paper_aligned_recovery_commit_materialization", cfg.recovery_commit_materialization)
    setattr(args, "paper_aligned_direct_density_control", cfg.direct_density_control)
    setattr(args, "paper_aligned_direct_update_prev_desc_on_hold", cfg.direct_update_prev_desc_on_hold)
    setattr(args, "paper_aligned_direct_density_upper_per_100", cfg.direct_density_upper_per_100)
    setattr(args, "paper_aligned_direct_density_hard_upper_per_100", cfg.direct_density_hard_upper_per_100)
    setattr(args, "paper_aligned_pose_render_assimilation_profile", cfg.pose_render_assimilation_profile)
    setattr(args, "paper_aligned_pose_render_texture_sampling", cfg.pose_render_texture_sampling)
    setattr(
        args,
        "paper_aligned_pose_render_texture_sampling_alpha",
        cfg.pose_render_texture_sampling_alpha,
    )
    setattr(
        args,
        "paper_aligned_pose_render_texture_sampling_min_selectivity",
        cfg.pose_render_texture_sampling_min_selectivity,
    )
    setattr(args, "paper_aligned_pose_render_keyframe_sampling", cfg.pose_render_keyframe_sampling)
    setattr(args, "paper_aligned_pose_render_edge_loss", cfg.pose_render_edge_loss)
    setattr(args, "paper_aligned_pose_render_psnr_loss", cfg.pose_render_psnr_loss)
    setattr(args, "paper_aligned_pose_render_psnr_loss_target_mask", cfg.pose_render_psnr_loss_target_mask)
    setattr(args, "paper_aligned_pose_render_psnr_loss_support_weight", cfg.pose_render_psnr_loss_support_weight)
    setattr(args, "paper_aligned_pose_render_psnr_loss_context_weight", cfg.pose_render_psnr_loss_context_weight)
    setattr(
        args,
        "paper_aligned_pose_render_psnr_loss_max_applied_ratio",
        cfg.pose_render_psnr_loss_max_applied_ratio,
    )
    setattr(
        args,
        "paper_aligned_pose_render_psnr_loss_ambiguity_low",
        cfg.pose_render_psnr_loss_ambiguity_low,
    )
    setattr(
        args,
        "paper_aligned_pose_render_psnr_loss_ambiguity_high",
        cfg.pose_render_psnr_loss_ambiguity_high,
    )
    setattr(
        args,
        "paper_aligned_pose_render_psnr_loss_ambiguity_min_applied_ratio",
        cfg.pose_render_psnr_loss_ambiguity_min_applied_ratio,
    )
    setattr(
        args,
        "paper_aligned_pose_render_psnr_loss_scene_guard",
        cfg.pose_render_psnr_loss_scene_guard,
    )
    setattr(
        args,
        "paper_aligned_pose_render_psnr_loss_scene_guard_raw_low",
        cfg.pose_render_psnr_loss_scene_guard_raw_low,
    )
    setattr(
        args,
        "paper_aligned_pose_render_psnr_loss_scene_guard_raw_high",
        cfg.pose_render_psnr_loss_scene_guard_raw_high,
    )
    setattr(
        args,
        "paper_aligned_pose_render_psnr_loss_scene_guard_min_ratio",
        cfg.pose_render_psnr_loss_scene_guard_min_ratio,
    )
    setattr(
        args,
        "paper_aligned_pose_render_psnr_loss_scene_guard_min_events",
        cfg.pose_render_psnr_loss_scene_guard_min_events,
    )
    setattr(
        args,
        "paper_aligned_pose_render_psnr_loss_scene_guard_min_applied",
        cfg.pose_render_psnr_loss_scene_guard_min_applied,
    )
    setattr(
        args,
        "paper_aligned_pose_render_psnr_loss_scene_guard_max_events",
        cfg.pose_render_psnr_loss_scene_guard_max_events,
    )
    setattr(
        args,
        "paper_aligned_pose_render_psnr_loss_structure_gate",
        cfg.pose_render_psnr_loss_structure_gate,
    )
    setattr(
        args,
        "paper_aligned_pose_render_psnr_loss_structure_min_score",
        cfg.pose_render_psnr_loss_structure_min_score,
    )
    setattr(
        args,
        "paper_aligned_pose_render_psnr_loss_structure_min_raw_mean",
        cfg.pose_render_psnr_loss_structure_min_raw_mean,
    )
    setattr(
        args,
        "paper_aligned_pose_render_psnr_loss_late_raw_stop",
        cfg.pose_render_psnr_loss_late_raw_stop,
    )
    setattr(
        args,
        "paper_aligned_pose_render_psnr_loss_late_raw_stop_min_mean",
        cfg.pose_render_psnr_loss_late_raw_stop_min_mean,
    )
    setattr(
        args,
        "paper_aligned_pose_render_psnr_loss_health_gate",
        cfg.pose_render_psnr_loss_health_gate,
    )
    setattr(
        args,
        "paper_aligned_pose_render_psnr_loss_health_min_score",
        cfg.pose_render_psnr_loss_health_min_score,
    )
    setattr(
        args,
        "paper_aligned_pose_render_psnr_loss_health_min_raw_loss",
        cfg.pose_render_psnr_loss_health_min_raw_loss,
    )
    setattr(args, "paper_aligned_pose_render_extra_optimization", cfg.pose_render_extra_optimization)
    setattr(args, "paper_aligned_pose_render_update_gate", cfg.pose_render_update_gate)
    setattr(
        args,
        "paper_aligned_pose_render_update_gate_psnr_health_min_score",
        cfg.pose_render_update_gate_psnr_health_min_score,
    )
    setattr(
        args,
        "paper_aligned_pose_render_update_gate_psnr_health_min_raw_loss",
        cfg.pose_render_update_gate_psnr_health_min_raw_loss,
    )
    setattr(
        args,
        "paper_aligned_pose_render_update_gate_psnr_health_scale",
        cfg.pose_render_update_gate_psnr_health_scale,
    )
    setattr(args, "paper_aligned_pose_render_pre_refine", cfg.pose_render_pre_refine)
    setattr(args, "paper_aligned_pose_render_init_weighting", cfg.pose_render_init_weighting)
    setattr(args, "paper_aligned_render_frame_policy", cfg.render_frame_policy)
    setattr(args, "paper_aligned_training_background_mode", cfg.training_background_mode)
    setattr(
        args,
        "paper_aligned_test_exposure_harmonization",
        cfg.test_exposure_harmonization,
    )
    setattr(
        args,
        "paper_aligned_test_exposure_guard_max_delta",
        cfg.test_exposure_guard_max_delta,
    )
    setattr(
        args,
        "paper_aligned_test_render_calibration",
        cfg.test_render_calibration,
    )
    return cfg
