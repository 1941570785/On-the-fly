from __future__ import annotations

from typing import Any


POSE_SAFE_DIRECT_DENSITY_MODE = "pose_safe_streaming_memory_v1"
INIT_WEIGHTING_MODE = "risk_opacity_v1"
INIT_WEIGHTING_ADAPTIVE_MODE = "risk_opacity_adaptive_v2"
INIT_WEIGHTING_MODES = {"off", INIT_WEIGHTING_MODE, INIT_WEIGHTING_ADAPTIVE_MODE}
POSE_RENDER_GATE_INFO_KEY = "_paper_aligned_pose_render_coupling"
ADAPTIVE_MIN_POSE_RISK = 0.20
ADAPTIVE_MAX_POSE_RISK = 0.32
ADAPTIVE_MIN_SAMPLE_OPACITY_SCALE = 0.88
ADAPTIVE_MIN_MATCH_OPACITY_SCALE = 0.95
ADAPTIVE_SCENE_MIN_EVENTS = 16
ADAPTIVE_SCENE_GENTLE_POSE_RISK_MEAN = 0.20
ADAPTIVE_SCENE_GENTLE_NOVELTY_MEAN = 0.35
ADAPTIVE_SCENE_MAX_RISK_ALPHA_MEAN = 1.0
ADAPTIVE_SCENE_LOW_POSE_RISK_MEAN = -1.0
ADAPTIVE_SCENE_LOW_NOVELTY_MEAN = -1.0
ADAPTIVE_SCENE_LOW_RISK_MIN_EVENTS = 64
ADAPTIVE_SCENE_STABLE_LOW_RISK_MIN_EVENTS = 32
ADAPTIVE_SCENE_STABLE_LOW_POSE_RISK_MEAN = 0.06
ADAPTIVE_SCENE_STABLE_LOW_NOVELTY_MEAN = 0.12
ADAPTIVE_SCENE_HIGH_UNCERTAINTY_BYPASS_RISK_ALPHA_MEAN = 2.0
ADAPTIVE_SCENE_HIGH_UNCERTAINTY_BYPASS_NOVELTY_MEAN = 2.0


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _pose_render_gate_info(keyframe_info: dict[str, Any] | None) -> dict[str, Any]:
    info = keyframe_info or {}
    gate: dict[str, Any] = {}
    nested = info.get(POSE_RENDER_GATE_INFO_KEY, None)
    if isinstance(nested, dict):
        gate.update(nested)
    for key in (
        "pose_risk_score",
        "pose_risk_high",
        "pose_render_risk_score",
        "pose_render_risk_high",
        "pose_render_risk_reason",
        "pose_render_pose_confidence",
        "pose_render_posterior_risk_score",
        "semantic_R_t",
        "representation_value_score",
        "novelty_value_score",
        "high_novelty_score",
        "support_needed_score",
        "utility_representation",
        "utility_coverage_gain",
        "stream_memory_representation_need",
        "new_view_event_score",
    ):
        if key in info:
            gate[key] = info[key]
    return gate


def _max_available(gate: dict[str, Any], keys: tuple[str, ...]) -> float:
    values = [_as_float(gate[key], 0.0) for key in keys if key in gate]
    return max(values) if values else 0.0


def pose_render_init_weighting_decision(
    *,
    mode: str | None,
    direct_density_mode: str | None,
    keyframe_info: dict[str, Any] | None,
    min_pose_risk: float = 0.16,
    max_pose_risk: float = 0.38,
    min_sample_opacity_scale: float = 0.82,
    min_match_opacity_scale: float = 0.92,
    max_representation_value: float = 0.65,
    max_novelty_value: float = 0.65,
    adaptive_novelty_threshold: float = 0.45,
    adaptive_risk_override_alpha: float = 0.65,
    adaptive_min_pose_risk: float = ADAPTIVE_MIN_POSE_RISK,
    adaptive_max_pose_risk: float = ADAPTIVE_MAX_POSE_RISK,
    adaptive_min_sample_opacity_scale: float = ADAPTIVE_MIN_SAMPLE_OPACITY_SCALE,
    adaptive_min_match_opacity_scale: float = ADAPTIVE_MIN_MATCH_OPACITY_SCALE,
    adaptive_scene_event_count: int = 0,
    adaptive_scene_pose_risk_mean: float = 0.0,
    adaptive_scene_novelty_mean: float = 0.0,
    adaptive_scene_risk_alpha_mean: float = 0.0,
    adaptive_scene_min_events: int = ADAPTIVE_SCENE_MIN_EVENTS,
    adaptive_scene_gentle_pose_risk_mean: float = ADAPTIVE_SCENE_GENTLE_POSE_RISK_MEAN,
    adaptive_scene_gentle_novelty_mean: float = ADAPTIVE_SCENE_GENTLE_NOVELTY_MEAN,
    adaptive_scene_max_risk_alpha_mean: float = ADAPTIVE_SCENE_MAX_RISK_ALPHA_MEAN,
    adaptive_scene_low_pose_risk_mean: float = ADAPTIVE_SCENE_LOW_POSE_RISK_MEAN,
    adaptive_scene_low_novelty_mean: float = ADAPTIVE_SCENE_LOW_NOVELTY_MEAN,
    adaptive_scene_low_risk_min_events: int = ADAPTIVE_SCENE_LOW_RISK_MIN_EVENTS,
    adaptive_scene_stable_low_risk_min_events: int = (
        ADAPTIVE_SCENE_STABLE_LOW_RISK_MIN_EVENTS
    ),
    adaptive_scene_stable_low_pose_risk_mean: float = (
        ADAPTIVE_SCENE_STABLE_LOW_POSE_RISK_MEAN
    ),
    adaptive_scene_stable_low_novelty_mean: float = (
        ADAPTIVE_SCENE_STABLE_LOW_NOVELTY_MEAN
    ),
    adaptive_scene_stable_low_risk_latched: bool = False,
    adaptive_scene_high_uncertainty_bypass_risk_alpha_mean: float = (
        ADAPTIVE_SCENE_HIGH_UNCERTAINTY_BYPASS_RISK_ALPHA_MEAN
    ),
    adaptive_scene_high_uncertainty_bypass_novelty_mean: float = (
        ADAPTIVE_SCENE_HIGH_UNCERTAINTY_BYPASS_NOVELTY_MEAN
    ),
) -> dict[str, Any]:
    mode_name = str(mode or "off")
    direct_mode = str(direct_density_mode or "off")
    adaptive_mode = mode_name == INIT_WEIGHTING_ADAPTIVE_MODE
    info = keyframe_info or {}
    gate = _pose_render_gate_info(info)
    pose_risk = _as_float(
        gate.get(
            "pose_render_risk_score",
            gate.get("pose_risk_score", gate.get("semantic_R_t", 0.0)),
        ),
        0.0,
    )
    pose_risk_high = _as_bool(
        gate.get("pose_render_risk_high", gate.get("pose_risk_high", False)), False
    )
    scene_events = max(0, int(adaptive_scene_event_count))
    scene_pose_mean = float(adaptive_scene_pose_risk_mean)
    scene_novelty_mean = float(adaptive_scene_novelty_mean)
    scene_alpha_mean = float(adaptive_scene_risk_alpha_mean)
    scene_min_events = max(0, int(adaptive_scene_min_events))
    scene_gentle_pose_mean = float(adaptive_scene_gentle_pose_risk_mean)
    scene_gentle_novelty_mean = float(adaptive_scene_gentle_novelty_mean)
    scene_max_alpha_mean = float(adaptive_scene_max_risk_alpha_mean)
    scene_low_pose_mean = float(adaptive_scene_low_pose_risk_mean)
    scene_low_novelty_mean = float(adaptive_scene_low_novelty_mean)
    scene_low_risk_min_events = max(0, int(adaptive_scene_low_risk_min_events))
    scene_stable_low_risk_min_events = max(
        0, int(adaptive_scene_stable_low_risk_min_events)
    )
    scene_stable_low_pose_mean = float(adaptive_scene_stable_low_pose_risk_mean)
    scene_stable_low_novelty_mean = float(adaptive_scene_stable_low_novelty_mean)
    scene_stable_low_latched = _as_bool(adaptive_scene_stable_low_risk_latched, False)
    scene_high_uncertainty_alpha_mean = float(
        adaptive_scene_high_uncertainty_bypass_risk_alpha_mean
    )
    scene_high_uncertainty_novelty_mean = float(
        adaptive_scene_high_uncertainty_bypass_novelty_mean
    )
    adaptive_scene_calibrated = bool(adaptive_mode and scene_events >= scene_min_events)
    adaptive_scene_high_uncertainty_bypass = bool(
        adaptive_scene_calibrated
        and scene_alpha_mean >= scene_high_uncertainty_alpha_mean
        and scene_novelty_mean >= scene_high_uncertainty_novelty_mean
    )
    adaptive_scene_low_risk_enabled = bool(
        scene_low_pose_mean >= 0.0 and scene_low_novelty_mean >= 0.0
    )
    adaptive_scene_stable_low_risk_bypass = bool(
        adaptive_scene_calibrated
        and adaptive_scene_low_risk_enabled
        and scene_events >= max(scene_min_events, scene_stable_low_risk_min_events)
        and scene_pose_mean <= scene_stable_low_pose_mean
        and scene_novelty_mean <= scene_stable_low_novelty_mean
    )
    adaptive_scene_mature_low_risk_bypass = bool(
        adaptive_scene_calibrated
        and adaptive_scene_low_risk_enabled
        and scene_events >= max(scene_min_events, scene_low_risk_min_events)
        and scene_pose_mean <= scene_low_pose_mean
        and scene_novelty_mean <= scene_low_novelty_mean
    )
    adaptive_scene_latched_low_risk_bypass = bool(
        adaptive_scene_calibrated
        and adaptive_scene_low_risk_enabled
        and scene_stable_low_latched
    )
    adaptive_scene_low_risk_bypass = bool(
        adaptive_scene_stable_low_risk_bypass
        or adaptive_scene_mature_low_risk_bypass
        or adaptive_scene_latched_low_risk_bypass
    )
    adaptive_scene_gentle = bool(
        adaptive_scene_calibrated
        and not adaptive_scene_high_uncertainty_bypass
        and not adaptive_scene_low_risk_bypass
        and scene_pose_mean >= scene_gentle_pose_mean
        and scene_novelty_mean >= scene_gentle_novelty_mean
        and scene_alpha_mean <= scene_max_alpha_mean
    )
    min_risk = (
        float(adaptive_min_pose_risk)
        if adaptive_scene_gentle
        else float(min_pose_risk)
    )
    raw_max_risk = (
        float(adaptive_max_pose_risk)
        if adaptive_scene_gentle
        else float(max_pose_risk)
    )
    max_risk = max(raw_max_risk, min_risk + 1e-6)
    risk_for_weight = max_risk if pose_risk_high else pose_risk
    risk_alpha = min(1.0, max(0.0, (risk_for_weight - min_risk) / (max_risk - min_risk)))
    min_sample_scale = (
        float(adaptive_min_sample_opacity_scale)
        if adaptive_scene_gentle
        else float(min_sample_opacity_scale)
    )
    min_match_scale = (
        float(adaptive_min_match_opacity_scale)
        if adaptive_scene_gentle
        else float(min_match_opacity_scale)
    )
    sample_scale = 1.0 - risk_alpha * (1.0 - min_sample_scale)
    match_scale = 1.0 - risk_alpha * (1.0 - min_match_scale)
    representation_value = _max_available(
        gate,
        (
            "stream_memory_representation_need",
            "utility_coverage_gain",
        ),
    )
    novelty_value = _max_available(
        gate,
        (
            "high_novelty_score",
            "new_view_event_score",
        ),
    )
    max_repr_value = float(max_representation_value)
    max_novel_value = float(max_novelty_value)
    adaptive_novelty_limit = float(adaptive_novelty_threshold)
    adaptive_override_alpha = max(0.0, min(1.0, float(adaptive_risk_override_alpha)))
    adaptive_risk_override = bool(
        adaptive_scene_gentle and (pose_risk_high or risk_alpha >= adaptive_override_alpha)
    )
    novelty_limit = adaptive_novelty_limit if adaptive_scene_gentle else max_novel_value
    debug: dict[str, Any] = {
        "mode": mode_name,
        "direct_density_mode": direct_mode,
        "applied": False,
        "reason": "",
        "is_test": _as_bool(info.get("is_test", False), False),
        "pose_risk_score": pose_risk,
        "pose_risk_high": pose_risk_high,
        "pose_render_risk_reason": str(gate.get("pose_render_risk_reason", "")),
        "min_pose_risk": min_risk,
        "max_pose_risk": max_risk,
        "risk_alpha": risk_alpha,
        "min_sample_opacity_scale": min_sample_scale,
        "min_match_opacity_scale": min_match_scale,
        "sample_opacity_scale": sample_scale,
        "match_opacity_scale": match_scale,
        "representation_value_score": representation_value,
        "novelty_value_score": novelty_value,
        "raw_representation_value_score": _as_float(
            gate.get("representation_value_score", 0.0), 0.0
        ),
        "raw_novelty_value_score": _as_float(gate.get("novelty_value_score", 0.0), 0.0),
        "max_representation_value": max_repr_value,
        "max_novelty_value": max_novel_value,
        "adaptive_novelty_threshold": adaptive_novelty_limit,
        "adaptive_risk_override_alpha": adaptive_override_alpha,
        "adaptive_risk_override": adaptive_risk_override,
        "effective_novelty_value_limit": novelty_limit,
        "adaptive_scene_gentle": adaptive_scene_gentle,
        "adaptive_scene_high_uncertainty_bypass": adaptive_scene_high_uncertainty_bypass,
        "adaptive_scene_low_risk_bypass": adaptive_scene_low_risk_bypass,
        "adaptive_scene_event_count": scene_events,
        "adaptive_scene_pose_risk_mean": scene_pose_mean,
        "adaptive_scene_novelty_mean": scene_novelty_mean,
        "adaptive_scene_risk_alpha_mean": scene_alpha_mean,
        "adaptive_scene_min_events": scene_min_events,
        "adaptive_scene_gentle_pose_risk_mean": scene_gentle_pose_mean,
        "adaptive_scene_gentle_novelty_mean": scene_gentle_novelty_mean,
        "adaptive_scene_max_risk_alpha_mean": scene_max_alpha_mean,
        "adaptive_scene_low_pose_risk_mean": scene_low_pose_mean,
        "adaptive_scene_low_novelty_mean": scene_low_novelty_mean,
        "adaptive_scene_low_risk_min_events": scene_low_risk_min_events,
        "adaptive_scene_stable_low_risk_bypass": (
            adaptive_scene_stable_low_risk_bypass
        ),
        "adaptive_scene_mature_low_risk_bypass": (
            adaptive_scene_mature_low_risk_bypass
        ),
        "adaptive_scene_latched_low_risk_bypass": (
            adaptive_scene_latched_low_risk_bypass
        ),
        "adaptive_scene_stable_low_risk_latched": scene_stable_low_latched,
        "adaptive_scene_stable_low_risk_min_events": (
            scene_stable_low_risk_min_events
        ),
        "adaptive_scene_stable_low_pose_risk_mean": scene_stable_low_pose_mean,
        "adaptive_scene_stable_low_novelty_mean": scene_stable_low_novelty_mean,
        "adaptive_scene_high_uncertainty_bypass_risk_alpha_mean": (
            scene_high_uncertainty_alpha_mean
        ),
        "adaptive_scene_high_uncertainty_bypass_novelty_mean": (
            scene_high_uncertainty_novelty_mean
        ),
        "adaptive_min_pose_risk": float(adaptive_min_pose_risk),
        "adaptive_max_pose_risk": float(adaptive_max_pose_risk),
        "adaptive_min_sample_opacity_scale": float(adaptive_min_sample_opacity_scale),
        "adaptive_min_match_opacity_scale": float(adaptive_min_match_opacity_scale),
    }
    if mode_name == "off":
        debug["reason"] = "off"
    elif mode_name not in INIT_WEIGHTING_MODES:
        debug["reason"] = "unknown_mode"
    elif direct_mode != POSE_SAFE_DIRECT_DENSITY_MODE:
        debug["reason"] = "direct_density_mismatch"
    elif debug["is_test"]:
        debug["reason"] = "test_frame"
    elif adaptive_scene_high_uncertainty_bypass:
        debug["reason"] = "adaptive_scene_high_uncertainty_bypass"
    elif adaptive_scene_low_risk_bypass:
        debug["reason"] = "adaptive_scene_low_risk_bypass"
    elif risk_alpha <= 0.0:
        debug["reason"] = "pose_risk_low"
    elif representation_value > max_repr_value:
        debug["reason"] = "representation_value_high"
    elif novelty_value > novelty_limit and not adaptive_risk_override:
        debug["reason"] = "adaptive_novelty_preserve" if adaptive_mode else "novelty_value_high"
    else:
        debug["applied"] = True
        debug["reason"] = (
            "risk_opacity_adaptive_scale" if adaptive_mode else "risk_opacity_scale"
        )
    return debug
