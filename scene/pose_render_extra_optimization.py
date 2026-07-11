from __future__ import annotations

from typing import Any

from scene.pose_render_edge_loss import pose_render_pose_gate_debug


POSE_SAFE_DIRECT_DENSITY_MODE = "pose_safe_streaming_memory_v1"
EXTRA_OPTIMIZATION_MODE = "pose_confidence_v1"
EXTRA_OPTIMIZATION_RENDER_RESPONSE_MODE = "pose_confidence_render_response_v2"
EXTRA_OPTIMIZATION_RENDER_ONLY_RESPONSE_MODE = "render_response_v3"
EXTRA_OPTIMIZATION_RENDER_ONLY_RESPONSE_SOFT_MODE = "render_response_v4"
EXTRA_OPTIMIZATION_RENDER_ONLY_RESPONSE_RESCUE_MODE = "render_response_v5"
EXTRA_OPTIMIZATION_RENDER_ONLY_RESPONSE_MASK_CONSERVATIVE_MODE = (
    "render_response_mask_conservative_v6"
)
EXTRA_OPTIMIZATION_RENDER_ONLY_RESPONSE_MODES = {
    EXTRA_OPTIMIZATION_RENDER_ONLY_RESPONSE_MODE,
    EXTRA_OPTIMIZATION_RENDER_ONLY_RESPONSE_SOFT_MODE,
    EXTRA_OPTIMIZATION_RENDER_ONLY_RESPONSE_RESCUE_MODE,
    EXTRA_OPTIMIZATION_RENDER_ONLY_RESPONSE_MASK_CONSERVATIVE_MODE,
}
EXTRA_OPTIMIZATION_MODES = {
    "off",
    EXTRA_OPTIMIZATION_MODE,
    EXTRA_OPTIMIZATION_RENDER_RESPONSE_MODE,
    EXTRA_OPTIMIZATION_RENDER_ONLY_RESPONSE_MODE,
    EXTRA_OPTIMIZATION_RENDER_ONLY_RESPONSE_SOFT_MODE,
    EXTRA_OPTIMIZATION_RENDER_ONLY_RESPONSE_RESCUE_MODE,
    EXTRA_OPTIMIZATION_RENDER_ONLY_RESPONSE_MASK_CONSERVATIVE_MODE,
}
POSE_RENDER_GATE_INFO_KEY = "_paper_aligned_pose_render_coupling"
POSE_RENDER_RESPONSE_INFO_KEY = "_paper_aligned_pose_render_response"
POSE_RENDER_TEXTURE_COVERAGE_INFO_KEY = (
    "_paper_aligned_pose_render_texture_sampling_coverage"
)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _pose_render_gate_info(keyframe_info: dict[str, Any] | None) -> dict[str, Any]:
    info = keyframe_info or {}
    gate: dict[str, Any] = {}
    nested = info.get(POSE_RENDER_GATE_INFO_KEY, None)
    if isinstance(nested, dict):
        gate.update(nested)
    for key in (
        "pose_render_pose_confidence",
        "pose_render_risk_score",
        "pose_risk_score",
        "semantic_R_t",
    ):
        if key in info:
            gate[key] = info[key]
    return gate


def _pose_render_response_info(keyframe_info: dict[str, Any] | None) -> dict[str, Any]:
    info = keyframe_info or {}
    response = info.get(POSE_RENDER_RESPONSE_INFO_KEY, None)
    return response if isinstance(response, dict) else {}


def pose_render_extra_optimization_decision(
    *,
    mode: str | None,
    direct_density_mode: str | None,
    keyframe_info: dict[str, Any] | None,
    base_iterations: int,
    render_frame_policy: str | None = None,
    min_confidence: float = 0.75,
    fraction: float = 0.25,
    max_extra: int = 8,
    max_pose_risk: float = 0.35,
    max_utility_drift: float = 0.60,
    min_pose_support: float = 0.55,
    min_match_support: float = 0.55,
) -> dict[str, Any]:
    mode_name = str(mode or "off")
    debug: dict[str, Any] = {
        "mode": mode_name,
        "direct_density_mode": str(direct_density_mode or "off"),
        "render_frame_policy": str(render_frame_policy or "off"),
        "applied": False,
        "reason": "",
        "base_iterations": int(base_iterations or 0),
        "extra_iterations": 0,
        "min_confidence": float(min_confidence),
        "fraction": float(fraction),
        "max_extra": int(max_extra or 0),
        "max_pose_risk": float(max_pose_risk),
        "max_utility_drift": float(max_utility_drift),
        "min_pose_support": float(min_pose_support),
        "min_match_support": float(min_match_support),
        "confidence": 0.0,
        "render_response_observations": 0,
        "render_response_improvement": 0.0,
        "render_response_scale": 0.0,
        "coverage_deficit": 0.0,
        "min_coverage_deficit": 0.08,
        "min_scene_coverage_deficit": 0.10,
        "min_sampling_applied_ratio": 0.15,
        "min_scene_pressure_events": 20,
        "min_scene_response_evaluated": 32,
        "max_scene_response_bad_ratio": 0.0,
        "render_response_micro_max_extra": 2,
    }
    if mode_name == "off":
        debug["reason"] = "off"
        return debug
    if mode_name not in EXTRA_OPTIMIZATION_MODES:
        debug["reason"] = "unknown_mode"
        return debug
    signal_enabled = (
        str(direct_density_mode or "off") == POSE_SAFE_DIRECT_DENSITY_MODE
        or str(render_frame_policy or "off") == "baseline_keyframe_lock_v1"
        or mode_name in EXTRA_OPTIMIZATION_RENDER_ONLY_RESPONSE_MODES
    )
    if not signal_enabled:
        debug["reason"] = "direct_density_mismatch"
        return debug
    info = keyframe_info or {}
    if bool(info.get("is_test", False)):
        debug["reason"] = "test_frame"
        return debug
    if int(base_iterations or 0) <= 0:
        debug["reason"] = "zero_base_iterations"
        return debug
    if int(max_extra or 0) <= 0 or float(fraction or 0.0) <= 0.0:
        debug["reason"] = "zero_extra_budget"
        return debug

    if mode_name in EXTRA_OPTIMIZATION_RENDER_ONLY_RESPONSE_MODES:
        if mode_name == EXTRA_OPTIMIZATION_RENDER_ONLY_RESPONSE_SOFT_MODE:
            debug["min_coverage_deficit"] = 0.075
            debug["min_scene_coverage_deficit"] = 0.075
            debug["max_scene_response_bad_ratio"] = 0.06
        elif mode_name in {
            EXTRA_OPTIMIZATION_RENDER_ONLY_RESPONSE_RESCUE_MODE,
            EXTRA_OPTIMIZATION_RENDER_ONLY_RESPONSE_MASK_CONSERVATIVE_MODE,
        }:
            debug["min_coverage_deficit"] = 0.065
            debug["min_scene_coverage_deficit"] = 0.065
            debug["max_scene_response_bad_ratio"] = 0.10
        training_background = info.get("_paper_aligned_training_background", None)
        if (
            mode_name == EXTRA_OPTIMIZATION_RENDER_ONLY_RESPONSE_MASK_CONSERVATIVE_MODE
            and isinstance(training_background, dict)
        ):
            mask_blocked = int(
                _as_float(training_background.get("dark_scene_mask_blocked", 0), 0.0)
            )
            dark_observations = int(
                _as_float(training_background.get("dark_scene_observations", 0), 0.0)
            )
            debug["training_background_mode"] = str(training_background.get("mode", ""))
            debug["training_background_dark_scene_mask_blocked"] = mask_blocked
            debug["training_background_dark_scene_observations"] = dark_observations
            if (
                str(training_background.get("mode", "")) == "dark_scene_fixed_black_v1"
                and mask_blocked > 0
                and dark_observations == 0
            ):
                debug["reason"] = "mask_blocked_scene_bypass"
                return debug
        coverage = info.get(POSE_RENDER_TEXTURE_COVERAGE_INFO_KEY, None)
        coverage_deficit = 0.0
        if isinstance(coverage, dict):
            coverage_deficit = _as_float(
                coverage.get("coverage_deficit", coverage.get("deficit", 0.0)),
                0.0,
            )
        debug["coverage_deficit"] = coverage_deficit
        min_coverage_deficit = float(debug["min_coverage_deficit"])
        if coverage_deficit < min_coverage_deficit:
            debug["reason"] = "coverage_sufficient"
            return debug
        scene_guard = (
            info.get(
                "_paper_aligned_pose_render_texture_sampling_scene_guard",
                None,
            )
            if mode_name != EXTRA_OPTIMIZATION_RENDER_ONLY_RESPONSE_MODE
            else None
        )
        if isinstance(scene_guard, dict):
            scene_events = int(_as_float(scene_guard.get("events", 0), 0.0))
            scene_applied = int(_as_float(scene_guard.get("applied", 0), 0.0))
            scene_coverage_deficit_mean = _as_float(
                scene_guard.get("coverage_deficit_mean", coverage_deficit),
                coverage_deficit,
            )
            sampling_applied_ratio = _as_float(
                scene_guard.get("sampling_applied_ratio", 0.0),
                0.0,
            )
            response_evaluated = int(
                _as_float(scene_guard.get("response_evaluated", 0), 0.0)
            )
            response_bad_ratio = _as_float(
                scene_guard.get("response_bad_ratio", 0.0),
                0.0,
            )
            debug["scene_events"] = scene_events
            debug["scene_applied"] = scene_applied
            debug["scene_coverage_deficit_mean"] = scene_coverage_deficit_mean
            debug["sampling_applied_ratio"] = sampling_applied_ratio
            debug["scene_response_evaluated"] = response_evaluated
            debug["scene_response_bad_ratio"] = response_bad_ratio
            if (
                scene_coverage_deficit_mean < min_coverage_deficit
                and sampling_applied_ratio < 0.50
            ):
                debug["reason"] = "scene_coverage_sufficient"
                return debug
            min_scene_pressure_events = int(debug["min_scene_pressure_events"])
            if scene_events < min_scene_pressure_events:
                debug["reason"] = "scene_pressure_warming_up"
                return debug
            min_scene_coverage_deficit = float(debug["min_scene_coverage_deficit"])
            min_sampling_applied_ratio = float(debug["min_sampling_applied_ratio"])
            if (
                scene_coverage_deficit_mean < min_scene_coverage_deficit
                or sampling_applied_ratio < min_sampling_applied_ratio
            ):
                debug["reason"] = "scene_pressure_low"
                return debug
            min_scene_response_evaluated = int(debug["min_scene_response_evaluated"])
            max_scene_response_bad_ratio = float(debug["max_scene_response_bad_ratio"])
            if (
                response_evaluated >= min_scene_response_evaluated
                and response_bad_ratio <= max_scene_response_bad_ratio
            ):
                pass
            elif response_evaluated >= 3 and response_bad_ratio > max_scene_response_bad_ratio:
                debug["reason"] = "scene_response_unstable"
                return debug
            else:
                debug["reason"] = "scene_response_warming_up"
                return debug

        response = _pose_render_response_info(keyframe_info)
        observations = int(_as_float(response.get("observations", 0), 0.0))
        improvement = _as_float(response.get("relative_improvement", 0.0), 0.0)
        first_rgb_mse = _as_float(response.get("first_rgb_mse", 0.0), 0.0)
        latest_rgb_mse = _as_float(response.get("latest_rgb_mse", 0.0), 0.0)
        debug.update(
            {
                "render_response_observations": observations,
                "render_response_improvement": improvement,
                "render_response_first_rgb_mse": first_rgb_mse,
                "render_response_latest_rgb_mse": latest_rgb_mse,
                "confidence": 1.0,
            }
        )
        if observations < 2:
            debug["reason"] = "render_response_pending"
            return debug
        if first_rgb_mse > 0.0 and latest_rgb_mse > first_rgb_mse * 1.02:
            debug["reason"] = "render_response_degraded"
            return debug
        if improvement < 0.03:
            debug["reason"] = "render_response_low"
            return debug
        response_scale = max(0.35, min(improvement / 0.12, 1.0))
        if mode_name != EXTRA_OPTIMIZATION_RENDER_ONLY_RESPONSE_MODE:
            render_response_micro_max_extra = int(debug["render_response_micro_max_extra"])
            max_extra = min(int(max_extra), render_response_micro_max_extra)
        extra = round(int(base_iterations) * float(fraction) * response_scale)
        extra = max(1, min(int(max_extra), int(extra)))
        debug.update(
            {
                "applied": True,
                "reason": "render_response_extra_optimization",
                "extra_iterations": extra,
                "render_response_scale": response_scale,
            }
        )
        return debug

    gate_debug = pose_render_pose_gate_debug(
        keyframe_info,
        max_pose_risk=max_pose_risk,
        max_utility_drift=max_utility_drift,
        min_pose_support=min_pose_support,
        min_match_support=min_match_support,
    )
    debug.update(gate_debug)
    if not bool(gate_debug.get("pose_gate_passed", False)):
        debug["reason"] = str(gate_debug.get("pose_gate_reason") or "pose_gate_rejected")
        return debug

    gate = _pose_render_gate_info(keyframe_info)
    pose_risk = _as_float(
        gate.get(
            "pose_render_risk_score",
            gate.get("pose_risk_score", gate.get("semantic_R_t", 1.0)),
        ),
        1.0,
    )
    confidence = _as_float(
        gate.get("pose_render_pose_confidence", max(0.0, 1.0 - pose_risk)),
        max(0.0, 1.0 - pose_risk),
    )
    confidence = max(0.0, min(1.0, confidence))
    debug["confidence"] = confidence
    if confidence < float(min_confidence):
        debug["reason"] = "confidence_low"
        return debug

    response_scale = 1.0
    if mode_name == EXTRA_OPTIMIZATION_RENDER_RESPONSE_MODE:
        response = _pose_render_response_info(keyframe_info)
        observations = int(_as_float(response.get("observations", 0), 0.0))
        improvement = _as_float(response.get("relative_improvement", 0.0), 0.0)
        first_rgb_mse = _as_float(response.get("first_rgb_mse", 0.0), 0.0)
        latest_rgb_mse = _as_float(response.get("latest_rgb_mse", 0.0), 0.0)
        debug.update(
            {
                "render_response_observations": observations,
                "render_response_improvement": improvement,
                "render_response_first_rgb_mse": first_rgb_mse,
                "render_response_latest_rgb_mse": latest_rgb_mse,
            }
        )
        if observations < 2:
            debug["reason"] = "render_response_pending"
            return debug
        if first_rgb_mse > 0.0 and latest_rgb_mse > first_rgb_mse * 1.02:
            debug["reason"] = "render_response_degraded"
            return debug
        if improvement < 0.03:
            debug["reason"] = "render_response_low"
            return debug
        response_scale = max(0.35, min(improvement / 0.12, 1.0))

    extra = round(int(base_iterations) * float(fraction) * confidence * response_scale)
    extra = max(1, min(int(max_extra), int(extra)))
    debug.update(
        {
            "applied": True,
            "reason": (
                "render_response_extra_optimization"
                if mode_name == EXTRA_OPTIMIZATION_RENDER_RESPONSE_MODE
                else "pose_confident_extra_optimization"
            ),
            "extra_iterations": extra,
            "render_response_scale": response_scale,
        }
    )
    return debug
