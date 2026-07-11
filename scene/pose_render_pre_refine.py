from __future__ import annotations

from typing import Any


POSE_SAFE_DIRECT_DENSITY_MODE = "pose_safe_streaming_memory_v1"
PRE_REFINE_MODE = "pose_only_v1"
PRE_REFINE_MODES = {"off", PRE_REFINE_MODE}
POSE_RENDER_GATE_INFO_KEY = "_paper_aligned_pose_render_coupling"


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


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
        "direct_keyframe_finalized",
        "pose_risk_score",
        "pose_risk_high",
        "pose_render_risk_score",
        "pose_render_risk_high",
        "pose_render_risk_reason",
        "pose_render_pose_confidence",
        "pose_render_posterior_risk_score",
        "pose_render_support_ratio",
        "pose_render_inlier_ratio",
        "pose_render_grid_coverage",
        "pose_render_grid_entropy",
        "pose_render_support_concentration",
        "semantic_R_t",
    ):
        if key in info:
            gate[key] = info[key]
    return gate


def pose_render_pre_refine_decision(
    *,
    mode: str | None,
    direct_density_mode: str | None,
    keyframe_info: dict[str, Any] | None,
    existing_gaussians: int,
    existing_keyframes: int,
    iterations: int = 2,
    min_pose_risk: float = 0.08,
    max_pose_risk: float = 0.38,
    min_existing_gaussians: int = 5000,
    min_existing_keyframes: int = 8,
) -> dict[str, Any]:
    mode_name = str(mode or "off")
    direct_mode = str(direct_density_mode or "off")
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
    debug: dict[str, Any] = {
        "mode": mode_name,
        "direct_density_mode": direct_mode,
        "run_pre_refine": False,
        "reason": "",
        "iterations": max(0, int(iterations or 0)),
        "is_test": _as_bool(info.get("is_test", False), False),
        "direct_keyframe_finalized": _as_bool(
            gate.get("direct_keyframe_finalized", True), True
        ),
        "pose_risk_score": pose_risk,
        "pose_risk_high": pose_risk_high,
        "pose_render_risk_reason": str(gate.get("pose_render_risk_reason", "")),
        "pose_render_pose_confidence": _as_float(
            gate.get("pose_render_pose_confidence", 1.0), 1.0
        ),
        "min_pose_risk": float(min_pose_risk),
        "max_pose_risk": float(max_pose_risk),
        "existing_gaussians": _as_int(existing_gaussians, 0),
        "existing_keyframes": _as_int(existing_keyframes, 0),
        "min_existing_gaussians": _as_int(min_existing_gaussians, 0),
        "min_existing_keyframes": _as_int(min_existing_keyframes, 0),
    }
    if mode_name == "off":
        debug["reason"] = "off"
    elif mode_name not in PRE_REFINE_MODES:
        debug["reason"] = "unknown_mode"
    elif direct_mode != POSE_SAFE_DIRECT_DENSITY_MODE:
        debug["reason"] = "direct_density_mismatch"
    elif debug["is_test"]:
        debug["reason"] = "test_frame"
    elif debug["iterations"] <= 0:
        debug["reason"] = "zero_iterations"
    elif not debug["direct_keyframe_finalized"]:
        debug["reason"] = "direct_keyframe_not_finalized"
    elif debug["existing_keyframes"] < debug["min_existing_keyframes"]:
        debug["reason"] = "map_not_mature_keyframes"
    elif debug["existing_gaussians"] < debug["min_existing_gaussians"]:
        debug["reason"] = "map_not_mature_gaussians"
    elif pose_risk_high or pose_risk > float(max_pose_risk):
        debug["reason"] = "pose_risk_too_high"
    elif pose_risk < float(min_pose_risk):
        debug["reason"] = "pose_risk_low"
    else:
        debug["run_pre_refine"] = True
        debug["reason"] = "pose_only_pre_refine"
    return debug
