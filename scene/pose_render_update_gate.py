from __future__ import annotations

from typing import Any

from scene.pose_render_edge_loss import (
    POSE_RENDER_GATE_INFO_KEY,
    POSE_SAFE_DIRECT_DENSITY_MODE,
    pose_render_training_signal_enabled,
)


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


def _gate_payload(keyframe_info: dict[str, Any] | None) -> dict[str, Any]:
    info = keyframe_info or {}
    payload = info.get(POSE_RENDER_GATE_INFO_KEY, None)
    if isinstance(payload, dict):
        return payload
    return {}


def _psnr_loss_payload(keyframe_info: dict[str, Any] | None) -> dict[str, Any]:
    info = keyframe_info or {}
    payload = info.get("_paper_aligned_pose_render_psnr_loss", None)
    if isinstance(payload, dict):
        return payload
    return {}


def pose_render_update_gate_decision(
    *,
    mode: str | None,
    direct_density_mode: str | None,
    keyframe_info: dict[str, Any] | None,
    render_frame_policy: str | None = None,
    min_confidence: float = 0.35,
    max_pose_risk: float = 0.55,
    max_utility_drift: float = 0.75,
    soft_min_scale: float = 0.65,
    psnr_health_min_score: float = 0.0,
    psnr_health_min_raw_loss: float = 0.0,
    psnr_health_scale: float = 1.0,
) -> dict[str, Any]:
    mode_name = str(mode or "off")
    psnr_health_mode = mode_name == "pose_confidence_soft_psnr_health_v1"
    soft_mode = mode_name == "pose_confidence_soft_v1" or psnr_health_mode
    info = keyframe_info or {}
    debug: dict[str, Any] = {
        "mode": mode_name,
        "direct_density_mode": str(direct_density_mode or "off"),
        "render_frame_policy": str(render_frame_policy or "off"),
        "allow_gaussian_update": True,
        "gaussian_update_scale": 1.0,
        "reason": "off",
        "confidence": 1.0,
        "min_confidence": float(min_confidence),
        "max_pose_risk": float(max_pose_risk),
        "max_utility_drift": float(max_utility_drift),
        "soft_min_scale": float(soft_min_scale),
        "psnr_health_min_score": float(psnr_health_min_score),
        "psnr_health_min_raw_loss": float(psnr_health_min_raw_loss),
        "psnr_health_scale": float(psnr_health_scale),
        "psnr_health_score": 0.0,
        "psnr_health_raw_loss": 0.0,
        "psnr_health_alias_scaled": False,
    }
    if mode_name == "off":
        return debug
    if mode_name not in {
        "pose_confidence_v1",
        "pose_confidence_soft_v1",
        "pose_confidence_soft_psnr_health_v1",
    }:
        debug.update({"allow_gaussian_update": True, "reason": "unknown_mode"})
        return debug
    signal_enabled = pose_render_training_signal_enabled(
        direct_density_mode, render_frame_policy
    )
    debug["render_training_signal_enabled"] = bool(signal_enabled)
    if not signal_enabled:
        debug.update({"allow_gaussian_update": True, "reason": "direct_density_mismatch"})
        return debug
    if bool(info.get("is_test", False)):
        debug.update({"allow_gaussian_update": False, "reason": "test_frame"})
        return debug

    payload = _gate_payload(info)
    if not payload:
        debug.update({"allow_gaussian_update": True, "reason": "missing_gate_allow"})
        return debug

    pose_risk = _as_float(
        payload.get(
            "pose_render_risk_score",
            payload.get("pose_risk_score", payload.get("semantic_R_t")),
        ),
        0.0,
    )
    utility_drift = _as_float(payload.get("utility_drift_risk"), 0.0)
    pose_support = _as_float(payload.get("pose_support_score"), 1.0)
    match_support = _as_float(payload.get("match_support_score"), 1.0)
    pose_risk_high = _as_bool(
        payload.get("pose_render_risk_high", payload.get("pose_risk_high")),
        False,
    )
    pose_risk_reference = _as_bool(payload.get("finalize_pose_risk_reference"), False)

    pose_confidence = 1.0 - min(max(pose_risk, 0.0) / max(float(max_pose_risk), 1e-6), 1.0)
    drift_confidence = 1.0 - min(
        max(utility_drift, 0.0) / max(float(max_utility_drift), 1e-6), 1.0
    )
    support_confidence = 0.5 * (
        min(max(pose_support, 0.0), 1.0) + min(max(match_support, 0.0), 1.0)
    )
    confidence = (
        0.45 * pose_confidence
        + 0.35 * support_confidence
        + 0.20 * drift_confidence
    )
    pose_risk_score_high = bool(pose_risk > float(max_pose_risk))
    if pose_risk_high or pose_risk_reference or pose_risk_score_high:
        confidence = min(confidence, 0.25)
    confidence = min(max(float(confidence), 0.0), 1.0)
    allow = bool(confidence >= float(min_confidence))
    reason = "pose_confidence_ok" if allow else "pose_confidence_low"
    if pose_risk_high:
        reason = "pose_risk_high"
    elif pose_risk_reference:
        reason = "pose_risk_reference"
    elif pose_risk_score_high:
        reason = "pose_risk_score_high"
    update_scale = 1.0
    if soft_mode:
        min_scale = min(max(float(soft_min_scale), 0.0), 1.0)
        normalized_confidence = confidence / max(float(min_confidence), 1e-6)
        update_scale = min_scale + (1.0 - min_scale) * min(
            max(normalized_confidence, 0.0), 1.0
        )
        allow = True
        if update_scale < 0.999:
            reason = "pose_confidence_soft_scaled"
        else:
            reason = "pose_confidence_ok"

    if psnr_health_mode:
        psnr_debug = _psnr_loss_payload(info)
        psnr_raw_loss = _as_float(psnr_debug.get("raw_loss"), 0.0)
        psnr_health_score = _as_float(
            psnr_debug.get("health_gate_score", psnr_debug.get("structure_score")),
            0.0,
        )
        psnr_scale = min(max(float(psnr_health_scale), 0.0), 1.0)
        debug.update(
            {
                "psnr_health_score": psnr_health_score,
                "psnr_health_raw_loss": psnr_raw_loss,
            }
        )
        if (
            bool(psnr_debug.get("applied", False))
            and psnr_raw_loss >= float(psnr_health_min_raw_loss)
            and psnr_health_score >= float(psnr_health_min_score)
            and psnr_scale < update_scale
        ):
            update_scale = psnr_scale
            allow = True
            reason = "psnr_health_alias_scaled"
            debug["psnr_health_alias_scaled"] = True

    debug.update(
        {
            "allow_gaussian_update": allow,
            "gaussian_update_scale": update_scale,
            "reason": reason,
            "confidence": confidence,
            "pose_risk_score": pose_risk,
            "utility_drift_risk": utility_drift,
            "pose_support_score": pose_support,
            "match_support_score": match_support,
            "pose_confidence": pose_confidence,
            "drift_confidence": drift_confidence,
            "support_confidence": support_confidence,
            "pose_risk_high": pose_risk_high,
            "pose_risk_score_high": pose_risk_score_high,
            "finalize_pose_risk_reference": pose_risk_reference,
        }
    )
    return debug
