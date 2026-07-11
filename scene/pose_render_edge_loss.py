from __future__ import annotations

from typing import Any


POSE_SAFE_DIRECT_DENSITY_MODE = "pose_safe_streaming_memory_v1"
BASELINE_RENDER_LOCK_POLICY = "baseline_keyframe_lock_v1"
POSE_SAFE_EDGE_LOSS_MODE = "gradient_pose_safe_v1"
POSE_ADAPTIVE_EDGE_LOSS_MODE = "gradient_pose_adaptive_v1"
EDGE_LOSS_MODES = {"gradient_v1", POSE_SAFE_EDGE_LOSS_MODE, POSE_ADAPTIVE_EDGE_LOSS_MODE}
POSE_RENDER_GATE_INFO_KEY = "_paper_aligned_pose_render_coupling"


def pose_render_training_signal_enabled(
    direct_density_mode: str | None,
    render_frame_policy: str | None = None,
) -> bool:
    return bool(
        str(direct_density_mode or "off") == POSE_SAFE_DIRECT_DENSITY_MODE
        or str(render_frame_policy or "off") == BASELINE_RENDER_LOCK_POLICY
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


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))


def _detached_float(value: Any, default: float = 0.0) -> float:
    try:
        if hasattr(value, "detach"):
            return float(value.detach().cpu().item())
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
        "direct_keyframe_finalized",
        "direct_finalization_decision",
        "finalize_pose_risk_reference",
        "pose_risk_high",
        "pose_risk_score",
        "pose_render_risk_high",
        "pose_render_risk_score",
        "pose_render_risk_reason",
        "pose_render_posterior_risk_score",
        "pose_render_pose_confidence",
        "semantic_R_t",
        "utility_drift_risk",
        "pose_support_score",
        "match_support_score",
        "active_memory_stable_pose_reference",
    ):
        if key in info:
            gate[key] = info[key]
    return gate


def pose_render_pose_gate_debug(
    keyframe_info: dict[str, Any] | None,
    *,
    max_pose_risk: float = 0.50,
    max_utility_drift: float = 0.65,
    min_pose_support: float = 0.45,
    min_match_support: float = 0.45,
) -> dict[str, Any]:
    gate = _pose_render_gate_info(keyframe_info)
    debug: dict[str, Any] = {
        "pose_gate_mode": "pose_safe_v1",
        "pose_gate_passed": False,
        "pose_gate_reason": "",
        "max_pose_risk": float(max_pose_risk),
        "max_utility_drift": float(max_utility_drift),
        "min_pose_support": float(min_pose_support),
        "min_match_support": float(min_match_support),
    }
    if not gate:
        debug["pose_gate_reason"] = "missing_pose_render_gate"
        return debug

    pose_risk_score = _as_float(
        gate.get(
            "pose_render_risk_score",
            gate.get("pose_risk_score", gate.get("semantic_R_t", None)),
        ),
        1.0,
    )
    utility_drift_risk = _as_float(gate.get("utility_drift_risk", None), 1.0)
    pose_support_score = _as_float(gate.get("pose_support_score", None), 0.0)
    match_support_score = _as_float(gate.get("match_support_score", None), 0.0)
    stable_pose_reference = _as_bool(
        gate.get("active_memory_stable_pose_reference", False), False
    )
    support_ok = bool(
        stable_pose_reference
        or (
            pose_support_score >= float(min_pose_support)
            and match_support_score >= float(min_match_support)
        )
    )
    debug.update(
        {
            "direct_keyframe_finalized": _as_bool(
                gate.get("direct_keyframe_finalized", True), True
            ),
            "direct_finalization_decision": str(
                gate.get("direct_finalization_decision", "")
            ),
            "pose_risk_score": pose_risk_score,
            "pose_risk_high": _as_bool(
                gate.get("pose_render_risk_high", gate.get("pose_risk_high", False)),
                False,
            ),
            "pose_render_risk_reason": str(gate.get("pose_render_risk_reason", "")),
            "finalize_pose_risk_reference": _as_bool(
                gate.get("finalize_pose_risk_reference", False), False
            ),
            "utility_drift_risk": utility_drift_risk,
            "pose_support_score": pose_support_score,
            "match_support_score": match_support_score,
            "active_memory_stable_pose_reference": stable_pose_reference,
            "pose_match_support_ok": support_ok,
        }
    )
    if not debug["direct_keyframe_finalized"]:
        debug["pose_gate_reason"] = "direct_keyframe_not_finalized"
    elif debug["pose_risk_high"]:
        debug["pose_gate_reason"] = "pose_risk_high"
    elif debug["finalize_pose_risk_reference"]:
        debug["pose_gate_reason"] = "pose_risk_reference"
    elif pose_risk_score > float(max_pose_risk):
        debug["pose_gate_reason"] = "pose_risk_score_high"
    elif utility_drift_risk > float(max_utility_drift):
        debug["pose_gate_reason"] = "utility_drift_risk_high"
    elif not support_ok:
        debug["pose_gate_reason"] = "low_pose_match_support"
    else:
        debug["pose_gate_passed"] = True
        debug["pose_gate_reason"] = "pose_safe"
    return debug


def pose_render_edge_loss_enabled(
    mode: str | None,
    direct_density_mode: str | None,
    keyframe_info: dict[str, Any] | None = None,
    weight: float = 0.0,
    max_pose_risk: float = 0.50,
    max_utility_drift: float = 0.65,
    min_pose_support: float = 0.45,
    min_match_support: float = 0.45,
) -> bool:
    mode_name = str(mode or "off")
    if mode_name not in EDGE_LOSS_MODES:
        return False
    if str(direct_density_mode or "off") != POSE_SAFE_DIRECT_DENSITY_MODE:
        return False
    if float(weight or 0.0) <= 0.0:
        return False
    info = keyframe_info or {}
    if bool(info.get("is_test", False)):
        return False
    if mode_name in {POSE_SAFE_EDGE_LOSS_MODE, POSE_ADAPTIVE_EDGE_LOSS_MODE}:
        gate_debug = pose_render_pose_gate_debug(
            keyframe_info,
            max_pose_risk=max_pose_risk,
            max_utility_drift=max_utility_drift,
            min_pose_support=min_pose_support,
            min_match_support=min_match_support,
        )
        if mode_name == POSE_SAFE_EDGE_LOSS_MODE:
            return bool(gate_debug.get("pose_gate_passed", False))
        return bool(
            gate_debug.get("pose_gate_reason") != "missing_pose_render_gate"
            and gate_debug.get("direct_keyframe_finalized", False)
            and not gate_debug.get("finalize_pose_risk_reference", False)
        )
    return True


def gradient_l1_edge_loss(image: Any, target: Any) -> Any:
    if tuple(image.shape) != tuple(target.shape):
        raise ValueError("image and target must have identical shapes")
    if image.ndim < 3 or image.shape[-2] < 2 or image.shape[-1] < 2:
        return image.new_zeros(())

    img_dx = image[..., :, 1:] - image[..., :, :-1]
    tgt_dx = target[..., :, 1:] - target[..., :, :-1]
    img_dy = image[..., 1:, :] - image[..., :-1, :]
    tgt_dy = target[..., 1:, :] - target[..., :-1, :]
    return 0.5 * ((img_dx - tgt_dx).abs().mean() + (img_dy - tgt_dy).abs().mean())


def pose_render_adaptive_edge_weight_debug(
    keyframe_info: dict[str, Any] | None,
    *,
    raw_loss: float,
    max_pose_risk: float = 0.50,
    max_utility_drift: float = 0.65,
    min_pose_support: float = 0.45,
    min_match_support: float = 0.45,
    min_weight_scale: float = 0.25,
    target_raw_loss: float = 0.02,
    risk_free_threshold: float = 0.25,
) -> dict[str, Any]:
    gate_debug = pose_render_pose_gate_debug(
        keyframe_info,
        max_pose_risk=max_pose_risk,
        max_utility_drift=max_utility_drift,
        min_pose_support=min_pose_support,
        min_match_support=min_match_support,
    )
    min_scale = _clamp(min_weight_scale, 0.0, 1.0)
    pose_risk_score = _as_float(gate_debug.get("pose_risk_score", 1.0), 1.0)
    if _as_bool(gate_debug.get("pose_risk_high", False), False):
        pose_risk_score = max(pose_risk_score, float(max_pose_risk))
    risk_free = _clamp(risk_free_threshold, 0.0, float(max_pose_risk))
    if pose_risk_score <= risk_free:
        pose_norm = 0.0
    else:
        pose_norm = _clamp(
            (pose_risk_score - risk_free)
            / max(float(max_pose_risk) - risk_free, 1e-6),
            0.0,
            1.0,
        )
    risk_scale = 1.0 - 0.50 * pose_norm

    utility_drift = _as_float(gate_debug.get("utility_drift_risk", 1.0), 1.0)
    drift_norm = _clamp(
        utility_drift / max(float(max_utility_drift), 1e-6), 0.0, 1.0
    )
    drift_scale = 1.0 - 0.25 * drift_norm

    if _as_bool(gate_debug.get("active_memory_stable_pose_reference", False), False):
        support_score = 1.0
    else:
        support_score = min(
            _as_float(gate_debug.get("pose_support_score", 0.0), 0.0),
            _as_float(gate_debug.get("match_support_score", 0.0), 0.0),
        )
    support_floor = min(float(min_pose_support), float(min_match_support))
    support_norm = _clamp(
        (support_score - support_floor) / max(1.0 - support_floor, 1e-6),
        0.0,
        1.0,
    )
    support_scale = 0.65 + 0.35 * support_norm

    raw_value = max(float(raw_loss), 0.0)
    target_value = max(float(target_raw_loss), 0.0)
    if target_value <= 0.0 or raw_value <= target_value:
        residual_clip_scale = 1.0
    else:
        residual_clip_scale = _clamp(target_value / max(raw_value, 1e-6), 0.0, 1.0)

    adaptive_scale = _clamp(
        risk_scale * drift_scale * support_scale * residual_clip_scale,
        min_scale,
        1.0,
    )
    gate_debug.update(
        {
            "adaptive_weight_scale": adaptive_scale,
            "min_weight_scale": min_scale,
            "target_raw_loss": target_value,
            "risk_free_threshold": risk_free,
            "risk_weight_scale": risk_scale,
            "drift_weight_scale": drift_scale,
            "support_weight_scale": support_scale,
            "residual_clip_scale": residual_clip_scale,
            "adaptive_support_score": support_score,
        }
    )
    return gate_debug


def pose_render_gradient_loss(
    image: Any,
    target: Any,
    *,
    mode: str | None,
    direct_density_mode: str | None,
    keyframe_info: dict[str, Any] | None = None,
    weight: float = 0.03,
    max_pose_risk: float = 0.50,
    max_utility_drift: float = 0.65,
    min_pose_support: float = 0.45,
    min_match_support: float = 0.45,
    min_weight_scale: float = 0.25,
    target_raw_loss: float = 0.02,
    risk_free_threshold: float = 0.25,
) -> tuple[Any, dict[str, Any]]:
    mode_name = str(mode or "off")
    info = keyframe_info or {}
    is_test = bool(info.get("is_test", False))
    debug: dict[str, Any] = {
        "mode": mode_name,
        "direct_density_mode": str(direct_density_mode or "off"),
        "is_test": is_test,
        "applied": False,
        "reason": "",
        "base_weight": float(weight or 0.0),
        "weight": float(weight or 0.0),
        "adaptive_weight_scale": 1.0,
        "residual_clip_scale": 1.0,
        "raw_loss": 0.0,
        "weighted_loss": 0.0,
    }
    if mode_name in {POSE_SAFE_EDGE_LOSS_MODE, POSE_ADAPTIVE_EDGE_LOSS_MODE} and not is_test:
        debug.update(
            pose_render_pose_gate_debug(
                keyframe_info,
                max_pose_risk=max_pose_risk,
                max_utility_drift=max_utility_drift,
                min_pose_support=min_pose_support,
                min_match_support=min_match_support,
            )
        )
    if not pose_render_edge_loss_enabled(
        mode,
        direct_density_mode,
        keyframe_info,
        weight=weight,
        max_pose_risk=max_pose_risk,
        max_utility_drift=max_utility_drift,
        min_pose_support=min_pose_support,
        min_match_support=min_match_support,
    ):
        debug["reason"] = str(debug.get("pose_gate_reason") or "disabled")
        return image.new_zeros(()), debug
    if tuple(image.shape) != tuple(target.shape):
        debug["reason"] = "shape_mismatch"
        return image.new_zeros(()), debug
    if image.ndim < 3 or image.shape[-2] < 2 or image.shape[-1] < 2:
        debug["reason"] = "invalid_spatial_shape"
        return image.new_zeros(()), debug

    raw_loss = gradient_l1_edge_loss(image, target)
    raw_loss_value = _detached_float(raw_loss, 0.0)
    effective_weight = float(weight or 0.0)
    reason = "gradient_edge_l1"
    if mode_name == POSE_ADAPTIVE_EDGE_LOSS_MODE:
        adaptive_debug = pose_render_adaptive_edge_weight_debug(
            keyframe_info,
            raw_loss=raw_loss_value,
            max_pose_risk=max_pose_risk,
            max_utility_drift=max_utility_drift,
            min_pose_support=min_pose_support,
            min_match_support=min_match_support,
            min_weight_scale=min_weight_scale,
            target_raw_loss=target_raw_loss,
            risk_free_threshold=risk_free_threshold,
        )
        debug.update(adaptive_debug)
        effective_weight = float(weight or 0.0) * float(
            adaptive_debug.get("adaptive_weight_scale", 1.0) or 1.0
        )
        reason = "gradient_edge_l1_adaptive"
    weighted_loss = effective_weight * raw_loss
    debug.update(
        {
            "applied": True,
            "reason": reason,
            "weight": effective_weight,
            "raw_loss": raw_loss_value,
            "weighted_loss": _detached_float(weighted_loss, 0.0),
        }
    )
    return weighted_loss, debug
