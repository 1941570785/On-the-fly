from __future__ import annotations

from typing import Any

import torch

from scene.pose_render_edge_loss import (
    POSE_RENDER_GATE_INFO_KEY,
    POSE_SAFE_DIRECT_DENSITY_MODE,
    pose_render_pose_gate_debug,
    pose_render_training_signal_enabled,
)


POSE_SAFE_PSNR_LOSS_MODE = "mse_pose_safe_v1"
POSE_RISK_ROBUST_PSNR_LOSS_MODE = "robust_mse_pose_risk_v1"
POSE_RISK_FREQUENCY_PSNR_LOSS_MODE = "freq_mse_pose_risk_v1"
PSNR_LOSS_MODES = {
    POSE_SAFE_PSNR_LOSS_MODE,
    POSE_RISK_ROBUST_PSNR_LOSS_MODE,
    POSE_RISK_FREQUENCY_PSNR_LOSS_MODE,
}
TARGET_MASK_OFF_MODE = "off"
TARGET_MASK_NONZERO_GT_MODE = "nonzero_gt_v1"
TARGET_MASK_MASK_AWARE_RANDOM_NONZERO_GT_MODE = "mask_aware_random_nonzero_gt_v1"
TARGET_MASK_MODES = {
    TARGET_MASK_OFF_MODE,
    TARGET_MASK_NONZERO_GT_MODE,
    TARGET_MASK_MASK_AWARE_RANDOM_NONZERO_GT_MODE,
}
SUPPORT_WEIGHT_OFF_MODE = "off"
SUPPORT_WEIGHT_RAW_POSE_MODE = "raw_pose_support_v1"
SUPPORT_WEIGHT_RAW_POSE_GATE_MODE = "raw_pose_support_gate_v1"
SUPPORT_WEIGHT_MODES = {
    SUPPORT_WEIGHT_OFF_MODE,
    SUPPORT_WEIGHT_RAW_POSE_MODE,
    SUPPORT_WEIGHT_RAW_POSE_GATE_MODE,
}
CONTEXT_WEIGHT_OFF_MODE = "off"
CONTEXT_WEIGHT_MASK_AWARE_NO_MASK_BOOST_MODE = "mask_aware_no_mask_boost_v1"
CONTEXT_WEIGHT_MASK_AWARE_NO_MASK_RAW_RESPONSE_BOOST_MODE = (
    "mask_aware_no_mask_raw_response_boost_v1"
)
CONTEXT_WEIGHT_MASK_AWARE_NO_MASK_RAW_RESPONSE_GATE_BOOST_MODE = (
    "mask_aware_no_mask_raw_response_gate_boost_v1"
)
CONTEXT_WEIGHT_MASK_AWARE_NO_MASK_SCENE_LOW_GATE_BOOST_MODE = (
    "mask_aware_no_mask_scene_low_gate_boost_v1"
)
CONTEXT_WEIGHT_MASK_AWARE_NO_MASK_SCENE_LOW_FAST_GATE_BOOST_MODE = (
    "mask_aware_no_mask_scene_low_fast_gate_boost_v1"
)
CONTEXT_WEIGHT_MODES = {
    CONTEXT_WEIGHT_OFF_MODE,
    CONTEXT_WEIGHT_MASK_AWARE_NO_MASK_BOOST_MODE,
    CONTEXT_WEIGHT_MASK_AWARE_NO_MASK_RAW_RESPONSE_BOOST_MODE,
    CONTEXT_WEIGHT_MASK_AWARE_NO_MASK_RAW_RESPONSE_GATE_BOOST_MODE,
    CONTEXT_WEIGHT_MASK_AWARE_NO_MASK_SCENE_LOW_GATE_BOOST_MODE,
    CONTEXT_WEIGHT_MASK_AWARE_NO_MASK_SCENE_LOW_FAST_GATE_BOOST_MODE,
}
STRUCTURE_GATE_OFF_MODE = "off"
STRUCTURE_GATE_GRADIENT_CORRELATION_MODE = "gradient_correlation_v1"
STRUCTURE_GATE_MODES = {
    STRUCTURE_GATE_OFF_MODE,
    STRUCTURE_GATE_GRADIENT_CORRELATION_MODE,
}


def _detached_float(value: Any, default: float = 0.0) -> float:
    try:
        if hasattr(value, "detach"):
            return float(value.detach().cpu().item())
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))


def _linear_support_scale(
    value: int | None,
    *,
    low: float,
    high: float,
    min_scale: float,
) -> float | None:
    if value is None:
        return None
    if float(high) <= float(low):
        return 1.0
    unit = _clamp((float(value) - float(low)) / (float(high) - float(low)), 0.0, 1.0)
    return _clamp(float(min_scale) + (1.0 - float(min_scale)) * unit, min_scale, 1.0)


def _pose_render_gate_info(keyframe_info: dict[str, Any] | None) -> dict[str, Any]:
    info = keyframe_info or {}
    gate: dict[str, Any] = {}
    nested = info.get(POSE_RENDER_GATE_INFO_KEY, None)
    if isinstance(nested, dict):
        gate.update(nested)
    for key in (
        "pose_render_correspondence_count",
        "pose_render_match_count",
        "pose_render_final_pose_inliers",
        "num_matches",
        "pose_inliers",
    ):
        if key in info:
            gate[key] = info[key]
    return gate


def pose_render_support_weight_scale(
    keyframe_info: dict[str, Any] | None,
    *,
    mode: str | None = SUPPORT_WEIGHT_OFF_MODE,
    min_scale: float = 0.45,
    correspondence_low: float = 3000.0,
    correspondence_high: float = 8000.0,
    inlier_low: float = 900.0,
    inlier_high: float = 2500.0,
) -> tuple[float, dict[str, Any]]:
    mode_name = str(mode or "off")
    min_scale = _clamp(float(min_scale), 0.0, 1.0)
    debug: dict[str, Any] = {
        "support_weight_mode": mode_name,
        "support_weight_scale": 1.0,
        "support_weight_reason": "off",
        "support_weight_min_scale": min_scale,
        "support_correspondence_count": -1,
        "support_final_pose_inliers": -1,
        "support_correspondence_low": float(correspondence_low),
        "support_correspondence_high": float(correspondence_high),
        "support_inlier_low": float(inlier_low),
        "support_inlier_high": float(inlier_high),
    }
    if mode_name == SUPPORT_WEIGHT_OFF_MODE:
        return 1.0, debug
    if mode_name not in SUPPORT_WEIGHT_MODES:
        debug["support_weight_reason"] = "unknown_support_weight_mode"
        return 1.0, debug

    gate = _pose_render_gate_info(keyframe_info)
    correspondence_count = _as_int(
        gate.get(
            "pose_render_correspondence_count",
            gate.get("pose_render_match_count", gate.get("num_matches")),
        ),
        None,
    )
    final_pose_inliers = _as_int(
        gate.get("pose_render_final_pose_inliers", gate.get("pose_inliers")),
        None,
    )
    if correspondence_count is not None:
        debug["support_correspondence_count"] = int(correspondence_count)
    if final_pose_inliers is not None:
        debug["support_final_pose_inliers"] = int(final_pose_inliers)

    scales: list[float] = []
    corr_scale = _linear_support_scale(
        correspondence_count,
        low=correspondence_low,
        high=correspondence_high,
        min_scale=min_scale,
    )
    if corr_scale is not None:
        scales.append(corr_scale)
    inlier_scale = _linear_support_scale(
        final_pose_inliers,
        low=inlier_low,
        high=inlier_high,
        min_scale=min_scale,
    )
    if inlier_scale is not None:
        scales.append(inlier_scale)

    if not scales:
        debug["support_weight_reason"] = "missing_raw_support"
        if mode_name == SUPPORT_WEIGHT_RAW_POSE_GATE_MODE:
            debug["support_weight_scale"] = 0.0
            return 0.0, debug
        return 1.0, debug

    scale = _clamp(min(scales), min_scale, 1.0)
    if mode_name == SUPPORT_WEIGHT_RAW_POSE_GATE_MODE:
        debug["support_weight_gate_min_scale"] = 0.90
        if scale < 0.90:
            debug["support_weight_scale"] = 0.0
            debug["support_weight_reason"] = "low_raw_pose_support"
            return 0.0, debug
        debug["support_weight_scale"] = scale
        debug["support_weight_reason"] = "raw_pose_support_gate_passed"
        return scale, debug

    debug["support_weight_scale"] = scale
    debug["support_weight_reason"] = "raw_pose_support_scaled"
    return scale, debug


def pose_render_context_weight_scale(
    keyframe_info: dict[str, Any] | None,
    *,
    mode: str | None = CONTEXT_WEIGHT_OFF_MODE,
    boost_scale: float = 1.5,
    raw_loss: Any | None = None,
    raw_loss_min: float = 0.0028,
) -> tuple[float, dict[str, Any]]:
    mode_name = str(mode or "off")
    scale = 1.0
    debug: dict[str, Any] = {
        "context_weight_mode": mode_name,
        "context_weight_scale": scale,
        "context_weight_reason": "off",
        "context_weight_boost_scale": float(boost_scale),
        "context_weight_raw_loss_min": float(raw_loss_min),
    }
    if raw_loss is not None:
        debug["context_weight_raw_loss"] = _detached_float(raw_loss, 0.0)
    if mode_name == CONTEXT_WEIGHT_OFF_MODE:
        return scale, debug
    if mode_name not in CONTEXT_WEIGHT_MODES:
        debug["context_weight_reason"] = "unknown_context_weight_mode"
        return 0.0, debug

    info = keyframe_info or {}
    background_info = info.get("_paper_aligned_training_background_frame")
    decision = ""
    if isinstance(background_info, dict):
        decision = str(background_info.get("decision", ""))
    debug["context_weight_background_decision"] = decision
    if decision == "mask_aware_dark_no_mask_random":
        raw_response_modes = {
            CONTEXT_WEIGHT_MASK_AWARE_NO_MASK_RAW_RESPONSE_BOOST_MODE,
            CONTEXT_WEIGHT_MASK_AWARE_NO_MASK_RAW_RESPONSE_GATE_BOOST_MODE,
            CONTEXT_WEIGHT_MASK_AWARE_NO_MASK_SCENE_LOW_GATE_BOOST_MODE,
            CONTEXT_WEIGHT_MASK_AWARE_NO_MASK_SCENE_LOW_FAST_GATE_BOOST_MODE,
        }
        if mode_name in raw_response_modes:
            raw_value = _detached_float(raw_loss, 0.0)
            debug["context_weight_raw_loss"] = raw_value
            if raw_value < float(raw_loss_min):
                if (
                    mode_name
                    == CONTEXT_WEIGHT_MASK_AWARE_NO_MASK_RAW_RESPONSE_GATE_BOOST_MODE
                    or mode_name
                    == CONTEXT_WEIGHT_MASK_AWARE_NO_MASK_SCENE_LOW_GATE_BOOST_MODE
                    or mode_name
                    == CONTEXT_WEIGHT_MASK_AWARE_NO_MASK_SCENE_LOW_FAST_GATE_BOOST_MODE
                ):
                    debug["context_weight_scale"] = 0.0
                    debug["context_weight_reason"] = (
                        "mask_aware_no_mask_raw_response_low_gate"
                    )
                    return 0.0, debug
                debug["context_weight_reason"] = (
                    "mask_aware_no_mask_raw_response_low"
                )
                return scale, debug
            scale = max(1.0, float(boost_scale))
            debug["context_weight_scale"] = scale
            debug["context_weight_reason"] = (
                "mask_aware_no_mask_raw_response_boost"
            )
            return scale, debug
        scale = max(1.0, float(boost_scale))
        debug["context_weight_scale"] = scale
        debug["context_weight_reason"] = "mask_aware_no_mask_boost"
        return scale, debug
    if decision in {"mask_aware_dark_random", "mask_aware_dark_fixed_black"}:
        debug["context_weight_reason"] = "mask_aware_masked"
        return scale, debug

    debug["context_weight_reason"] = "missing_or_unmatched_context"
    return scale, debug


def target_valid_rgb_mask(target: Any, eps: float = 1e-6) -> Any:
    if getattr(target, "ndim", 0) < 3:
        raise ValueError("target must have a channel dimension")
    return target.detach().abs().sum(dim=-3) > float(eps)


def _expand_valid_mask(valid_mask: Any, reference: Any) -> Any:
    mask = valid_mask.bool()
    if mask.ndim == reference.ndim - 1:
        mask = mask.unsqueeze(-3)
    if mask.ndim != reference.ndim:
        raise ValueError("valid_mask must match image spatial dimensions")
    return mask.expand_as(reference)


def mean_squared_rgb_loss(image: Any, target: Any, valid_mask: Any | None = None) -> Any:
    if tuple(image.shape) != tuple(target.shape):
        raise ValueError("image and target must have identical shapes")
    if int(image.numel()) <= 0:
        return image.new_zeros(())
    residual = (image - target).pow(2)
    if valid_mask is not None:
        expanded_mask = _expand_valid_mask(valid_mask, residual)
        if int(expanded_mask.sum().item()) <= 0:
            return image.new_zeros(())
        return residual[expanded_mask].mean()
    return residual.mean()


def pose_render_structure_alignment_score(
    image: Any,
    target: Any,
    *,
    max_side: int = 160,
    eps: float = 1e-6,
) -> tuple[float, dict[str, Any]]:
    debug: dict[str, Any] = {
        "structure_gate_mode": STRUCTURE_GATE_GRADIENT_CORRELATION_MODE,
        "structure_score": 0.0,
        "structure_max_side": int(max_side),
    }
    if tuple(image.shape) != tuple(target.shape) or getattr(image, "ndim", 0) < 3:
        debug["structure_reason"] = "invalid_structure_shape"
        return 0.0, debug

    image_gray = image.detach().mean(dim=-3, keepdim=True)
    target_gray = target.detach().mean(dim=-3, keepdim=True)
    height = int(image_gray.shape[-2])
    width = int(image_gray.shape[-1])
    if height < 2 or width < 2:
        debug["structure_score"] = 1.0
        debug["structure_reason"] = "flat_structure"
        return 1.0, debug

    stride = max(1, int(max(height, width) // max(1, int(max_side))))
    if stride > 1:
        image_gray = image_gray[..., ::stride, ::stride]
        target_gray = target_gray[..., ::stride, ::stride]
    debug["structure_stride"] = int(stride)

    image_grad_x = image_gray[..., :, 1:] - image_gray[..., :, :-1]
    image_grad_y = image_gray[..., 1:, :] - image_gray[..., :-1, :]
    target_grad_x = target_gray[..., :, 1:] - target_gray[..., :, :-1]
    target_grad_y = target_gray[..., 1:, :] - target_gray[..., :-1, :]
    image_vec = torch.cat((image_grad_x.reshape(-1), image_grad_y.reshape(-1)))
    target_vec = torch.cat((target_grad_x.reshape(-1), target_grad_y.reshape(-1)))
    target_norm = target_vec.norm()
    image_norm = image_vec.norm()
    debug["structure_target_norm"] = _detached_float(target_norm, 0.0)
    debug["structure_image_norm"] = _detached_float(image_norm, 0.0)
    if _detached_float(target_norm, 0.0) <= float(eps):
        debug["structure_score"] = 1.0
        debug["structure_reason"] = "target_flat_structure"
        return 1.0, debug
    if _detached_float(image_norm, 0.0) <= float(eps):
        debug["structure_reason"] = "render_flat_structure"
        return 0.0, debug

    score_tensor = (image_vec * target_vec).sum() / (
        image_norm.clamp_min(float(eps)) * target_norm.clamp_min(float(eps))
    )
    score = _clamp(_detached_float(score_tensor, 0.0), -1.0, 1.0)
    debug["structure_score"] = score
    debug["structure_reason"] = "gradient_correlation"
    return score, debug


def _pose_render_risk_score(keyframe_info: dict[str, Any] | None) -> float:
    gate = _pose_render_gate_info(keyframe_info)
    for key in ("pose_render_risk_score", "pose_risk_score", "semantic_R_t"):
        if key in gate:
            return _clamp(_detached_float(gate.get(key), 0.0), 0.0, 1.0)
    return 0.0


def pose_risk_robust_rgb_loss(
    image: Any,
    target: Any,
    *,
    valid_mask: Any | None = None,
    keyframe_info: dict[str, Any] | None = None,
    max_pose_risk: float = 0.30,
    min_weight: float = 0.70,
    eps: float = 1e-6,
) -> tuple[Any, dict[str, Any]]:
    residual_sq = (image - target).pow(2)
    pixel_rmse = (residual_sq.mean(dim=-3, keepdim=True) + float(eps)).sqrt()

    if valid_mask is not None:
        pixel_mask = _expand_valid_mask(valid_mask, pixel_rmse)
        scale_source = pixel_rmse[pixel_mask]
    else:
        pixel_mask = None
        scale_source = pixel_rmse.reshape(-1)
    if int(scale_source.numel()) <= 0:
        zero = image.new_zeros(())
        return zero, {
            "robust_pose_risk_score": 0.0,
            "robust_pose_risk_unit": 0.0,
            "robust_scale": 0.0,
            "robust_scale_eff": 0.0,
            "robust_weight_min": 0.0,
            "robust_weight_mean": 0.0,
            "robust_weight_max": 0.0,
        }

    scale = scale_source.detach().mean().clamp_min(float(eps))
    pose_risk = _pose_render_risk_score(keyframe_info)
    risk_denominator = max(float(max_pose_risk), float(eps))
    pose_risk_unit = _clamp(pose_risk / risk_denominator, 0.0, 1.0)
    scale_multiplier = 1.15 - 0.25 * pose_risk_unit
    scale_eff = (scale * float(scale_multiplier)).clamp_min(float(eps))

    min_weight = _clamp(float(min_weight), 0.0, 1.0)
    robust_ratio = pixel_rmse.detach() / (4.0 * scale_eff + float(eps))
    soft_weight = 1.0 / (1.0 + robust_ratio.pow(2))
    weights = min_weight + (1.0 - min_weight) * soft_weight
    expanded_weights = weights.expand_as(residual_sq)

    if valid_mask is not None:
        expanded_mask = _expand_valid_mask(valid_mask, residual_sq)
        numerator = (residual_sq * expanded_weights)[expanded_mask].sum()
        denominator = expanded_weights[expanded_mask].sum().clamp_min(float(eps))
        weight_values = weights[pixel_mask].detach()
    else:
        numerator = (residual_sq * expanded_weights).sum()
        denominator = expanded_weights.sum().clamp_min(float(eps))
        weight_values = weights.detach().reshape(-1)

    loss = numerator / denominator
    debug = {
        "robust_pose_risk_score": pose_risk,
        "robust_pose_risk_unit": pose_risk_unit,
        "robust_scale": _detached_float(scale, 0.0),
        "robust_scale_eff": _detached_float(scale_eff, 0.0),
        "robust_weight_min": _detached_float(weight_values.min(), 0.0),
        "robust_weight_mean": _detached_float(weight_values.mean(), 0.0),
        "robust_weight_max": _detached_float(weight_values.max(), 0.0),
    }
    return loss, debug


def pose_risk_frequency_rgb_loss(
    image: Any,
    target: Any,
    *,
    valid_mask: Any | None = None,
    keyframe_info: dict[str, Any] | None = None,
    max_pose_risk: float = 0.30,
    max_boost: float = 0.25,
    eps: float = 1e-6,
) -> tuple[Any, dict[str, Any]]:
    residual_sq = (image - target).pow(2)
    target_gray = target.detach().mean(dim=-3, keepdim=True)
    grad_x = target_gray.new_zeros(target_gray.shape)
    grad_y = target_gray.new_zeros(target_gray.shape)
    grad_x[..., :, 1:] = (target_gray[..., :, 1:] - target_gray[..., :, :-1]).abs()
    grad_y[..., 1:, :] = (target_gray[..., 1:, :] - target_gray[..., :-1, :]).abs()
    frequency = grad_x + grad_y

    if valid_mask is not None:
        pixel_mask = _expand_valid_mask(valid_mask, frequency)
        frequency_source = frequency[pixel_mask]
    else:
        pixel_mask = None
        frequency_source = frequency.reshape(-1)
    if int(frequency_source.numel()) <= 0:
        zero = image.new_zeros(())
        return zero, {
            "frequency_boost": 0.0,
            "frequency_pose_risk_score": 0.0,
            "frequency_pose_risk_unit": 0.0,
            "frequency_weight_min": 1.0,
            "frequency_weight_mean": 1.0,
            "frequency_weight_max": 1.0,
        }

    pose_risk = _pose_render_risk_score(keyframe_info)
    risk_denominator = max(float(max_pose_risk), float(eps))
    pose_risk_unit = _clamp(pose_risk / risk_denominator, 0.0, 1.0)
    boost = _clamp(float(max_boost), 0.0, 1.0) * (1.0 - 0.50 * pose_risk_unit)
    frequency_mean = frequency_source.detach().mean().clamp_min(float(eps))
    normalized_frequency = (frequency / frequency_mean).detach().clamp(0.0, 3.0)
    weights = 1.0 + float(boost) * normalized_frequency
    expanded_weights = weights.expand_as(residual_sq)

    if valid_mask is not None:
        expanded_mask = _expand_valid_mask(valid_mask, residual_sq)
        numerator = (residual_sq * expanded_weights)[expanded_mask].sum()
        denominator = expanded_weights[expanded_mask].sum().clamp_min(float(eps))
        weight_values = weights[pixel_mask].detach()
    else:
        numerator = (residual_sq * expanded_weights).sum()
        denominator = expanded_weights.sum().clamp_min(float(eps))
        weight_values = weights.detach().reshape(-1)

    loss = numerator / denominator
    debug = {
        "frequency_boost": float(boost),
        "frequency_pose_risk_score": pose_risk,
        "frequency_pose_risk_unit": pose_risk_unit,
        "frequency_weight_min": _detached_float(weight_values.min(), 1.0),
        "frequency_weight_mean": _detached_float(weight_values.mean(), 1.0),
        "frequency_weight_max": _detached_float(weight_values.max(), 1.0),
    }
    return loss, debug


def pose_render_mse_loss_enabled(
    mode: str | None,
    direct_density_mode: str | None,
    keyframe_info: dict[str, Any] | None = None,
    weight: float = 0.0,
    render_frame_policy: str | None = None,
    max_pose_risk: float = 0.30,
    max_utility_drift: float = 0.55,
    min_pose_support: float = 0.45,
    min_match_support: float = 0.45,
) -> bool:
    if str(mode or "off") not in PSNR_LOSS_MODES:
        return False
    if not pose_render_training_signal_enabled(
        direct_density_mode, render_frame_policy
    ):
        return False
    if float(weight or 0.0) <= 0.0:
        return False
    info = keyframe_info or {}
    if bool(info.get("is_test", False)):
        return False
    gate_debug = pose_render_pose_gate_debug(
        keyframe_info,
        max_pose_risk=max_pose_risk,
        max_utility_drift=max_utility_drift,
        min_pose_support=min_pose_support,
        min_match_support=min_match_support,
    )
    return bool(gate_debug.get("pose_gate_passed", False))


def pose_render_mse_loss(
    image: Any,
    target: Any,
    *,
    mode: str | None,
    direct_density_mode: str | None,
    render_frame_policy: str | None = None,
    keyframe_info: dict[str, Any] | None = None,
    weight: float = 0.10,
    target_mask_mode: str | None = TARGET_MASK_OFF_MODE,
    support_weight_mode: str | None = SUPPORT_WEIGHT_OFF_MODE,
    context_weight_mode: str | None = CONTEXT_WEIGHT_OFF_MODE,
    support_weight_min_scale: float = 0.45,
    support_weight_correspondence_low: float = 3000.0,
    support_weight_correspondence_high: float = 8000.0,
    support_weight_inlier_low: float = 900.0,
    support_weight_inlier_high: float = 2500.0,
    structure_gate_mode: str | None = STRUCTURE_GATE_OFF_MODE,
    structure_min_score: float = 0.0,
    min_target_valid_ratio: float = 0.05,
    max_pose_risk: float = 0.30,
    max_utility_drift: float = 0.55,
    min_pose_support: float = 0.45,
    min_match_support: float = 0.45,
) -> tuple[Any, dict[str, Any]]:
    mode_name = str(mode or "off")
    info = keyframe_info or {}
    is_test = bool(info.get("is_test", False))
    debug: dict[str, Any] = {
        "mode": mode_name,
        "direct_density_mode": str(direct_density_mode or "off"),
        "render_frame_policy": str(render_frame_policy or "off"),
        "is_test": is_test,
        "applied": False,
        "reason": "",
        "base_weight": float(weight or 0.0),
        "weight": float(weight or 0.0),
        "target_mask_mode": str(target_mask_mode or "off"),
        "target_mask_applied": False,
        "target_valid_ratio": 1.0,
        "support_weight_mode": str(support_weight_mode or "off"),
        "support_weight_scale": 1.0,
        "support_weight_reason": "off",
        "context_weight_mode": str(context_weight_mode or "off"),
        "context_weight_scale": 1.0,
        "context_weight_reason": "off",
        "structure_gate_mode": str(structure_gate_mode or "off"),
        "structure_min_score": float(structure_min_score or 0.0),
        "raw_loss": 0.0,
        "weighted_loss": 0.0,
    }
    background_info = info.get("_paper_aligned_training_background_frame")
    if isinstance(background_info, dict):
        debug["training_background_mode"] = str(background_info.get("mode", ""))
        debug["training_background_decision"] = str(
            background_info.get("decision", "")
        )
        if "target_mean" in background_info:
            debug["training_background_target_mean"] = _detached_float(
                background_info.get("target_mean"),
                0.0,
            )
        if "valid_ratio" in background_info:
            debug["training_background_valid_ratio"] = _detached_float(
                background_info.get("valid_ratio"),
                0.0,
            )
    if mode_name in PSNR_LOSS_MODES and not is_test:
        debug.update(
            pose_render_pose_gate_debug(
                keyframe_info,
                max_pose_risk=max_pose_risk,
                max_utility_drift=max_utility_drift,
                min_pose_support=min_pose_support,
                min_match_support=min_match_support,
            )
        )
    signal_enabled = pose_render_training_signal_enabled(
        direct_density_mode, render_frame_policy
    )
    debug["render_training_signal_enabled"] = bool(signal_enabled)
    if not pose_render_mse_loss_enabled(
        mode,
        direct_density_mode,
        keyframe_info,
        weight=weight,
        render_frame_policy=render_frame_policy,
        max_pose_risk=max_pose_risk,
        max_utility_drift=max_utility_drift,
        min_pose_support=min_pose_support,
        min_match_support=min_match_support,
    ):
        if mode_name in PSNR_LOSS_MODES and not signal_enabled:
            debug["reason"] = "direct_density_mismatch"
        else:
            debug["reason"] = str(debug.get("pose_gate_reason") or "disabled")
        return image.new_zeros(()), debug
    if tuple(image.shape) != tuple(target.shape):
        debug["reason"] = "shape_mismatch"
        return image.new_zeros(()), debug
    if int(image.numel()) <= 0:
        debug["reason"] = "invalid_spatial_shape"
        return image.new_zeros(()), debug

    valid_mask = None
    mask_mode = str(target_mask_mode or "off")
    use_nonzero_target_mask = mask_mode == TARGET_MASK_NONZERO_GT_MODE
    if mask_mode == TARGET_MASK_MASK_AWARE_RANDOM_NONZERO_GT_MODE:
        use_nonzero_target_mask = (
            str(debug.get("training_background_decision", ""))
            == "mask_aware_dark_random"
        )
    if use_nonzero_target_mask:
        valid_mask = target_valid_rgb_mask(target)
        valid_count = int(valid_mask.sum().item())
        total_count = int(valid_mask.numel())
        valid_ratio = float(valid_count) / float(max(total_count, 1))
        debug.update(
            {
                "target_mask_applied": True,
                "target_valid_ratio": valid_ratio,
            }
        )
        if valid_ratio < float(min_target_valid_ratio):
            debug["reason"] = "invalid_target_mask"
            return image.new_zeros(()), debug
    elif mask_mode not in TARGET_MASK_MODES:
        debug["reason"] = "unknown_target_mask_mode"
        return image.new_zeros(()), debug

    support_scale, support_debug = pose_render_support_weight_scale(
        keyframe_info,
        mode=support_weight_mode,
        min_scale=support_weight_min_scale,
        correspondence_low=support_weight_correspondence_low,
        correspondence_high=support_weight_correspondence_high,
        inlier_low=support_weight_inlier_low,
        inlier_high=support_weight_inlier_high,
    )
    debug.update(support_debug)
    supported_weight = float(weight or 0.0) * float(support_scale)
    debug["weight"] = supported_weight
    if supported_weight <= 0.0:
        debug["reason"] = str(debug.get("support_weight_reason") or "zero_support_weight")
        return image.new_zeros(()), debug

    structure_mode = str(structure_gate_mode or "off")
    if structure_mode == STRUCTURE_GATE_GRADIENT_CORRELATION_MODE:
        structure_score, structure_debug = pose_render_structure_alignment_score(
            image,
            target,
        )
        debug.update(structure_debug)
        debug["structure_min_score"] = float(structure_min_score or 0.0)
        if structure_score < float(structure_min_score or 0.0):
            debug["reason"] = "psnr_structure_gate_low"
            debug["weight"] = 0.0
            return image.new_zeros(()), debug
    elif structure_mode not in STRUCTURE_GATE_MODES:
        debug["reason"] = "unknown_structure_gate_mode"
        return image.new_zeros(()), debug

    robust_debug: dict[str, Any] = {}
    if mode_name == POSE_RISK_ROBUST_PSNR_LOSS_MODE:
        raw_loss, robust_debug = pose_risk_robust_rgb_loss(
            image,
            target,
            valid_mask=valid_mask,
            keyframe_info=keyframe_info,
            max_pose_risk=max_pose_risk,
        )
        reason = "robust_mse_rgb_pose_safe"
    elif mode_name == POSE_RISK_FREQUENCY_PSNR_LOSS_MODE:
        raw_loss, robust_debug = pose_risk_frequency_rgb_loss(
            image,
            target,
            valid_mask=valid_mask,
            keyframe_info=keyframe_info,
            max_pose_risk=max_pose_risk,
        )
        reason = "freq_mse_rgb_pose_safe"
    else:
        raw_loss = mean_squared_rgb_loss(image, target, valid_mask=valid_mask)
        reason = "mse_rgb_pose_safe"
    context_scale, context_debug = pose_render_context_weight_scale(
        keyframe_info,
        mode=context_weight_mode,
        raw_loss=raw_loss,
    )
    debug.update(context_debug)
    effective_weight = supported_weight * float(context_scale)
    debug["weight"] = effective_weight
    if effective_weight <= 0.0:
        debug.update(
            {
                "raw_loss": _detached_float(raw_loss, 0.0),
                "weighted_loss": 0.0,
            }
        )
        debug.update(robust_debug)
        debug["reason"] = str(debug.get("context_weight_reason") or "zero_context_weight")
        return image.new_zeros(()), debug
    weighted_loss = effective_weight * raw_loss
    debug.update(
        {
            "applied": True,
            "reason": reason,
            "raw_loss": _detached_float(raw_loss, 0.0),
            "weighted_loss": _detached_float(weighted_loss, 0.0),
        }
    )
    debug.update(robust_debug)
    return weighted_loss, debug
