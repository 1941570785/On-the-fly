from __future__ import annotations

from typing import Any


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _clamp01(value: Any, default: float = 0.0) -> float:
    v = _as_float(value, default)
    return max(0.0, min(1.0, v))


def _normalized_shortfall(value: float, target: float) -> float:
    return 1.0 - _clamp01(float(value) / max(float(target), 1e-6))


def pose_render_posterior_risk_info(
    *,
    pose_debug: dict[str, Any] | None,
    viewpoint_scores: dict[str, Any] | None,
    min_num_inliers: int,
    num_matches: int | None = None,
    pose_inliers: int | None = None,
    semantic_pose_risk: float = 0.0,
    semantic_q: float = 1.0,
    semantic_b_r: float = 1.0,
    high_threshold: float = 0.55,
) -> dict[str, Any]:
    """Estimate pose risk after pose initialization, before render updates absorb it."""

    pose_debug = dict(pose_debug or {})
    viewpoint_scores = dict(viewpoint_scores or {})
    min_inliers = max(1, int(min_num_inliers or 1))

    match_count = max(
        0,
        _as_int(num_matches, 0),
        _as_int(pose_debug.get("match_count_total"), 0),
    )
    correspondences = max(
        0,
        _as_int(pose_debug.get("num_2d3d_correspondences"), 0),
        match_count,
    )
    pnp_inliers = max(0, _as_int(pose_debug.get("num_pnp_inliers"), 0))
    miniba_inliers = max(0, _as_int(pose_debug.get("num_miniba_inliers"), 0))
    final_pose_inliers = max(0, _as_int(pose_inliers, max(miniba_inliers, pnp_inliers)))
    if final_pose_inliers == 0:
        final_pose_inliers = max(miniba_inliers, pnp_inliers)

    support_ratio = _clamp01(final_pose_inliers / max(2.0 * float(min_inliers), 1.0))
    inlier_ratio = _clamp01(final_pose_inliers / max(float(correspondences), 1.0))
    pnp_ratio = _clamp01(pnp_inliers / max(float(correspondences), 1.0))
    miniba_ratio = _clamp01(miniba_inliers / max(float(correspondences), 1.0))
    grid_coverage = _clamp01(viewpoint_scores.get("inlier_grid_coverage"), 0.75)
    grid_entropy = _clamp01(viewpoint_scores.get("inlier_grid_entropy"), 0.75)
    support_concentration = _clamp01(viewpoint_scores.get("support_concentration"), 0.0)

    posterior_risk = _clamp01(
        0.25 * (1.0 - support_ratio)
        + 0.25 * _normalized_shortfall(inlier_ratio, 0.65)
        + 0.14 * _normalized_shortfall(pnp_ratio, 0.55)
        + 0.12 * _normalized_shortfall(miniba_ratio, 0.55)
        + 0.10 * (1.0 - grid_coverage)
        + 0.06 * (1.0 - grid_entropy)
        + 0.08 * support_concentration
    )

    semantic_pose_risk = _clamp01(semantic_pose_risk, 0.0)
    semantic_q = _clamp01(semantic_q, 1.0)
    semantic_b_r = _clamp01(semantic_b_r, 1.0)
    render_risk = max(semantic_pose_risk, posterior_risk)
    risk_high = bool(
        render_risk >= float(high_threshold)
        or semantic_q < 0.38
        or semantic_b_r < 0.25
    )
    if semantic_pose_risk >= float(high_threshold) or semantic_q < 0.38 or semantic_b_r < 0.25:
        reason = "semantic_pose_risk_high"
    elif posterior_risk >= float(high_threshold) or support_ratio < 0.55 or inlier_ratio < 0.18:
        reason = "posterior_pose_support_weak"
    elif grid_coverage < 0.55 or grid_entropy < 0.60 or support_concentration > 0.35:
        reason = "posterior_spatial_support_weak"
    else:
        reason = "posterior_pose_support_ok"

    confidence = 1.0 - render_risk
    return {
        "pose_render_posterior_risk_score": float(posterior_risk),
        "pose_render_risk_score": float(render_risk),
        "pose_render_pose_confidence": float(confidence),
        "pose_render_risk_high": bool(risk_high),
        "pose_render_risk_reason": reason,
        "pose_render_support_ratio": float(support_ratio),
        "pose_render_inlier_ratio": float(inlier_ratio),
        "pose_render_pnp_inlier_ratio": float(pnp_ratio),
        "pose_render_miniba_inlier_ratio": float(miniba_ratio),
        "pose_render_correspondence_count": int(correspondences),
        "pose_render_match_count": int(match_count),
        "pose_render_final_pose_inliers": int(final_pose_inliers),
        "pose_render_grid_coverage": float(grid_coverage),
        "pose_render_grid_entropy": float(grid_entropy),
        "pose_render_support_concentration": float(support_concentration),
    }


def augment_pose_render_payload_with_posterior_risk(
    payload: dict[str, Any],
    *,
    pose_debug: dict[str, Any] | None,
    viewpoint_scores: dict[str, Any] | None,
    min_num_inliers: int,
    high_threshold: float = 0.55,
) -> dict[str, Any]:
    semantic_pose_risk = _clamp01(
        payload.get("pose_risk_score", payload.get("semantic_R_t")), 0.0
    )
    risk_info = pose_render_posterior_risk_info(
        pose_debug=pose_debug,
        viewpoint_scores=viewpoint_scores,
        min_num_inliers=min_num_inliers,
        num_matches=_as_int(payload.get("num_matches"), 0),
        pose_inliers=_as_int(payload.get("pose_inliers"), 0),
        semantic_pose_risk=semantic_pose_risk,
        semantic_q=_clamp01(payload.get("semantic_Q_t"), 1.0),
        semantic_b_r=_clamp01(payload.get("semantic_B_R_t"), 1.0),
        high_threshold=high_threshold,
    )
    payload["semantic_pose_risk_score"] = float(semantic_pose_risk)
    payload.update(risk_info)
    payload["pose_risk_score"] = float(risk_info["pose_render_risk_score"])
    payload["pose_risk_high"] = bool(
        payload.get("pose_risk_high", False) or risk_info["pose_render_risk_high"]
    )
    return payload
