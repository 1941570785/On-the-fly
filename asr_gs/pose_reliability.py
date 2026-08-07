from __future__ import annotations

import math
from typing import Mapping

from asr_gs.config import PoseReliabilityConfig


POLICY_NAME = "support_residual_secondary_review"


def should_retry_failed_pose(
    diagnostics: Mapping[str, object],
    config: PoseReliabilityConfig,
) -> bool:
    """Retry only MiniBA failures backed by sufficient geometric support."""
    if str(diagnostics.get("failure_reason", "")) != "miniba_inliers_too_few":
        return False
    return (
        int(diagnostics.get("num_2d3d_correspondences", 0))
        >= config.retry_min_2d3d
        and int(diagnostics.get("num_pnp_inliers", 0))
        >= config.retry_min_pnp_inliers
    )


def should_review_weak_pose(
    diagnostics: Mapping[str, object],
    config: PoseReliabilityConfig,
) -> bool:
    """Flag a successful pose whose support or MiniBA residual is weak."""
    if str(diagnostics.get("failure_reason", "")):
        return False
    correspondence_count = int(
        diagnostics.get("num_2d3d_correspondences", 0)
    )
    if correspondence_count < config.multi_hypothesis_min_2d3d:
        return False
    pnp_inliers = int(diagnostics.get("num_pnp_inliers", 0))
    sampled = max(
        1,
        int(diagnostics.get("num_pnp_candidate_correspondences", 0)),
    )
    pnp_ratio = pnp_inliers / sampled
    residual = float(
        diagnostics.get("direct_pose_miniba_residual", 0.0)
    )
    return (
        (
            math.isfinite(residual)
            and residual > config.multi_hypothesis_max_miniba_residual
        )
        or pnp_inliers < config.multi_hypothesis_min_pnp_inliers
        or pnp_ratio < config.multi_hypothesis_max_pnp_ratio
    )


def pose_candidate_score(diagnostics: Mapping[str, object]) -> float:
    """Rank a pose by geometric support, residual, and bounded motion."""
    pnp_inliers = float(diagnostics.get("num_pnp_inliers", 0))
    miniba_inliers = float(diagnostics.get("num_miniba_inliers", 0))
    rotation = float(
        diagnostics.get("direct_pose_motion_rotation_deg", 0.0)
    )
    translation = float(
        diagnostics.get("direct_pose_motion_translation", 0.0)
    )
    residual = float(
        diagnostics.get("direct_pose_miniba_residual", 0.0)
    )
    residual_penalty = 400.0 * residual if math.isfinite(residual) else 0.0
    return (
        pnp_inliers
        + 0.5 * miniba_inliers
        - rotation
        - 25.0 * translation
        - residual_penalty
    )


def pose_candidate_improves(
    current: Mapping[str, object],
    candidate: Mapping[str, object],
    config: PoseReliabilityConfig,
) -> bool:
    """Accept a replacement only when its gain is supported and bounded."""
    current_support = float(current.get("num_pnp_inliers", 0)) + 0.5 * float(
        current.get("num_miniba_inliers", 0)
    )
    candidate_support = float(
        candidate.get("num_pnp_inliers", 0)
    ) + 0.5 * float(candidate.get("num_miniba_inliers", 0))
    current_residual = float(
        current.get("direct_pose_miniba_residual", float("inf"))
    )
    candidate_residual = float(
        candidate.get("direct_pose_miniba_residual", float("inf"))
    )
    support_gain = candidate_support > current_support * (
        1.0 + config.min_support_gain
    )
    residual_gain = (
        math.isfinite(current_residual)
        and math.isfinite(candidate_residual)
        and candidate_residual
        < current_residual * (1.0 - config.min_residual_gain)
        and candidate_support
        >= current_support * config.min_residual_support_ratio
    )
    if not support_gain and not residual_gain:
        return False

    current_rotation = float(
        current.get("direct_pose_motion_rotation_deg", 0.0)
    )
    candidate_rotation = float(
        candidate.get("direct_pose_motion_rotation_deg", 0.0)
    )
    current_translation = float(
        current.get("direct_pose_motion_translation", 0.0)
    )
    candidate_translation = float(
        candidate.get("direct_pose_motion_translation", 0.0)
    )
    max_rotation = max(
        config.rotation_floor_deg,
        current_rotation + config.rotation_margin_deg,
        current_rotation * config.rotation_scale,
    )
    max_translation = max(
        config.translation_floor,
        current_translation * config.translation_scale
        + config.translation_margin,
    )
    return (
        candidate_rotation <= max_rotation
        and candidate_translation <= max_translation
        and pose_candidate_score(candidate) > pose_candidate_score(current)
    )
