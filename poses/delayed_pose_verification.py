from __future__ import annotations

import math
from typing import Any, Sequence

import torch


def _point_count(keyframe: Any) -> int:
    desc = getattr(keyframe, "desc_kpts", None)
    mask = getattr(desc, "has_pt3d", None)
    return int(mask.sum().item()) if isinstance(mask, torch.Tensor) else 0


def _is_quarantined(keyframe: Any) -> bool:
    info = getattr(keyframe, "info", {}) or {}
    return bool(
        info.get("_pose_reference_quarantined", False)
        or info.get("_pose_initialization_risk", {}).get("isolated", False)
    )


def select_delayed_reference_groups(
    keyframes: Sequence[Any],
    target_position: int,
    *,
    solve_count: int = 8,
    validation_count: int = 6,
    min_point_count: int = 16,
) -> tuple[list[Any], list[Any]]:
    """Choose disjoint nearby train-frame references on both temporal sides."""
    if not 0 <= int(target_position) < len(keyframes):
        raise IndexError("target_position is outside keyframes")
    candidates: list[tuple[int, int, Any]] = []
    for position, keyframe in enumerate(keyframes):
        if position == int(target_position):
            continue
        if bool(getattr(keyframe, "is_test", False)):
            continue
        if _is_quarantined(keyframe) or _point_count(keyframe) < int(min_point_count):
            continue
        candidates.append((abs(position - int(target_position)), position, keyframe))
    candidates.sort(key=lambda item: (item[0], item[1]))

    required = max(0, int(solve_count)) + max(0, int(validation_count))
    ranked = candidates[:required]
    solve: list[Any] = []
    validation: list[Any] = []
    for rank, (_, _, keyframe) in enumerate(ranked):
        if len(solve) < int(solve_count) and (
            rank % 2 == 0 or len(validation) >= int(validation_count)
        ):
            solve.append(keyframe)
        elif len(validation) < int(validation_count):
            validation.append(keyframe)
        else:
            solve.append(keyframe)
    return solve, validation


def merge_global_reference_groups(
    local_solve: Sequence[Any],
    local_validation: Sequence[Any],
    global_ranked: Sequence[Any],
    *,
    solve_count: int,
    validation_count: int,
    global_solve_count: int = 4,
    global_validation_count: int = 3,
) -> tuple[list[Any], list[Any]]:
    """Mix disjoint local and nonlocal references without changing rank order."""
    solve: list[Any] = []
    validation: list[Any] = []
    used: set[int] = set()

    def add(output: list[Any], values: Sequence[Any], limit: int) -> None:
        if len(output) >= int(limit):
            return
        for value in values:
            identity = id(value)
            if identity in used:
                continue
            output.append(value)
            used.add(identity)
            if len(output) >= int(limit):
                return

    add(solve, global_ranked, min(int(global_solve_count), int(solve_count)))
    add(solve, local_solve, int(solve_count))
    add(solve, global_ranked, int(solve_count))
    add(
        validation,
        global_ranked,
        min(int(global_validation_count), int(validation_count)),
    )
    add(validation, local_validation, int(validation_count))
    add(validation, global_ranked, int(validation_count))
    return solve, validation


def summarize_candidate_validation(
    pre_errors: torch.Tensor,
    post_errors: torch.Tensor,
    pre_valid: torch.Tensor,
    post_valid: torch.Tensor,
    *,
    max_error: float,
    min_support: int,
    min_relative_improvement: float,
    max_mean_ratio: float,
    max_p90_ratio: float,
    correction_translation: float,
    correction_rotation_deg: float,
    max_translation: float,
    max_rotation_deg: float,
) -> dict[str, object]:
    """Compare an independently generated pose candidate on held-out matches."""
    finite = torch.isfinite(pre_errors) & torch.isfinite(post_errors)
    common = pre_valid & post_valid & finite
    informative = common & (
        (pre_errors <= float(max_error)) | (post_errors <= float(max_error))
    )
    pre = pre_errors[informative]
    post = post_errors[informative]
    support = int(informative.sum().item())

    debug: dict[str, object] = {
        "accepted": False,
        "reason": "insufficient_validation_support",
        "validation_support": support,
        "correction_translation": float(correction_translation),
        "correction_rotation_deg": float(correction_rotation_deg),
    }
    if support < int(min_support):
        return debug

    pre_median = float(pre.median().item())
    post_median = float(post.median().item())
    pre_mean = float(pre.mean().item())
    post_mean = float(post.mean().item())
    pre_p90 = float(torch.quantile(pre, 0.9).item())
    post_p90 = float(torch.quantile(post, 0.9).item())
    relative_improvement = (pre_median - post_median) / max(pre_median, 1e-12)
    mean_ratio = post_mean / max(pre_mean, 1e-12)
    p90_ratio = post_p90 / max(pre_p90, 1e-12)
    debug.update(
        {
            "pre_median": pre_median,
            "post_median": post_median,
            "pre_mean": pre_mean,
            "post_mean": post_mean,
            "pre_p90": pre_p90,
            "post_p90": post_p90,
            "relative_median_improvement": relative_improvement,
            "mean_ratio": mean_ratio,
            "p90_ratio": p90_ratio,
        }
    )
    checks = (
        (
            math.isfinite(correction_translation)
            and correction_translation <= float(max_translation)
        ),
        (
            math.isfinite(correction_rotation_deg)
            and correction_rotation_deg <= float(max_rotation_deg)
        ),
        relative_improvement >= float(min_relative_improvement),
        mean_ratio <= float(max_mean_ratio),
        p90_ratio <= float(max_p90_ratio),
    )
    reasons = (
        "translation_limit",
        "rotation_limit",
        "median_not_improved",
        "mean_regressed",
        "p90_regressed",
    )
    for passed, reason in zip(checks, reasons):
        if not passed:
            debug["reason"] = reason
            return debug
    debug["accepted"] = True
    debug["reason"] = "held_out_reprojection_improved"
    return debug
