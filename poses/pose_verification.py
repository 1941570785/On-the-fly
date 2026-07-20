from __future__ import annotations

import math
from typing import Any

import torch


def _pose_4x4(pose: torch.Tensor) -> torch.Tensor:
    if pose.shape == (4, 4):
        return pose
    if pose.shape == (3, 4):
        out = torch.eye(4, dtype=pose.dtype, device=pose.device)
        out[:3] = pose
        return out
    raise ValueError(f"Expected a 3x4 or 4x4 pose, got {tuple(pose.shape)}")


def compute_reprojection_errors(
    pose: torch.Tensor,
    xyz: torch.Tensor,
    uv: torch.Tensor,
    *,
    focal: torch.Tensor,
    centre: torch.Tensor,
    min_depth: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-correspondence pixel errors for a world-to-camera pose."""
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError("xyz must have shape [N, 3]")
    if uv.shape != (xyz.shape[0], 2):
        raise ValueError("uv must have shape [N, 2]")
    pose4 = _pose_4x4(pose)
    camera_xyz = xyz @ pose4[:3, :3].transpose(0, 1) + pose4[:3, 3]
    depth = camera_xyz[:, 2]
    finite = (
        torch.isfinite(camera_xyz).all(dim=-1)
        & torch.isfinite(uv).all(dim=-1)
        & torch.isfinite(pose4).all()
    )
    valid = finite & (depth > float(min_depth))
    focal_value = focal.reshape(-1)[0].to(dtype=xyz.dtype, device=xyz.device)
    centre_value = centre.reshape(2).to(dtype=xyz.dtype, device=xyz.device)
    safe_depth = torch.where(valid, depth, torch.ones_like(depth))
    projected = camera_xyz[:, :2] * (focal_value / safe_depth[:, None]) + centre_value
    errors = torch.linalg.vector_norm(projected - uv, dim=-1)
    errors = torch.where(valid & torch.isfinite(errors), errors, torch.full_like(errors, float("inf")))
    return errors, valid


def summarize_reprojection_errors(
    errors: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, float | int]:
    valid = valid_mask.to(dtype=torch.bool) & torch.isfinite(errors)
    values = errors[valid]
    if values.numel() == 0:
        return {
            "valid_count": 0,
            "mean": float("inf"),
            "median": float("inf"),
            "p90": float("inf"),
            "max": float("inf"),
        }
    return {
        "valid_count": int(values.numel()),
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "p90": float(torch.quantile(values, 0.90).item()),
        "max": float(values.max().item()),
    }


def select_robust_correspondences(
    errors: torch.Tensor,
    uv: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    width: int,
    height: int,
    mad_scale: float = 2.5,
    max_cutoff: float = float("inf"),
    min_support: int = 24,
    grid_rows: int = 4,
    grid_cols: int = 6,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """MAD-filter correspondences while preserving spatially distributed support."""
    if errors.ndim != 1 or uv.shape != (errors.shape[0], 2):
        raise ValueError("errors and uv must have shapes [N] and [N, 2]")
    valid = (
        valid_mask.to(dtype=torch.bool)
        & torch.isfinite(errors)
        & torch.isfinite(uv).all(dim=-1)
    )
    selected = torch.zeros_like(valid)
    values = errors[valid]
    if values.numel() == 0:
        return selected, {
            "valid_count": 0,
            "selected_count": 0,
            "rejected_count": 0,
            "median": float("inf"),
            "mad": float("inf"),
            "cutoff": float("inf"),
            "occupied_cells": 0,
            "selected_cells": 0,
            "restored_for_min_support": 0,
        }

    median = values.median()
    mad = (values - median).abs().median()
    robust_sigma = 1.4826 * mad
    cutoff_value = max(1.0, float((median + max(0.0, mad_scale) * robust_sigma).item()))
    if math.isfinite(max_cutoff) and max_cutoff > 0:
        cutoff_value = min(cutoff_value, float(max_cutoff))
    selected = valid & (errors <= cutoff_value)

    rows = max(1, int(grid_rows))
    cols = max(1, int(grid_cols))
    safe_width = max(1, int(width))
    safe_height = max(1, int(height))
    x = torch.floor(uv[:, 0] * cols / safe_width).to(torch.long).clamp(0, cols - 1)
    y = torch.floor(uv[:, 1] * rows / safe_height).to(torch.long).clamp(0, rows - 1)
    cells = y * cols + x
    bounded = valid
    if math.isfinite(max_cutoff) and max_cutoff > 0:
        bounded = bounded & (errors <= float(max_cutoff))
    occupied = torch.unique(cells[bounded])
    for cell in occupied:
        indices = torch.where(bounded & (cells == cell))[0]
        if indices.numel() > 0:
            best = indices[torch.argmin(errors[indices])]
            selected[best] = True

    before_restore = int(selected.sum().item())
    target = min(max(0, int(min_support)), int(bounded.sum().item()))
    if before_restore < target:
        candidates = torch.where(bounded)[0]
        order = torch.argsort(errors[candidates])
        selected[candidates[order[:target]]] = True
    selected_cells = int(torch.unique(cells[selected]).numel()) if bool(selected.any()) else 0
    selected_count = int(selected.sum().item())
    valid_count = int(valid.sum().item())
    return selected, {
        "valid_count": valid_count,
        "selected_count": selected_count,
        "rejected_count": valid_count - selected_count,
        "median": float(median.item()),
        "mad": float(mad.item()),
        "cutoff": float(cutoff_value),
        "occupied_cells": int(occupied.numel()),
        "selected_cells": selected_cells,
        "restored_for_min_support": max(0, selected_count - before_restore),
    }


def pose_correction_magnitude(
    initial_pose: torch.Tensor,
    refined_pose: torch.Tensor,
) -> dict[str, float]:
    initial = _pose_4x4(initial_pose)
    refined = _pose_4x4(refined_pose)
    relative_rotation = refined[:3, :3] @ initial[:3, :3].transpose(0, 1)
    cosine = ((torch.trace(relative_rotation) - 1.0) * 0.5).clamp(-1.0, 1.0)
    rotation_deg = torch.rad2deg(torch.acos(cosine))
    initial_centre = -(initial[:3, :3].transpose(0, 1) @ initial[:3, 3])
    refined_centre = -(refined[:3, :3].transpose(0, 1) @ refined[:3, 3])
    translation = torch.linalg.vector_norm(refined_centre - initial_centre)
    return {
        "rotation_deg": float(rotation_deg.item()),
        "translation": float(translation.item()),
    }


def decide_pose_refinement(
    *,
    pre: dict[str, Any],
    post: dict[str, Any],
    correction: dict[str, Any],
    max_rotation_deg: float,
    max_translation: float,
    min_relative_median_improvement: float = 0.02,
    max_p90_ratio: float = 1.01,
    min_support_ratio: float = 0.80,
) -> dict[str, float | int | bool | str]:
    pre_median = float(pre.get("median", float("inf")))
    post_median = float(post.get("median", float("inf")))
    pre_mean = float(pre.get("mean", pre_median))
    post_mean = float(post.get("mean", post_median))
    pre_p90 = float(pre.get("p90", float("inf")))
    post_p90 = float(post.get("p90", float("inf")))
    pre_count = int(pre.get("valid_count", 0) or 0)
    post_count = int(post.get("valid_count", 0) or 0)
    rotation_deg = float(correction.get("rotation_deg", float("inf")))
    translation = float(correction.get("translation", float("inf")))
    finite = all(
        math.isfinite(value)
        for value in (
            pre_mean,
            post_mean,
            pre_median,
            post_median,
            pre_p90,
            post_p90,
            rotation_deg,
            translation,
        )
    )
    relative_improvement = (
        (pre_median - post_median) / max(abs(pre_median), 1e-6)
        if finite
        else float("-inf")
    )
    support_ratio = post_count / max(pre_count, 1)
    relative_mean_improvement = (
        (pre_mean - post_mean) / max(abs(pre_mean), 1e-6)
        if finite
        else float("-inf")
    )

    reason = "accepted"
    if not finite:
        reason = "non_finite_diagnostics"
    elif pre_count <= 0 or post_count <= 0:
        reason = "insufficient_valid_support"
    elif relative_improvement < float(min_relative_median_improvement):
        reason = "median_not_improved"
    elif post_mean > pre_mean:
        reason = "mean_degraded"
    elif post_p90 > pre_p90 * float(max_p90_ratio):
        reason = "p90_degraded"
    elif support_ratio < float(min_support_ratio):
        reason = "support_collapsed"
    elif rotation_deg > float(max_rotation_deg):
        reason = "rotation_correction_too_large"
    elif translation > float(max_translation):
        reason = "translation_correction_too_large"
    return {
        "accepted": reason == "accepted",
        "reason": reason,
        "relative_median_improvement": float(relative_improvement),
        "relative_mean_improvement": float(relative_mean_improvement),
        "support_ratio": float(support_ratio),
        "rotation_correction_deg": rotation_deg,
        "translation_correction": translation,
        "max_rotation_deg": float(max_rotation_deg),
        "max_translation": float(max_translation),
    }


def choose_verified_pose(
    initial_pose: torch.Tensor,
    refined_pose: torch.Tensor,
    decision: dict[str, Any],
) -> torch.Tensor:
    return (refined_pose if bool(decision.get("accepted", False)) else initial_pose).clone()
