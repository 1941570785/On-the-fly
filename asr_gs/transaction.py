"""Snapshot, guard, and rollback utilities for module C."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch


STATE_KEYS = ("val", "exp_avg", "exp_avg_sq", "lr")
DEFAULT_PARAMETER_GRADIENT_SCALES = {
    "xyz": 0.10,
    "scaling": 0.10,
    "rotation": 0.10,
    "opacity": 0.25,
    "f_dc": 0.50,
    "f_rest": 0.50,
}


def snapshot_selected_gaussian_state(
    gaussian_params: Mapping[str, Mapping[str, Any]],
    selection: torch.Tensor,
) -> dict[str, dict[str, tuple[str, torch.Tensor]]]:
    """Capture selected Gaussian rows and optimizer state for exact rollback."""
    row_count = int(selection.numel())
    snapshot: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}
    for name, state in gaussian_params.items():
        captured: dict[str, tuple[str, torch.Tensor]] = {}
        for key in STATE_KEYS:
            value = state.get(key)
            if not torch.is_tensor(value):
                continue
            if value.ndim > 0 and value.shape[0] == row_count:
                captured[key] = ("selected", value.detach()[selection].clone())
            else:
                captured[key] = ("full", value.detach().clone())
        if captured:
            snapshot[name] = captured
    return snapshot


@torch.no_grad()
def restore_selected_gaussian_state(
    gaussian_params: Mapping[str, Mapping[str, Any]],
    selection: torch.Tensor,
    snapshot: Mapping[str, Mapping[str, tuple[str, torch.Tensor]]],
) -> None:
    """Restore a snapshot without replacing optimizer-owned tensor objects."""
    for name, captured in snapshot.items():
        state = gaussian_params[name]
        for key, (capture_mode, saved) in captured.items():
            value = state[key]
            if capture_mode == "selected":
                value[selection] = saved
            else:
                value.copy_(saved)


@torch.no_grad()
def overwrite_selected_gaussian_snapshot(
    gaussian_params: Mapping[str, Mapping[str, Any]],
    selection: torch.Tensor,
    snapshot: Mapping[str, Mapping[str, tuple[str, torch.Tensor]]],
) -> None:
    """Reuse snapshot storage for the latest accepted candidate."""
    for name, captured in snapshot.items():
        state = gaussian_params[name]
        for key, (capture_mode, saved) in captured.items():
            value = state[key]
            if capture_mode == "selected":
                saved.copy_(value.detach()[selection])
            else:
                saved.copy_(value.detach())


@torch.no_grad()
def overwrite_selected_gaussian_value_snapshot(
    gaussian_params: Mapping[str, Mapping[str, Any]],
    selection: torch.Tensor,
    snapshot: Mapping[str, Mapping[str, tuple[str, torch.Tensor]]],
) -> None:
    """Retain candidate values while leaving pre-refinement optimizer state intact."""
    for name, captured in snapshot.items():
        if "val" not in captured:
            continue
        state = gaussian_params[name]
        capture_mode, saved = captured["val"]
        value = state["val"]
        if capture_mode == "selected":
            saved.copy_(value.detach()[selection])
        else:
            saved.copy_(value.detach())


def scale_gaussian_gradients(
    gaussian_params: Mapping[str, Mapping[str, Any]],
    scales: Mapping[str, float] = DEFAULT_PARAMETER_GRADIENT_SCALES,
) -> None:
    """Apply C-specific conservative steps while retaining every Gaussian group."""
    for name, state in gaussian_params.items():
        value = state.get("val")
        if not torch.is_tensor(value) or value.grad is None:
            continue
        value.grad.mul_(float(scales.get(name, 1.0)))


@torch.no_grad()
def rendering_response_gap(
    residual: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    min_selectivity: float = 0.18,
) -> dict[str, Any]:
    """Measure spatially selective underfit using an existing rendered view."""
    response = residual.detach().float().abs()
    if response.ndim == 3:
        response = response.mean(dim=0)
    if response.ndim != 2 or response.numel() == 0:
        return {
            "coverage_deficit": 0.0,
            "selectivity": 0.0,
            "selective": False,
            "threshold": 0.0,
        }

    structure = torch.zeros_like(response)
    if response.shape[1] > 1:
        structure[:, 1:] += (response[:, 1:] - response[:, :-1]).abs()
    if response.shape[0] > 1:
        structure[1:, :] += (response[1:, :] - response[:-1, :]).abs()
    response = 0.65 * response + 0.35 * structure

    if valid_mask is not None:
        mask = valid_mask.detach().bool()
        while mask.ndim > 2:
            mask = mask.any(dim=0)
        if mask.shape != response.shape:
            return {
                "coverage_deficit": 0.0,
                "selectivity": 0.0,
                "selective": False,
                "threshold": 0.0,
            }
        values = response[mask]
    else:
        values = response.reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() < 4:
        return {
            "coverage_deficit": 0.0,
            "selectivity": 0.0,
            "selective": False,
            "threshold": 0.0,
        }

    normalized = (values / values.mean().clamp_min(1e-8)).clamp(0.25, 4.0)
    q50, q75, q90 = torch.quantile(
        normalized,
        torch.tensor(
            [0.50, 0.75, 0.90],
            device=normalized.device,
            dtype=normalized.dtype,
        ),
    )
    mad = torch.median((normalized - q50).abs())
    selectivity = float((q90 - q50).item())
    selective = bool(selectivity >= float(min_selectivity))
    threshold = torch.maximum(q75, q50 + 1.5 * mad)
    coverage_deficit = (
        float((normalized > threshold).float().mean().item()) if selective else 0.0
    )
    return {
        "coverage_deficit": coverage_deficit,
        "selectivity": selectivity,
        "selective": selective,
        "threshold": float(threshold.item()),
    }


def refinement_time_budget_seconds(
    cumulative_base_runtime_seconds: float,
    cumulative_refinement_runtime_seconds: float,
    *,
    target_ratio: float = 0.08,
) -> float:
    target = max(0.0, float(target_ratio)) * max(
        0.0, float(cumulative_base_runtime_seconds)
    )
    return max(0.0, target - max(0.0, float(cumulative_refinement_runtime_seconds)))


def select_refinement_reference_indices(
    keyframe_is_test: list[bool] | tuple[bool, ...],
    *,
    current_index: int,
    max_references: int = 2,
) -> list[int]:
    """Select recent historical training views without touching RNG state."""
    limit = max(0, min(int(current_index), len(keyframe_is_test)))
    selected = [
        index
        for index in range(limit - 1, -1, -1)
        if not bool(keyframe_is_test[index])
    ][: max(0, int(max_references))]
    return sorted(selected)


def refinement_reference_guard(
    reference: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    candidate: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    *,
    relative_tolerance: float = 0.001,
) -> dict[str, Any]:
    """Reject a current-view update that regresses recent training views."""
    decision = {
        "accepted": False,
        "reason": "",
        "relative_tolerance": float(relative_tolerance),
    }
    if not reference or len(reference) != len(candidate):
        decision["reason"] = "reference_metrics_unavailable"
        return decision

    tolerance = max(0.0, float(relative_tolerance))
    epsilon = 1e-8
    for initial, current in zip(reference, candidate):
        try:
            initial_total = float(initial["total_loss"])
            current_total = float(current["total_loss"])
            initial_rgb = float(initial["rgb_mse"])
            current_rgb = float(current["rgb_mse"])
        except (KeyError, TypeError, ValueError):
            decision["reason"] = "non_finite_reference_metrics"
            return decision
        if not all(
            math.isfinite(value)
            for value in (initial_total, current_total, initial_rgb, current_rgb)
        ):
            decision["reason"] = "non_finite_reference_metrics"
            return decision
        if current_total > initial_total * (1.0 + tolerance) + epsilon:
            decision["reason"] = "reference_total_loss_guard_failed"
            return decision
        if current_rgb > initial_rgb * (1.0 + tolerance) + epsilon:
            decision["reason"] = "reference_rgb_mse_guard_failed"
            return decision

    decision.update({"accepted": True, "reason": "reference_guard_passed"})
    return decision


def _finite_metrics(metrics: Mapping[str, Any]) -> bool:
    try:
        return all(
            math.isfinite(float(metrics[name]))
            for name in ("total_loss", "rgb_mse", "dssim", "depth")
        )
    except (KeyError, TypeError, ValueError):
        return False


def refinement_candidate_acceptance(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    dssim_relative_tolerance: float = 0.002,
    depth_relative_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Check whether an extra-refinement candidate is safe to retain."""
    decision = {
        "accepted": False,
        "reason": "",
        "dssim_relative_tolerance": float(dssim_relative_tolerance),
        "depth_relative_tolerance": float(depth_relative_tolerance),
    }
    if not _finite_metrics(reference):
        decision["reason"] = "non_finite_reference"
        return decision
    if not _finite_metrics(candidate):
        decision["reason"] = "non_finite_candidate"
        return decision

    reference_total = float(reference["total_loss"])
    candidate_total = float(candidate["total_loss"])
    if candidate_total >= reference_total:
        decision["reason"] = "total_loss_not_improved"
        return decision
    if float(candidate["rgb_mse"]) > float(reference["rgb_mse"]):
        decision["reason"] = "rgb_mse_not_improved"
        return decision

    epsilon = 1e-8
    dssim_limit = (
        float(reference["dssim"]) * (1.0 + float(dssim_relative_tolerance))
        + epsilon
    )
    if float(candidate["dssim"]) > dssim_limit:
        decision["reason"] = "dssim_guard_failed"
        return decision

    depth_limit = (
        float(reference["depth"]) * (1.0 + float(depth_relative_tolerance))
        + epsilon
    )
    if float(candidate["depth"]) > depth_limit:
        decision["reason"] = "depth_guard_failed"
        return decision

    decision.update(
        {
            "accepted": True,
            "reason": "candidate_accepted",
            "total_loss_improvement": reference_total - candidate_total,
        }
    )
    return decision
