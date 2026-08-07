from __future__ import annotations

from typing import Any, Mapping

from asr_gs.config import TransactionalRefinementConfig


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def updated_render_response(
    previous: Mapping[str, Any] | None,
    latest_rgb_mse: float,
    *,
    history_size: int = 4,
    projection_coverage_deficit: float = 0.0,
    representation_coverage_deficit: float = 0.0,
    representation_gap_selectivity: float = 0.0,
) -> dict[str, Any]:
    latest = float(latest_rgb_mse)
    response = previous if isinstance(previous, Mapping) else {}
    observations = int(response.get("observations", 0) or 0) + 1
    first = _as_float(response.get("first_rgb_mse", latest), latest)
    best = min(_as_float(response.get("best_rgb_mse", latest), latest), latest)
    recent = response.get("recent_rgb_mse", [])
    if not isinstance(recent, (list, tuple)):
        recent = []
    bounded_size = max(1, int(history_size))
    values = [
        _as_float(value, latest)
        for value in list(recent)[-(bounded_size - 1) :]
    ]
    values.append(latest)
    denominator = max(abs(first), 1e-8)
    return {
        "observations": observations,
        "first_rgb_mse": first,
        "latest_rgb_mse": latest,
        "best_rgb_mse": best,
        "relative_improvement": (first - latest) / denominator,
        "best_relative_improvement": (first - best) / denominator,
        "recent_rgb_mse": values,
        "projection_coverage_deficit": float(
            projection_coverage_deficit
        ),
        "representation_coverage_deficit": float(
            representation_coverage_deficit
        ),
        "representation_gap_selectivity": float(
            representation_gap_selectivity
        ),
    }


def extra_refinement_decision(
    *,
    response: Mapping[str, Any] | None,
    base_iterations: int,
    is_test: bool,
    already_finalized: bool,
    config: TransactionalRefinementConfig,
) -> dict[str, Any]:
    """Select a bounded Gaussian-only transaction from rendering response."""
    decision: dict[str, Any] = {
        "enabled": bool(config.enabled),
        "applied": False,
        "reason": "",
        "base_iterations": int(base_iterations),
        "extra_iterations": 0,
        "fraction": float(config.fraction),
        "max_extra": int(config.max_extra_iterations),
    }
    if not config.enabled:
        decision["reason"] = "disabled"
        return decision
    if is_test:
        decision["reason"] = "test_frame"
        return decision
    if already_finalized:
        decision["reason"] = "already_finalized"
        return decision
    if base_iterations <= 0:
        decision["reason"] = "zero_base_iterations"
        return decision

    values = response if isinstance(response, Mapping) else {}
    observations = int(values.get("observations", 0) or 0)
    first = _as_float(values.get("first_rgb_mse", 0.0))
    latest = _as_float(values.get("latest_rgb_mse", 0.0))
    best = _as_float(values.get("best_rgb_mse", latest), latest)
    improvement = _as_float(values.get("relative_improvement", 0.0))
    projection_coverage_deficit = _as_float(
        values.get("projection_coverage_deficit", 0.0)
    )
    representation_coverage_deficit = _as_float(
        values.get("representation_coverage_deficit", 0.0)
    )
    coverage_deficit = max(
        projection_coverage_deficit,
        representation_coverage_deficit,
    )
    decision.update(
        {
            "observations": observations,
            "first_rgb_mse": first,
            "latest_rgb_mse": latest,
            "relative_improvement": improvement,
            "coverage_deficit": coverage_deficit,
            "projection_coverage_deficit": projection_coverage_deficit,
            "representation_coverage_deficit": (
                representation_coverage_deficit
            ),
        }
    )

    if observations < int(config.min_response_observations):
        decision["reason"] = "response_pending"
        return decision
    if coverage_deficit < float(config.min_coverage_deficit):
        decision["reason"] = "coverage_sufficient"
        return decision
    if coverage_deficit > float(config.max_coverage_deficit):
        decision["reason"] = "coverage_gap_too_large"
        return decision
    if first > 0.0 and latest > first * 1.02:
        decision["reason"] = "response_degraded"
        return decision
    if improvement < float(config.min_relative_improvement):
        decision["reason"] = "response_low"
        return decision

    recent = values.get("recent_rgb_mse", [])
    if isinstance(recent, (list, tuple)) and len(recent) >= 4:
        recent_values = [_as_float(value) for value in recent[-4:]]
        decreases = sum(
            current < previous
            for previous, current in zip(recent_values, recent_values[1:])
        )
        rebound = max(0.0, (latest - best) / max(abs(best), 1e-8))
        decision.update(
            {
                "recent_decreases": int(decreases),
                "rebound_ratio": rebound,
            }
        )
        if (
            decreases < 2
            or rebound > float(config.max_response_rebound_ratio)
        ):
            decision["reason"] = "response_unstable"
            return decision

    scale = max(
        0.35,
        min(improvement / float(config.response_scale_reference), 1.0),
    )
    requested = round(int(base_iterations) * float(config.fraction) * scale)
    extra = max(
        1,
        min(int(config.max_extra_iterations), int(requested)),
    )
    decision.update(
        {
            "applied": True,
            "reason": "transaction_requested",
            "extra_iterations": extra,
            "response_scale": scale,
        }
    )
    return decision
