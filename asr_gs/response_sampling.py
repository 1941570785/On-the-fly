from __future__ import annotations

from typing import Any, Mapping

from asr_gs.config import ResponseSamplingConfig


def sampling_scene_guard(
    stats: Mapping[str, object],
    config: ResponseSamplingConfig,
) -> dict[str, object]:
    """Summarize whether prior sampling interventions remained useful."""
    evaluated = int(stats.get("response_evaluated", 0) or 0)
    bad = int(stats.get("response_bad", 0) or 0)
    bad_ratio = bad / max(evaluated, 1)
    disabled = (
        evaluated >= config.guard_min_evaluated
        and bad_ratio >= config.guard_max_bad_ratio
    )
    return {
        "disabled": disabled,
        "reason": "response_bad_ratio" if disabled else "active",
        "response_evaluated": evaluated,
        "response_bad": bad,
        "response_bad_ratio": bad_ratio,
    }


def sampling_response_verdict(
    response: Mapping[str, object],
    config: ResponseSamplingConfig,
) -> dict[str, object] | None:
    """Classify the rendering response after one sampling intervention."""
    observations = int(response.get("observations", 0) or 0)
    if observations < config.response_min_observations:
        return None
    first = float(response.get("first_rgb_mse", 0.0) or 0.0)
    latest = float(response.get("latest_rgb_mse", 0.0) or 0.0)
    improvement = float(response.get("relative_improvement", 0.0) or 0.0)
    degraded = (
        first > 0.0
        and latest
        > first * (1.0 + config.response_max_degradation_ratio)
    )
    low = improvement < config.response_min_improvement
    return {
        "verdict": "bad" if degraded or low else "good",
        "observations": observations,
        "relative_improvement": improvement,
        "first_rgb_mse": first,
        "latest_rgb_mse": latest,
        "degraded": degraded,
        "low_improvement": low,
    }


def response_guided_sampling_probability(
    base_probability: Any,
    residual_edge_response: Any,
    config: ResponseSamplingConfig,
    *,
    coverage_deficit: float = 1.0,
    scene_guard: Mapping[str, object] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Reallocate baseline Bernoulli mass using rendering response."""
    debug: dict[str, Any] = {
        "enabled": bool(config.enabled),
        "applied": False,
        "reason": "",
        "alpha": 0.0,
        "selectivity": 0.0,
        "coverage_deficit": float(coverage_deficit),
    }
    if not config.enabled:
        debug["reason"] = "disabled"
        return base_probability, debug
    if bool((scene_guard or {}).get("disabled", False)):
        debug["reason"] = "scene_guard_disabled"
        debug["scene_guard"] = dict(scene_guard or {})
        return base_probability, debug
    if float(coverage_deficit) < config.min_coverage_deficit:
        debug["reason"] = "coverage_sufficient"
        return base_probability, debug
    if base_probability.numel() == 0 or residual_edge_response.numel() == 0:
        debug["reason"] = "empty"
        return base_probability, debug
    if tuple(base_probability.shape) != tuple(residual_edge_response.shape):
        debug["reason"] = "shape_mismatch"
        return base_probability, debug

    base = base_probability.clamp_min(0.0)
    mass = base.sum()
    mass_value = float(mass.detach().cpu().item())
    debug["mass_before"] = mass_value
    if mass_value <= 1e-12:
        debug["reason"] = "zero_mass"
        return base_probability, debug

    guide = residual_edge_response.detach().clamp_min(0.0)
    guide = (guide / guide.mean().clamp_min(1e-6)).clamp(
        float(config.guide_min),
        float(config.guide_max),
    )
    flat = guide.flatten()
    q50 = flat.quantile(0.50)
    q90 = flat.quantile(0.90)
    selectivity = float((q90 - q50).detach().cpu().item())
    debug.update(
        {
            "selectivity": selectivity,
            "guide_q50": float(q50.detach().cpu().item()),
            "guide_q90": float(q90.detach().cpu().item()),
        }
    )
    if selectivity < float(config.min_selectivity):
        debug["reason"] = "low_selectivity"
        return base_probability, debug

    effective_alpha = max(0.0, min(float(config.alpha), 0.25))
    effective_alpha *= max(0.25, min(selectivity / 0.75, 1.0))
    weights = (1.0 + effective_alpha * (guide - 1.0)).clamp(
        float(config.weight_min),
        float(config.weight_max),
    )
    weighted = base * weights
    normalized = weighted * (mass / weighted.sum().clamp_min(1e-12))
    probability = normalized.clamp(0.0, 1.0)

    debug.update(
        {
            "applied": True,
            "reason": "response_guided",
            "alpha": effective_alpha,
            "mass_before_clipping": float(
                normalized.sum().detach().cpu().item()
            ),
            "mass_after_clipping": float(
                probability.sum().detach().cpu().item()
            ),
            "clipped_pixels": int(
                (normalized > 1.0).sum().detach().cpu().item()
            ),
        }
    )
    return probability, debug
