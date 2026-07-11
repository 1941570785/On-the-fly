from __future__ import annotations

from typing import Any

TEXTURE_SAMPLING_RESIDUAL_EDGE_MODE = "residual_edge_v1"
TEXTURE_SAMPLING_RESPONSE_GUARD_MODE = "residual_edge_response_guard_v2"
TEXTURE_SAMPLING_RESPONSE_GUARD_BYPASS_MODE = "residual_edge_response_guard_v4"
TEXTURE_SAMPLING_RESPONSE_GUARD_NON_DARK_MODE = (
    "residual_edge_response_guard_non_dark_v5"
)
TEXTURE_SAMPLING_RESPONSE_GUARD_MASK_CONSERVATIVE_MODE = (
    "residual_edge_response_guard_mask_conservative_v6"
)
TEXTURE_SAMPLING_SCENE_GUARD_KEY = (
    "_paper_aligned_pose_render_texture_sampling_scene_guard"
)
TEXTURE_SAMPLING_COVERAGE_KEY = "_paper_aligned_pose_render_texture_sampling_coverage"
SCENE_LOW_RESPONSE_KEY = "_paper_aligned_pose_render_scene_low_response"


def _to_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def pose_render_texture_sampling_enabled(
    mode: str | None,
    direct_density_mode: str | None,
    keyframe_info: dict[str, Any] | None = None,
    render_frame_policy: str | None = None,
) -> bool:
    mode_name = str(mode or "off")
    if mode_name not in {
        TEXTURE_SAMPLING_RESIDUAL_EDGE_MODE,
        TEXTURE_SAMPLING_RESPONSE_GUARD_MODE,
        TEXTURE_SAMPLING_RESPONSE_GUARD_BYPASS_MODE,
        TEXTURE_SAMPLING_RESPONSE_GUARD_NON_DARK_MODE,
        TEXTURE_SAMPLING_RESPONSE_GUARD_MASK_CONSERVATIVE_MODE,
    }:
        return False
    signal_enabled = (
        str(direct_density_mode or "off") == "pose_safe_streaming_memory_v1"
        or str(render_frame_policy or "off") == "baseline_keyframe_lock_v1"
        or mode_name
        in {
            TEXTURE_SAMPLING_RESPONSE_GUARD_MODE,
            TEXTURE_SAMPLING_RESPONSE_GUARD_BYPASS_MODE,
            TEXTURE_SAMPLING_RESPONSE_GUARD_NON_DARK_MODE,
            TEXTURE_SAMPLING_RESPONSE_GUARD_MASK_CONSERVATIVE_MODE,
        }
    )
    if not signal_enabled:
        return False
    info = keyframe_info or {}
    if bool(info.get("is_test", False)):
        return False
    return True


def _response_guard_allows_sampling(
    mode: str | None,
    keyframe_info: dict[str, Any] | None,
    debug: dict[str, Any],
) -> bool:
    mode_name = str(mode or "off")
    if mode_name not in {
        TEXTURE_SAMPLING_RESPONSE_GUARD_MODE,
        TEXTURE_SAMPLING_RESPONSE_GUARD_BYPASS_MODE,
        TEXTURE_SAMPLING_RESPONSE_GUARD_NON_DARK_MODE,
        TEXTURE_SAMPLING_RESPONSE_GUARD_MASK_CONSERVATIVE_MODE,
    }:
        return True
    info = keyframe_info or {}
    training_background = info.get("_paper_aligned_training_background", None)
    if mode_name in {
        TEXTURE_SAMPLING_RESPONSE_GUARD_NON_DARK_MODE,
        TEXTURE_SAMPLING_RESPONSE_GUARD_MASK_CONSERVATIVE_MODE,
    } and isinstance(training_background, dict):
        dark_decision = str(training_background.get("dark_scene_decision", ""))
        mask_blocked = int(_to_float(training_background.get("dark_scene_mask_blocked", 0), 0.0))
        dark_observations = int(
            _to_float(training_background.get("dark_scene_observations", 0), 0.0)
        )
        debug["training_background_mode"] = str(training_background.get("mode", ""))
        debug["training_background_dark_scene_decision"] = dark_decision
        debug["training_background_dark_scene_mask_blocked"] = mask_blocked
        debug["training_background_dark_scene_observations"] = dark_observations
        fixed_black_mode = (
            str(training_background.get("mode", "")) == "dark_scene_fixed_black_v1"
        )
        if (
            mode_name == TEXTURE_SAMPLING_RESPONSE_GUARD_MASK_CONSERVATIVE_MODE
            and fixed_black_mode
            and mask_blocked > 0
            and dark_observations == 0
        ):
            debug["reason"] = "mask_blocked_scene_bypass"
            return False
        if (
            fixed_black_mode
            and dark_decision == "fixed_black"
            and mask_blocked == 0
        ):
            debug["reason"] = "dark_fixed_scene_bypass"
            return False
    scene_low_response = (
        info.get(SCENE_LOW_RESPONSE_KEY, None)
        if mode_name != TEXTURE_SAMPLING_RESPONSE_GUARD_MODE
        else None
    )
    if isinstance(scene_low_response, dict):
        debug["scene_low_response_disabled"] = bool(
            scene_low_response.get("disabled", False)
        )
        debug["scene_low_response_reason"] = str(
            scene_low_response.get("reason", "")
        )
        if bool(scene_low_response.get("disabled", False)):
            debug["reason"] = "scene_low_response_latched"
            return False
    scene_guard = info.get(TEXTURE_SAMPLING_SCENE_GUARD_KEY, None)
    if not isinstance(scene_guard, dict):
        debug["scene_guard_disabled"] = False
        debug["scene_guard_reason"] = "uninitialized"
        return True
    disabled = bool(scene_guard.get("disabled", False))
    coverage_bypass_latched = bool(scene_guard.get("coverage_bypass_latched", False))
    debug["scene_guard_disabled"] = disabled
    debug["coverage_bypass_latched"] = coverage_bypass_latched
    debug["scene_guard_reason"] = str(scene_guard.get("reason", ""))
    debug["scene_guard_response_evaluated"] = int(
        _to_float(scene_guard.get("response_evaluated", 0), 0.0)
    )
    debug["scene_guard_response_bad_ratio"] = _to_float(
        scene_guard.get("response_bad_ratio", 0.0),
        0.0,
    )
    scene_events = int(_to_float(scene_guard.get("events", 0), 0.0))
    sampling_applied_ratio = _to_float(
        scene_guard.get("sampling_applied_ratio", 0.0),
        0.0,
    )
    debug["scene_guard_events"] = scene_events
    debug["sampling_applied_ratio"] = sampling_applied_ratio
    max_sampling_applied_ratio = 0.40
    debug["max_sampling_applied_ratio"] = max_sampling_applied_ratio
    if disabled:
        debug["reason"] = "scene_guard_disabled"
        return False
    if (
        mode_name == TEXTURE_SAMPLING_RESPONSE_GUARD_BYPASS_MODE
        and coverage_bypass_latched
    ):
        debug["reason"] = "coverage_bypass_latched"
        return False
    if (
        mode_name != TEXTURE_SAMPLING_RESPONSE_GUARD_MODE
        and scene_events >= 20
        and sampling_applied_ratio >= max_sampling_applied_ratio
    ):
        debug["reason"] = "scene_sampling_quota"
        return False
    coverage = info.get(TEXTURE_SAMPLING_COVERAGE_KEY, None)
    min_coverage_deficit = _to_float(
        scene_guard.get("min_coverage_deficit", 0.0),
        0.0,
    )
    if isinstance(coverage, dict):
        coverage_deficit = _to_float(
            coverage.get("coverage_deficit", coverage.get("deficit", 1.0)),
            1.0,
        )
        debug["coverage"] = _to_float(coverage.get("coverage", 0.0), 0.0)
        debug["coverage_deficit"] = coverage_deficit
        debug["min_coverage_deficit"] = min_coverage_deficit
        if coverage_deficit < min_coverage_deficit:
            debug["reason"] = "coverage_sufficient"
            return False
    return True


def residual_edge_guided_sampling_probability(
    sample_proba: Any,
    residual_edge: Any,
    *,
    mode: str | None,
    direct_density_mode: str | None,
    render_frame_policy: str | None = None,
    keyframe_info: dict[str, Any] | None = None,
    alpha: float = 0.12,
    min_selectivity: float = 0.18,
) -> tuple[Any, dict[str, Any]]:
    debug: dict[str, Any] = {
        "mode": str(mode or "off"),
        "direct_density_mode": str(direct_density_mode or "off"),
        "render_frame_policy": str(render_frame_policy or "off"),
        "applied": False,
        "reason": "",
        "alpha": 0.0,
        "selectivity": 0.0,
    }
    if not pose_render_texture_sampling_enabled(
        mode,
        direct_density_mode,
        keyframe_info,
        render_frame_policy=render_frame_policy,
    ):
        debug["reason"] = "disabled"
        return sample_proba, debug
    if not _response_guard_allows_sampling(mode, keyframe_info, debug):
        return sample_proba, debug
    if sample_proba.numel() == 0 or residual_edge.numel() == 0:
        debug["reason"] = "empty"
        return sample_proba, debug
    if tuple(sample_proba.shape) != tuple(residual_edge.shape):
        debug["reason"] = "shape_mismatch"
        return sample_proba, debug

    base = sample_proba.clamp_min(0.0)
    budget = base.sum()
    if float(budget.detach().cpu().item()) <= 1e-12:
        debug["reason"] = "zero_budget"
        return sample_proba, debug

    guide = residual_edge.detach().clamp_min(0.0)
    mean = guide.mean().clamp_min(1e-6)
    guide = (guide / mean).clamp(0.25, 4.0)
    flat = guide.flatten()
    q50 = flat.quantile(0.50)
    q90 = flat.quantile(0.90)
    selectivity = float((q90 - q50).detach().cpu().item())
    debug["selectivity"] = selectivity
    if selectivity < float(min_selectivity):
        debug["reason"] = "low_selectivity"
        return sample_proba, debug

    effective_alpha = max(0.0, min(float(alpha), 0.25))
    effective_alpha *= max(0.25, min(selectivity / 0.75, 1.0))
    weights = (1.0 + effective_alpha * (guide - 1.0)).clamp(0.55, 1.75)
    guided = base * weights
    guided_sum = guided.sum().clamp_min(1e-12)
    guided = guided * (budget / guided_sum)

    debug.update(
        {
            "applied": True,
            "reason": "residual_edge_guided",
            "alpha": float(effective_alpha),
            "budget_before": float(budget.detach().cpu().item()),
            "budget_after": float(guided.sum().detach().cpu().item()),
            "guide_q50": float(q50.detach().cpu().item()),
            "guide_q90": float(q90.detach().cpu().item()),
        }
    )
    return guided, debug
