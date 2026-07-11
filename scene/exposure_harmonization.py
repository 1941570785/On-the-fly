from __future__ import annotations

from typing import Any


EXPOSURE_HARMONIZATION_OFF = "off"
EXPOSURE_HARMONIZATION_NEIGHBOR_AVERAGE = "neighbor_average_v1"
EXPOSURE_HARMONIZATION_SOURCE_TIME_INTERP = "source_time_interp_v1"
EXPOSURE_HARMONIZATION_SOURCE_TIME_GUARDED = "source_time_guarded_v1"
EXPOSURE_HARMONIZATION_SOURCE_TIME_ADAPTIVE = "source_time_adaptive_v2"
EXPOSURE_HARMONIZATION_DARK_SCENE_OFF_GUARDED = "dark_scene_off_guarded_v1"


def _info(keyframe: Any) -> dict[str, Any]:
    data = getattr(keyframe, "info", None)
    return data if isinstance(data, dict) else {}


def _source_frame_id(keyframe: Any, fallback: int) -> int:
    info = _info(keyframe)
    for key in ("_paper_aligned_source_frame_id", "source_frame_id"):
        if key in info:
            try:
                return int(info[key])
            except (TypeError, ValueError):
                break
    return int(fallback)


def _is_test_keyframe(keyframe: Any) -> bool:
    return bool(_info(keyframe).get("is_test", False))


def adaptive_source_time_exposure_scene_decision(
    *,
    train_keyframes: int,
    test_keyframes: int,
    texture_sampling_events: int,
    texture_sampling_applied: int,
    coverage_deficit_mean: float | None,
    min_train_test_ratio: float = 1.0,
    max_train_test_ratio: float = 4.0,
    max_sampling_applied_ratio: float = 0.02,
    min_coverage_deficit: float = 0.04,
    max_coverage_deficit: float = 0.075,
) -> dict[str, Any]:
    test_count = max(int(test_keyframes), 0)
    train_count = max(int(train_keyframes), 0)
    events = max(int(texture_sampling_events), 0)
    applied = max(int(texture_sampling_applied), 0)
    train_test_ratio = float(train_count) / float(max(test_count, 1))
    sampling_applied_ratio = float(applied) / float(max(events, 1))
    coverage_missing = coverage_deficit_mean is None
    coverage_value = 0.0 if coverage_missing else float(coverage_deficit_mean)

    selected_mode = EXPOSURE_HARMONIZATION_NEIGHBOR_AVERAGE
    reason = "density_or_sampling_guard"
    if test_count <= 0 or train_count <= 0:
        reason = "insufficient_keyframes"
    elif coverage_missing:
        reason = "missing_coverage_deficit"
    elif not (min_train_test_ratio <= train_test_ratio <= max_train_test_ratio):
        reason = "train_test_density_out_of_range"
    elif sampling_applied_ratio > max_sampling_applied_ratio:
        reason = "texture_sampling_pressure_high"
    elif not (min_coverage_deficit <= coverage_value <= max_coverage_deficit):
        reason = "coverage_deficit_out_of_range"
    else:
        selected_mode = EXPOSURE_HARMONIZATION_SOURCE_TIME_INTERP
        reason = "moderate_density_coverage_gap"

    return {
        "mode": EXPOSURE_HARMONIZATION_SOURCE_TIME_ADAPTIVE,
        "selected_mode": selected_mode,
        "reason": reason,
        "train_keyframes": train_count,
        "test_keyframes": test_count,
        "train_test_ratio": train_test_ratio,
        "texture_sampling_events": events,
        "texture_sampling_applied": applied,
        "texture_sampling_applied_ratio": sampling_applied_ratio,
        "coverage_deficit_mean": coverage_value,
        "coverage_deficit_missing": bool(coverage_missing),
        "min_train_test_ratio": float(min_train_test_ratio),
        "max_train_test_ratio": float(max_train_test_ratio),
        "max_sampling_applied_ratio": float(max_sampling_applied_ratio),
        "min_coverage_deficit": float(min_coverage_deficit),
        "max_coverage_deficit": float(max_coverage_deficit),
    }


def dark_scene_test_exposure_scene_decision(
    training_background_stats: dict[str, Any] | None,
    *,
    max_running_mean_for_off: float = 0.40,
) -> dict[str, Any]:
    stats = training_background_stats if isinstance(training_background_stats, dict) else {}
    mode = str(stats.get("mode", ""))
    observations = int(stats.get("dark_scene_observations", 0) or 0)
    mask_blocked = int(stats.get("dark_scene_mask_blocked", 0) or 0)
    running_mean_raw = stats.get("dark_scene_running_mean", None)
    running_mean_missing = running_mean_raw is None
    running_mean = 0.0 if running_mean_missing else float(running_mean_raw)

    selected_mode = EXPOSURE_HARMONIZATION_NEIGHBOR_AVERAGE
    reason = "unsupported_background_mode"
    if mode != "dark_scene_fixed_black_v1":
        reason = "unsupported_background_mode"
    elif mask_blocked > 0:
        reason = "masked_scene"
    elif observations <= 0 or running_mean_missing:
        reason = "missing_brightness_observation"
    elif running_mean <= float(max_running_mean_for_off):
        selected_mode = EXPOSURE_HARMONIZATION_OFF
        reason = "dim_unmasked_scene"
    else:
        reason = "brightness_out_of_range"

    return {
        "mode": EXPOSURE_HARMONIZATION_DARK_SCENE_OFF_GUARDED,
        "selected_mode": selected_mode,
        "reason": reason,
        "training_background_mode": mode,
        "dark_scene_observations": observations,
        "dark_scene_mask_blocked": mask_blocked,
        "dark_scene_running_mean": running_mean,
        "dark_scene_running_mean_missing": bool(running_mean_missing),
        "max_running_mean_for_off": float(max_running_mean_for_off),
    }


def _nearest_non_test_indices(keyframes: list[Any], index: int) -> tuple[int | None, int | None]:
    prev_index = None
    for candidate in range(index - 1, -1, -1):
        if not _is_test_keyframe(keyframes[candidate]):
            prev_index = candidate
            break

    next_index = None
    for candidate in range(index + 1, len(keyframes)):
        if not _is_test_keyframe(keyframes[candidate]):
            next_index = candidate
            break
    return prev_index, next_index


def _baseline_neighbor_average_exposure(keyframes: list[Any], index: int):
    if len(keyframes) <= 1:
        return keyframes[index].exposure
    prev_index = index - 1 if index != 0 else min(1, len(keyframes) - 1)
    next_index = index + 1 if index != len(keyframes) - 1 else max(0, len(keyframes) - 2)
    return (keyframes[prev_index].exposure + keyframes[next_index].exposure) / 2


def _max_abs_exposure_delta(left: Any, right: Any) -> float:
    try:
        delta = left - right
    except Exception:
        try:
            return abs(float(left) - float(right))
        except Exception:
            return float("inf")

    if hasattr(delta, "detach"):
        delta = delta.detach()
    if hasattr(delta, "abs"):
        delta = delta.abs()
    else:
        try:
            return abs(float(delta))
        except Exception:
            return float("inf")

    if hasattr(delta, "max"):
        delta = delta.max()
    if hasattr(delta, "item"):
        return abs(float(delta.item()))
    try:
        return abs(float(delta))
    except Exception:
        return float("inf")


def harmonized_test_exposure(
    keyframes: list[Any],
    index: int,
    *,
    mode: str | None = EXPOSURE_HARMONIZATION_NEIGHBOR_AVERAGE,
    max_neighbor_exposure_delta: float | None = None,
):
    mode_name = str(mode or EXPOSURE_HARMONIZATION_NEIGHBOR_AVERAGE)
    if mode_name == EXPOSURE_HARMONIZATION_OFF:
        return keyframes[index].exposure
    source_time_modes = {
        EXPOSURE_HARMONIZATION_SOURCE_TIME_INTERP,
        EXPOSURE_HARMONIZATION_SOURCE_TIME_GUARDED,
    }
    if mode_name not in source_time_modes:
        return _baseline_neighbor_average_exposure(keyframes, index)

    prev_index, next_index = _nearest_non_test_indices(keyframes, index)
    if prev_index is None and next_index is None:
        return keyframes[index].exposure
    if prev_index is None:
        return keyframes[next_index].exposure
    if next_index is None:
        return keyframes[prev_index].exposure

    prev_source = _source_frame_id(keyframes[prev_index], prev_index)
    next_source = _source_frame_id(keyframes[next_index], next_index)
    test_source = _source_frame_id(keyframes[index], index)
    if next_source == prev_source:
        return _baseline_neighbor_average_exposure(keyframes, index)

    if mode_name == EXPOSURE_HARMONIZATION_SOURCE_TIME_GUARDED:
        max_delta = float(
            max_neighbor_exposure_delta
            if max_neighbor_exposure_delta is not None
            else 0.0
        )
        if max_delta > 0.0:
            exposure_delta = _max_abs_exposure_delta(
                keyframes[prev_index].exposure,
                keyframes[next_index].exposure,
            )
            if exposure_delta > max_delta:
                return _baseline_neighbor_average_exposure(keyframes, index)

    weight_next = (float(test_source) - float(prev_source)) / (
        float(next_source) - float(prev_source)
    )
    weight_next = max(0.0, min(1.0, weight_next))
    return (
        keyframes[prev_index].exposure * (1.0 - weight_next)
        + keyframes[next_index].exposure * weight_next
    )
