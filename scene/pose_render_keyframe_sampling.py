from __future__ import annotations

from typing import Any, Sequence

from scene.pose_render_edge_loss import (
    POSE_RENDER_GATE_INFO_KEY,
    POSE_SAFE_DIRECT_DENSITY_MODE,
)


POSE_CONFIDENCE_TEMPORAL_SAMPLING_MODE = "pose_confidence_temporal_v1"
KEYFRAME_SAMPLING_MODES = {POSE_CONFIDENCE_TEMPORAL_SAMPLING_MODE}


def _as_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(float(lower), min(float(upper), float(value)))


def _gate_payload(keyframe_info: dict[str, Any] | None) -> dict[str, Any]:
    info = keyframe_info or {}
    gate: dict[str, Any] = {}
    nested = info.get(POSE_RENDER_GATE_INFO_KEY, None)
    if isinstance(nested, dict):
        gate.update(nested)
    for key in (
        "direct_keyframe_finalized",
        "finalize_pose_risk_reference",
        "pose_risk_high",
        "pose_risk_score",
        "pose_render_risk_high",
        "pose_render_risk_score",
        "semantic_R_t",
        "utility_drift_risk",
        "pose_support_score",
        "match_support_score",
    ):
        if key in info:
            gate[key] = info[key]
    return gate


def _pose_sampling_confidence(
    keyframe_info: dict[str, Any] | None,
    *,
    max_pose_risk: float,
    max_utility_drift: float,
    min_pose_support: float,
    min_match_support: float,
) -> tuple[float, dict[str, Any]]:
    gate = _gate_payload(keyframe_info)
    if not gate:
        return 1.0, {"reason": "missing_pose_render_gate", "high_risk": False}

    pose_risk = _as_float(
        gate.get(
            "pose_render_risk_score",
            gate.get("pose_risk_score", gate.get("semantic_R_t", None)),
        ),
        0.0,
    )
    utility_drift = _as_float(gate.get("utility_drift_risk", None), 0.0)
    pose_support = _as_float(gate.get("pose_support_score", None), 1.0)
    match_support = _as_float(gate.get("match_support_score", None), 1.0)
    pose_risk_high = _as_bool(
        gate.get("pose_render_risk_high", gate.get("pose_risk_high", False)),
        False,
    )
    pose_risk_reference = _as_bool(gate.get("finalize_pose_risk_reference", False), False)
    finalized = _as_bool(gate.get("direct_keyframe_finalized", True), True)

    pose_score_high = bool(pose_risk > float(max_pose_risk))
    utility_high = bool(utility_drift > float(max_utility_drift))
    support_low = bool(
        pose_support < float(min_pose_support)
        or match_support < float(min_match_support)
    )
    high_risk = bool(
        (not finalized)
        or pose_risk_high
        or pose_risk_reference
        or pose_score_high
        or utility_high
        or support_low
    )

    if high_risk:
        confidence = 0.0
    else:
        pose_confidence = 1.0 - _clamp(
            pose_risk / max(float(max_pose_risk), 1e-6)
        )
        drift_confidence = 1.0 - _clamp(
            utility_drift / max(float(max_utility_drift), 1e-6)
        )
        support_confidence = 0.5 * (
            _clamp(pose_support) + _clamp(match_support)
        )
        confidence = (
            0.45 * pose_confidence
            + 0.35 * support_confidence
            + 0.20 * drift_confidence
        )
    debug = {
        "reason": "pose_confidence",
        "high_risk": high_risk,
        "direct_keyframe_finalized": finalized,
        "pose_risk_score": pose_risk,
        "utility_drift_risk": utility_drift,
        "pose_support_score": pose_support,
        "match_support_score": match_support,
        "pose_risk_high": pose_risk_high,
        "pose_risk_score_high": pose_score_high,
        "pose_risk_reference": pose_risk_reference,
        "utility_drift_high": utility_high,
        "support_low": support_low,
    }
    return _clamp(confidence), debug


def pose_render_keyframe_sampling_weights(
    *,
    mode: str | None,
    direct_density_mode: str | None,
    keyframe_infos: Sequence[dict[str, Any] | None],
    min_weight: float = 0.35,
    max_pose_risk: float = 0.38,
    max_utility_drift: float = 0.75,
    min_pose_support: float = 0.35,
    min_match_support: float = 0.35,
) -> tuple[list[float], dict[str, Any]]:
    mode_name = str(mode or "off")
    direct_mode = str(direct_density_mode or "off")
    count = len(keyframe_infos)
    debug: dict[str, Any] = {
        "mode": mode_name,
        "direct_density_mode": direct_mode,
        "applied": False,
        "reason": "",
        "candidate_count": count,
        "min_weight": float(min_weight),
        "max_pose_risk": float(max_pose_risk),
        "max_utility_drift": float(max_utility_drift),
        "min_pose_support": float(min_pose_support),
        "min_match_support": float(min_match_support),
        "high_risk_candidates": 0,
        "missing_pose_render_gate": 0,
    }
    uniform = [1.0 for _ in keyframe_infos]
    if count == 0:
        debug["reason"] = "empty_candidates"
        return [], debug
    if mode_name == "off":
        debug["reason"] = "off"
        return uniform, debug
    if mode_name not in KEYFRAME_SAMPLING_MODES:
        debug["reason"] = "unknown_mode"
        return uniform, debug
    if direct_mode != POSE_SAFE_DIRECT_DENSITY_MODE:
        debug["reason"] = "direct_density_mismatch"
        return uniform, debug

    floor = _clamp(float(min_weight))
    weights: list[float] = []
    confidence_sum = 0.0
    for info in keyframe_infos:
        confidence, item_debug = _pose_sampling_confidence(
            info,
            max_pose_risk=max_pose_risk,
            max_utility_drift=max_utility_drift,
            min_pose_support=min_pose_support,
            min_match_support=min_match_support,
        )
        if item_debug.get("reason") == "missing_pose_render_gate":
            debug["missing_pose_render_gate"] = int(
                debug.get("missing_pose_render_gate", 0)
            ) + 1
            weight = 1.0
        else:
            if bool(item_debug.get("high_risk", False)):
                debug["high_risk_candidates"] = int(
                    debug.get("high_risk_candidates", 0)
                ) + 1
                weight = floor
            else:
                weight = 1.0
        weights.append(_clamp(weight, floor, 1.0))
        confidence_sum += float(confidence)

    total_weight = sum(weights)
    debug.update(
        {
            "applied": True,
            "reason": "pose_confidence_temporal",
            "weight_min": min(weights),
            "weight_max": max(weights),
            "weight_mean": total_weight / float(count),
            "confidence_mean": confidence_sum / float(count),
        }
    )
    return weights, debug


def choose_pose_render_keyframe_id(
    *,
    candidate_ids: Sequence[int],
    keyframe_infos: Sequence[dict[str, Any] | None],
    mode: str | None,
    direct_density_mode: str | None,
    rng: Any = None,
    min_weight: float = 0.35,
    max_pose_risk: float = 0.38,
    max_utility_drift: float = 0.75,
    min_pose_support: float = 0.35,
    min_match_support: float = 0.35,
) -> tuple[int, dict[str, Any]]:
    if rng is None:
        import numpy as np

        rng = np.random
    weights, debug = pose_render_keyframe_sampling_weights(
        mode=mode,
        direct_density_mode=direct_density_mode,
        keyframe_infos=keyframe_infos,
        min_weight=min_weight,
        max_pose_risk=max_pose_risk,
        max_utility_drift=max_utility_drift,
        min_pose_support=min_pose_support,
        min_match_support=min_match_support,
    )
    if len(candidate_ids) == 0:
        raise ValueError("candidate_ids must not be empty")
    if len(candidate_ids) != len(weights):
        debug.update({"applied": False, "reason": "candidate_weight_mismatch"})
        chosen = int(rng.choice(candidate_ids))
        debug["chosen_keyframe_id"] = chosen
        return chosen, debug

    uniform_weights = bool(max(weights) - min(weights) <= 1e-12)
    if uniform_weights:
        probabilities = [1.0 / float(len(weights)) for _ in weights]
        chosen = int(rng.choice(candidate_ids))
    else:
        total = float(sum(weights))
        if total <= 1e-12:
            probabilities = [1.0 / float(len(weights)) for _ in weights]
        else:
            probabilities = [float(weight) / total for weight in weights]
        chosen = int(rng.choice(candidate_ids, p=probabilities))
    debug.update(
        {
            "chosen_keyframe_id": chosen,
            "used_weighted_probabilities": not uniform_weights,
            "chosen_probability": probabilities[list(candidate_ids).index(chosen)]
            if chosen in candidate_ids
            else 0.0,
            "probability_min": min(probabilities),
            "probability_max": max(probabilities),
        }
    )
    return chosen, debug
