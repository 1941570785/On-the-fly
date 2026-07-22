from __future__ import annotations

import json
import math
import statistics
from collections import deque
from pathlib import Path
from typing import Any

from paper_aligned_policy.viewpoint_coverage import rotation_degrees_between


POSE_INITIALIZATION_RISK_MODES = {
    "off",
    "observe_v1",
    "isolate_v1",
    "verify_v1",
    "verify_v2",
}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _clamp01(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, _as_float(value, default)))


def _normalized_shortfall(value: float, target: float) -> float:
    return 1.0 - _clamp01(float(value) / max(float(target), 1e-6))


def _history_pose(item: Any) -> Any:
    if isinstance(item, dict):
        return item.get("Rt", item.get("pose", item.get("current_Rt")))
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        return item[1]
    return None


def _recent_rotation_anomaly(current_Rt: Any, pose_history: Any) -> dict[str, float]:
    poses = [pose for pose in (_history_pose(item) for item in (pose_history or [])) if pose is not None]
    poses = poses[-9:]
    if not poses or current_Rt is None:
        return {
            "current_rotation_step_deg": 0.0,
            "historical_rotation_step_median_deg": 0.0,
            "rotation_anomaly": 0.0,
        }

    current_step = rotation_degrees_between(current_Rt, poses[-1])
    historical_steps = [
        rotation_degrees_between(poses[index], poses[index - 1])
        for index in range(1, len(poses))
    ]
    historical_steps = [step for step in historical_steps if math.isfinite(step)]
    median_step = statistics.median(historical_steps) if historical_steps else current_step
    expected_upper = float(median_step) + 5.0
    anomaly_scale = max(5.0, 1.5 * float(median_step))
    anomaly = _clamp01((float(current_step) - expected_upper) / anomaly_scale)
    return {
        "current_rotation_step_deg": float(current_step),
        "historical_rotation_step_median_deg": float(median_step),
        "rotation_anomaly": float(anomaly),
    }


class PoseInitializationRiskGate:
    """Post-pose risk observer with an optional representation-isolation decision."""

    def __init__(
        self,
        *,
        mode: str = "off",
        absolute_threshold: float = 0.10,
        adaptive_sigma: float = 2.0,
        warmup: int = 8,
        history_size: int = 64,
        cooldown_frames: int = 12,
    ) -> None:
        normalized_mode = str(mode or "off").strip().lower()
        if normalized_mode not in POSE_INITIALIZATION_RISK_MODES:
            raise ValueError(f"Unsupported pose initialization risk mode: {mode}")
        self.mode = normalized_mode
        self.absolute_threshold = _clamp01(absolute_threshold, 0.10)
        self.adaptive_sigma = max(0.0, _as_float(adaptive_sigma, 2.0))
        self.warmup = max(0, _as_int(warmup, 8))
        self.history_size = max(1, _as_int(history_size, 64))
        self.cooldown_frames = max(0, _as_int(cooldown_frames, 12))
        self.risk_history: deque[float] = deque(maxlen=self.history_size)
        self.events: list[dict[str, Any]] = []
        self.last_isolated_frame_id = -1

    def _risk_threshold(self) -> tuple[float, dict[str, float]]:
        if not self.risk_history:
            return self.absolute_threshold, {
                "risk_history_median": 0.0,
                "risk_history_mad": 0.0,
                "adaptive_threshold": self.absolute_threshold,
            }
        values = list(self.risk_history)
        median = float(statistics.median(values))
        mad = float(statistics.median(abs(value - median) for value in values))
        robust_sigma = 1.4826 * mad
        adaptive = min(0.90, median + self.adaptive_sigma * robust_sigma)
        threshold = max(self.absolute_threshold, adaptive)
        return float(threshold), {
            "risk_history_median": median,
            "risk_history_mad": mad,
            "adaptive_threshold": float(adaptive),
        }

    def evaluate(
        self,
        *,
        frame_id: int,
        pose_debug: dict[str, Any] | None,
        viewpoint_scores: dict[str, Any] | None,
        min_num_inliers: int,
        recent_pose_fail_rate: float,
        current_Rt: Any,
        pose_history: Any,
        baseline_selected: bool,
        is_test: bool,
        is_bootstrap: bool,
    ) -> dict[str, Any]:
        pose_debug = dict(pose_debug or {})
        viewpoint_scores = dict(viewpoint_scores or {})
        min_inliers = max(1, _as_int(min_num_inliers, 1))

        match_count = max(0, _as_int(pose_debug.get("match_count_total"), 0))
        correspondences = max(
            match_count,
            _as_int(pose_debug.get("num_2d3d_correspondences"), 0),
        )
        pnp_inliers = max(0, _as_int(pose_debug.get("num_pnp_inliers"), 0))
        miniba_inliers = max(0, _as_int(pose_debug.get("num_miniba_inliers"), 0))
        final_inliers = max(pnp_inliers, miniba_inliers)
        if miniba_inliers > 0:
            final_inliers = miniba_inliers

        absolute_support = _clamp01(final_inliers / max(2.0 * min_inliers, 1.0))
        consensus_ratio = _clamp01(final_inliers / max(float(correspondences), 1.0))
        stage_retention = (
            _clamp01(miniba_inliers / max(float(pnp_inliers), 1.0))
            if pnp_inliers > 0
            else 0.0
        )
        pose_uncertainty = _clamp01(
            0.45 * (1.0 - absolute_support)
            + 0.40 * _normalized_shortfall(consensus_ratio, 0.35)
            + 0.15 * (1.0 - stage_retention)
        )

        grid_coverage = _clamp01(viewpoint_scores.get("inlier_grid_coverage"), 0.0)
        grid_entropy = _clamp01(viewpoint_scores.get("inlier_grid_entropy"), 0.0)
        anchor_health = _clamp01(viewpoint_scores.get("anchor_health_score"), 0.0)
        selected_references = max(
            0,
            _as_int(viewpoint_scores.get("selected_reference_count"), 0),
        )
        reference_health = _clamp01(selected_references / 2.0)
        state_support_gap = _clamp01(
            0.35 * (1.0 - grid_coverage)
            + 0.20 * (1.0 - grid_entropy)
            + 0.30 * (1.0 - anchor_health)
            + 0.15 * (1.0 - reference_health)
        )

        rotation = _recent_rotation_anomaly(current_Rt, pose_history)
        fail_rate = _clamp01(recent_pose_fail_rate)
        temporal_degradation = _clamp01(
            0.55 * fail_rate + 0.45 * rotation["rotation_anomaly"]
        )
        risk_score = _clamp01(
            0.45 * pose_uncertainty
            + 0.35 * state_support_gap
            + 0.20 * temporal_degradation
        )
        threshold, threshold_debug = self._risk_threshold()
        verification_floor = 0.005
        verification_threshold = max(
            verification_floor,
            float(threshold_debug.get("adaptive_threshold", verification_floor)),
        )
        warmed_up = len(self.risk_history) >= self.warmup
        multi_signal_risk = bool(
            pose_uncertainty >= 0.12
            and (state_support_gap >= 0.12 or temporal_degradation >= 0.15)
        )
        severe_pose_risk = pose_uncertainty >= 0.30

        verification_mode = self.mode in {"verify_v1", "verify_v2"}
        test_pose_only_verification = bool(is_test and self.mode == "verify_v2")
        eligible = bool(
            baseline_selected
            and not is_bootstrap
            and (not is_test or test_pose_only_verification)
        )
        risk_evidence_trigger = bool(
            eligible
            and warmed_up
            and risk_score >= threshold
            and (multi_signal_risk or severe_pose_risk)
        )
        risk_trigger = bool(self.mode == "isolate_v1" and risk_evidence_trigger)
        verification_signal = bool(
            pose_uncertainty >= 0.03
            or state_support_gap >= 0.015
            or temporal_degradation >= 0.05
        )
        verification_candidate = bool(
            eligible
            and warmed_up
            and risk_score >= verification_threshold
            and verification_signal
        )
        verification_trigger = bool(verification_mode and verification_candidate)
        cooldown_active = bool(
            self.last_isolated_frame_id >= 0
            and int(frame_id) - self.last_isolated_frame_id < self.cooldown_frames
        )
        isolate = bool(risk_trigger and not cooldown_active)
        if isolate:
            self.last_isolated_frame_id = int(frame_id)
        if not baseline_selected:
            decision = "bypass_not_selected"
        elif is_test and not test_pose_only_verification:
            decision = "bypass_test"
        elif is_bootstrap:
            decision = "bypass_bootstrap"
        elif self.mode == "off":
            decision = "off"
        elif self.mode == "observe_v1":
            decision = "observe"
        elif verification_mode and not warmed_up:
            decision = "verify_warmup"
        elif verification_trigger:
            decision = "verify_candidate"
        elif verification_mode:
            decision = "verify_bypass"
        elif not warmed_up:
            decision = "warmup_admit"
        elif isolate:
            decision = "isolate"
        elif risk_trigger and cooldown_active:
            decision = "cooldown_admit"
        else:
            decision = "admit"

        event = {
            "frame_id": int(frame_id),
            "mode": self.mode,
            "decision": decision,
            "eligible": eligible,
            "isolated": isolate,
            "risk_trigger": risk_trigger,
            "risk_evidence_trigger": risk_evidence_trigger,
            "verification_candidate": verification_candidate,
            "verification_trigger": verification_trigger,
            "verification_signal": verification_signal,
            "verification_risk_threshold": float(verification_threshold),
            "verification_risk_floor": float(verification_floor),
            "verification_attempted": False,
            "verification_accepted": False,
            "cooldown_active": cooldown_active,
            "last_isolated_frame_id": int(self.last_isolated_frame_id),
            "baseline_selected": bool(baseline_selected),
            "is_test": bool(is_test),
            "test_pose_only_verification": test_pose_only_verification,
            "is_bootstrap": bool(is_bootstrap),
            "pose_uncertainty": float(pose_uncertainty),
            "state_support_gap": float(state_support_gap),
            "temporal_degradation": float(temporal_degradation),
            "risk_score": float(risk_score),
            "risk_threshold": float(threshold),
            "warmed_up": bool(warmed_up),
            "history_count_before": int(len(self.risk_history)),
            "multi_signal_risk": multi_signal_risk,
            "severe_pose_risk": severe_pose_risk,
            "absolute_support": float(absolute_support),
            "consensus_ratio": float(consensus_ratio),
            "stage_retention": float(stage_retention),
            "match_count_total": int(match_count),
            "num_2d3d_correspondences": int(correspondences),
            "num_pnp_inliers": int(pnp_inliers),
            "num_miniba_inliers": int(miniba_inliers),
            "pnp_inlier_ratio": float(
                pnp_inliers / max(float(correspondences), 1.0)
            ),
            "miniba_inlier_ratio": float(
                miniba_inliers / max(float(correspondences), 1.0)
            ),
            "grid_coverage": float(grid_coverage),
            "grid_entropy": float(grid_entropy),
            "anchor_health": float(anchor_health),
            "selected_reference_count": int(selected_references),
            "recent_pose_fail_rate": float(fail_rate),
            **rotation,
            **threshold_debug,
        }
        self.events.append(event)
        if eligible:
            self.risk_history.append(float(risk_score))
        return event

    def summary(self) -> dict[str, Any]:
        eligible_events = [event for event in self.events if event["eligible"]]
        isolated_events = [event for event in eligible_events if event["isolated"]]
        verification_candidates = [
            event
            for event in eligible_events
            if event.get("verification_candidate", False)
        ]
        verification_attempts = [
            event for event in eligible_events if event.get("verification_attempted", False)
        ]
        verification_accepts = [
            event for event in eligible_events if event.get("verification_accepted", False)
        ]
        cooldown_events = [
            event
            for event in eligible_events
            if event["decision"] == "cooldown_admit"
        ]

        def mean(key: str) -> float:
            if not eligible_events:
                return 0.0
            return float(sum(float(event[key]) for event in eligible_events) / len(eligible_events))

        return {
            "events": int(len(self.events)),
            "eligible": int(len(eligible_events)),
            "isolated": int(len(isolated_events)),
            "cooldown_blocked": int(len(cooldown_events)),
            "isolated_ratio": float(len(isolated_events) / max(len(eligible_events), 1)),
            "verification_candidates": int(len(verification_candidates)),
            "verification_attempts": int(len(verification_attempts)),
            "verification_accepted": int(len(verification_accepts)),
            "verification_accept_rate": float(
                len(verification_accepts) / max(len(verification_attempts), 1)
            ),
            "risk_score_mean": mean("risk_score"),
            "pose_uncertainty_mean": mean("pose_uncertainty"),
            "state_support_gap_mean": mean("state_support_gap"),
            "temporal_degradation_mean": mean("temporal_degradation"),
        }

    def flush(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "mode": self.mode,
                "absolute_threshold": self.absolute_threshold,
                "adaptive_sigma": self.adaptive_sigma,
                "warmup": self.warmup,
                "history_size": self.history_size,
                "cooldown_frames": self.cooldown_frames,
            },
            "summary": self.summary(),
            "events": self.events,
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
