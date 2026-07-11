from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


POSE_RISK_UTILITY_MODES = {"off", "observe_v1", "active_v1"}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _clamp01(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, _as_float(value, default)))


def representation_utility_score(
    *,
    coverage_deficit: float,
    residual_selectivity: float,
    new_view_event_score: float,
    selectivity_reference: float = 1.8,
) -> tuple[float, dict[str, float]]:
    coverage_value = _clamp01(coverage_deficit)
    selectivity_value = _clamp01(
        _as_float(residual_selectivity) / max(_as_float(selectivity_reference, 1.8), 1e-6)
    )
    novelty_value = _clamp01(new_view_event_score)
    utility = _clamp01(
        0.50 * coverage_value
        + 0.30 * selectivity_value
        + 0.20 * novelty_value
    )
    return utility, {
        "coverage_value": float(coverage_value),
        "selectivity_value": float(selectivity_value),
        "novelty_value": float(novelty_value),
    }


def pose_risk_candidate(risk_event: dict[str, Any] | None) -> bool:
    event = dict(risk_event or {})
    return bool(
        event.get("eligible", False)
        and event.get("warmed_up", False)
        and _as_float(event.get("risk_score"))
        >= _as_float(event.get("risk_threshold"), 1.0)
        and (
            event.get("multi_signal_risk", False)
            or event.get("severe_pose_risk", False)
        )
    )


def pose_review_acceptance(
    *,
    start_loss: float,
    end_loss: float,
    rotation_delta_deg: float,
    translation_delta: float,
    max_rotation_delta_deg: float,
    max_translation_delta: float,
    max_loss_increase_ratio: float = 0.0,
) -> tuple[bool, str]:
    start = max(_as_float(start_loss), 1e-8)
    end = _as_float(end_loss, float("inf"))
    if end > start * (1.0 + max(0.0, _as_float(max_loss_increase_ratio))):
        return False, "loss_degraded"
    if _as_float(rotation_delta_deg) > max(
        0.0, _as_float(max_rotation_delta_deg)
    ):
        return False, "rotation_step_exceeded"
    if _as_float(translation_delta) > max(0.0, _as_float(max_translation_delta)):
        return False, "translation_step_exceeded"
    return True, "accepted"


class PoseRiskUtilityAdmissionGate:
    """Combine post-pose uncertainty with expected representation value."""

    def __init__(
        self,
        *,
        mode: str = "off",
        utility_threshold: float = 0.25,
        selectivity_reference: float = 1.8,
    ) -> None:
        normalized_mode = str(mode or "off").strip().lower()
        if normalized_mode not in POSE_RISK_UTILITY_MODES:
            raise ValueError(f"Unsupported pose risk utility mode: {mode}")
        self.mode = normalized_mode
        self.utility_threshold = _clamp01(utility_threshold, 0.25)
        self.selectivity_reference = max(
            1e-6, _as_float(selectivity_reference, 1.8)
        )
        self.events: list[dict[str, Any]] = []

    def evaluate(
        self,
        *,
        frame_id: int,
        risk_event: dict[str, Any] | None,
        render_probe: dict[str, Any] | None,
        baseline_selected: bool,
        is_test: bool,
        is_bootstrap: bool,
    ) -> dict[str, Any]:
        risk_event = dict(risk_event or {})
        render_probe = dict(render_probe or {})
        candidate = pose_risk_candidate(risk_event)
        coverage_deficit = _clamp01(render_probe.get("coverage_deficit"), 0.0)
        residual_selectivity = max(
            0.0, _as_float(render_probe.get("residual_selectivity"), 0.0)
        )
        new_view_event_score = _clamp01(
            render_probe.get(
                "new_view_event_score",
                risk_event.get("new_view_event_score", 0.0),
            )
        )
        utility_score, utility_debug = representation_utility_score(
            coverage_deficit=coverage_deficit,
            residual_selectivity=residual_selectivity,
            new_view_event_score=new_view_event_score,
            selectivity_reference=self.selectivity_reference,
        )
        suggested_decision = (
            "review_admit"
            if candidate and utility_score >= self.utility_threshold
            else "isolate_low_utility"
            if candidate
            else "admit"
        )

        if not baseline_selected:
            decision = "bypass_not_selected"
        elif is_test:
            decision = "bypass_test"
        elif is_bootstrap:
            decision = "bypass_bootstrap"
        elif self.mode == "off":
            decision = "off"
        elif self.mode == "observe_v1":
            decision = "observe"
        else:
            decision = suggested_decision

        event = {
            "frame_id": int(frame_id),
            "mode": self.mode,
            "decision": decision,
            "suggested_decision": suggested_decision,
            "baseline_selected": bool(baseline_selected),
            "is_test": bool(is_test),
            "is_bootstrap": bool(is_bootstrap),
            "risk_candidate": bool(candidate),
            "probe_required": bool(candidate and not render_probe),
            "review": bool(decision == "review_admit"),
            "isolated": bool(decision == "isolate_low_utility"),
            "risk_score": _as_float(risk_event.get("risk_score")),
            "risk_threshold": _as_float(risk_event.get("risk_threshold")),
            "pose_uncertainty": _as_float(risk_event.get("pose_uncertainty")),
            "utility_score": float(utility_score),
            "utility_threshold": float(self.utility_threshold),
            "coverage_deficit": float(coverage_deficit),
            "residual_selectivity": float(residual_selectivity),
            "new_view_event_score": float(new_view_event_score),
            **utility_debug,
        }
        self.events.append(event)
        return event

    def summary(self) -> dict[str, Any]:
        decisions = [str(event["decision"]) for event in self.events]
        candidates = [event for event in self.events if event["risk_candidate"]]

        def mean(key: str) -> float:
            if not candidates:
                return 0.0
            return float(
                sum(float(event[key]) for event in candidates) / len(candidates)
            )

        return {
            "events": int(len(self.events)),
            "risk_candidates": int(len(candidates)),
            "review_admit": int(decisions.count("review_admit")),
            "isolate_low_utility": int(decisions.count("isolate_low_utility")),
            "observe": int(decisions.count("observe")),
            "utility_score_mean": mean("utility_score"),
            "coverage_deficit_mean": mean("coverage_deficit"),
            "residual_selectivity_mean": mean("residual_selectivity"),
        }

    def flush(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "mode": self.mode,
                "utility_threshold": self.utility_threshold,
                "selectivity_reference": self.selectivity_reference,
            },
            "summary": self.summary(),
            "events": self.events,
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
