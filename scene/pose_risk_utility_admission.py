from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


POSE_RISK_UTILITY_MODES = {
    "off",
    "observe_v1",
    "active_v1",
    "pose_quarantine_v1",
    "pose_quarantine_utility_v1",
    "pose_quarantine_severe_v1",
}


def pose_reference_quarantine_enabled(mode: str) -> bool:
    return str(mode or "").strip().lower() in {
        "pose_quarantine_v1",
        "pose_quarantine_utility_v1",
        "pose_quarantine_severe_v1",
    }


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _clamp01(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, _as_float(value, default)))


def snapshot_optimizer_parameter_state(
    optimizer: Any,
    parameter_names: set[str],
) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for name, parameter in optimizer.params.items():
        if name not in parameter_names:
            continue
        snapshot[name] = {
            key: parameter[key].detach().clone()
            for key in ("val", "exp_avg", "exp_avg_sq")
            if key in parameter
        }
    return snapshot


def restore_optimizer_parameter_state(
    optimizer: Any,
    snapshot: dict[str, dict[str, Any]],
    *,
    restore_values: bool = True,
) -> None:
    for name, state in snapshot.items():
        parameter = optimizer.params[name]
        for key, value in state.items():
            if key == "val" and not restore_values:
                continue
            parameter[key].data.copy_(value)


def scale_optimizer_parameter_learning_rates(
    optimizer: Any,
    parameter_names: set[str],
    *,
    scale: float,
) -> dict[str, Any]:
    factor = _as_float(scale, 1.0)
    if factor <= 0.0:
        factor = 1.0
    snapshot: dict[str, Any] = {}
    for name, parameter in optimizer.params.items():
        if name not in parameter_names or "lr" not in parameter:
            continue
        learning_rate = parameter["lr"]
        snapshot[name] = (
            learning_rate.detach().clone()
            if hasattr(learning_rate, "detach")
            else learning_rate
        )
        parameter["lr"] = learning_rate * factor
    return snapshot


def restore_optimizer_parameter_learning_rates(
    optimizer: Any,
    snapshot: dict[str, Any],
) -> None:
    for name, learning_rate in snapshot.items():
        optimizer.params[name]["lr"] = learning_rate


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


def pose_review_candidate(
    risk_event: dict[str, Any] | None,
    *,
    use_verification_candidates: bool = False,
) -> bool:
    event = dict(risk_event or {})
    verification_candidate = bool(
        use_verification_candidates
        and event.get("eligible", False)
        and event.get("warmed_up", False)
        and event.get("verification_candidate", False)
    )
    return bool(pose_risk_candidate(event) or verification_candidate)


def filter_pose_reference_indices(
    keyframes: list[Any],
    indices: list[int],
    *,
    enabled: bool,
) -> list[int]:
    original = [int(index) for index in indices]
    if not enabled:
        return original
    filtered = [
        index
        for index in original
        if not bool(
            keyframes[index].info.get("_pose_reference_quarantined", False)
        )
    ]
    return filtered if filtered else original


def pose_review_acceptance(
    *,
    start_loss: float,
    end_loss: float,
    rotation_delta_deg: float,
    translation_delta: float,
    max_rotation_delta_deg: float,
    max_translation_delta: float,
    max_loss_increase_ratio: float = 0.0,
    min_relative_loss_improvement: float = 0.0,
    validation_support_ratio: float = 1.0,
    min_validation_support_ratio: float = 0.0,
) -> tuple[bool, str]:
    if not math.isfinite(float(start_loss)) or not math.isfinite(float(end_loss)):
        return False, "non_finite_loss"
    if _as_float(validation_support_ratio) < max(
        0.0, _as_float(min_validation_support_ratio)
    ):
        return False, "validation_support_lost"
    start = max(_as_float(start_loss), 1e-8)
    end = _as_float(end_loss)
    if end > start * (1.0 + max(0.0, _as_float(max_loss_increase_ratio))):
        return False, "loss_degraded"
    relative_improvement = (start - end) / start
    if relative_improvement < max(
        0.0, _as_float(min_relative_loss_improvement)
    ):
        return False, "loss_improvement_too_small"
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
        utility_threshold: float = 0.24,
        selectivity_reference: float = 1.8,
        isolation_risk_margin: float = 0.04,
        isolation_cooldown_frames: int = 24,
        quarantine_risk_margin: float = 0.08,
        quarantine_cooldown_frames: int = 64,
        use_verification_candidates: bool = False,
        review_test_candidates: bool = False,
    ) -> None:
        normalized_mode = str(mode or "off").strip().lower()
        if normalized_mode not in POSE_RISK_UTILITY_MODES:
            raise ValueError(f"Unsupported pose risk utility mode: {mode}")
        self.mode = normalized_mode
        self.utility_threshold = _clamp01(utility_threshold, 0.24)
        self.selectivity_reference = max(
            1e-6, _as_float(selectivity_reference, 1.8)
        )
        self.isolation_risk_margin = max(
            0.0, _as_float(isolation_risk_margin, 0.04)
        )
        self.isolation_cooldown_frames = max(
            0, int(isolation_cooldown_frames)
        )
        self.quarantine_risk_margin = max(
            0.0, _as_float(quarantine_risk_margin, 0.08)
        )
        self.quarantine_cooldown_frames = max(
            0, int(quarantine_cooldown_frames)
        )
        self.use_verification_candidates = bool(use_verification_candidates)
        self.review_test_candidates = bool(review_test_candidates)
        self.last_isolated_frame_id = -1
        self.last_quarantined_frame_id = -1
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
        verification_candidate_routed = bool(
            self.use_verification_candidates
            and risk_event.get("eligible", False)
            and risk_event.get("warmed_up", False)
            and risk_event.get("verification_candidate", False)
        )
        candidate = pose_review_candidate(
            risk_event,
            use_verification_candidates=self.use_verification_candidates,
        )
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
        risk_score = _as_float(risk_event.get("risk_score"))
        risk_threshold = _as_float(risk_event.get("risk_threshold"))
        risk_margin = risk_score - risk_threshold
        strong_isolation_risk = bool(
            risk_event.get("severe_pose_risk", False)
            or risk_margin >= self.isolation_risk_margin
        )
        cooldown_active = bool(
            self.last_isolated_frame_id >= 0
            and int(frame_id) - self.last_isolated_frame_id
            < self.isolation_cooldown_frames
        )
        quarantine_risk = bool(
            risk_event.get("severe_pose_risk", False)
            or risk_margin >= self.quarantine_risk_margin
        )
        quarantine_cooldown_active = bool(
            self.last_quarantined_frame_id >= 0
            and int(frame_id) - self.last_quarantined_frame_id
            < self.quarantine_cooldown_frames
        )
        if self.mode == "pose_quarantine_severe_v1":
            if not candidate:
                suggested_decision = "admit"
            elif not bool(risk_event.get("severe_pose_risk", False)):
                suggested_decision = "admit_conservative"
            elif quarantine_cooldown_active:
                suggested_decision = "quarantine_cooldown_admit"
            else:
                suggested_decision = "conservative_render_pose_quarantine"
        elif self.mode == "pose_quarantine_utility_v1":
            if not candidate:
                suggested_decision = "admit"
            elif utility_score >= self.utility_threshold:
                suggested_decision = "render_admit_high_utility"
            elif not quarantine_risk:
                suggested_decision = "admit_conservative"
            elif quarantine_cooldown_active:
                suggested_decision = "quarantine_cooldown_admit"
            else:
                suggested_decision = "conservative_render_pose_quarantine"
        elif self.mode == "pose_quarantine_v1":
            if not candidate:
                suggested_decision = "admit"
            elif not quarantine_risk:
                suggested_decision = "admit_conservative"
            elif quarantine_cooldown_active:
                suggested_decision = "quarantine_cooldown_admit"
            elif utility_score >= self.utility_threshold:
                suggested_decision = "render_admit_pose_quarantine"
            else:
                suggested_decision = "conservative_render_pose_quarantine"
        else:
            if not candidate:
                suggested_decision = "admit"
            elif utility_score >= self.utility_threshold:
                suggested_decision = "review_admit"
            elif not strong_isolation_risk:
                suggested_decision = "admit_conservative"
            elif cooldown_active:
                suggested_decision = "cooldown_admit"
            else:
                suggested_decision = "isolate_low_utility"

        if not baseline_selected:
            decision = "bypass_not_selected"
        elif is_test:
            decision = (
                "review_admit"
                if self.review_test_candidates
                and suggested_decision == "review_admit"
                else "bypass_test"
            )
        elif is_bootstrap:
            decision = "bypass_bootstrap"
        elif self.mode == "off":
            decision = "off"
        elif self.mode == "observe_v1":
            decision = "observe"
        else:
            decision = suggested_decision
        if decision == "isolate_low_utility":
            self.last_isolated_frame_id = int(frame_id)
        if decision in {
            "render_admit_pose_quarantine",
            "conservative_render_pose_quarantine",
        }:
            self.last_quarantined_frame_id = int(frame_id)

        event = {
            "frame_id": int(frame_id),
            "source_frame_id": int(risk_event.get("source_frame_id", frame_id)),
            "image_name": str(risk_event.get("image_name", "")),
            "estimated_Rt": risk_event.get("estimated_Rt"),
            "gt_Rt": risk_event.get("gt_Rt"),
            "mode": self.mode,
            "decision": decision,
            "suggested_decision": suggested_decision,
            "baseline_selected": bool(baseline_selected),
            "is_test": bool(is_test),
            "is_bootstrap": bool(is_bootstrap),
            "risk_candidate": bool(candidate),
            "verification_candidate_routed": verification_candidate_routed,
            "probe_required": bool(candidate and not render_probe),
            "review": bool(decision == "review_admit"),
            "isolated": bool(decision == "isolate_low_utility"),
            "pose_reference_quarantined": bool(
                decision
                in {
                    "render_admit_pose_quarantine",
                    "conservative_render_pose_quarantine",
                }
            ),
            "risk_score": risk_score,
            "risk_threshold": risk_threshold,
            "risk_margin": float(risk_margin),
            "strong_isolation_risk": strong_isolation_risk,
            "cooldown_active": cooldown_active,
            "quarantine_risk": quarantine_risk,
            "quarantine_cooldown_active": quarantine_cooldown_active,
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
            "admit_conservative": int(decisions.count("admit_conservative")),
            "cooldown_admit": int(decisions.count("cooldown_admit")),
            "quarantine_cooldown_admit": int(
                decisions.count("quarantine_cooldown_admit")
            ),
            "pose_reference_quarantined": int(
                sum(bool(event["pose_reference_quarantined"]) for event in self.events)
            ),
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
                "isolation_risk_margin": self.isolation_risk_margin,
                "isolation_cooldown_frames": self.isolation_cooldown_frames,
                "quarantine_risk_margin": self.quarantine_risk_margin,
                "quarantine_cooldown_frames": self.quarantine_cooldown_frames,
                "use_verification_candidates": self.use_verification_candidates,
                "review_test_candidates": self.review_test_candidates,
            },
            "summary": self.summary(),
            "events": self.events,
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
