from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .config import Thresholds


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


class SemanticV1RuntimePolicy:
    def __init__(
        self,
        thresholds: Thresholds | None = None,
        recovery_delay_frames: int | None = None,
        recovery_max_attempts: int | None = None,
        recovery_attempts_per_tick: int | None = None,
    ) -> None:
        self.th = thresholds or Thresholds()
        self.recovery_pool: list[dict[str, Any]] = []
        self.recovery_pool_max = 0
        self.total_frames = 0
        self.decision_counts = {"direct_admit": 0, "defer_recoverable": 0, "discard": 0}
        self.recovery_delay_frames = int(8 if recovery_delay_frames is None else recovery_delay_frames)
        self.recovery_max_attempts = int(3 if recovery_max_attempts is None else recovery_max_attempts)
        self.recovery_attempts_per_tick = int(3 if recovery_attempts_per_tick is None else recovery_attempts_per_tick)
        self.recovery_attempt_count = 0
        self.recovery_success_count = 0
        self.recovery_discard_count = 0
        self.recovery_success_events: list[dict[str, Any]] = []
        self.recovery_discard_events: list[dict[str, Any]] = []
        self.recovery_pool_timeline: list[dict[str, Any]] = []
        self._recovered_sources_queue: list[dict[str, Any]] = []

    def _scores(self, baseline_should_add: bool, evidence: dict[str, Any]) -> dict[str, float]:
        median_disp = float(evidence.get("median_displacement", 0.0) or 0.0)
        disp_th = float(evidence.get("displacement_threshold", 1.0) or 1.0)
        disp_ratio = _clamp01(median_disp / max(disp_th, 1e-6))
        num_matches = float(evidence.get("num_matches", 0.0) or 0.0)
        min_inliers = float(evidence.get("min_num_inliers_threshold", 100.0) or 100.0)
        support_ratio = _clamp01(num_matches / max(2.0 * min_inliers, 1.0))
        pose_fail_rate = _clamp01(float(evidence.get("recent_pose_fail_rate", 0.0) or 0.0))
        is_test = bool(evidence.get("is_test", False))

        pose_uncertainty = _clamp01(float(evidence.get("pose_uncertainty", 1.0 - support_ratio) or 0.0))
        state_support_gap = _clamp01(float(evidence.get("state_support_gap", 1.0 - support_ratio) or 0.0))
        temporal_degradation = _clamp01(
            float(evidence.get("temporal_degradation", pose_fail_rate) or 0.0)
        )
        R_t = _clamp01(0.40 * pose_uncertainty + 0.35 * state_support_gap + 0.25 * temporal_degradation)

        baseline_bonus = 1.0 if baseline_should_add else 0.0
        test_bonus = 1.0 if is_test else 0.0
        representation_gain = _clamp01(float(evidence.get("representation_gain", disp_ratio) or 0.0))
        view_motion_gain = _clamp01(float(evidence.get("view_motion_gain", disp_ratio) or 0.0))
        chain_support_gain = _clamp01(float(evidence.get("chain_support_gain", support_ratio) or 0.0))
        V_t = _clamp01(
            0.35 * representation_gain
            + 0.30 * view_motion_gain
            + 0.25 * chain_support_gain
            + 0.10 * baseline_bonus
            + 0.05 * test_bonus
        )

        default_context = 0.55 * chain_support_gain + 0.25 * (1.0 - pose_fail_rate) + 0.20 * support_ratio
        C_t = _clamp01(float(evidence.get("recovery_context_score", default_context) or 0.0))
        B_R_t = self._risk_band_score(R_t)
        Q_t = _clamp01(0.40 * B_R_t + 0.30 * V_t + 0.30 * C_t)
        return {"R_t": R_t, "V_t": V_t, "C_t": C_t, "B_R_t": B_R_t, "Q_t": Q_t}

    def _risk_band_score(self, R_t: float) -> float:
        risk = _clamp01(float(R_t))
        high = float(self.th.tau_R_high)
        if risk <= high:
            return 1.0
        return _clamp01((1.0 - risk) / max(1.0 - high, 1e-6))

    def _decision(
        self,
        baseline_should_add: bool,
        R_t: float,
        V_t: float,
        B_R_t: float,
        Q_t: float,
    ) -> str:
        direct = bool(baseline_should_add) and (R_t <= self.th.tau_R_low) and (V_t >= self.th.tau_V)
        defer = (
            (B_R_t >= self.th.tau_B)
            and (V_t >= self.th.tau_V_min)
            and (Q_t >= self.th.tau_Q)
        )
        if direct:
            return "direct_admit"
        if defer:
            return "defer_recoverable"
        return "discard"

    def _recovery_priority(self, item: dict[str, Any], current_step: int) -> float:
        age = max(0, current_step - int(item["defer_step"]))
        return (
            0.45 * float(item["Q_t"])
            + 0.25 * float(item["V_t"])
            + 0.15 * (1.0 - float(item["R_t"]))
            + 0.15 * min(age / 50.0, 1.0)
        )

    def _tick_recovery(
        self,
        current_step: int,
        current_scores: dict[str, float],
        current_frame_id: int,
    ) -> dict[str, Any]:
        pool_before = len(self.recovery_pool)
        if not self.recovery_pool:
            tick = {
                "attempted": 0,
                "success": 0,
                "discarded": 0,
                "success_frame_ids": [],
                "pool_before": pool_before,
                "pool_after": len(self.recovery_pool),
            }
            self.recovery_pool_timeline.append({"tick": int(current_step), **tick})
            return tick

        eligible = [
            item
            for item in self.recovery_pool
            if (current_step - int(item["defer_step"])) >= self.recovery_delay_frames
        ]
        if not eligible:
            tick = {
                "attempted": 0,
                "success": 0,
                "discarded": 0,
                "success_frame_ids": [],
                "pool_before": pool_before,
                "pool_after": len(self.recovery_pool),
            }
            self.recovery_pool_timeline.append({"tick": int(current_step), **tick})
            return tick

        eligible.sort(key=lambda x: self._recovery_priority(x, current_step), reverse=True)
        chosen = eligible[: self.recovery_attempts_per_tick]
        attempts = 0
        success = 0
        discarded = 0
        success_ids: list[int] = []
        for item in chosen:
            attempts += 1
            self.recovery_attempt_count += 1
            q_blend = 0.7 * float(item["Q_t"]) + 0.3 * float(current_scores["Q_t"])
            v_blend = 0.7 * float(item["V_t"]) + 0.3 * float(current_scores["V_t"])
            r_blend = 0.7 * float(item["R_t"]) + 0.3 * float(current_scores["R_t"])
            b_blend = self._risk_band_score(r_blend)
            ok = (
                (q_blend >= self.th.tau_Q)
                and (v_blend >= self.th.tau_V_min)
                and (b_blend >= self.th.tau_B)
            )
            item["attempts"] = int(item.get("attempts", 0)) + 1
            if ok:
                success += 1
                self.recovery_success_count += 1
                source_id = int(item["frame_id"])
                success_ids.append(source_id)
                self._recovered_sources_queue.append(
                    {
                        "source_frame_id": source_id,
                        "source_input_index": int(item.get("source_payload", {}).get("source_input_index", source_id)),
                        "source_payload": item.get("source_payload", {}),
                        "pool_enter_tick": int(item["defer_step"]),
                        "recovery_attempt_tick": int(current_step),
                        "current_tick_frame_id": int(current_frame_id),
                        "recovery_attempt_count": int(item["attempts"]),
                        "scores": {
                            "R_t": float(item.get("R_t", 0.0)),
                            "V_t": float(item.get("V_t", 0.0)),
                            "B_R_t": float(item.get("B_R_t", 0.0)),
                            "Q_t": float(item.get("Q_t", 0.0)),
                            "q_blend": float(q_blend),
                            "v_blend": float(v_blend),
                            "r_blend": float(r_blend),
                            "b_blend": float(b_blend),
                        },
                    }
                )
                self.recovery_success_events.append(
                    {
                        "frame_id": source_id,
                        "attempts": int(item["attempts"]),
                        "recover_step": int(current_step),
                        "q_blend": float(q_blend),
                        "v_blend": float(v_blend),
                        "r_blend": float(r_blend),
                        "b_blend": float(b_blend),
                    }
                )
                self.recovery_pool.remove(item)
            elif int(item["attempts"]) >= self.recovery_max_attempts:
                discarded += 1
                self.recovery_discard_count += 1
                self.recovery_discard_events.append(
                    {
                        "frame_id": int(item["frame_id"]),
                        "attempts": int(item["attempts"]),
                        "discard_step": int(current_step),
                    }
                )
                self.recovery_pool.remove(item)

        tick = {
            "attempted": attempts,
            "success": success,
            "discarded": discarded,
            "success_frame_ids": success_ids,
            "pool_before": pool_before,
            "pool_after": len(self.recovery_pool),
        }
        self.recovery_pool_timeline.append({"tick": int(current_step), **tick})
        return tick

    def decide(
        self,
        frame_id: int,
        baseline_should_add: bool,
        evidence: dict[str, Any] | None = None,
        source_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.total_frames += 1
        ev = evidence or {}
        scores = self._scores(baseline_should_add, ev)
        action = self._decision(
            baseline_should_add,
            scores["R_t"],
            scores["V_t"],
            scores["B_R_t"],
            scores["Q_t"],
        )
        self.decision_counts[action] = int(self.decision_counts.get(action, 0)) + 1
        recovery_tick = self._tick_recovery(self.total_frames, scores, int(frame_id))
        if action == "defer_recoverable":
            self.recovery_pool.append(
                {
                    "frame_id": int(frame_id),
                    "defer_step": int(self.total_frames),
                    "attempts": 0,
                    "R_t": scores["R_t"],
                    "V_t": scores["V_t"],
                    "B_R_t": scores["B_R_t"],
                    "Q_t": scores["Q_t"],
                    "source_payload": source_payload or {},
                }
            )
            self.recovery_pool_max = max(self.recovery_pool_max, len(self.recovery_pool))
        return {
            "action": action,
            "admit_to_chain": action == "direct_admit",
            "scores": scores,
            "recovery_pool_size": len(self.recovery_pool),
            "recovery_tick": recovery_tick,
            "thresholds": asdict(self.th),
        }

    def pop_recovered_sources(self) -> list[dict[str, Any]]:
        items = self._recovered_sources_queue
        self._recovered_sources_queue = []
        return items

    def summary(self) -> dict[str, Any]:
        return {
            "total_frames": int(self.total_frames),
            "recovery_pool_size": int(len(self.recovery_pool)),
            "recovery_pool_max": int(self.recovery_pool_max),
            "recovery_attempt_count": int(self.recovery_attempt_count),
            "recovery_success_count": int(self.recovery_success_count),
            "recovery_discard_count": int(self.recovery_discard_count),
            "decision_counts": {k: int(v) for k, v in self.decision_counts.items()},
            "recovery_success_events": self.recovery_success_events,
            "recovery_discard_events": self.recovery_discard_events,
            "recovery_pool_timeline": self.recovery_pool_timeline,
            "thresholds": asdict(self.th),
            "recovery_config": {
                "delay_frames": int(self.recovery_delay_frames),
                "max_attempts": int(self.recovery_max_attempts),
                "attempts_per_tick": int(self.recovery_attempts_per_tick),
            },
        }
