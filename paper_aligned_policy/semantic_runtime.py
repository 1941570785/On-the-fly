from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .config import Thresholds


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


class SemanticV1RuntimePolicy:
    def __init__(self, thresholds: Thresholds | None = None) -> None:
        self.th = thresholds or Thresholds()
        self.recovery_pool: list[dict[str, Any]] = []
        self.recovery_pool_max = 0
        self.total_frames = 0
        self.decision_counts = {"direct_admit": 0, "defer_recoverable": 0, "discard": 0}
        self.recovery_delay_frames = 8
        self.recovery_max_attempts = 3
        self.recovery_attempts_per_tick = 3
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

        risk_from_instability = 0.55 * (1.0 - support_ratio) + 0.45 * pose_fail_rate
        if not baseline_should_add:
            risk_from_instability = 0.6 * risk_from_instability + 0.4
        R_t = _clamp01(risk_from_instability)

        baseline_bonus = 1.0 if baseline_should_add else 0.0
        test_bonus = 0.2 if is_test else 0.0
        V_t = _clamp01(0.45 * support_ratio + 0.35 * disp_ratio + 0.20 * baseline_bonus + test_bonus)

        C_t = _clamp01(0.6 * support_ratio + 0.4 * (1.0 - pose_fail_rate))
        B_R_t = _clamp01(1.0 - abs(R_t - 0.575) / 0.175)
        Q_t = _clamp01(0.45 * B_R_t + 0.25 * V_t + 0.30 * C_t)
        return {"R_t": R_t, "V_t": V_t, "C_t": C_t, "B_R_t": B_R_t, "Q_t": Q_t}

    def _decision(self, R_t: float, V_t: float, Q_t: float) -> str:
        direct = (R_t <= self.th.tau_R_low) and (V_t >= self.th.tau_V)
        defer = (
            (R_t > self.th.tau_R_low)
            and (R_t < self.th.tau_R_high)
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

    def _tick_recovery(self, current_step: int, current_scores: dict[str, float]) -> dict[str, Any]:
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
            ok = (q_blend >= self.th.tau_Q) and (v_blend >= self.th.tau_V_min) and (r_blend < self.th.tau_R_high)
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
                        "recovery_attempt_count": int(item["attempts"]),
                        "scores": {
                            "R_t": float(item.get("R_t", 0.0)),
                            "V_t": float(item.get("V_t", 0.0)),
                            "Q_t": float(item.get("Q_t", 0.0),
                            ),
                            "q_blend": float(q_blend),
                            "v_blend": float(v_blend),
                            "r_blend": float(r_blend),
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
        action = self._decision(scores["R_t"], scores["V_t"], scores["Q_t"])
        self.decision_counts[action] = int(self.decision_counts.get(action, 0)) + 1
        recovery_tick = self._tick_recovery(self.total_frames, scores)
        if action == "defer_recoverable":
            self.recovery_pool.append(
                {
                    "frame_id": int(frame_id),
                    "defer_step": int(self.total_frames),
                    "attempts": 0,
                    "R_t": scores["R_t"],
                    "V_t": scores["V_t"],
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
