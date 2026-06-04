from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import apply_coupled_innovation_defaults, resolve_coupled_innovation_config
from .direct_density_control import DirectDensityController
from .recovery_commit_control import RecoveryCommitController
from .semantic_runtime import SemanticV1RuntimePolicy


def _to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    if isinstance(x, (int, float)):
        return x != 0
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


class PaperAlignedRuntimeGate:
    def __init__(self, args: Any) -> None:
        self.requested_mode = str(getattr(args, "risk_admission_mode", "off") or "off")
        self.coupled_config = apply_coupled_innovation_defaults(args)
        self.mode = str(getattr(args, "risk_admission_mode", "off") or "off")
        self.training_risk_mode = self.mode
        self.trace_path = str(getattr(args, "paper_aligned_contract_trace_path", "") or "").strip()
        self.recovery_bridge_mode = str(
            getattr(args, "paper_aligned_recovery_commit_bridge", "true_source_commit") or "true_source_commit"
        )
        self.recovery_commit_control_mode = str(
            getattr(args, "paper_aligned_recovery_commit_control", "off") or "off"
        )
        self.recovery_window_size = int(getattr(args, "paper_aligned_recovery_window_size", 30) or 30)
        self.trace_events: list[dict[str, Any]] = []
        self._event_index: dict[int, int] = {}
        self._pending_true_source_commits: list[dict[str, Any]] = []
        self._held_true_source_commits: list[dict[str, Any]] = []
        self.true_recovery_commit_events: list[dict[str, Any]] = []
        self.recovery_commit_control_events: list[dict[str, Any]] = []
        self.recovery_commit_materialization_events: list[dict[str, Any]] = []
        self.keyframe_timeline_events: list[dict[str, Any]] = []
        self.chosen_kfs_reference_events: list[dict[str, Any]] = []
        self.matching_support_events: list[dict[str, Any]] = []
        self.pnp_miniba_reference_events: list[dict[str, Any]] = []
        self.local_map_anchor_events: list[dict[str, Any]] = []
        self.support_integration_events: list[dict[str, Any]] = []
        self.chosen_kfs_candidate_events: list[dict[str, Any]] = []
        self.pose_reference_pool_events: list[dict[str, Any]] = []
        self.matching_to_pose_path_bridge_events: list[dict[str, Any]] = []
        self.lifecycle_gate_events: list[dict[str, Any]] = []
        self.recovery_pose_path_events: list[dict[str, Any]] = []
        self.frame_stage_reachability_events: list[dict[str, Any]] = []
        self.recovery_pose_outcome_fix_events: list[dict[str, Any]] = []
        self.recovery_2d3d_support_events: list[dict[str, Any]] = []
        self.recovery_reference_3d_association_events: list[dict[str, Any]] = []
        self.recovery_pnp_consensus_events: list[dict[str, Any]] = []
        self.recovery_ref_subset_events: list[dict[str, Any]] = []
        self.anchor_transition_bridge_events: list[dict[str, Any]] = []
        self.recovery_support_trace_events: list[dict[str, Any]] = []
        self.direct_density_control_events: list[dict[str, Any]] = []
        self.direct_density_control_v2_events: list[dict[str, Any]] = []
        self.direct_density_control_v2_1_events: list[dict[str, Any]] = []
        self.direct_density_control_v2_2_events: list[dict[str, Any]] = []
        self.direct_density_control_v2_2_1_events: list[dict[str, Any]] = []
        self.direct_density_control_v2_2_2_events: list[dict[str, Any]] = []
        self.direct_density_control_v2_2_2_1_events: list[dict[str, Any]] = []
        self.trace_unavailable_reasons: dict[str, str] = {
            "match_graph_neighbor_ids": "No persistent match graph object is exposed; pairwise matches live on DescribedKeypoints.matches.",
            "match_graph_id": "No stable match graph id exists for keyframes in the current runtime.",
        }
        self.semantic_policy: SemanticV1RuntimePolicy | None = None
        self.recovery_commit_controller = RecoveryCommitController(args)
        self.direct_density_controller = DirectDensityController(args)
        self._anchor_count_at_last_direct_finalize = 1
        if self.mode == "paper_aligned_semantic_v1":
            cfg = self.coupled_config
            self.semantic_policy = SemanticV1RuntimePolicy(
                thresholds=cfg.thresholds if cfg.enabled else None,
                recovery_delay_frames=cfg.recovery_delay_frames if cfg.enabled else None,
                recovery_max_attempts=cfg.recovery_max_attempts if cfg.enabled else None,
                recovery_attempts_per_tick=cfg.recovery_attempts_per_tick if cfg.enabled else None,
            )

    def _get_event(self, frame_id: int) -> dict[str, Any] | None:
        idx = self._event_index.get(frame_id)
        if idx is None:
            return None
        if idx < 0 or idx >= len(self.trace_events):
            return None
        return self.trace_events[idx]

    def _source_payload(self, frame_id: int, info: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
        return {
            "source_frame_id": int(frame_id),
            "source_input_index": int(frame_id),
            "source_image_name": str(info.get("image_name", info.get("image_path", "")) or ""),
            "image_path": str(info.get("image_path", "")),
            "source_info": dict(info),
            "image_tensor": info.get("_image_tensor"),
            "desc_kpts": info.get("_desc_kpts"),
            "inlier_evidence": info.get("_inlier_evidence", {}),
            "local_context": info.get("_local_context", {}),
            "D_t_evidence": dict(evidence),
        }

    def pop_pending_true_source_commits(self, current_tick_frame_id: int | None = None) -> list[dict[str, Any]]:
        items = list(self._pending_true_source_commits)
        self._pending_true_source_commits = []
        if current_tick_frame_id is not None and self._held_true_source_commits:
            remaining: list[dict[str, Any]] = []
            for item in self._held_true_source_commits:
                if int(item.get("_next_retry_tick", 0)) <= int(current_tick_frame_id):
                    items.append(item)
                else:
                    remaining.append(item)
            self._held_true_source_commits = remaining
        return items

    def _recent_recovery_commit_count(self, current_tick_frame_id: int) -> int:
        if current_tick_frame_id <= 0:
            return 0
        lo = max(1, int(current_tick_frame_id) - self.recovery_window_size + 1)
        count = 0
        for ev in self.true_recovery_commit_events:
            tick = int(ev.get("recovery_attempt_tick", -1))
            if lo <= tick <= int(current_tick_frame_id) and bool(ev.get("final_keyframe_incremented", False)):
                count += 1
        return count

    def _current_keyframe_density(self, current_tick_frame_id: int) -> float:
        if current_tick_frame_id <= 0:
            return 0.0
        final_count = self._current_keyframe_count()
        return (100.0 * float(final_count)) / float(max(current_tick_frame_id, 1))

    def _current_keyframe_count(self) -> int:
        return int(sum(1 for e in self.trace_events if bool(e.get("final_keyframe_incremented", False))))

    def _main_chain_gap_p90_recent(self) -> float:
        """Recent main-chain gap estimate; never use a huge sentinel (breaks density guard)."""
        ticks = sorted(
            int(e.get("frame_id", -1))
            for e in self.trace_events
            if bool(e.get("final_keyframe_incremented", False))
        )
        if len(ticks) < 2:
            return 0.0
        gaps = [ticks[i] - ticks[i - 1] for i in range(1, len(ticks))]
        if not gaps:
            return 0.0
        if len(gaps) < 2:
            return float(max(gaps))
        gaps.sort()
        idx = int(round((len(gaps) - 1) * 0.9))
        idx = max(0, min(len(gaps) - 1, idx))
        return float(gaps[idx])

    def _finalized_ticks(self) -> list[int]:
        return sorted(
            int(e.get("frame_id", -1))
            for e in self.trace_events
            if bool(e.get("final_keyframe_incremented", False))
        )

    def _local_window_stats(self, current_tick_frame_id: int, window_size: int) -> dict[str, float | int]:
        frame_id = int(current_tick_frame_id)
        window_size = max(1, int(window_size))
        lo = max(1, frame_id - window_size + 1)
        ticks = self._finalized_ticks()
        in_window = [t for t in ticks if lo <= t <= frame_id]
        local_kf = len(in_window)
        local_density = 100.0 * float(local_kf) / float(window_size)
        gaps: list[int] = []
        for i in range(1, len(in_window)):
            gaps.append(in_window[i] - in_window[i - 1])
        if in_window:
            gaps.append(frame_id - in_window[-1])
        elif frame_id > 0:
            gaps.append(frame_id)
        local_gap_max = float(max(gaps)) if gaps else 0.0
        last_tick = ticks[-1] if ticks else 0
        local_gap_after_hold = float(frame_id - last_tick) if last_tick > 0 else float(frame_id)
        return {
            "local_window_density": local_density,
            "local_window_keyframes": local_kf,
            "local_window_gap_max": local_gap_max,
            "local_window_gap_after_if_hold": local_gap_after_hold,
        }

    def _keyframe_growth_recent(self, current_tick_frame_id: int, window: int = 100) -> int:
        if current_tick_frame_id <= 0:
            return 0
        ticks = sorted(
            int(e.get("frame_id", -1))
            for e in self.trace_events
            if bool(e.get("final_keyframe_incremented", False))
        )
        if not ticks:
            return 0
        now_lo = max(1, int(current_tick_frame_id) - int(window) + 1)
        prev_lo = max(1, int(current_tick_frame_id) - int(2 * window) + 1)
        prev_hi = max(0, int(current_tick_frame_id) - int(window))
        now_cnt = sum(1 for t in ticks if now_lo <= t <= int(current_tick_frame_id))
        prev_cnt = sum(1 for t in ticks if prev_lo <= t <= prev_hi)
        return int(now_cnt - prev_cnt)

    def _final_committed_source_ticks(self) -> list[int]:
        return sorted(
            int(e.get("frame_id", -1))
            for e in self.trace_events
            if bool(e.get("final_keyframe_incremented", False))
        )

    def _last_committed_source_before(self, source_frame_id: int) -> tuple[int, str]:
        ticks: list[tuple[int, str]] = sorted(
            (
                int(e.get("frame_id", -1)),
                str(e.get("action", "")),
            )
            for e in self.trace_events
            if bool(e.get("final_keyframe_incremented", False))
        )
        prev_tick = -1
        prev_action = ""
        for tick, action in ticks:
            if tick < int(source_frame_id):
                prev_tick = tick
                prev_action = action
            else:
                break
        return prev_tick, prev_action

    def _source_already_committed(self, source_frame_id: int) -> bool:
        event = self._get_event(int(source_frame_id))
        if event is None:
            return False
        return bool(event.get("final_keyframe_incremented", False)) or bool(
            event.get("source_recovery_committed", False)
        )

    def _current_anchor_count(self) -> int | None:
        # Runtime does not expose authoritative anchor cardinality here.
        # Do not reinterpret anchor_update call count as anchor count.
        return None

    def _recent_materialization_stats(self, current_tick_frame_id: int, window: int = 80) -> dict[str, Any]:
        lo = max(1, int(current_tick_frame_id) - int(window) + 1)
        attempts = [
            e
            for e in self.recovery_commit_materialization_events
            if lo <= int(e.get("current_tick_frame_id", e.get("recovery_attempt_tick", -1)) or -1)
            <= int(current_tick_frame_id)
        ]
        materialized = [
            e
            for e in attempts
            if bool(e.get("materialized", e.get("final_keyframe_incremented", False)))
        ]
        pose_failed = [
            e
            for e in attempts
            if "inliers_too_few" in str(e.get("failure_reason", ""))
            or "inliers_too_few" in str(e.get("materialization_failure_reason", ""))
        ]
        return {
            "recent_runtime_attempt_count": len(attempts),
            "recent_materialized_count": len(materialized),
            "recent_pose_fail_count": len(pose_failed),
            "recent_failed_no_materialization_count": max(0, len(attempts) - len(materialized)),
            "recent_materialization_rate": float(len(materialized)) / float(max(len(attempts), 1)),
            "recent_pose_fail_rate": float(len(pose_failed)) / float(max(len(attempts), 1)),
        }

    def _source_pose_fail_context(self, source_frame_id: int) -> dict[str, Any]:
        failures = [
            e
            for e in self.recovery_commit_materialization_events
            if int(e.get("source_frame_id", -1) or -1) == int(source_frame_id)
            and (
                "inliers_too_few" in str(e.get("failure_reason", ""))
                or "inliers_too_few" in str(e.get("materialization_failure_reason", ""))
            )
        ]
        if not failures:
            return {
                "source_pose_fail_count": 0,
                "source_last_pose_fail_reason": "",
                "keyframes_since_last_pose_fail": 999,
            }
        last = failures[-1]
        last_kf_count = int(last.get("keyframe_count_at_failure", self._current_keyframe_count()) or 0)
        return {
            "source_pose_fail_count": len(failures),
            "source_last_pose_fail_reason": str(last.get("failure_reason", "") or last.get("materialization_failure_reason", "")),
            "keyframes_since_last_pose_fail": max(0, self._current_keyframe_count() - last_kf_count),
        }

    def decide_recovered_commit(
        self,
        recovered: dict[str, Any],
        current_tick_frame_id: int,
    ) -> dict[str, Any]:
        source_frame_id = int(recovered.get("source_frame_id", -1))
        source_input_index = int(recovered.get("source_input_index", source_frame_id))
        source_event = self._get_event(source_frame_id) or {}
        prev_committed_source, prev_committed_action = self._last_committed_source_before(source_input_index)
        if prev_committed_source >= 0:
            source_gap_to_last_committed = max(0, source_input_index - prev_committed_source)
            current_open_gap = max(0, int(current_tick_frame_id) - prev_committed_source)
        else:
            source_gap_to_last_committed = 999
            current_open_gap = 999
        predicted_gap_if_hold = max(
            source_gap_to_last_committed + int(self.recovery_commit_controller.retry_interval),
            current_open_gap,
        )
        episode_trigger = max(
            int(getattr(self.recovery_commit_controller, "v2_source_gap_trigger", 0)),
            int(getattr(self.recovery_commit_controller, "v4_gap_trigger", 0)),
            int(getattr(self.recovery_commit_controller, "v5_gap_trigger", 0)),
        )
        long_gap_episode_id = prev_committed_source if current_open_gap >= int(episode_trigger) else 0
        source_num_inliers = max(
            int(source_event.get("num_pnp_inliers", 0) or 0),
            int(source_event.get("num_miniba_inliers", 0) or 0),
        )
        source_action = str(source_event.get("action", ""))
        is_surrogate = bool(source_frame_id == int(current_tick_frame_id))
        is_contamination_risk = bool(source_action not in {"defer_recoverable", "direct_admit"})
        anchor_count_before = self._current_anchor_count()
        growth_short = self._keyframe_growth_recent(
            current_tick_frame_id,
            window=int(getattr(self.recovery_commit_controller, "v5_growth_window_short", 100)),
        )
        growth_long = self._keyframe_growth_recent(
            current_tick_frame_id,
            window=int(getattr(self.recovery_commit_controller, "v5_growth_window_long", 200)),
        )
        density_now = self._current_keyframe_density(current_tick_frame_id)
        mat_stats = self._recent_materialization_stats(current_tick_frame_id, window=80)
        source_pose_fail = self._source_pose_fail_context(source_frame_id)
        context = {
            "current_tick_frame_id": int(current_tick_frame_id),
            "recent_recovery_commit_count": self._recent_recovery_commit_count(current_tick_frame_id),
            "keyframe_density_per_100": density_now,
            "current_keyframe_count": self._current_keyframe_count(),
            "main_chain_gap_p90_recent": self._main_chain_gap_p90_recent(),
            "keyframe_growth_recent": self._keyframe_growth_recent(current_tick_frame_id, window=100),
            "recent_keyframe_growth_short": growth_short,
            "recent_keyframe_growth_long": growth_long,
            "starvation_risk": bool(
                growth_short
                <= int(getattr(self.recovery_commit_controller, "v5_growth_plateau_min_short", 10))
                and density_now
                < float(getattr(self.recovery_commit_controller, "v5_density_lower", 28.0))
            ),
            "source_already_committed": self._source_already_committed(
                source_frame_id
            ),
            "source_gap_to_last_committed": source_gap_to_last_committed,
            "predicted_gap_if_hold": predicted_gap_if_hold,
            "long_gap_episode_id": long_gap_episode_id,
            "source_committed_action": prev_committed_action,
            "anchor_count_before": anchor_count_before,
            "anchor_count_available": bool(anchor_count_before is not None),
            "source_num_inliers": int(source_num_inliers),
            "is_surrogate": is_surrogate,
            "is_contamination_risk": is_contamination_risk,
            "open_gap_unclosed": bool(current_open_gap > int(episode_trigger)),
            **mat_stats,
            **source_pose_fail,
        }
        decision = self.recovery_commit_controller.decide(recovered, context)
        payload = {
            "source_frame_id": source_frame_id,
            "current_frame_id": int(current_tick_frame_id),
            "source_input_index": source_input_index,
            "current_tick_frame_id": int(current_tick_frame_id),
            "pool_enter_tick": int(recovered.get("pool_enter_tick", -1)),
            "recovery_attempt_tick": int(recovered.get("recovery_attempt_tick", -1)),
            "recovery_attempt_count": int(recovered.get("recovery_attempt_count", 0)),
            "control_mode": self.recovery_commit_control_mode,
            "decision": decision.action,
            "decision_reason": decision.reason,
            "control_decision": "allow_commit" if decision.action == "commit" else decision.action,
            "source_gap_to_last_committed": source_gap_to_last_committed,
            "predicted_gap_if_hold": predicted_gap_if_hold,
            "long_gap_episode_id": long_gap_episode_id,
            "episode_override_count": int(decision.debug.get("episode_override_count", 0)),
            "density_before": float(decision.debug.get("density_before", 0.0)),
            "density_after": float(decision.debug.get("density_after", 0.0)),
            "override_reason": str(decision.debug.get("override_reason", "")),
            "commit_channel": str(decision.debug.get("commit_channel", "")),
            "R_t": float(decision.debug.get("R_t", 0.0)),
            "V_t": float(decision.debug.get("V_t", 0.0)),
            "Q_t": float(decision.debug.get("Q_t", 0.0)),
            "num_matches": int(decision.debug.get("num_matches", 0) or 0),
            "num_inliers": int(decision.debug.get("num_inliers", 0) or 0),
            "window_id": int(decision.debug.get("window_id", 0) or 0),
            "window_candidate_rank": int(decision.debug.get("window_candidate_rank", 0) or 0),
            "window_support_score": float(decision.debug.get("window_support_score", 0.0) or 0.0),
            "support_score": float(decision.debug.get("support_score", 0.0) or 0.0),
            "materialization_feasibility_score": float(
                decision.debug.get("materialization_feasibility_score", 0.0) or 0.0
            ),
            "feature_missing": str(decision.debug.get("feature_missing", "")),
            "window_normal_commit_count": int(decision.debug.get("window_normal_commit_count", 0) or 0),
            "window_gap_critical_commit_count": int(
                decision.debug.get("window_gap_critical_commit_count", 0) or 0
            ),
            "window_budget_used": int(decision.debug.get("window_budget_used", 0) or 0),
            "rescue_budget_used": int(decision.debug.get("rescue_budget_used", 0) or 0),
            "gap_rescue_budget_used": int(decision.debug.get("gap_rescue_budget_used", 0) or 0),
            "density_state": str(decision.debug.get("density_state", "")),
            "expected_min_keyframes": float(decision.debug.get("expected_min_keyframes", 0.0) or 0.0),
            "keyframe_deficit": float(decision.debug.get("keyframe_deficit", 0.0) or 0.0),
            "recent_keyframe_growth": int(decision.debug.get("recent_keyframe_growth", 0) or 0),
            "growth_plateau": bool(decision.debug.get("growth_plateau", False)),
            "rescue_channel": str(decision.debug.get("rescue_channel", "")),
            "anchor_count_before": int(decision.debug.get("anchor_count_before", 0) or 0),
            "anchor_count_after_if_available": decision.debug.get("anchor_count_after_if_available", None),
            "anchor_guard_triggered": bool(decision.debug.get("anchor_guard_triggered", False)),
            "anchor_guard_action": str(decision.debug.get("anchor_guard_action", "")),
            "anchor_target_state": str(decision.debug.get("anchor_target_state", "")),
            "anchor_override_reason": str(decision.debug.get("anchor_override_reason", "")),
            "retry_limit_extended": bool(decision.debug.get("retry_limit_extended", False)),
            "retry_extension_reason": str(decision.debug.get("retry_extension_reason", "")),
            "budget_override_reason": str(decision.debug.get("budget_override_reason", "")),
            "coverage_rescue_triggered": bool(decision.debug.get("coverage_rescue_triggered", False)),
            "gap_rescue_triggered": bool(decision.debug.get("gap_rescue_triggered", False)),
            "blocked_by_hard_semantics": bool(decision.debug.get("blocked_by_hard_semantics", False)),
            "attempt_budget_used": int(decision.debug.get("attempt_budget_used", 0) or 0),
            "materialized_budget_used": int(decision.debug.get("materialized_budget_used", 0) or 0),
            "attempt_failed_no_materialization": bool(
                decision.debug.get("attempt_failed_no_materialization", False)
            ),
            "pose_fail_count": int(decision.debug.get("pose_fail_count", 0) or 0),
            "last_pose_fail_reason": str(decision.debug.get("last_pose_fail_reason", "")),
            "cooldown_active": bool(decision.debug.get("cooldown_active", False)),
            "cooldown_release_reason": str(decision.debug.get("cooldown_release_reason", "")),
            "gap_candidate_pose_infeasible": bool(
                decision.debug.get("gap_candidate_pose_infeasible", False)
            ),
            "gap_unfillable_due_to_pose_support": bool(
                decision.debug.get("gap_unfillable_due_to_pose_support", False)
            ),
            "early_coverage_rescue": bool(decision.debug.get("early_coverage_rescue", False)),
            "recent_materialization_rate": float(
                decision.debug.get("recent_materialization_rate", 0.0) or 0.0
            ),
            "recent_pose_fail_rate": float(decision.debug.get("recent_pose_fail_rate", 0.0) or 0.0),
            "keyframe_growth_recent": int(decision.debug.get("keyframe_growth_recent", 0) or 0),
            "starvation_risk": bool(decision.debug.get("starvation_risk", False)),
            "is_gap_critical": bool(decision.debug.get("is_gap_critical", False)),
            "is_coverage_floor": bool(decision.debug.get("is_coverage_floor", False)),
            "is_support_topk": bool(decision.debug.get("is_support_topk", False)),
            "is_coverage_sparse": bool(decision.debug.get("is_coverage_sparse", False)),
            "is_duplicate": bool(decision.debug.get("is_duplicate", False)),
            "is_surrogate": bool(decision.debug.get("is_surrogate", False)),
            "is_contamination_risk": bool(decision.debug.get("is_contamination_risk", False)),
            "blocked_reason": str(decision.debug.get("blocked_reason", "")),
            "runtime_commit_attempted": False,
            "runtime_commit_success": False,
            "source_resolution_success": False,
            "add_keyframe_attempted": False,
            "add_keyframe_success": False,
            "materialized": False,
            "final_timeline_recorded": False,
            "materialization_failure_reason": "",
            "debug": decision.debug,
        }
        self.recovery_commit_control_events.append(payload)
        return payload

    def hold_recovered_source(
        self,
        recovered: dict[str, Any],
        current_tick_frame_id: int,
        reason: str,
    ) -> None:
        item = dict(recovered)
        retries = int(item.get("_hold_retries", 0)) + 1
        item["_hold_retries"] = retries
        item["_hold_reason"] = str(reason)
        item["_next_retry_tick"] = int(current_tick_frame_id) + int(
            self.recovery_commit_controller.retry_interval
        )
        self._held_true_source_commits.append(item)

    def reject_recovered_source(self, recovered: dict[str, Any], reason: str) -> None:
        event = self._get_event(int(recovered.get("source_frame_id", -1)))
        if event is not None:
            event["source_recovery_commit"] = True
            event["source_recovery_payload_present"] = True
            event["source_recovery_committed"] = False
            event["source_recovery_failure_reason"] = str(reason)

    def decide(
        self,
        frame_id: int,
        info: dict[str, Any],
        baseline_should_add: bool,
        phase: str = "incremental",
        evidence: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        action = "direct_admit" if baseline_should_add else "discard"
        decision_meta: dict[str, Any] = {}

        if self.mode == "paper_aligned_baseline_passthrough":
            action = "direct_admit" if baseline_should_add else "discard"
        elif self.mode == "paper_aligned_semantic_v1":
            assert self.semantic_policy is not None
            semantic_input = dict(evidence or {})
            semantic = self.semantic_policy.decide(
                frame_id=frame_id,
                baseline_should_add=baseline_should_add,
                evidence=semantic_input,
                source_payload=self._source_payload(frame_id, info, semantic_input),
            )
            action = str(semantic["action"])
            bridge_tag = "none"
            if (
                self.recovery_bridge_mode == "semantic_surrogate"
                and action != "direct_admit"
                and int((semantic.get("recovery_tick", {}) or {}).get("success", 0)) > 0
                and bool(baseline_should_add)
            ):
                action = "current_frame_surrogate_commit"
                bridge_tag = "semantic_surrogate"

            if self.recovery_bridge_mode == "true_source_commit":
                for recovered in self.semantic_policy.pop_recovered_sources():
                    recovered["current_tick_frame_id"] = int(frame_id)
                    recovered["current_tick_image_name"] = str(info.get("image_name", ""))
                    recovered["bridge_type"] = "true_source_commit"
                    self._pending_true_source_commits.append(recovered)

            decision_meta = {
                "R_t": semantic["scores"]["R_t"],
                "V_t": semantic["scores"]["V_t"],
                "Q_t": semantic["scores"]["Q_t"],
                "C_t": semantic["scores"]["C_t"],
                "B_R_t": semantic["scores"]["B_R_t"],
                "recovery_pool_size": semantic["recovery_pool_size"],
                "recovery_tick": semantic["recovery_tick"],
                "recovery_bridge_tag": bridge_tag,
                "thresholds": semantic["thresholds"],
            }
            if self.coupled_config.enabled:
                decision_meta["coupled_model"] = self.coupled_config.model_name
                decision_meta["module_switches"] = self.coupled_config.module_switches()

        admit = action in ("direct_admit", "current_frame_surrogate_commit")
        event = {
            "frame_id": int(frame_id),
            "image_name": str(info.get("image_name", "")),
            "is_test": _to_bool(info.get("is_test", False)),
            "phase_at_decision": phase,
            "baseline_should_add": bool(baseline_should_add),
            "action": action,
            "admit_to_chain": admit,
            "mode": self.mode,
            "requested_mode": self.requested_mode,
            "coupled_model_enabled": bool(self.coupled_config.enabled),
            "decision_meta": decision_meta,
            "pose_init_attempted": False,
            "pose_init_success": None,
            "pose_fail_detail": "",
            "num_2d3d_correspondences": None,
            "num_pnp_inliers": None,
            "num_miniba_inliers": None,
            "keyframe_add_called": False,
            "gaussian_update_called": False,
            "anchor_update_called": False,
            "final_keyframe_incremented": False,
            "drop_reason": "",
            "source_recovery_commit": False,
            "source_recovery_payload_present": False,
            "source_recovery_committed": False,
            "source_recovery_failure_reason": "",
        }
        self._event_index[int(frame_id)] = len(self.trace_events)
        self.trace_events.append(event)
        return admit, action

    def mark_pose_attempt(self, frame_id: int) -> None:
        event = self._get_event(frame_id)
        if event is not None:
            event["pose_init_attempted"] = True

    def mark_pose_result(self, frame_id: int, success: bool) -> None:
        event = self._get_event(frame_id)
        if event is not None:
            event["pose_init_attempted"] = True
            event["pose_init_success"] = bool(success)
            if not success and not event.get("drop_reason"):
                event["drop_reason"] = "pose_init_failed"

    def annotate_pose_debug(self, frame_id: int, debug: dict[str, Any]) -> None:
        event = self._get_event(frame_id)
        if event is None or not isinstance(debug, dict):
            return
        event["num_2d3d_correspondences"] = debug.get("num_2d3d_correspondences")
        event["num_pnp_inliers"] = debug.get("num_pnp_inliers")
        event["num_miniba_inliers"] = debug.get("num_miniba_inliers")
        detail = str(debug.get("failure_reason", "") or "")
        if detail:
            event["pose_fail_detail"] = detail

    def mark_keyframe_add(self, frame_id: int) -> None:
        event = self._get_event(frame_id)
        if event is not None:
            event["keyframe_add_called"] = True

    def mark_gaussian_update(self, frame_id: int) -> None:
        event = self._get_event(frame_id)
        if event is not None:
            event["gaussian_update_called"] = True

    def mark_anchor_update(self, frame_id: int) -> None:
        event = self._get_event(frame_id)
        if event is not None:
            event["anchor_update_called"] = True

    def mark_final_keyframe_increment(self, frame_id: int) -> None:
        event = self._get_event(frame_id)
        if event is not None:
            event["final_keyframe_incremented"] = True

    def mark_drop_reason(self, frame_id: int, reason: str) -> None:
        event = self._get_event(frame_id)
        if event is not None and reason:
            event["drop_reason"] = str(reason)
            if not event.get("pose_fail_detail"):
                event["pose_fail_detail"] = str(reason)

    def mark_true_source_recovery_attempt(
        self,
        source_frame_id: int,
        current_tick_frame_id: int,
        current_tick_image_name: str,
        pool_enter_tick: int,
        recovery_attempt_tick: int,
    ) -> None:
        event = self._get_event(source_frame_id)
        if event is not None:
            event["source_recovery_commit"] = True
            event["source_recovery_payload_present"] = True
            event["source_recovery_current_tick_frame_id"] = int(current_tick_frame_id)
            event["source_recovery_current_tick_image_name"] = str(current_tick_image_name)
            event["source_recovery_pool_enter_tick"] = int(pool_enter_tick)
            event["source_recovery_attempt_tick"] = int(recovery_attempt_tick)

    def mark_true_source_recovery_result(self, source_frame_id: int, committed: bool, reason: str = "") -> None:
        event = self._get_event(source_frame_id)
        if event is not None:
            event["source_recovery_committed"] = bool(committed)
            event["source_recovery_failure_reason"] = str(reason or "")
            if committed:
                event["source_recovery_commit"] = True
                event["source_recovery_payload_present"] = True

    def append_true_recovery_commit_event(self, payload: dict[str, Any]) -> None:
        self.true_recovery_commit_events.append(dict(payload))

    def append_recovery_commit_materialization_event(self, payload: dict[str, Any]) -> None:
        self.recovery_commit_materialization_events.append(dict(payload))

    def append_keyframe_timeline_event(self, payload: dict[str, Any]) -> None:
        self.keyframe_timeline_events.append(dict(payload))

    def append_chosen_kfs_reference_event(self, payload: dict[str, Any]) -> None:
        self.chosen_kfs_reference_events.append(dict(payload))

    def append_matching_support_event(self, payload: dict[str, Any]) -> None:
        self.matching_support_events.append(dict(payload))

    def append_pnp_miniba_reference_event(self, payload: dict[str, Any]) -> None:
        self.pnp_miniba_reference_events.append(dict(payload))

    def append_local_map_anchor_event(self, payload: dict[str, Any]) -> None:
        self.local_map_anchor_events.append(dict(payload))

    def append_support_integration_event(self, payload: dict[str, Any]) -> None:
        self.support_integration_events.append(dict(payload))

    def append_chosen_kfs_candidate_events(self, payloads: list[dict[str, Any]]) -> None:
        self.chosen_kfs_candidate_events.extend(dict(p) for p in payloads)

    def append_pose_reference_pool_events(self, payloads: list[dict[str, Any]]) -> None:
        self.pose_reference_pool_events.extend(dict(p) for p in payloads)

    def append_matching_to_pose_path_bridge_event(self, payload: dict[str, Any]) -> None:
        self.matching_to_pose_path_bridge_events.append(dict(payload))

    def append_lifecycle_gate_event(self, payload: dict[str, Any]) -> None:
        self.lifecycle_gate_events.append(dict(payload))

    def append_recovery_pose_path_event(self, payload: dict[str, Any]) -> None:
        self.recovery_pose_path_events.append(dict(payload))

    def append_frame_stage_reachability_event(self, payload: dict[str, Any]) -> None:
        self.frame_stage_reachability_events.append(dict(payload))

    def append_recovery_pose_outcome_fix_event(self, payload: dict[str, Any]) -> None:
        self.recovery_pose_outcome_fix_events.append(dict(payload))

    def append_recovery_2d3d_support_event(self, payload: dict[str, Any]) -> None:
        self.recovery_2d3d_support_events.append(dict(payload))

    def append_recovery_reference_3d_association_event(self, payload: dict[str, Any]) -> None:
        self.recovery_reference_3d_association_events.append(dict(payload))

    def append_recovery_pnp_consensus_event(self, payload: dict[str, Any]) -> None:
        self.recovery_pnp_consensus_events.append(dict(payload))

    def append_recovery_ref_subset_event(self, payload: dict[str, Any]) -> None:
        self.recovery_ref_subset_events.append(dict(payload))

    def append_anchor_transition_bridge_event(self, payload: dict[str, Any]) -> None:
        self.anchor_transition_bridge_events.append(dict(payload))

    def append_recovery_support_trace_event(self, payload: dict[str, Any]) -> None:
        self.recovery_support_trace_events.append(dict(payload))

    def decide_direct_finalization(
        self,
        *,
        frame_id: int,
        runtime_action: str,
        baseline_should_add: bool,
        is_test: bool,
        is_bootstrap_phase: bool,
        anchor_changed: bool,
        support_triggered: bool,
        median_displacement: float,
        displacement_threshold: float,
        num_matches: int,
        min_num_inliers: int,
        pose_inliers: int,
    ):
        last_tick, _ = self._last_committed_source_before(frame_id)
        source_gap = int(frame_id - last_tick) if last_tick >= 0 else int(frame_id)
        gap_before = float(self._main_chain_gap_p90_recent())
        density_before = float(self._current_keyframe_density(frame_id))
        local_density_before = density_before
        keyframe_growth_recent = int(self._keyframe_growth_recent(frame_id))
        ctrl = self.direct_density_controller
        expected_kf = max(
            1.0,
            float(frame_id)
            * float(getattr(ctrl, "baseline_density_per_100", 27.0))
            / 100.0,
        )
        baseline_relative_density = float(
            100.0
            * float(self._current_keyframe_count())
            / max(
                float(getattr(ctrl, "baseline_relative_lower_ratio", 0.8)) * expected_kf,
                1.0,
            )
        )
        disp_ratio = float(median_displacement / max(displacement_threshold, 1e-6))
        match_ratio = float(num_matches / max(2.0 * min_num_inliers, 1.0))
        novelty_proxy = min(1.0, 0.5 * min(disp_ratio, 2.0) + 0.5 * min(match_ratio, 2.0))
        local_stats = self._local_window_stats(
            frame_id,
            int(getattr(ctrl, "local_window_size", 100) or 100),
        )
        local_window_density = float(local_stats["local_window_density"])
        local_window_keyframes = int(local_stats["local_window_keyframes"])
        local_window_gap_max = float(local_stats["local_window_gap_max"])
        local_window_gap_after_if_hold = float(local_stats["local_window_gap_after_if_hold"])
        gap_after_hold = float(max(source_gap, gap_before, local_window_gap_after_if_hold))
        decision = self.direct_density_controller.decide(
            frame_id=int(frame_id),
            runtime_action=str(runtime_action),
            baseline_should_add=bool(baseline_should_add),
            is_test=bool(is_test),
            is_bootstrap_phase=bool(is_bootstrap_phase),
            density_before=density_before,
            local_density_before=local_window_density,
            keyframe_growth_recent=keyframe_growth_recent,
            baseline_relative_density=baseline_relative_density,
            source_gap_to_last_keyframe=source_gap,
            main_chain_gap_before=gap_before,
            main_chain_gap_after_if_hold=gap_after_hold,
            anchor_changed=bool(anchor_changed),
            support_triggered=bool(support_triggered),
            median_displacement=float(median_displacement),
            displacement_threshold=float(displacement_threshold),
            num_matches=int(num_matches),
            min_num_inliers=int(min_num_inliers),
            pose_inliers=int(pose_inliers),
            novelty_proxy=novelty_proxy,
            current_keyframe_count=int(self._current_keyframe_count()),
            local_window_density=local_window_density,
            local_window_keyframes=local_window_keyframes,
            local_window_gap_max=local_window_gap_max,
            local_window_gap_after_if_hold=local_window_gap_after_if_hold,
        )
        if ctrl.is_v22:
            decision.debug.update(
                {
                    "local_window_density": local_window_density,
                    "local_window_keyframes": local_window_keyframes,
                    "local_window_gap_max": local_window_gap_max,
                    "local_window_gap_after_if_hold": local_window_gap_after_if_hold,
                }
            )
        return decision

    def append_direct_density_control_event(self, payload: dict[str, Any]) -> None:
        self.direct_density_control_events.append(dict(payload))

    def append_direct_density_control_v2_event(self, payload: dict[str, Any]) -> None:
        self.direct_density_control_v2_events.append(dict(payload))

    def append_direct_density_control_v2_1_event(self, payload: dict[str, Any]) -> None:
        self.direct_density_control_v2_1_events.append(dict(payload))

    def append_direct_density_control_v2_2_event(self, payload: dict[str, Any]) -> None:
        self.direct_density_control_v2_2_events.append(dict(payload))

    def append_direct_density_control_v2_2_1_event(self, payload: dict[str, Any]) -> None:
        self.direct_density_control_v2_2_1_events.append(dict(payload))

    def append_direct_density_control_v2_2_2_event(self, payload: dict[str, Any]) -> None:
        self.direct_density_control_v2_2_2_events.append(dict(payload))

    def append_direct_density_control_v2_2_2_1_event(self, payload: dict[str, Any]) -> None:
        self.direct_density_control_v2_2_2_1_events.append(dict(payload))

    def flush_trace(self) -> None:
        if not self.trace_path:
            return
        out = Path(self.trace_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "mode": self.mode,
            "requested_mode": self.requested_mode,
            "coupled_innovation_config": self.coupled_config.to_trace_dict() if self.coupled_config.enabled else {},
            "stage_metric_contract": self.coupled_config.stage_metric_contract() if self.coupled_config.enabled else {
                "online_quality_metric_fields": [],
                "offline_stage_metric_fields": [],
            },
            "num_events": len(self.trace_events),
            "direct_admit": int(sum(1 for e in self.trace_events if e.get("action") == "direct_admit")),
            "true_recovery_commit": int(sum(1 for e in self.trace_events if e.get("source_recovery_committed", False))),
            "recovery_signal_bridge": int(
                sum(1 for e in self.trace_events if e.get("action") == "current_frame_surrogate_commit")
            ),
            "defer_recoverable": int(sum(1 for e in self.trace_events if e.get("action") == "defer_recoverable")),
            "discard": int(sum(1 for e in self.trace_events if e.get("action") == "discard")),
            "direct_not_finalized": int(
                sum(
                    1
                    for e in self.trace_events
                    if e.get("action") in ("direct_admit", "current_frame_surrogate_commit")
                    and not e.get("final_keyframe_incremented", False)
                )
            ),
            "events": self.trace_events,
            "true_recovery_commit_events": self.true_recovery_commit_events,
            "recovery_commit_control_mode": self.recovery_commit_control_mode,
            "recovery_commit_control_events": self.recovery_commit_control_events,
            "recovery_commit_materialization_events": self.recovery_commit_materialization_events,
            "keyframe_timeline_events": self.keyframe_timeline_events,
            "chosen_kfs_reference_events": self.chosen_kfs_reference_events,
            "matching_support_events": self.matching_support_events,
            "pnp_miniba_reference_events": self.pnp_miniba_reference_events,
            "local_map_anchor_events": self.local_map_anchor_events,
            "support_integration_events": self.support_integration_events,
            "chosen_kfs_candidate_events": self.chosen_kfs_candidate_events,
            "pose_reference_pool_events": self.pose_reference_pool_events,
            "matching_to_pose_path_bridge_events": self.matching_to_pose_path_bridge_events,
            "lifecycle_gate_events": self.lifecycle_gate_events,
            "recovery_pose_path_events": self.recovery_pose_path_events,
            "frame_stage_reachability_events": self.frame_stage_reachability_events,
            "recovery_pose_outcome_fix_events": self.recovery_pose_outcome_fix_events,
            "recovery_2d3d_support_events": self.recovery_2d3d_support_events,
            "recovery_reference_3d_association_events": self.recovery_reference_3d_association_events,
            "recovery_pnp_consensus_events": self.recovery_pnp_consensus_events,
            "recovery_ref_subset_events": self.recovery_ref_subset_events,
            "anchor_transition_bridge_events": self.anchor_transition_bridge_events,
            "recovery_support_trace_events": self.recovery_support_trace_events,
            "direct_density_control_mode": str(
                getattr(self.direct_density_controller, "mode", "off")
            ),
            "direct_density_control_events": self.direct_density_control_events,
            "direct_density_control_v2_events": self.direct_density_control_v2_events,
            "direct_density_control_v2_1_events": self.direct_density_control_v2_1_events,
            "direct_density_control_v2_2_events": self.direct_density_control_v2_2_events,
            "direct_density_control_v2_2_1_events": self.direct_density_control_v2_2_1_events,
            "direct_density_control_v2_2_2_events": self.direct_density_control_v2_2_2_events,
            "direct_density_control_v2_2_2_1_events": self.direct_density_control_v2_2_2_1_events,
            "trace_unavailable_reasons": self.trace_unavailable_reasons,
        }
        if self.semantic_policy is not None:
            payload["semantic_summary"] = self.semantic_policy.summary()
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
