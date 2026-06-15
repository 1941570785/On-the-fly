from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
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
        self.viewpoint_coverage_events: list[dict[str, Any]] = []
        self.pose_only_reference_pool: list[Any] = []
        self.pose_only_reference_pool_max_size = 32
        self.pose_only_reference_ttl_frames = 180
        self.pose_only_reference_min_age_frames = 8
        self.pose_only_reference_min_3d_points = 400
        self.pose_only_reference_max_per_query = 1
        self.pose_only_reference_min_match_score = 180.0
        self.pose_only_reference_register_min_interval_frames = 0
        self.pose_only_reference_selection_cooldown_frames = 0
        self.pose_only_reference_age_bonus = 0.0
        self.pose_only_reference_selection_strategy = "match_age"
        self.pose_only_reference_risk_gate_enabled = False
        self.pose_only_reference_min_support_concentration = 0.065
        self.pose_only_reference_low_new_view_max = 0.18
        self.pose_only_reference_high_new_view_min = 0.20
        self.pose_only_reference_high_new_view_support_min = 0.10
        self.pose_only_reference_high_new_view_anchor_health_max = 0.70
        self.pose_only_reference_high_new_view_entropy_max = 1.01
        self.pose_only_reference_early_turn_bridge_frame_max = 0
        self.pose_only_reference_late_extreme_new_view_frame_min = 10**9
        self.pose_only_reference_late_extreme_new_view_min = 1.01
        self.pose_only_reference_late_extreme_new_view_support_min = 0.0
        self.pose_only_reference_growth_stall_max = 0
        self.pose_only_reference_require_negative_growth = False
        self.pose_only_reference_allow_high_new_view_rescue = True
        self.pose_only_reference_repetitive_entropy_min = 1.01
        self.pose_only_reference_repetitive_entropy_support_max = 0.0
        self._last_pose_only_reference_register_frame = -1
        self.pose_only_reference_registered_count = 0
        self.pose_only_reference_selected_count = 0
        self.trace_unavailable_reasons: dict[str, str] = {
            "match_graph_neighbor_ids": "No persistent match graph object is exposed; pairwise matches live on DescribedKeypoints.matches.",
            "match_graph_id": "No stable match graph id exists for keyframes in the current runtime.",
        }
        self.semantic_policy: SemanticV1RuntimePolicy | None = None
        self.recovery_commit_controller = RecoveryCommitController(args)
        self.direct_density_controller = DirectDensityController(args)
        if (
            getattr(self.direct_density_controller, "is_pose_rep_active_memory_v2", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v4", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v5", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v6", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v8", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v9", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v24", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v25", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v26", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v30", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v31", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v33", False)
        ):
            self.pose_only_reference_pool_max_size = 24
            self.pose_only_reference_ttl_frames = 140
            self.pose_only_reference_min_age_frames = 18
            self.pose_only_reference_min_match_score = 220.0
            self.pose_only_reference_register_min_interval_frames = 12
            self.pose_only_reference_selection_cooldown_frames = 8
            self.pose_only_reference_age_bonus = 30.0
        if getattr(self.direct_density_controller, "is_pose_rep_active_memory_v4", False):
            self.pose_only_reference_pool_max_size = 16
            self.pose_only_reference_ttl_frames = 120
            self.pose_only_reference_min_age_frames = 24
            self.pose_only_reference_min_match_score = 240.0
            self.pose_only_reference_register_min_interval_frames = 18
            self.pose_only_reference_selection_cooldown_frames = 12
            self.pose_only_reference_age_bonus = 20.0
            self.pose_only_reference_risk_gate_enabled = True
        if getattr(self.direct_density_controller, "is_pose_rep_active_memory_v5", False):
            self.pose_only_reference_pool_max_size = 12
            self.pose_only_reference_ttl_frames = 100
            self.pose_only_reference_min_age_frames = 28
            self.pose_only_reference_min_match_score = 260.0
            self.pose_only_reference_register_min_interval_frames = 24
            self.pose_only_reference_selection_cooldown_frames = 16
            self.pose_only_reference_age_bonus = 12.0
            self.pose_only_reference_risk_gate_enabled = True
            self.pose_only_reference_min_support_concentration = 0.08
            self.pose_only_reference_low_new_view_max = 0.165
            self.pose_only_reference_high_new_view_min = 0.20
            self.pose_only_reference_high_new_view_support_min = 0.11
            self.pose_only_reference_high_new_view_anchor_health_max = 0.70
        if (
            getattr(self.direct_density_controller, "is_pose_rep_active_memory_v6", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v8", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v9", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v24", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v25", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v26", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v30", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v31", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v33", False)
        ):
            self.pose_only_reference_pool_max_size = 12
            self.pose_only_reference_ttl_frames = 100
            self.pose_only_reference_min_age_frames = 28
            self.pose_only_reference_min_match_score = 260.0
            self.pose_only_reference_register_min_interval_frames = 24
            self.pose_only_reference_selection_cooldown_frames = 16
            self.pose_only_reference_age_bonus = 12.0
            self.pose_only_reference_risk_gate_enabled = True
            self.pose_only_reference_min_support_concentration = 0.08
            self.pose_only_reference_low_new_view_max = 0.165
            self.pose_only_reference_high_new_view_min = 0.20
            self.pose_only_reference_high_new_view_support_min = 0.11
            self.pose_only_reference_high_new_view_anchor_health_max = 0.70
            self.pose_only_reference_require_negative_growth = True
            self.pose_only_reference_allow_high_new_view_rescue = False
            self.pose_only_reference_repetitive_entropy_min = 0.94
            self.pose_only_reference_repetitive_entropy_support_max = 0.08
        if (
            getattr(self.direct_density_controller, "is_pose_rep_active_memory_v9", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v24", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v25", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v26", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v30", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v31", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v33", False)
        ):
            self.pose_only_reference_allow_high_new_view_rescue = True
            self.pose_only_reference_high_new_view_support_min = 0.10
            self.pose_only_reference_high_new_view_anchor_health_max = 0.70
            self.pose_only_reference_high_new_view_entropy_max = 0.92
        if getattr(self.direct_density_controller, "is_pose_rep_active_memory_v24", False):
            self.pose_only_reference_max_per_query = 2
            self.pose_only_reference_min_match_score = 280.0
            self.pose_only_reference_selection_cooldown_frames = 12
            self.pose_only_reference_age_bonus = 8.0
            self.pose_only_reference_selection_strategy = "risk_aware"
        if (
            getattr(self.direct_density_controller, "is_pose_rep_active_memory_v25", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v30", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v31", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v33", False)
        ):
            self.pose_only_reference_max_per_query = 1
            self.pose_only_reference_min_match_score = 280.0
            self.pose_only_reference_selection_cooldown_frames = 12
            self.pose_only_reference_age_bonus = 8.0
            self.pose_only_reference_selection_strategy = "risk_aware"
        if (
            getattr(self.direct_density_controller, "is_pose_rep_active_memory_v26", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v30", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v31", False)
            or getattr(self.direct_density_controller, "is_pose_rep_active_memory_v33", False)
        ):
            self.pose_only_reference_late_extreme_new_view_frame_min = 1200
            self.pose_only_reference_late_extreme_new_view_min = 0.35
            self.pose_only_reference_late_extreme_new_view_support_min = 0.15
        self._anchor_count_at_last_direct_finalize = 1
        if self.mode == "paper_aligned_semantic_v1":
            cfg = self.coupled_config
            self.semantic_policy = SemanticV1RuntimePolicy(
                thresholds=cfg.thresholds,
                recovery_delay_frames=cfg.recovery_delay_frames,
                recovery_max_attempts=cfg.recovery_max_attempts,
                recovery_attempts_per_tick=cfg.recovery_attempts_per_tick,
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
        source_payload = recovered.get("source_payload", {}) or {}
        is_density_hold_recovery = bool(
            source_event.get("density_hold_recovery_enqueued")
            or source_payload.get("density_hold_context")
        )
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
            "is_density_hold_recovery": is_density_hold_recovery,
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

    def enqueue_density_hold_recovery_candidate(
        self,
        *,
        frame_id: int,
        info: dict[str, Any],
        evidence: dict[str, Any] | None = None,
        hold_decision: str = "",
        hold_reason: str = "",
        density_debug: dict[str, Any] | None = None,
    ) -> bool:
        event = self._get_event(int(frame_id))
        if self.semantic_policy is None or event is None:
            return False

        decision_meta = event.get("decision_meta")
        if not isinstance(decision_meta, dict):
            decision_meta = {}
            event["decision_meta"] = decision_meta

        if str(event.get("action", "")) not in {"direct_admit", "current_frame_surrogate_commit"}:
            event["density_hold_recovery_enqueued"] = False
            event["density_hold_recovery_block_reason"] = "not_direct_candidate"
            return False
        if bool(event.get("final_keyframe_incremented", False)):
            event["density_hold_recovery_enqueued"] = False
            event["density_hold_recovery_block_reason"] = "already_finalized"
            return False

        scores = {
            "R_t": float(decision_meta.get("R_t", 1.0) or 0.0),
            "V_t": float(decision_meta.get("V_t", 0.0) or 0.0),
            "B_R_t": float(decision_meta.get("B_R_t", 0.0) or 0.0),
            "Q_t": float(decision_meta.get("Q_t", 0.0) or 0.0),
        }
        th = self.semantic_policy.th
        recoverable = bool(
            scores["R_t"] <= float(th.tau_R_high)
            and scores["B_R_t"] >= float(th.tau_B)
            and scores["V_t"] >= float(th.tau_V_min)
            and scores["Q_t"] >= float(th.tau_Q)
        )
        if not recoverable:
            event["density_hold_recovery_enqueued"] = False
            event["density_hold_recovery_block_reason"] = "semantic_scores_not_recoverable"
            return False

        source_payload = self._source_payload(int(frame_id), info, dict(evidence or {}))
        source_payload["density_hold_context"] = {
            "hold_decision": str(hold_decision),
            "hold_reason": str(hold_reason),
            "density_debug": dict(density_debug or {}),
        }
        enqueued = self.semantic_policy.enqueue_density_hold_candidate(
            int(frame_id),
            scores,
            source_payload=source_payload,
            hold_reason=str(hold_reason),
        )
        tag = "density_hold_recoverable" if enqueued else "density_hold_duplicate"
        event["direct_admit_but_held_for_density"] = True
        event["density_hold_recovery_enqueued"] = bool(enqueued)
        event["density_hold_recovery_bridge_tag"] = tag
        event["density_hold_recovery_hold_decision"] = str(hold_decision)
        event["density_hold_recovery_hold_reason"] = str(hold_reason)
        decision_meta["density_hold_recovery_bridge_tag"] = tag
        decision_meta["recovery_pool_size"] = len(self.semantic_policy.recovery_pool)
        decision_meta["density_hold_recovery_enqueued"] = bool(enqueued)
        return bool(enqueued)

    def is_recovery_pose_path_candidate(self, frame_id: int) -> bool:
        event = self._get_event(int(frame_id))
        if event is None:
            return False
        action = str(event.get("action", ""))
        if action == "defer_recoverable":
            return True
        return bool(
            action == "direct_admit"
            and event.get("direct_admit_but_held_for_density")
            and event.get("density_hold_recovery_enqueued")
        )

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
        semantic_scores: dict[str, Any] | None = None,
        viewpoint_scores: dict[str, Any] | None = None,
    ):
        if semantic_scores is None:
            event = self._get_event(int(frame_id))
            meta = event.get("decision_meta", {}) if isinstance(event, dict) else {}
            if isinstance(meta, dict):
                semantic_scores = {
                    "R_t": meta.get("R_t"),
                    "V_t": meta.get("V_t"),
                    "Q_t": meta.get("Q_t"),
                    "C_t": meta.get("C_t"),
                    "B_R_t": meta.get("B_R_t"),
                }
            else:
                semantic_scores = {}
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
            semantic_scores=semantic_scores,
            viewpoint_scores=viewpoint_scores,
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

    def append_viewpoint_coverage_event(self, payload: dict[str, Any]) -> None:
        self.viewpoint_coverage_events.append(dict(payload))

    def _pose_only_pool_enabled(self) -> bool:
        return bool(
            getattr(self.direct_density_controller, "is_pose_rep_active_memory", False)
        )

    def _clone_pose_only_value(self, value: Any) -> Any:
        if hasattr(value, "detach") and hasattr(value, "clone"):
            return value.detach().clone()
        if hasattr(value, "clone"):
            return value.clone()
        return copy.deepcopy(value)

    def _clone_pose_only_desc(self, desc_kpts: Any) -> Any:
        desc = copy.copy(desc_kpts)
        for name in ("kpts", "feats", "valid", "has_pt3d", "pts_conf", "pts3d", "depth"):
            if hasattr(desc_kpts, name):
                setattr(desc, name, self._clone_pose_only_value(getattr(desc_kpts, name)))
        desc.matches = {}
        return desc

    def _pose_only_support_count(self, desc_kpts: Any) -> int:
        has_pt3d = getattr(desc_kpts, "has_pt3d", None)
        if has_pt3d is None:
            return 0
        try:
            total = has_pt3d.sum()
            return int(total.item() if hasattr(total, "item") else total)
        except Exception:
            return 0

    def _tensor_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if hasattr(value, "item"):
                return float(value.item())
            return float(value)
        except Exception:
            return float(default)

    def _pose_only_reference_info_metric(
        self,
        ref: Any,
        name: str,
        default: float = 0.0,
    ) -> float:
        info = getattr(ref, "info", {}) or {}
        return self._tensor_float(
            info.get(f"_paper_aligned_pose_only_{name}", info.get(name, default)),
            default,
        )

    def _pose_only_reference_geometry_score(self, ref: Any, support_count: int, age: int) -> float:
        support_concentration = self._pose_only_reference_info_metric(
            ref, "support_concentration", 0.0
        )
        new_view = self._pose_only_reference_info_metric(
            ref, "new_view_event_score", 0.0
        )
        anchor_health = self._pose_only_reference_info_metric(
            ref, "anchor_health_score", 0.0
        )
        entropy = self._pose_only_reference_info_metric(
            ref, "inlier_grid_entropy", 0.0
        )
        pnp_inliers = self._pose_only_reference_info_metric(ref, "pnp_inliers", 0.0)
        miniba_inliers = self._pose_only_reference_info_metric(ref, "miniba_inliers", 0.0)
        support_ratio = min(2.0, float(support_count) / max(float(self.pose_only_reference_min_3d_points), 1.0))
        age_norm = min(float(age), float(self.pose_only_reference_ttl_frames)) / max(
            float(self.pose_only_reference_ttl_frames), 1.0
        )
        mid_age_bonus = max(0.0, 1.0 - abs(age_norm - 0.55))
        high_view_bridge = bool(
            new_view >= float(self.pose_only_reference_high_new_view_min)
            and support_concentration >= float(self.pose_only_reference_high_new_view_support_min)
            and anchor_health <= float(self.pose_only_reference_high_new_view_anchor_health_max)
            and entropy <= float(self.pose_only_reference_high_new_view_entropy_max)
        )
        low_new_view_context = new_view <= float(self.pose_only_reference_low_new_view_max)
        entropy_penalty = max(0.0, entropy - float(self.pose_only_reference_high_new_view_entropy_max))
        anchor_penalty = max(0.0, anchor_health - float(self.pose_only_reference_high_new_view_anchor_health_max))
        return float(
            260.0 * support_concentration
            + (70.0 if high_view_bridge else 0.0)
            + (8.0 if low_new_view_context else 0.0)
            + 8.0 * support_ratio
            + 0.020 * pnp_inliers
            + 0.015 * miniba_inliers
            + 18.0 * mid_age_bonus
            - 90.0 * entropy_penalty
            - 70.0 * anchor_penalty
        )

    def _purge_pose_only_references(self, frame_id: int) -> None:
        min_source_frame = int(frame_id) - int(self.pose_only_reference_ttl_frames)
        self.pose_only_reference_pool = [
            ref
            for ref in self.pose_only_reference_pool
            if int(ref.info.get("_paper_aligned_source_frame_id", -1)) >= min_source_frame
        ]

    def _pose_only_reference_risk_gate_decision(
        self, debug: dict[str, Any], frame_id: int | None = None
    ) -> tuple[bool, str, dict[str, float]]:
        support_concentration = self._tensor_float(debug.get("support_concentration"), 1.0)
        new_view_event = self._tensor_float(debug.get("new_view_event_score"), 0.0)
        anchor_health = self._tensor_float(debug.get("anchor_health_score"), 0.0)
        inlier_grid_entropy = self._tensor_float(debug.get("inlier_grid_entropy"), 0.0)
        viewpoint_rotation_window_max = self._tensor_float(
            debug.get("viewpoint_rotation_deg_window_max", debug.get("viewpoint_rotation_window_max", 0.0)),
            0.0,
        )
        frame_index = int(frame_id if frame_id is not None else debug.get("frame_id", -1) or -1)
        growth_recent = self._tensor_float(
            debug.get("keyframe_growth_recent", debug.get("recent_keyframe_growth", 0.0)),
            0.0,
        )
        metrics = {
            "pose_only_support_concentration": float(support_concentration),
            "pose_only_new_view_event_score": float(new_view_event),
            "pose_only_anchor_health_score": float(anchor_health),
            "pose_only_keyframe_growth_recent": float(growth_recent),
            "pose_only_inlier_grid_entropy": float(inlier_grid_entropy),
            "pose_only_viewpoint_rotation_window_max": float(viewpoint_rotation_window_max),
            "pose_only_frame_id": float(frame_index),
        }
        if bool(self.pose_only_reference_require_negative_growth):
            if growth_recent >= 0.0:
                return False, "pose_only_growth_not_stalled", metrics
        elif growth_recent > float(self.pose_only_reference_growth_stall_max):
            return False, "pose_only_growth_not_stalled", metrics
        if (
            inlier_grid_entropy >= float(self.pose_only_reference_repetitive_entropy_min)
            and support_concentration < float(self.pose_only_reference_repetitive_entropy_support_max)
        ):
            return False, "pose_only_repetitive_entropy_low_support", metrics
        if support_concentration < float(self.pose_only_reference_min_support_concentration):
            return False, "pose_only_low_support_diversity", metrics
        low_new_view_context = new_view_event <= float(self.pose_only_reference_low_new_view_max)
        high_new_view_candidate = new_view_event >= float(
            self.pose_only_reference_high_new_view_min
        )
        if high_new_view_candidate and not bool(
            self.pose_only_reference_allow_high_new_view_rescue
        ):
            return False, "pose_only_high_new_view_pnp_disabled", metrics
        if (
            frame_index >= int(self.pose_only_reference_late_extreme_new_view_frame_min)
            and new_view_event >= float(self.pose_only_reference_late_extreme_new_view_min)
            and support_concentration < float(self.pose_only_reference_late_extreme_new_view_support_min)
        ):
            return False, "pose_only_late_extreme_new_view_thin_support", metrics
        high_new_view_rescue = (
            high_new_view_candidate
            and support_concentration >= float(self.pose_only_reference_high_new_view_support_min)
            and anchor_health <= float(self.pose_only_reference_high_new_view_anchor_health_max)
            and inlier_grid_entropy <= float(self.pose_only_reference_high_new_view_entropy_max)
        )
        if low_new_view_context or high_new_view_rescue:
            return True, "", metrics
        if (
            high_new_view_candidate
            and anchor_health > float(self.pose_only_reference_high_new_view_anchor_health_max)
        ):
            return False, "pose_only_anchor_healthy_high_new_view", metrics
        return False, "pose_only_risk_gate_not_met", metrics

    def _apply_pose_support_to_desc(self, desc_kpts: Any, pose_support: dict[str, Any] | None) -> None:
        if not pose_support or not hasattr(desc_kpts, "update_3D_pts"):
            return
        match_indices = pose_support.get("match_indices")
        pts3d = pose_support.get("pts3d")
        pts_conf = pose_support.get("pts_conf")
        if match_indices is None or pts3d is None or pts_conf is None:
            return
        try:
            if hasattr(match_indices, "numel") and int(match_indices.numel()) == 0:
                return
            depth = pose_support.get("depth")
            if depth is None and hasattr(pts_conf, "new_zeros"):
                depth = pts_conf.new_zeros(pts_conf.shape)
            if depth is None:
                depth = pts_conf
            desc_kpts.update_3D_pts(pts3d, depth, pts_conf, match_indices)
        except Exception:
            return

    def register_pose_only_reference(
        self,
        *,
        frame_id: int,
        info: dict[str, Any],
        desc_kpts: Any,
        Rt: Any,
        density_debug: dict[str, Any] | None = None,
        pose_debug: dict[str, Any] | None = None,
        pose_support: dict[str, Any] | None = None,
    ) -> bool:
        debug = dict(density_debug or {})
        reason = ""
        if not self._pose_only_pool_enabled():
            reason = "pool_disabled"
        elif bool(info.get("is_test", False)):
            reason = "test_frame"
        elif not bool(debug.get("active_memory_context", False)):
            reason = "not_active_memory_context"
        elif str(debug.get("active_memory_frame_role", "")) != "tracking_only":
            reason = "not_tracking_only_role"
        elif desc_kpts is None or Rt is None:
            reason = "missing_pose_reference_payload"

        if reason:
            self.pose_reference_pool_events.append(
                {
                    "event_type": "pose_only_register",
                    "frame_id": int(frame_id),
                    "registered": False,
                    "block_reason": reason,
                    "pool_size_after": len(self.pose_only_reference_pool),
                }
            )
            return False

        risk_gate_metrics: dict[str, float] = {}
        if self.pose_only_reference_risk_gate_enabled:
            risk_gate_ok, risk_gate_reason, risk_gate_metrics = (
                self._pose_only_reference_risk_gate_decision(debug, frame_id=int(frame_id))
            )
            if not risk_gate_ok:
                self.pose_reference_pool_events.append(
                    {
                        "event_type": "pose_only_register",
                        "frame_id": int(frame_id),
                        "registered": False,
                        "block_reason": risk_gate_reason,
                        "pool_size_after": len(self.pose_only_reference_pool),
                        **risk_gate_metrics,
                    }
                )
                return False

        self._purge_pose_only_references(int(frame_id))
        register_gap = (
            int(frame_id) - int(self._last_pose_only_reference_register_frame)
            if int(self._last_pose_only_reference_register_frame) >= 0
            else 10**9
        )
        if register_gap < int(self.pose_only_reference_register_min_interval_frames):
            self.pose_reference_pool_events.append(
                {
                    "event_type": "pose_only_register",
                    "frame_id": int(frame_id),
                    "registered": False,
                    "block_reason": "pose_only_register_cooldown",
                    "pose_only_register_gap": int(register_gap),
                    "pose_only_register_min_interval_frames": int(
                        self.pose_only_reference_register_min_interval_frames
                    ),
                    "pool_size_after": len(self.pose_only_reference_pool),
                }
            )
            return False
        desc = self._clone_pose_only_desc(desc_kpts)
        self._apply_pose_support_to_desc(desc, pose_support)
        support_count = self._pose_only_support_count(desc)
        if support_count < int(self.pose_only_reference_min_3d_points):
            self.pose_reference_pool_events.append(
                {
                    "event_type": "pose_only_register",
                    "frame_id": int(frame_id),
                    "registered": False,
                    "block_reason": "insufficient_3d_support",
                    "pose_only_3d_support_count": support_count,
                    "pool_size_after": len(self.pose_only_reference_pool),
                }
            )
            return False

        try:
            r_w2c = Rt[:3, :2].detach().clone()
            t_w2c = Rt[:3, 3].detach().clone()
        except Exception:
            r_w2c = getattr(Rt, "rW2C", None)
            t_w2c = getattr(Rt, "tW2C", None)
        ref_info = dict(info)
        ref_info["_paper_aligned_source_frame_id"] = int(frame_id)
        ref_info["_paper_aligned_commit_origin"] = "pose_only_reference"
        ref_info["_paper_aligned_pose_only_reference"] = True
        ref_info["_paper_aligned_support_eligible_recovery_keyframe"] = False
        for key, value in risk_gate_metrics.items():
            ref_info[f"_paper_aligned_{key}"] = value
        ref_info["_paper_aligned_pose_only_pnp_inliers"] = int((pose_debug or {}).get("num_pnp_inliers", 0) or 0)
        ref_info["_paper_aligned_pose_only_miniba_inliers"] = int((pose_debug or {}).get("num_miniba_inliers", 0) or 0)
        reference = SimpleNamespace(
            index=1_000_000 + int(frame_id),
            info=ref_info,
            desc_kpts=desc,
            rW2C=r_w2c,
            tW2C=t_w2c,
            is_test=bool(info.get("is_test", False)),
        )
        self.pose_only_reference_pool.append(reference)
        if len(self.pose_only_reference_pool) > int(self.pose_only_reference_pool_max_size):
            self.pose_only_reference_pool = self.pose_only_reference_pool[
                -int(self.pose_only_reference_pool_max_size) :
            ]
        self._last_pose_only_reference_register_frame = int(frame_id)
        self.pose_only_reference_registered_count += 1
        self.pose_reference_pool_events.append(
            {
                "event_type": "pose_only_register",
                "frame_id": int(frame_id),
                "reference_keyframe_id": int(reference.index),
                "reference_source_frame_id": int(frame_id),
                "reference_commit_origin": "pose_only_reference",
                "registered": True,
                "pose_only_3d_support_count": support_count,
                "pose_only_pool_size_after": len(self.pose_only_reference_pool),
                **risk_gate_metrics,
                "pose_only_pnp_inliers": int((pose_debug or {}).get("num_pnp_inliers", 0) or 0),
                "pose_only_miniba_inliers": int((pose_debug or {}).get("num_miniba_inliers", 0) or 0),
            }
        )
        return True

    def select_pose_only_references(
        self,
        *,
        frame_id: int,
        curr_desc_kpts: Any,
        matcher: Any,
        max_refs: int | None = None,
    ) -> list[Any]:
        if not self._pose_only_pool_enabled() or curr_desc_kpts is None or matcher is None:
            return []
        self._purge_pose_only_references(int(frame_id))
        limit = int(max_refs or self.pose_only_reference_max_per_query)
        selection_strategy = str(self.pose_only_reference_selection_strategy)
        scored: list[tuple[float, float, float, int, int, Any]] = []
        candidate_count = 0
        cooldown_skip_count = 0
        for ref in self.pose_only_reference_pool:
            source_frame_id = int(ref.info.get("_paper_aligned_source_frame_id", -1))
            if source_frame_id >= int(frame_id):
                continue
            if int(frame_id) - source_frame_id < int(self.pose_only_reference_min_age_frames):
                continue
            age = int(frame_id) - source_frame_id
            last_selected = int(
                ref.info.get("_paper_aligned_pose_only_last_selected_frame", -1)
            )
            if (
                last_selected >= 0
                and int(frame_id) - last_selected
                < int(self.pose_only_reference_selection_cooldown_frames)
            ):
                cooldown_skip_count += 1
                continue
            support_count = self._pose_only_support_count(ref.desc_kpts)
            score = self._tensor_float(matcher.evaluate_match(ref.desc_kpts, curr_desc_kpts))
            candidate_count += 1
            selected = bool(
                support_count >= int(self.pose_only_reference_min_3d_points)
                and score >= float(self.pose_only_reference_min_match_score)
            )
            if selected:
                age_bonus = min(float(age), float(self.pose_only_reference_ttl_frames)) / max(
                    float(self.pose_only_reference_ttl_frames), 1.0
                )
                geometry_score = 0.0
                if selection_strategy == "risk_aware":
                    geometry_score = self._pose_only_reference_geometry_score(ref, support_count, age)
                    diversity_score = score + geometry_score
                else:
                    diversity_score = score + float(self.pose_only_reference_age_bonus) * age_bonus
                scored.append((diversity_score, score, geometry_score, age, source_frame_id, ref))
        scored.sort(key=lambda item: (-item[0], -item[2], -item[3]))
        selected_refs = [item[5] for item in scored[:limit]]
        for ref in selected_refs:
            ref.info["_paper_aligned_pose_only_last_selected_frame"] = int(frame_id)
        if candidate_count or selected_refs:
            self.pose_reference_pool_events.append(
                {
                    "event_type": "pose_only_select_summary",
                    "frame_id": int(frame_id),
                    "pose_only_pool_size": len(self.pose_only_reference_pool),
                    "pose_only_candidate_count": int(candidate_count),
                    "pose_only_selected_count": len(selected_refs),
                    "pose_only_cooldown_skip_count": int(cooldown_skip_count),
                    "pose_only_selected_source_frame_ids": [
                        int(ref.info.get("_paper_aligned_source_frame_id", -1))
                        for ref in selected_refs
                    ],
                    "pose_only_selected_reference_ids": [int(ref.index) for ref in selected_refs],
                    "pose_only_selection_strategy": selection_strategy,
                }
            )
            for diversity_score, score, geometry_score, age, _source_frame_id, ref in scored[:limit]:
                self.pose_reference_pool_events.append(
                    {
                        "event_type": "pose_only_select_reference",
                        "frame_id": int(frame_id),
                        "reference_keyframe_id": int(ref.index),
                        "reference_source_frame_id": int(
                            ref.info.get("_paper_aligned_source_frame_id", -1)
                        ),
                        "reference_commit_origin": "pose_only_reference",
                        "pose_only_match_score": float(score),
                        "pose_only_diversity_score": float(diversity_score),
                        "pose_only_geometry_score": float(geometry_score),
                        "pose_only_selection_strategy": selection_strategy,
                        "pose_only_reference_age": int(age),
                        "pose_only_3d_support_count": self._pose_only_support_count(
                            ref.desc_kpts
                        ),
                        "candidate_selected": True,
                    }
                )
        self.pose_only_reference_selected_count += len(selected_refs)
        return selected_refs

    def pose_only_reference_pool_summary(self) -> dict[str, Any]:
        return {
            "enabled": self._pose_only_pool_enabled(),
            "pool_size": len(self.pose_only_reference_pool),
            "registered_count": int(self.pose_only_reference_registered_count),
            "selected_count": int(self.pose_only_reference_selected_count),
            "max_size": int(self.pose_only_reference_pool_max_size),
            "ttl_frames": int(self.pose_only_reference_ttl_frames),
            "min_age_frames": int(self.pose_only_reference_min_age_frames),
            "max_per_query": int(self.pose_only_reference_max_per_query),
            "min_3d_points": int(self.pose_only_reference_min_3d_points),
            "min_match_score": float(self.pose_only_reference_min_match_score),
            "register_min_interval_frames": int(
                self.pose_only_reference_register_min_interval_frames
            ),
            "selection_cooldown_frames": int(
                self.pose_only_reference_selection_cooldown_frames
            ),
            "age_bonus": float(self.pose_only_reference_age_bonus),
            "selection_strategy": str(self.pose_only_reference_selection_strategy),
            "risk_gate_enabled": bool(self.pose_only_reference_risk_gate_enabled),
            "min_support_concentration": float(self.pose_only_reference_min_support_concentration),
            "low_new_view_max": float(self.pose_only_reference_low_new_view_max),
            "high_new_view_min": float(self.pose_only_reference_high_new_view_min),
            "high_new_view_support_min": float(self.pose_only_reference_high_new_view_support_min),
            "high_new_view_anchor_health_max": float(
                self.pose_only_reference_high_new_view_anchor_health_max
            ),
            "high_new_view_entropy_max": float(
                self.pose_only_reference_high_new_view_entropy_max
            ),
            "early_turn_bridge_frame_max": int(
                self.pose_only_reference_early_turn_bridge_frame_max
            ),
            "late_extreme_new_view_frame_min": int(
                self.pose_only_reference_late_extreme_new_view_frame_min
            ),
            "late_extreme_new_view_min": float(
                self.pose_only_reference_late_extreme_new_view_min
            ),
            "late_extreme_new_view_support_min": float(
                self.pose_only_reference_late_extreme_new_view_support_min
            ),
            "growth_stall_max": int(self.pose_only_reference_growth_stall_max),
            "requires_negative_growth": bool(
                self.pose_only_reference_require_negative_growth
            ),
            "allow_high_new_view_rescue": bool(
                self.pose_only_reference_allow_high_new_view_rescue
            ),
            "repetitive_entropy_min": float(
                self.pose_only_reference_repetitive_entropy_min
            ),
            "repetitive_entropy_support_max": float(
                self.pose_only_reference_repetitive_entropy_support_max
            ),
        }

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
            "viewpoint_coverage_events": self.viewpoint_coverage_events,
            "pose_only_reference_pool_summary": self.pose_only_reference_pool_summary(),
            "trace_unavailable_reasons": self.trace_unavailable_reasons,
        }
        if self.semantic_policy is not None:
            payload["semantic_summary"] = self.semantic_policy.summary()
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
