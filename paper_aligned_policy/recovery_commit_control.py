from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _b(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    if isinstance(x, (int, float)):
        return x != 0
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


@dataclass
class RecoveryCommitDecision:
    action: str  # commit | hold | reject
    reason: str
    debug: dict[str, Any]


class RecoveryCommitController:
    """
    Post-success commit control for recovered sources.

    This layer does NOT change R/V/Q or tau-based tri-state decision.
    It only gates whether a recovery-success source is materialized into main chain.
    """

    def __init__(self, args: Any) -> None:
        self.mode = str(getattr(args, "paper_aligned_recovery_commit_control", "off") or "off")
        self.window_size = int(getattr(args, "paper_aligned_recovery_window_size", 30) or 30)
        self.max_per_window = int(
            getattr(args, "paper_aligned_recovery_commit_max_per_window", 5) or 5
        )
        self.max_candidate_age = int(
            getattr(args, "paper_aligned_recovery_candidate_max_age", 180) or 180
        )
        self.max_hold_retries = int(getattr(args, "paper_aligned_recovery_hold_max_retries", 4) or 4)
        self.retry_interval = int(getattr(args, "paper_aligned_recovery_hold_retry_interval", 6) or 6)
        # Baseline-aligned density reference for forest1-like settings (343 / 1194 ~= 28.7/100)
        self.baseline_density_upper = float(
            getattr(args, "paper_aligned_recovery_density_upper_per_100", 46.0) or 46.0
        )
        self.min_geom_matches = int(
            getattr(args, "paper_aligned_recovery_min_geom_matches", 80) or 80
        )
        self.min_v_for_commit = float(
            getattr(args, "paper_aligned_recovery_min_v_for_commit", 0.35) or 0.35
        )
        self.min_q_for_commit = float(
            getattr(args, "paper_aligned_recovery_min_q_for_commit", 0.20) or 0.20
        )
        self.main_chain_gap_small = float(
            getattr(args, "paper_aligned_recovery_main_chain_gap_small", 2.0) or 2.0
        )
        self.gap_override_threshold = float(
            getattr(args, "paper_aligned_recovery_gap_override_threshold", 5.0) or 5.0
        )
        self.gap_override_max_per_interval = int(
            getattr(args, "paper_aligned_recovery_gap_override_max_per_interval", 2) or 2
        )
        self.gap_override_density_upper = float(
            getattr(args, "paper_aligned_recovery_gap_override_density_upper_per_100", 55.0) or 55.0
        )
        self.gap_override_min_v = float(
            getattr(args, "paper_aligned_recovery_gap_override_min_v", 0.25) or 0.25
        )
        self.gap_override_min_q = float(
            getattr(args, "paper_aligned_recovery_gap_override_min_q", 0.10) or 0.10
        )
        self.v2_source_gap_trigger = int(
            getattr(args, "paper_aligned_recovery_v2_source_gap_trigger", 18) or 18
        )
        self.v2_predicted_gap_trigger = int(
            getattr(args, "paper_aligned_recovery_v2_predicted_gap_trigger", 20) or 20
        )
        self.v2_episode_override_budget = int(
            getattr(args, "paper_aligned_recovery_v2_episode_override_budget", 4) or 4
        )
        self.v3_density_upper = float(
            getattr(args, "paper_aligned_recovery_v3_density_upper_per_100", 42.0) or 42.0
        )
        self.v3_gap_critical_trigger = int(
            getattr(args, "paper_aligned_recovery_v3_gap_critical_trigger", 18) or 18
        )
        self.v3_gap_hard_limit = int(
            getattr(args, "paper_aligned_recovery_v3_gap_hard_limit", 20) or 20
        )
        self.v3_window_size = int(getattr(args, "paper_aligned_recovery_v3_window_size", 50) or 50)
        self.v3_max_normal_per_window = int(
            getattr(args, "paper_aligned_recovery_v3_max_normal_per_window", 1) or 1
        )
        self.v3_max_gap_critical_per_window = int(
            getattr(args, "paper_aligned_recovery_v3_max_gap_critical_per_window", 2) or 2
        )
        self.v3_min_source_gap = int(getattr(args, "paper_aligned_recovery_v3_min_source_gap", 5) or 5)
        self.v3_support_topk = int(getattr(args, "paper_aligned_recovery_v3_support_topk", 1) or 1)
        self.v3_anchor_guard_enabled = _b(
            getattr(args, "paper_aligned_recovery_v3_anchor_guard_enabled", True), True
        )
        self.v3_anchor_soft_limit = int(getattr(args, "paper_aligned_recovery_v3_anchor_soft_limit", 5) or 5)
        self.v3_min_v = float(getattr(args, "paper_aligned_recovery_v3_min_v", 0.25) or 0.25)
        self.v3_min_q = float(getattr(args, "paper_aligned_recovery_v3_min_q", 0.10) or 0.10)
        self.v3_max_r = float(getattr(args, "paper_aligned_recovery_v3_max_r", 0.75) or 0.75)
        self.v4_density_lower = float(
            getattr(args, "paper_aligned_recovery_v4_density_lower_per_100", 28.0) or 28.0
        )
        self.v4_density_target = float(
            getattr(args, "paper_aligned_recovery_v4_density_target_per_100", 35.0) or 35.0
        )
        self.v4_density_upper = float(
            getattr(args, "paper_aligned_recovery_v4_density_upper_per_100", 42.0) or 42.0
        )
        self.v4_density_hard_upper = float(
            getattr(args, "paper_aligned_recovery_v4_density_hard_upper_per_100", 50.0) or 50.0
        )
        self.v4_gap_trigger = int(getattr(args, "paper_aligned_recovery_v4_gap_trigger", 18) or 18)
        self.v4_gap_hard_limit = int(
            getattr(args, "paper_aligned_recovery_v4_gap_hard_limit", 20) or 20
        )
        self.v4_gap_rescue_budget_per_episode = int(
            getattr(args, "paper_aligned_recovery_v4_gap_rescue_budget_per_episode", 4) or 4
        )
        self.v4_window_size = int(getattr(args, "paper_aligned_recovery_v4_window_size", 50) or 50)
        self.v4_topk_below = int(
            getattr(args, "paper_aligned_recovery_v4_normal_topk_below_lower", 2) or 2
        )
        self.v4_topk_in = int(
            getattr(args, "paper_aligned_recovery_v4_normal_topk_in_band", 1) or 1
        )
        self.v4_topk_above = int(
            getattr(args, "paper_aligned_recovery_v4_normal_topk_above_upper", 0) or 0
        )
        self.v4_min_source_gap = int(getattr(args, "paper_aligned_recovery_v4_min_source_gap", 4) or 4)
        self.v4_min_v = float(getattr(args, "paper_aligned_recovery_v4_min_v", 0.25) or 0.25)
        self.v4_min_q = float(getattr(args, "paper_aligned_recovery_v4_min_q", 0.10) or 0.10)
        self.v4_max_r = float(getattr(args, "paper_aligned_recovery_v4_max_r", 0.75) or 0.75)
        self.v4_anchor_target_min = int(
            getattr(args, "paper_aligned_recovery_v4_anchor_target_min", 4) or 4
        )
        self.v4_anchor_target_max = int(
            getattr(args, "paper_aligned_recovery_v4_anchor_target_max", 5) or 5
        )
        self.v4_anchor_soft_upper = int(
            getattr(args, "paper_aligned_recovery_v4_anchor_soft_upper", 6) or 6
        )
        self.v4_retry_extension_for_gap = _b(
            getattr(args, "paper_aligned_recovery_v4_retry_extension_for_gap", True), True
        )
        self.v4_retry_extension_for_coverage = _b(
            getattr(args, "paper_aligned_recovery_v4_retry_extension_for_coverage", True), True
        )
        self.v5_density_lower = float(
            getattr(args, "paper_aligned_recovery_v5_density_lower_per_100", 28.0) or 28.0
        )
        self.v5_density_target = float(
            getattr(args, "paper_aligned_recovery_v5_density_target_per_100", 35.0) or 35.0
        )
        self.v5_density_upper = float(
            getattr(args, "paper_aligned_recovery_v5_density_upper_per_100", 42.0) or 42.0
        )
        self.v5_density_hard_upper = float(
            getattr(args, "paper_aligned_recovery_v5_density_hard_upper_per_100", 50.0) or 50.0
        )
        self.v5_gap_trigger = int(getattr(args, "paper_aligned_recovery_v5_gap_trigger", 18) or 18)
        self.v5_gap_hard_limit = int(getattr(args, "paper_aligned_recovery_v5_gap_hard_limit", 20) or 20)
        self.v5_gap_rescue_base_budget = int(
            getattr(args, "paper_aligned_recovery_v5_gap_rescue_base_budget", 4) or 4
        )
        self.v5_window_size = int(getattr(args, "paper_aligned_recovery_v5_window_size", 50) or 50)
        self.v5_topk_below = int(
            getattr(args, "paper_aligned_recovery_v5_normal_topk_below_lower", 2) or 2
        )
        self.v5_topk_in = int(
            getattr(args, "paper_aligned_recovery_v5_normal_topk_in_band", 1) or 1
        )
        self.v5_topk_above = int(
            getattr(args, "paper_aligned_recovery_v5_normal_topk_above_upper", 0) or 0
        )
        self.v5_coverage_topk = int(getattr(args, "paper_aligned_recovery_v5_coverage_topk", 3) or 3)
        self.v5_min_source_gap = int(getattr(args, "paper_aligned_recovery_v5_min_source_gap", 4) or 4)
        self.v5_min_v = float(getattr(args, "paper_aligned_recovery_v5_min_v", 0.25) or 0.25)
        self.v5_min_q = float(getattr(args, "paper_aligned_recovery_v5_min_q", 0.10) or 0.10)
        self.v5_max_r = float(getattr(args, "paper_aligned_recovery_v5_max_r", 0.75) or 0.75)
        self.v5_anchor_target_min = int(
            getattr(args, "paper_aligned_recovery_v5_anchor_target_min", 4) or 4
        )
        self.v5_anchor_target_max = int(
            getattr(args, "paper_aligned_recovery_v5_anchor_target_max", 5) or 5
        )
        self.v5_anchor_soft_upper = int(
            getattr(args, "paper_aligned_recovery_v5_anchor_soft_upper", 6) or 6
        )
        self.v5_growth_window_short = int(
            getattr(args, "paper_aligned_recovery_v5_growth_window_short", 100) or 100
        )
        self.v5_growth_window_long = int(
            getattr(args, "paper_aligned_recovery_v5_growth_window_long", 200) or 200
        )
        self.v5_growth_plateau_min_short = int(
            getattr(args, "paper_aligned_recovery_v5_growth_plateau_min_short", 10) or 10
        )
        self.v5_growth_plateau_min_long = int(
            getattr(args, "paper_aligned_recovery_v5_growth_plateau_min_long", 20) or 20
        )
        self.v5_retry_extension_for_gap = _b(
            getattr(args, "paper_aligned_recovery_v5_retry_extension_for_gap", True), True
        )
        self.v5_retry_extension_for_coverage = _b(
            getattr(args, "paper_aligned_recovery_v5_retry_extension_for_coverage", True), True
        )
        self._gap_interval_index = 0
        self._in_gap_interval = False
        self._gap_interval_override_count = 0
        self._v2_episode_override_counts: dict[int, int] = {}
        self._v3_windows: dict[int, dict[str, Any]] = {}
        self._v4_windows: dict[int, dict[str, Any]] = {}
        self._v4_episode_rescue_counts: dict[int, int] = {}
        self._v5_windows: dict[int, dict[str, Any]] = {}
        self._v5_episode_rescue_counts: dict[int, int] = {}

    def _adaptive_budget(self, main_chain_gap_p90: float, keyframe_density_per_100: float) -> int:
        if main_chain_gap_p90 >= 6.0:
            return self.max_per_window + 2
        if main_chain_gap_p90 >= 4.0:
            return self.max_per_window + 1
        if keyframe_density_per_100 > self.baseline_density_upper * 1.1:
            return max(2, self.max_per_window - 2)
        if keyframe_density_per_100 > self.baseline_density_upper:
            return max(3, self.max_per_window - 1)
        return self.max_per_window

    def _v3_support_score(
        self,
        *,
        num_matches: int,
        num_inliers: int,
        v_t: float,
        q_t: float,
        r_t: float,
    ) -> float:
        risk_gain = max(0.0, self.v3_max_r - r_t)
        return (
            0.50 * float(max(num_matches, 0))
            + 0.80 * float(max(num_inliers, 0))
            + 100.0 * float(max(v_t, 0.0))
            + 100.0 * float(max(q_t, 0.0))
            + 50.0 * risk_gain
        )

    def _v4_support_score(
        self,
        *,
        num_matches: int,
        num_inliers: int,
        v_t: float,
        q_t: float,
        r_t: float,
    ) -> float:
        risk_gain = max(0.0, self.v4_max_r - r_t)
        return (
            0.45 * float(max(num_matches, 0))
            + 0.90 * float(max(num_inliers, 0))
            + 110.0 * float(max(v_t, 0.0))
            + 110.0 * float(max(q_t, 0.0))
            + 60.0 * risk_gain
        )

    def _decide_balanced_v4(
        self,
        candidate: dict[str, Any],
        context: dict[str, Any],
        base_debug: dict[str, Any],
    ) -> RecoveryCommitDecision:
        scores = candidate.get("scores", {}) or {}
        source_payload = candidate.get("source_payload", {}) or {}
        inlier_evidence = source_payload.get("inlier_evidence", {}) or {}
        current_tick = int(context.get("current_tick_frame_id", -1))
        source_input = int(candidate.get("source_input_index", candidate.get("source_frame_id", -1)))
        age = max(0, current_tick - source_input)
        hold_retries = int(candidate.get("_hold_retries", 0))
        density_before = _f(context.get("keyframe_density_per_100", 0.0))
        density_after = density_before + (100.0 / float(max(current_tick, 1)))
        source_gap_to_last_committed = int(context.get("source_gap_to_last_committed", 0))
        predicted_gap_if_hold = int(context.get("predicted_gap_if_hold", source_gap_to_last_committed))
        long_gap_episode_id = int(context.get("long_gap_episode_id", 0))
        keyframe_growth_recent = int(context.get("keyframe_growth_recent", 0))
        starvation_risk = bool(context.get("starvation_risk", False))

        v_t = _f(scores.get("V_t"))
        q_t = _f(scores.get("Q_t"))
        r_t = _f(scores.get("R_t"))
        num_matches = int(_f(inlier_evidence.get("num_matches", 0), 0.0))
        num_inliers = int(_f(context.get("source_num_inliers", 0), 0.0))
        anchor_count_available = bool(context.get("anchor_count_available", False))
        anchor_count_before_raw = context.get("anchor_count_before", None)
        anchor_count_before = int(anchor_count_before_raw) if anchor_count_before_raw is not None else -1

        is_duplicate = bool(context.get("source_already_committed", False))
        is_surrogate = bool(context.get("is_surrogate", False))
        is_contamination_risk = bool(context.get("is_contamination_risk", False))
        is_semantic_valid = bool(
            (not is_duplicate)
            and (not is_surrogate)
            and (not is_contamination_risk)
            and (r_t < self.v4_max_r)
            and (v_t >= self.v4_min_v)
            and (q_t >= self.v4_min_q)
            and (density_after <= self.v4_density_hard_upper)
            and (source_input >= 0)
        )
        geom_ok = num_matches >= self.min_geom_matches
        age_ok = age <= self.max_candidate_age

        if density_after > self.v4_density_hard_upper:
            density_state = "above_hard"
        elif density_after > self.v4_density_upper:
            density_state = "above_upper"
        elif density_after < self.v4_density_lower:
            density_state = "below_lower"
        else:
            density_state = "in_band"

        is_gap_critical = bool(
            source_gap_to_last_committed >= self.v4_gap_trigger
            or predicted_gap_if_hold > self.v4_gap_hard_limit
        )
        is_coverage_floor = bool(density_before < self.v4_density_lower or starvation_risk)
        is_too_close = source_gap_to_last_committed < self.v4_min_source_gap

        if density_state == "below_lower":
            normal_topk = self.v4_topk_below
        elif density_state == "in_band":
            normal_topk = self.v4_topk_in
        else:
            normal_topk = self.v4_topk_above

        window_id = max(0, int(current_tick // max(self.v4_window_size, 1)))
        ws = self._v4_windows.setdefault(
            window_id,
            {"normal_commits": 0, "coverage_commits": 0, "support_scores": {}},
        )
        support_score = self._v4_support_score(
            num_matches=num_matches,
            num_inliers=num_inliers,
            v_t=v_t,
            q_t=q_t,
            r_t=r_t,
        )
        ws["support_scores"][int(source_input)] = float(support_score)
        ranked = sorted(
            ((int(k), float(v)) for k, v in ws["support_scores"].items()),
            key=lambda x: (-x[1], x[0]),
        )
        rank_map = {sid: idx + 1 for idx, (sid, _score) in enumerate(ranked)}
        window_candidate_rank = int(rank_map.get(int(source_input), len(ranked) + 1))
        is_support_topk = window_candidate_rank <= max(normal_topk, 0)

        if not anchor_count_available:
            anchor_target_state = "unavailable"
        elif anchor_count_before < self.v4_anchor_target_min:
            anchor_target_state = "below_band"
        elif anchor_count_before <= self.v4_anchor_target_max:
            anchor_target_state = "in_band"
        else:
            anchor_target_state = "above_band"
        anchor_soft_guard_triggered = bool(
            anchor_count_available and anchor_count_before > self.v4_anchor_target_max
        )

        episode_count = int(self._v4_episode_rescue_counts.get(long_gap_episode_id, 0))
        gap_rescue_budget_used = episode_count
        can_gap_rescue = bool(
            is_gap_critical
            and long_gap_episode_id > 0
            and episode_count < self.v4_gap_rescue_budget_per_episode
        )
        retry_limit_extended = False
        retry_extension_reason = ""

        debug = dict(base_debug)
        debug.update(
            {
                "commit_channel": "",
                "R_t": r_t,
                "V_t": v_t,
                "Q_t": q_t,
                "support_score": float(support_score),
                "num_matches": num_matches,
                "num_inliers": num_inliers,
                "support_count": num_inliers,
                "source_gap_to_last_committed": source_gap_to_last_committed,
                "predicted_gap_if_hold": predicted_gap_if_hold,
                "density_before": density_before,
                "density_after": density_after,
                "density_state": density_state,
                "window_id": window_id,
                "window_candidate_rank": window_candidate_rank,
                "window_support_score": float(support_score),
                "window_budget_used": int(ws["normal_commits"] + ws["coverage_commits"]),
                "window_normal_commit_count": int(ws["normal_commits"]),
                "window_coverage_commit_count": int(ws["coverage_commits"]),
                "long_gap_episode_id": long_gap_episode_id,
                "gap_rescue_budget_used": gap_rescue_budget_used,
                "keyframe_growth_recent": keyframe_growth_recent,
                "starvation_risk": starvation_risk,
                "anchor_count_before": anchor_count_before,
                "anchor_target_state": anchor_target_state,
                "anchor_soft_guard_triggered": anchor_soft_guard_triggered,
                "anchor_override_reason": "",
                "retry_limit_extended": False,
                "retry_extension_reason": "",
                "blocked_reason": "",
                "is_gap_critical": is_gap_critical,
                "is_coverage_floor": is_coverage_floor,
                "is_support_topk": is_support_topk,
                "is_duplicate": is_duplicate,
                "is_surrogate": is_surrogate,
                "is_contamination_risk": is_contamination_risk,
            }
        )

        if not is_semantic_valid or (not geom_ok):
            debug["blocked_reason"] = "invalid_semantics_or_support"
            if is_gap_critical:
                return RecoveryCommitDecision("reject", "v4_gap_rescue_reject_invalid_semantics", debug)
            return RecoveryCommitDecision("reject", "v4_reject_invalid_semantics", debug)
        if not age_ok:
            debug["blocked_reason"] = "reject_age"
            return RecoveryCommitDecision("reject", "v3_reject_age", debug)
        if hold_retries >= self.max_hold_retries:
            if is_gap_critical and self.v4_retry_extension_for_gap:
                retry_limit_extended = True
                retry_extension_reason = "gap_critical"
            elif is_coverage_floor and self.v4_retry_extension_for_coverage:
                retry_limit_extended = True
                retry_extension_reason = "coverage_floor"
            else:
                debug["blocked_reason"] = "reject_retry_limit"
                return RecoveryCommitDecision("reject", "v3_reject_retry_limit", debug)
        debug["retry_limit_extended"] = retry_limit_extended
        debug["retry_extension_reason"] = retry_extension_reason

        # A) gap-rescue commit channel
        if is_gap_critical:
            if density_state == "above_hard":
                debug["blocked_reason"] = "gap_rescue_hard_density"
                return RecoveryCommitDecision("hold", "v4_gap_rescue_hold_hard_density", debug)
            if density_state == "above_upper" and predicted_gap_if_hold <= (self.v4_gap_hard_limit + 2):
                debug["blocked_reason"] = "gap_rescue_density_above_upper_noncritical_tail"
                return RecoveryCommitDecision("hold", "v4_hold_density_above_upper", debug)
            if not can_gap_rescue:
                debug["blocked_reason"] = "gap_rescue_budget"
                return RecoveryCommitDecision("hold", "v3_hold_window_budget", debug)
            self._v4_episode_rescue_counts[long_gap_episode_id] = episode_count + 1
            debug["gap_rescue_budget_used"] = self._v4_episode_rescue_counts[long_gap_episode_id]
            debug["commit_channel"] = "gap_rescue"
            if anchor_soft_guard_triggered:
                debug["anchor_override_reason"] = "gap_rescue"
            return RecoveryCommitDecision("commit", "v4_gap_rescue_commit", debug)

        # B) coverage-floor commit channel
        if is_coverage_floor:
            debug["commit_channel"] = "coverage_floor"
            if density_state == "above_hard":
                debug["blocked_reason"] = "coverage_floor_hard_density"
                return RecoveryCommitDecision("hold", "v4_gap_rescue_hold_hard_density", debug)
            if not is_support_topk:
                debug["blocked_reason"] = "not_topk"
                return RecoveryCommitDecision("hold", "v4_coverage_floor_hold_not_topk", debug)
            if is_too_close:
                debug["blocked_reason"] = "too_close"
                return RecoveryCommitDecision("hold", "v4_coverage_floor_hold_too_close", debug)
            if density_state == "above_upper":
                debug["blocked_reason"] = "density_above_upper"
                return RecoveryCommitDecision("hold", "v4_hold_density_above_upper", debug)
            ws["coverage_commits"] = int(ws["coverage_commits"]) + 1
            debug["window_budget_used"] = int(ws["normal_commits"] + ws["coverage_commits"])
            return RecoveryCommitDecision("commit", "v4_coverage_floor_commit", debug)

        # C) target-band normal sparse commit channel
        debug["commit_channel"] = "target_band_normal"
        if density_state == "above_upper":
            debug["blocked_reason"] = "density_above_upper"
            return RecoveryCommitDecision("hold", "v4_hold_density_above_upper", debug)
        if not is_support_topk:
            debug["blocked_reason"] = "not_topk"
            return RecoveryCommitDecision("hold", "v4_hold_not_topk", debug)
        if is_too_close:
            debug["blocked_reason"] = "too_close"
            return RecoveryCommitDecision("hold", "v4_hold_too_close", debug)
        if anchor_soft_guard_triggered and density_state != "below_lower":
            debug["blocked_reason"] = "anchor_soft_guard"
            return RecoveryCommitDecision("hold", "v4_hold_window_budget", debug)
        normal_budget = 2 if density_state == "below_lower" else 1
        if int(ws["normal_commits"]) >= normal_budget:
            debug["blocked_reason"] = "window_budget"
            return RecoveryCommitDecision("hold", "v4_hold_window_budget", debug)
        ws["normal_commits"] = int(ws["normal_commits"]) + 1
        debug["window_budget_used"] = int(ws["normal_commits"] + ws["coverage_commits"])
        return RecoveryCommitDecision("commit", "v4_support_ranked_sparse_commit", debug)

    def _v5_support_score(
        self,
        *,
        num_matches: int,
        num_inliers: int,
        v_t: float,
        q_t: float,
        r_t: float,
    ) -> float:
        risk_gain = max(0.0, self.v5_max_r - r_t)
        return (
            0.45 * float(max(num_matches, 0))
            + 0.90 * float(max(num_inliers, 0))
            + 120.0 * float(max(v_t, 0.0))
            + 120.0 * float(max(q_t, 0.0))
            + 60.0 * risk_gain
        )

    def _decide_rescue_v5(
        self,
        candidate: dict[str, Any],
        context: dict[str, Any],
        base_debug: dict[str, Any],
    ) -> RecoveryCommitDecision:
        scores = candidate.get("scores", {}) or {}
        source_payload = candidate.get("source_payload", {}) or {}
        inlier_evidence = source_payload.get("inlier_evidence", {}) or {}
        current_tick = int(context.get("current_tick_frame_id", -1))
        source_input = int(candidate.get("source_input_index", candidate.get("source_frame_id", -1)))
        age = max(0, current_tick - source_input)
        hold_retries = int(candidate.get("_hold_retries", 0))
        current_keyframes = int(context.get("current_keyframe_count", 0))
        density_before = _f(context.get("keyframe_density_per_100", 0.0))
        density_after = density_before + (100.0 / float(max(current_tick, 1)))
        source_gap_to_last_committed = int(context.get("source_gap_to_last_committed", 0))
        predicted_gap_if_hold = int(context.get("predicted_gap_if_hold", source_gap_to_last_committed))
        long_gap_episode_id = int(context.get("long_gap_episode_id", 0))
        open_gap_unclosed = bool(context.get("open_gap_unclosed", False))

        recent_growth_short = int(context.get("recent_keyframe_growth_short", 0))
        recent_growth_long = int(context.get("recent_keyframe_growth_long", 0))
        recent_keyframe_growth = recent_growth_short
        growth_plateau = bool(
            (current_tick >= self.v5_growth_window_short and recent_growth_short <= self.v5_growth_plateau_min_short)
            or (current_tick >= self.v5_growth_window_long and recent_growth_long <= self.v5_growth_plateau_min_long)
        )

        v_t = _f(scores.get("V_t"))
        q_t = _f(scores.get("Q_t"))
        r_t = _f(scores.get("R_t"))
        num_matches = int(_f(inlier_evidence.get("num_matches", 0), 0.0))
        num_inliers = int(_f(context.get("source_num_inliers", 0), 0.0))
        anchor_count_available = bool(context.get("anchor_count_available", False))
        anchor_count_before_raw = context.get("anchor_count_before", None)
        anchor_count_before = int(anchor_count_before_raw) if anchor_count_before_raw is not None else -1

        is_duplicate = bool(context.get("source_already_committed", False))
        is_surrogate = bool(context.get("is_surrogate", False))
        is_contamination_risk = bool(context.get("is_contamination_risk", False))
        expected_min_keyframes = float(max(current_tick, 1)) * self.v5_density_lower / 100.0
        keyframe_deficit = max(0.0, expected_min_keyframes - float(max(current_keyframes, 0)))
        coverage_rescue_triggered = bool(keyframe_deficit > 0.0 or growth_plateau)
        gap_rescue_triggered = bool(
            source_gap_to_last_committed >= self.v5_gap_trigger
            or predicted_gap_if_hold > self.v5_gap_hard_limit
            or open_gap_unclosed
        )
        blocked_by_hard_semantics = False
        retry_limit_extended = False
        retry_extension_reason = ""
        budget_override_reason = ""

        if density_after > self.v5_density_hard_upper:
            density_state = "above_hard"
        elif density_after > self.v5_density_upper:
            density_state = "above_upper"
        elif density_after < self.v5_density_lower:
            density_state = "below_lower"
        else:
            density_state = "in_band"

        if density_state == "below_lower":
            normal_topk = self.v5_topk_below
        elif density_state == "in_band":
            normal_topk = self.v5_topk_in
        else:
            normal_topk = self.v5_topk_above

        window_id = max(0, int(current_tick // max(self.v5_window_size, 1)))
        ws = self._v5_windows.setdefault(
            window_id,
            {"normal_commits": 0, "rescue_commits": 0, "support_scores": {}},
        )
        support_score = self._v5_support_score(
            num_matches=num_matches,
            num_inliers=num_inliers,
            v_t=v_t,
            q_t=q_t,
            r_t=r_t,
        )
        ws["support_scores"][int(source_input)] = float(support_score)
        ranked = sorted(
            ((int(k), float(v)) for k, v in ws["support_scores"].items()),
            key=lambda x: (-x[1], x[0]),
        )
        rank_map = {sid: idx + 1 for idx, (sid, _score) in enumerate(ranked)}
        window_candidate_rank = int(rank_map.get(int(source_input), len(ranked) + 1))
        is_support_topk = window_candidate_rank <= max(normal_topk, 0)
        is_coverage_topk = window_candidate_rank <= max(self.v5_coverage_topk, 1)
        is_extreme_support = bool(
            len(ranked) > 0 and support_score >= 0.98 * float(ranked[0][1]) and window_candidate_rank <= 1
        )

        if not anchor_count_available:
            anchor_target_state = "unavailable"
        elif anchor_count_before < self.v5_anchor_target_min:
            anchor_target_state = "below_band"
        elif anchor_count_before <= self.v5_anchor_target_max:
            anchor_target_state = "in_band"
        else:
            anchor_target_state = "above_band"
        anchor_soft_guard_triggered = bool(
            anchor_count_available and anchor_count_before > self.v5_anchor_target_max
        )
        is_too_close = source_gap_to_last_committed < self.v5_min_source_gap

        episode_id = long_gap_episode_id if long_gap_episode_id > 0 else int(source_input // max(self.v5_window_size, 1))
        episode_used = int(self._v5_episode_rescue_counts.get(episode_id, 0))
        dynamic_extra_budget = 0
        if source_gap_to_last_committed >= self.v5_gap_trigger:
            dynamic_extra_budget += 1 + max(0, (source_gap_to_last_committed - self.v5_gap_trigger) // 4)
        if predicted_gap_if_hold > self.v5_gap_hard_limit:
            dynamic_extra_budget += 1 + max(0, (predicted_gap_if_hold - self.v5_gap_hard_limit) // 4)
        if open_gap_unclosed:
            dynamic_extra_budget += 1
        gap_dynamic_budget = self.v5_gap_rescue_base_budget + dynamic_extra_budget
        can_gap_rescue = bool(gap_rescue_triggered and episode_used < gap_dynamic_budget)

        debug = dict(base_debug)
        debug.update(
            {
                "commit_channel": "",
                "rescue_channel": "",
                "R_t": r_t,
                "V_t": v_t,
                "Q_t": q_t,
                "support_score": float(support_score),
                "num_matches": num_matches,
                "num_inliers": num_inliers,
                "support_count": num_inliers,
                "source_gap_to_last_committed": source_gap_to_last_committed,
                "predicted_gap_if_hold": predicted_gap_if_hold,
                "density_before": density_before,
                "density_after": density_after,
                "density_state": density_state,
                "expected_min_keyframes": expected_min_keyframes,
                "keyframe_deficit": keyframe_deficit,
                "recent_keyframe_growth": recent_keyframe_growth,
                "keyframe_growth_recent": recent_keyframe_growth,
                "growth_plateau": growth_plateau,
                "window_id": window_id,
                "window_candidate_rank": window_candidate_rank,
                "window_support_score": float(support_score),
                "window_budget_used": int(ws["normal_commits"] + ws["rescue_commits"]),
                "window_normal_commit_count": int(ws["normal_commits"]),
                "window_rescue_commit_count": int(ws["rescue_commits"]),
                "long_gap_episode_id": long_gap_episode_id,
                "rescue_budget_used": episode_used,
                "gap_rescue_budget_used": episode_used,
                "retry_limit_extended": retry_limit_extended,
                "retry_extension_reason": retry_extension_reason,
                "budget_override_reason": budget_override_reason,
                "coverage_rescue_triggered": coverage_rescue_triggered,
                "gap_rescue_triggered": gap_rescue_triggered,
                "blocked_by_hard_semantics": blocked_by_hard_semantics,
                "anchor_count_before": anchor_count_before,
                "anchor_target_state": anchor_target_state,
                "anchor_soft_guard_triggered": anchor_soft_guard_triggered,
                "anchor_override_reason": "",
                "blocked_reason": "",
                "is_gap_critical": gap_rescue_triggered,
                "is_coverage_floor": coverage_rescue_triggered,
                "is_support_topk": is_support_topk,
                "is_duplicate": is_duplicate,
                "is_surrogate": is_surrogate,
                "is_contamination_risk": is_contamination_risk,
            }
        )

        hard_semantics_ok = bool(
            (not is_duplicate)
            and (not is_surrogate)
            and (not is_contamination_risk)
            and (source_input >= 0)
            and (r_t < self.v5_max_r)
            and (v_t >= self.v5_min_v)
            and (q_t >= self.v5_min_q)
            and (num_matches >= self.min_geom_matches)
        )
        if (not hard_semantics_ok) or density_state == "above_hard":
            debug["blocked_by_hard_semantics"] = True
            debug["blocked_reason"] = "hard_semantics"
            if gap_rescue_triggered:
                return RecoveryCommitDecision("reject", "v5_gap_rescue_reject_invalid_semantics", debug)
            if coverage_rescue_triggered:
                return RecoveryCommitDecision("reject", "v5_coverage_rescue_reject_invalid_semantics", debug)
            return RecoveryCommitDecision("reject", "v5_reject_invalid_semantics", debug)

        if age > self.max_candidate_age:
            debug["blocked_reason"] = "reject_age"
            return RecoveryCommitDecision("reject", "v5_reject_age", debug)

        if hold_retries >= self.max_hold_retries:
            if gap_rescue_triggered and self.v5_retry_extension_for_gap:
                retry_limit_extended = True
                retry_extension_reason = "gap_rescue"
            elif coverage_rescue_triggered and self.v5_retry_extension_for_coverage:
                retry_limit_extended = True
                retry_extension_reason = "coverage_rescue"
            else:
                debug["blocked_reason"] = "reject_retry_limit"
                return RecoveryCommitDecision("reject", "v5_reject_retry_limit", debug)
            debug["retry_limit_extended"] = retry_limit_extended
            debug["retry_extension_reason"] = retry_extension_reason

        # A) dynamic gap rescue channel
        if gap_rescue_triggered:
            debug["commit_channel"] = "gap_rescue"
            debug["rescue_channel"] = "gap_rescue"
            if density_state == "above_upper" and predicted_gap_if_hold <= self.v5_gap_hard_limit:
                debug["blocked_reason"] = "density_above_upper_not_critical"
                return RecoveryCommitDecision("hold", "v5_hold_density_above_upper", debug)
            if not can_gap_rescue:
                debug["blocked_reason"] = "gap_rescue_budget"
                return RecoveryCommitDecision("hold", "v5_hold_gap_rescue_budget", debug)
            self._v5_episode_rescue_counts[episode_id] = episode_used + 1
            debug["rescue_budget_used"] = self._v5_episode_rescue_counts[episode_id]
            debug["gap_rescue_budget_used"] = self._v5_episode_rescue_counts[episode_id]
            if dynamic_extra_budget > 0:
                debug["budget_override_reason"] = "dynamic_gap_budget_extension"
            if anchor_soft_guard_triggered:
                debug["anchor_override_reason"] = "gap_rescue"
            ws["rescue_commits"] = int(ws["rescue_commits"]) + 1
            debug["window_budget_used"] = int(ws["normal_commits"] + ws["rescue_commits"])
            return RecoveryCommitDecision("commit", "v5_gap_rescue_commit", debug)

        # B) coverage deficit / growth plateau rescue channel
        if coverage_rescue_triggered:
            debug["commit_channel"] = "coverage_rescue"
            debug["rescue_channel"] = "coverage_rescue"
            if density_state == "above_upper" and (not is_extreme_support):
                debug["blocked_reason"] = "coverage_density_above_upper"
                return RecoveryCommitDecision("hold", "v5_hold_density_above_upper", debug)
            if (not is_coverage_topk) and (not growth_plateau):
                debug["blocked_reason"] = "coverage_not_topk"
                return RecoveryCommitDecision("hold", "v5_coverage_floor_hold_not_topk", debug)
            if growth_plateau and window_candidate_rank > 1:
                debug["blocked_reason"] = "growth_plateau_pick_best_only"
                return RecoveryCommitDecision("hold", "v5_growth_rescue_hold_not_best", debug)
            # coverage rescue can override too_close / anchor_soft / normal budget by design
            debug["budget_override_reason"] = "coverage_deficit_or_growth_plateau"
            if is_too_close:
                debug["anchor_override_reason"] = "too_close_override_by_coverage_rescue"
            ws["rescue_commits"] = int(ws["rescue_commits"]) + 1
            debug["window_budget_used"] = int(ws["normal_commits"] + ws["rescue_commits"])
            return RecoveryCommitDecision("commit", "v5_coverage_rescue_commit", debug)

        # C) target-band normal sparse channel
        debug["commit_channel"] = "target_band_normal"
        if density_state == "above_upper":
            if not is_extreme_support:
                debug["blocked_reason"] = "density_above_upper"
                return RecoveryCommitDecision("hold", "v5_hold_density_above_upper", debug)
            debug["budget_override_reason"] = "extreme_support_above_upper"
        if not is_support_topk:
            debug["blocked_reason"] = "not_topk"
            return RecoveryCommitDecision("hold", "v5_hold_not_topk", debug)
        if is_too_close:
            debug["blocked_reason"] = "too_close"
            return RecoveryCommitDecision("hold", "v5_hold_too_close", debug)
        if anchor_soft_guard_triggered and density_state != "below_lower":
            debug["blocked_reason"] = "anchor_soft_guard"
            return RecoveryCommitDecision("hold", "v5_hold_anchor_soft_guard", debug)
        normal_budget = 2 if density_state == "below_lower" else 1
        if int(ws["normal_commits"]) >= normal_budget:
            debug["blocked_reason"] = "window_budget"
            return RecoveryCommitDecision("hold", "v5_hold_window_budget", debug)
        ws["normal_commits"] = int(ws["normal_commits"]) + 1
        debug["window_budget_used"] = int(ws["normal_commits"] + ws["rescue_commits"])
        return RecoveryCommitDecision("commit", "v5_support_ranked_sparse_commit", debug)

    def _decide_strict_v3(
        self,
        candidate: dict[str, Any],
        context: dict[str, Any],
        base_debug: dict[str, Any],
    ) -> RecoveryCommitDecision:
        scores = candidate.get("scores", {}) or {}
        source_payload = candidate.get("source_payload", {}) or {}
        inlier_evidence = source_payload.get("inlier_evidence", {}) or {}
        current_tick = int(context.get("current_tick_frame_id", -1))
        source_input = int(candidate.get("source_input_index", candidate.get("source_frame_id", -1)))
        age = max(0, current_tick - source_input)
        hold_retries = int(candidate.get("_hold_retries", 0))
        density_before = _f(context.get("keyframe_density_per_100", 0.0))
        density_after = density_before + (100.0 / float(max(current_tick, 1)))
        source_gap_to_last_committed = int(context.get("source_gap_to_last_committed", 0))
        predicted_gap_if_hold = int(context.get("predicted_gap_if_hold", source_gap_to_last_committed))
        main_chain_gap_p90 = _f(context.get("main_chain_gap_p90_recent", 0.0))

        v_t = _f(scores.get("V_t"))
        q_t = _f(scores.get("Q_t"))
        r_t = _f(scores.get("R_t"))
        num_matches = int(_f(inlier_evidence.get("num_matches", 0), 0.0))
        num_inliers = int(_f(context.get("source_num_inliers", 0), 0.0))
        anchor_count_available = bool(context.get("anchor_count_available", False))
        anchor_count_before_raw = context.get("anchor_count_before", None)
        anchor_count_before = int(anchor_count_before_raw) if anchor_count_before_raw is not None else -1

        is_duplicate = bool(context.get("source_already_committed", False))
        is_surrogate = bool(context.get("is_surrogate", False))
        is_contamination_risk = bool(context.get("is_contamination_risk", False))
        geom_ok = num_matches >= self.min_geom_matches
        value_ok = v_t >= self.v3_min_v and q_t >= self.v3_min_q and r_t < self.v3_max_r
        age_ok = age <= self.max_candidate_age
        not_too_close = source_gap_to_last_committed >= self.v3_min_source_gap
        density_ok = density_after <= self.v3_density_upper
        is_gap_critical = bool(
            source_gap_to_last_committed >= self.v3_gap_critical_trigger
            or predicted_gap_if_hold > self.v3_gap_hard_limit
            or (
                main_chain_gap_p90 >= float(max(self.v3_gap_hard_limit - 1, 1))
                and source_gap_to_last_committed >= self.v3_min_source_gap
            )
        )
        is_coverage_sparse = bool(
            source_gap_to_last_committed >= self.v3_min_source_gap
            or predicted_gap_if_hold >= self.v3_gap_critical_trigger
            or main_chain_gap_p90 >= 4.0
        )

        window_id = max(0, int(current_tick // max(self.v3_window_size, 1)))
        ws = self._v3_windows.setdefault(
            window_id,
            {
                "normal_commits": 0,
                "gap_critical_commits": 0,
                "support_scores": {},
            },
        )
        support_score = self._v3_support_score(
            num_matches=num_matches,
            num_inliers=num_inliers,
            v_t=v_t,
            q_t=q_t,
            r_t=r_t,
        )
        ws["support_scores"][int(source_input)] = float(support_score)
        ranked = sorted(
            ((int(k), float(v)) for k, v in ws["support_scores"].items()),
            key=lambda x: (-x[1], x[0]),
        )
        rank_map = {sid: idx + 1 for idx, (sid, _score) in enumerate(ranked)}
        window_candidate_rank = int(rank_map.get(int(source_input), len(ranked) + 1))
        is_support_topk = window_candidate_rank <= max(self.v3_support_topk, 1)
        support_override_ok = bool(
            is_support_topk and len(ranked) > 0 and support_score >= 0.95 * float(ranked[0][1])
        )

        anchor_guard_triggered = bool(
            self.v3_anchor_guard_enabled
            and anchor_count_available
            and anchor_count_before >= self.v3_anchor_soft_limit
        )
        anchor_guard_action = "none"
        anchor_override_reason = ""
        if anchor_guard_triggered:
            if is_gap_critical:
                anchor_guard_action = "override"
                anchor_override_reason = "gap_critical"
            elif support_override_ok:
                anchor_guard_action = "override"
                anchor_override_reason = "support_topk_override"
            else:
                anchor_guard_action = "hold"

        debug = dict(base_debug)
        debug.update(
            {
                "commit_channel": "",
                "R_t": r_t,
                "V_t": v_t,
                "Q_t": q_t,
                "num_matches": num_matches,
                "num_inliers": num_inliers,
                "window_id": window_id,
                "window_candidate_rank": window_candidate_rank,
                "window_support_score": float(support_score),
                "window_normal_commit_count": int(ws["normal_commits"]),
                "window_gap_critical_commit_count": int(ws["gap_critical_commits"]),
                "density_before": density_before,
                "density_after": density_after,
                "anchor_count_before": anchor_count_before,
                "anchor_count_available": anchor_count_available,
                "anchor_count_after_if_available": None,
                "anchor_guard_triggered": anchor_guard_triggered,
                "anchor_guard_action": anchor_guard_action,
                "anchor_override_reason": anchor_override_reason,
                "blocked_reason": "",
                "is_gap_critical": is_gap_critical,
                "is_support_topk": is_support_topk,
                "is_coverage_sparse": is_coverage_sparse,
                "is_duplicate": is_duplicate,
                "is_surrogate": is_surrogate,
                "is_contamination_risk": is_contamination_risk,
            }
        )

        if is_duplicate or is_surrogate or is_contamination_risk:
            debug["blocked_reason"] = "invalid_semantics"
            return RecoveryCommitDecision("reject", "v3_reject_invalid_semantics", debug)
        if not age_ok:
            debug["blocked_reason"] = "reject_age"
            return RecoveryCommitDecision("reject", "v3_reject_age", debug)
        if hold_retries >= self.max_hold_retries:
            debug["blocked_reason"] = "reject_retry_limit"
            return RecoveryCommitDecision("reject", "v3_reject_retry_limit", debug)
        if not geom_ok or not value_ok:
            debug["blocked_reason"] = "low_support"
            return RecoveryCommitDecision("reject", "v3_reject_low_support", debug)
        if not density_ok:
            debug["blocked_reason"] = "density_high"
            return RecoveryCommitDecision("hold", "v3_hold_density_high", debug)
        if (not is_gap_critical) and (not not_too_close):
            debug["blocked_reason"] = "too_close_to_existing_keyframe"
            return RecoveryCommitDecision("hold", "v3_hold_too_close_to_existing_keyframe", debug)
        if anchor_guard_triggered and anchor_guard_action == "hold":
            debug["blocked_reason"] = "anchor_guard"
            return RecoveryCommitDecision("hold", "v3_hold_anchor_guard", debug)

        if is_gap_critical:
            if int(ws["gap_critical_commits"]) >= int(self.v3_max_gap_critical_per_window):
                debug["blocked_reason"] = "window_gap_critical_budget"
                return RecoveryCommitDecision("hold", "v3_hold_window_budget", debug)
            ws["gap_critical_commits"] = int(ws["gap_critical_commits"]) + 1
            debug["window_gap_critical_commit_count"] = int(ws["gap_critical_commits"])
            debug["commit_channel"] = "gap_critical"
            return RecoveryCommitDecision("commit", "v3_gap_critical_commit", debug)

        if not is_coverage_sparse:
            debug["blocked_reason"] = "coverage_not_sparse"
            return RecoveryCommitDecision("reject", "v3_reject_low_support", debug)
        if not is_support_topk:
            debug["blocked_reason"] = "not_support_topk"
            return RecoveryCommitDecision("hold", "v3_hold_not_topk", debug)
        if int(ws["normal_commits"]) >= int(self.v3_max_normal_per_window):
            debug["blocked_reason"] = "window_normal_budget"
            return RecoveryCommitDecision("hold", "v3_hold_window_budget", debug)

        ws["normal_commits"] = int(ws["normal_commits"]) + 1
        debug["window_normal_commit_count"] = int(ws["normal_commits"])
        debug["commit_channel"] = "support_ranked_sparse"
        return RecoveryCommitDecision("commit", "v3_support_ranked_sparse_commit", debug)

    def decide(
        self,
        candidate: dict[str, Any],
        context: dict[str, Any],
    ) -> RecoveryCommitDecision:
        if self.mode == "off":
            return RecoveryCommitDecision("commit", "control_off", {})

        source_payload = candidate.get("source_payload", {}) or {}
        inlier_evidence = source_payload.get("inlier_evidence", {}) or {}
        scores = candidate.get("scores", {}) or {}

        current_tick = int(context.get("current_tick_frame_id", -1))
        source_input = int(candidate.get("source_input_index", candidate.get("source_frame_id", -1)))
        age = max(0, current_tick - source_input)
        hold_retries = int(candidate.get("_hold_retries", 0))

        num_matches = int(_f(inlier_evidence.get("num_matches", 0), 0.0))
        geom_ok = num_matches >= self.min_geom_matches
        value_ok = (_f(scores.get("V_t")) >= self.min_v_for_commit) and (
            _f(scores.get("Q_t")) >= self.min_q_for_commit
        )
        age_ok = age <= self.max_candidate_age

        density = _f(context.get("keyframe_density_per_100", 0.0))
        main_chain_gap_p90 = _f(context.get("main_chain_gap_p90_recent", 0.0))
        recent_recovery_commits = int(context.get("recent_recovery_commit_count", 0))
        source_gap_to_last_committed = int(context.get("source_gap_to_last_committed", 0))
        predicted_gap_if_hold = int(context.get("predicted_gap_if_hold", source_gap_to_last_committed))
        long_gap_episode_id = int(context.get("long_gap_episode_id", 0))
        source_committed_action = str(context.get("source_committed_action", ""))

        limit = self.max_per_window
        if self.mode == "adaptive":
            limit = self._adaptive_budget(main_chain_gap_p90, density)
        is_gap_high_risk = main_chain_gap_p90 >= self.gap_override_threshold
        if is_gap_high_risk and not self._in_gap_interval:
            self._in_gap_interval = True
            self._gap_interval_index += 1
            self._gap_interval_override_count = 0
        elif (not is_gap_high_risk) and self._in_gap_interval:
            self._in_gap_interval = False
            self._gap_interval_override_count = 0

        source_already_committed = bool(context.get("source_already_committed", False))

        debug = {
            "geom_ok": geom_ok,
            "value_ok": value_ok,
            "age_ok": age_ok,
            "num_matches": num_matches,
            "V_t": _f(scores.get("V_t")),
            "Q_t": _f(scores.get("Q_t")),
            "R_t": _f(scores.get("R_t")),
            "candidate_age": age,
            "recent_recovery_commit_count": recent_recovery_commits,
            "window_size": self.window_size,
            "window_limit": limit,
            "keyframe_density_per_100": density,
            "baseline_density_upper_per_100": self.baseline_density_upper,
            "main_chain_gap_p90_recent": main_chain_gap_p90,
            "hold_retries": hold_retries,
            "source_already_committed": source_already_committed,
            "gap_override_threshold": self.gap_override_threshold,
            "is_gap_high_risk": is_gap_high_risk,
            "gap_interval_index": self._gap_interval_index if is_gap_high_risk else 0,
            "gap_interval_override_count": self._gap_interval_override_count,
            "gap_override_max_per_interval": self.gap_override_max_per_interval,
            "source_gap_to_last_committed": source_gap_to_last_committed,
            "predicted_gap_if_hold": predicted_gap_if_hold,
            "long_gap_episode_id": long_gap_episode_id,
            "source_committed_action": source_committed_action,
            "v2_source_gap_trigger": self.v2_source_gap_trigger,
            "v2_predicted_gap_trigger": self.v2_predicted_gap_trigger,
            "v2_episode_override_budget": self.v2_episode_override_budget,
        }

        if source_already_committed:
            debug["blocked_reason"] = "duplicate_source_committed"
            return RecoveryCommitDecision("reject", "duplicate_source_committed", debug)
        if self.mode == "recovery_commit_strict_v3":
            return self._decide_strict_v3(candidate, context, debug)
        if self.mode == "recovery_commit_balanced_v4":
            return self._decide_balanced_v4(candidate, context, debug)
        if self.mode == "recovery_commit_rescue_v5":
            return self._decide_rescue_v5(candidate, context, debug)
        if not age_ok:
            debug["blocked_reason"] = "reject_age"
            return RecoveryCommitDecision("reject", "reject_age", debug)
        if not geom_ok:
            debug["blocked_reason"] = "geometry_support_low"
            return RecoveryCommitDecision("reject", "geometry_support_low", debug)

        allow_gap_override = self.mode == "conservative_gap_aware" and is_gap_high_risk
        override_value_ok = (_f(scores.get("V_t")) >= self.gap_override_min_v) and (
            _f(scores.get("Q_t")) >= self.gap_override_min_q
        )
        override_density_ok = density <= self.gap_override_density_upper
        override_budget_ok = self._gap_interval_override_count < self.gap_override_max_per_interval
        debug["gap_override_value_ok"] = override_value_ok
        debug["gap_override_density_ok"] = override_density_ok
        debug["gap_override_budget_ok"] = override_budget_ok
        debug["density_before"] = density
        debug["density_after"] = density + (100.0 / float(max(current_tick, 1)))
        debug["override_reason"] = ""
        debug["episode_override_count"] = 0

        if (
            allow_gap_override
            and recent_recovery_commits >= limit
            and override_value_ok
            and override_density_ok
            and override_budget_ok
        ):
            self._gap_interval_override_count += 1
            debug["gap_interval_override_count"] = self._gap_interval_override_count
            debug["override_reason"] = "gap_aware_v1_override"
            debug["episode_override_count"] = self._gap_interval_override_count
            return RecoveryCommitDecision("commit", "gap_aware_v1_override", debug)

        # v2: explicitly target source-gap long-tail episodes.
        if self.mode == "conservative_gap_aware_v2":
            episode_count = self._v2_episode_override_counts.get(long_gap_episode_id, 0)
            v2_density_ok = density <= self.gap_override_density_upper
            v2_value_ok = (_f(scores.get("V_t")) >= self.gap_override_min_v) and (
                _f(scores.get("Q_t")) >= self.gap_override_min_q
            )
            r_t = _f(scores.get("R_t"))
            v2_risk_ok = (r_t < 0.75) or (0.40 <= r_t < 0.75)
            v2_episode_budget_ok = (long_gap_episode_id > 0) and (episode_count < self.v2_episode_override_budget)
            source_gap_trigger = source_gap_to_last_committed >= self.v2_source_gap_trigger
            predicted_gap_trigger = predicted_gap_if_hold > self.v2_predicted_gap_trigger
            v2_trigger = source_gap_trigger and predicted_gap_trigger
            v2_blocked_path = bool((recent_recovery_commits >= limit))
            debug["v2_density_ok"] = v2_density_ok
            debug["v2_value_ok"] = v2_value_ok
            debug["v2_risk_ok"] = v2_risk_ok
            debug["v2_episode_budget_ok"] = v2_episode_budget_ok
            debug["v2_source_gap_trigger_active"] = source_gap_trigger
            debug["v2_predicted_gap_trigger_active"] = predicted_gap_trigger
            debug["v2_trigger_active"] = v2_trigger
            debug["v2_blocked_path"] = v2_blocked_path
            if (
                v2_trigger
                and v2_blocked_path
                and v2_density_ok
                and v2_value_ok
                and v2_risk_ok
                and v2_episode_budget_ok
            ):
                self._v2_episode_override_counts[long_gap_episode_id] = episode_count + 1
                debug["episode_override_count"] = self._v2_episode_override_counts[long_gap_episode_id]
                debug["override_reason"] = "gap_aware_v2_source_gap_override"
                return RecoveryCommitDecision("commit", "gap_aware_v2_source_gap_override", debug)

        # Density/rate protection first for conservative and adaptive modes
        if recent_recovery_commits >= limit:
            debug["blocked_reason"] = "hold_window_rate_limit"
            return RecoveryCommitDecision("hold", "hold_window_rate_limit", debug)
        if density > self.baseline_density_upper and main_chain_gap_p90 <= self.main_chain_gap_small:
            debug["blocked_reason"] = "hold_density_high"
            return RecoveryCommitDecision("hold", "hold_density_high", debug)

        if hold_retries >= self.max_hold_retries:
            debug["blocked_reason"] = "reject_retry_limit"
            return RecoveryCommitDecision("reject", "reject_retry_limit", debug)

        if not value_ok:
            debug["blocked_reason"] = "value_support_low"
            return RecoveryCommitDecision("hold", "value_support_low", debug)

        return RecoveryCommitDecision("commit", "normal_commit", debug)
