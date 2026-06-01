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
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


@dataclass
class DirectFinalizationDecision:
    finalize: bool
    decision: str
    reason: str
    debug: dict[str, Any]


class DirectDensityController:
    """Gate direct keyframe finalization without changing R/V/Q admission semantics."""

    def __init__(self, args: Any) -> None:
        self.mode = str(getattr(args, "paper_aligned_direct_density_control", "off") or "off")
        self.density_lower = float(getattr(args, "paper_aligned_direct_density_lower_per_100", 28.0) or 28.0)
        self.density_target = float(getattr(args, "paper_aligned_direct_density_target_per_100", 35.0) or 35.0)
        self.density_upper = float(getattr(args, "paper_aligned_direct_density_upper_per_100", 45.0) or 45.0)
        self.density_hard_upper = float(
            getattr(args, "paper_aligned_direct_density_hard_upper_per_100", 50.0) or 50.0
        )
        self.gap_hard_limit = int(getattr(args, "paper_aligned_direct_gap_hard_limit", 20) or 20)
        self.redundant_source_gap = int(
            getattr(args, "paper_aligned_direct_redundant_source_gap", 3) or 3
        )
        if self.mode == "conservative":
            self.density_upper = min(self.density_upper, 40.0)
            self.density_target = min(self.density_target, 32.0)
            self.redundant_source_gap = min(self.redundant_source_gap, 2)

    @property
    def enabled(self) -> bool:
        return self.mode not in {"", "off", "none"}

    def decide(
        self,
        *,
        frame_id: int,
        runtime_action: str,
        baseline_should_add: bool,
        is_test: bool,
        is_bootstrap_phase: bool,
        density_before: float,
        source_gap_to_last_keyframe: int,
        main_chain_gap_before: float,
        main_chain_gap_after_if_hold: float,
        anchor_changed: bool,
        support_triggered: bool,
        median_displacement: float,
        displacement_threshold: float,
        num_matches: int,
        min_num_inliers: int,
        pose_inliers: int,
        novelty_proxy: float,
    ) -> DirectFinalizationDecision:
        dbg: dict[str, Any] = {
            "mode": self.mode,
            "density_before": density_before,
            "source_gap_to_last_keyframe": source_gap_to_last_keyframe,
            "main_chain_gap_before": main_chain_gap_before,
            "main_chain_gap_after_if_hold": main_chain_gap_after_if_hold,
            "anchor_changed": anchor_changed,
            "support_triggered": support_triggered,
            "median_displacement": median_displacement,
            "displacement_threshold": displacement_threshold,
            "num_matches": num_matches,
            "pose_inliers": pose_inliers,
            "novelty_proxy": novelty_proxy,
        }
        direct_candidate = runtime_action in {"direct_admit", "current_frame_surrogate_commit"}
        dbg["direct_admit_candidate"] = direct_candidate

        if not self.enabled or not direct_candidate:
            return DirectFinalizationDecision(
                finalize=True,
                decision="finalize",
                reason="control_off_or_not_direct",
                debug=dbg,
            )
        if is_test or is_bootstrap_phase:
            dbg["direct_keyframe_finalized"] = True
            return DirectFinalizationDecision(
                finalize=True,
                decision="finalize",
                reason="test_or_bootstrap_bypass",
                debug=dbg,
            )

        gap_critical = bool(
            main_chain_gap_after_if_hold >= self.gap_hard_limit
            or main_chain_gap_before >= self.gap_hard_limit - 2
        )
        displacement_novelty = bool(
            median_displacement >= 1.35 * max(displacement_threshold, 1e-6)
        )
        # Support bridge inflates match counts; do not treat match count alone as novelty.
        match_novelty = bool(
            num_matches >= 1.5 * max(min_num_inliers, 1) and not support_triggered
        )
        high_novelty = bool(
            displacement_novelty
            or match_novelty
            or (novelty_proxy >= 0.72 and displacement_novelty)
        )
        support_needed = bool(
            support_triggered
            and num_matches >= max(min_num_inliers, 1)
            and density_before < self.density_target
        )
        anchor_boundary = bool(anchor_changed)

        if gap_critical:
            dbg["finalize_gap_critical"] = True
            return DirectFinalizationDecision(
                finalize=True,
                decision="finalize_gap_critical",
                reason="gap_critical_override",
                debug=dbg,
            )
        if density_before < self.density_lower:
            dbg["finalize_support_needed"] = True
            return DirectFinalizationDecision(
                finalize=True,
                decision="finalize_support_needed",
                reason="density_below_lower_band",
                debug=dbg,
            )
        if anchor_boundary:
            dbg["finalize_anchor_boundary"] = True
            return DirectFinalizationDecision(
                finalize=True,
                decision="finalize_anchor_boundary",
                reason="finalize_anchor_boundary",
                debug=dbg,
            )
        if support_needed:
            return DirectFinalizationDecision(
                finalize=True,
                decision="finalize_support_needed",
                reason="support_needed_low_density",
                debug=dbg,
            )

        redundant = bool(
            source_gap_to_last_keyframe <= self.redundant_source_gap
            and main_chain_gap_before <= 5.0
            and not anchor_changed
            and density_before >= self.density_target
            and not gap_critical
        )
        if redundant and not displacement_novelty:
            dbg["hold_redundant"] = True
            return DirectFinalizationDecision(
                finalize=False,
                decision="hold_redundant",
                reason="redundant_close_gap_safe_chain",
                debug=dbg,
            )

        if density_before > self.density_upper:
            if gap_critical or displacement_novelty:
                decision = "finalize_gap_critical" if gap_critical else "finalize_high_novelty"
                return DirectFinalizationDecision(
                    finalize=True,
                    decision=decision,
                    reason=decision,
                    debug=dbg,
                )
            dbg["hold_density_high"] = True
            return DirectFinalizationDecision(
                finalize=False,
                decision="hold_density_high",
                reason="density_above_upper_band",
                debug=dbg,
            )

        if density_before > self.density_target and not displacement_novelty and not match_novelty:
            if source_gap_to_last_keyframe <= self.redundant_source_gap:
                dbg["hold_redundant"] = True
                return DirectFinalizationDecision(
                    finalize=False,
                    decision="hold_redundant",
                    reason="above_target_close_gap_low_novelty",
                    debug=dbg,
                )

        if high_novelty:
            return DirectFinalizationDecision(
                finalize=True,
                decision="finalize_high_novelty",
                reason="finalize_high_novelty",
                debug=dbg,
            )

        dbg["direct_keyframe_finalized"] = True
        return DirectFinalizationDecision(
            finalize=True,
            decision="finalize",
            reason="default_finalize",
            debug=dbg,
        )
