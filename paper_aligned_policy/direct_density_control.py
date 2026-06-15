from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _clamp01(x: Any, default: float = 0.0) -> float:
    v = _f(x, default)
    return max(0.0, min(1.0, v))


@dataclass
class DirectFinalizationDecision:
    finalize: bool
    decision: str
    reason: str
    debug: dict[str, Any]


@dataclass
class _BudgetWindow:
    window_id: int = -1
    high_novelty_used: int = 0
    support_needed_used: int = 0
    early_rescue_used: int = 0
    gap_rescue_used: int = 0
    post500_gap_rescue_used: int = 0
    value_hold_used: int = 0


class DirectDensityController:
    """Gate direct keyframe finalization without changing R/V/Q admission semantics."""

    def __init__(self, args: Any) -> None:
        self.mode = str(getattr(args, "paper_aligned_direct_density_control", "off") or "off")
        self.density_lower = float(getattr(args, "paper_aligned_direct_density_lower_per_100", 25.0) or 25.0)
        self.density_target = float(getattr(args, "paper_aligned_direct_density_target_per_100", 35.0) or 35.0)
        self.density_upper = float(getattr(args, "paper_aligned_direct_density_upper_per_100", 75.0) or 75.0)
        self.density_hard_upper = float(
            getattr(args, "paper_aligned_direct_density_hard_upper_per_100", 90.0) or 90.0
        )
        self.gap_hard_limit = int(getattr(args, "paper_aligned_direct_gap_hard_limit", 20) or 20)
        self.redundant_source_gap = int(
            getattr(args, "paper_aligned_direct_redundant_source_gap", 3) or 3
        )
        self.hysteresis_margin = float(
            getattr(args, "paper_aligned_direct_density_hysteresis_margin", 3.0) or 3.0
        )
        self.high_novelty_budget_per_100 = int(
            getattr(args, "paper_aligned_direct_high_novelty_budget_per_100", 8) or 8
        )
        self.support_needed_budget_per_100 = int(
            getattr(args, "paper_aligned_direct_support_needed_budget_per_100", 6) or 6
        )
        self.min_growth_per_100 = float(
            getattr(args, "paper_aligned_direct_min_growth_per_100", 20.0) or 20.0
        )
        self.baseline_relative_lower_ratio = float(
            getattr(args, "paper_aligned_direct_baseline_relative_lower_ratio", 0.8) or 0.8
        )
        self.baseline_density_per_100 = float(
            getattr(args, "paper_aligned_direct_baseline_density_per_100", 27.0) or 27.0
        )
        self.prev_desc_update_on_hold = str(
            getattr(args, "paper_aligned_direct_update_prev_desc_on_hold", "off") or "off"
        )
        self.hold_tracking_bridge_mode = str(
            getattr(args, "paper_aligned_direct_hold_tracking_bridge_mode", "light") or "light"
        )
        if self.mode == "pose_rep_decouple_v1":
            self.density_lower = 18.0
            self.density_target = 28.0
            self.density_upper = 42.0
            self.density_hard_upper = 58.0
            self.gap_hard_limit = 15
            self.redundant_source_gap = 2
            self.min_growth_per_100 = 10.0
            self.baseline_relative_lower_ratio = 0.55
            if self.prev_desc_update_on_hold == "off":
                self.prev_desc_update_on_hold = "light"
        if self.mode in {
            "pose_rep_value_decouple_v2",
            "pose_rep_value_decouple_v3",
            "pose_rep_active_memory_v1",
            "pose_rep_active_memory_v2",
            "pose_rep_active_memory_v4",
            "pose_rep_active_memory_v5",
            "pose_rep_active_memory_v6",
            "pose_rep_active_memory_v8",
            "pose_rep_active_memory_v9",
            "pose_rep_active_memory_v24",
            "pose_rep_active_memory_v25",
            "pose_rep_active_memory_v26",
            "pose_rep_active_memory_v27",
        }:
            self.density_lower = 16.0
            self.density_target = 30.0
            self.density_upper = 70.0
            self.density_hard_upper = 92.0
            self.gap_hard_limit = 18
            self.redundant_source_gap = 2
            self.min_growth_per_100 = 8.0
            self.baseline_relative_lower_ratio = 0.45
            if self.prev_desc_update_on_hold == "off":
                self.prev_desc_update_on_hold = "light"
        if self.mode == "conservative":
            self.density_upper = min(self.density_upper, 40.0)
            self.density_target = min(self.density_target, 32.0)
            self.redundant_source_gap = min(self.redundant_source_gap, 2)
        if self.mode == "target_band_v1":
            self.density_lower = float(
                getattr(args, "paper_aligned_direct_density_lower_per_100", 28.0) or 28.0
            )
        self.early_rescue_budget_per_100 = int(
            getattr(args, "paper_aligned_direct_v2_2_early_rescue_budget_per_100", 8) or 8
        )
        self.early_rescue_density_stop = float(
            getattr(args, "paper_aligned_direct_v2_2_early_rescue_density_stop", 32.0) or 32.0
        )
        self.local_window_size = int(
            getattr(args, "paper_aligned_direct_local_window_size", 100) or 100
        )
        self.local_density_lower = float(
            getattr(args, "paper_aligned_direct_local_density_lower_per_100", 20.0) or 20.0
        )
        self.gap_critical_limit = int(getattr(args, "paper_aligned_direct_gap_hard_limit", 20) or 20)
        self.soft_gap_threshold = int(
            getattr(args, "paper_aligned_direct_v2_2_1_soft_gap_threshold", 6) or 6
        )
        self.hard_gap_threshold = int(
            getattr(args, "paper_aligned_direct_v2_2_1_hard_gap_threshold", 20) or 20
        )
        self.gap_rescue_budget_per_100 = int(
            getattr(args, "paper_aligned_direct_v2_2_1_gap_rescue_budget_per_100", 4) or 4
        )
        self.gap_rescue_density_upper_500 = float(
            getattr(args, "paper_aligned_direct_v2_2_1_gap_rescue_density_upper_500", 50.0) or 50.0
        )
        self.gap_rescue_density_upper_later = float(
            getattr(args, "paper_aligned_direct_v2_2_1_gap_rescue_density_upper_later", 45.0) or 45.0
        )
        self.preemptive_gap_threshold = int(
            getattr(args, "paper_aligned_direct_v2_2_2_preemptive_gap_threshold", 18) or 18
        )
        self.post500_gap_rescue_budget_per_100 = int(
            getattr(args, "paper_aligned_direct_v2_2_2_1_post500_gap_rescue_budget_per_100", 4) or 4
        )
        if self.mode == "pose_rep_decouple_v1":
            self.local_density_lower = 14.0
            self.soft_gap_threshold = 5
            self.hard_gap_threshold = 15
            self.gap_critical_limit = 15
        if self.mode in {
            "pose_rep_value_decouple_v2",
            "pose_rep_value_decouple_v3",
            "pose_rep_active_memory_v1",
            "pose_rep_active_memory_v2",
            "pose_rep_active_memory_v4",
            "pose_rep_active_memory_v5",
            "pose_rep_active_memory_v6",
            "pose_rep_active_memory_v8",
            "pose_rep_active_memory_v9",
            "pose_rep_active_memory_v24",
            "pose_rep_active_memory_v25",
            "pose_rep_active_memory_v26",
            "pose_rep_active_memory_v27",
        }:
            self.local_density_lower = 12.0
            self.soft_gap_threshold = 8
            self.hard_gap_threshold = 18
            self.gap_critical_limit = 18
            self.representation_value_hold_max = float(
                getattr(args, "paper_aligned_direct_representation_value_hold_max", 0.42)
                or 0.42
            )
            self.pose_reference_value_min = float(
                getattr(args, "paper_aligned_direct_pose_reference_value_min", 0.55)
                or 0.55
            )
            self.value_hold_budget_per_100 = int(
                getattr(args, "paper_aligned_direct_value_hold_budget_per_100", 5)
                or 5
            )
        if self.mode in {
            "pose_rep_value_decouple_v3",
            "pose_rep_active_memory_v1",
            "pose_rep_active_memory_v2",
            "pose_rep_active_memory_v4",
            "pose_rep_active_memory_v5",
            "pose_rep_active_memory_v6",
            "pose_rep_active_memory_v8",
            "pose_rep_active_memory_v9",
            "pose_rep_active_memory_v24",
            "pose_rep_active_memory_v25",
            "pose_rep_active_memory_v26",
            "pose_rep_active_memory_v27",
        }:
            self.value_hold_budget_per_100 = int(
                getattr(args, "paper_aligned_direct_value_hold_budget_per_100", 48)
                or 48
            )
            self.representation_value_hold_max = float(
                getattr(args, "paper_aligned_direct_representation_value_hold_max", 0.35)
                or 0.35
            )
            self.pose_reference_value_min = float(
                getattr(args, "paper_aligned_direct_pose_reference_value_min", 0.65)
                or 0.65
            )
        if self.mode == "pose_rep_active_memory_v27":
            self.density_upper = 72.0
            self.density_hard_upper = 94.0
            self.value_hold_budget_per_100 = int(
                getattr(args, "paper_aligned_direct_value_hold_budget_per_100", 36)
                or 36
            )
            self.representation_value_hold_max = float(
                getattr(args, "paper_aligned_direct_representation_value_hold_max", 0.38)
                or 0.38
            )
            self.pose_reference_value_min = float(
                getattr(args, "paper_aligned_direct_pose_reference_value_min", 0.58)
                or 0.58
            )
            self.utility_representation_min = float(
                getattr(args, "paper_aligned_direct_utility_representation_min", 0.38)
                or 0.38
            )
            self.utility_pose_reference_min = float(
                getattr(args, "paper_aligned_direct_utility_pose_reference_min", 0.58)
                or 0.58
            )
        if self.mode in {"pose_rep_active_memory_v1", "pose_rep_active_memory_v2"}:
            self.value_hold_budget_per_100 = int(
                getattr(args, "paper_aligned_direct_value_hold_budget_per_100", 85)
                or 85
            )
            self.representation_value_hold_max = float(
                getattr(args, "paper_aligned_direct_representation_value_hold_max", 0.38)
                or 0.38
            )
        if self.mode in {"target_band_v2_2_1", "target_band_v2_2_2", "target_band_v2_2_2_1"}:
            self.gap_critical_limit = self.hard_gap_threshold
        self._budget = _BudgetWindow()
        self._last_density_state = "unknown"

    @property
    def enabled(self) -> bool:
        return self.mode not in {"", "off", "none"}

    @property
    def is_v2(self) -> bool:
        return self.mode in {"target_band_v2", "target_band_v2_1", "target_band_v2_2"}

    @property
    def is_v21(self) -> bool:
        return self.mode == "target_band_v2_1"

    @property
    def is_v22(self) -> bool:
        return self.mode in {
            "target_band_v2_2",
            "target_band_v2_2_1",
            "target_band_v2_2_2",
            "target_band_v2_2_2_1",
        }

    @property
    def is_v221(self) -> bool:
        return self.mode == "target_band_v2_2_1"

    @property
    def is_v222(self) -> bool:
        return self.mode == "target_band_v2_2_2"

    @property
    def is_v2221(self) -> bool:
        return self.mode == "target_band_v2_2_2_1"

    @property
    def is_pose_rep_decouple(self) -> bool:
        return self.mode in {
            "pose_rep_decouple_v1",
            "pose_rep_value_decouple_v2",
            "pose_rep_value_decouple_v3",
            "pose_rep_active_memory_v1",
            "pose_rep_active_memory_v2",
            "pose_rep_active_memory_v4",
            "pose_rep_active_memory_v5",
            "pose_rep_active_memory_v6",
            "pose_rep_active_memory_v8",
            "pose_rep_active_memory_v9",
            "pose_rep_active_memory_v24",
            "pose_rep_active_memory_v25",
            "pose_rep_active_memory_v26",
            "pose_rep_active_memory_v27",
        }

    @property
    def is_pose_rep_value_decouple(self) -> bool:
        return self.mode in {
            "pose_rep_value_decouple_v2",
            "pose_rep_value_decouple_v3",
            "pose_rep_active_memory_v1",
            "pose_rep_active_memory_v2",
            "pose_rep_active_memory_v4",
            "pose_rep_active_memory_v5",
            "pose_rep_active_memory_v6",
            "pose_rep_active_memory_v8",
            "pose_rep_active_memory_v9",
            "pose_rep_active_memory_v24",
            "pose_rep_active_memory_v25",
            "pose_rep_active_memory_v26",
            "pose_rep_active_memory_v27",
        }

    @property
    def is_pose_rep_value_v3(self) -> bool:
        return self.mode in {
            "pose_rep_value_decouple_v3",
            "pose_rep_active_memory_v1",
            "pose_rep_active_memory_v2",
            "pose_rep_active_memory_v4",
            "pose_rep_active_memory_v5",
            "pose_rep_active_memory_v6",
            "pose_rep_active_memory_v8",
            "pose_rep_active_memory_v9",
            "pose_rep_active_memory_v24",
            "pose_rep_active_memory_v25",
            "pose_rep_active_memory_v26",
            "pose_rep_active_memory_v27",
        }

    @property
    def is_pose_rep_active_memory_v1(self) -> bool:
        return self.mode == "pose_rep_active_memory_v1"

    @property
    def is_pose_rep_active_memory_v2(self) -> bool:
        return self.mode == "pose_rep_active_memory_v2"

    @property
    def is_pose_rep_active_memory_v4(self) -> bool:
        return self.mode == "pose_rep_active_memory_v4"

    @property
    def is_pose_rep_active_memory_v5(self) -> bool:
        return self.mode == "pose_rep_active_memory_v5"

    @property
    def is_pose_rep_active_memory_v6(self) -> bool:
        return self.mode == "pose_rep_active_memory_v6"

    @property
    def is_pose_rep_active_memory_v8(self) -> bool:
        return self.mode == "pose_rep_active_memory_v8"

    @property
    def is_pose_rep_active_memory_v9(self) -> bool:
        return self.mode == "pose_rep_active_memory_v9"

    @property
    def is_pose_rep_active_memory_v24(self) -> bool:
        return self.mode == "pose_rep_active_memory_v24"

    @property
    def is_pose_rep_active_memory_v25(self) -> bool:
        return self.mode == "pose_rep_active_memory_v25"

    @property
    def is_pose_rep_active_memory_v26(self) -> bool:
        return self.mode == "pose_rep_active_memory_v26"

    @property
    def is_pose_rep_active_memory_v27(self) -> bool:
        return self.mode == "pose_rep_active_memory_v27"

    @property
    def is_pose_rep_active_memory(self) -> bool:
        return self.mode in {
            "pose_rep_active_memory_v1",
            "pose_rep_active_memory_v2",
            "pose_rep_active_memory_v4",
            "pose_rep_active_memory_v5",
            "pose_rep_active_memory_v6",
            "pose_rep_active_memory_v8",
            "pose_rep_active_memory_v9",
            "pose_rep_active_memory_v24",
            "pose_rep_active_memory_v25",
            "pose_rep_active_memory_v26",
            "pose_rep_active_memory_v27",
        }

    @property
    def is_v222_family(self) -> bool:
        return self.is_v222 or self.is_v2221

    def _gap_rescue_density_cap(self, frame_id: int) -> float:
        if (self.is_v221 or self.is_v222_family) and int(frame_id) <= 500:
            return self.gap_rescue_density_upper_500
        if self.is_v221 or self.is_v222_family:
            return self.gap_rescue_density_upper_later
        return self.density_upper

    def should_update_prev_desc_on_hold(self, decision: str, density_state: str) -> bool:
        if self.prev_desc_update_on_hold == "on":
            return True
        if self.prev_desc_update_on_hold == "off":
            return False
        if self.prev_desc_update_on_hold == "light":
            if self.hold_tracking_bridge_mode == "none":
                return False
            if self.is_pose_rep_decouple:
                return decision in {
                    "hold_redundant",
                    "hold_density_high",
                    "hold_low_representation_value",
                }
            return decision == "hold_redundant" and density_state == "in_band"
        return False

    def should_enqueue_hold_recovery(self) -> bool:
        return self.enabled and not self.is_pose_rep_decouple

    def _window_id(self, frame_id: int) -> int:
        return int(frame_id) // 100

    def _sync_budget_window(self, frame_id: int) -> None:
        wid = self._window_id(frame_id)
        if wid != self._budget.window_id:
            self._budget = _BudgetWindow(window_id=wid)

    def _density_state(
        self,
        density_before: float,
        *,
        starvation_risk: bool,
    ) -> str:
        eff_lower = self.density_lower
        eff_upper = self.density_upper
        eff_hard = self.density_hard_upper
        if self._last_density_state == "above_upper":
            eff_upper += self.hysteresis_margin
            eff_hard += self.hysteresis_margin
        elif self._last_density_state == "below_lower":
            eff_lower -= self.hysteresis_margin
        if density_before >= eff_hard:
            state = "above_hard"
        elif density_before >= eff_upper:
            state = "above_upper"
        elif density_before < eff_lower or starvation_risk:
            state = "below_lower"
        elif density_before <= self.density_target + self.hysteresis_margin * 0.5:
            state = "in_band"
        else:
            state = "in_band"
        self._last_density_state = state
        return state

    def decide(
        self,
        *,
        frame_id: int,
        runtime_action: str,
        baseline_should_add: bool,
        is_test: bool,
        is_bootstrap_phase: bool,
        density_before: float,
        local_density_before: float,
        keyframe_growth_recent: int,
        baseline_relative_density: float,
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
        current_keyframe_count: int = 0,
        local_window_density: float = 0.0,
        local_window_keyframes: int = 0,
        local_window_gap_max: float = 0.0,
        local_window_gap_after_if_hold: float = 0.0,
        semantic_scores: dict[str, Any] | None = None,
        viewpoint_scores: dict[str, Any] | None = None,
    ) -> DirectFinalizationDecision:
        if self.is_pose_rep_value_decouple:
            return self._decide_pose_rep_value_v2(
                frame_id=frame_id,
                runtime_action=runtime_action,
                baseline_should_add=baseline_should_add,
                is_test=is_test,
                is_bootstrap_phase=is_bootstrap_phase,
                density_before=density_before,
                local_density_before=local_density_before,
                local_window_density=local_window_density,
                local_window_keyframes=local_window_keyframes,
                local_window_gap_max=local_window_gap_max,
                local_window_gap_after_if_hold=local_window_gap_after_if_hold,
                keyframe_growth_recent=keyframe_growth_recent,
                baseline_relative_density=baseline_relative_density,
                source_gap_to_last_keyframe=source_gap_to_last_keyframe,
                main_chain_gap_before=main_chain_gap_before,
                main_chain_gap_after_if_hold=main_chain_gap_after_if_hold,
                anchor_changed=anchor_changed,
                support_triggered=support_triggered,
                median_displacement=median_displacement,
                displacement_threshold=displacement_threshold,
                num_matches=num_matches,
                min_num_inliers=min_num_inliers,
                pose_inliers=pose_inliers,
                novelty_proxy=novelty_proxy,
                current_keyframe_count=current_keyframe_count,
                semantic_scores=semantic_scores,
                viewpoint_scores=viewpoint_scores,
            )
        if self.is_pose_rep_decouple or self.is_v221 or self.is_v222_family or self.mode == "target_band_v2_2":
            return self._decide_v22(
                frame_id=frame_id,
                runtime_action=runtime_action,
                baseline_should_add=baseline_should_add,
                is_test=is_test,
                is_bootstrap_phase=is_bootstrap_phase,
                density_before=density_before,
                local_density_before=local_density_before,
                local_window_density=local_window_density,
                local_window_keyframes=local_window_keyframes,
                local_window_gap_max=local_window_gap_max,
                local_window_gap_after_if_hold=local_window_gap_after_if_hold,
                keyframe_growth_recent=keyframe_growth_recent,
                baseline_relative_density=baseline_relative_density,
                source_gap_to_last_keyframe=source_gap_to_last_keyframe,
                main_chain_gap_before=main_chain_gap_before,
                main_chain_gap_after_if_hold=main_chain_gap_after_if_hold,
                anchor_changed=anchor_changed,
                support_triggered=support_triggered,
                median_displacement=median_displacement,
                displacement_threshold=displacement_threshold,
                num_matches=num_matches,
                min_num_inliers=min_num_inliers,
                pose_inliers=pose_inliers,
                novelty_proxy=novelty_proxy,
                current_keyframe_count=current_keyframe_count,
            )
        if self.is_v21:
            return self._decide_v21(
                frame_id=frame_id,
                runtime_action=runtime_action,
                baseline_should_add=baseline_should_add,
                is_test=is_test,
                is_bootstrap_phase=is_bootstrap_phase,
                density_before=density_before,
                local_density_before=local_density_before,
                keyframe_growth_recent=keyframe_growth_recent,
                baseline_relative_density=baseline_relative_density,
                source_gap_to_last_keyframe=source_gap_to_last_keyframe,
                main_chain_gap_before=main_chain_gap_before,
                main_chain_gap_after_if_hold=main_chain_gap_after_if_hold,
                anchor_changed=anchor_changed,
                support_triggered=support_triggered,
                median_displacement=median_displacement,
                displacement_threshold=displacement_threshold,
                num_matches=num_matches,
                min_num_inliers=min_num_inliers,
                pose_inliers=pose_inliers,
                novelty_proxy=novelty_proxy,
                current_keyframe_count=current_keyframe_count,
            )
        if self.mode == "target_band_v2":
            return self._decide_v2(
                frame_id=frame_id,
                runtime_action=runtime_action,
                baseline_should_add=baseline_should_add,
                is_test=is_test,
                is_bootstrap_phase=is_bootstrap_phase,
                density_before=density_before,
                local_density_before=local_density_before,
                keyframe_growth_recent=keyframe_growth_recent,
                baseline_relative_density=baseline_relative_density,
                source_gap_to_last_keyframe=source_gap_to_last_keyframe,
                main_chain_gap_before=main_chain_gap_before,
                main_chain_gap_after_if_hold=main_chain_gap_after_if_hold,
                anchor_changed=anchor_changed,
                support_triggered=support_triggered,
                median_displacement=median_displacement,
                displacement_threshold=displacement_threshold,
                num_matches=num_matches,
                min_num_inliers=min_num_inliers,
                pose_inliers=pose_inliers,
                novelty_proxy=novelty_proxy,
                current_keyframe_count=current_keyframe_count,
            )
        return self._decide_v1(
            frame_id=frame_id,
            runtime_action=runtime_action,
            is_test=is_test,
            is_bootstrap_phase=is_bootstrap_phase,
            density_before=density_before,
            source_gap_to_last_keyframe=source_gap_to_last_keyframe,
            main_chain_gap_before=main_chain_gap_before,
            main_chain_gap_after_if_hold=main_chain_gap_after_if_hold,
            anchor_changed=anchor_changed,
            support_triggered=support_triggered,
            median_displacement=median_displacement,
            displacement_threshold=displacement_threshold,
            num_matches=num_matches,
            min_num_inliers=min_num_inliers,
            pose_inliers=pose_inliers,
            novelty_proxy=novelty_proxy,
        )

    def _decide_pose_rep_value_v2(
        self,
        *,
        frame_id: int,
        runtime_action: str,
        baseline_should_add: bool,
        is_test: bool,
        is_bootstrap_phase: bool,
        density_before: float,
        local_density_before: float,
        local_window_density: float,
        local_window_keyframes: int,
        local_window_gap_max: float,
        local_window_gap_after_if_hold: float,
        keyframe_growth_recent: int,
        baseline_relative_density: float,
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
        current_keyframe_count: int,
        semantic_scores: dict[str, Any] | None,
        viewpoint_scores: dict[str, Any] | None = None,
    ) -> DirectFinalizationDecision:
        self._sync_budget_window(frame_id)
        min_growth_window = max(1, int(self.min_growth_per_100 * 0.5))
        expected_min_keyframes = float(frame_id) * self.density_lower / 100.0
        keyframe_debt = max(0.0, expected_min_keyframes - float(current_keyframe_count))
        baseline_expected_kf = float(frame_id) * self.baseline_density_per_100 / 100.0
        baseline_min_kf = self.baseline_relative_lower_ratio * baseline_expected_kf

        scores = dict(semantic_scores or {})
        viewpoint = dict(viewpoint_scores or {})
        viewpoint_rotation_window_max = _f(
            viewpoint.get("viewpoint_rotation_deg_window_max"), 0.0
        )
        viewpoint_grid_coverage = _clamp01(
            viewpoint.get("inlier_grid_coverage"), 0.0
        )
        inlier_grid_entropy = _clamp01(
            viewpoint.get("inlier_grid_entropy"), 0.0
        )
        support_concentration = _clamp01(
            viewpoint.get("support_concentration"), 0.0
        )
        anchor_health_score = _clamp01(
            viewpoint.get("anchor_health_score"), 0.0
        )
        new_view_event_score = _clamp01(
            viewpoint.get("new_view_event_score"), 0.0
        )
        semantic_R = _clamp01(scores.get("R_t"), 0.5)
        semantic_V = _clamp01(scores.get("V_t"), 0.0)
        semantic_Q = _clamp01(scores.get("Q_t"), 0.5)
        semantic_C = _clamp01(scores.get("C_t"), 0.0)
        semantic_BR = _clamp01(scores.get("B_R_t"), 0.0)
        risk_score = semantic_R
        disp_ratio = _f(median_displacement, 0.0) / max(_f(displacement_threshold, 0.0), 1e-6)
        motion_value = max(0.0, min(1.0, disp_ratio / 2.0))
        match_support = max(0.0, min(1.0, _f(num_matches, 0.0) / max(2.0 * max(min_num_inliers, 1), 1.0)))
        pose_support = max(0.0, min(1.0, _f(pose_inliers, 0.0) / max(2.0 * max(min_num_inliers, 1), 1.0)))
        novelty_value = _clamp01(novelty_proxy, 0.0)
        source_redundancy = 1.0 if source_gap_to_last_keyframe <= self.redundant_source_gap else 0.0
        redundancy_penalty = _clamp01(
            0.50 * source_redundancy * pose_support
            + 0.20 * source_redundancy * match_support
        )
        representation_value = _clamp01(
            0.18 * semantic_V
            + 0.10 * semantic_C
            + 0.27 * motion_value
            + 0.20 * novelty_value
            + (0.08 if support_triggered else 0.0)
            - redundancy_penalty
        )
        pose_reference_value = _clamp01(
            0.30 * (1.0 - risk_score)
            + 0.22 * semantic_Q
            + 0.18 * semantic_BR
            + 0.18 * pose_support
            + 0.12 * match_support
        )

        growth_ok = keyframe_growth_recent >= min_growth_window
        baseline_relative_kf_ok = current_keyframe_count >= int(baseline_min_kf)
        local_under_dense = bool(
            local_window_density < self.local_density_lower
            and density_before < self.density_upper
        )
        growth_starved = bool(not growth_ok and density_before < self.density_target)
        starvation_risk = bool(
            density_before < self.density_lower
            or keyframe_debt > 0.5
            or growth_starved
            or (not baseline_relative_kf_ok and density_before < self.density_target)
            or local_under_dense
        )
        density_state = self._density_state(density_before, starvation_risk=starvation_risk)
        hard_lim = self.hard_gap_threshold
        gap_critical = bool(
            main_chain_gap_after_if_hold > hard_lim
            or local_window_gap_after_if_hold > hard_lim
            or local_window_gap_max > hard_lim
            or source_gap_to_last_keyframe > hard_lim
        )
        gap_safe = bool(
            main_chain_gap_after_if_hold <= hard_lim
            and local_window_gap_after_if_hold <= hard_lim
            and source_gap_to_last_keyframe <= hard_lim
        )
        pose_risk_high = bool(
            risk_score >= 0.55
            or semantic_BR < 0.25
            or semantic_Q < 0.38
            or pose_inliers < max(min_num_inliers, 1)
        )
        recovery_pool_size = max(0, int(_f(scores.get("recovery_pool_size"), 0.0)))
        utility_recovery_pressure_score = _clamp01((float(recovery_pool_size) - 3.0) / 16.0)
        utility_gap_pressure = _clamp01(
            max(
                _f(main_chain_gap_after_if_hold, 0.0),
                _f(local_window_gap_after_if_hold, 0.0),
                _f(source_gap_to_last_keyframe, 0.0),
            )
            / max(float(self.hard_gap_threshold), 1.0)
        )
        utility_view_change = _clamp01(
            max(
                new_view_event_score,
                viewpoint_rotation_window_max / 45.0,
                1.0 - viewpoint_grid_coverage,
            )
        )
        utility_density_pressure = _clamp01(
            0.55 * (_f(density_before, 0.0) / max(self.density_hard_upper, 1.0))
            + 0.30 * (_f(local_window_density, 0.0) / max(self.density_hard_upper, 1.0))
            + 0.15 * (_f(keyframe_growth_recent, 0.0) / 60.0)
        )
        utility_compute_cost = _clamp01(
            0.70 * utility_density_pressure
            + 0.10 * utility_recovery_pressure_score
            + 0.10 * source_redundancy
            + 0.10 * (_f(current_keyframe_count, 0.0) / 1200.0)
        )
        utility_drift_risk = _clamp01(
            0.34 * risk_score
            + 0.20 * (1.0 - semantic_Q)
            + 0.17 * (1.0 - semantic_BR)
            + 0.13 * (1.0 - pose_support)
            + 0.08 * (1.0 - match_support)
            + 0.08 * support_concentration
        )
        utility_coverage_gain = _clamp01(
            0.24 * semantic_V
            + 0.18 * semantic_C
            + 0.18 * novelty_value
            + 0.14 * motion_value
            + 0.16 * utility_view_change
            + 0.10 * (1.0 - source_redundancy)
        )
        utility_recovery_gain = _clamp01(
            0.30 * utility_recovery_pressure_score
            + 0.20 * (1.0 - semantic_C)
            + 0.16 * novelty_value
            + 0.14 * motion_value
            + 0.10 * utility_gap_pressure
            + 0.10 * (1.0 - anchor_health_score)
        )
        utility_pose_reference = _clamp01(
            pose_reference_value
            + 0.08 * (1.0 - utility_drift_risk)
            + 0.04 * match_support
            - 0.06 * utility_compute_cost
        )
        utility_representation = _clamp01(
            0.56 * utility_coverage_gain
            + 0.24 * utility_recovery_gain
            + 0.12 * (1.0 - source_redundancy)
            - 0.20 * utility_compute_cost
            - 0.16 * redundancy_penalty
        )
        utility_total = _clamp01(
            0.44 * utility_pose_reference
            + 0.44 * utility_representation
            + 0.12 * utility_recovery_gain
            - 0.12 * utility_drift_risk
            - 0.10 * utility_compute_cost
        )
        utility_recovery_pressure_context = bool(
            self.is_pose_rep_active_memory_v27
            and recovery_pool_size >= 6
            and utility_recovery_gain >= 0.35
            and utility_coverage_gain >= 0.45
        )
        utility_representation_role = bool(
            self.is_pose_rep_active_memory_v27
            and (
                utility_representation >= self.utility_representation_min
                or utility_recovery_pressure_context
                or (anchor_changed and utility_coverage_gain >= 0.35)
                or (
                    source_gap_to_last_keyframe > self.redundant_source_gap
                    and utility_coverage_gain >= 0.45
                    and utility_drift_risk < 0.55
                )
            )
        )
        utility_tracking_only_role = bool(
            self.is_pose_rep_active_memory_v27
            and utility_pose_reference >= self.utility_pose_reference_min
            and utility_representation < self.utility_representation_min
            and utility_drift_risk < 0.55
        )
        high_recent_growth_representation_guard = bool(
            self.is_pose_rep_value_v3
            and keyframe_growth_recent >= max(24, int(self.min_growth_per_100 * 2.5))
        )
        low_semantic_coverage_representation_guard = bool(
            self.is_pose_rep_value_v3
            and (semantic_C < 0.82 or semantic_Q < 0.78)
            and motion_value >= 0.55
        )
        anchor_boundary_representation_guard = bool(
            self.is_pose_rep_value_v3 and anchor_changed
        )
        long_sequence_maturity_guard = bool(
            self.is_pose_rep_value_v3 and int(frame_id) < 300
        )
        long_stream_low_growth_context = bool(
            self.is_pose_rep_value_v3
            and not long_sequence_maturity_guard
            and keyframe_growth_recent <= max(12, min_growth_window)
            and density_before >= self.density_upper
            and local_density_before >= 60.0
            and source_gap_to_last_keyframe <= self.redundant_source_gap
            and semantic_Q >= 0.85
            and semantic_BR >= 0.8
            and not low_semantic_coverage_representation_guard
        )
        active_memory_redundancy_pressure = _clamp01(
            0.28 * source_redundancy
            + 0.20 * pose_support
            + 0.16 * match_support
            + 0.18 * _clamp01((density_before - 45.0) / 35.0)
            + 0.18 * _clamp01((local_density_before - 35.0) / 45.0)
        )
        active_memory_low_parallax = bool(disp_ratio <= 0.85 and motion_value <= 0.45)
        active_memory_redundancy_pressure_high = bool(
            active_memory_redundancy_pressure >= 0.70
        )
        active_memory_stable_pose_reference = bool(
            risk_score <= 0.25
            and semantic_Q >= 0.85
            and semantic_C >= 0.85
            and semantic_BR >= 0.80
            and pose_support >= 0.75
            and match_support >= 0.75
        )
        active_memory_low_representation_value = bool(
            representation_value < self.representation_value_hold_max
        )
        active_memory_low_marginal_representation = bool(
            active_memory_low_representation_value
            and active_memory_redundancy_pressure_high
        )
        value_hold_budget_available = bool(
            self._budget.value_hold_used < self.value_hold_budget_per_100
        )
        active_memory_low_turn_dense_context = bool(
            (
                self.is_pose_rep_active_memory_v2
                or self.is_pose_rep_active_memory_v4
                or self.is_pose_rep_active_memory_v5
                or self.is_pose_rep_active_memory_v6
                or self.is_pose_rep_active_memory_v8
                or self.is_pose_rep_active_memory_v9
                or self.is_pose_rep_active_memory_v24
                or self.is_pose_rep_active_memory_v25
                or self.is_pose_rep_active_memory_v26
                or self.is_pose_rep_active_memory_v27
            )
            and int(frame_id) >= 300
            and density_before >= 55.0
            and local_density_before >= 35.0
            and viewpoint_rotation_window_max <= 8.0
            and viewpoint_grid_coverage >= 0.95
            and active_memory_stable_pose_reference
            and active_memory_low_marginal_representation
            and not anchor_changed
            and gap_safe
            and not starvation_risk
            and density_state != "below_lower"
        )
        active_memory_context_candidate = bool(
            self.is_pose_rep_active_memory
            and int(frame_id) >= 300
            and density_before >= 55.0
            and (local_density_before >= 45.0 or active_memory_low_turn_dense_context)
            and active_memory_stable_pose_reference
            and active_memory_low_marginal_representation
            and not anchor_changed
            and gap_safe
            and not starvation_risk
            and density_state != "below_lower"
        )
        active_memory_context = bool(
            active_memory_context_candidate and value_hold_budget_available
        )
        active_memory_marginal_value = representation_value
        active_memory_frame_role = (
            "tracking_only"
            if active_memory_context or utility_tracking_only_role
            else "representation"
        )
        representation_value_high = bool(
            representation_value >= self.representation_value_hold_max
            or utility_representation_role
            or (high_recent_growth_representation_guard and not active_memory_context)
            or low_semantic_coverage_representation_guard
            or anchor_boundary_representation_guard
            or (
                source_gap_to_last_keyframe > self.redundant_source_gap
                and semantic_V >= 0.65
                and semantic_C >= 0.55
                and novelty_value >= 0.55
            )
            or (
                source_gap_to_last_keyframe > self.redundant_source_gap
                and disp_ratio >= 1.75
            )
        )
        bootstrap_value_hold_guard = int(frame_id) <= 100
        if self.is_pose_rep_active_memory_v27:
            value_hold_allowed = bool(
                utility_tracking_only_role
                and not utility_representation_role
                and not representation_value_high
                and not pose_risk_high
                and gap_safe
                and not starvation_risk
                and density_state != "below_lower"
                and value_hold_budget_available
                and not bootstrap_value_hold_guard
            )
        else:
            value_hold_allowed = bool(
                not representation_value_high
                and pose_reference_value >= self.pose_reference_value_min
                and not pose_risk_high
                and gap_safe
                and not starvation_risk
                and density_state != "below_lower"
                and value_hold_budget_available
                and not bootstrap_value_hold_guard
                and (
                    not self.is_pose_rep_value_v3
                    or long_stream_low_growth_context
                    or active_memory_context
                )
            )
        block_reason = ""
        if self.is_pose_rep_active_memory_v27 and value_hold_allowed:
            block_reason = "utility_tracking_only_role"
        elif self.is_pose_rep_active_memory_v27 and utility_representation_role:
            block_reason = "utility_representation_gain"
        elif self.is_pose_rep_active_memory_v27 and not utility_tracking_only_role:
            block_reason = "utility_tracking_role_low"
        elif high_recent_growth_representation_guard and not active_memory_context:
            block_reason = "high_recent_growth_representation_guard"
        elif low_semantic_coverage_representation_guard:
            block_reason = "low_semantic_coverage_representation_guard"
        elif anchor_boundary_representation_guard:
            block_reason = "anchor_boundary_representation_guard"
        elif representation_value_high:
            block_reason = "representation_value_high"
        elif pose_risk_high:
            block_reason = "pose_risk_high"
        elif not gap_safe:
            block_reason = "gap_not_safe"
        elif starvation_risk or density_state == "below_lower":
            block_reason = "keyframe_density_debt"
        elif pose_reference_value < self.pose_reference_value_min:
            block_reason = "pose_reference_value_low"
        elif not value_hold_budget_available:
            block_reason = "value_hold_budget_exhausted"
        elif bootstrap_value_hold_guard:
            block_reason = "early_bootstrap_value_hold_guard"
        elif long_sequence_maturity_guard:
            block_reason = "long_sequence_maturity_guard"
        elif active_memory_context:
            block_reason = "active_memory_tracking_only_context"

        dbg: dict[str, Any] = {
            "mode": self.mode,
            "frame_id": frame_id,
            "density_before": density_before,
            "density_after": density_before,
            "local_density_before": local_density_before,
            "local_window_density": local_window_density,
            "local_window_keyframes": local_window_keyframes,
            "local_window_gap_max": local_window_gap_max,
            "local_window_gap_after_if_hold": local_window_gap_after_if_hold,
            "baseline_relative_density": baseline_relative_density,
            "keyframe_count_current": current_keyframe_count,
            "expected_min_keyframes": round(expected_min_keyframes, 2),
            "keyframe_debt": round(keyframe_debt, 2),
            "recent_keyframe_growth": keyframe_growth_recent,
            "keyframe_growth_recent": keyframe_growth_recent,
            "source_gap_to_last_keyframe": source_gap_to_last_keyframe,
            "main_chain_gap_before": main_chain_gap_before,
            "main_chain_gap_after_if_hold": main_chain_gap_after_if_hold,
            "anchor_changed": anchor_changed,
            "support_triggered": support_triggered,
            "density_state": density_state,
            "direct_admit_candidate": runtime_action in {"direct_admit", "current_frame_surrogate_commit"},
            "baseline_direct_admit_candidate": bool(baseline_should_add and runtime_action == "direct_admit"),
            "starvation_risk": starvation_risk,
            "local_density_rescue_triggered": local_under_dense,
            "gap_critical_triggered": gap_critical,
            "gap_critical_finalized": False,
            "semantic_R_t": semantic_R,
            "semantic_V_t": semantic_V,
            "semantic_Q_t": semantic_Q,
            "semantic_C_t": semantic_C,
            "semantic_B_R_t": semantic_BR,
            "pose_risk_score": risk_score,
            "motion_value_score": motion_value,
            "match_support_score": match_support,
            "pose_support_score": pose_support,
            "active_memory_context": active_memory_context,
            "active_memory_context_candidate": active_memory_context_candidate,
            "active_memory_low_turn_dense_context": active_memory_low_turn_dense_context,
            "viewpoint_rotation_window_max": viewpoint_rotation_window_max,
            "viewpoint_grid_coverage": viewpoint_grid_coverage,
            "inlier_grid_entropy": inlier_grid_entropy,
            "support_concentration": support_concentration,
            "anchor_health_score": anchor_health_score,
            "new_view_event_score": new_view_event_score,
            "active_memory_frame_role": active_memory_frame_role,
            "active_memory_marginal_value": active_memory_marginal_value,
            "active_memory_redundancy_pressure": active_memory_redundancy_pressure,
            "active_memory_redundancy_pressure_high": active_memory_redundancy_pressure_high,
            "active_memory_low_parallax": active_memory_low_parallax,
            "active_memory_stable_pose_reference": active_memory_stable_pose_reference,
            "active_memory_low_representation_value": active_memory_low_representation_value,
            "active_memory_low_marginal_representation": active_memory_low_marginal_representation,
            "source_redundancy_score": source_redundancy,
            "representation_redundancy_penalty": redundancy_penalty,
            "representation_value_score": representation_value,
            "pose_reference_value_score": pose_reference_value,
            "recovery_pool_size": recovery_pool_size,
            "utility_pose_reference": utility_pose_reference,
            "utility_representation": utility_representation,
            "utility_coverage_gain": utility_coverage_gain,
            "utility_recovery_gain": utility_recovery_gain,
            "utility_recovery_pressure_score": utility_recovery_pressure_score,
            "utility_compute_cost": utility_compute_cost,
            "utility_drift_risk": utility_drift_risk,
            "utility_view_change": utility_view_change,
            "utility_gap_pressure": utility_gap_pressure,
            "utility_total": utility_total,
            "utility_frame_role": active_memory_frame_role,
            "utility_tracking_only_role": utility_tracking_only_role,
            "utility_representation_role": utility_representation_role,
            "utility_recovery_pressure_context": utility_recovery_pressure_context,
            "representation_value_hold_max": self.representation_value_hold_max,
            "pose_reference_value_min": self.pose_reference_value_min,
            "value_hold_budget_per_100": self.value_hold_budget_per_100,
            "value_hold_budget_used": self._budget.value_hold_used,
            "value_hold_budget_available": value_hold_budget_available,
            "bootstrap_value_hold_guard": bootstrap_value_hold_guard,
            "value_hold_allowed": value_hold_allowed,
            "value_hold_block_reason": block_reason,
            "high_novelty_score": novelty_value if representation_value_high else 0.0,
            "support_needed_score": min(1.0, num_matches / max(min_num_inliers, 1)) if support_triggered else 0.0,
            "novelty_value_score": novelty_value,
            "hold_low_representation_value": False,
            "hold_density_high": False,
            "hold_redundant": False,
            "hold_redundant_allowed": False,
            "hold_density_high_allowed": False,
            "hold_density_high_blocked_by_gap": False,
            "hold_redundant_blocked_by_gap": False,
            "hold_redundant_blocked_by_lower_guard": False,
            "hold_density_high_blocked_by_starvation": False,
            "hold_density_high_blocked_by_local_under_density": False,
            "finalize_high_novelty": False,
            "finalize_support_needed": False,
            "finalize_gap_critical": False,
            "finalize_high_representation_value": False,
            "finalize_pose_risk_reference": False,
            "representation_value_high": representation_value_high,
            "pose_risk_high": pose_risk_high,
            "high_recent_growth_representation_guard": high_recent_growth_representation_guard,
            "low_semantic_coverage_representation_guard": low_semantic_coverage_representation_guard,
            "anchor_boundary_representation_guard": anchor_boundary_representation_guard,
            "long_sequence_maturity_guard": long_sequence_maturity_guard,
            "long_stream_low_growth_context": long_stream_low_growth_context,
            "density_only_hold_disabled": self.is_pose_rep_value_v3,
        }
        direct_candidate = dbg["direct_admit_candidate"]
        if not self.enabled or not direct_candidate:
            dbg["direct_keyframe_finalized"] = True
            return DirectFinalizationDecision(True, "finalize", "control_off_or_not_direct", dbg)
        if is_test or is_bootstrap_phase:
            dbg["direct_keyframe_finalized"] = True
            return DirectFinalizationDecision(True, "finalize", "test_or_bootstrap_bypass", dbg)

        def _finalize(decision: str, reason: str) -> DirectFinalizationDecision:
            dbg["direct_keyframe_finalized"] = True
            dbg["keyframe_finalized"] = True
            if decision == "finalize_gap_critical_v2":
                dbg["gap_critical_finalized"] = True
                dbg["finalize_gap_critical"] = True
            if decision == "finalize_support_needed":
                dbg["finalize_support_needed"] = True
            if decision == "finalize_high_representation_value":
                dbg["finalize_high_representation_value"] = True
                dbg["finalize_high_novelty"] = bool(novelty_value >= 0.65 or disp_ratio >= 1.25)
            if decision == "finalize_pose_risk_reference":
                dbg["finalize_pose_risk_reference"] = True
            if decision == "finalize_local_density_rescue":
                dbg["finalize_local_density_rescue"] = True
            return DirectFinalizationDecision(True, decision, reason, dbg)

        if gap_critical:
            return _finalize("finalize_gap_critical_v2", "value_hold_blocked_by_gap")
        if local_under_dense:
            dbg["hold_density_high_blocked_by_local_under_density"] = True
            return _finalize("finalize_local_density_rescue", "local_under_density_preempts_value_hold")
        if starvation_risk or density_state == "below_lower":
            dbg["hold_redundant_blocked_by_lower_guard"] = True
            return _finalize("finalize_growth_rescue", "keyframe_density_debt_preempts_value_hold")
        if pose_risk_high:
            return _finalize("finalize_pose_risk_reference", "pose_risk_high")
        if representation_value_high:
            reason = (
                "utility_representation_gain"
                if self.is_pose_rep_active_memory_v27 and utility_representation_role
                else "representation_value_high"
            )
            return _finalize("finalize_high_representation_value", reason)
        if anchor_changed and representation_value >= 0.34:
            return _finalize("finalize_anchor_boundary", "anchor_boundary_representation_context")

        if value_hold_allowed:
            self._budget.value_hold_used += 1
            dbg["value_hold_budget_used"] = self._budget.value_hold_used
            dbg["hold_low_representation_value"] = True
            dbg["direct_keyframe_finalized"] = False
            dbg["keyframe_finalized"] = False
            return DirectFinalizationDecision(
                False,
                "hold_low_representation_value",
                "pose_reference_only_low_representation_value",
                dbg,
            )

        if (
            not self.is_pose_rep_value_v3
            and density_before >= self.density_hard_upper
            and gap_safe
        ):
            dbg["hold_density_high"] = True
            dbg["hold_density_high_allowed"] = True
            dbg["direct_keyframe_finalized"] = False
            dbg["keyframe_finalized"] = False
            return DirectFinalizationDecision(False, "hold_density_high", "dense_band_pose_reference_only", dbg)

        return _finalize("finalize", block_reason or "default_value_aware_finalize")

    def _decide_v2(
        self,
        *,
        frame_id: int,
        runtime_action: str,
        baseline_should_add: bool,
        is_test: bool,
        is_bootstrap_phase: bool,
        density_before: float,
        local_density_before: float,
        keyframe_growth_recent: int,
        baseline_relative_density: float,
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
        current_keyframe_count: int,
    ) -> DirectFinalizationDecision:
        self._sync_budget_window(frame_id)
        dbg: dict[str, Any] = {
            "mode": self.mode,
            "frame_id": frame_id,
            "density_before": density_before,
            "local_density_before": local_density_before,
            "baseline_relative_density": baseline_relative_density,
            "keyframe_growth_recent": keyframe_growth_recent,
            "source_gap_to_last_keyframe": source_gap_to_last_keyframe,
            "main_chain_gap_before": main_chain_gap_before,
            "main_chain_gap_after_if_hold": main_chain_gap_after_if_hold,
            "anchor_changed": anchor_changed,
            "support_triggered": support_triggered,
            "novelty_proxy": novelty_proxy,
            "high_novelty_budget_used": self._budget.high_novelty_used,
            "support_needed_budget_used": self._budget.support_needed_used,
            "high_novelty_budget_exhausted": False,
            "support_needed_budget_exhausted": False,
            "hold_redundant_allowed": True,
            "hold_redundant_blocked_by_lower_guard": False,
            "blocked_override_reason": "",
            "direct_admit_candidate": runtime_action in {"direct_admit", "current_frame_surrogate_commit"},
        }
        direct_candidate = dbg["direct_admit_candidate"]

        if not self.enabled or not direct_candidate:
            return DirectFinalizationDecision(True, "finalize", "control_off_or_not_direct", dbg)
        if is_test or is_bootstrap_phase:
            dbg["direct_keyframe_finalized"] = True
            return DirectFinalizationDecision(True, "finalize", "test_or_bootstrap_bypass", dbg)

        expected_kf = max(1, int(frame_id * self.baseline_density_per_100 / 100.0))
        baseline_relative_kf_ok = current_keyframe_count >= int(
            self.baseline_relative_lower_ratio * expected_kf
        )
        growth_floor_ok = keyframe_growth_recent >= int(self.min_growth_per_100 * 0.5) or density_before >= self.min_growth_per_100
        starvation_risk = bool(
            density_before < self.density_lower
            or not baseline_relative_kf_ok
            or (density_before < self.baseline_relative_lower_ratio * self.baseline_density_per_100)
        )
        dbg["starvation_risk"] = starvation_risk
        dbg["baseline_relative_kf_ok"] = baseline_relative_kf_ok

        density_state = self._density_state(density_before, starvation_risk=starvation_risk)
        dbg["density_state"] = density_state

        gap_critical = bool(
            main_chain_gap_after_if_hold >= self.gap_hard_limit
            or main_chain_gap_before >= self.gap_hard_limit - 2
        )
        displacement_novelty = bool(
            median_displacement >= 1.35 * max(displacement_threshold, 1e-6)
        )
        extreme_novelty = bool(
            displacement_novelty and novelty_proxy >= 0.80
        )
        high_novelty_score = float(novelty_proxy if displacement_novelty else 0.0)
        support_needed_score = float(
            min(1.0, num_matches / max(min_num_inliers, 1)) if support_triggered else 0.0
        )
        dbg["high_novelty_score"] = high_novelty_score
        dbg["support_needed_score"] = support_needed_score
        dbg["finalize_gap_critical"] = gap_critical

        want_support = bool(
            support_triggered
            and num_matches >= max(min_num_inliers, 1)
            and (density_before < self.density_target or starvation_risk)
        )
        want_high_novelty = bool(displacement_novelty and not support_triggered)

        def _hold(decision: str, reason: str, extra: dict[str, Any] | None = None) -> DirectFinalizationDecision:
            if extra:
                dbg.update(extra)
            dbg["hold_density_high"] = decision == "hold_density_high"
            dbg["hold_redundant"] = decision in {"hold_redundant", "direct_admit_redundant_hold"}
            dbg["direct_keyframe_finalized"] = False
            return DirectFinalizationDecision(False, decision, reason, dbg)

        def _fin(decision: str, reason: str, *, use_high_budget: bool = False, use_support_budget: bool = False) -> DirectFinalizationDecision:
            if use_high_budget:
                self._budget.high_novelty_used += 1
                dbg["high_novelty_budget_used"] = self._budget.high_novelty_used
                dbg["finalize_high_novelty"] = True
            if use_support_budget:
                self._budget.support_needed_used += 1
                dbg["support_needed_budget_used"] = self._budget.support_needed_used
                dbg["finalize_support_needed"] = True
            dbg["direct_keyframe_finalized"] = True
            dbg["keyframe_finalized"] = True
            return DirectFinalizationDecision(True, decision, reason, dbg)

        if gap_critical:
            return _fin("finalize_gap_critical", "gap_critical_override")

        if anchor_changed:
            dbg["finalize_anchor_boundary"] = True
            return _fin("finalize_anchor_boundary", "anchor_boundary")

        if starvation_risk or density_state == "below_lower":
            dbg["hold_redundant_blocked_by_lower_guard"] = True
            dbg["hold_redundant_allowed"] = False
            if want_support:
                if self._budget.support_needed_used < self.support_needed_budget_per_100:
                    return _fin("finalize_support_needed", "growth_rescue_support_needed", use_support_budget=True)
            if want_high_novelty or source_gap_to_last_keyframe > self.redundant_source_gap:
                if self._budget.high_novelty_used < self.high_novelty_budget_per_100:
                    return _fin("finalize_high_novelty", "growth_rescue_high_novelty", use_high_budget=True)
            return _fin("finalize_growth_rescue", "density_below_lower_growth_rescue")

        if density_state == "above_hard":
            if gap_critical or extreme_novelty:
                return _fin(
                    "finalize_gap_critical" if gap_critical else "finalize_high_novelty",
                    "above_hard_extreme_only",
                    use_high_budget=not gap_critical,
                )
            if want_support and self._budget.support_needed_used < self.support_needed_budget_per_100:
                return _fin("finalize_support_needed", "above_hard_support_rescue", use_support_budget=True)
            return _hold("hold_density_high", "above_hard_upper_block", {"blocked_override_reason": "above_hard_band"})

        if density_state == "above_upper":
            if want_high_novelty:
                if self._budget.high_novelty_used >= self.high_novelty_budget_per_100:
                    dbg["high_novelty_budget_exhausted"] = True
                    dbg["blocked_override_reason"] = "high_novelty_budget_exhausted"
                    return _hold("hold_density_high", "high_novelty_budget_exhausted")
                if extreme_novelty or displacement_novelty:
                    return _fin("finalize_high_novelty", "budgeted_high_novelty", use_high_budget=True)
            if want_support:
                if self._budget.support_needed_used >= self.support_needed_budget_per_100:
                    dbg["support_needed_budget_exhausted"] = True
                    return _hold("hold_density_high", "support_needed_budget_exhausted")
                return _fin("finalize_support_needed", "budgeted_support_needed", use_support_budget=True)
            return _hold("hold_density_high", "density_above_upper_band")

        redundant = bool(
            source_gap_to_last_keyframe <= self.redundant_source_gap
            and main_chain_gap_before <= 5.0
            and not anchor_changed
            and density_state == "in_band"
            and not gap_critical
            and not starvation_risk
        )
        if redundant and not displacement_novelty:
            dbg["hold_redundant_allowed"] = True
            return _hold("hold_redundant", "redundant_close_gap_in_band")

        if want_support and density_before <= self.density_target + 2.0:
            if self._budget.support_needed_used >= self.support_needed_budget_per_100:
                dbg["support_needed_budget_exhausted"] = True
            else:
                return _fin("finalize_support_needed", "in_band_support_needed", use_support_budget=True)

        if want_high_novelty:
            if self._budget.high_novelty_used >= self.high_novelty_budget_per_100:
                dbg["high_novelty_budget_exhausted"] = True
                if density_before > self.density_target:
                    return _hold("hold_density_high", "high_novelty_budget_exhausted")
            else:
                return _fin("finalize_high_novelty", "budgeted_high_novelty", use_high_budget=True)

        return _fin("finalize", "default_in_band_finalize")

    def _decide_v21(
        self,
        *,
        frame_id: int,
        runtime_action: str,
        baseline_should_add: bool,
        is_test: bool,
        is_bootstrap_phase: bool,
        density_before: float,
        local_density_before: float,
        keyframe_growth_recent: int,
        baseline_relative_density: float,
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
        current_keyframe_count: int,
    ) -> DirectFinalizationDecision:
        self._sync_budget_window(frame_id)
        min_growth_window = max(1, int(self.min_growth_per_100 * 0.5))
        expected_min_keyframes = float(frame_id) * self.density_lower / 100.0
        keyframe_debt = max(0.0, expected_min_keyframes - float(current_keyframe_count))
        baseline_expected_kf = float(frame_id) * self.baseline_density_per_100 / 100.0
        baseline_min_kf = self.baseline_relative_lower_ratio * baseline_expected_kf

        dbg: dict[str, Any] = {
            "mode": self.mode,
            "frame_id": frame_id,
            "density_before": density_before,
            "density_after": density_before,
            "local_density_before": local_density_before,
            "baseline_relative_density": baseline_relative_density,
            "keyframe_count_current": current_keyframe_count,
            "expected_min_keyframes": round(expected_min_keyframes, 2),
            "keyframe_debt": round(keyframe_debt, 2),
            "recent_keyframe_growth": keyframe_growth_recent,
            "keyframe_growth_recent": keyframe_growth_recent,
            "source_gap_to_last_keyframe": source_gap_to_last_keyframe,
            "main_chain_gap_before": main_chain_gap_before,
            "main_chain_gap_after_if_hold": main_chain_gap_after_if_hold,
            "anchor_changed": anchor_changed,
            "support_triggered": support_triggered,
            "novelty_proxy": novelty_proxy,
            "high_novelty_budget_used": self._budget.high_novelty_used,
            "support_needed_budget_used": self._budget.support_needed_used,
            "high_novelty_budget_exhausted": False,
            "support_needed_budget_exhausted": False,
            "hold_density_high_allowed": False,
            "hold_density_high_blocked_by_starvation": False,
            "hold_redundant_allowed": False,
            "hold_redundant_blocked_by_lower_guard": False,
            "blocked_override_reason": "",
            "direct_admit_candidate": runtime_action in {"direct_admit", "current_frame_surrogate_commit"},
            "starvation_preempts_hold": False,
        }
        direct_candidate = dbg["direct_admit_candidate"]

        if not self.enabled or not direct_candidate:
            return DirectFinalizationDecision(True, "finalize", "control_off_or_not_direct", dbg)
        if is_test or is_bootstrap_phase:
            dbg["direct_keyframe_finalized"] = True
            return DirectFinalizationDecision(True, "finalize", "test_or_bootstrap_bypass", dbg)

        baseline_relative_kf_ok = current_keyframe_count >= int(baseline_min_kf)
        growth_ok = keyframe_growth_recent >= min_growth_window
        starvation_risk = bool(
            density_before < self.density_lower
            or local_density_before < self.density_lower
            or keyframe_debt > 0.5
            or not growth_ok
            or not baseline_relative_kf_ok
            or density_before < self.baseline_relative_lower_ratio * self.baseline_density_per_100
        )
        starvation_preempts_hold = starvation_risk
        dbg["starvation_risk"] = starvation_risk
        dbg["starvation_preempts_hold"] = starvation_preempts_hold
        dbg["baseline_relative_kf_ok"] = baseline_relative_kf_ok

        density_state = self._density_state(density_before, starvation_risk=starvation_risk)
        dbg["density_state"] = density_state

        gap_critical = bool(
            main_chain_gap_after_if_hold >= self.gap_hard_limit
            or main_chain_gap_before >= self.gap_hard_limit - 2
        )
        gap_safe = main_chain_gap_before <= 5.0 and not gap_critical
        displacement_novelty = bool(
            median_displacement >= 1.35 * max(displacement_threshold, 1e-6)
        )
        extreme_novelty = bool(displacement_novelty and novelty_proxy >= 0.80)
        high_novelty_score = float(novelty_proxy if displacement_novelty else 0.0)
        support_needed_score = float(
            min(1.0, num_matches / max(min_num_inliers, 1)) if support_triggered else 0.0
        )
        dbg["high_novelty_score"] = high_novelty_score
        dbg["support_needed_score"] = support_needed_score
        dbg["finalize_gap_critical"] = gap_critical

        hold_density_high_allowed = bool(
            density_before > self.density_upper
            and local_density_before > self.density_upper
            and not starvation_risk
            and keyframe_debt <= 0.0
            and growth_ok
            and gap_safe
        )
        dbg["hold_density_high_allowed"] = hold_density_high_allowed
        if starvation_preempts_hold:
            dbg["hold_density_high_blocked_by_starvation"] = True

        want_support = bool(
            support_triggered
            and num_matches >= max(min_num_inliers, 1)
            and (density_before < self.density_target or starvation_risk)
        )
        want_high_novelty = bool(displacement_novelty and not support_triggered)
        budget_relaxed = bool(starvation_risk or keyframe_debt > 0)

        def _fin(
            decision: str,
            reason: str,
            *,
            use_high_budget: bool = False,
            use_support_budget: bool = False,
            exempt_budget: bool = False,
        ) -> DirectFinalizationDecision:
            if use_high_budget and not (exempt_budget or budget_relaxed):
                self._budget.high_novelty_used += 1
                dbg["high_novelty_budget_used"] = self._budget.high_novelty_used
                dbg["finalize_high_novelty"] = True
            elif decision == "finalize_high_novelty":
                dbg["finalize_high_novelty"] = True
            if use_support_budget and not (exempt_budget or budget_relaxed):
                self._budget.support_needed_used += 1
                dbg["support_needed_budget_used"] = self._budget.support_needed_used
                dbg["finalize_support_needed"] = True
            elif decision in {"finalize_support_needed", "finalize_growth_rescue"}:
                if decision == "finalize_growth_rescue":
                    dbg["finalize_growth_rescue"] = True
                else:
                    dbg["finalize_support_needed"] = True
            dbg["direct_keyframe_finalized"] = True
            dbg["keyframe_finalized"] = True
            return DirectFinalizationDecision(True, decision, reason, dbg)

        def _hold(decision: str, reason: str, extra: dict[str, Any] | None = None) -> DirectFinalizationDecision:
            if starvation_preempts_hold and decision in {"hold_density_high", "hold_redundant"}:
                dbg["hold_density_high_blocked_by_starvation"] = decision == "hold_density_high"
                if decision == "hold_redundant":
                    dbg["hold_redundant_blocked_by_lower_guard"] = True
                return _fin("finalize_growth_rescue", "starvation_preempts_hold", exempt_budget=True)
            if extra:
                dbg.update(extra)
            dbg["hold_density_high"] = decision == "hold_density_high"
            dbg["hold_redundant"] = decision in {"hold_redundant", "direct_admit_redundant_hold"}
            dbg["direct_keyframe_finalized"] = False
            return DirectFinalizationDecision(False, decision, reason, dbg)

        if gap_critical:
            return _fin("finalize_gap_critical", "gap_critical_override", exempt_budget=True)

        if anchor_changed:
            return _fin("finalize_anchor_boundary", "anchor_boundary", exempt_budget=True)

        if starvation_risk or keyframe_debt > 0 or density_state == "below_lower":
            dbg["hold_redundant_blocked_by_lower_guard"] = True
            dbg["hold_redundant_allowed"] = False
            if want_support:
                return _fin(
                    "finalize_support_needed",
                    "growth_rescue_support_needed",
                    use_support_budget=not budget_relaxed,
                    exempt_budget=budget_relaxed,
                )
            if want_high_novelty and (budget_relaxed or self._budget.high_novelty_used < self.high_novelty_budget_per_100):
                return _fin(
                    "finalize_high_novelty",
                    "growth_rescue_high_novelty",
                    use_high_budget=not budget_relaxed,
                    exempt_budget=budget_relaxed,
                )
            return _fin("finalize_growth_rescue", "keyframe_debt_or_below_lower", exempt_budget=True)

        if density_state == "above_hard":
            if extreme_novelty or gap_critical:
                return _fin("finalize_high_novelty", "above_hard_extreme_only", use_high_budget=True)
            if want_support and self._budget.support_needed_used < self.support_needed_budget_per_100:
                return _fin("finalize_support_needed", "above_hard_support_rescue", use_support_budget=True)
            if hold_density_high_allowed:
                return _hold("hold_density_high", "above_hard_upper_block")
            return _fin("finalize", "above_hard_no_hold_finalize")

        if density_state == "above_upper":
            if want_high_novelty:
                if (
                    not budget_relaxed
                    and self._budget.high_novelty_used >= self.high_novelty_budget_per_100
                ):
                    dbg["high_novelty_budget_exhausted"] = True
                    if hold_density_high_allowed:
                        return _hold("hold_density_high", "high_novelty_budget_exhausted")
                    return _fin("finalize", "budget_exhausted_finalize_default")
                if extreme_novelty or displacement_novelty:
                    return _fin("finalize_high_novelty", "budgeted_high_novelty", use_high_budget=True)
            if want_support:
                if (
                    not budget_relaxed
                    and self._budget.support_needed_used >= self.support_needed_budget_per_100
                ):
                    dbg["support_needed_budget_exhausted"] = True
                    if hold_density_high_allowed:
                        return _hold("hold_density_high", "support_needed_budget_exhausted")
                    return _fin("finalize", "support_budget_exhausted_finalize")
                return _fin("finalize_support_needed", "budgeted_support_needed", use_support_budget=True)
            if hold_density_high_allowed:
                return _hold("hold_density_high", "density_above_upper_band")
            return _fin("finalize", "above_upper_no_hold_finalize")

        redundant = bool(
            source_gap_to_last_keyframe <= self.redundant_source_gap
            and gap_safe
            and not anchor_changed
            and density_state == "in_band"
            and keyframe_debt <= 0
            and not starvation_risk
        )
        if redundant and not displacement_novelty:
            dbg["hold_redundant_allowed"] = True
            return _hold("hold_redundant", "redundant_close_gap_in_band")

        if want_support and density_before <= self.density_target + 2.0:
            if (
                not budget_relaxed
                and self._budget.support_needed_used >= self.support_needed_budget_per_100
            ):
                dbg["support_needed_budget_exhausted"] = True
            else:
                return _fin("finalize_support_needed", "in_band_support_needed", use_support_budget=True)

        if want_high_novelty:
            if (
                not budget_relaxed
                and self._budget.high_novelty_used >= self.high_novelty_budget_per_100
            ):
                dbg["high_novelty_budget_exhausted"] = True
                if hold_density_high_allowed:
                    return _hold("hold_density_high", "high_novelty_budget_exhausted_in_band")
            else:
                return _fin("finalize_high_novelty", "budgeted_high_novelty", use_high_budget=True)

        return _fin("finalize", "default_in_band_finalize")

    def _decide_v22(
        self,
        *,
        frame_id: int,
        runtime_action: str,
        baseline_should_add: bool,
        is_test: bool,
        is_bootstrap_phase: bool,
        density_before: float,
        local_density_before: float,
        local_window_density: float,
        local_window_keyframes: int,
        local_window_gap_max: float,
        local_window_gap_after_if_hold: float,
        keyframe_growth_recent: int,
        baseline_relative_density: float,
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
        current_keyframe_count: int,
    ) -> DirectFinalizationDecision:
        self._sync_budget_window(frame_id)
        min_growth_window = max(1, int(self.min_growth_per_100 * 0.5))
        expected_min_keyframes = float(frame_id) * self.density_lower / 100.0
        keyframe_debt = max(0.0, expected_min_keyframes - float(current_keyframe_count))
        baseline_expected_kf = float(frame_id) * self.baseline_density_per_100 / 100.0
        baseline_min_kf = self.baseline_relative_lower_ratio * baseline_expected_kf

        dbg: dict[str, Any] = {
            "mode": self.mode,
            "frame_id": frame_id,
            "density_before": density_before,
            "density_after": density_before,
            "local_density_before": local_density_before,
            "local_window_density": local_window_density,
            "local_window_keyframes": local_window_keyframes,
            "local_window_gap_max": local_window_gap_max,
            "local_window_gap_after_if_hold": local_window_gap_after_if_hold,
            "baseline_relative_density": baseline_relative_density,
            "keyframe_count_current": current_keyframe_count,
            "expected_min_keyframes": round(expected_min_keyframes, 2),
            "keyframe_debt": round(keyframe_debt, 2),
            "recent_keyframe_growth": keyframe_growth_recent,
            "keyframe_growth_recent": keyframe_growth_recent,
            "source_gap_to_last_keyframe": source_gap_to_last_keyframe,
            "main_chain_gap_before": main_chain_gap_before,
            "main_chain_gap_after_if_hold": main_chain_gap_after_if_hold,
            "gap_critical_triggered": False,
            "gap_critical_finalized": False,
            "early_rescue_triggered": False,
            "early_rescue_budget_used": self._budget.early_rescue_used,
            "early_rescue_budget_exhausted": False,
            "local_density_rescue_triggered": False,
            "local_gap_rescue_triggered": False,
            "hold_density_high_allowed": False,
            "hold_density_high_blocked_by_gap": False,
            "hold_density_high_blocked_by_starvation": False,
            "hold_density_high_blocked_by_local_under_density": False,
            "hold_redundant_allowed": False,
            "hold_redundant_blocked_by_lower_guard": False,
            "hold_redundant_blocked_by_gap": False,
            "direct_admit_candidate": runtime_action in {"direct_admit", "current_frame_surrogate_commit"},
            "soft_gap_threshold": self.soft_gap_threshold
            if (self.is_v221 or self.is_v222_family)
            else self.gap_critical_limit,
            "preemptive_gap_threshold": self.preemptive_gap_threshold
            if self.is_v222_family
            else self.hard_gap_threshold,
            "hard_gap_threshold": self.hard_gap_threshold
            if (self.is_v221 or self.is_v222_family)
            else self.gap_critical_limit,
            "local_window_gap_before": local_window_gap_max,
            "density_cap": self._gap_rescue_density_cap(frame_id),
            "density_cap_for_gap_rescue": self._gap_rescue_density_cap(frame_id),
            "density_cap_exception_for_hard_gap": False,
            "post500_pre_gap_rescue_triggered": False,
            "post500_pre_gap_rescue_finalized": False,
            "post500_gap_rescue_budget_used": self._budget.post500_gap_rescue_used,
            "post500_gap_rescue_budget_exhausted": False,
            "hard_gap_rescue_finalized": False,
            "hard_gap_soft_budget_bypassed": False,
            "hard_gap_density_cap_exception": False,
            "candidate_missing_context": False,
            "predicted_next_gap_if_no_future_candidate": 0.0,
            "hold_density_high_blocked_by_hard_gap": False,
            "hold_redundant_blocked_by_hard_gap": False,
            "hold_density_high_blocked_by_pre_gap": False,
            "density_cap_exceeded": False,
            "preemptive_gap_rescue_triggered": False,
            "soft_gap_rescue_triggered": False,
            "gap_tail_rescue_triggered": False,
            "hard_gap_rescue_triggered": False,
            "gap_rescue_budget_used": self._budget.gap_rescue_used,
            "gap_rescue_budget_exhausted": False,
            "finalize_gap_tail_rescue_v2_2_1": False,
            "finalize_hard_gap_rescue_v2_2_1": False,
            "finalize_gap_tail_preemptive_v2_2_2": False,
            "finalize_hard_gap_rescue_v2_2_2": False,
            "hold_gap_rescue_blocked_by_density_cap": False,
        }
        direct_candidate = dbg["direct_admit_candidate"]
        if not self.enabled or not direct_candidate:
            return DirectFinalizationDecision(True, "finalize", "control_off_or_not_direct", dbg)
        if is_test or is_bootstrap_phase:
            return DirectFinalizationDecision(True, "finalize", "test_or_bootstrap_bypass", dbg)
        if bool(baseline_should_add) and runtime_action == "direct_admit":
            dbg["baseline_direct_admit_candidate"] = True

        growth_ok = keyframe_growth_recent >= min_growth_window
        baseline_relative_kf_ok = current_keyframe_count >= int(baseline_min_kf)
        growth_starved = bool(not growth_ok and density_before < self.density_target)
        local_under_dense = bool(
            local_window_density < self.local_density_lower
            and density_before < self.density_upper
        )
        starvation_risk = bool(
            density_before < self.density_lower
            or keyframe_debt > 0.5
            or growth_starved
            or (not baseline_relative_kf_ok and density_before < self.density_target)
            or local_under_dense
        )
        dbg["starvation_risk"] = starvation_risk
        density_state = self._density_state(density_before, starvation_risk=starvation_risk)
        dbg["density_state"] = density_state

        hard_lim = self.hard_gap_threshold if (self.is_v221 or self.is_v222_family) else self.gap_critical_limit
        soft_lim = self.soft_gap_threshold if (self.is_v221 or self.is_v222_family) else self.gap_critical_limit
        preemptive_lim = self.preemptive_gap_threshold if self.is_v222_family else hard_lim
        gap_hard = bool(
            self.is_v222_family
            and (
                main_chain_gap_after_if_hold > hard_lim
                or local_window_gap_after_if_hold > hard_lim
                or (
                    self.is_v2221
                    and source_gap_to_last_keyframe > hard_lim
                )
            )
        )
        gap_preemptive_v222 = bool(
            self.is_v222_family
            and not gap_hard
            and (
                main_chain_gap_after_if_hold >= preemptive_lim
                or local_window_gap_after_if_hold >= preemptive_lim
            )
        )
        gap_preemptive_source_v2221 = bool(
            self.is_v2221
            and int(frame_id) > 500
            and not gap_hard
            and source_gap_to_last_keyframe >= preemptive_lim
        )
        predicted_next_gap = max(
            main_chain_gap_after_if_hold,
            main_chain_gap_before + 2.0,
            local_window_gap_max,
            local_window_gap_after_if_hold,
        )
        if int(frame_id) > 500 and local_window_gap_max >= 9.0:
            predicted_next_gap = max(predicted_next_gap, 11.0)
        dbg["predicted_next_gap_if_no_future_candidate"] = float(predicted_next_gap)
        candidate_missing_periodic = bool(
            self.is_v2221
            and not gap_hard
            and local_window_gap_max >= 10.0
            and predicted_next_gap >= 11.0
            and (
                int(frame_id) > 500
                or main_chain_gap_before >= 6.0
            )
        )
        candidate_missing_post500_rescue = bool(
            candidate_missing_periodic and int(frame_id) > 500
        )
        candidate_missing_periodic_bypass_budget = bool(
            candidate_missing_periodic and local_window_gap_max >= 11.0
        )
        post500_pre_gap = bool(
            self.is_v2221
            and int(frame_id) > 500
            and not gap_hard
            and not gap_preemptive_source_v2221
            and not candidate_missing_post500_rescue
            and (
                main_chain_gap_before >= 8.0
                or (
                    local_window_gap_max >= 10.0
                    and source_gap_to_last_keyframe >= 8.0
                )
                or (
                    predicted_next_gap >= 10.0
                    and source_gap_to_last_keyframe >= 8.0
                )
            )
        )
        dbg["post500_pre_gap_rescue_triggered"] = post500_pre_gap
        dbg["candidate_missing_context"] = bool(
            candidate_missing_periodic
            or (
                self.is_v2221
                and int(frame_id) > 500
                and main_chain_gap_before >= 8.0
                and predicted_next_gap >= 10.0
            )
        )
        gap_hard_v222 = gap_hard
        gap_critical = bool(
            gap_hard
            or (
                not self.is_v222_family
                and (
                    main_chain_gap_after_if_hold > hard_lim
                    or local_window_gap_after_if_hold > hard_lim
                    or local_window_gap_max > hard_lim
                    or source_gap_to_last_keyframe > hard_lim
                )
            )
        )
        dbg["gap_critical_triggered"] = gap_critical
        dbg["hard_gap_rescue_triggered"] = gap_hard
        dbg["preemptive_gap_rescue_triggered"] = bool(
            gap_preemptive_v222 or gap_preemptive_source_v2221
        )
        eff_soft_lim = soft_lim
        if (self.is_v221 or self.is_v222_family) and int(frame_id) <= 500:
            eff_soft_lim = min(soft_lim, 5)
        gap_tail_soft = bool(
            (self.is_v221 or self.is_v222_family)
            and not gap_critical
            and not gap_preemptive_v222
            and not post500_pre_gap
            and (
                main_chain_gap_after_if_hold > eff_soft_lim
                or local_window_gap_after_if_hold > eff_soft_lim
                or local_window_gap_max > eff_soft_lim
                or source_gap_to_last_keyframe > eff_soft_lim
            )
        )
        dbg["soft_gap_rescue_triggered"] = gap_tail_soft
        dbg["gap_tail_rescue_triggered"] = bool(
            gap_tail_soft or gap_critical or gap_preemptive_v222
        )
        displacement_novelty = bool(
            median_displacement >= 1.35 * max(displacement_threshold, 1e-6)
        )
        extreme_novelty = bool(displacement_novelty and novelty_proxy >= 0.80)
        dbg["high_novelty_score"] = float(novelty_proxy if displacement_novelty else 0.0)
        dbg["support_needed_score"] = float(
            min(1.0, num_matches / max(min_num_inliers, 1)) if support_triggered else 0.0
        )
        want_support = bool(
            support_triggered
            and num_matches >= max(min_num_inliers, 1)
            and density_before < self.density_target + 5.0
        )
        want_high_novelty = bool(
            displacement_novelty
            and not support_triggered
            and density_before <= self.density_upper
        )
        budget_relaxed = bool(density_before < self.density_lower or keyframe_debt > 0)
        early_phase = int(frame_id) <= 500
        early_rescue_active = bool(
            early_phase
            and density_before
            < max(self.early_rescue_density_stop, self.density_lower + 7.0)
        )
        early_rescue_needed = bool(
            early_rescue_active
            and (
                density_before < self.density_lower
                or keyframe_debt > 0
                or growth_starved
                or local_under_dense
            )
        )
        dbg["early_rescue_triggered"] = early_rescue_needed
        local_density_rescue = bool(
            local_window_density < self.local_density_lower
            and density_before < self.density_upper
        )
        local_gap_rescue = bool(
            not self.is_v2221
            and local_window_gap_max > self.gap_critical_limit
            and not gap_critical
            and density_before < self.density_upper
        )
        dbg["local_density_rescue_triggered"] = local_density_rescue
        dbg["local_gap_rescue_triggered"] = local_gap_rescue

        gap_safe = bool(
            main_chain_gap_after_if_hold <= self.gap_critical_limit
            and local_window_gap_after_if_hold <= self.gap_critical_limit
        )
        hold_density_high_allowed = bool(
            density_before > self.density_upper
            and local_window_density > self.density_upper
            and not starvation_risk
            and keyframe_debt <= 0
            and gap_safe
            and growth_ok
        )
        dbg["hold_density_high_allowed"] = hold_density_high_allowed
        baseline_skeleton_local_cap = max(
            self.density_upper,
            self.gap_rescue_density_upper_later,
        )
        baseline_skeleton_global_cap = self.density_hard_upper
        if early_phase:
            baseline_skeleton_global_cap += 10.0
        baseline_skeleton_reserve = bool(
            self.is_v2221
            and bool(baseline_should_add)
            and runtime_action == "direct_admit"
            and gap_safe
            and not anchor_changed
            and not starvation_risk
            and keyframe_debt <= 0
            and local_window_density <= baseline_skeleton_local_cap
            and density_before <= baseline_skeleton_global_cap
        )
        dbg["baseline_skeleton_reserve"] = baseline_skeleton_reserve
        dbg["baseline_skeleton_local_cap"] = baseline_skeleton_local_cap
        dbg["baseline_skeleton_global_cap"] = baseline_skeleton_global_cap

        def _fin(
            decision: str,
            reason: str,
            *,
            use_high_budget: bool = False,
            use_support_budget: bool = False,
            use_early_budget: bool = False,
            use_gap_budget: bool = False,
            use_post500_budget: bool = False,
            exempt_budget: bool = False,
            hard_cap_exception: bool = False,
        ) -> DirectFinalizationDecision:
            if use_gap_budget:
                self._budget.gap_rescue_used += 1
                dbg["gap_rescue_budget_used"] = self._budget.gap_rescue_used
            if use_post500_budget:
                self._budget.post500_gap_rescue_used += 1
                dbg["post500_gap_rescue_budget_used"] = self._budget.post500_gap_rescue_used
            if use_early_budget:
                self._budget.early_rescue_used += 1
                dbg["early_rescue_budget_used"] = self._budget.early_rescue_used
                dbg["finalize_early_growth_rescue"] = decision == "finalize_early_growth_rescue"
            if use_high_budget and not (exempt_budget or budget_relaxed):
                self._budget.high_novelty_used += 1
                dbg["high_novelty_budget_used"] = self._budget.high_novelty_used
                dbg["finalize_high_novelty"] = True
            elif decision == "finalize_high_novelty":
                dbg["finalize_high_novelty"] = True
            if use_support_budget and not (exempt_budget or budget_relaxed):
                self._budget.support_needed_used += 1
                dbg["support_needed_budget_used"] = self._budget.support_needed_used
                dbg["finalize_support_needed"] = True
            elif decision in {"finalize_support_needed", "finalize_growth_rescue", "finalize_local_density_rescue"}:
                dbg[decision.replace("finalize_", "finalize_")] = True
            if decision == "finalize_gap_critical_v2_2":
                dbg["gap_critical_finalized"] = True
                dbg["finalize_gap_critical"] = True
            if decision == "finalize_local_gap_rescue":
                dbg["finalize_local_gap_rescue"] = True
            if decision == "finalize_local_density_rescue":
                dbg["finalize_local_density_rescue"] = True
            if decision == "finalize_gap_tail_rescue_v2_2_1":
                dbg["finalize_gap_tail_rescue_v2_2_1"] = True
                dbg["gap_tail_rescue_triggered"] = True
            if decision == "finalize_hard_gap_rescue_v2_2_1":
                dbg["finalize_hard_gap_rescue_v2_2_1"] = True
                dbg["hard_gap_rescue_triggered"] = True
                dbg["gap_critical_finalized"] = True
            if decision == "finalize_hard_gap_rescue_v2_2_2":
                dbg["finalize_hard_gap_rescue_v2_2_2"] = True
                dbg["hard_gap_rescue_triggered"] = True
                dbg["hard_gap_rescue_finalized"] = True
                dbg["gap_critical_finalized"] = True
            if decision == "finalize_hard_gap_rescue_v2_2_2_1":
                dbg["finalize_hard_gap_rescue_v2_2_2_1"] = True
                dbg["hard_gap_rescue_triggered"] = True
                dbg["hard_gap_rescue_finalized"] = True
                dbg["hard_gap_soft_budget_bypassed"] = True
                dbg["gap_critical_finalized"] = True
            if decision == "finalize_gap_tail_preemptive_v2_2_2":
                dbg["finalize_gap_tail_preemptive_v2_2_2"] = True
                dbg["preemptive_gap_rescue_triggered"] = True
            if decision == "finalize_post500_pre_gap_rescue_v2_2_2_1":
                dbg["post500_pre_gap_rescue_finalized"] = True
                dbg["post500_pre_gap_rescue_triggered"] = True
            if hard_cap_exception:
                cap = self._gap_rescue_density_cap(frame_id)
                if density_before > cap:
                    dbg["density_cap_exception_for_hard_gap"] = True
                    dbg["hard_gap_density_cap_exception"] = True
            dbg["direct_keyframe_finalized"] = True
            dbg["keyframe_finalized"] = True
            return DirectFinalizationDecision(True, decision, reason, dbg)

        def _gap_rescue_cap_ok(*, allow_over_cap_for_hard: bool = False) -> bool:
            cap = self._gap_rescue_density_cap(frame_id)
            dbg["density_cap"] = cap
            dbg["density_cap_for_gap_rescue"] = cap
            if allow_over_cap_for_hard and (gap_hard or gap_preemptive_source_v2221):
                if density_before > 50.0 and not (
                    main_chain_gap_after_if_hold > hard_lim
                    or local_window_gap_after_if_hold > hard_lim
                    or source_gap_to_last_keyframe > hard_lim
                ):
                    dbg["density_cap_exceeded"] = True
                    return False
                if density_before > cap:
                    dbg["density_cap_exception_for_hard_gap"] = True
                    dbg["hard_gap_density_cap_exception"] = True
                return True
            ok = bool(
                density_before <= cap
                and local_window_density <= cap + 2.0
            )
            if not ok:
                dbg["density_cap_exceeded"] = True
                dbg["hold_gap_rescue_blocked_by_density_cap"] = True
            return ok

        def _try_v222_family_rescue(*, from_hold: bool = False) -> DirectFinalizationDecision | None:
            if gap_hard:
                if self.is_v2221:
                    return _fin(
                        "finalize_hard_gap_rescue_v2_2_2_1",
                        "hard_gap_v2_2_2_1_from_hold" if from_hold else "hard_gap_v2_2_2_1",
                        exempt_budget=True,
                        hard_cap_exception=True,
                    )
                if self.is_v222:
                    return _fin(
                        "finalize_hard_gap_rescue_v2_2_2",
                        "hard_gap_v2_2_2_from_hold" if from_hold else "hard_gap_v2_2_2",
                        exempt_budget=True,
                    )
            if gap_preemptive_source_v2221:
                if _gap_rescue_cap_ok(allow_over_cap_for_hard=True):
                    return _fin(
                        "finalize_hard_gap_rescue_v2_2_2_1",
                        "preemptive_source_gap_v2_2_2_1_from_hold"
                        if from_hold
                        else "preemptive_source_gap_v2_2_2_1",
                        exempt_budget=True,
                        hard_cap_exception=True,
                    )
                if from_hold:
                    dbg["hold_density_high_blocked_by_hard_gap"] = True
                return None
            if candidate_missing_post500_rescue and self.is_v2221:
                cap = self._gap_rescue_density_cap(frame_id)
                dbg["density_cap"] = cap
                if density_before > cap:
                    if from_hold:
                        dbg["hold_density_high_blocked_by_pre_gap"] = True
                    return None
                use_post500_budget = (
                    self._budget.post500_gap_rescue_used
                    < self.post500_gap_rescue_budget_per_100
                )
                if not use_post500_budget and not candidate_missing_periodic_bypass_budget:
                    dbg["post500_gap_rescue_budget_exhausted"] = True
                    if (
                        predicted_next_gap >= 11.0
                        or source_gap_to_last_keyframe >= 11
                    ):
                        return _fin(
                            "finalize_post500_pre_gap_rescue_v2_2_2_1",
                            "candidate_missing_post500_budget_bypass_v2_2_2_1",
                            exempt_budget=True,
                        )
                    if from_hold:
                        dbg["hold_density_high_blocked_by_pre_gap"] = True
                    return None
                if not use_post500_budget:
                    dbg["post500_gap_rescue_budget_exhausted"] = True
                return _fin(
                    "finalize_post500_pre_gap_rescue_v2_2_2_1",
                    "candidate_missing_post500_from_hold"
                    if from_hold
                    else "candidate_missing_post500_v2_2_2_1",
                    use_post500_budget=use_post500_budget,
                    exempt_budget=True,
                )
            if post500_pre_gap and self.is_v2221:
                if not _gap_rescue_cap_ok():
                    if from_hold:
                        dbg["hold_density_high_blocked_by_pre_gap"] = True
                    return None
                if self._budget.post500_gap_rescue_used < self.post500_gap_rescue_budget_per_100:
                    return _fin(
                        "finalize_post500_pre_gap_rescue_v2_2_2_1",
                        "post500_pre_gap_from_hold" if from_hold else "post500_pre_gap_v2_2_2_1",
                        use_post500_budget=True,
                        exempt_budget=True,
                    )
                dbg["post500_gap_rescue_budget_exhausted"] = True
                if (
                    predicted_next_gap >= 11.0
                    or source_gap_to_last_keyframe >= 11
                ):
                    return _fin(
                        "finalize_post500_pre_gap_rescue_v2_2_2_1",
                        "post500_pre_gap_budget_bypass_v2_2_2_1",
                        exempt_budget=True,
                    )
                if from_hold:
                    dbg["hold_density_high_blocked_by_pre_gap"] = True
                return None
            if self.is_v222 or (self.is_v2221 and int(frame_id) <= 500):
                if gap_preemptive_v222:
                    if _gap_rescue_cap_ok():
                        return _fin(
                            "finalize_gap_tail_preemptive_v2_2_2",
                            "preemptive_gap_v2_2_2_from_hold"
                            if from_hold
                            else "preemptive_gap_v2_2_2",
                            exempt_budget=True,
                        )
                    if from_hold:
                        dbg["hold_density_high_blocked_by_gap"] = True
                        dbg["hold_redundant_blocked_by_gap"] = True
                    return None
                if gap_tail_soft and _gap_rescue_cap_ok():
                    if self._budget.gap_rescue_used < self.gap_rescue_budget_per_100:
                        return _fin(
                            "finalize_gap_tail_rescue_v2_2_1",
                            "gap_tail_soft_v2_2_2_from_hold"
                            if from_hold
                            else "gap_tail_soft_v2_2_2",
                            use_gap_budget=True,
                            exempt_budget=True,
                        )
                    dbg["gap_rescue_budget_exhausted"] = True
                    if from_hold and not gap_hard and not candidate_missing_periodic:
                        dbg["hold_gap_rescue_budget_exhausted"] = True
                        dbg["hold_density_high_blocked_by_gap"] = True
                        dbg["hold_redundant_blocked_by_gap"] = True
                        return DirectFinalizationDecision(
                            False,
                            "hold_gap_rescue_budget_exhausted",
                            "gap_rescue_budget_exhausted",
                            dbg,
                        )
                return None
            return None

        def _try_gap_tail_rescue(*, from_hold: bool = False) -> DirectFinalizationDecision | None:
            if self.is_v222_family:
                rescued = _try_v222_family_rescue(from_hold=from_hold)
                if rescued is not None:
                    return rescued
            if not (self.is_v221 or self.is_v222_family):
                return None
            if self.is_v222_family:
                return None
            if not _gap_rescue_cap_ok():
                return None
            if gap_critical:
                dbg["hard_gap_rescue_triggered"] = True
                return _fin(
                    "finalize_hard_gap_rescue_v2_2_1",
                    "hard_gap_v2_2_1_from_hold" if from_hold else "hard_gap_v2_2_1",
                    exempt_budget=True,
                )
            if gap_tail_soft:
                if self._budget.gap_rescue_used < self.gap_rescue_budget_per_100:
                    return _fin(
                        "finalize_gap_tail_rescue_v2_2_1",
                        "gap_tail_soft_v2_2_1_from_hold" if from_hold else "gap_tail_soft_v2_2_1",
                        use_gap_budget=True,
                        exempt_budget=True,
                    )
                dbg["gap_rescue_budget_exhausted"] = True
                if from_hold:
                    dbg["hold_density_high_blocked_by_gap"] = True
                    dbg["hold_redundant_blocked_by_gap"] = True
                    return DirectFinalizationDecision(
                        False,
                        "hold_gap_rescue_budget_exhausted",
                        "gap_rescue_budget_exhausted",
                        dbg,
                    )
            return None

        def _hold(decision: str, reason: str) -> DirectFinalizationDecision:
            if self.is_v221 or self.is_v222_family:
                rescued = _try_gap_tail_rescue(from_hold=True)
                if rescued is not None:
                    return rescued
            elif (
                source_gap_to_last_keyframe > 6
                and density_before >= self.density_lower
                and density_before <= self.density_upper
                and not starvation_risk
            ):
                dbg["gap_critical_triggered"] = True
                return _fin(
                    "finalize_gap_critical_v2_2",
                    "soft_gap_preempts_hold",
                    exempt_budget=True,
                )
            if gap_critical or not gap_safe:
                dbg["hold_density_high_blocked_by_gap"] = True
                dbg["hold_redundant_blocked_by_gap"] = True
                dbg["gap_critical_finalized"] = True
                return _fin("finalize_gap_critical_v2_2", "hold_blocked_by_gap_critical", exempt_budget=True)
            if starvation_risk or keyframe_debt > 0:
                dbg["hold_redundant_blocked_by_lower_guard"] = True
                if early_rescue_needed and self._budget.early_rescue_used < self.early_rescue_budget_per_100:
                    return _fin("finalize_early_growth_rescue", "starvation_preempts_hold", use_early_budget=True, exempt_budget=True)
                return _fin("finalize_growth_rescue", "starvation_preempts_hold", exempt_budget=True)
            if local_window_density < self.local_density_lower:
                dbg["hold_density_high_blocked_by_local_under_density"] = True
                return _fin("finalize_local_density_rescue", "local_under_density_preempts_hold", exempt_budget=True)
            dbg["hold_density_high"] = decision == "hold_density_high"
            dbg["hold_redundant"] = decision == "hold_redundant"
            dbg["direct_keyframe_finalized"] = False
            hold_reason = reason
            if (self.is_v221 or self.is_v222_family) and gap_safe:
                hold_reason = f"{reason}_gap_safe"
            return DirectFinalizationDecision(False, decision, hold_reason, dbg)

        if self.is_v2221 and gap_hard:
            return _fin(
                "finalize_hard_gap_rescue_v2_2_2_1",
                "priority_hard_gap_v2_2_2_1",
                exempt_budget=True,
                hard_cap_exception=True,
            )

        if baseline_skeleton_reserve:
            return _fin(
                "finalize_baseline_skeleton_reserve",
                "baseline_skeleton_reserve",
                exempt_budget=True,
            )

        if self.is_v221 or self.is_v222_family:
            rescued = _try_gap_tail_rescue(from_hold=False)
            if rescued is not None:
                return rescued
        elif gap_critical:
            return _fin("finalize_gap_critical_v2_2", "gap_critical_v2_2", exempt_budget=True)

        if anchor_changed:
            anchor_boundary_density_hold = bool(
                gap_safe
                and not starvation_risk
                and keyframe_debt <= 0
                and density_before >= self.density_target
                and local_window_density >= self.local_density_lower
                and not want_support
            )
            dbg["anchor_boundary_density_hold"] = anchor_boundary_density_hold
            if anchor_boundary_density_hold:
                return _hold("hold_density_high", "anchor_boundary_density_hold")
            return _fin("finalize_anchor_boundary", "anchor_boundary", exempt_budget=True)

        if early_rescue_needed:
            if self._budget.early_rescue_used < self.early_rescue_budget_per_100:
                return _fin("finalize_early_growth_rescue", "early_lower_bound_rescue", use_early_budget=True, exempt_budget=True)
            dbg["early_rescue_budget_exhausted"] = True

        if local_density_rescue:
            return _fin("finalize_local_density_rescue", "local_window_under_density", exempt_budget=True)

        if local_gap_rescue:
            return _fin("finalize_local_gap_rescue", "local_window_gap_rescue", exempt_budget=True)

        if starvation_risk or keyframe_debt > 0 or density_state == "below_lower":
            dbg["hold_redundant_blocked_by_lower_guard"] = True
            if want_support:
                return _fin("finalize_support_needed", "growth_rescue_support", use_support_budget=not budget_relaxed, exempt_budget=budget_relaxed)
            if want_high_novelty and (budget_relaxed or self._budget.high_novelty_used < self.high_novelty_budget_per_100):
                return _fin("finalize_high_novelty", "growth_rescue_novelty", use_high_budget=not budget_relaxed, exempt_budget=budget_relaxed)
            return _fin("finalize_growth_rescue", "keyframe_debt_rescue", exempt_budget=True)

        if density_state in {"above_hard", "above_upper"}:
            if want_support and self._budget.support_needed_used < self.support_needed_budget_per_100:
                return _fin("finalize_support_needed", "dense_band_support", use_support_budget=True)
            if (
                want_high_novelty
                and extreme_novelty
                and self._budget.high_novelty_used < self.high_novelty_budget_per_100
            ):
                return _fin("finalize_high_novelty", "dense_band_budgeted_novelty", use_high_budget=True)
            dbg["hold_density_high_blocked_by_starvation"] = bool(
                not hold_density_high_allowed and starvation_risk
            )
            if not early_phase or density_before >= self.density_upper:
                return _hold("hold_density_high", "at_or_above_upper_default_hold")

        redundant = bool(
            source_gap_to_last_keyframe <= self.redundant_source_gap
            and gap_safe
            and density_state == "in_band"
            and keyframe_debt <= 0
            and not starvation_risk
        )
        if redundant and not displacement_novelty:
            dbg["hold_redundant_allowed"] = True
            return _hold("hold_redundant", "redundant_in_band")

        if want_support and self._budget.support_needed_used < self.support_needed_budget_per_100:
            return _fin("finalize_support_needed", "in_band_support", use_support_budget=True)

        if want_high_novelty:
            if self._budget.high_novelty_used >= self.high_novelty_budget_per_100:
                dbg["high_novelty_budget_exhausted"] = True
                if hold_density_high_allowed or density_before >= self.density_target:
                    return _hold("hold_density_high", "in_band_budget_hold")
            else:
                return _fin("finalize_high_novelty", "in_band_high_novelty", use_high_budget=True)

        if (
            int(frame_id) > 500
            and density_before >= self.density_upper
            and gap_safe
            and not starvation_risk
        ):
            return _hold("hold_density_high", "long_run_above_upper_hold")

        return _fin("finalize", "default_in_band")

    def _decide_v1(
        self,
        *,
        frame_id: int,
        runtime_action: str,
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
            return DirectFinalizationDecision(True, "finalize", "control_off_or_not_direct", dbg)
        if is_test or is_bootstrap_phase:
            dbg["direct_keyframe_finalized"] = True
            return DirectFinalizationDecision(True, "finalize", "test_or_bootstrap_bypass", dbg)

        gap_critical = bool(
            main_chain_gap_after_if_hold >= self.gap_hard_limit
            or main_chain_gap_before >= self.gap_hard_limit - 2
        )
        displacement_novelty = bool(
            median_displacement >= 1.35 * max(displacement_threshold, 1e-6)
        )
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

        if gap_critical:
            dbg["finalize_gap_critical"] = True
            return DirectFinalizationDecision(True, "finalize_gap_critical", "gap_critical_override", dbg)
        if density_before < self.density_lower:
            return DirectFinalizationDecision(True, "finalize_support_needed", "density_below_lower_band", dbg)
        if anchor_changed:
            return DirectFinalizationDecision(True, "finalize_anchor_boundary", "finalize_anchor_boundary", dbg)
        if support_needed:
            return DirectFinalizationDecision(True, "finalize_support_needed", "support_needed_low_density", dbg)

        redundant = bool(
            source_gap_to_last_keyframe <= self.redundant_source_gap
            and main_chain_gap_before <= 5.0
            and not anchor_changed
            and density_before >= self.density_target
            and not gap_critical
        )
        if redundant and not displacement_novelty:
            dbg["hold_redundant"] = True
            return DirectFinalizationDecision(False, "hold_redundant", "redundant_close_gap_safe_chain", dbg)

        if density_before > self.density_upper:
            if gap_critical or displacement_novelty:
                decision = "finalize_gap_critical" if gap_critical else "finalize_high_novelty"
                return DirectFinalizationDecision(True, decision, decision, dbg)
            dbg["hold_density_high"] = True
            return DirectFinalizationDecision(False, "hold_density_high", "density_above_upper_band", dbg)

        if density_before > self.density_target and not displacement_novelty and not match_novelty:
            if source_gap_to_last_keyframe <= self.redundant_source_gap:
                dbg["hold_redundant"] = True
                return DirectFinalizationDecision(False, "hold_redundant", "above_target_close_gap_low_novelty", dbg)

        if high_novelty:
            return DirectFinalizationDecision(True, "finalize_high_novelty", "finalize_high_novelty", dbg)

        dbg["direct_keyframe_finalized"] = True
        return DirectFinalizationDecision(True, "finalize", "default_finalize", dbg)
