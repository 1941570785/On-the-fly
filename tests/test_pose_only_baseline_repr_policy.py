from __future__ import annotations

import unittest
from types import SimpleNamespace

from paper_aligned_policy.direct_density_control import DirectDensityController


def _args(**overrides):
    base = {
        "paper_aligned_direct_density_control": "pose_only_ssm_baseline_repr_v1",
        "paper_aligned_direct_update_prev_desc_on_hold": "light",
        "paper_aligned_direct_hold_tracking_bridge_mode": "light",
        "paper_aligned_direct_density_lower_per_100": None,
        "paper_aligned_direct_density_target_per_100": None,
        "paper_aligned_direct_density_upper_per_100": None,
        "paper_aligned_direct_density_hard_upper_per_100": None,
        "paper_aligned_direct_gap_hard_limit": None,
        "paper_aligned_direct_redundant_source_gap": None,
        "paper_aligned_direct_density_hysteresis_margin": None,
        "paper_aligned_direct_high_novelty_budget_per_100": None,
        "paper_aligned_direct_support_needed_budget_per_100": None,
        "paper_aligned_direct_min_growth_per_100": None,
        "paper_aligned_direct_baseline_relative_lower_ratio": None,
        "paper_aligned_direct_baseline_density_per_100": None,
        "paper_aligned_direct_local_window_size": None,
        "paper_aligned_direct_local_density_lower_per_100": None,
        "paper_aligned_direct_v2_2_1_soft_gap_threshold": None,
        "paper_aligned_direct_v2_2_1_hard_gap_threshold": None,
        "paper_aligned_direct_v2_2_1_gap_rescue_budget_per_100": None,
        "paper_aligned_direct_v2_2_1_gap_rescue_density_upper_500": None,
        "paper_aligned_direct_v2_2_1_gap_rescue_density_upper_later": None,
        "paper_aligned_direct_v2_2_2_preemptive_gap_threshold": None,
        "paper_aligned_direct_v2_2_2_1_post500_gap_rescue_budget_per_100": None,
        "paper_aligned_direct_v2_2_early_rescue_budget_per_100": None,
        "paper_aligned_direct_v2_2_early_rescue_density_stop": None,
        "paper_aligned_direct_representation_value_hold_max": None,
        "paper_aligned_direct_pose_reference_value_min": None,
        "paper_aligned_direct_value_hold_budget_per_100": None,
        "paper_aligned_direct_utility_representation_min": None,
        "paper_aligned_direct_utility_pose_reference_min": None,
        "paper_aligned_stream_memory_candidate_budget_per_100": None,
        "paper_aligned_pose_memory_geometry_context": "off",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _decision(controller, *, baseline_should_add: bool, is_test: bool = False, is_bootstrap: bool = False):
    return controller.decide(
        frame_id=240,
        runtime_action="direct_admit",
        baseline_should_add=baseline_should_add,
        is_test=is_test,
        is_bootstrap_phase=is_bootstrap,
        density_before=88.0,
        local_density_before=82.0,
        local_window_density=80.0,
        local_window_keyframes=80,
        local_window_gap_max=2.0,
        local_window_gap_after_if_hold=3.0,
        keyframe_growth_recent=60,
        baseline_relative_density=2.1,
        source_gap_to_last_keyframe=1,
        main_chain_gap_before=1.0,
        main_chain_gap_after_if_hold=2.0,
        anchor_changed=False,
        support_triggered=True,
        median_displacement=48.0,
        displacement_threshold=30.0,
        num_matches=1600,
        min_num_inliers=100,
        pose_inliers=900,
        novelty_proxy=0.55,
        current_keyframe_count=210,
        semantic_scores={"R_t": 0.18, "V_t": 0.62, "Q_t": 0.74, "B_R_t": 0.90, "C_t": 0.55},
        viewpoint_scores={
            "new_view_event_score": 0.30,
            "support_concentration": 0.16,
            "anchor_health_score": 0.42,
            "inlier_grid_entropy": 0.82,
            "viewpoint_rotation_deg_window_max": 22.0,
        },
    )


class PoseOnlyBaselineRepresentationPolicyTest(unittest.TestCase):
    def test_non_baseline_direct_admit_is_pose_only_and_not_materialized(self):
        controller = DirectDensityController(_args())

        decision = _decision(controller, baseline_should_add=False)

        self.assertFalse(decision.finalize)
        self.assertEqual(decision.decision, "hold_pose_only_baseline_repr")
        self.assertEqual(decision.reason, "non_baseline_pose_only_reference")
        self.assertTrue(decision.debug["active_memory_context"])
        self.assertEqual(decision.debug["active_memory_frame_role"], "tracking_only")
        self.assertEqual(decision.debug["stream_memory_frame_identity"], "pose-only")
        self.assertEqual(decision.debug["stream_memory_write_action"], "pose_only_write")
        self.assertFalse(controller.should_enqueue_hold_recovery(decision.decision))
        self.assertTrue(
            controller.should_update_prev_desc_on_hold(
                decision.decision, decision.debug["density_state"]
            )
        )

    def test_baseline_direct_admit_keeps_baseline_representation_frame(self):
        controller = DirectDensityController(_args())

        decision = _decision(controller, baseline_should_add=True)

        self.assertTrue(decision.finalize)
        self.assertEqual(decision.decision, "finalize_baseline_repr")
        self.assertEqual(decision.reason, "baseline_representation_keyframe")
        self.assertFalse(decision.debug["active_memory_context"])
        self.assertEqual(decision.debug["active_memory_frame_role"], "representation")
        self.assertEqual(decision.debug["stream_memory_frame_identity"], "render+pose")

    def test_test_and_bootstrap_frames_stay_materialized_for_baseline_compatibility(self):
        controller = DirectDensityController(_args())

        self.assertTrue(_decision(controller, baseline_should_add=False, is_test=True).finalize)
        self.assertTrue(_decision(controller, baseline_should_add=False, is_bootstrap=True).finalize)


if __name__ == "__main__":
    unittest.main()
