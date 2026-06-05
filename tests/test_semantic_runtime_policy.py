from __future__ import annotations

import unittest

from paper_aligned_policy.config import Thresholds
from paper_aligned_policy.semantic_runtime import SemanticV1RuntimePolicy


class SemanticRuntimePolicyTests(unittest.TestCase):
    def test_doc_scores_keep_baseline_gate_out_of_risk_score(self):
        policy = SemanticV1RuntimePolicy()

        add_scores = policy._scores(
            True,
            {
                "num_matches": 100,
                "min_num_inliers_threshold": 100,
                "median_displacement": 0.8,
                "displacement_threshold": 1.0,
                "recent_pose_fail_rate": 0.2,
            },
        )
        hold_scores = policy._scores(
            False,
            {
                "num_matches": 100,
                "min_num_inliers_threshold": 100,
                "median_displacement": 0.8,
                "displacement_threshold": 1.0,
                "recent_pose_fail_rate": 0.2,
            },
        )

        self.assertAlmostEqual(add_scores["R_t"], hold_scores["R_t"])
        self.assertGreater(add_scores["V_t"], hold_scores["V_t"])
        self.assertIn("B_R_t", hold_scores)
        self.assertIn("C_t", hold_scores)

    def test_defer_requires_recoverable_risk_band_and_delayed_future_context(self):
        policy = SemanticV1RuntimePolicy(
            thresholds=Thresholds(tau_R_low=0.25, tau_R_high=0.75, tau_B=0.25, tau_V_min=0.25, tau_Q=0.15),
            recovery_delay_frames=2,
            recovery_attempts_per_tick=1,
        )
        evidence = {
            "num_matches": 100,
            "min_num_inliers_threshold": 100,
            "median_displacement": 0.9,
            "displacement_threshold": 1.0,
            "recent_pose_fail_rate": 0.25,
        }

        first = policy.decide(10, baseline_should_add=False, evidence=evidence, source_payload={"source_input_index": 10})
        self.assertEqual(first["action"], "defer_recoverable")
        self.assertEqual(policy.pop_recovered_sources(), [])

        second = policy.decide(11, baseline_should_add=False, evidence=evidence, source_payload={"source_input_index": 11})
        self.assertEqual(second["recovery_tick"]["attempted"], 0)
        self.assertEqual(policy.pop_recovered_sources(), [])

        third = policy.decide(12, baseline_should_add=False, evidence=evidence, source_payload={"source_input_index": 12})
        self.assertEqual(third["recovery_tick"]["attempted"], 1)
        recovered = policy.pop_recovered_sources()
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0]["source_frame_id"], 10)
        self.assertEqual(recovered[0]["current_tick_frame_id"], 12)
        self.assertGreater(recovered[0]["current_tick_frame_id"], recovered[0]["source_frame_id"])

    def test_non_baseline_frame_cannot_direct_admit_even_when_scores_are_strong(self):
        policy = SemanticV1RuntimePolicy(
            thresholds=Thresholds(tau_R_low=0.40, tau_R_high=0.75, tau_B=0.12, tau_V=0.55, tau_V_min=0.45, tau_Q=0.10)
        )

        decision = policy.decide(
            20,
            baseline_should_add=False,
            evidence={
                "pose_uncertainty": 0.0,
                "state_support_gap": 0.0,
                "temporal_degradation": 0.0,
                "representation_gain": 0.90,
                "view_motion_gain": 0.90,
                "chain_support_gain": 0.90,
                "recovery_context_score": 1.0,
            },
            source_payload={"source_input_index": 20},
        )

        self.assertGreaterEqual(decision["scores"]["V_t"], policy.th.tau_V)
        self.assertEqual(decision["action"], "defer_recoverable")
        self.assertEqual(decision["recovery_pool_size"], 1)

    def test_low_risk_moderate_value_frame_can_enter_recovery_pool(self):
        policy = SemanticV1RuntimePolicy(
            thresholds=Thresholds(tau_R_low=0.40, tau_R_high=0.75, tau_B=0.25, tau_V=0.55, tau_V_min=0.25, tau_Q=0.10)
        )

        decision = policy.decide(
            21,
            baseline_should_add=False,
            evidence={
                "pose_uncertainty": 0.20,
                "state_support_gap": 0.20,
                "temporal_degradation": 0.20,
                "representation_gain": 0.40,
                "view_motion_gain": 0.40,
                "chain_support_gain": 0.40,
                "recovery_context_score": 0.80,
            },
            source_payload={"source_input_index": 21},
        )

        self.assertEqual(decision["action"], "defer_recoverable")
        self.assertLess(decision["scores"]["V_t"], policy.th.tau_V)
        self.assertGreaterEqual(decision["scores"]["B_R_t"], policy.th.tau_B)
        self.assertEqual(decision["recovery_pool_size"], 1)

    def test_zero_risk_moderate_value_frame_is_safe_recoverable_not_discarded(self):
        policy = SemanticV1RuntimePolicy(
            thresholds=Thresholds(tau_R_low=0.40, tau_R_high=0.75, tau_B=0.12, tau_V=0.55, tau_V_min=0.45, tau_Q=0.10)
        )

        decision = policy.decide(
            22,
            baseline_should_add=False,
            evidence={
                "pose_uncertainty": 0.0,
                "state_support_gap": 0.0,
                "temporal_degradation": 0.0,
                "representation_gain": 0.52,
                "view_motion_gain": 0.52,
                "chain_support_gain": 0.52,
                "recovery_context_score": 1.0,
            },
            source_payload={"source_input_index": 22},
        )

        self.assertEqual(decision["scores"]["R_t"], 0.0)
        self.assertEqual(decision["scores"]["B_R_t"], 1.0)
        self.assertLess(decision["scores"]["V_t"], policy.th.tau_V)
        self.assertEqual(decision["action"], "defer_recoverable")

    def test_high_risk_low_band_frame_is_discarded_even_when_value_is_visible(self):
        policy = SemanticV1RuntimePolicy(
            thresholds=Thresholds(tau_R_low=0.25, tau_R_high=0.70, tau_B=0.30, tau_V_min=0.25, tau_Q=0.10)
        )

        decision = policy.decide(
            1,
            baseline_should_add=True,
            evidence={
                "num_matches": 0,
                "min_num_inliers_threshold": 100,
                "median_displacement": 1.0,
                "displacement_threshold": 1.0,
                "recent_pose_fail_rate": 1.0,
            },
        )

        self.assertEqual(decision["action"], "discard")


if __name__ == "__main__":
    unittest.main()
