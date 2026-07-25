from pathlib import Path
import unittest

from scene.pose_render_extra_optimization import (
    pose_render_extra_optimization_decision,
    updated_render_response,
)


class PoseRenderExtraOptimizationTests(unittest.TestCase):
    def test_confident_pose_safe_frame_gets_bounded_extra_iterations(self):
        debug = pose_render_extra_optimization_decision(
            mode="pose_confidence_v1",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "direct_keyframe_finalized": True,
                    "pose_render_risk_score": 0.08,
                    "pose_render_risk_high": False,
                    "finalize_pose_risk_reference": False,
                    "utility_drift_risk": 0.10,
                    "pose_support_score": 0.88,
                    "match_support_score": 0.90,
                    "pose_render_pose_confidence": 0.92,
                },
            },
            base_iterations=30,
            min_confidence=0.75,
            fraction=0.25,
            max_extra=8,
        )

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["extra_iterations"], 7)
        self.assertEqual(debug["reason"], "pose_confident_extra_optimization")

    def test_risky_or_test_frames_do_not_get_extra_iterations(self):
        risky = pose_render_extra_optimization_decision(
            mode="pose_confidence_v1",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "direct_keyframe_finalized": True,
                    "pose_render_risk_score": 0.72,
                    "pose_render_risk_high": True,
                    "finalize_pose_risk_reference": False,
                    "utility_drift_risk": 0.10,
                    "pose_support_score": 0.88,
                    "match_support_score": 0.90,
                    "pose_render_pose_confidence": 0.92,
                },
            },
            base_iterations=30,
        )
        test_frame = pose_render_extra_optimization_decision(
            mode="pose_confidence_v1",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={"is_test": True},
            base_iterations=30,
        )

        self.assertFalse(risky["applied"])
        self.assertEqual(risky["extra_iterations"], 0)
        self.assertFalse(test_frame["applied"])
        self.assertEqual(test_frame["reason"], "test_frame")

    def test_render_response_mode_waits_for_latest_frame_response(self):
        debug = pose_render_extra_optimization_decision(
            mode="pose_confidence_render_response_v2",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "direct_keyframe_finalized": True,
                    "pose_render_risk_score": 0.08,
                    "pose_render_risk_high": False,
                    "finalize_pose_risk_reference": False,
                    "utility_drift_risk": 0.10,
                    "pose_support_score": 0.88,
                    "match_support_score": 0.90,
                    "pose_render_pose_confidence": 0.92,
                },
            },
            base_iterations=30,
        )

        self.assertFalse(debug["applied"])
        self.assertEqual(debug["extra_iterations"], 0)
        self.assertEqual(debug["reason"], "render_response_pending")

    def test_render_response_mode_rejects_low_or_degraded_improvement(self):
        keyframe_info = {
            "is_test": False,
            "_paper_aligned_pose_render_coupling": {
                "direct_keyframe_finalized": True,
                "pose_render_risk_score": 0.08,
                "pose_render_risk_high": False,
                "finalize_pose_risk_reference": False,
                "utility_drift_risk": 0.10,
                "pose_support_score": 0.88,
                "match_support_score": 0.90,
                "pose_render_pose_confidence": 0.92,
            },
            "_paper_aligned_pose_render_response": {
                "observations": 5,
                "first_rgb_mse": 0.020,
                "latest_rgb_mse": 0.0198,
                "relative_improvement": 0.010,
            },
        }

        debug = pose_render_extra_optimization_decision(
            mode="pose_confidence_render_response_v2",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info=keyframe_info,
            base_iterations=30,
        )

        self.assertFalse(debug["applied"])
        self.assertEqual(debug["extra_iterations"], 0)
        self.assertEqual(debug["reason"], "render_response_low")

    def test_render_response_mode_grants_bounded_extra_iterations_after_drop(self):
        debug = pose_render_extra_optimization_decision(
            mode="pose_confidence_render_response_v2",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "direct_keyframe_finalized": True,
                    "pose_render_risk_score": 0.08,
                    "pose_render_risk_high": False,
                    "finalize_pose_risk_reference": False,
                    "utility_drift_risk": 0.10,
                    "pose_support_score": 0.88,
                    "match_support_score": 0.90,
                    "pose_render_pose_confidence": 0.92,
                },
                "_paper_aligned_pose_render_response": {
                    "observations": 5,
                    "first_rgb_mse": 0.020,
                    "latest_rgb_mse": 0.016,
                    "relative_improvement": 0.20,
                },
            },
            base_iterations=30,
            min_confidence=0.75,
            fraction=0.25,
            max_extra=8,
        )

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["extra_iterations"], 7)
        self.assertEqual(debug["reason"], "render_response_extra_optimization")
        self.assertGreaterEqual(debug["render_response_scale"], 0.99)

    def test_render_response_v3_refines_high_deficit_frame_without_pose_gate(self):
        debug = pose_render_extra_optimization_decision(
            mode="render_response_v3",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_texture_sampling_coverage": {
                    "coverage": 0.88,
                    "coverage_deficit": 0.12,
                },
                "_paper_aligned_pose_render_texture_sampling_scene_guard": {
                    "events": 20,
                    "applied": 16,
                    "coverage_deficit_mean": 0.12,
                    "sampling_applied_ratio": 0.80,
                    "response_evaluated": 32,
                    "response_bad_ratio": 0.0,
                },
                "_paper_aligned_pose_render_response": {
                    "observations": 5,
                    "first_rgb_mse": 0.020,
                    "latest_rgb_mse": 0.016,
                    "relative_improvement": 0.20,
                },
            },
            base_iterations=30,
            fraction=0.25,
            max_extra=8,
        )

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["reason"], "render_response_extra_optimization")
        self.assertEqual(debug["extra_iterations"], 8)
        self.assertEqual(debug["confidence"], 1.0)
        self.assertAlmostEqual(debug["coverage_deficit"], 0.12)

    def test_render_response_v3_requires_four_response_observations(self):
        debug = pose_render_extra_optimization_decision(
            mode="render_response_v3",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_texture_sampling_coverage": {
                    "coverage": 0.88,
                    "coverage_deficit": 0.12,
                },
                "_paper_aligned_pose_render_response": {
                    "observations": 3,
                    "first_rgb_mse": 0.020,
                    "latest_rgb_mse": 0.016,
                    "best_rgb_mse": 0.016,
                    "recent_rgb_mse": [0.020, 0.018, 0.016],
                    "relative_improvement": 0.20,
                },
            },
            base_iterations=30,
            fraction=0.25,
            max_extra=8,
        )

        self.assertFalse(debug["applied"])
        self.assertEqual(debug["reason"], "render_response_pending")
        self.assertEqual(debug["extra_iterations"], 0)

    def test_render_response_v3_rejects_unstable_recent_response(self):
        debug = pose_render_extra_optimization_decision(
            mode="render_response_v3",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_texture_sampling_coverage": {
                    "coverage": 0.88,
                    "coverage_deficit": 0.12,
                },
                "_paper_aligned_pose_render_response": {
                    "observations": 4,
                    "first_rgb_mse": 0.020,
                    "latest_rgb_mse": 0.016,
                    "best_rgb_mse": 0.015,
                    "recent_rgb_mse": [0.020, 0.015, 0.017, 0.016],
                    "relative_improvement": 0.20,
                },
            },
            base_iterations=30,
            fraction=0.25,
            max_extra=8,
        )

        self.assertFalse(debug["applied"])
        self.assertEqual(debug["reason"], "render_response_unstable")
        self.assertGreater(debug["render_response_rebound_ratio"], 0.01)

    def test_render_response_v3_rejects_gap_too_large_for_visible_refinement(self):
        debug = pose_render_extra_optimization_decision(
            mode="render_response_v3",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_texture_sampling_coverage": {
                    "coverage": 0.60,
                    "coverage_deficit": 0.40,
                },
                "_paper_aligned_pose_render_response": {
                    "observations": 4,
                    "first_rgb_mse": 0.020,
                    "latest_rgb_mse": 0.014,
                    "best_rgb_mse": 0.014,
                    "recent_rgb_mse": [0.020, 0.018, 0.016, 0.014],
                    "relative_improvement": 0.30,
                },
            },
            base_iterations=30,
            fraction=0.25,
            max_extra=8,
        )

        self.assertFalse(debug["applied"])
        self.assertEqual(debug["reason"], "coverage_gap_too_large")
        self.assertEqual(debug["extra_iterations"], 0)

    def test_render_response_v3_does_not_repeat_a_finalized_keyframe(self):
        debug = pose_render_extra_optimization_decision(
            mode="render_response_v3",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_extra_refinement_state": {
                    "state": "DONE",
                    "attempted": True,
                },
                "_paper_aligned_pose_render_texture_sampling_coverage": {
                    "coverage": 0.88,
                    "coverage_deficit": 0.12,
                },
                "_paper_aligned_pose_render_response": {
                    "observations": 4,
                    "first_rgb_mse": 0.020,
                    "latest_rgb_mse": 0.014,
                    "best_rgb_mse": 0.014,
                    "recent_rgb_mse": [0.020, 0.018, 0.016, 0.014],
                    "relative_improvement": 0.30,
                },
            },
            base_iterations=30,
            fraction=0.25,
            max_extra=8,
        )

        self.assertFalse(debug["applied"])
        self.assertEqual(debug["reason"], "refinement_already_finalized")
        self.assertEqual(debug["extra_iterations"], 0)

    def test_render_response_v3_is_self_authorized_without_render_lock_policy(self):
        debug = pose_render_extra_optimization_decision(
            mode="render_response_v3",
            direct_density_mode="off",
            render_frame_policy="off",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_texture_sampling_coverage": {
                    "coverage": 0.88,
                    "coverage_deficit": 0.12,
                },
                "_paper_aligned_pose_render_texture_sampling_scene_guard": {
                    "events": 20,
                    "applied": 16,
                    "coverage_deficit_mean": 0.12,
                    "sampling_applied_ratio": 0.80,
                    "response_evaluated": 32,
                    "response_bad_ratio": 0.0,
                },
                "_paper_aligned_pose_render_response": {
                    "observations": 5,
                    "first_rgb_mse": 0.020,
                    "latest_rgb_mse": 0.016,
                    "relative_improvement": 0.20,
                },
            },
            base_iterations=30,
            fraction=0.25,
            max_extra=8,
        )

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["reason"], "render_response_extra_optimization")
        self.assertEqual(debug["extra_iterations"], 8)

    def test_render_response_v3_rejects_coverage_sufficient_frame(self):
        debug = pose_render_extra_optimization_decision(
            mode="render_response_v3",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_texture_sampling_coverage": {
                    "coverage": 0.96,
                    "coverage_deficit": 0.04,
                },
                "_paper_aligned_pose_render_response": {
                    "observations": 5,
                    "first_rgb_mse": 0.020,
                    "latest_rgb_mse": 0.016,
                    "relative_improvement": 0.20,
                },
            },
            base_iterations=30,
        )

        self.assertFalse(debug["applied"])
        self.assertEqual(debug["reason"], "coverage_sufficient")
        self.assertEqual(debug["extra_iterations"], 0)

    def test_render_response_v3_uses_selective_representation_gap_when_projection_is_full(self):
        debug = pose_render_extra_optimization_decision(
            mode="render_response_v3",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_texture_sampling_coverage": {
                    "coverage": 0.99,
                    "coverage_deficit": 0.01,
                },
                "_paper_aligned_pose_render_response": {
                    "observations": 4,
                    "first_rgb_mse": 0.020,
                    "latest_rgb_mse": 0.014,
                    "best_rgb_mse": 0.014,
                    "recent_rgb_mse": [0.020, 0.018, 0.016, 0.014],
                    "relative_improvement": 0.30,
                    "representation_coverage_deficit": 0.12,
                    "representation_gap_selectivity": 0.45,
                },
            },
            base_iterations=30,
            fraction=0.25,
            max_extra=8,
        )

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["coverage_deficit"], 0.12)
        self.assertEqual(debug["projection_coverage_deficit"], 0.01)
        self.assertEqual(debug["representation_coverage_deficit"], 0.12)

    def test_render_response_v3_uses_current_view_despite_low_scene_pressure(self):
        debug = pose_render_extra_optimization_decision(
            mode="render_response_v3",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_texture_sampling_coverage": {
                    "coverage": 0.88,
                    "coverage_deficit": 0.12,
                },
                "_paper_aligned_pose_render_texture_sampling_scene_guard": {
                    "coverage_deficit_mean": 0.02,
                    "sampling_applied_ratio": 0.04,
                },
                "_paper_aligned_pose_render_response": {
                    "observations": 5,
                    "first_rgb_mse": 0.020,
                    "latest_rgb_mse": 0.016,
                    "relative_improvement": 0.20,
                },
            },
            base_iterations=30,
        )

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["reason"], "render_response_extra_optimization")
        self.assertEqual(debug["extra_iterations"], 8)

    def test_render_response_v3_ignores_scene_mean_when_current_view_has_deficit(self):
        debug = pose_render_extra_optimization_decision(
            mode="render_response_v3",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_texture_sampling_coverage": {
                    "coverage": 0.88,
                    "coverage_deficit": 0.12,
                },
                "_paper_aligned_pose_render_texture_sampling_scene_guard": {
                    "events": 40,
                    "applied": 32,
                    "coverage_deficit_mean": 0.07,
                    "sampling_applied_ratio": 0.80,
                },
                "_paper_aligned_pose_render_response": {
                    "observations": 5,
                    "first_rgb_mse": 0.020,
                    "latest_rgb_mse": 0.016,
                    "relative_improvement": 0.20,
                },
            },
            base_iterations=30,
            fraction=0.25,
            max_extra=8,
        )

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["reason"], "render_response_extra_optimization")
        self.assertEqual(debug["extra_iterations"], 8)

    def test_render_response_v3_ignores_scene_sampling_ratio(self):
        debug = pose_render_extra_optimization_decision(
            mode="render_response_v3",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_texture_sampling_coverage": {
                    "coverage": 0.88,
                    "coverage_deficit": 0.12,
                },
                "_paper_aligned_pose_render_texture_sampling_scene_guard": {
                    "events": 40,
                    "applied": 2,
                    "coverage_deficit_mean": 0.12,
                    "sampling_applied_ratio": 0.05,
                },
                "_paper_aligned_pose_render_response": {
                    "observations": 5,
                    "first_rgb_mse": 0.020,
                    "latest_rgb_mse": 0.016,
                    "relative_improvement": 0.20,
                },
            },
            base_iterations=30,
            fraction=0.25,
            max_extra=8,
        )

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["reason"], "render_response_extra_optimization")
        self.assertEqual(debug["extra_iterations"], 8)

    def test_render_response_v3_does_not_require_scene_pressure_warmup(self):
        debug = pose_render_extra_optimization_decision(
            mode="render_response_v3",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_texture_sampling_coverage": {
                    "coverage": 0.88,
                    "coverage_deficit": 0.12,
                },
                "_paper_aligned_pose_render_texture_sampling_scene_guard": {
                    "events": 10,
                    "applied": 8,
                    "coverage_deficit_mean": 0.12,
                    "sampling_applied_ratio": 0.80,
                },
                "_paper_aligned_pose_render_response": {
                    "observations": 5,
                    "first_rgb_mse": 0.020,
                    "latest_rgb_mse": 0.016,
                    "relative_improvement": 0.20,
                },
            },
            base_iterations=30,
            fraction=0.25,
            max_extra=8,
        )

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["reason"], "render_response_extra_optimization")
        self.assertEqual(debug["extra_iterations"], 8)

    def test_render_response_v3_uses_per_view_response_despite_scene_instability(self):
        debug = pose_render_extra_optimization_decision(
            mode="render_response_v3",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_texture_sampling_coverage": {
                    "coverage": 0.88,
                    "coverage_deficit": 0.12,
                },
                "_paper_aligned_pose_render_texture_sampling_scene_guard": {
                    "events": 40,
                    "applied": 32,
                    "coverage_deficit_mean": 0.12,
                    "sampling_applied_ratio": 0.80,
                    "response_evaluated": 16,
                    "response_bad_ratio": 0.0625,
                },
                "_paper_aligned_pose_render_response": {
                    "observations": 5,
                    "first_rgb_mse": 0.020,
                    "latest_rgb_mse": 0.016,
                    "relative_improvement": 0.20,
                },
            },
            base_iterations=30,
            fraction=0.25,
            max_extra=8,
        )

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["reason"], "render_response_extra_optimization")
        self.assertEqual(debug["extra_iterations"], 8)

    def test_render_response_v3_does_not_require_scene_response_warmup(self):
        debug = pose_render_extra_optimization_decision(
            mode="render_response_v3",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_texture_sampling_coverage": {
                    "coverage": 0.88,
                    "coverage_deficit": 0.12,
                },
                "_paper_aligned_pose_render_texture_sampling_scene_guard": {
                    "events": 40,
                    "applied": 32,
                    "coverage_deficit_mean": 0.12,
                    "sampling_applied_ratio": 0.80,
                    "response_evaluated": 12,
                    "response_bad_ratio": 0.0,
                },
                "_paper_aligned_pose_render_response": {
                    "observations": 5,
                    "first_rgb_mse": 0.020,
                    "latest_rgb_mse": 0.016,
                    "relative_improvement": 0.20,
                },
            },
            base_iterations=30,
            fraction=0.25,
            max_extra=8,
        )

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["reason"], "render_response_extra_optimization")
        self.assertEqual(debug["extra_iterations"], 8)

    def test_render_response_v4_allows_moderate_scene_pressure_with_small_bad_ratio(self):
        debug = pose_render_extra_optimization_decision(
            mode="render_response_v4",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_texture_sampling_coverage": {
                    "coverage": 0.92,
                    "coverage_deficit": 0.08,
                },
                "_paper_aligned_pose_render_texture_sampling_scene_guard": {
                    "events": 40,
                    "applied": 15,
                    "coverage_deficit_mean": 0.078,
                    "sampling_applied_ratio": 0.375,
                    "response_evaluated": 40,
                    "response_bad_ratio": 0.06,
                },
                "_paper_aligned_pose_render_response": {
                    "observations": 5,
                    "first_rgb_mse": 0.020,
                    "latest_rgb_mse": 0.016,
                    "relative_improvement": 0.20,
                },
            },
            base_iterations=30,
            fraction=0.25,
            max_extra=8,
        )

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["reason"], "render_response_extra_optimization")
        self.assertEqual(debug["extra_iterations"], 2)
        self.assertEqual(debug["max_scene_response_bad_ratio"], 0.06)
        self.assertEqual(debug["min_scene_coverage_deficit"], 0.075)

    def test_render_response_v4_still_rejects_coverage_sufficient_frame(self):
        debug = pose_render_extra_optimization_decision(
            mode="render_response_v4",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_texture_sampling_coverage": {
                    "coverage": 0.98,
                    "coverage_deficit": 0.02,
                },
                "_paper_aligned_pose_render_texture_sampling_scene_guard": {
                    "events": 40,
                    "applied": 15,
                    "coverage_deficit_mean": 0.078,
                    "sampling_applied_ratio": 0.375,
                    "response_evaluated": 40,
                    "response_bad_ratio": 0.05,
                },
                "_paper_aligned_pose_render_response": {
                    "observations": 5,
                    "first_rgb_mse": 0.020,
                    "latest_rgb_mse": 0.016,
                    "relative_improvement": 0.20,
                },
            },
            base_iterations=30,
        )

        self.assertFalse(debug["applied"])
        self.assertEqual(debug["reason"], "coverage_sufficient")
        self.assertEqual(debug["extra_iterations"], 0)

    def test_render_response_v5_rescues_mild_gap_with_stable_response(self):
        debug = pose_render_extra_optimization_decision(
            mode="render_response_v5",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_texture_sampling_coverage": {
                    "coverage": 0.93,
                    "coverage_deficit": 0.07,
                },
                "_paper_aligned_pose_render_texture_sampling_scene_guard": {
                    "events": 40,
                    "applied": 14,
                    "coverage_deficit_mean": 0.07,
                    "sampling_applied_ratio": 0.35,
                    "response_evaluated": 40,
                    "response_bad_ratio": 0.08,
                },
                "_paper_aligned_pose_render_response": {
                    "observations": 5,
                    "first_rgb_mse": 0.020,
                    "latest_rgb_mse": 0.016,
                    "relative_improvement": 0.20,
                },
            },
            base_iterations=30,
            fraction=0.25,
            max_extra=8,
        )

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["reason"], "render_response_extra_optimization")
        self.assertEqual(debug["extra_iterations"], 2)
        self.assertEqual(debug["max_scene_response_bad_ratio"], 0.10)
        self.assertEqual(debug["min_scene_coverage_deficit"], 0.065)

    def test_render_response_v5_still_rejects_low_scene_pressure(self):
        debug = pose_render_extra_optimization_decision(
            mode="render_response_v5",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_texture_sampling_coverage": {
                    "coverage": 0.93,
                    "coverage_deficit": 0.07,
                },
                "_paper_aligned_pose_render_texture_sampling_scene_guard": {
                    "events": 40,
                    "applied": 0,
                    "coverage_deficit_mean": 0.001,
                    "sampling_applied_ratio": 0.0,
                    "response_evaluated": 40,
                    "response_bad_ratio": 0.0,
                },
                "_paper_aligned_pose_render_response": {
                    "observations": 5,
                    "first_rgb_mse": 0.020,
                    "latest_rgb_mse": 0.016,
                    "relative_improvement": 0.20,
                },
            },
            base_iterations=30,
            fraction=0.25,
            max_extra=8,
        )

        self.assertFalse(debug["applied"])
        self.assertEqual(debug["reason"], "scene_coverage_sufficient")
        self.assertEqual(debug["extra_iterations"], 0)

    def test_render_response_mask_conservative_bypasses_mask_blocked_scene(self):
        debug = pose_render_extra_optimization_decision(
            mode="render_response_mask_conservative_v6",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_training_background": {
                    "mode": "dark_scene_fixed_black_v1",
                    "dark_scene_decision": None,
                    "dark_scene_observations": 0,
                    "dark_scene_mask_blocked": 128,
                },
                "_paper_aligned_pose_render_texture_sampling_coverage": {
                    "coverage": 0.93,
                    "coverage_deficit": 0.07,
                },
                "_paper_aligned_pose_render_texture_sampling_scene_guard": {
                    "events": 40,
                    "applied": 14,
                    "coverage_deficit_mean": 0.07,
                    "sampling_applied_ratio": 0.35,
                    "response_evaluated": 40,
                    "response_bad_ratio": 0.08,
                },
                "_paper_aligned_pose_render_response": {
                    "observations": 5,
                    "first_rgb_mse": 0.020,
                    "latest_rgb_mse": 0.016,
                    "relative_improvement": 0.20,
                },
            },
            base_iterations=30,
            fraction=0.25,
            max_extra=8,
        )

        self.assertFalse(debug["applied"])
        self.assertEqual(debug["reason"], "mask_blocked_scene_bypass")
        self.assertEqual(debug["training_background_dark_scene_mask_blocked"], 128)
        self.assertEqual(debug["extra_iterations"], 0)

    def test_render_response_history_keeps_the_latest_four_observations(self):
        response = {}
        for value in (0.040, 0.030, 0.025, 0.024, 0.023):
            response = updated_render_response(
                response,
                value,
                representation_coverage_deficit=0.12,
                representation_gap_selectivity=0.45,
            )

        self.assertEqual(response["observations"], 5)
        self.assertEqual(response["first_rgb_mse"], 0.040)
        self.assertEqual(response["latest_rgb_mse"], 0.023)
        self.assertEqual(response["best_rgb_mse"], 0.023)
        self.assertEqual(response["representation_coverage_deficit"], 0.12)
        self.assertEqual(response["representation_gap_selectivity"], 0.45)
        self.assertEqual(response["recent_rgb_mse"], [0.030, 0.025, 0.024, 0.023])

    def test_scene_model_focuses_transactional_extra_iterations_on_latest_keyframe(self):
        source = Path("scene/scene_model.py").read_text(encoding="utf-8")

        self.assertIn("pose_render_extra_optimization_decision", source)
        self.assertIn("_run_transactional_gaussian_refinement", source)
        self.assertIn("refinement_candidate_acceptance", source)
        self.assertIn("scale_gaussian_gradients", source)

    def test_scene_model_defers_render_response_extra_decision_until_after_base_loop(self):
        source = Path("scene/scene_model.py").read_text(encoding="utf-8")

        mode_check = 'self.pose_render_extra_optimization == "pose_confidence_render_response_v2"'
        render_mode_check = 'self.pose_render_extra_optimization == "render_response_v3"'
        render_v4_mode_check = 'self.pose_render_extra_optimization == "render_response_v4"'
        render_v6_mode_check = 'self.pose_render_extra_optimization == "render_response_mask_conservative_v6"'
        scene_guard_refresh = 'TEXTURE_SAMPLING_SCENE_GUARD_KEY] = self._pose_render_texture_sampling_scene_guard()'
        loop_check = "while i < n_iters"
        response_key = '"_paper_aligned_pose_render_response"'

        self.assertIn(mode_check, source)
        self.assertIn(render_mode_check, source)
        self.assertIn(render_v4_mode_check, source)
        self.assertIn(render_v6_mode_check, source)
        self.assertIn(scene_guard_refresh, source)
        self.assertIn(response_key, source)
        self.assertLess(source.index(loop_check), source.rindex(mode_check))
        refresh_block_start = source.rindex("if (", 0, source.rindex(scene_guard_refresh))
        refresh_block = source[refresh_block_start : source.rindex(scene_guard_refresh)]
        self.assertNotIn('"render_response_v3"', refresh_block)


if __name__ == "__main__":
    unittest.main()
