import unittest
from pathlib import Path

from scene.pose_render_init_weighting import pose_render_init_weighting_decision


class PoseRenderInitWeightingTests(unittest.TestCase):
    def test_low_risk_keeps_baseline_opacity(self):
        debug = pose_render_init_weighting_decision(
            mode="risk_opacity_v1",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_render_risk_score": 0.05,
                    "pose_render_risk_high": False,
                },
            },
        )

        self.assertFalse(debug["applied"])
        self.assertEqual(debug["reason"], "pose_risk_low")
        self.assertAlmostEqual(debug["sample_opacity_scale"], 1.0)
        self.assertAlmostEqual(debug["match_opacity_scale"], 1.0)

    def test_medium_risk_softly_downweights_new_gaussians(self):
        debug = pose_render_init_weighting_decision(
            mode="risk_opacity_v1",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_risk_score": 0.0,
                    "pose_render_risk_score": 0.27,
                    "pose_render_risk_high": False,
                },
            },
            min_pose_risk=0.16,
            max_pose_risk=0.38,
            min_sample_opacity_scale=0.82,
            min_match_opacity_scale=0.92,
        )

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["reason"], "risk_opacity_scale")
        self.assertGreater(debug["sample_opacity_scale"], 0.82)
        self.assertLess(debug["sample_opacity_scale"], 1.0)
        self.assertGreater(debug["match_opacity_scale"], debug["sample_opacity_scale"])

    def test_high_risk_uses_max_downweight_without_blocking(self):
        debug = pose_render_init_weighting_decision(
            mode="risk_opacity_v1",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_render_risk_score": 0.70,
                    "pose_render_risk_high": True,
                },
            },
            min_sample_opacity_scale=0.80,
            min_match_opacity_scale=0.90,
        )

        self.assertTrue(debug["applied"])
        self.assertAlmostEqual(debug["risk_alpha"], 1.0)
        self.assertAlmostEqual(debug["sample_opacity_scale"], 0.80)
        self.assertAlmostEqual(debug["match_opacity_scale"], 0.90)

    def test_high_representation_value_protects_new_gaussians(self):
        debug = pose_render_init_weighting_decision(
            mode="risk_opacity_v1",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_render_risk_score": 0.28,
                    "stream_memory_representation_need": 0.82,
                    "novelty_value_score": 0.15,
                },
            },
            min_pose_risk=0.20,
            max_pose_risk=0.32,
            max_representation_value=0.65,
            max_novelty_value=0.65,
        )

        self.assertFalse(debug["applied"])
        self.assertEqual(debug["reason"], "representation_value_high")
        self.assertGreater(debug["risk_alpha"], 0.0)
        self.assertAlmostEqual(debug["representation_value_score"], 0.82)

    def test_high_new_view_value_protects_new_gaussians(self):
        debug = pose_render_init_weighting_decision(
            mode="risk_opacity_v1",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_render_risk_score": 0.28,
                    "representation_value_score": 0.20,
                    "new_view_event_score": 0.78,
                },
            },
            min_pose_risk=0.20,
            max_pose_risk=0.32,
            max_representation_value=0.65,
            max_novelty_value=0.65,
        )

        self.assertFalse(debug["applied"])
        self.assertEqual(debug["reason"], "novelty_value_high")
        self.assertGreater(debug["risk_alpha"], 0.0)
        self.assertAlmostEqual(debug["novelty_value_score"], 0.78)

    def test_adaptive_mode_preserves_high_novelty_when_risk_is_moderate(self):
        debug = pose_render_init_weighting_decision(
            mode="risk_opacity_adaptive_v2",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_render_risk_score": 0.27,
                    "new_view_event_score": 0.55,
                },
            },
            min_pose_risk=0.16,
            max_pose_risk=0.38,
            max_novelty_value=0.65,
            adaptive_novelty_threshold=0.45,
            adaptive_risk_override_alpha=0.65,
            adaptive_scene_event_count=96,
            adaptive_scene_pose_risk_mean=0.22,
            adaptive_scene_novelty_mean=0.41,
        )

        self.assertFalse(debug["applied"])
        self.assertEqual(debug["reason"], "adaptive_novelty_preserve")
        self.assertGreater(debug["risk_alpha"], 0.0)
        self.assertAlmostEqual(debug["adaptive_novelty_threshold"], 0.45)
        self.assertFalse(debug["adaptive_risk_override"])

    def test_adaptive_mode_keeps_scaling_when_risk_overrides_novelty(self):
        debug = pose_render_init_weighting_decision(
            mode="risk_opacity_adaptive_v2",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_render_risk_score": 0.34,
                    "new_view_event_score": 0.72,
                },
            },
            min_pose_risk=0.16,
            max_pose_risk=0.38,
            max_novelty_value=0.65,
            adaptive_novelty_threshold=0.45,
            adaptive_risk_override_alpha=0.65,
            adaptive_scene_event_count=32,
            adaptive_scene_pose_risk_mean=0.22,
            adaptive_scene_novelty_mean=0.41,
        )

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["reason"], "risk_opacity_adaptive_scale")
        self.assertTrue(debug["adaptive_risk_override"])
        self.assertLess(debug["sample_opacity_scale"], 1.0)

    def test_adaptive_mode_uses_gentler_band_after_scene_calibration(self):
        low_debug = pose_render_init_weighting_decision(
            mode="risk_opacity_adaptive_v2",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_render_risk_score": 0.18,
                    "new_view_event_score": 0.10,
                },
            },
            adaptive_scene_event_count=32,
            adaptive_scene_pose_risk_mean=0.22,
            adaptive_scene_novelty_mean=0.41,
        )
        high_debug = pose_render_init_weighting_decision(
            mode="risk_opacity_adaptive_v2",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_render_risk_score": 0.40,
                    "new_view_event_score": 0.10,
                },
            },
            adaptive_scene_event_count=32,
            adaptive_scene_pose_risk_mean=0.22,
            adaptive_scene_novelty_mean=0.41,
        )
        uncalibrated_debug = pose_render_init_weighting_decision(
            mode="risk_opacity_adaptive_v2",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_render_risk_score": 0.18,
                    "new_view_event_score": 0.10,
                },
            },
            adaptive_scene_event_count=32,
            adaptive_scene_pose_risk_mean=0.18,
            adaptive_scene_novelty_mean=0.41,
        )

        self.assertFalse(low_debug["applied"])
        self.assertEqual(low_debug["reason"], "pose_risk_low")
        self.assertTrue(low_debug["adaptive_scene_gentle"])
        self.assertAlmostEqual(low_debug["min_pose_risk"], 0.20)
        self.assertAlmostEqual(low_debug["max_pose_risk"], 0.32)
        self.assertTrue(high_debug["applied"])
        self.assertAlmostEqual(high_debug["sample_opacity_scale"], 0.88)
        self.assertAlmostEqual(high_debug["match_opacity_scale"], 0.95)
        self.assertTrue(uncalibrated_debug["applied"])
        self.assertFalse(uncalibrated_debug["adaptive_scene_gentle"])
        self.assertAlmostEqual(uncalibrated_debug["min_pose_risk"], 0.16)
        self.assertAlmostEqual(uncalibrated_debug["max_pose_risk"], 0.38)
        self.assertAlmostEqual(uncalibrated_debug["min_sample_opacity_scale"], 0.82)
        self.assertAlmostEqual(uncalibrated_debug["min_match_opacity_scale"], 0.92)

    def test_adaptive_scene_risk_alpha_cap_keeps_high_risk_novelty_scaling(self):
        debug = pose_render_init_weighting_decision(
            mode="risk_opacity_adaptive_v2",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_render_risk_score": 0.24,
                    "new_view_event_score": 0.55,
                },
            },
            adaptive_scene_event_count=32,
            adaptive_scene_pose_risk_mean=0.23,
            adaptive_scene_novelty_mean=0.46,
            adaptive_scene_risk_alpha_mean=0.47,
            adaptive_scene_max_risk_alpha_mean=0.42,
            adaptive_scene_high_uncertainty_bypass_risk_alpha_mean=2.0,
        )

        self.assertFalse(debug["adaptive_scene_gentle"])
        self.assertTrue(debug["applied"])
        self.assertEqual(debug["reason"], "risk_opacity_adaptive_scale")
        self.assertAlmostEqual(debug["effective_novelty_value_limit"], 0.65)

    def test_adaptive_scene_risk_alpha_cap_is_disabled_by_default(self):
        debug = pose_render_init_weighting_decision(
            mode="risk_opacity_adaptive_v2",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_render_risk_score": 0.24,
                    "new_view_event_score": 0.55,
                },
            },
            adaptive_scene_event_count=32,
            adaptive_scene_pose_risk_mean=0.23,
            adaptive_scene_novelty_mean=0.46,
            adaptive_scene_risk_alpha_mean=0.35,
        )

        self.assertTrue(debug["adaptive_scene_gentle"])
        self.assertFalse(debug["applied"])
        self.assertEqual(debug["reason"], "adaptive_novelty_preserve")

    def test_adaptive_scene_low_risk_bypasses_opacity_weighting(self):
        debug = pose_render_init_weighting_decision(
            mode="risk_opacity_adaptive_v2",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_render_risk_score": 0.25,
                    "new_view_event_score": 0.18,
                },
            },
            adaptive_scene_event_count=96,
            adaptive_scene_pose_risk_mean=0.10,
            adaptive_scene_novelty_mean=0.22,
            adaptive_scene_risk_alpha_mean=0.08,
            adaptive_scene_low_pose_risk_mean=0.14,
            adaptive_scene_low_novelty_mean=0.30,
        )

        self.assertFalse(debug["applied"])
        self.assertTrue(debug["adaptive_scene_low_risk_bypass"])
        self.assertEqual(debug["reason"], "adaptive_scene_low_risk_bypass")

    def test_adaptive_scene_low_risk_bypass_waits_for_mature_low_risk_window(self):
        early = pose_render_init_weighting_decision(
            mode="risk_opacity_adaptive_v2",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_render_risk_score": 0.25,
                    "new_view_event_score": 0.18,
                },
            },
            adaptive_scene_event_count=32,
            adaptive_scene_pose_risk_mean=0.10,
            adaptive_scene_novelty_mean=0.22,
            adaptive_scene_risk_alpha_mean=0.08,
            adaptive_scene_low_pose_risk_mean=0.14,
            adaptive_scene_low_novelty_mean=0.30,
            adaptive_scene_low_risk_min_events=64,
        )
        mature = pose_render_init_weighting_decision(
            mode="risk_opacity_adaptive_v2",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_render_risk_score": 0.25,
                    "new_view_event_score": 0.18,
                },
            },
            adaptive_scene_event_count=96,
            adaptive_scene_pose_risk_mean=0.10,
            adaptive_scene_novelty_mean=0.22,
            adaptive_scene_risk_alpha_mean=0.08,
            adaptive_scene_low_pose_risk_mean=0.14,
            adaptive_scene_low_novelty_mean=0.30,
            adaptive_scene_low_risk_min_events=64,
        )

        self.assertFalse(early["adaptive_scene_low_risk_bypass"])
        self.assertTrue(early["applied"])
        self.assertTrue(mature["adaptive_scene_low_risk_bypass"])
        self.assertFalse(mature["applied"])
        self.assertEqual(mature["adaptive_scene_low_risk_min_events"], 64)

    def test_adaptive_scene_stable_low_risk_bypasses_before_mature_window(self):
        debug = pose_render_init_weighting_decision(
            mode="risk_opacity_adaptive_v2",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_render_risk_score": 0.25,
                    "new_view_event_score": 0.08,
                },
            },
            adaptive_scene_event_count=32,
            adaptive_scene_pose_risk_mean=0.04,
            adaptive_scene_novelty_mean=0.09,
            adaptive_scene_risk_alpha_mean=0.04,
            adaptive_scene_low_pose_risk_mean=0.14,
            adaptive_scene_low_novelty_mean=0.30,
            adaptive_scene_low_risk_min_events=64,
            adaptive_scene_stable_low_risk_min_events=32,
            adaptive_scene_stable_low_pose_risk_mean=0.06,
            adaptive_scene_stable_low_novelty_mean=0.12,
        )

        self.assertTrue(debug["adaptive_scene_stable_low_risk_bypass"])
        self.assertTrue(debug["adaptive_scene_low_risk_bypass"])
        self.assertFalse(debug["applied"])
        self.assertEqual(debug["reason"], "adaptive_scene_low_risk_bypass")

    def test_adaptive_scene_stable_low_risk_latch_keeps_bypass_active(self):
        debug = pose_render_init_weighting_decision(
            mode="risk_opacity_adaptive_v2",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_render_risk_score": 0.28,
                    "new_view_event_score": 0.22,
                },
            },
            adaptive_scene_event_count=40,
            adaptive_scene_pose_risk_mean=0.08,
            adaptive_scene_novelty_mean=0.18,
            adaptive_scene_risk_alpha_mean=0.12,
            adaptive_scene_low_pose_risk_mean=0.14,
            adaptive_scene_low_novelty_mean=0.30,
            adaptive_scene_low_risk_min_events=64,
            adaptive_scene_stable_low_risk_min_events=16,
            adaptive_scene_stable_low_pose_risk_mean=0.02,
            adaptive_scene_stable_low_novelty_mean=0.05,
            adaptive_scene_stable_low_risk_latched=True,
        )

        self.assertTrue(debug["adaptive_scene_stable_low_risk_latched"])
        self.assertFalse(debug["adaptive_scene_stable_low_risk_bypass"])
        self.assertTrue(debug["adaptive_scene_low_risk_bypass"])
        self.assertFalse(debug["applied"])

    def test_adaptive_scene_optional_bypass_is_disabled_by_default(self):
        debug = pose_render_init_weighting_decision(
            mode="risk_opacity_adaptive_v2",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_render_risk_score": 0.25,
                    "new_view_event_score": 0.18,
                },
            },
            adaptive_scene_event_count=32,
            adaptive_scene_pose_risk_mean=0.10,
            adaptive_scene_novelty_mean=0.22,
            adaptive_scene_risk_alpha_mean=0.08,
        )

        self.assertFalse(debug["adaptive_scene_low_risk_bypass"])
        self.assertTrue(debug["applied"])
        self.assertEqual(debug["reason"], "risk_opacity_adaptive_scale")

    def test_adaptive_scene_high_uncertainty_novelty_bypasses_weighting(self):
        debug = pose_render_init_weighting_decision(
            mode="risk_opacity_adaptive_v2",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_render_risk_score": 0.31,
                    "new_view_event_score": 0.55,
                },
            },
            adaptive_scene_event_count=32,
            adaptive_scene_pose_risk_mean=0.23,
            adaptive_scene_novelty_mean=0.46,
            adaptive_scene_risk_alpha_mean=0.47,
            adaptive_scene_high_uncertainty_bypass_risk_alpha_mean=0.44,
            adaptive_scene_high_uncertainty_bypass_novelty_mean=0.44,
        )

        self.assertFalse(debug["applied"])
        self.assertTrue(debug["adaptive_scene_high_uncertainty_bypass"])
        self.assertEqual(debug["reason"], "adaptive_scene_high_uncertainty_bypass")

    def test_adaptive_scene_high_uncertainty_bypass_is_disabled_by_default(self):
        debug = pose_render_init_weighting_decision(
            mode="risk_opacity_adaptive_v2",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_render_risk_score": 0.31,
                    "new_view_event_score": 0.55,
                },
            },
            adaptive_scene_event_count=32,
            adaptive_scene_pose_risk_mean=0.23,
            adaptive_scene_novelty_mean=0.46,
            adaptive_scene_risk_alpha_mean=0.47,
        )

        self.assertFalse(debug["adaptive_scene_high_uncertainty_bypass"])
        self.assertTrue(debug["adaptive_scene_gentle"])

    def test_scene_model_records_base_and_effective_adaptive_thresholds(self):
        source = (
            Path(__file__)
            .resolve()
            .parents[1]
            .joinpath("scene", "scene_model.py")
            .read_text(encoding="utf-8")
        )

        self.assertIn(
            '"min_pose_risk": self.pose_render_init_weighting_min_pose_risk',
            source,
        )
        self.assertIn('"effective_min_pose_risk_sum": 0.0', source)
        self.assertIn('"effective_min_sample_opacity_scale_sum": 0.0', source)
        self.assertIn('debug.get("min_sample_opacity_scale", 1.0)', source)

    def test_args_exposes_adaptive_init_weighting_mode(self):
        source = (
            Path(__file__)
            .resolve()
            .parents[1]
            .joinpath("args.py")
            .read_text(encoding="utf-8")
        )

        self.assertIn("'risk_opacity_adaptive_v2'", source)
        self.assertIn("paper_aligned_pose_render_init_weighting_adaptive_novelty_threshold", source)
        self.assertIn("paper_aligned_pose_render_init_weighting_adaptive_risk_override_alpha", source)
        self.assertIn("paper_aligned_pose_render_init_weighting_adaptive_min_pose_risk", source)
        self.assertIn("paper_aligned_pose_render_init_weighting_adaptive_min_sample_opacity_scale", source)
        self.assertIn("paper_aligned_pose_render_init_weighting_adaptive_scene_min_events", source)
        self.assertIn("paper_aligned_pose_render_init_weighting_adaptive_scene_gentle_pose_risk_mean", source)
        self.assertIn("paper_aligned_pose_render_init_weighting_adaptive_scene_max_risk_alpha_mean", source)
        self.assertIn("paper_aligned_pose_render_init_weighting_adaptive_scene_low_pose_risk_mean", source)
        self.assertIn("paper_aligned_pose_render_init_weighting_adaptive_scene_low_risk_min_events", source)
        self.assertIn("paper_aligned_pose_render_init_weighting_adaptive_scene_stable_low_risk_min_events", source)
        self.assertIn("paper_aligned_pose_render_init_weighting_adaptive_scene_high_uncertainty_bypass_risk_alpha_mean", source)

    def test_scene_model_records_adaptive_init_weighting_reasons(self):
        source = (
            Path(__file__)
            .resolve()
            .parents[1]
            .joinpath("scene", "scene_model.py")
            .read_text(encoding="utf-8")
        )

        self.assertIn('"adaptive_novelty_preserve": 0', source)
        self.assertIn('"risk_opacity_adaptive_scale": 0', source)
        self.assertIn("adaptive_novelty_threshold=", source)
        self.assertIn("adaptive_risk_override_alpha=", source)
        self.assertIn("adaptive_min_pose_risk=", source)
        self.assertIn("adaptive_min_sample_opacity_scale=", source)
        self.assertIn("adaptive_scene_event_count=", source)
        self.assertIn("adaptive_scene_pose_risk_mean=", source)
        self.assertIn("adaptive_scene_risk_alpha_mean=", source)
        self.assertIn("adaptive_scene_low_risk_min_events=", source)
        self.assertIn("adaptive_scene_stable_low_risk_min_events=", source)
        self.assertIn("adaptive_scene_stable_low_risk_latched=", source)
        self.assertIn('"adaptive_scene_low_risk_bypass": 0', source)
        self.assertIn('"adaptive_scene_stable_low_risk_bypass": 0', source)
        self.assertIn('"adaptive_scene_stable_low_risk_latched": 0', source)
        self.assertIn('"early_trace": []', source)
        self.assertIn('"adaptive_scene_high_uncertainty_bypass": 0', source)

    def test_scene_model_applies_init_weighting_before_inverse_sigmoid(self):
        source = (
            Path(__file__)
            .resolve()
            .parents[1]
            .joinpath("scene", "scene_model.py")
            .read_text(encoding="utf-8")
        )

        opacity_init = source.index("opacities = torch.ones")
        decision = source.index("pose_render_init_weighting_decision", opacity_init)
        sample_scale = source.index("sample_opacity_scale", decision)
        inverse = source.index("opacities = inverse_sigmoid(opacities)", sample_scale)
        self.assertLess(opacity_init, decision)
        self.assertLess(decision, sample_scale)
        self.assertLess(sample_scale, inverse)


if __name__ == "__main__":
    unittest.main()
