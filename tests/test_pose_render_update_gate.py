import inspect
import unittest
from pathlib import Path

from scene.pose_render_update_gate import pose_render_update_gate_decision


class PoseRenderUpdateGateTests(unittest.TestCase):
    def test_off_mode_preserves_gaussian_update(self):
        debug = pose_render_update_gate_decision(
            mode="off",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={"is_test": False},
        )
        self.assertTrue(debug["allow_gaussian_update"])
        self.assertEqual(debug["reason"], "off")

    def test_stable_pose_confidence_allows_gaussian_update(self):
        debug = pose_render_update_gate_decision(
            mode="pose_confidence_v1",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_risk_score": 0.10,
                    "utility_drift_risk": 0.10,
                    "pose_support_score": 0.90,
                    "match_support_score": 0.80,
                },
            },
        )
        self.assertTrue(debug["allow_gaussian_update"])
        self.assertEqual(debug["reason"], "pose_confidence_ok")
        self.assertGreater(debug["confidence"], 0.35)

    def test_high_pose_risk_blocks_gaussian_update_but_not_pose_step(self):
        debug = pose_render_update_gate_decision(
            mode="pose_confidence_v1",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_risk_score": 0.70,
                    "pose_risk_high": True,
                    "utility_drift_risk": 0.10,
                    "pose_support_score": 0.90,
                    "match_support_score": 0.80,
                },
            },
            min_confidence=0.35,
        )
        self.assertFalse(debug["allow_gaussian_update"])
        self.assertEqual(debug["reason"], "pose_risk_high")

    def test_posterior_pose_render_risk_overrides_zero_semantic_risk(self):
        debug = pose_render_update_gate_decision(
            mode="pose_confidence_v1",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_risk_score": 0.0,
                    "pose_render_risk_score": 0.68,
                    "pose_render_risk_high": True,
                    "pose_render_risk_reason": "posterior_pose_support_weak",
                    "utility_drift_risk": 0.20,
                },
            },
        )
        self.assertFalse(debug["allow_gaussian_update"])
        self.assertEqual(debug["reason"], "pose_risk_high")
        self.assertAlmostEqual(debug["pose_risk_score"], 0.68)

    def test_pose_risk_above_runtime_threshold_blocks_even_without_high_flag(self):
        debug = pose_render_update_gate_decision(
            mode="pose_confidence_v1",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_risk_score": 0.0,
                    "pose_render_risk_score": 0.36,
                    "pose_render_risk_high": False,
                    "utility_drift_risk": 0.05,
                    "pose_support_score": 0.98,
                    "match_support_score": 0.98,
                },
            },
            max_pose_risk=0.30,
        )

        self.assertFalse(debug["allow_gaussian_update"])
        self.assertEqual(debug["reason"], "pose_risk_score_high")

    def test_soft_pose_confidence_scales_gaussian_update_instead_of_blocking(self):
        debug = pose_render_update_gate_decision(
            mode="pose_confidence_soft_v1",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_render_risk_score": 0.30,
                    "pose_render_risk_high": False,
                    "utility_drift_risk": 0.05,
                    "pose_support_score": 0.85,
                    "match_support_score": 0.86,
                },
            },
            min_confidence=0.75,
            max_pose_risk=0.35,
            soft_min_scale=0.65,
        )

        self.assertTrue(debug["allow_gaussian_update"])
        self.assertEqual(debug["reason"], "pose_confidence_soft_scaled")
        self.assertGreaterEqual(debug["gaussian_update_scale"], 0.65)
        self.assertLess(debug["gaussian_update_scale"], 1.0)

    def test_psnr_health_mode_caps_gaussian_update_for_high_structure_raw_residuals(self):
        debug = pose_render_update_gate_decision(
            mode="pose_confidence_soft_psnr_health_v1",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_render_risk_score": 0.10,
                    "pose_render_risk_high": False,
                    "utility_drift_risk": 0.05,
                    "pose_support_score": 0.92,
                    "match_support_score": 0.90,
                },
                "_paper_aligned_pose_render_psnr_loss": {
                    "applied": True,
                    "raw_loss": 0.0058,
                    "health_gate_score": 0.86,
                },
            },
            min_confidence=0.35,
            max_pose_risk=0.35,
            soft_min_scale=0.65,
            psnr_health_min_score=0.84,
            psnr_health_min_raw_loss=0.0045,
            psnr_health_scale=0.55,
        )

        self.assertTrue(debug["allow_gaussian_update"])
        self.assertEqual(debug["reason"], "psnr_health_alias_scaled")
        self.assertAlmostEqual(debug["gaussian_update_scale"], 0.55)
        self.assertAlmostEqual(debug["psnr_health_score"], 0.86)
        self.assertAlmostEqual(debug["psnr_health_raw_loss"], 0.0058)

    def test_psnr_health_mode_preserves_update_when_raw_residual_is_low(self):
        debug = pose_render_update_gate_decision(
            mode="pose_confidence_soft_psnr_health_v1",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_render_risk_score": 0.10,
                    "utility_drift_risk": 0.05,
                    "pose_support_score": 0.92,
                    "match_support_score": 0.90,
                },
                "_paper_aligned_pose_render_psnr_loss": {
                    "applied": True,
                    "raw_loss": 0.0034,
                    "health_gate_score": 0.90,
                },
            },
            min_confidence=0.35,
            max_pose_risk=0.35,
            soft_min_scale=0.65,
            psnr_health_min_score=0.84,
            psnr_health_min_raw_loss=0.0045,
            psnr_health_scale=0.55,
        )

        self.assertTrue(debug["allow_gaussian_update"])
        self.assertEqual(debug["reason"], "pose_confidence_ok")
        self.assertAlmostEqual(debug["gaussian_update_scale"], 1.0)

    def test_baseline_render_lock_policy_scales_without_direct_density(self):
        self.assertIn(
            "render_frame_policy",
            inspect.signature(pose_render_update_gate_decision).parameters,
        )
        debug = pose_render_update_gate_decision(
            mode="pose_confidence_soft_v1",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_render_risk_score": 0.30,
                    "pose_render_risk_high": False,
                    "utility_drift_risk": 0.05,
                    "pose_support_score": 0.85,
                    "match_support_score": 0.86,
                },
            },
            min_confidence=0.75,
            max_pose_risk=0.35,
            soft_min_scale=0.65,
        )

        self.assertTrue(debug["allow_gaussian_update"])
        self.assertEqual(debug["direct_density_mode"], "off")
        self.assertEqual(debug["render_frame_policy"], "baseline_keyframe_lock_v1")
        self.assertEqual(debug["reason"], "pose_confidence_soft_scaled")
        self.assertLess(debug["gaussian_update_scale"], 1.0)

    def test_missing_gate_allows_update_for_baseline_compatibility(self):
        debug = pose_render_update_gate_decision(
            mode="pose_confidence_v1",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={"is_test": False},
        )
        self.assertTrue(debug["allow_gaussian_update"])
        self.assertEqual(debug["reason"], "missing_gate_allow")

    def test_scene_model_scales_gaussian_gradients_before_optimizer_step(self):
        source = Path("scene/scene_model.py").read_text(encoding="utf-8")

        self.assertIn("gaussian_update_scale", source)
        self.assertIn("_apply_pose_render_gaussian_update_scale", source)
        self.assertIn("render_frame_policy=self.paper_aligned_render_frame_policy", source)
        self.assertIn("val.grad.mul_(scale)", source)


if __name__ == "__main__":
    unittest.main()
