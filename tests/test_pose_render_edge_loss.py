import unittest

import torch

from scene.pose_render_edge_loss import (
    gradient_l1_edge_loss,
    pose_render_edge_loss_enabled,
    pose_render_gradient_loss,
    pose_render_pose_gate_debug,
)


class PoseRenderEdgeLossTests(unittest.TestCase):
    def test_edge_loss_is_explicit_pose_safe_and_train_only(self):
        self.assertFalse(
            pose_render_edge_loss_enabled(
                "off", "pose_safe_streaming_memory_v1", {"is_test": False}, 0.03
            )
        )
        self.assertFalse(
            pose_render_edge_loss_enabled("gradient_v1", "off", {"is_test": False}, 0.03)
        )
        self.assertFalse(
            pose_render_edge_loss_enabled(
                "gradient_v1", "pose_safe_streaming_memory_v1", {"is_test": True}, 0.03
            )
        )
        self.assertFalse(
            pose_render_edge_loss_enabled(
                "gradient_v1", "pose_safe_streaming_memory_v1", {"is_test": False}, 0.0
            )
        )
        self.assertTrue(
            pose_render_edge_loss_enabled(
                "gradient_v1", "pose_safe_streaming_memory_v1", {"is_test": False}, 0.03
            )
        )

    def test_pose_safe_gate_uses_runtime_pose_risk(self):
        safe_info = {
            "is_test": False,
            "_paper_aligned_pose_render_coupling": {
                "direct_keyframe_finalized": True,
                "pose_risk_score": 0.25,
                "pose_risk_high": False,
                "finalize_pose_risk_reference": False,
                "utility_drift_risk": 0.20,
                "pose_support_score": 0.70,
                "match_support_score": 0.65,
            },
        }
        risky_info = {
            "is_test": False,
            "_paper_aligned_pose_render_coupling": {
                "direct_keyframe_finalized": True,
                "pose_risk_score": 0.62,
                "pose_risk_high": True,
                "finalize_pose_risk_reference": True,
                "utility_drift_risk": 0.20,
                "pose_support_score": 0.70,
                "match_support_score": 0.65,
            },
        }

        self.assertTrue(
            pose_render_edge_loss_enabled(
                "gradient_pose_safe_v1",
                "pose_safe_streaming_memory_v1",
                safe_info,
                0.01,
            )
        )
        self.assertTrue(pose_render_pose_gate_debug(safe_info)["pose_gate_passed"])
        self.assertFalse(
            pose_render_edge_loss_enabled(
                "gradient_pose_safe_v1",
                "pose_safe_streaming_memory_v1",
                risky_info,
                0.01,
            )
        )
        self.assertFalse(
            pose_render_edge_loss_enabled(
                "gradient_pose_safe_v1",
                "pose_safe_streaming_memory_v1",
                {"is_test": False},
                0.01,
            )
        )
        self.assertEqual(
            pose_render_pose_gate_debug(risky_info)["pose_gate_reason"],
            "pose_risk_high",
        )

    def test_pose_safe_gate_prefers_posterior_pose_render_risk(self):
        info = {
            "is_test": False,
            "_paper_aligned_pose_render_coupling": {
                "direct_keyframe_finalized": True,
                "pose_risk_score": 0.0,
                "pose_render_risk_score": 0.72,
                "pose_render_risk_high": True,
                "pose_render_risk_reason": "posterior_pose_support_weak",
                "utility_drift_risk": 0.10,
                "pose_support_score": 0.90,
                "match_support_score": 0.90,
            },
        }

        debug = pose_render_pose_gate_debug(info)

        self.assertFalse(debug["pose_gate_passed"])
        self.assertAlmostEqual(debug["pose_risk_score"], 0.72)
        self.assertEqual(debug["pose_gate_reason"], "pose_risk_high")

    def test_gradient_loss_zero_for_identical_images(self):
        image = torch.rand(3, 8, 8)
        self.assertAlmostEqual(float(gradient_l1_edge_loss(image, image)), 0.0)

    def test_gradient_loss_is_positive_and_backpropagates(self):
        render = torch.zeros(3, 8, 8, requires_grad=True)
        target = torch.zeros(3, 8, 8)
        target[:, 2:6, 2:6] = 1.0

        weighted_loss, debug = pose_render_gradient_loss(
            render,
            target,
            mode="gradient_v1",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={"is_test": False},
            weight=0.03,
        )
        weighted_loss.backward()

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["reason"], "gradient_edge_l1")
        self.assertGreater(debug["raw_loss"], 0.0)
        self.assertGreater(float(render.grad.abs().sum()), 0.0)

    def test_pose_safe_gradient_loss_is_gated_before_backward(self):
        render = torch.zeros(3, 8, 8, requires_grad=True)
        target = torch.zeros(3, 8, 8)
        target[:, 2:6, 2:6] = 1.0

        weighted_loss, debug = pose_render_gradient_loss(
            render,
            target,
            mode="gradient_pose_safe_v1",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "direct_keyframe_finalized": True,
                    "pose_risk_score": 0.80,
                    "pose_risk_high": False,
                    "finalize_pose_risk_reference": False,
                    "utility_drift_risk": 0.20,
                    "pose_support_score": 0.70,
                    "match_support_score": 0.65,
                },
            },
            weight=0.01,
        )

        self.assertFalse(debug["applied"])
        self.assertEqual(debug["reason"], "pose_risk_score_high")
        self.assertEqual(float(weighted_loss), 0.0)

    def test_adaptive_gradient_loss_scales_down_risky_large_residual_frames(self):
        render = torch.zeros(3, 8, 8, requires_grad=True)
        target = torch.zeros(3, 8, 8)
        target[:, 2:6, 2:6] = 1.0

        weighted_loss, debug = pose_render_gradient_loss(
            render,
            target,
            mode="gradient_pose_adaptive_v1",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "direct_keyframe_finalized": True,
                    "pose_render_risk_score": 0.44,
                    "pose_render_risk_high": True,
                    "finalize_pose_risk_reference": False,
                    "utility_drift_risk": 0.55,
                    "pose_support_score": 0.46,
                    "match_support_score": 0.47,
                },
            },
            weight=0.01,
            min_weight_scale=0.20,
            target_raw_loss=0.02,
        )
        weighted_loss.backward()

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["reason"], "gradient_edge_l1_adaptive")
        self.assertAlmostEqual(debug["base_weight"], 0.01)
        self.assertLess(debug["weight"], 0.01)
        self.assertGreaterEqual(debug["adaptive_weight_scale"], 0.20)
        self.assertLess(debug["residual_clip_scale"], 1.0)
        self.assertGreater(float(render.grad.abs().sum()), 0.0)

    def test_adaptive_gradient_loss_preserves_confident_low_residual_frames(self):
        render = torch.zeros(3, 8, 8, requires_grad=True)
        target = torch.zeros(3, 8, 8)
        target[:, :, 4:] = 0.02

        _, debug = pose_render_gradient_loss(
            render,
            target,
            mode="gradient_pose_adaptive_v1",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "direct_keyframe_finalized": True,
                    "pose_render_risk_score": 0.04,
                    "pose_render_risk_high": False,
                    "finalize_pose_risk_reference": False,
                    "utility_drift_risk": 0.0,
                    "pose_support_score": 0.96,
                    "match_support_score": 0.94,
                    "active_memory_stable_pose_reference": True,
                },
            },
            weight=0.01,
            min_weight_scale=0.20,
            target_raw_loss=0.02,
        )

        self.assertTrue(debug["applied"])
        self.assertGreater(debug["weight"], 0.0085)
        self.assertAlmostEqual(debug["residual_clip_scale"], 1.0)
        self.assertLessEqual(debug["adaptive_weight_scale"], 1.0)

    def test_adaptive_gradient_loss_keeps_base_weight_inside_risk_free_band(self):
        render = torch.zeros(3, 8, 8, requires_grad=True)
        target = torch.zeros(3, 8, 8)
        target[:, :, 4:] = 0.02

        _, debug = pose_render_gradient_loss(
            render,
            target,
            mode="gradient_pose_adaptive_v1",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "direct_keyframe_finalized": True,
                    "pose_render_risk_score": 0.20,
                    "pose_render_risk_high": False,
                    "finalize_pose_risk_reference": False,
                    "utility_drift_risk": 0.0,
                    "pose_support_score": 0.96,
                    "match_support_score": 0.94,
                    "active_memory_stable_pose_reference": True,
                },
            },
            weight=0.01,
            min_weight_scale=0.20,
            target_raw_loss=0.02,
            risk_free_threshold=0.25,
        )

        self.assertTrue(debug["applied"])
        self.assertAlmostEqual(debug["risk_weight_scale"], 1.0)
        self.assertAlmostEqual(debug["weight"], 0.01)


if __name__ == "__main__":
    unittest.main()
