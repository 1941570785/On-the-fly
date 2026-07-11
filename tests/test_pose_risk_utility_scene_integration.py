import inspect
import unittest

import torch

from scene.pose_risk_utility_admission import pose_review_acceptance
from scene.scene_model import SceneModel


class _ProbeScene:
    width = 4
    height = 4

    def __init__(self):
        self.xyz = torch.ones(1, 3)
        self.grad_enabled_during_render = None

    def render(self, width, height, view_matrix, scaling_modifier, bg):
        self.grad_enabled_during_render = torch.is_grad_enabled()
        main_gaussian = torch.full((1, height, width), -1, dtype=torch.int32)
        main_gaussian[:, :, : width // 2] = 0
        return {
            "render": torch.zeros(3, height, width),
            "mainGaussID": main_gaussian,
        }


class PoseRiskUtilitySceneIntegrationTests(unittest.TestCase):
    def test_probe_uses_no_grad_and_reports_render_coverage(self):
        scene = _ProbeScene()
        result = SceneModel.probe_pose_risk_utility(
            scene,
            image=torch.ones(3, 4, 4),
            Rt=torch.eye(4),
            mask=None,
            downsample=1,
        )

        self.assertFalse(scene.grad_enabled_during_render)
        self.assertAlmostEqual(result["render_coverage"], 0.5)
        self.assertAlmostEqual(result["coverage_deficit"], 0.5)
        self.assertGreaterEqual(result["residual_selectivity"], 0.0)

    def test_pose_review_accepts_improvement_within_pose_bounds(self):
        accepted, reason = pose_review_acceptance(
            start_loss=0.20,
            end_loss=0.18,
            rotation_delta_deg=0.4,
            translation_delta=0.01,
            max_rotation_delta_deg=1.5,
            max_translation_delta=0.05,
        )

        self.assertTrue(accepted)
        self.assertEqual(reason, "accepted")

    def test_pose_review_rejects_degradation_or_excessive_step(self):
        degraded = pose_review_acceptance(
            start_loss=0.20,
            end_loss=0.21,
            rotation_delta_deg=0.1,
            translation_delta=0.01,
            max_rotation_delta_deg=1.5,
            max_translation_delta=0.05,
        )
        excessive = pose_review_acceptance(
            start_loss=0.20,
            end_loss=0.18,
            rotation_delta_deg=2.0,
            translation_delta=0.01,
            max_rotation_delta_deg=1.5,
            max_translation_delta=0.05,
        )

        self.assertEqual(degraded, (False, "loss_degraded"))
        self.assertEqual(excessive, (False, "rotation_step_exceeded"))

    def test_scene_review_is_pose_only_bounded_and_restorable(self):
        source = inspect.getsource(SceneModel.review_pose_risk_keyframe)

        self.assertIn('pose_params = {"rW2C", "tW2C"}', source)
        self.assertIn("min(2,", source)
        self.assertIn("keyframe.set_Rt(start_Rt)", source)
        self.assertIn("pose_review_acceptance", source)


if __name__ == "__main__":
    unittest.main()
