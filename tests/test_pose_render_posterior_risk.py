import unittest
from pathlib import Path

from scene.pose_render_posterior_risk import (
    augment_pose_render_payload_with_posterior_risk,
    pose_render_posterior_risk_info,
)


class PoseRenderPosteriorRiskTests(unittest.TestCase):
    def test_strong_pose_support_keeps_risk_low(self):
        info = pose_render_posterior_risk_info(
            pose_debug={
                "num_2d3d_correspondences": 1800,
                "num_pnp_inliers": 1500,
                "num_miniba_inliers": 1450,
            },
            viewpoint_scores={
                "inlier_grid_coverage": 0.94,
                "inlier_grid_entropy": 0.92,
                "support_concentration": 0.08,
            },
            min_num_inliers=100,
            num_matches=2200,
            pose_inliers=1450,
            semantic_pose_risk=0.04,
            semantic_q=0.96,
            semantic_b_r=0.95,
        )

        self.assertLess(info["pose_render_posterior_risk_score"], 0.20)
        self.assertLess(info["pose_render_risk_score"], 0.20)
        self.assertFalse(info["pose_render_risk_high"])

    def test_weak_pose_init_support_raises_render_risk(self):
        info = pose_render_posterior_risk_info(
            pose_debug={
                "num_2d3d_correspondences": 900,
                "num_pnp_inliers": 120,
                "num_miniba_inliers": 86,
            },
            viewpoint_scores={
                "inlier_grid_coverage": 0.48,
                "inlier_grid_entropy": 0.55,
                "support_concentration": 0.38,
            },
            min_num_inliers=100,
            num_matches=920,
            pose_inliers=86,
            semantic_pose_risk=0.03,
            semantic_q=0.92,
            semantic_b_r=0.94,
        )

        self.assertGreater(info["pose_render_posterior_risk_score"], 0.55)
        self.assertGreater(info["pose_render_risk_score"], 0.55)
        self.assertTrue(info["pose_render_risk_high"])
        self.assertEqual(info["pose_render_risk_reason"], "posterior_pose_support_weak")

    def test_semantic_pose_risk_is_preserved(self):
        info = pose_render_posterior_risk_info(
            pose_debug={
                "num_2d3d_correspondences": 2000,
                "num_pnp_inliers": 1900,
                "num_miniba_inliers": 1800,
            },
            viewpoint_scores={"inlier_grid_coverage": 0.95, "support_concentration": 0.05},
            min_num_inliers=100,
            num_matches=2100,
            pose_inliers=1800,
            semantic_pose_risk=0.72,
            semantic_q=0.35,
            semantic_b_r=0.20,
        )

        self.assertAlmostEqual(info["pose_render_risk_score"], 0.72)
        self.assertTrue(info["pose_render_risk_high"])
        self.assertEqual(info["pose_render_risk_reason"], "semantic_pose_risk_high")

    def test_payload_augmentation_promotes_posterior_risk_to_legacy_field(self):
        payload = {
            "pose_risk_score": 0.0,
            "pose_risk_high": False,
            "semantic_Q_t": 0.92,
            "semantic_B_R_t": 0.94,
            "num_matches": 920,
            "pose_inliers": 86,
        }

        out = augment_pose_render_payload_with_posterior_risk(
            payload,
            pose_debug={
                "num_2d3d_correspondences": 900,
                "num_pnp_inliers": 120,
                "num_miniba_inliers": 86,
            },
            viewpoint_scores={
                "inlier_grid_coverage": 0.48,
                "inlier_grid_entropy": 0.55,
                "support_concentration": 0.38,
            },
            min_num_inliers=100,
        )

        self.assertIs(out, payload)
        self.assertGreater(out["pose_render_risk_score"], 0.55)
        self.assertAlmostEqual(out["pose_risk_score"], out["pose_render_risk_score"])
        self.assertTrue(out["pose_risk_high"])
        self.assertTrue(out["pose_render_risk_high"])

    def test_training_loop_bridges_posterior_risk_into_payload(self):
        train_source = (
            Path(__file__).resolve().parents[1].joinpath("train.py").read_text(encoding="utf-8")
        )

        self.assertIn("augment_pose_render_payload_with_posterior_risk", train_source)
        self.assertIn("pose_render_posterior_risk", train_source)


if __name__ == "__main__":
    unittest.main()
