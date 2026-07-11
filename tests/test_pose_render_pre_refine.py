import unittest
from pathlib import Path

from scene.pose_render_pre_refine import pose_render_pre_refine_decision


class PoseRenderPreRefineTests(unittest.TestCase):
    def test_off_mode_does_not_run(self):
        debug = pose_render_pre_refine_decision(
            mode="off",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={"is_test": False},
            existing_gaussians=50000,
            existing_keyframes=20,
        )

        self.assertFalse(debug["run_pre_refine"])
        self.assertEqual(debug["reason"], "off")

    def test_medium_posterior_risk_runs_after_map_matures(self):
        debug = pose_render_pre_refine_decision(
            mode="pose_only_v1",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "direct_keyframe_finalized": True,
                    "pose_risk_score": 0.0,
                    "pose_render_risk_score": 0.22,
                    "pose_render_risk_high": False,
                },
            },
            existing_gaussians=50000,
            existing_keyframes=20,
            min_pose_risk=0.08,
            max_pose_risk=0.38,
        )

        self.assertTrue(debug["run_pre_refine"])
        self.assertEqual(debug["reason"], "pose_only_pre_refine")
        self.assertAlmostEqual(debug["pose_risk_score"], 0.22)

    def test_low_risk_and_high_risk_do_not_run(self):
        low = pose_render_pre_refine_decision(
            mode="pose_only_v1",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_render_risk_score": 0.03,
                    "pose_render_risk_high": False,
                },
            },
            existing_gaussians=50000,
            existing_keyframes=20,
        )
        high = pose_render_pre_refine_decision(
            mode="pose_only_v1",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_render_risk_score": 0.62,
                    "pose_render_risk_high": True,
                },
            },
            existing_gaussians=50000,
            existing_keyframes=20,
        )

        self.assertFalse(low["run_pre_refine"])
        self.assertEqual(low["reason"], "pose_risk_low")
        self.assertFalse(high["run_pre_refine"])
        self.assertEqual(high["reason"], "pose_risk_too_high")

    def test_immature_map_does_not_run(self):
        debug = pose_render_pre_refine_decision(
            mode="pose_only_v1",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_coupling": {
                    "pose_render_risk_score": 0.22,
                    "pose_render_risk_high": False,
                },
            },
            existing_gaussians=3000,
            existing_keyframes=20,
        )

        self.assertFalse(debug["run_pre_refine"])
        self.assertEqual(debug["reason"], "map_not_mature_gaussians")

    def test_training_loop_pre_refines_before_adding_new_gaussians(self):
        root = Path(__file__).resolve().parents[1]
        train_source = root.joinpath("train.py").read_text(encoding="utf-8")
        scene_model_source = root.joinpath("scene", "scene_model.py").read_text(
            encoding="utf-8"
        )

        pre_refine_call = train_source.index("scene_model.pose_render_pre_refine_keyframe")
        gaussian_call = train_source.index("scene_model.add_new_gaussians()", pre_refine_call)
        self.assertLess(pre_refine_call, gaussian_call)
        self.assertIn("pose_render_pre_refine_decision", scene_model_source)
        self.assertIn("pose_render_pre_refine", scene_model_source)

    def test_scene_model_pre_refine_is_strictly_pose_only(self):
        scene_model_source = (
            Path(__file__)
            .resolve()
            .parents[1]
            .joinpath("scene", "scene_model.py")
            .read_text(encoding="utf-8")
        )

        self.assertIn('pose_params = {"rW2C", "tW2C"}', scene_model_source)
        self.assertIn('if name not in pose_params:', scene_model_source)
        self.assertIn('param_dict["val"].grad = None', scene_model_source)
        self.assertIn('keyframe.optimizer.step()', scene_model_source)
        self.assertIn('render_pkg["mainGaussID"][0] >= 0', scene_model_source)


if __name__ == "__main__":
    unittest.main()
