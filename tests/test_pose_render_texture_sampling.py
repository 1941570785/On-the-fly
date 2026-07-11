from pathlib import Path
import unittest

import torch

from scene.pose_render_texture_sampling import (
    pose_render_texture_sampling_enabled,
    residual_edge_guided_sampling_probability,
)


class PoseRenderTextureSamplingTests(unittest.TestCase):
    def test_sampling_is_explicit_and_enabled_by_pose_safe_or_baseline_lock(self):
        self.assertFalse(
            pose_render_texture_sampling_enabled(
                "off", "pose_safe_streaming_memory_v1", {"is_test": False}
            )
        )
        self.assertFalse(
            pose_render_texture_sampling_enabled(
                "residual_edge_v1", "off", {"is_test": False}
            )
        )
        self.assertTrue(
            pose_render_texture_sampling_enabled(
                "residual_edge_v1",
                "off",
                {"is_test": False},
                render_frame_policy="baseline_keyframe_lock_v1",
            )
        )
        self.assertFalse(
            pose_render_texture_sampling_enabled(
                "residual_edge_v1", "pose_safe_streaming_memory_v1", {"is_test": True}
            )
        )
        self.assertTrue(
            pose_render_texture_sampling_enabled(
                "residual_edge_v1", "pose_safe_streaming_memory_v1", {"is_test": False}
            )
        )

    def test_residual_edge_sampling_preserves_budget_and_boosts_residual_region(self):
        base = torch.ones(8, 8)
        residual_edge = torch.ones(8, 8)
        residual_edge[2:6, 2:6] = 8.0

        guided, debug = residual_edge_guided_sampling_probability(
            base,
            residual_edge,
            mode="residual_edge_v1",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={"is_test": False},
            alpha=0.12,
            min_selectivity=0.10,
        )

        self.assertTrue(debug["applied"])
        self.assertAlmostEqual(float(guided.sum()), float(base.sum()), places=5)
        self.assertGreater(float(guided[2:6, 2:6].mean()), float(base[2:6, 2:6].mean()))
        self.assertLess(float(guided[:2, :2].mean()), float(base[:2, :2].mean()))

    def test_residual_edge_sampling_applies_under_baseline_render_lock_policy(self):
        base = torch.ones(8, 8)
        residual_edge = torch.ones(8, 8)
        residual_edge[2:6, 2:6] = 8.0

        guided, debug = residual_edge_guided_sampling_probability(
            base,
            residual_edge,
            mode="residual_edge_v1",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info={"is_test": False},
            alpha=0.12,
            min_selectivity=0.10,
        )

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["render_frame_policy"], "baseline_keyframe_lock_v1")
        self.assertAlmostEqual(float(guided.sum()), float(base.sum()), places=5)

    def test_response_guard_sampling_is_self_authorized_without_render_lock_policy(self):
        base = torch.ones(8, 8)
        residual_edge = torch.ones(8, 8)
        residual_edge[2:6, 2:6] = 8.0

        guided, debug = residual_edge_guided_sampling_probability(
            base,
            residual_edge,
            mode="residual_edge_response_guard_v2",
            direct_density_mode="off",
            render_frame_policy="off",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_texture_sampling_scene_guard": {
                    "disabled": False,
                    "events": 10,
                    "sampling_applied_ratio": 0.0,
                    "min_coverage_deficit": 0.05,
                },
                "_paper_aligned_pose_render_texture_sampling_coverage": {
                    "coverage_deficit": 0.20,
                },
            },
            alpha=0.12,
            min_selectivity=0.10,
        )

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["reason"], "residual_edge_guided")
        self.assertAlmostEqual(float(guided.sum()), float(base.sum()), places=5)

    def test_response_guard_v4_sampling_is_self_authorized_without_render_lock_policy(self):
        base = torch.ones(8, 8)
        residual_edge = torch.ones(8, 8)
        residual_edge[2:6, 2:6] = 8.0

        guided, debug = residual_edge_guided_sampling_probability(
            base,
            residual_edge,
            mode="residual_edge_response_guard_v4",
            direct_density_mode="off",
            render_frame_policy="off",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_texture_sampling_scene_guard": {
                    "disabled": False,
                    "coverage_bypass_latched": False,
                    "events": 10,
                    "sampling_applied_ratio": 0.0,
                    "min_coverage_deficit": 0.05,
                },
                "_paper_aligned_pose_render_texture_sampling_coverage": {
                    "coverage_deficit": 0.20,
                },
            },
            alpha=0.12,
            min_selectivity=0.10,
        )

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["reason"], "residual_edge_guided")
        self.assertAlmostEqual(float(guided.sum()), float(base.sum()), places=5)

    def test_response_guard_v4_blocks_sampling_when_high_coverage_bypass_is_latched(self):
        base = torch.ones(8, 8)
        residual_edge = torch.ones(8, 8)
        residual_edge[2:6, 2:6] = 8.0

        guided, debug = residual_edge_guided_sampling_probability(
            base,
            residual_edge,
            mode="residual_edge_response_guard_v4",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_texture_sampling_scene_guard": {
                    "disabled": False,
                    "coverage_bypass_latched": True,
                    "coverage_bypass_reason": "high_coverage_low_sampling_pressure",
                    "events": 40,
                    "sampling_applied_ratio": 0.0,
                    "min_coverage_deficit": 0.05,
                },
                "_paper_aligned_pose_render_texture_sampling_coverage": {
                    "coverage_deficit": 0.001,
                },
            },
            alpha=0.12,
            min_selectivity=0.10,
        )

        self.assertFalse(debug["applied"])
        self.assertTrue(debug["coverage_bypass_latched"])
        self.assertEqual(debug["reason"], "coverage_bypass_latched")
        self.assertTrue(torch.equal(guided, base))

    def test_non_dark_response_guard_blocks_sampling_for_fixed_black_dark_scene(self):
        base = torch.ones(8, 8)
        residual_edge = torch.ones(8, 8)
        residual_edge[2:6, 2:6] = 8.0

        guided, debug = residual_edge_guided_sampling_probability(
            base,
            residual_edge,
            mode="residual_edge_response_guard_non_dark_v5",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_training_background": {
                    "mode": "dark_scene_fixed_black_v1",
                    "dark_scene_decision": "fixed_black",
                    "dark_scene_mask_blocked": 0,
                },
                "_paper_aligned_pose_render_texture_sampling_scene_guard": {
                    "disabled": False,
                    "events": 40,
                    "sampling_applied_ratio": 0.0,
                    "min_coverage_deficit": 0.05,
                },
                "_paper_aligned_pose_render_texture_sampling_coverage": {
                    "coverage_deficit": 0.20,
                },
            },
            alpha=0.12,
            min_selectivity=0.10,
        )

        self.assertFalse(debug["applied"])
        self.assertEqual(debug["reason"], "dark_fixed_scene_bypass")
        self.assertTrue(torch.equal(guided, base))

    def test_mask_conservative_response_guard_bypasses_mask_blocked_scene(self):
        base = torch.ones(8, 8)
        residual_edge = torch.ones(8, 8)
        residual_edge[2:6, 2:6] = 8.0

        guided, debug = residual_edge_guided_sampling_probability(
            base,
            residual_edge,
            mode="residual_edge_response_guard_mask_conservative_v6",
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
                "_paper_aligned_pose_render_texture_sampling_scene_guard": {
                    "disabled": False,
                    "events": 40,
                    "sampling_applied_ratio": 0.0,
                    "min_coverage_deficit": 0.05,
                },
                "_paper_aligned_pose_render_texture_sampling_coverage": {
                    "coverage_deficit": 0.20,
                },
            },
            alpha=0.12,
            min_selectivity=0.10,
        )

        self.assertFalse(debug["applied"])
        self.assertEqual(debug["reason"], "mask_blocked_scene_bypass")
        self.assertEqual(debug["training_background_dark_scene_mask_blocked"], 128)
        self.assertTrue(torch.equal(guided, base))

    def test_flat_residual_edge_does_not_redistribute_sampling(self):
        base = torch.rand(8, 8)
        residual_edge = torch.ones(8, 8)

        guided, debug = residual_edge_guided_sampling_probability(
            base,
            residual_edge,
            mode="residual_edge_v1",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info={"is_test": False},
            alpha=0.12,
            min_selectivity=0.10,
        )

        self.assertFalse(debug["applied"])
        self.assertEqual(debug["reason"], "low_selectivity")
        self.assertTrue(torch.equal(guided, base))

    def test_response_guard_mode_blocks_sampling_when_scene_guard_is_disabled(self):
        base = torch.ones(8, 8)
        residual_edge = torch.ones(8, 8)
        residual_edge[2:6, 2:6] = 8.0

        guided, debug = residual_edge_guided_sampling_probability(
            base,
            residual_edge,
            mode="residual_edge_response_guard_v2",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_texture_sampling_scene_guard": {
                    "disabled": True,
                    "reason": "response_bad_ratio",
                },
            },
            alpha=0.12,
            min_selectivity=0.10,
        )

        self.assertFalse(debug["applied"])
        self.assertEqual(debug["reason"], "scene_guard_disabled")
        self.assertTrue(torch.equal(guided, base))

    def test_response_guard_mode_blocks_sampling_when_render_coverage_is_sufficient(self):
        base = torch.ones(8, 8)
        residual_edge = torch.ones(8, 8)
        residual_edge[2:6, 2:6] = 8.0

        guided, debug = residual_edge_guided_sampling_probability(
            base,
            residual_edge,
            mode="residual_edge_response_guard_v2",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_texture_sampling_scene_guard": {
                    "disabled": False,
                    "min_coverage_deficit": 0.05,
                },
                "_paper_aligned_pose_render_texture_sampling_coverage": {
                    "coverage_deficit": 0.01,
                },
            },
            alpha=0.12,
            min_selectivity=0.10,
        )

        self.assertFalse(debug["applied"])
        self.assertEqual(debug["reason"], "coverage_sufficient")
        self.assertTrue(torch.equal(guided, base))

    def test_response_guard_mode_allows_selective_sampling_before_scene_guard_trips(self):
        base = torch.ones(8, 8)
        residual_edge = torch.ones(8, 8)
        residual_edge[2:6, 2:6] = 8.0

        guided, debug = residual_edge_guided_sampling_probability(
            base,
            residual_edge,
            mode="residual_edge_response_guard_v2",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_texture_sampling_scene_guard": {
                    "disabled": False,
                    "min_coverage_deficit": 0.05,
                },
                "_paper_aligned_pose_render_texture_sampling_coverage": {
                    "coverage_deficit": 0.20,
                },
            },
            alpha=0.12,
            min_selectivity=0.10,
        )

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["reason"], "residual_edge_guided")
        self.assertAlmostEqual(float(guided.sum()), float(base.sum()), places=5)

    def test_response_guard_v2_ignores_later_scene_sampling_quota(self):
        base = torch.ones(8, 8)
        residual_edge = torch.ones(8, 8)
        residual_edge[2:6, 2:6] = 8.0

        guided, debug = residual_edge_guided_sampling_probability(
            base,
            residual_edge,
            mode="residual_edge_response_guard_v2",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_texture_sampling_scene_guard": {
                    "disabled": False,
                    "events": 40,
                    "applied": 28,
                    "sampling_applied_ratio": 0.70,
                    "min_coverage_deficit": 0.05,
                },
                "_paper_aligned_pose_render_texture_sampling_coverage": {
                    "coverage_deficit": 0.20,
                },
            },
            alpha=0.12,
            min_selectivity=0.10,
        )

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["reason"], "residual_edge_guided")
        self.assertAlmostEqual(float(guided.sum()), float(base.sum()), places=5)

    def test_response_guard_v2_ignores_scene_sampling_ratio(self):
        base = torch.ones(8, 8)
        residual_edge = torch.ones(8, 8)
        residual_edge[2:6, 2:6] = 8.0

        guided, debug = residual_edge_guided_sampling_probability(
            base,
            residual_edge,
            mode="residual_edge_response_guard_v2",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_texture_sampling_scene_guard": {
                    "disabled": False,
                    "events": 40,
                    "applied": 17,
                    "sampling_applied_ratio": 0.425,
                    "min_coverage_deficit": 0.05,
                },
                "_paper_aligned_pose_render_texture_sampling_coverage": {
                    "coverage_deficit": 0.20,
                },
            },
            alpha=0.12,
            min_selectivity=0.10,
        )

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["reason"], "residual_edge_guided")
        self.assertAlmostEqual(float(guided.sum()), float(base.sum()), places=5)

    def test_response_guard_v2_ignores_later_scene_low_response_latch(self):
        base = torch.ones(8, 8)
        residual_edge = torch.ones(8, 8)
        residual_edge[2:6, 2:6] = 8.0

        guided, debug = residual_edge_guided_sampling_probability(
            base,
            residual_edge,
            mode="residual_edge_response_guard_v2",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_scene_low_response": {
                    "disabled": True,
                    "reason": "scene_low_response",
                },
                "_paper_aligned_pose_render_texture_sampling_scene_guard": {
                    "disabled": False,
                    "events": 40,
                    "sampling_applied_ratio": 0.70,
                    "min_coverage_deficit": 0.05,
                },
                "_paper_aligned_pose_render_texture_sampling_coverage": {
                    "coverage_deficit": 0.20,
                },
            },
            alpha=0.12,
            min_selectivity=0.10,
        )

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["reason"], "residual_edge_guided")
        self.assertAlmostEqual(float(guided.sum()), float(base.sum()), places=5)

    def test_response_guard_v4_retains_scene_sampling_quota(self):
        base = torch.ones(8, 8)
        residual_edge = torch.ones(8, 8)
        residual_edge[2:6, 2:6] = 8.0

        guided, debug = residual_edge_guided_sampling_probability(
            base,
            residual_edge,
            mode="residual_edge_response_guard_v4",
            direct_density_mode="off",
            render_frame_policy="off",
            keyframe_info={
                "is_test": False,
                "_paper_aligned_pose_render_texture_sampling_scene_guard": {
                    "disabled": False,
                    "events": 40,
                    "applied": 28,
                    "sampling_applied_ratio": 0.70,
                    "min_coverage_deficit": 0.05,
                },
                "_paper_aligned_pose_render_texture_sampling_coverage": {
                    "coverage_deficit": 0.20,
                },
            },
            alpha=0.12,
            min_selectivity=0.10,
        )

        self.assertFalse(debug["applied"])
        self.assertEqual(debug["reason"], "scene_sampling_quota")
        self.assertTrue(torch.equal(guided, base))

    def test_scene_model_links_response_guard_sampling_to_render_response_verdicts(self):
        source = Path("scene/scene_model.py").read_text(encoding="utf-8")

        self.assertIn("TEXTURE_SAMPLING_RESPONSE_GUARD_MODE", source)
        self.assertIn("_pose_render_texture_sampling_scene_guard", source)
        self.assertIn("_update_pose_render_texture_sampling_response_guard", source)
        self.assertIn(
            "TEXTURE_SAMPLING_RESPONSE_GUARD_MODE,",
            source,
        )
        self.assertIn('"scene_guard_min_coverage_deficit": 0.08', source)
        self.assertIn(
            "_paper_aligned_pose_render_texture_sampling_response_evaluated",
            source,
        )
        self.assertIn("TEXTURE_SAMPLING_COVERAGE_KEY", source)
        sampling_block_start = source.index("if render_residual_edge is not None:")
        guard_index = source.index(
            "keyframe.info[TEXTURE_SAMPLING_SCENE_GUARD_KEY]",
        )
        sampling_index = source.index(
            "residual_edge_guided_sampling_probability(",
            sampling_block_start,
        )
        self.assertLess(guard_index, sampling_block_start)
        self.assertLess(guard_index, sampling_index)

    def test_scene_model_links_v4_high_coverage_bypass_to_response_tracking(self):
        source = Path("scene/scene_model.py").read_text(encoding="utf-8")

        self.assertIn("TEXTURE_SAMPLING_RESPONSE_GUARD_BYPASS_MODE", source)
        self.assertIn("coverage_bypass_latched", source)
        self.assertIn("_pose_render_response_tracking_bypassed", source)

    def test_scene_model_links_mask_conservative_sampling_to_response_tracking(self):
        source = Path("scene/scene_model.py").read_text(encoding="utf-8")

        self.assertIn("TEXTURE_SAMPLING_RESPONSE_GUARD_MASK_CONSERVATIVE_MODE", source)
        self.assertIn("_dark_scene_mask_blocked_scene_latched", source)


if __name__ == "__main__":
    unittest.main()
