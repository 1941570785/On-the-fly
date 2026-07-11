import inspect
import unittest
from pathlib import Path

import torch

from scene.pose_render_psnr_loss import (
    mean_squared_rgb_loss,
    target_valid_rgb_mask,
    pose_render_mse_loss,
    pose_render_mse_loss_enabled,
    pose_render_structure_alignment_score,
    pose_render_support_weight_scale,
)
from scene.scene_model import SceneModel


def _pose_safe_info():
    return {
        "is_test": False,
        "_paper_aligned_pose_render_coupling": {
            "direct_keyframe_finalized": True,
            "pose_render_risk_score": 0.18,
            "pose_render_risk_high": False,
            "finalize_pose_risk_reference": False,
            "utility_drift_risk": 0.20,
            "pose_support_score": 0.72,
            "match_support_score": 0.68,
        },
    }


def _pose_safe_info_with_raw_support(correspondences=None, final_inliers=None):
    info = _pose_safe_info()
    gate = info["_paper_aligned_pose_render_coupling"]
    if correspondences is not None:
        gate["pose_render_correspondence_count"] = correspondences
    if final_inliers is not None:
        gate["pose_render_final_pose_inliers"] = final_inliers
    return info


def _pose_safe_info_with_risk(risk):
    info = _pose_safe_info()
    gate = info["_paper_aligned_pose_render_coupling"]
    gate["pose_render_risk_score"] = risk
    gate["pose_risk_score"] = risk
    return info


class PoseRenderPsnrLossTests(unittest.TestCase):
    def test_mean_squared_rgb_loss_matches_psnr_mse_target(self):
        render = torch.zeros(3, 4, 4, requires_grad=True)
        target = torch.full((3, 4, 4), 0.5)

        loss = mean_squared_rgb_loss(render, target)
        loss.backward()

        self.assertAlmostEqual(float(loss.detach()), 0.25)
        self.assertGreater(float(render.grad.abs().sum()), 0.0)


    def test_target_valid_rgb_mask_excludes_black_gt_pixels(self):
        target = torch.zeros(3, 2, 2)
        target[:, 0, 0] = 0.5
        mask = target_valid_rgb_mask(target)

        self.assertEqual(tuple(mask.shape), (2, 2))
        self.assertTrue(bool(mask[0, 0]))
        self.assertFalse(bool(mask[1, 1]))

    def test_masked_mse_uses_only_nonzero_gt_pixels(self):
        render = torch.ones(3, 2, 2, requires_grad=True)
        target = torch.zeros(3, 2, 2)
        target[:, 0, 0] = 0.5
        valid_mask = target_valid_rgb_mask(target)

        loss = mean_squared_rgb_loss(render, target, valid_mask=valid_mask)
        loss.backward()

        self.assertAlmostEqual(float(loss.detach()), 0.25)
        self.assertGreater(float(render.grad[:, 0, 0].abs().sum()), 0.0)
        self.assertEqual(float(render.grad[:, 1, 1].abs().sum()), 0.0)

    def test_pose_safe_mse_loss_applies_and_backpropagates(self):
        render = torch.zeros(3, 4, 4, requires_grad=True)
        target = torch.full((3, 4, 4), 0.5)

        weighted_loss, debug = pose_render_mse_loss(
            render,
            target,
            mode="mse_pose_safe_v1",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info=_pose_safe_info(),
            weight=0.10,
            max_pose_risk=0.30,
            max_utility_drift=0.55,
            min_pose_support=0.45,
            min_match_support=0.45,
        )
        weighted_loss.backward()

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["reason"], "mse_rgb_pose_safe")
        self.assertAlmostEqual(debug["raw_loss"], 0.25)
        self.assertAlmostEqual(debug["weighted_loss"], 0.025)
        self.assertGreater(float(render.grad.abs().sum()), 0.0)


    def test_baseline_render_lock_policy_enables_mse_without_direct_density(self):
        self.assertIn(
            "render_frame_policy",
            inspect.signature(pose_render_mse_loss_enabled).parameters,
        )
        self.assertIn(
            "render_frame_policy",
            inspect.signature(pose_render_mse_loss).parameters,
        )
        self.assertTrue(
            pose_render_mse_loss_enabled(
                "mse_pose_safe_v1",
                "off",
                _pose_safe_info(),
                0.10,
                render_frame_policy="baseline_keyframe_lock_v1",
                max_pose_risk=0.30,
            )
        )

        render = torch.zeros(3, 4, 4, requires_grad=True)
        target = torch.full((3, 4, 4), 0.5)
        weighted_loss, debug = pose_render_mse_loss(
            render,
            target,
            mode="mse_pose_safe_v1",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info=_pose_safe_info(),
            weight=0.10,
            max_pose_risk=0.30,
        )

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["direct_density_mode"], "off")
        self.assertEqual(debug["render_frame_policy"], "baseline_keyframe_lock_v1")
        self.assertEqual(debug["reason"], "mse_rgb_pose_safe")
        self.assertAlmostEqual(float(weighted_loss.detach()), 0.025)

    def test_structure_alignment_score_is_high_for_matching_edges(self):
        target = torch.zeros(3, 8, 8)
        target[:, :, 4:] = 1.0
        render = target.clone()

        score, debug = pose_render_structure_alignment_score(render, target)

        self.assertGreater(score, 0.99)
        self.assertEqual(debug["structure_gate_mode"], "gradient_correlation_v1")

    def test_structure_gate_rejects_mismatched_edge_layout(self):
        render = torch.zeros(3, 8, 8, requires_grad=True)
        target = torch.zeros(3, 8, 8)
        target[:, :, 4:] = 1.0

        weighted_loss, debug = pose_render_mse_loss(
            render,
            target,
            mode="mse_pose_safe_v1",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info=_pose_safe_info(),
            weight=0.10,
            structure_gate_mode="gradient_correlation_v1",
            structure_min_score=0.55,
            max_pose_risk=0.30,
        )

        self.assertFalse(debug["applied"])
        self.assertEqual(debug["reason"], "psnr_structure_gate_low")
        self.assertLess(debug["structure_score"], 0.55)
        self.assertEqual(float(weighted_loss), 0.0)


    def test_raw_pose_support_weight_scale_reduces_weak_correspondence_frames(self):
        scale, debug = pose_render_support_weight_scale(
            _pose_safe_info_with_raw_support(correspondences=3400, final_inliers=2800),
            mode="raw_pose_support_v1",
            min_scale=0.45,
            correspondence_low=3000,
            correspondence_high=8000,
        )

        self.assertLess(scale, 0.55)
        self.assertGreaterEqual(scale, 0.45)
        self.assertEqual(debug["support_weight_mode"], "raw_pose_support_v1")
        self.assertEqual(debug["support_correspondence_count"], 3400)

    def test_raw_pose_support_weight_scale_keeps_v2_when_counts_are_missing(self):
        render = torch.zeros(3, 4, 4, requires_grad=True)
        target = torch.full((3, 4, 4), 0.5)

        weighted_loss, debug = pose_render_mse_loss(
            render,
            target,
            mode="mse_pose_safe_v1",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info=_pose_safe_info(),
            weight=0.10,
            support_weight_mode="raw_pose_support_v1",
            max_pose_risk=0.30,
        )

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["support_weight_mode"], "raw_pose_support_v1")
        self.assertEqual(debug["support_weight_reason"], "missing_raw_support")
        self.assertAlmostEqual(debug["support_weight_scale"], 1.0)
        self.assertAlmostEqual(debug["weighted_loss"], 0.025)
        self.assertAlmostEqual(float(weighted_loss.detach()), 0.025)

    def test_raw_pose_support_gate_rejects_low_support_frames(self):
        scale, debug = pose_render_support_weight_scale(
            _pose_safe_info_with_raw_support(correspondences=3400, final_inliers=2800),
            mode="raw_pose_support_gate_v1",
            min_scale=0.45,
            correspondence_low=3000,
            correspondence_high=8000,
        )

        self.assertEqual(scale, 0.0)
        self.assertEqual(debug["support_weight_reason"], "low_raw_pose_support")

    def test_raw_pose_support_gate_disables_mse_when_counts_are_missing(self):
        render = torch.zeros(3, 4, 4, requires_grad=True)
        target = torch.full((3, 4, 4), 0.5)

        weighted_loss, debug = pose_render_mse_loss(
            render,
            target,
            mode="mse_pose_safe_v1",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info=_pose_safe_info(),
            weight=0.10,
            support_weight_mode="raw_pose_support_gate_v1",
            max_pose_risk=0.30,
        )

        self.assertFalse(debug["applied"])
        self.assertEqual(debug["reason"], "missing_raw_support")
        self.assertEqual(debug["support_weight_scale"], 0.0)
        self.assertEqual(float(weighted_loss), 0.0)

    def test_raw_pose_support_weight_scale_is_applied_to_mse_weight(self):
        render = torch.zeros(3, 4, 4, requires_grad=True)
        target = torch.full((3, 4, 4), 0.5)

        weighted_loss, debug = pose_render_mse_loss(
            render,
            target,
            mode="mse_pose_safe_v1",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info=_pose_safe_info_with_raw_support(correspondences=3400, final_inliers=2800),
            weight=0.10,
            support_weight_mode="raw_pose_support_v1",
            max_pose_risk=0.30,
        )

        self.assertTrue(debug["applied"])
        self.assertLess(debug["support_weight_scale"], 0.55)
        self.assertAlmostEqual(debug["weight"], 0.10 * debug["support_weight_scale"])
        self.assertAlmostEqual(float(weighted_loss.detach()), 0.25 * debug["weight"])

    def test_robust_pose_risk_mse_preserves_uniform_psnr_residual(self):
        render = torch.zeros(3, 4, 4, requires_grad=True)
        target = torch.full((3, 4, 4), 0.5)

        weighted_loss, debug = pose_render_mse_loss(
            render,
            target,
            mode="robust_mse_pose_risk_v1",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info=_pose_safe_info_with_risk(0.10),
            weight=0.10,
            max_pose_risk=0.30,
        )
        weighted_loss.backward()

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["reason"], "robust_mse_rgb_pose_safe")
        self.assertAlmostEqual(debug["raw_loss"], 0.25)
        self.assertAlmostEqual(debug["weighted_loss"], 0.025)
        self.assertAlmostEqual(debug["robust_weight_min"], debug["robust_weight_max"])
        self.assertGreater(float(render.grad.abs().sum()), 0.0)

    def test_robust_pose_risk_mse_soft_downweights_local_outlier_without_masking(self):
        render = torch.full((3, 2, 2), 0.1, requires_grad=True)
        target = torch.zeros(3, 2, 2)
        with torch.no_grad():
            render[:, 0, 0] = 1.1
        mse = mean_squared_rgb_loss(render, target)

        weighted_loss, debug = pose_render_mse_loss(
            render,
            target,
            mode="robust_mse_pose_risk_v1",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info=_pose_safe_info_with_risk(0.18),
            weight=1.0,
            max_pose_risk=0.30,
        )
        weighted_loss.backward()

        self.assertTrue(debug["applied"])
        self.assertLess(debug["raw_loss"], float(mse.detach()))
        self.assertGreater(debug["robust_weight_min"], 0.0)
        self.assertLess(debug["robust_weight_min"], debug["robust_weight_mean"])
        self.assertLess(debug["robust_weight_mean"], debug["robust_weight_max"])
        self.assertGreater(float(render.grad[:, 1, 1].abs().sum()), 0.0)

    def test_robust_pose_risk_mse_tightens_scale_as_pose_risk_rises(self):
        render = torch.full((3, 2, 2), 0.1)
        target = torch.zeros(3, 2, 2)
        render[:, 0, 0] = 0.6

        _, low_debug = pose_render_mse_loss(
            render,
            target,
            mode="robust_mse_pose_risk_v1",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info=_pose_safe_info_with_risk(0.03),
            weight=1.0,
            max_pose_risk=0.30,
        )
        _, high_debug = pose_render_mse_loss(
            render,
            target,
            mode="robust_mse_pose_risk_v1",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info=_pose_safe_info_with_risk(0.27),
            weight=1.0,
            max_pose_risk=0.30,
        )

        self.assertLess(high_debug["robust_scale_eff"], low_debug["robust_scale_eff"])
        self.assertLess(high_debug["raw_loss"], low_debug["raw_loss"])

    def test_frequency_pose_risk_mse_preserves_flat_psnr_residual(self):
        render = torch.zeros(3, 4, 4, requires_grad=True)
        target = torch.full((3, 4, 4), 0.5)

        weighted_loss, debug = pose_render_mse_loss(
            render,
            target,
            mode="freq_mse_pose_risk_v1",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info=_pose_safe_info_with_risk(0.10),
            weight=0.10,
            max_pose_risk=0.30,
        )
        weighted_loss.backward()

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["reason"], "freq_mse_rgb_pose_safe")
        self.assertAlmostEqual(debug["raw_loss"], 0.25)
        self.assertAlmostEqual(debug["weighted_loss"], 0.025)
        self.assertAlmostEqual(debug["frequency_weight_min"], debug["frequency_weight_max"])
        self.assertGreater(float(render.grad.abs().sum()), 0.0)

    def test_frequency_pose_risk_mse_gently_prioritizes_target_edges(self):
        render = torch.zeros(3, 3, 3, requires_grad=True)
        target = torch.zeros(3, 3, 3)
        target[:, :, 2] = 1.0
        mse = mean_squared_rgb_loss(render, target)

        weighted_loss, debug = pose_render_mse_loss(
            render,
            target,
            mode="freq_mse_pose_risk_v1",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info=_pose_safe_info_with_risk(0.10),
            weight=1.0,
            max_pose_risk=0.30,
        )
        weighted_loss.backward()

        self.assertTrue(debug["applied"])
        self.assertGreater(debug["raw_loss"], float(mse.detach()))
        self.assertGreater(debug["frequency_weight_max"], debug["frequency_weight_mean"])
        self.assertGreater(debug["frequency_weight_mean"], debug["frequency_weight_min"])
        self.assertGreater(float(render.grad[:, :, 2].abs().sum()), 0.0)

    def test_frequency_pose_risk_mse_reduces_edge_boost_as_pose_risk_rises(self):
        render = torch.zeros(3, 3, 3)
        target = torch.zeros(3, 3, 3)
        target[:, :, 2] = 1.0

        _, low_debug = pose_render_mse_loss(
            render,
            target,
            mode="freq_mse_pose_risk_v1",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info=_pose_safe_info_with_risk(0.03),
            weight=1.0,
            max_pose_risk=0.30,
        )
        _, high_debug = pose_render_mse_loss(
            render,
            target,
            mode="freq_mse_pose_risk_v1",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info=_pose_safe_info_with_risk(0.27),
            weight=1.0,
            max_pose_risk=0.30,
        )

        self.assertLess(high_debug["frequency_boost"], low_debug["frequency_boost"])
        self.assertLess(high_debug["raw_loss"], low_debug["raw_loss"])

    def test_nonzero_gt_target_mask_keeps_baseline_lock_loss_on_valid_pixels(self):
        render = torch.ones(3, 2, 2, requires_grad=True)
        target = torch.zeros(3, 2, 2)
        target[:, 0, 0] = 0.5

        weighted_loss, debug = pose_render_mse_loss(
            render,
            target,
            mode="mse_pose_safe_v1",
            direct_density_mode="off",
            render_frame_policy="baseline_keyframe_lock_v1",
            keyframe_info=_pose_safe_info(),
            weight=0.10,
            target_mask_mode="nonzero_gt_v1",
            max_pose_risk=0.30,
        )
        weighted_loss.backward()

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["target_mask_mode"], "nonzero_gt_v1")
        self.assertAlmostEqual(debug["target_valid_ratio"], 0.25)
        self.assertAlmostEqual(debug["raw_loss"], 0.25)
        self.assertEqual(float(render.grad[:, 1, 1].abs().sum()), 0.0)

    def test_mse_loss_is_train_direct_density_and_pose_safe_gated(self):
        self.assertFalse(
            pose_render_mse_loss_enabled(
                "off", "pose_safe_streaming_memory_v1", _pose_safe_info(), 0.10
            )
        )
        self.assertFalse(
            pose_render_mse_loss_enabled(
                "mse_pose_safe_v1", "off", _pose_safe_info(), 0.10
            )
        )
        test_info = _pose_safe_info()
        test_info["is_test"] = True
        self.assertFalse(
            pose_render_mse_loss_enabled(
                "mse_pose_safe_v1",
                "pose_safe_streaming_memory_v1",
                test_info,
                0.10,
            )
        )
        risky_info = _pose_safe_info()
        risky_info["_paper_aligned_pose_render_coupling"][
            "pose_render_risk_score"
        ] = 0.72
        self.assertFalse(
            pose_render_mse_loss_enabled(
                "mse_pose_safe_v1",
                "pose_safe_streaming_memory_v1",
                risky_info,
                0.10,
                max_pose_risk=0.30,
            )
        )
        self.assertTrue(
            pose_render_mse_loss_enabled(
                "mse_pose_safe_v1",
                "pose_safe_streaming_memory_v1",
                _pose_safe_info(),
                0.10,
                max_pose_risk=0.30,
            )
        )

    def test_mse_loss_returns_zero_for_shape_mismatch(self):
        render = torch.zeros(3, 4, 4, requires_grad=True)
        target = torch.zeros(3, 4, 5)

        weighted_loss, debug = pose_render_mse_loss(
            render,
            target,
            mode="mse_pose_safe_v1",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_info=_pose_safe_info(),
            weight=0.10,
        )

        self.assertFalse(debug["applied"])
        self.assertEqual(debug["reason"], "shape_mismatch")
        self.assertEqual(float(weighted_loss), 0.0)

    def test_train_bridges_baseline_lock_raw_pose_support_into_keyframe_info(self):
        source = (
            Path(__file__)
            .resolve()
            .parents[1]
            .joinpath("train.py")
            .read_text(encoding="utf-8")
        )

        self.assertIn("baseline_render_lock_pose_support", source)
        self.assertIn("paper_aligned_render_frame_policy", source)
        self.assertIn("fin_dec is None", source)
        self.assertIn("augment_pose_render_payload_with_posterior_risk", source)

    def test_scene_model_integrates_psnr_loss_before_backward_and_metadata(self):
        source = (
            Path(__file__)
            .resolve()
            .parents[1]
            .joinpath("scene", "scene_model.py")
            .read_text(encoding="utf-8")
        )

        self.assertIn("pose_render_mse_loss", source)
        self.assertIn("self.pose_render_psnr_loss_stats", source)
        self.assertIn("self.paper_aligned_render_frame_policy", source)
        self.assertIn("render_frame_policy=self.paper_aligned_render_frame_policy", source)
        self.assertIn("target_mask_mode=self.pose_render_psnr_loss_target_mask", source)
        self.assertIn("self.pose_render_psnr_loss_max_applied_ratio", source)
        self.assertIn("psnr_budget_exhausted", source)
        self.assertIn("self.pose_render_psnr_loss_ambiguity_low", source)
        self.assertIn("psnr_residual_ambiguity_band", source)
        self.assertIn("self.pose_render_psnr_loss_scene_guard", source)
        self.assertIn("self.pose_render_psnr_loss_scene_guard_max_events", source)
        self.assertIn("self.pose_render_psnr_loss_structure_gate", source)
        self.assertIn("self.pose_render_psnr_loss_structure_min_raw_mean", source)
        self.assertIn("pose_render_structure_alignment_score", source)
        self.assertIn("pose_render_structure_alignment_score(image, gt_image)", source)
        self.assertIn("structure_gate_window_expired", source)
        self.assertIn("structure_min_raw_mean", source)
        self.assertIn("self.pose_render_psnr_loss_late_raw_stop", source)
        self.assertIn("psnr_late_raw_mean_stop", source)
        self.assertIn("self.pose_render_psnr_loss_health_gate", source)
        self.assertIn("psnr_health_gate_low", source)
        self.assertIn("health_gate_score", source)
        self.assertIn("self.paper_aligned_training_background_mode", source)
        self.assertIn("self._training_background_step", source)
        self.assertIn("_training_background_for_keyframe", source)
        self.assertIn("deterministic_random_v1", source)
        self.assertIn("bg=self._training_background_for_keyframe(keyframe, lvl)", source)
        self.assertIn("_pose_render_psnr_scene_guard_should_block", source)
        self.assertIn("psnr_scene_guard_blocked", source)
        self.assertIn("psnr_scene_guard_late_window_expired", source)
        self.assertIn("psnr_structure_gate_low", source)
        call = source.index("psnr_loss, psnr_loss_debug = pose_render_mse_loss")
        scene_guard = source.index("_pose_render_psnr_scene_guard_should_block", call)
        structure_gate = source.index("pose_render_structure_alignment_score", scene_guard)
        self.assertLess(scene_guard, structure_gate)
        backward = source.index("loss.backward()", call)
        self.assertLess(call, backward)
        self.assertIn('"pose_render_psnr_loss": self._pose_render_psnr_loss_summary()', source)

    def test_precommit_scene_guard_waits_and_records_candidate_raw_loss(self):
        model = object.__new__(SceneModel)
        model.pose_render_psnr_loss_scene_guard = "raw_loss_ratio_precommit_guard_v1"
        model.pose_render_psnr_loss_scene_guard_triggered = False
        model.pose_render_psnr_loss_scene_guard_raw_low = 0.0028
        model.pose_render_psnr_loss_scene_guard_raw_high = 0.0055
        model.pose_render_psnr_loss_scene_guard_min_ratio = 0.10
        model.pose_render_psnr_loss_scene_guard_min_events = 384
        model.pose_render_psnr_loss_scene_guard_min_applied = 48
        model.pose_render_psnr_loss_scene_guard_max_events = 1000
        model.pose_render_psnr_loss_stats = {
            "events": 0,
            "applied": 0,
            "raw_loss_sum": 0.0,
            "psnr_scene_guard_precommit_wait": 0,
            "scene_guard_candidate_count": 0,
            "scene_guard_candidate_raw_loss_sum": 0.0,
        }
        debug = {
            "applied": True,
            "raw_loss": 0.005,
            "pose_gate_mode": "pose_safe_v1",
            "pose_gate_passed": True,
        }

        should_block, guard_debug = model._pose_render_psnr_scene_guard_should_block(debug)

        self.assertTrue(should_block)
        self.assertEqual(guard_debug["reason"], "psnr_scene_guard_precommit_wait")
        self.assertEqual(guard_debug["scene_guard_candidate_count"], 1)
        debug.update(guard_debug)
        debug.update({"applied": False, "weight": 0.0, "weighted_loss": 0.0})
        model._record_pose_render_psnr_loss(debug)

        stats = model.pose_render_psnr_loss_stats
        self.assertEqual(stats["events"], 1)
        self.assertEqual(stats["applied"], 0)
        self.assertEqual(stats["psnr_scene_guard_precommit_wait"], 1)
        self.assertEqual(stats["scene_guard_candidate_count"], 1)
        self.assertAlmostEqual(stats["scene_guard_candidate_raw_loss_sum"], 0.005)

    def test_precommit_scene_guard_triggers_from_candidate_statistics(self):
        model = object.__new__(SceneModel)
        model.pose_render_psnr_loss_scene_guard = "raw_loss_ratio_precommit_guard_v1"
        model.pose_render_psnr_loss_scene_guard_triggered = False
        model.pose_render_psnr_loss_scene_guard_raw_low = 0.0028
        model.pose_render_psnr_loss_scene_guard_raw_high = 0.0055
        model.pose_render_psnr_loss_scene_guard_min_ratio = 0.10
        model.pose_render_psnr_loss_scene_guard_min_events = 384
        model.pose_render_psnr_loss_scene_guard_min_applied = 48
        model.pose_render_psnr_loss_scene_guard_max_events = 1000
        model.pose_render_psnr_loss_stats = {
            "events": 383,
            "applied": 0,
            "raw_loss_sum": 0.0,
            "scene_guard_candidate_count": 205,
            "scene_guard_candidate_raw_loss_sum": 205 * 0.00549,
        }
        debug = {
            "applied": True,
            "raw_loss": 0.00549,
            "pose_gate_mode": "pose_safe_v1",
            "pose_gate_passed": True,
        }

        should_block, guard_debug = model._pose_render_psnr_scene_guard_should_block(debug)

        self.assertTrue(should_block)
        self.assertTrue(model.pose_render_psnr_loss_scene_guard_triggered)
        self.assertEqual(guard_debug["reason"], "psnr_scene_guard_triggered")
        self.assertEqual(guard_debug["scene_guard_trigger_events"], 384)
        self.assertEqual(guard_debug["scene_guard_trigger_applied"], 206)
        self.assertGreater(guard_debug["scene_guard_trigger_ratio"], 0.10)

    def test_render_gap_precommit_guard_triggers_on_high_residual_with_coverage_gap(self):
        model = object.__new__(SceneModel)
        model.pose_render_psnr_loss_scene_guard = (
            "raw_loss_ratio_precommit_render_gap_guard_v1"
        )
        model.pose_render_psnr_loss_scene_guard_triggered = False
        model.pose_render_psnr_loss_scene_guard_raw_low = 0.0028
        model.pose_render_psnr_loss_scene_guard_raw_high = 0.0055
        model.pose_render_psnr_loss_scene_guard_min_ratio = 0.10
        model.pose_render_psnr_loss_scene_guard_min_events = 384
        model.pose_render_psnr_loss_scene_guard_min_applied = 48
        model.pose_render_psnr_loss_scene_guard_max_events = 1000
        model.pose_render_psnr_loss_stats = {
            "events": 383,
            "applied": 0,
            "raw_loss_sum": 0.0,
            "scene_guard_candidate_count": 182,
            "scene_guard_candidate_raw_loss_sum": 182 * 0.0066,
        }
        model.pose_render_texture_sampling_stats = {
            "events": 120,
            "applied": 42,
            "coverage_deficit_sum": 120 * 0.078,
        }
        debug = {
            "applied": True,
            "raw_loss": 0.0066,
            "pose_gate_mode": "pose_safe_v1",
            "pose_gate_passed": True,
        }

        should_block, guard_debug = model._pose_render_psnr_scene_guard_should_block(debug)

        self.assertTrue(should_block)
        self.assertTrue(model.pose_render_psnr_loss_scene_guard_triggered)
        self.assertEqual(guard_debug["reason"], "psnr_scene_guard_triggered")
        self.assertTrue(guard_debug["scene_guard_render_gap_triggered"])
        self.assertGreaterEqual(guard_debug["scene_guard_coverage_deficit_mean"], 0.075)
        self.assertGreater(
            guard_debug["scene_guard_projected_raw_loss_mean"],
            model.pose_render_psnr_loss_scene_guard_raw_high,
        )

    def test_render_gap_precommit_guard_keeps_high_coverage_scene_open(self):
        model = object.__new__(SceneModel)
        model.pose_render_psnr_loss_scene_guard = (
            "raw_loss_ratio_precommit_render_gap_guard_v1"
        )
        model.pose_render_psnr_loss_scene_guard_triggered = False
        model.pose_render_psnr_loss_scene_guard_raw_low = 0.0028
        model.pose_render_psnr_loss_scene_guard_raw_high = 0.0055
        model.pose_render_psnr_loss_scene_guard_min_ratio = 0.10
        model.pose_render_psnr_loss_scene_guard_min_events = 384
        model.pose_render_psnr_loss_scene_guard_min_applied = 48
        model.pose_render_psnr_loss_scene_guard_max_events = 1000
        model.pose_render_psnr_loss_stats = {
            "events": 383,
            "applied": 0,
            "raw_loss_sum": 0.0,
            "scene_guard_candidate_count": 150,
            "scene_guard_candidate_raw_loss_sum": 150 * 0.0071,
        }
        model.pose_render_texture_sampling_stats = {
            "events": 120,
            "applied": 0,
            "coverage_deficit_sum": 120 * 0.001,
        }
        debug = {
            "applied": True,
            "raw_loss": 0.0071,
            "pose_gate_mode": "pose_safe_v1",
            "pose_gate_passed": True,
        }

        should_block, guard_debug = model._pose_render_psnr_scene_guard_should_block(debug)

        self.assertFalse(should_block)
        self.assertFalse(model.pose_render_psnr_loss_scene_guard_triggered)
        self.assertFalse(guard_debug.get("scene_guard_render_gap_triggered", False))
        self.assertLess(guard_debug.get("scene_guard_coverage_deficit_mean", 0.0), 0.075)

    def test_coverage_precommit_guard_keeps_low_pressure_raw_band_open(self):
        model = object.__new__(SceneModel)
        model.pose_render_psnr_loss_scene_guard = (
            "raw_loss_ratio_precommit_coverage_guard_v1"
        )
        model.pose_render_psnr_loss_scene_guard_triggered = False
        model.pose_render_psnr_loss_scene_guard_raw_low = 0.0028
        model.pose_render_psnr_loss_scene_guard_raw_high = 0.0055
        model.pose_render_psnr_loss_scene_guard_min_ratio = 0.10
        model.pose_render_psnr_loss_scene_guard_min_events = 384
        model.pose_render_psnr_loss_scene_guard_min_applied = 48
        model.pose_render_psnr_loss_scene_guard_max_events = 1000
        model.pose_render_psnr_loss_stats = {
            "events": 383,
            "applied": 0,
            "raw_loss_sum": 0.0,
            "scene_guard_candidate_count": 183,
            "scene_guard_candidate_raw_loss_sum": 183 * 0.0030,
        }
        model.pose_render_texture_sampling_stats = {
            "events": 120,
            "applied": 2,
            "coverage_deficit_sum": 120 * 0.015,
        }
        debug = {
            "applied": True,
            "raw_loss": 0.0030,
            "pose_gate_mode": "pose_safe_v1",
            "pose_gate_passed": True,
        }

        should_block, guard_debug = model._pose_render_psnr_scene_guard_should_block(debug)

        self.assertFalse(should_block)
        self.assertTrue(guard_debug.get("scene_guard_coverage_aware", False))
        self.assertFalse(
            guard_debug.get("scene_guard_coverage_pressure_sufficient", False)
        )
        self.assertFalse(model.pose_render_psnr_loss_scene_guard_triggered)

    def test_coverage_precommit_guard_triggers_raw_band_when_pressure_is_high(self):
        model = object.__new__(SceneModel)
        model.pose_render_psnr_loss_scene_guard = (
            "raw_loss_ratio_precommit_coverage_guard_v1"
        )
        model.pose_render_psnr_loss_scene_guard_triggered = False
        model.pose_render_psnr_loss_scene_guard_raw_low = 0.0028
        model.pose_render_psnr_loss_scene_guard_raw_high = 0.0055
        model.pose_render_psnr_loss_scene_guard_min_ratio = 0.10
        model.pose_render_psnr_loss_scene_guard_min_events = 384
        model.pose_render_psnr_loss_scene_guard_min_applied = 48
        model.pose_render_psnr_loss_scene_guard_max_events = 1000
        model.pose_render_psnr_loss_stats = {
            "events": 383,
            "applied": 0,
            "raw_loss_sum": 0.0,
            "scene_guard_candidate_count": 183,
            "scene_guard_candidate_raw_loss_sum": 183 * 0.0030,
        }
        model.pose_render_texture_sampling_stats = {
            "events": 120,
            "applied": 42,
            "coverage_deficit_sum": 120 * 0.080,
        }
        debug = {
            "applied": True,
            "raw_loss": 0.0030,
            "pose_gate_mode": "pose_safe_v1",
            "pose_gate_passed": True,
        }

        should_block, guard_debug = model._pose_render_psnr_scene_guard_should_block(debug)

        self.assertTrue(should_block)
        self.assertTrue(guard_debug["scene_guard_coverage_aware"])
        self.assertTrue(guard_debug["scene_guard_coverage_pressure_sufficient"])
        self.assertEqual(guard_debug["reason"], "psnr_scene_guard_triggered")
        self.assertTrue(model.pose_render_psnr_loss_scene_guard_triggered)

    def test_args_exposes_pose_safe_psnr_loss(self):
        source = (
            Path(__file__)
            .resolve()
            .parents[1]
            .joinpath("args.py")
            .read_text(encoding="utf-8")
        )

        self.assertIn("--paper_aligned_pose_render_psnr_loss", source)
        self.assertIn("'mse_pose_safe_v1'", source)
        self.assertIn("--paper_aligned_pose_render_psnr_loss_weight", source)
        self.assertIn("--paper_aligned_training_background_mode", source)
        self.assertIn("'target_mean_v1'", source)
        self.assertIn("'deterministic_random_v1'", source)
        self.assertIn("--paper_aligned_pose_render_psnr_loss_health_gate", source)
        self.assertIn("--paper_aligned_pose_render_psnr_loss_health_min_score", source)
        self.assertIn("--paper_aligned_pose_render_psnr_loss_health_min_raw_loss", source)


if __name__ == "__main__":
    unittest.main()
