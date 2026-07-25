from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from asr_gs.config import (
    FINAL_CONFIG,
    resolve_config,
    resolve_runtime_config,
)
from asr_gs.diagnostics import should_track_render_response
from asr_gs.pose_reliability import (
    POLICY_NAME,
    pose_candidate_improves,
    should_retry_failed_pose,
    should_review_weak_pose,
)
from asr_gs.refinement_policy import extra_refinement_decision
from asr_gs.response_sampling import (
    response_guided_sampling_probability,
    sampling_response_verdict,
    sampling_scene_guard,
)
from asr_gs.transaction import (
    refinement_candidate_acceptance,
    restore_selected_gaussian_state,
    snapshot_selected_gaussian_state,
)


class ReleaseConfigurationTests(unittest.TestCase):
    def test_final_configuration_locks_reported_a_and_k8_c(self):
        self.assertEqual(POLICY_NAME, "support_residual_secondary_review")
        self.assertEqual(FINAL_CONFIG.pose.retry_attempts, 2)
        self.assertEqual(FINAL_CONFIG.pose.multi_hypothesis_attempts, 2)
        self.assertEqual(FINAL_CONFIG.sampling.alpha, 0.03)
        self.assertEqual(FINAL_CONFIG.sampling.min_selectivity, 1.8)
        self.assertEqual(FINAL_CONFIG.refinement.max_extra_iterations, 8)
        self.assertEqual(FINAL_CONFIG.refinement.fraction, 0.25)

    def test_ablation_resolution_disables_only_requested_component(self):
        config = resolve_config("asr-gs", {"b"})
        self.assertTrue(config.pose.enabled)
        self.assertFalse(config.sampling.enabled)
        self.assertTrue(config.refinement.enabled)

    def test_baseline_disables_all_asr_gs_components(self):
        config = resolve_config("baseline")
        self.assertFalse(config.pose.enabled)
        self.assertFalse(config.sampling.enabled)
        self.assertFalse(config.refinement.enabled)

    def test_viewer_inference_without_training_args_uses_safe_config(self):
        config = resolve_runtime_config(
            SimpleNamespace(anchor_overlap=0.3),
            inference_mode=True,
        )
        self.assertEqual(config.method, "baseline")
        self.assertFalse(config.pose.enabled)
        self.assertFalse(config.sampling.enabled)
        self.assertFalse(config.refinement.enabled)


class SamplingContractTests(unittest.TestCase):
    def test_sampling_only_mode_still_tracks_render_response(self):
        self.assertTrue(
            should_track_render_response(
                sampling_enabled=True,
                refinement_enabled=False,
                is_latest_keyframe=True,
                is_test=False,
            )
        )
        self.assertFalse(
            should_track_render_response(
                sampling_enabled=False,
                refinement_enabled=False,
                is_latest_keyframe=True,
                is_test=False,
            )
        )

    def test_guided_probability_is_finite_and_clipped(self):
        base = torch.tensor([[0.9, 0.8], [0.7, 0.6]])
        response = torch.tensor([[100.0, 0.0], [0.0, 0.0]])
        guided, debug = response_guided_sampling_probability(
            base,
            response,
            FINAL_CONFIG.sampling,
        )
        self.assertTrue(torch.isfinite(guided).all())
        self.assertGreaterEqual(float(guided.min()), 0.0)
        self.assertLessEqual(float(guided.max()), 1.0)
        self.assertAlmostEqual(
            debug["mass_before"],
            debug["mass_before_clipping"],
            places=5,
        )
        self.assertLessEqual(
            debug["mass_after_clipping"],
            debug["mass_before_clipping"] + 1e-6,
        )

    def test_sampling_ablation_is_exact_baseline_bypass(self):
        config = resolve_config("asr-gs", {"b"}).sampling
        base = torch.rand(5, 7)
        response = torch.rand(5, 7)
        guided, debug = response_guided_sampling_probability(
            base,
            response,
            config,
        )
        self.assertTrue(torch.equal(guided, base))
        self.assertEqual(debug["reason"], "disabled")

    def test_sampling_requires_a_representation_coverage_gap(self):
        base = torch.full((3, 3), 0.5)
        response = torch.arange(9, dtype=torch.float32).view(3, 3)
        guided, debug = response_guided_sampling_probability(
            base,
            response,
            FINAL_CONFIG.sampling,
            coverage_deficit=0.01,
        )
        self.assertTrue(torch.equal(guided, base))
        self.assertEqual(debug["reason"], "coverage_sufficient")

    def test_bad_historical_response_disables_sampling(self):
        guard = sampling_scene_guard(
            {"response_evaluated": 3, "response_bad": 3},
            FINAL_CONFIG.sampling,
        )
        self.assertTrue(guard["disabled"])
        verdict = sampling_response_verdict(
            {
                "observations": 4,
                "first_rgb_mse": 1.0,
                "latest_rgb_mse": 1.1,
                "relative_improvement": -0.1,
            },
            FINAL_CONFIG.sampling,
        )
        self.assertEqual(verdict["verdict"], "bad")


class RefinementContractTests(unittest.TestCase):
    def test_transaction_restores_values_and_optimizer_state(self):
        values = torch.arange(12, dtype=torch.float32).view(4, 3)
        exp_avg = torch.arange(4, dtype=torch.float32).view(4, 1)
        parameters = {
            "xyz": {
                "val": values.clone(),
                "exp_avg": exp_avg.clone(),
                "exp_avg_sq": (exp_avg + 10).clone(),
                "lr": torch.tensor(0.25),
            }
        }
        selection = torch.tensor([True, False, True, False])
        snapshot = snapshot_selected_gaussian_state(parameters, selection)
        parameters["xyz"]["val"][selection] += 100
        parameters["xyz"]["exp_avg"][selection] += 100
        parameters["xyz"]["exp_avg_sq"][selection] += 100
        parameters["xyz"]["lr"].fill_(1.0)

        restore_selected_gaussian_state(parameters, selection, snapshot)

        self.assertTrue(torch.equal(parameters["xyz"]["val"], values))
        self.assertTrue(
            torch.equal(parameters["xyz"]["exp_avg"], exp_avg)
        )
        self.assertEqual(float(parameters["xyz"]["lr"]), 0.25)

    def test_transaction_rejects_a_photometric_regression(self):
        reference = {
            "total_loss": 1.0,
            "rgb_mse": 0.5,
            "dssim": 0.2,
            "depth": 0.1,
        }
        candidate = {
            "total_loss": 0.9,
            "rgb_mse": 0.51,
            "dssim": 0.19,
            "depth": 0.09,
        }
        decision = refinement_candidate_acceptance(reference, candidate)
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["reason"], "rgb_mse_not_improved")

    def test_refinement_decision_cannot_exceed_k8(self):
        decision = extra_refinement_decision(
            response={
                "observations": 8,
                "first_rgb_mse": 1.0,
                "latest_rgb_mse": 0.5,
                "best_rgb_mse": 0.5,
                "relative_improvement": 0.5,
                "recent_rgb_mse": [0.9, 0.8, 0.7, 0.5],
                "projection_coverage_deficit": 0.15,
                "representation_coverage_deficit": 0.15,
            },
            base_iterations=300,
            is_test=False,
            already_finalized=False,
            config=FINAL_CONFIG.refinement,
        )
        self.assertTrue(decision["applied"])
        self.assertEqual(decision["extra_iterations"], 8)

    def test_refinement_ablation_is_exact_bypass(self):
        config = resolve_config("asr-gs", {"c"}).refinement
        decision = extra_refinement_decision(
            response={},
            base_iterations=30,
            is_test=False,
            already_finalized=False,
            config=config,
        )
        self.assertFalse(decision["applied"])
        self.assertEqual(decision["reason"], "disabled")


class PoseReliabilityContractTests(unittest.TestCase):
    def test_supported_miniba_failure_is_retried(self):
        diagnostics = {
            "failure_reason": "miniba_inliers_too_few",
            "num_2d3d_correspondences": 1000,
            "num_pnp_inliers": 200,
        }
        self.assertTrue(
            should_retry_failed_pose(diagnostics, FINAL_CONFIG.pose)
        )

    def test_weak_success_is_reviewed(self):
        diagnostics = {
            "failure_reason": "",
            "num_2d3d_correspondences": 3000,
            "num_pnp_candidate_correspondences": 2000,
            "num_pnp_inliers": 500,
            "direct_pose_miniba_residual": 0.5,
        }
        self.assertTrue(should_review_weak_pose(diagnostics, FINAL_CONFIG.pose))

    def test_replacement_requires_supported_bounded_gain(self):
        current = {
            "num_pnp_inliers": 500,
            "num_miniba_inliers": 1000,
            "direct_pose_miniba_residual": 1.0,
            "direct_pose_motion_rotation_deg": 2.0,
            "direct_pose_motion_translation": 0.05,
        }
        candidate = {
            "num_pnp_inliers": 700,
            "num_miniba_inliers": 1200,
            "direct_pose_miniba_residual": 0.7,
            "direct_pose_motion_rotation_deg": 2.5,
            "direct_pose_motion_translation": 0.06,
        }
        self.assertTrue(
            pose_candidate_improves(current, candidate, FINAL_CONFIG.pose)
        )


if __name__ == "__main__":
    unittest.main()
