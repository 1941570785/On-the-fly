import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from args import get_args
from poses.pose_initializer import (
    PoseInitializer,
    score_direct_pose_candidate_v18,
    should_accept_direct_pose_candidate_v18,
    should_retry_direct_pose_initialization_v18,
    should_run_direct_pose_multi_hypothesis_v18,
)
from poses.ransac import EstimatorType, RANSACEstimator
from scene.pose_initialization_risk import PoseInitializationRiskGate


def _rt(rotation_deg: float = 0.0) -> torch.Tensor:
    angle = torch.tensor(rotation_deg * 3.141592653589793 / 180.0)
    c = torch.cos(angle)
    s = torch.sin(angle)
    return torch.tensor(
        [
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _weak_pose_debug() -> dict[str, int]:
    return {
        "match_count_total": 500,
        "num_2d3d_correspondences": 480,
        "num_pnp_inliers": 70,
        "num_miniba_inliers": 45,
    }


def _weak_viewpoint() -> dict[str, float | int]:
    return {
        "inlier_grid_coverage": 0.25,
        "inlier_grid_entropy": 0.35,
        "anchor_health_score": 0.20,
        "selected_reference_count": 1,
    }


class _FakeMiniBA:
    def __init__(self, translation_x: float, rotation_deg: float = 0.0):
        self.translation_x = translation_x
        self.rotation_deg = rotation_deg
        self.calls = 0

    def __call__(self, rotations, translations, focal, xyz, centre, uv):
        self.calls += 1
        output_rotations = rotations.clone()
        output_rotations[0] = _rt(self.rotation_deg)[:3, :2]
        output_translation = translations.clone()
        output_translation[0, 0] = self.translation_x
        residual = torch.zeros_like(uv)
        mask = torch.ones_like(uv)
        return (
            output_rotations,
            output_translation,
            focal,
            xyz,
            residual,
            residual,
            mask,
        )


class _FakePnP:
    def __init__(self, translation_x: float, rotation_deg: float = 0.0):
        self.translation_x = translation_x
        self.rotation_deg = rotation_deg
        self.calls = 0
        self.last_count = 0

    def __call__(self, uvs, xyz, focal, centre, rotation_init, translation_init, confs):
        self.calls += 1
        self.last_count = len(xyz)
        pose = _rt(self.rotation_deg)
        pose[0, 3] = self.translation_x
        return pose[:3], torch.ones(len(xyz), dtype=torch.bool, device=xyz.device)


class PoseVerificationRiskIntegrationTests(unittest.TestCase):
    def test_pose_safe_v18_retries_only_supported_miniba_failures(self):
        supported_failure = {
            "failure_reason": "miniba_inliers_too_few",
            "num_2d3d_correspondences": 500,
            "num_pnp_inliers": 20,
        }
        self.assertTrue(
            should_retry_direct_pose_initialization_v18(
                supported_failure,
                is_test=True,
            )
        )
        self.assertFalse(
            should_retry_direct_pose_initialization_v18(
                {**supported_failure, "num_pnp_inliers": 19},
                is_test=True,
            )
        )
        self.assertFalse(
            should_retry_direct_pose_initialization_v18(
                {**supported_failure, "failure_reason": "pnp_inliers_too_few"},
                is_test=True,
            )
        )

    def test_pose_safe_v18_resamples_weak_successful_hypotheses(self):
        weak = {
            "failure_reason": "",
            "num_2d3d_correspondences": 3000,
            "num_pnp_candidate_correspondences": 2000,
            "num_pnp_inliers": 699,
            "direct_pose_miniba_residual": 0.5,
        }
        self.assertTrue(should_run_direct_pose_multi_hypothesis_v18(weak))
        self.assertFalse(
            should_run_direct_pose_multi_hypothesis_v18(
                {
                    **weak,
                    "num_pnp_inliers": 1000,
                    "direct_pose_miniba_residual": 0.5,
                }
            )
        )
        self.assertTrue(
            should_run_direct_pose_multi_hypothesis_v18(
                {
                    **weak,
                    "num_pnp_inliers": 1000,
                    "direct_pose_miniba_residual": 1.01,
                }
            )
        )

    def test_pose_safe_v18_candidate_requires_gain_and_motion_consistency(self):
        current = {
            "num_pnp_inliers": 600,
            "num_miniba_inliers": 1000,
            "direct_pose_miniba_residual": 1.0,
            "direct_pose_motion_rotation_deg": 2.0,
            "direct_pose_motion_translation": 0.10,
        }
        better = {
            **current,
            "num_pnp_inliers": 800,
            "num_miniba_inliers": 1100,
            "direct_pose_miniba_residual": 0.7,
            "direct_pose_motion_rotation_deg": 3.0,
            "direct_pose_motion_translation": 0.20,
        }
        motion_jump = {
            **better,
            "direct_pose_motion_translation": 0.50,
        }
        self.assertGreater(
            score_direct_pose_candidate_v18(better),
            score_direct_pose_candidate_v18(current),
        )
        self.assertTrue(
            should_accept_direct_pose_candidate_v18(current, better)
        )
        self.assertFalse(
            should_accept_direct_pose_candidate_v18(current, motion_jump)
        )

    def test_ransac_uses_the_supplied_sampling_generator(self):
        estimator = object.__new__(RANSACEstimator)
        estimator.N = 8
        estimator.m = 4
        estimator.type = EstimatorType.P4P
        estimator.models = torch.zeros(8, 3, 4)
        sampled_indices = []

        def capture_estimate(
            mkpts1,
            mkpts2,
            idxs,
            focal,
            centre,
            rotation,
            translation,
        ):
            sampled_indices.append(idxs.clone())

        estimator.estimate = capture_estimate
        estimator.get_inlier_mask = (
            lambda mkpts1, mkpts2, focal, centre: torch.ones(
                estimator.N,
                len(mkpts1),
                dtype=torch.bool,
            )
        )
        inputs = {
            "mkpts1": torch.zeros(12, 2),
            "mkpts2": torch.zeros(12, 3),
            "focal": torch.ones(1),
            "centre": torch.zeros(2),
            "R6D_init": torch.eye(3)[:, :2],
            "t_init": torch.zeros(3),
        }

        estimator(
            **inputs,
            generator=torch.Generator().manual_seed(17),
        )
        estimator(
            **inputs,
            generator=torch.Generator().manual_seed(17),
        )

        self.assertTrue(torch.equal(sampled_indices[0], sampled_indices[1]))

    def test_pose_initializer_uses_expanded_anchor_evidence_without_replacing_baseline_state(self):
        initializer = PoseInitializer.__new__(PoseInitializer)
        initializer.f = torch.tensor([100.0])
        initializer.centre = torch.tensor([50.0, 50.0])
        initializer.max_pnp_error = 10.0
        initializer.num_pts_miniba_incr = 4
        initializer.miniBA_incr = _FakeMiniBA(translation_x=0.01)
        initializer.PnPRANSAC = _FakePnP(translation_x=0.01)
        initializer.last_incremental_pose_support = {}
        baseline_candidates = {
            "uvs": torch.tensor(
                [[50.0, 50.0], [60.0, 50.0], [50.0, 60.0], [60.0, 60.0]]
            ),
            "pts3d": torch.tensor(
                [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0], [0.1, 0.1, 1.0]]
            ),
            "pts_conf": torch.ones(4),
            "match_indices": torch.arange(4),
        }
        initializer.last_incremental_pose_candidates = baseline_candidates
        anchor_evidence = {
            "uvs": torch.tensor(
                [
                    [51.0, 50.0],
                    [61.0, 50.0],
                    [51.0, 60.0],
                    [61.0, 60.0],
                    [41.0, 50.0],
                    [51.0, 40.0],
                ]
            ),
            "pts3d": torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [0.1, 0.0, 1.0],
                    [0.0, 0.1, 1.0],
                    [0.1, 0.1, 1.0],
                    [-0.1, 0.0, 1.0],
                    [0.0, -0.1, 1.0],
                ]
            ),
            "pts_conf": torch.ones(6),
            "match_indices": torch.arange(6),
        }

        verified, debug = initializer.verify_incremental_pose(
            _rt(),
            {"verification_trigger": True, "risk_score": 0.8},
            image_width=100,
            image_height=100,
            pose_history=[(7, _rt()), (8, _rt()), (9, _rt())],
            min_support=4,
            min_relative_median_improvement=0.05,
            pose_evidence_override=anchor_evidence,
        )

        self.assertTrue(debug["accepted"])
        self.assertEqual(debug["candidate_source"], "stable_anchor_2d3d")
        self.assertEqual(initializer.PnPRANSAC.last_count, 6)
        self.assertIs(initializer.last_incremental_pose_candidates, baseline_candidates)
        self.assertAlmostEqual(float(verified[0, 3]), 0.01, places=5)

    def test_stable_anchor_evidence_can_recover_beyond_the_local_pre_error_gate(self):
        initializer = PoseInitializer.__new__(PoseInitializer)
        initializer.f = torch.tensor([500.0])
        initializer.centre = torch.tensor([50.0, 50.0])
        initializer.max_pnp_error = 5.0
        initializer.num_pts_miniba_incr = 4
        initializer.miniBA_incr = _FakeMiniBA(translation_x=0.02)
        initializer.PnPRANSAC = _FakePnP(translation_x=0.02)
        initializer.last_incremental_pose_support = {}
        initializer.last_incremental_pose_candidates = {}
        points = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [0.0, 0.1, 1.0],
                [0.1, 0.1, 1.0],
                [-0.1, 0.0, 1.0],
                [0.0, -0.1, 1.0],
            ]
        )
        anchor_evidence = {
            "uvs": torch.stack(
                (
                    500.0 * points[:, 0] + 60.0,
                    500.0 * points[:, 1] + 50.0,
                ),
                dim=-1,
            ),
            "pts3d": points,
            "pts_conf": torch.ones(6),
            "match_indices": torch.arange(6),
        }

        verified, debug = initializer.verify_incremental_pose(
            _rt(),
            {"verification_trigger": True, "risk_score": 0.8},
            image_width=160,
            image_height=160,
            pose_history=[(7, _rt()), (8, _rt()), (9, _rt())],
            min_support=4,
            min_relative_median_improvement=0.05,
            pose_evidence_override=anchor_evidence,
            pose_evidence_pre_error_scale=3.0,
        )

        self.assertTrue(debug["accepted"])
        self.assertEqual(debug["pose_evidence_pre_error_limit"], 15.0)
        self.assertEqual(initializer.PnPRANSAC.last_count, 6)
        self.assertAlmostEqual(float(verified[0, 3]), 0.02, places=5)

    def test_verify_mode_marks_candidate_without_isolating_frame(self):
        gate = PoseInitializationRiskGate(
            mode="verify_v1",
            absolute_threshold=0.10,
            adaptive_sigma=0.0,
            warmup=0,
        )

        event = gate.evaluate(
            frame_id=10,
            pose_debug=_weak_pose_debug(),
            viewpoint_scores=_weak_viewpoint(),
            min_num_inliers=100,
            recent_pose_fail_rate=0.60,
            current_Rt=_rt(35.0),
            pose_history=[(8, _rt(0.0)), (9, _rt(1.0))],
            baseline_selected=True,
            is_test=False,
            is_bootstrap=False,
        )

        self.assertTrue(event["verification_trigger"])
        self.assertFalse(event["isolated"])
        self.assertEqual(event["decision"], "verify_candidate")

    def test_verify_mode_uses_scene_tail_threshold_below_hard_isolation_floor(self):
        gate = PoseInitializationRiskGate(
            mode="verify_v1",
            absolute_threshold=0.10,
            adaptive_sigma=2.0,
            warmup=3,
        )
        gate.risk_history.extend([0.0020, 0.0025, 0.0030, 0.0022])

        event = gate.evaluate(
            frame_id=20,
            pose_debug={
                "match_count_total": 420,
                "num_2d3d_correspondences": 400,
                "num_pnp_inliers": 250,
                "num_miniba_inliers": 230,
            },
            viewpoint_scores={
                "inlier_grid_coverage": 0.95,
                "inlier_grid_entropy": 0.95,
                "anchor_health_score": 0.90,
                "selected_reference_count": 3,
            },
            min_num_inliers=100,
            recent_pose_fail_rate=0.0,
            current_Rt=_rt(2.0),
            pose_history=[(18, _rt(0.0)), (19, _rt(1.0))],
            baseline_selected=True,
            is_test=False,
            is_bootstrap=False,
        )

        self.assertLess(event["verification_risk_threshold"], event["risk_threshold"])
        self.assertGreater(event["risk_score"], event["verification_risk_threshold"])
        self.assertTrue(event["verification_trigger"])

    def test_observe_mode_records_the_same_candidate_without_running_verification(self):
        gate = PoseInitializationRiskGate(
            mode="observe_v1",
            absolute_threshold=0.10,
            adaptive_sigma=2.0,
            warmup=3,
        )
        gate.risk_history.extend([0.0020, 0.0025, 0.0030, 0.0022])

        event = gate.evaluate(
            frame_id=20,
            pose_debug={
                "match_count_total": 420,
                "num_2d3d_correspondences": 400,
                "num_pnp_inliers": 250,
                "num_miniba_inliers": 230,
            },
            viewpoint_scores={
                "inlier_grid_coverage": 0.95,
                "inlier_grid_entropy": 0.95,
                "anchor_health_score": 0.90,
                "selected_reference_count": 3,
            },
            min_num_inliers=100,
            recent_pose_fail_rate=0.0,
            current_Rt=_rt(2.0),
            pose_history=[(18, _rt(0.0)), (19, _rt(1.0))],
            baseline_selected=True,
            is_test=False,
            is_bootstrap=False,
        )

        self.assertTrue(event["verification_candidate"])
        self.assertFalse(event["verification_trigger"])
        self.assertEqual(event["decision"], "observe")

    def test_cli_accepts_verify_mode_and_pose_only_parameters(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "train.py",
                "-s",
                td,
                "-m",
                str(Path(td) / "out"),
                "--pose_initialization_risk_mode",
                "verify_v1",
                "--pose_verification_mad_scale",
                "2.8",
                "--pose_verification_min_support",
                "32",
                "--pose_verification_min_improvement",
                "0.08",
            ]
            with patch.object(sys, "argv", argv):
                args = get_args()

        self.assertEqual(args.pose_initialization_risk_mode, "verify_v1")
        self.assertAlmostEqual(args.pose_verification_mad_scale, 2.8)
        self.assertEqual(args.pose_verification_min_support, 32)
        self.assertAlmostEqual(args.pose_verification_min_improvement, 0.08)

    def test_cli_accepts_independently_validated_verify_v2_mode(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "train.py",
                "-s",
                td,
                "-m",
                str(Path(td) / "out"),
                "--pose_initialization_risk_mode",
                "verify_v2",
                "--pose_verification_candidate_mode",
                "balanced_step_v21",
            ]
            with patch.object(sys, "argv", argv):
                args = get_args()

        self.assertEqual(args.pose_initialization_risk_mode, "verify_v2")
        self.assertEqual(
            args.pose_verification_candidate_mode, "balanced_step_v21"
        )
        self.assertAlmostEqual(args.pose_verification_v2_min_improvement, 0.0)

    def test_cli_accepts_stable_anchor_reference_expansion(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "train.py",
                "-s",
                td,
                "-m",
                str(Path(td) / "out"),
                "--pose_verification_anchor_reference_mode",
                "stable_anchor_v1",
                "--pose_verification_anchor_pool_size",
                "40",
                "--pose_verification_anchor_max_refs",
                "3",
                "--pose_verification_anchor_min_age_frames",
                "36",
                "--pose_verification_anchor_min_support_count",
                "500",
                "--pose_verification_anchor_max_risk_score",
                "0.07",
                "--pose_verification_anchor_min_match_score",
                "200",
                "--pose_verification_anchor_min_source_separation",
                "18",
                "--pose_verification_anchor_candidate_scope",
                "anchor_only_v1",
                "--pose_verification_anchor_pre_error_scale",
                "3.5",
            ]
            with patch.object(sys, "argv", argv):
                args = get_args()

        self.assertEqual(
            args.pose_verification_anchor_reference_mode,
            "stable_anchor_v1",
        )
        self.assertEqual(args.pose_verification_anchor_pool_size, 40)
        self.assertEqual(args.pose_verification_anchor_max_refs, 3)
        self.assertEqual(args.pose_verification_anchor_min_age_frames, 36)
        self.assertEqual(args.pose_verification_anchor_min_support_count, 500)
        self.assertAlmostEqual(
            args.pose_verification_anchor_max_risk_score,
            0.07,
        )
        self.assertAlmostEqual(
            args.pose_verification_anchor_min_match_score,
            200.0,
        )
        self.assertEqual(
            args.pose_verification_anchor_min_source_separation,
            18,
        )
        self.assertEqual(
            args.pose_verification_anchor_candidate_scope,
            "anchor_only_v1",
        )
        self.assertAlmostEqual(
            args.pose_verification_anchor_pre_error_scale,
            3.5,
        )

    def test_cli_accepts_failed_verification_reference_guard(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "train.py",
                "-s",
                td,
                "-m",
                str(Path(td) / "out"),
                "--pose_verification_reference_policy",
                "conservative_quarantine_v1",
                "--pose_verification_reference_risk_threshold",
                "0.12",
                "--pose_verification_reference_cooldown_frames",
                "20",
            ]
            with patch.object(sys, "argv", argv):
                args = get_args()

        self.assertEqual(
            args.pose_verification_reference_policy,
            "conservative_quarantine_v1",
        )
        self.assertAlmostEqual(
            args.pose_verification_reference_risk_threshold,
            0.12,
        )
        self.assertEqual(
            args.pose_verification_reference_cooldown_frames,
            20,
        )

    def test_cli_accepts_acceptance_independent_reference_guard(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "train.py",
                "-s",
                td,
                "-m",
                str(Path(td) / "out"),
                "--pose_verification_reference_policy",
                "conservative_high_risk_v2",
            ]
            with patch.object(sys, "argv", argv):
                args = get_args()

        self.assertEqual(
            args.pose_verification_reference_policy,
            "conservative_high_risk_v2",
        )

    def test_cli_accepts_deterministic_registration_sampling(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "train.py",
                "-s",
                td,
                "-m",
                str(Path(td) / "out"),
                "--pose_verification_registration_sampling_mode",
                "frame_deterministic_v1",
            ]
            with patch.object(sys, "argv", argv):
                args = get_args()

        self.assertEqual(
            args.pose_verification_registration_sampling_mode,
            "frame_deterministic_v1",
        )

    def test_cli_accepts_deterministic_opencv_registration_solver(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "train.py",
                "-s",
                td,
                "-m",
                str(Path(td) / "out"),
                "--pose_verification_registration_solver_mode",
                "deterministic_opencv_v2",
            ]
            with patch.object(sys, "argv", argv):
                args = get_args()

        self.assertEqual(
            args.pose_verification_registration_solver_mode,
            "deterministic_opencv_v2",
        )

    def test_cli_accepts_fixed_async_joint_pose_budget(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "train.py",
                "-s",
                td,
                "-m",
                str(Path(td) / "out"),
                "--pose_verification_async_pose_protection_mode",
                "fixed_joint_budget_v1",
            ]
            with patch.object(sys, "argv", argv):
                args = get_args()

        self.assertEqual(
            args.pose_verification_async_pose_protection_mode,
            "fixed_joint_budget_v1",
        )

    def test_cli_accepts_opt_in_frozen_sparse_reference_geometry(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "train.py",
                "-s",
                td,
                "-m",
                str(Path(td) / "out"),
                "--pose_verification_reference_geometry_mode",
                "frozen_first_valid_v1",
            ]
            with patch.object(sys, "argv", argv):
                args = get_args()

        self.assertEqual(
            args.pose_verification_reference_geometry_mode,
            "frozen_first_valid_v1",
        )

    def test_cli_accepts_guarded_frozen_reference_geometry(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "train.py",
                "-s",
                td,
                "-m",
                str(Path(td) / "out"),
                "--pose_verification_reference_geometry_mode",
                "guarded_frozen_v3",
                "--pose_verification_frozen_min_match_support",
                "12",
                "--pose_verification_frozen_min_live_ratio",
                "0.6",
            ]
            with patch.object(sys, "argv", argv):
                args = get_args()

        self.assertEqual(
            args.pose_verification_reference_geometry_mode,
            "guarded_frozen_v3",
        )
        self.assertEqual(
            args.pose_verification_frozen_min_match_support,
            12,
        )
        self.assertAlmostEqual(
            args.pose_verification_frozen_min_live_ratio,
            0.6,
        )

    def test_cli_accepts_guarded_frozen_global_points_with_live_pose(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "train.py",
                "-s",
                td,
                "-m",
                str(Path(td) / "out"),
                "--pose_verification_reference_geometry_mode",
                "guarded_frozen_live_pose_v4",
            ]
            with patch.object(sys, "argv", argv):
                args = get_args()

        self.assertEqual(
            args.pose_verification_reference_geometry_mode,
            "guarded_frozen_live_pose_v4",
        )

    def test_cli_accepts_frame_homogeneous_frozen_geometry(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "train.py",
                "-s",
                td,
                "-m",
                str(Path(td) / "out"),
                "--pose_verification_reference_geometry_mode",
                "guarded_frozen_homogeneous_v5",
                "--pose_verification_frozen_min_reference_count",
                "3",
                "--pose_verification_frozen_min_total_support",
                "72",
            ]
            with patch.object(sys, "argv", argv):
                args = get_args()

        self.assertEqual(
            args.pose_verification_reference_geometry_mode,
            "guarded_frozen_homogeneous_v5",
        )
        self.assertEqual(
            args.pose_verification_frozen_min_reference_count,
            3,
        )
        self.assertEqual(
            args.pose_verification_frozen_min_total_support,
            72,
        )

    def test_cli_accepts_global_frozen_support_guard(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "train.py",
                "-s",
                td,
                "-m",
                str(Path(td) / "out"),
                "--pose_verification_reference_geometry_mode",
                "frozen_global_support_guard_v6",
                "--pose_verification_frozen_min_total_support",
                "24",
            ]
            with patch.object(sys, "argv", argv):
                args = get_args()

        self.assertEqual(
            args.pose_verification_reference_geometry_mode,
            "frozen_global_support_guard_v6",
        )
        self.assertEqual(
            args.pose_verification_frozen_min_total_support,
            24,
        )

    def test_frame_homogeneous_policy_uses_only_supported_frozen_refs(self):
        mode, indices, debug = (
            PoseInitializer._select_guarded_frozen_frame_policy(
                [31, 0, 29, 0],
                total_reference_count=4,
                min_reference_count=2,
                min_total_support=48,
            )
        )

        self.assertEqual(mode, "frozen_subset")
        self.assertEqual(indices, [0, 2])
        self.assertEqual(debug["candidate_total_support"], 60)

    def test_frame_homogeneous_policy_falls_back_all_live_atomically(self):
        mode, indices, debug = (
            PoseInitializer._select_guarded_frozen_frame_policy(
                [47, 0, 0],
                total_reference_count=3,
                min_reference_count=2,
                min_total_support=48,
            )
        )

        self.assertEqual(mode, "live_fallback")
        self.assertEqual(indices, [0, 1, 2])
        self.assertEqual(debug["candidate_reference_count"], 1)

    def test_global_frozen_support_policy_preserves_all_frozen_references(self):
        mode, indices, debug = (
            PoseInitializer._select_frozen_global_support_policy(
                [12, 8, 14],
                all_snapshots_available=True,
                total_reference_count=3,
                min_total_support=24,
            )
        )

        self.assertEqual(mode, "frozen_all")
        self.assertEqual(indices, [0, 1, 2])
        self.assertEqual(debug["candidate_total_support"], 34)

    def test_global_frozen_support_policy_falls_back_below_pnp_support(self):
        mode, indices, debug = (
            PoseInitializer._select_frozen_global_support_policy(
                [3, 4, 0],
                all_snapshots_available=True,
                total_reference_count=3,
                min_total_support=24,
            )
        )

        self.assertEqual(mode, "live_fallback")
        self.assertEqual(indices, [0, 1, 2])
        self.assertEqual(debug["candidate_total_support"], 7)

    def test_guarded_frozen_geometry_falls_back_when_match_support_is_stale(self):
        class _Desc:
            def __init__(self):
                self.pts3d = torch.arange(15, dtype=torch.float32).view(5, 3)
                self.pts_conf = torch.ones(5)
                self.has_pt3d = torch.ones(5, dtype=torch.bool)

        class _Keyframe:
            def __init__(self):
                self.desc_kpts = _Desc()
                self._pose_verification_geometry_snapshot_mode = (
                    "guarded_frozen_v3"
                )
                self._pose_verification_frozen_min_match_support = 3
                self._pose_verification_frozen_min_live_ratio = 0.5
                self.snapshot = (
                    self.desc_kpts.pts3d + 100.0,
                    torch.ones(5),
                    torch.tensor([True, False, False, False, False]),
                )
                self.current_rt = torch.eye(4)
                self.snapshot_rt = torch.eye(4)
                self.snapshot_rt[0, 3] = 1.0

            def get_pose_verification_geometry(self):
                return self.snapshot

            def get_pose_verification_geometry_Rt(self):
                return self.snapshot_rt

            def get_geometry_Rt(self):
                return self.current_rt

        keyframe = _Keyframe()
        points, confidence, mask, pose, used_snapshot, debug = (
            PoseInitializer._keyframe_pose_reference_bundle(
                keyframe,
                torch.arange(5),
                verification_evidence=False,
            )
        )

        self.assertIs(points, keyframe.desc_kpts.pts3d)
        self.assertIs(confidence, keyframe.desc_kpts.pts_conf)
        self.assertIs(mask, keyframe.desc_kpts.has_pt3d)
        self.assertIs(pose, keyframe.current_rt)
        self.assertFalse(used_snapshot)
        self.assertEqual(debug["reason"], "frozen_support_too_low")

    def test_guarded_frozen_geometry_uses_coherent_snapshot_when_supported(self):
        class _Desc:
            def __init__(self):
                self.pts3d = torch.arange(15, dtype=torch.float32).view(5, 3)
                self.pts_conf = torch.ones(5)
                self.has_pt3d = torch.ones(5, dtype=torch.bool)

        class _Keyframe:
            def __init__(self):
                self.desc_kpts = _Desc()
                self._pose_verification_geometry_snapshot_mode = (
                    "guarded_frozen_v3"
                )
                self._pose_verification_frozen_min_match_support = 3
                self._pose_verification_frozen_min_live_ratio = 0.5
                self.snapshot = (
                    self.desc_kpts.pts3d + 100.0,
                    torch.ones(5),
                    torch.tensor([True, True, True, False, False]),
                )
                self.current_rt = torch.eye(4)
                self.snapshot_rt = torch.eye(4)
                self.snapshot_rt[0, 3] = 1.0

            def get_pose_verification_geometry(self):
                return self.snapshot

            def get_pose_verification_geometry_Rt(self):
                return self.snapshot_rt

            def get_geometry_Rt(self):
                return self.current_rt

        keyframe = _Keyframe()
        points, _, _, pose, used_snapshot, debug = (
            PoseInitializer._keyframe_pose_reference_bundle(
                keyframe,
                torch.arange(5),
                verification_evidence=False,
            )
        )

        self.assertIs(points, keyframe.snapshot[0])
        self.assertIs(pose, keyframe.snapshot_rt)
        self.assertTrue(used_snapshot)
        self.assertEqual(debug["reason"], "frozen_support_guard_passed")

    def test_guarded_frozen_global_points_keep_the_live_reference_pose(self):
        class _Desc:
            def __init__(self):
                self.pts3d = torch.arange(15, dtype=torch.float32).view(5, 3)
                self.pts_conf = torch.ones(5)
                self.has_pt3d = torch.ones(5, dtype=torch.bool)

        class _Keyframe:
            def __init__(self):
                self.desc_kpts = _Desc()
                self._pose_verification_geometry_snapshot_mode = (
                    "guarded_frozen_live_pose_v4"
                )
                self._pose_verification_frozen_min_match_support = 3
                self._pose_verification_frozen_min_live_ratio = 0.5
                self.snapshot = (
                    self.desc_kpts.pts3d + 100.0,
                    torch.ones(5),
                    torch.tensor([True, True, True, False, False]),
                )
                self.current_rt = torch.eye(4)
                self.snapshot_rt = torch.eye(4)
                self.snapshot_rt[0, 3] = 1.0

            def get_pose_verification_geometry(self):
                return self.snapshot

            def get_pose_verification_geometry_Rt(self):
                return self.snapshot_rt

            def get_geometry_Rt(self):
                return self.current_rt

        keyframe = _Keyframe()
        points, _, _, pose, used_snapshot, debug = (
            PoseInitializer._keyframe_pose_reference_bundle(
                keyframe,
                torch.arange(5),
                verification_evidence=False,
            )
        )

        self.assertIs(points, keyframe.snapshot[0])
        self.assertIs(pose, keyframe.current_rt)
        self.assertTrue(used_snapshot)
        self.assertEqual(
            debug["reason"],
            "frozen_support_guard_passed_live_pose",
        )

    def test_verification_only_geometry_keeps_initial_pose_live(self):
        class _Desc:
            def __init__(self):
                self.pts3d = torch.tensor([[1.0, 2.0, 3.0]])
                self.pts_conf = torch.tensor([0.7])
                self.has_pt3d = torch.tensor([True])

        class _Keyframe:
            def __init__(self):
                self.desc_kpts = _Desc()
                self._pose_verification_geometry_snapshot_mode = (
                    "frozen_verification_only_v2"
                )
                self.snapshot = (
                    torch.tensor([[4.0, 5.0, 6.0]]),
                    torch.tensor([0.9]),
                    torch.tensor([True]),
                )

            def get_pose_verification_geometry(self):
                return self.snapshot

        keyframe = _Keyframe()
        initial_geometry = PoseInitializer._keyframe_pose_reference_geometry(
            keyframe,
            verification_evidence=False,
        )
        verification_geometry = PoseInitializer._keyframe_pose_reference_geometry(
            keyframe,
            verification_evidence=True,
        )

        self.assertIs(initial_geometry[0], keyframe.desc_kpts.pts3d)
        self.assertIs(initial_geometry[1], keyframe.desc_kpts.pts_conf)
        self.assertIs(initial_geometry[2], keyframe.desc_kpts.has_pt3d)
        self.assertIs(verification_geometry, keyframe.snapshot)

    def test_cli_accepts_opt_in_photometric_pose_review_controls(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "train.py",
                "-s",
                td,
                "-m",
                str(Path(td) / "out"),
                "--pose_risk_utility_use_verification_candidates",
                "--pose_risk_utility_review_test_candidates",
                "--pose_risk_utility_review_min_relative_improvement",
                "0.01",
            ]
            with patch.object(sys, "argv", argv):
                args = get_args()

        self.assertTrue(args.pose_risk_utility_use_verification_candidates)
        self.assertTrue(args.pose_risk_utility_review_test_candidates)
        self.assertAlmostEqual(
            args.pose_risk_utility_review_min_relative_improvement,
            0.01,
        )

    def test_cli_accepts_a_specific_photometric_review_controls(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "train.py",
                "-s",
                td,
                "-m",
                str(Path(td) / "out"),
                "--pose_verification_photometric_review",
                "--pose_verification_photometric_iterations",
                "4",
                "--pose_verification_photometric_min_relative_improvement",
                "0.01",
                "--pose_verification_photometric_min_support_ratio",
                "0.95",
                "--pose_verification_photometric_scope",
                "test_only",
                "--pose_verification_photometric_seed",
                "raw",
                "--pose_verification_photometric_lr_scale",
                "5.0",
            ]
            with patch.object(sys, "argv", argv):
                args = get_args()

        self.assertTrue(args.pose_verification_photometric_review)
        self.assertEqual(args.pose_verification_photometric_iterations, 4)
        self.assertAlmostEqual(
            args.pose_verification_photometric_min_relative_improvement,
            0.01,
        )
        self.assertAlmostEqual(
            args.pose_verification_photometric_min_support_ratio,
            0.95,
        )
        self.assertEqual(
            args.pose_verification_photometric_scope,
            "test_only",
        )
        self.assertEqual(
            args.pose_verification_photometric_seed,
            "raw",
        )
        self.assertAlmostEqual(
            args.pose_verification_photometric_lr_scale,
            5.0,
        )

    def test_pose_initializer_accepts_verified_candidate_and_records_diagnostics(self):
        initializer = PoseInitializer.__new__(PoseInitializer)
        initializer.f = torch.tensor([100.0])
        initializer.centre = torch.tensor([50.0, 50.0])
        initializer.max_pnp_error = 10.0
        initializer.num_pts_miniba_incr = 4
        initializer.miniBA_incr = _FakeMiniBA(translation_x=0.01)
        initializer.PnPRANSAC = _FakePnP(translation_x=0.01)
        initializer.last_incremental_pose_support = {
            "uvs": torch.tensor(
                [[51.0, 50.0], [61.0, 50.0], [51.0, 60.0], [61.0, 60.0]]
            ),
            "pts3d": torch.tensor(
                [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0], [0.1, 0.1, 1.0]]
            ),
            "pts_conf": torch.ones(4),
            "match_indices": torch.arange(4),
        }
        initializer.last_incremental_pose_candidates = {
            "uvs": torch.tensor(
                [
                    [51.0, 50.0],
                    [61.0, 50.0],
                    [51.0, 60.0],
                    [61.0, 60.0],
                    [41.0, 50.0],
                    [51.0, 40.0],
                ]
            ),
            "pts3d": torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [0.1, 0.0, 1.0],
                    [0.0, 0.1, 1.0],
                    [0.1, 0.1, 1.0],
                    [-0.1, 0.0, 1.0],
                    [0.0, -0.1, 1.0],
                ]
            ),
            "pts_conf": torch.ones(6),
            "match_indices": torch.arange(6),
        }

        verified, debug = initializer.verify_incremental_pose(
            _rt(),
            {"verification_trigger": True, "risk_score": 0.8},
            image_width=100,
            image_height=100,
            pose_history=[(7, _rt()), (8, _rt()), (9, _rt())],
            min_support=4,
            min_relative_median_improvement=0.05,
        )

        self.assertTrue(debug["triggered"])
        self.assertTrue(debug["accepted"])
        self.assertEqual(debug["reason"], "accepted")
        self.assertAlmostEqual(float(verified[0, 3]), 0.01, places=5)
        self.assertEqual(initializer.miniBA_incr.calls, 1)
        self.assertEqual(initializer.PnPRANSAC.calls, 1)
        self.assertEqual(debug["verification_pnp_inliers"], 6)
        self.assertEqual(initializer.PnPRANSAC.last_count, 6)
        self.assertEqual(debug["candidate_source"], "full_2d3d")
        self.assertLess(debug["post_reprojection_median"], debug["pre_reprojection_median"])

    def test_pose_initializer_v2_solves_and_validates_on_disjoint_references(self):
        initializer = PoseInitializer.__new__(PoseInitializer)
        initializer.f = torch.tensor([100.0])
        initializer.centre = torch.tensor([50.0, 50.0])
        initializer.max_pnp_error = 10.0
        initializer.num_pts_miniba_incr = 4
        initializer.miniBA_incr = _FakeMiniBA(translation_x=0.01)
        initializer.PnPRANSAC = _FakePnP(translation_x=0.01)
        initializer.last_incremental_pose_support = {}
        initializer.last_incremental_pose_candidates = {
            "uvs": torch.tensor(
                [
                    [51.0, 50.0],
                    [61.0, 50.0],
                    [51.0, 60.0],
                    [61.0, 60.0],
                    [41.0, 50.0],
                    [51.0, 40.0],
                    [41.0, 40.0],
                    [61.0, 40.0],
                ]
            ),
            "pts3d": torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [0.1, 0.0, 1.0],
                    [0.0, 0.1, 1.0],
                    [0.1, 0.1, 1.0],
                    [-0.1, 0.0, 1.0],
                    [0.0, -0.1, 1.0],
                    [-0.1, -0.1, 1.0],
                    [0.1, -0.1, 1.0],
                ]
            ),
            "pts_conf": torch.ones(8),
            "match_indices": torch.arange(8),
            "corr_ref_ids": torch.tensor([1, 1, 1, 1, 2, 2, 2, 2]),
        }

        verified, debug = initializer.verify_incremental_pose(
            _rt(),
            {"verification_trigger": True, "risk_score": 0.8},
            image_width=100,
            image_height=100,
            pose_history=[(7, _rt()), (8, _rt()), (9, _rt())],
            min_support=4,
            min_relative_median_improvement=0.05,
            independent_validation=True,
        )

        self.assertTrue(debug["accepted"])
        self.assertEqual(debug["split_strategy"], "reference_holdout")
        self.assertEqual(debug["solve_reference_ids"], [1])
        self.assertEqual(debug["validation_reference_ids"], [2])
        self.assertEqual(debug["solve_count"], 4)
        self.assertEqual(debug["validation_count"], 4)
        self.assertEqual(initializer.PnPRANSAC.last_count, 4)
        self.assertAlmostEqual(float(verified[0, 3]), 0.01, places=5)
        self.assertLess(
            debug["post_reprojection_median"], debug["pre_reprojection_median"]
        )

    def test_pose_initializer_selects_a_validated_partial_pose_step(self):
        initializer = PoseInitializer.__new__(PoseInitializer)
        initializer.f = torch.tensor([100.0])
        initializer.centre = torch.tensor([50.0, 50.0])
        initializer.max_pnp_error = 10.0
        initializer.num_pts_miniba_incr = 4
        initializer.miniBA_incr = _FakeMiniBA(translation_x=0.02)
        initializer.PnPRANSAC = _FakePnP(translation_x=0.02)
        initializer.last_incremental_pose_support = {}
        initializer.last_incremental_pose_candidates = {
            "uvs": torch.tensor(
                [
                    [51.0, 50.0],
                    [61.0, 50.0],
                    [51.0, 60.0],
                    [61.0, 60.0],
                    [41.0, 50.0],
                    [51.0, 40.0],
                    [41.0, 40.0],
                    [61.0, 40.0],
                ]
            ),
            "pts3d": torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [0.1, 0.0, 1.0],
                    [0.0, 0.1, 1.0],
                    [0.1, 0.1, 1.0],
                    [-0.1, 0.0, 1.0],
                    [0.0, -0.1, 1.0],
                    [-0.1, -0.1, 1.0],
                    [0.1, -0.1, 1.0],
                ]
            ),
            "pts_conf": torch.ones(8),
            "match_indices": torch.arange(8),
            "corr_ref_ids": torch.tensor([1, 1, 1, 1, 2, 2, 2, 2]),
        }

        verified, debug = initializer.verify_incremental_pose(
            _rt(),
            {"verification_trigger": True, "risk_score": 0.8},
            image_width=100,
            image_height=100,
            pose_history=[(7, _rt()), (8, _rt()), (9, _rt())],
            min_support=4,
            min_relative_median_improvement=0.05,
            independent_validation=True,
            candidate_mode="balanced_step_v21",
        )

        self.assertTrue(debug["accepted"])
        self.assertAlmostEqual(debug["selected_step_alpha"], 0.5, places=5)
        self.assertAlmostEqual(float(verified[0, 3]), 0.01, places=5)
        self.assertLess(
            debug["post_reprojection_median"], debug["pre_reprojection_median"]
        )
        self.assertEqual(len(debug["step_candidates"]), 4)

    def test_multihypothesis_mode_can_select_an_independent_pnp_candidate(self):
        initializer = PoseInitializer.__new__(PoseInitializer)
        initializer.f = torch.tensor([100.0])
        initializer.centre = torch.tensor([50.0, 50.0])
        initializer.max_pnp_error = 10.0
        initializer.num_pts_miniba_incr = 4
        initializer.miniBA_incr = _FakeMiniBA(translation_x=-0.02)
        initializer.PnPRANSAC = _FakePnP(translation_x=-0.02)
        initializer.last_incremental_pose_support = {}
        initializer.last_incremental_pose_candidates = {
            "uvs": torch.tensor(
                [
                    [51.0, 50.0],
                    [61.0, 50.0],
                    [51.0, 60.0],
                    [61.0, 60.0],
                    [41.0, 50.0],
                    [51.0, 40.0],
                    [41.0, 40.0],
                    [61.0, 40.0],
                ]
            ),
            "pts3d": torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [0.1, 0.0, 1.0],
                    [0.0, 0.1, 1.0],
                    [0.1, 0.1, 1.0],
                    [-0.1, 0.0, 1.0],
                    [0.0, -0.1, 1.0],
                    [-0.1, -0.1, 1.0],
                    [0.1, -0.1, 1.0],
                ]
            ),
            "pts_conf": torch.ones(8),
            "match_indices": torch.arange(8),
            "corr_ref_ids": torch.tensor([1, 1, 1, 1, 2, 2, 2, 2]),
        }
        pnp_candidate = _rt()
        pnp_candidate[0, 3] = 0.01

        with patch(
            "poses.pose_initializer.estimate_opencv_pnp_candidates",
            return_value=(
                [
                    {
                        "name": "opencv_test",
                        "pose": pnp_candidate,
                        "inlier_count": 4,
                    }
                ],
                {"successful_candidates": 1},
            ),
        ):
            verified, debug = initializer.verify_incremental_pose(
                _rt(),
                {"verification_trigger": True, "risk_score": 0.8},
                image_width=100,
                image_height=100,
                pose_history=[(7, _rt()), (8, _rt()), (9, _rt())],
                min_support=4,
                min_relative_median_improvement=0.05,
                independent_validation=True,
                candidate_mode="multihypothesis_v23",
            )

        self.assertTrue(debug["accepted"])
        self.assertEqual(debug["selected_candidate_source"], "opencv_test")
        self.assertAlmostEqual(float(verified[0, 3]), 0.01, places=5)
        self.assertEqual(debug["opencv_pnp_successful_candidates"], 1)

    def test_multiview_mode_can_select_a_heldout_validated_relative_candidate(self):
        initializer = PoseInitializer.__new__(PoseInitializer)
        initializer.f = torch.tensor([100.0])
        initializer.centre = torch.tensor([50.0, 50.0])
        initializer.max_pnp_error = 10.0
        initializer.num_pts_miniba_incr = 4
        initializer.miniBA_incr = _FakeMiniBA(translation_x=-0.02)
        initializer.PnPRANSAC = _FakePnP(translation_x=-0.02)
        initializer.last_incremental_pose_support = {}
        points = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [0.0, 0.1, 1.0],
                [0.1, 0.1, 1.0],
                [-0.1, 0.0, 1.0],
                [0.0, -0.1, 1.0],
                [-0.1, -0.1, 1.0],
                [0.1, -0.1, 1.0],
            ]
        )
        initializer.last_incremental_pose_candidates = {
            "uvs": torch.tensor(
                [
                    [51.0, 50.0],
                    [61.0, 50.0],
                    [51.0, 60.0],
                    [61.0, 60.0],
                    [41.0, 50.0],
                    [51.0, 40.0],
                    [41.0, 40.0],
                    [61.0, 40.0],
                ]
            ),
            "pts3d": points,
            "pts_conf": torch.ones(8),
            "match_indices": torch.arange(8),
            "corr_ref_ids": torch.tensor([1, 1, 1, 1, 2, 2, 2, 2]),
            "ref_uvs": torch.tensor(
                [
                    [50.0, 50.0],
                    [60.0, 50.0],
                    [50.0, 60.0],
                    [60.0, 60.0],
                    [40.0, 50.0],
                    [50.0, 40.0],
                    [40.0, 40.0],
                    [60.0, 40.0],
                ]
            ),
            "ref_Rts": torch.stack([_rt()] * 8),
        }
        relative_candidate = _rt()
        relative_candidate[0, 3] = 0.01
        history = []
        for frame_id, translation_x in ((7, 0.04), (8, 0.03), (9, 0.02)):
            history_pose = _rt()
            history_pose[0, 3] = translation_x
            history.append((frame_id, history_pose))

        with (
            patch(
                "poses.pose_initializer.estimate_opencv_pnp_candidates",
                return_value=([], {"successful_candidates": 0}),
            ),
            patch(
                "poses.pose_initializer."
                "estimate_multireference_relative_pose_candidates",
                return_value=(
                    [
                        {
                            "name": "relative_test",
                            "pose": relative_candidate,
                            "inlier_count": 8,
                        }
                    ],
                    {"successful_candidates": 1},
                ),
            ),
        ):
            verified, debug = initializer.verify_incremental_pose(
                _rt(),
                {
                    "verification_trigger": True,
                    "risk_score": 0.8,
                    "frame_id": 10,
                },
                image_width=100,
                image_height=100,
                pose_history=history,
                min_support=4,
                min_relative_median_improvement=0.05,
                independent_validation=True,
                candidate_mode="multiview_relative_v24",
                max_temporal_score_ratio=0.90,
            )

        self.assertTrue(debug["accepted"])
        self.assertEqual(debug["selected_candidate_source"], "relative_test")
        self.assertAlmostEqual(float(verified[0, 3]), 0.01, places=5)
        self.assertEqual(debug["relative_pose_successful_candidates"], 1)
        self.assertLessEqual(debug["selected_temporal_score_ratio"], 0.90)

    def test_epipolar_validation_can_override_map_biased_reprojection(self):
        initializer = PoseInitializer.__new__(PoseInitializer)
        initializer.f = torch.tensor([100.0])
        initializer.centre = torch.tensor([50.0, 50.0])
        initializer.max_pnp_error = 10.0
        initializer.num_pts_miniba_incr = 4
        initializer.miniBA_incr = _FakeMiniBA(
            translation_x=0.15,
            rotation_deg=5.0,
        )
        initializer.PnPRANSAC = _FakePnP(
            translation_x=0.15,
            rotation_deg=5.0,
        )
        initializer.last_incremental_pose_support = {}

        true_pose = _rt(5.0)
        true_pose[0, 3] = 0.15
        reference_one = _rt()
        reference_one[0, 3] = -0.20
        reference_two = _rt()
        reference_two[1, 3] = 0.25
        points = torch.tensor(
            [
                [-0.20, -0.10, 2.0],
                [0.10, -0.20, 2.4],
                [0.30, 0.10, 2.8],
                [-0.10, 0.30, 2.2],
                [0.40, -0.30, 3.2],
                [-0.30, 0.20, 2.7],
                [0.20, 0.35, 3.0],
                [-0.35, -0.25, 2.5],
            ]
        )

        def project(pose: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
            camera = xyz @ pose[:3, :3].transpose(0, 1) + pose[:3, 3]
            return camera[:, :2] / camera[:, 2:] * initializer.f + initializer.centre

        current_uv = project(true_pose, points)
        reference_poses = torch.stack(
            [reference_one] * 4 + [reference_two] * 4
        )
        reference_uv = torch.cat(
            (
                project(reference_one, points[:4]),
                project(reference_two, points[4:]),
            )
        )
        biased_depth = torch.full((len(points), 1), 2.5)
        biased_xyz = torch.cat(
            (
                (current_uv - initializer.centre) / initializer.f * biased_depth,
                biased_depth,
            ),
            dim=-1,
        )
        initializer.last_incremental_pose_candidates = {
            "uvs": current_uv,
            "pts3d": biased_xyz,
            "pts_conf": torch.ones(8),
            "match_indices": torch.arange(8),
            "corr_ref_ids": torch.tensor([1, 1, 1, 1, 2, 2, 2, 2]),
            "ref_uvs": reference_uv,
            "ref_Rts": reference_poses,
        }

        verified, debug = initializer.verify_incremental_pose(
            _rt(),
            {"verification_trigger": True, "risk_score": 0.8},
            image_width=100,
            image_height=100,
            pose_history=[(7, _rt()), (8, _rt()), (9, _rt())],
            min_support=4,
            min_relative_median_improvement=0.05,
            independent_validation=True,
            candidate_mode="balanced_epipolar_v22",
        )

        self.assertTrue(debug["accepted"])
        self.assertGreater(debug["selected_step_alpha"], 0.0)
        self.assertGreater(abs(float(verified[0, 3])), 0.01)
        self.assertLess(
            debug["post_epipolar_median"], debug["pre_epipolar_median"]
        )
        self.assertGreater(
            debug["post_reprojection_median"], debug["pre_reprojection_median"]
        )

    def test_pose_initializer_v2_falls_back_before_solver_without_holdout_support(self):
        initializer = PoseInitializer.__new__(PoseInitializer)
        initializer.f = torch.tensor([100.0])
        initializer.centre = torch.tensor([50.0, 50.0])
        initializer.max_pnp_error = 10.0
        initializer.num_pts_miniba_incr = 4
        initializer.miniBA_incr = _FakeMiniBA(translation_x=0.01)
        initializer.PnPRANSAC = _FakePnP(translation_x=0.01)
        initializer.last_incremental_pose_support = {}
        initializer.last_incremental_pose_candidates = {
            "uvs": torch.tensor(
                [
                    [51.0, 50.0],
                    [61.0, 50.0],
                    [51.0, 60.0],
                    [61.0, 60.0],
                    [41.0, 50.0],
                    [51.0, 40.0],
                ]
            ),
            "pts3d": torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [0.1, 0.0, 1.0],
                    [0.0, 0.1, 1.0],
                    [0.1, 0.1, 1.0],
                    [-0.1, 0.0, 1.0],
                    [0.0, -0.1, 1.0],
                ]
            ),
            "pts_conf": torch.ones(6),
            "match_indices": torch.arange(6),
            "corr_ref_ids": torch.tensor([1, 1, 1, 1, 1, 2]),
        }
        initial = _rt()

        verified, debug = initializer.verify_incremental_pose(
            initial,
            {"verification_trigger": True, "risk_score": 0.8},
            image_width=100,
            image_height=100,
            pose_history=[(7, _rt()), (8, _rt()), (9, _rt())],
            min_support=4,
            independent_validation=True,
        )

        self.assertFalse(debug["accepted"])
        self.assertEqual(debug["reason"], "insufficient_independent_support")
        self.assertTrue(torch.equal(verified, initial))
        self.assertNotEqual(verified.data_ptr(), initial.data_ptr())
        self.assertEqual(initializer.PnPRANSAC.calls, 0)
        self.assertEqual(initializer.miniBA_incr.calls, 0)

    def test_pose_initializer_bypass_is_exact_and_does_not_call_solver(self):
        initializer = PoseInitializer.__new__(PoseInitializer)
        initializer.miniBA_incr = _FakeMiniBA(translation_x=0.01)
        initial = _rt()

        verified, debug = initializer.verify_incremental_pose(
            initial,
            {"verification_trigger": False, "risk_score": 0.1},
            image_width=100,
            image_height=100,
            pose_history=[],
        )

        self.assertTrue(torch.equal(verified, initial))
        self.assertNotEqual(verified.data_ptr(), initial.data_ptr())
        self.assertEqual(debug["reason"], "risk_not_triggered")
        self.assertEqual(initializer.miniBA_incr.calls, 0)

    def test_train_invokes_pose_verification_before_keyframe_construction(self):
        source = Path("train.py").read_text(encoding="utf-8")
        risk_index = source.index("pose_initialization_risk_gate.evaluate")
        verification_index = source.index("pose_initializer.verify_incremental_pose")
        keyframe_index = source.index("keyframe = Keyframe(", verification_index)

        self.assertLess(risk_index, verification_index)
        self.assertLess(verification_index, keyframe_index)
        self.assertIn('"initial_estimated_Rt": _pose_matrix_for_trace(Rt)', source)
        self.assertIn('"gt_Rt": _pose_matrix_for_trace(info.get("Rt"))', source)

    def test_train_routes_verify_v2_to_independent_validation_and_stage_trace(self):
        source = Path("train.py").read_text(encoding="utf-8")

        self.assertIn(
            'pose_initialization_risk_mode in {"verify_v1", "verify_v2"}',
            source,
        )
        self.assertIn(
            'independent_validation=(pose_initialization_risk_mode == "verify_v2")',
            source,
        )
        self.assertIn('"pose_verification_candidate_mode"', source)
        self.assertIn("candidate_mode=str(", source)
        self.assertIn('"pose_verification_v2_min_improvement"', source)
        self.assertIn(
            '"pose_verification_max_temporal_score_ratio"',
            source,
        )
        self.assertIn('"post_a_Rt": _pose_matrix_for_trace(Rt)', source)
        self.assertIn('"a_stage_is_test": bool(info.get("is_test", False))', source)

    def test_train_builds_anchor_evidence_after_risk_trigger_and_before_verification(self):
        source = Path("train.py").read_text(encoding="utf-8")
        risk_index = source.index("pose_initialization_risk_gate.evaluate")
        anchor_index = source.index("pose_verification_anchor_evidence")
        verification_index = source.index(
            "pose_initializer.verify_incremental_pose",
            anchor_index,
        )

        self.assertLess(risk_index, anchor_index)
        self.assertLess(anchor_index, verification_index)
        self.assertIn("sample_stable_pose_anchor_candidates(", source)
        self.assertIn("rank_stable_pose_anchor_records(", source)
        self.assertIn(
            "resolve_stable_pose_anchor_probe_references(",
            source,
        )
        self.assertIn("_snapshot_pose_match_state(", source)
        self.assertIn("_restore_pose_match_state(", source)
        self.assertIn(
            "pose_evidence_override=pose_verification_anchor_evidence",
            source,
        )

    def test_train_quarantines_only_failed_high_risk_verification_references(self):
        source = Path("train.py").read_text(encoding="utf-8")
        verification_index = source.index(
            "pose_initializer.verify_incremental_pose"
        )
        guard_index = source.index(
            "decide_failed_verification_reference_quarantine(",
            verification_index,
        )
        keyframe_index = source.index("keyframe = Keyframe(", guard_index)

        self.assertLess(verification_index, guard_index)
        self.assertLess(guard_index, keyframe_index)
        self.assertIn(
            'info["_pose_reference_quarantined"] = True',
            source,
        )
        self.assertIn(
            '"verification_reference_quarantined"',
            source,
        )
        self.assertIn(
            "or pose_verification_reference_guard_enabled",
            source,
        )

    def test_train_routes_independent_frame_seed_to_registration_and_verification(self):
        source = Path("train.py").read_text(encoding="utf-8")

        self.assertIn(
            "pose_verification_registration_sampling_mode",
            source,
        )
        self.assertIn(
            "sampling_seed=pose_registration_sampling_seed",
            source,
        )
        self.assertIn(
            "sampling_seed=pose_verification_sampling_seed",
            source,
        )
        self.assertIn(
            "registration_solver_mode=pose_registration_solver_mode",
            source,
        )

    def test_scene_model_protects_pose_after_fixed_async_joint_budget(self):
        source = Path("scene/scene_model.py").read_text(encoding="utf-8")

        self.assertIn("pose_verification_async_pose_protection_mode", source)
        self.assertIn("async_pose_update_enabled(", source)
        self.assertIn("update_pose=pose_update_enabled", source)

    def test_train_routes_opt_in_candidates_and_review_threshold(self):
        train_source = Path("train.py").read_text(encoding="utf-8")
        scene_source = Path("scene/scene_model.py").read_text(encoding="utf-8")

        self.assertIn("pose_review_candidate(", train_source)
        self.assertIn(
            "use_verification_candidates=bool(",
            train_source,
        )
        self.assertIn(
            "review_test_candidates=bool(",
            train_source,
        )
        self.assertIn(
            '"pose_risk_utility_review_min_relative_improvement"',
            train_source,
        )
        self.assertIn(
            "min_relative_loss_improvement=min_relative_loss_improvement",
            scene_source,
        )

    def test_train_runs_a_photometric_review_before_gaussian_insertion(self):
        train_source = Path("train.py").read_text(encoding="utf-8")
        scene_source = Path("scene/scene_model.py").read_text(encoding="utf-8")
        add_keyframe_index = train_source.index("scene_model.add_keyframe(keyframe)")
        review_index = train_source.index(
            "pose_verification_photometric_review",
            add_keyframe_index,
        )
        add_gaussians_index = train_source.index(
            "scene_model.add_new_gaussians()",
            review_index,
        )

        self.assertLess(add_keyframe_index, review_index)
        self.assertLess(review_index, add_gaussians_index)
        self.assertIn(
            'pose_initialization_risk_mode == "verify_v2"',
            train_source,
        )
        self.assertIn(
            '"verification_trigger", False',
            train_source,
        )
        self.assertIn('"geometric_verification_accepted"', train_source)
        self.assertIn('"photometric_verification_accepted"', train_source)
        self.assertIn('"post_a_Rt": _pose_matrix_for_trace(Rt)', train_source)
        self.assertIn("initial_validation_support", scene_source)
        self.assertIn("validation_support_ratio", scene_source)
        self.assertIn("snapshot_optimizer_parameter_state(", scene_source)
        self.assertIn("restore_optimizer_parameter_state(", scene_source)
        self.assertIn("scale_optimizer_parameter_learning_rates(", scene_source)
        self.assertIn("restore_optimizer_parameter_learning_rates(", scene_source)
        self.assertIn('"pose_verification_photometric_lr_scale"', train_source)
        self.assertIn('"pose_verification_photometric_scope"', train_source)
        self.assertIn('"pose_verification_photometric_seed"', train_source)
        self.assertIn('"verification_candidate"', train_source)
        self.assertIn('"test_only"', train_source)
        self.assertIn("restore_values=not accepted", scene_source)


if __name__ == "__main__":
    unittest.main()
