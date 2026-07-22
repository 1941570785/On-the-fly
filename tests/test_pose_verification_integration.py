import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from args import get_args
from poses.pose_initializer import PoseInitializer
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
    def __init__(self, translation_x: float):
        self.translation_x = translation_x
        self.calls = 0

    def __call__(self, rotations, translations, focal, xyz, centre, uv):
        self.calls += 1
        output_translation = translations.clone()
        output_translation[0, 0] = self.translation_x
        residual = torch.zeros_like(uv)
        mask = torch.ones_like(uv)
        return rotations, output_translation, focal, xyz, residual, residual, mask


class _FakePnP:
    def __init__(self, translation_x: float):
        self.translation_x = translation_x
        self.calls = 0
        self.last_count = 0

    def __call__(self, uvs, xyz, focal, centre, rotation_init, translation_init, confs):
        self.calls += 1
        self.last_count = len(xyz)
        pose = _rt()
        pose[0, 3] = self.translation_x
        return pose[:3], torch.ones(len(xyz), dtype=torch.bool, device=xyz.device)


class PoseVerificationRiskIntegrationTests(unittest.TestCase):
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
            ]
            with patch.object(sys, "argv", argv):
                args = get_args()

        self.assertEqual(args.pose_initialization_risk_mode, "verify_v2")

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


if __name__ == "__main__":
    unittest.main()
