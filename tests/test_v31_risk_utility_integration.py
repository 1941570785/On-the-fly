import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from args import get_args


class V31RiskUtilityIntegrationTests(unittest.TestCase):
    def test_cli_accepts_joint_admission_and_review_options(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "train.py",
                "-s",
                td,
                "-m",
                str(Path(td) / "out"),
                "--pose_risk_utility_admission_mode",
                "active_v1",
                "--pose_risk_utility_threshold",
                "0.31",
                "--pose_risk_utility_selectivity_reference",
                "1.7",
                "--pose_risk_utility_probe_downsample",
                "3",
                "--pose_risk_utility_isolation_risk_margin",
                "0.05",
                "--pose_risk_utility_isolation_cooldown_frames",
                "20",
                "--pose_risk_utility_quarantine_risk_margin",
                "0.09",
                "--pose_risk_utility_quarantine_cooldown_frames",
                "48",
                "--pose_risk_utility_review_iterations",
                "2",
                "--pose_risk_utility_review_min_coverage",
                "0.20",
                "--pose_risk_utility_review_max_rotation_deg",
                "1.2",
                "--pose_risk_utility_review_max_translation",
                "0.04",
            ]
            with patch.object(sys, "argv", argv):
                args = get_args()

        self.assertEqual(args.pose_risk_utility_admission_mode, "active_v1")
        self.assertAlmostEqual(args.pose_risk_utility_threshold, 0.31)
        self.assertAlmostEqual(args.pose_risk_utility_selectivity_reference, 1.7)
        self.assertEqual(args.pose_risk_utility_probe_downsample, 3)
        self.assertAlmostEqual(args.pose_risk_utility_isolation_risk_margin, 0.05)
        self.assertEqual(args.pose_risk_utility_isolation_cooldown_frames, 20)
        self.assertAlmostEqual(args.pose_risk_utility_quarantine_risk_margin, 0.09)
        self.assertEqual(args.pose_risk_utility_quarantine_cooldown_frames, 48)
        self.assertEqual(args.pose_risk_utility_review_iterations, 2)
        self.assertAlmostEqual(args.pose_risk_utility_review_min_coverage, 0.20)
        self.assertAlmostEqual(args.pose_risk_utility_review_max_rotation_deg, 1.2)
        self.assertAlmostEqual(args.pose_risk_utility_review_max_translation, 0.04)

    def test_training_loop_probes_only_risk_candidates(self):
        source = Path("train.py").read_text(encoding="utf-8")

        self.assertIn("PoseRiskUtilityAdmissionGate", source)
        self.assertIn("pose_risk_candidate", source)
        self.assertIn("scene_model.probe_pose_risk_utility", source)
        self.assertIn("pose_risk_utility_trace.json", source)
        self.assertIn('pose_risk_utility_decision["isolated"]', source)

    def test_high_value_review_runs_before_gaussian_growth(self):
        source = Path("train.py").read_text(encoding="utf-8")
        review_position = source.index("scene_model.review_pose_risk_keyframe")
        gaussian_position = source.index("scene_model.add_new_gaussians()", review_position)

        self.assertLess(review_position, gaussian_position)
        self.assertIn('pose_risk_utility_decision["review"]', source)

    def test_joint_policy_uses_observe_only_pose_risk_estimator(self):
        source = Path("train.py").read_text(encoding="utf-8")

        self.assertIn(
            'pose_initialization_risk_mode = "observe_v1"',
            source,
        )
        self.assertIn('"estimated_Rt": _pose_matrix_for_trace(Rt)', source)
        self.assertIn('"gt_Rt": _pose_matrix_for_trace(info.get("gt_Rt"))', source)

    def test_pose_quarantine_is_filtered_only_from_pose_reference_selection(self):
        train_source = Path("train.py").read_text(encoding="utf-8")
        scene_source = Path("scene/scene_model.py").read_text(encoding="utf-8")

        self.assertIn('info["_pose_reference_quarantined"] = True', train_source)
        self.assertIn("exclude_pose_quarantined=", train_source)
        self.assertIn("exclude_pose_quarantined: bool = False", scene_source)
        self.assertIn("filter_pose_reference_indices", scene_source)


if __name__ == "__main__":
    unittest.main()
