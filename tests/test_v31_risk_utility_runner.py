import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.compare_v31_risk_utility_results import (
    canonical_frame_id,
    evaluate_three_way_pose,
)
from tools.run_v31_risk_utility_experiment import (
    VARIANTS,
    build_command,
    build_specs,
    parse_args,
)


def w2c_at(x, y=0.0):
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = -float(x)
    pose[1, 3] = -float(y)
    return pose


class V31RiskUtilityRunnerTests(unittest.TestCase):
    def test_comparison_uses_final_metadata_for_all_pose_trajectories(self):
        source = Path("tools/compare_v31_risk_utility_results.py").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            '"new": metadata_trajectory(method_dirs["new"])',
            source,
        )
        self.assertIn('"pose_reference_quarantined"', source)

    def test_canonical_frame_id_aligns_padded_names(self):
        self.assertEqual(canonical_frame_id("1.jpg"), "1")
        self.assertEqual(canonical_frame_id("000001.JPG"), "1")
        self.assertEqual(canonical_frame_id("frame_02.png"), "frame_02")

    def test_runner_keeps_v31_fixed_and_varies_joint_policy(self):
        args = parse_args(["--dry_run"])
        with tempfile.TemporaryDirectory() as td:
            specs = build_specs(Path(td), ["bonsai", "forest1"])
            commands = {
                (spec.scene, spec.variant): build_command(spec, args)
                for spec in specs
            }

        self.assertEqual(args.test_hold, 0)
        self.assertEqual(
            [name for name, _ in VARIANTS],
            [
                "V31_RU_observe",
                "V31_RU_active",
                "V31_RU_active_no_review",
                "V31_RU_pose_quarantine",
            ],
        )
        for command in commands.values():
            joined = " ".join(command)
            self.assertIn("baseline_render_lock_intra_frame_v31", joined)
            self.assertIn("--pose_initialization_risk_mode observe_v1", joined)
            self.assertIn("--pose_risk_utility_threshold 0.24", joined)
            self.assertIn("--pose_risk_utility_isolation_risk_margin 0.04", joined)
            self.assertIn(
                "--pose_risk_utility_isolation_cooldown_frames 24", joined
            )
            self.assertIn("--pose_risk_utility_quarantine_risk_margin 0.08", joined)
            self.assertIn(
                "--pose_risk_utility_quarantine_cooldown_frames 64", joined
            )
        self.assertIn(
            "--test_hold 8",
            " ".join(commands[("bonsai", "V31_RU_pose_quarantine")]),
        )
        self.assertIn(
            "--test_hold 10",
            " ".join(commands[("forest1", "V31_RU_pose_quarantine")]),
        )
        self.assertIn(
            "observe_v1", " ".join(commands[("bonsai", "V31_RU_observe")])
        )
        self.assertIn(
            "active_v1", " ".join(commands[("bonsai", "V31_RU_active")])
        )
        self.assertEqual(
            commands[("bonsai", "V31_RU_active_no_review")][-2:],
            ["--pose_risk_utility_review_iterations", "0"],
        )
        self.assertIn(
            "--pose_risk_utility_admission_mode pose_quarantine_v1",
            " ".join(commands[("bonsai", "V31_RU_pose_quarantine")]),
        )

    def test_three_way_pose_evaluation_uses_one_common_frame_set(self):
        gt = {
            "1": w2c_at(0.0, 0.0),
            "2": w2c_at(1.0, 0.0),
            "3": w2c_at(1.0, 1.0),
            "4": w2c_at(2.0, 1.0),
        }
        baseline = {key: pose.copy() for key, pose in gt.items()}
        v31 = {key: pose.copy() for key, pose in gt.items()}
        new = {key: pose.copy() for key, pose in gt.items()}
        new["3"][0, 3] -= 0.2
        new["extra"] = w2c_at(5.0)

        result = evaluate_three_way_pose(
            {
                "baseline": {"estimated": baseline, "gt": gt},
                "v31": {"estimated": v31, "gt": gt},
                "new": {"estimated": new, "gt": gt},
            },
            rpe_delta=1,
        )

        self.assertEqual(result["common_frame_count"], 4)
        self.assertAlmostEqual(result["baseline"]["ape_trans_mean"], 0.0)
        self.assertAlmostEqual(result["v31"]["rpe_trans_mean"], 0.0)
        self.assertGreater(result["new"]["ape_trans_mean"], 0.0)
        self.assertGreater(result["new"]["rpe_trans_mean"], 0.0)
        self.assertIn("ape_rot_deg_mean", result["new"])
        self.assertIn("rpe_rot_deg_mean", result["new"])


if __name__ == "__main__":
    unittest.main()
