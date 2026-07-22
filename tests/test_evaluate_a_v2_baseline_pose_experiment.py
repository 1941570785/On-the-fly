import math
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from tools.evaluate_a_v2_baseline_pose_experiment import (
    POSE_FIELDS,
    aggregate_repeat_rows,
    compute_fixed_alignment_stage_report,
    reference_scene_spec,
)


def pose_c2w(x: float, y: float, z: float, rz_deg: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    angle = math.radians(rz_deg)
    pose[:3, :3] = [
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    pose[:3, 3] = [x, y, z]
    return pose


class AV2BaselinePoseEvaluatorTests(unittest.TestCase):
    def test_report_uses_the_trace_verification_accepted_key(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "tools" / "evaluate_a_v2_baseline_pose_experiment.py").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            'trace_summary.get("verification_accepted", 0)',
            source,
        )

    def test_reference_loader_uses_official_pose_scene_specs(self):
        for scene in (
            "bonsai",
            "counter",
            "garden",
            "forest1",
            "forest2",
            "university2",
            "desk",
            "xyz",
            "long_office",
        ):
            spec = reference_scene_spec(scene)
            self.assertEqual(spec.name, scene)
            self.assertTrue(spec.reference_path)

    def test_direct_script_entrypoint_resolves_repo_and_comparison_tools(self):
        root = Path(__file__).resolve().parents[1]
        process = subprocess.run(
            [
                sys.executable,
                str(root / "tools" / "evaluate_a_v2_baseline_pose_experiment.py"),
                "--help",
            ],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("--run_root", process.stdout)

    def test_stage_report_uses_one_anchor_alignment_and_detects_retained_gain(self):
        reference = {
            "1": pose_c2w(0.0, 0.0, 0.0),
            "2": pose_c2w(1.0, 0.0, 0.0),
            "3": pose_c2w(0.0, 1.0, 0.0),
            "4": pose_c2w(0.0, 0.0, 1.0),
        }
        initial = {name: pose.copy() for name, pose in reference.items()}
        initial["4"] = pose_c2w(0.2, 0.0, 1.0, rz_deg=5.0)
        post_a = {name: pose.copy() for name, pose in initial.items()}
        post_a["4"] = reference["4"].copy()
        final = {name: pose.copy() for name, pose in post_a.items()}

        report = compute_fixed_alignment_stage_report(
            reference,
            initial,
            post_a,
            final,
            ["1", "2", "3", "4"],
            accepted_frame_ids={"4"},
        )

        self.assertEqual(report["alignment_frame_ids"], ["1", "2", "3"])
        self.assertEqual(report["alignment_fallback"], "none")
        self.assertGreater(report["stages"]["initial"]["T_APE"], 0.0)
        self.assertLess(
            report["stages"]["post_a"]["T_APE"],
            report["stages"]["initial"]["T_APE"],
        )
        self.assertAlmostEqual(report["stages"]["post_a"]["T_APE"], 0.0)
        self.assertAlmostEqual(report["accepted_both_improved_rate"], 1.0)
        self.assertAlmostEqual(report["retention_rate"], 1.0)
        self.assertEqual(report["accepted_count"], 1)

    def test_repeat_aggregation_reports_mean_and_sample_standard_deviation(self):
        rows = [
            {
                "dataset": "MipNeRF360",
                "variant": "a_v2",
                "repeat": 1,
                "T_APE": 10.0,
                "R_APE": 0.1,
                "T_RPE": 20.0,
                "R_RPE": 0.2,
            },
            {
                "dataset": "MipNeRF360",
                "variant": "a_v2",
                "repeat": 2,
                "T_APE": 14.0,
                "R_APE": 0.3,
                "T_RPE": 24.0,
                "R_RPE": 0.4,
            },
        ]

        summary = aggregate_repeat_rows(
            rows,
            group_fields=("dataset", "variant"),
            metric_fields=POSE_FIELDS,
        )

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["n"], 2)
        self.assertAlmostEqual(summary[0]["T_APE_mean"], 12.0)
        self.assertAlmostEqual(summary[0]["T_APE_std"], math.sqrt(8.0))
        self.assertAlmostEqual(summary[0]["R_RPE_mean"], 0.3)


if __name__ == "__main__":
    unittest.main()
