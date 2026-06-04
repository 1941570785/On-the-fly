from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from paper_aligned_policy.stage_metrics import build_stage_metric_evaluation
from tools.build_stage_metric_evaluation_v1 import main as stage_metric_main


TRACE = {
    "requested_mode": "on_the_fly_innovation_v1",
    "mode": "paper_aligned_semantic_v1",
    "events": [
        {
            "frame_id": 1,
            "action": "direct_admit",
            "final_keyframe_incremented": True,
            "pose_init_attempted": True,
            "pose_init_success": True,
            "num_2d3d_correspondences": 120,
            "num_pnp_inliers": 50,
            "num_miniba_inliers": 45,
            "decision_meta": {"R_t": 0.30, "V_t": 0.80, "Q_t": 0.30},
        },
        {
            "frame_id": 2,
            "action": "defer_recoverable",
            "final_keyframe_incremented": False,
            "pose_init_attempted": True,
            "pose_init_success": True,
            "num_2d3d_correspondences": 90,
            "num_pnp_inliers": 40,
            "num_miniba_inliers": 38,
            "source_recovery_committed": True,
            "decision_meta": {"R_t": 0.55, "V_t": 0.50, "Q_t": 0.20},
        },
        {
            "frame_id": 3,
            "action": "discard",
            "final_keyframe_incremented": False,
            "pose_init_attempted": False,
            "pose_init_success": False,
            "num_2d3d_correspondences": 0,
            "num_pnp_inliers": 0,
            "num_miniba_inliers": 0,
            "decision_meta": {"R_t": 0.90, "V_t": 0.10, "Q_t": 0.02},
        },
        {
            "frame_id": 4,
            "action": "direct_admit",
            "final_keyframe_incremented": True,
            "pose_init_attempted": True,
            "pose_init_success": True,
            "num_2d3d_correspondences": 180,
            "num_pnp_inliers": 70,
            "num_miniba_inliers": 60,
            "decision_meta": {"R_t": 0.20, "V_t": 0.90, "Q_t": 0.40},
        },
    ],
    "recovery_pose_path_events": [
        {"source_frame_id": 2, "current_frame_id": 8, "pnp_success": True, "pnp_inliers": 44, "miniba_success": True, "miniba_inliers": 41}
    ],
    "recovery_commit_materialization_events": [
        {"source_frame_id": 2, "current_tick_frame_id": 8, "materialized": True, "source_equals_current_frame": False}
    ],
    "pnp_miniba_reference_events": [
        {"frame_id": 1, "pnp_ref_keyframe_ids": [0], "miniba_ref_keyframe_ids": [0], "pnp_success": True},
        {"frame_id": 4, "pnp_ref_keyframe_ids": [1, 2], "miniba_ref_keyframe_ids": [1], "pnp_success": True},
    ],
}

LIFECYCLE_ROWS = [
    {"frame_id": "1", "psnr": "20.0", "ssim": "0.50", "lpips": "0.40", "absolute_relative_translation_error": "0.10", "absolute_relative_rotation_error": "1.0", "lifecycle_state": "evaluated", "state_risk_bucket": "low"},
    {"frame_id": "2", "psnr": "18.0", "ssim": "0.45", "lpips": "0.50", "absolute_relative_translation_error": "0.20", "absolute_relative_rotation_error": "2.0", "lifecycle_state": "deferred", "state_risk_bucket": "medium"},
    {"frame_id": "3", "psnr": "15.0", "ssim": "0.30", "lpips": "0.70", "absolute_relative_translation_error": "0.50", "absolute_relative_rotation_error": "5.0", "lifecycle_state": "discarded", "state_risk_bucket": "high"},
    {"frame_id": "4", "psnr": "22.0", "ssim": "0.60", "lpips": "0.35", "absolute_relative_translation_error": "0.08", "absolute_relative_rotation_error": "0.8", "lifecycle_state": "evaluated", "state_risk_bucket": "low"},
]


class StageMetricEvaluationTests(unittest.TestCase):
    def test_builds_stage_rows_with_offline_quality_pose_and_lifecycle_metrics(self):
        result = build_stage_metric_evaluation(TRACE, lifecycle_rows=LIFECYCLE_ROWS, run_quality={"PSNR": 21.0, "SSIM": 0.55, "LPIPS": 0.37, "R_deg": 1.2, "t": 0.12})
        by_stage = {row["stage_name"]: row for row in result["stage_rows"]}

        self.assertEqual(by_stage["direct_admit"]["frame_count"], 2)
        self.assertEqual(by_stage["direct_admit"]["final_keyframe_count"], 2)
        self.assertAlmostEqual(by_stage["direct_admit"]["psnr_mean"], 21.0)
        self.assertAlmostEqual(by_stage["direct_admit"]["ssim_mean"], 0.55)
        self.assertAlmostEqual(by_stage["direct_admit"]["lpips_mean"], 0.375)
        self.assertAlmostEqual(by_stage["direct_admit"]["absolute_relative_translation_error_mean"], 0.09)
        self.assertAlmostEqual(by_stage["direct_admit"]["absolute_relative_rotation_error_mean"], 0.9)
        self.assertEqual(by_stage["direct_admit"]["pnp_inliers_median"], 60.0)
        self.assertEqual(by_stage["direct_admit"]["miniba_inliers_median"], 52.5)

        self.assertEqual(by_stage["defer_recoverable"]["frame_count"], 1)
        self.assertEqual(by_stage["recovery_attempt"]["frame_count"], 1)
        self.assertEqual(by_stage["true_source_materialized"]["frame_count"], 1)
        self.assertEqual(by_stage["discard"]["frame_count"], 1)
        self.assertEqual(result["overall_summary"]["keyframe_gap_p90"], 3.0)
        self.assertEqual(result["overall_summary"]["true_source_materialized_count"], 1)

    def test_declares_quality_metrics_as_offline_only_not_online_decision_inputs(self):
        result = build_stage_metric_evaluation(TRACE, lifecycle_rows=LIFECYCLE_ROWS)

        self.assertEqual(result["metric_contract"]["online_quality_metric_fields"], [])
        self.assertIn("PSNR", result["metric_contract"]["offline_stage_metric_fields"])
        self.assertIn("absolute_relative_translation_error", result["metric_contract"]["offline_stage_metric_fields"])
        self.assertNotIn("PSNR", result["online_decision_metric_fields_seen"])
        self.assertNotIn("LPIPS", result["online_decision_metric_fields_seen"])

    def test_cli_writes_summary_table_frame_table_and_report(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            trace_path = base / "semantic_trace.json"
            lifecycle_path = base / "lifecycle.csv"
            output_dir = base / "out"
            trace_path.write_text(json.dumps(TRACE), encoding="utf-8")
            with lifecycle_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(LIFECYCLE_ROWS[0]))
                writer.writeheader()
                writer.writerows(LIFECYCLE_ROWS)

            rc = stage_metric_main([
                "--trace_json",
                str(trace_path),
                "--lifecycle_csv",
                str(lifecycle_path),
                "--output_dir",
                str(output_dir),
            ])

            self.assertEqual(rc, 0)
            self.assertTrue((output_dir / "stage_metric_summary.json").exists())
            self.assertTrue((output_dir / "stage_metric_table.csv").exists())
            self.assertTrue((output_dir / "stage_metric_frame_table.csv").exists())
            self.assertTrue((output_dir / "stage_metric_report.md").exists())
            summary = json.loads((output_dir / "stage_metric_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["overall_summary"]["true_source_materialized_count"], 1)
            self.assertEqual(summary["metric_contract"]["online_quality_metric_fields"], [])


if __name__ == "__main__":
    unittest.main()
