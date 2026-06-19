from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scene.frame_metrics import (
    FRAME_METRIC_FIELDS,
    build_frame_metric_row,
    infer_dataset_scene_from_output_dir,
    write_frame_metrics_csv,
)


class FrameMetricsCsvTests(unittest.TestCase):
    def test_builds_scalar_frame_metric_row_for_interval_and_stage_tools(self):
        row = build_frame_metric_row(
            dataset_name="StaticHikes",
            scene_name="forest1",
            frame_idx=4,
            original_image_name="101.jpg",
            sequence_order=10,
            stream_frame_idx=250,
            is_test_view=True,
            is_keyframe=True,
            is_registered=True,
            est_rt=[
                [1.0, 0.0, 0.0, 1.5],
                [0.0, 1.0, 0.0, -2.0],
                [0.0, 0.0, 1.0, 3.25],
                [0.0, 0.0, 0.0, 1.0],
            ],
            quality={"psnr": 20.5, "ssim": 0.61, "lpips": 0.32},
            pose_error={"abs_trans_error": 0.12, "abs_rot_error_deg": 1.7},
            output_dir="results/StaticHikes/forest1/run",
            baseline_eval={
                "frame": True,
                "hold": 10,
                "sequence_index": 20,
                "original_index": 20,
            },
        )

        self.assertEqual(row["dataset_name"], "StaticHikes")
        self.assertEqual(row["scene_name"], "forest1")
        self.assertEqual(row["frame_idx"], 4)
        self.assertEqual(row["original_frame_idx"], 101)
        self.assertEqual(row["stream_frame_idx"], 250)
        self.assertEqual(row["render_image_name"], "101.jpg")
        self.assertTrue(row["baseline_eval_frame"])
        self.assertEqual(row["baseline_eval_hold"], 10)
        self.assertEqual(row["baseline_eval_sequence_index"], 20)
        self.assertEqual(row["baseline_eval_original_index"], 20)
        self.assertEqual(row["split"], "test")
        self.assertEqual(row["pose_format"], "w2c")
        self.assertEqual(row["est_tx"], 1.5)
        self.assertEqual(row["est_ty"], -2.0)
        self.assertEqual(row["est_tz"], 3.25)
        self.assertEqual(row["psnr"], 20.5)
        self.assertEqual(row["ssim"], 0.61)
        self.assertEqual(row["lpips"], 0.32)
        self.assertEqual(row["abs_trans_error"], 0.12)
        self.assertEqual(row["abs_rot_error_deg"], 1.7)

    def test_writes_frame_metrics_csv_with_stable_header(self):
        row = build_frame_metric_row(
            dataset_name="StaticHikes",
            scene_name="forest1",
            frame_idx=0,
            original_image_name="1.jpg",
            sequence_order=0,
            stream_frame_idx=0,
            is_test_view=False,
            is_keyframe=True,
            is_registered=True,
            est_rt=None,
            quality={},
            pose_error={},
            output_dir="results/StaticHikes/forest1/run",
        )

        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "frame_metrics.csv"
            write_frame_metrics_csv(csv_path, [row])
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(rows[0]["original_frame_idx"], "1")
        self.assertEqual(rows[0]["stream_frame_idx"], "0")
        self.assertEqual(rows[0]["split"], "train")
        self.assertEqual(rows[0]["psnr"], "")
        self.assertIn("abs_rot_error_deg", rows[0])
        self.assertEqual(list(rows[0].keys()), FRAME_METRIC_FIELDS)

    def test_infers_dataset_and_scene_from_results_path(self):
        self.assertEqual(
            infer_dataset_scene_from_output_dir("results/StaticHikes/forest1/A5/short300"),
            ("StaticHikes", "forest1"),
        )


if __name__ == "__main__":
    unittest.main()
