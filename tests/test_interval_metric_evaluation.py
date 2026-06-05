from __future__ import annotations

import unittest

from paper_aligned_policy.interval_metrics import build_interval_metric_evaluation


FRAME_ROWS = [
    {"original_frame_idx": "0", "psnr": "20.0", "ssim": "0.60", "lpips": "0.30", "abs_trans_error": "0.20", "abs_rot_error_deg": "1.0"},
    {"original_frame_idx": "40", "psnr": "22.0", "ssim": "0.64", "lpips": "0.26", "abs_trans_error": "0.18", "abs_rot_error_deg": "1.2"},
    {"original_frame_idx": "100", "psnr": "17.0", "ssim": "0.48", "lpips": "0.48", "abs_trans_error": "0.90", "abs_rot_error_deg": "7.0"},
    {"original_frame_idx": "140", "psnr": "16.0", "ssim": "0.46", "lpips": "0.50", "abs_trans_error": "1.10", "abs_rot_error_deg": "8.0"},
]


class IntervalMetricEvaluationTests(unittest.TestCase):
    def test_flags_interval_quality_drop_and_baseline_gap(self):
        result = build_interval_metric_evaluation(
            FRAME_ROWS,
            interval_size=100,
            baseline={"PSNR": 18.34, "SSIM": 0.52, "LPIPS": 0.40, "R_deg": 1.74, "t": 0.23},
        )

        rows = result["interval_rows"]
        self.assertEqual([row["interval"] for row in rows], ["0-100", "100-200"])
        self.assertAlmostEqual(rows[0]["psnr_mean"], 21.0)
        self.assertAlmostEqual(rows[1]["psnr_mean"], 16.5)
        self.assertTrue(rows[1]["quality_drop_from_previous"])
        self.assertIn("psnr_drop", rows[1]["drop_reasons"])
        self.assertIn("ssim_drop", rows[1]["drop_reasons"])
        self.assertIn("lpips_increase", rows[1]["drop_reasons"])
        self.assertIn("translation_error_increase", rows[1]["drop_reasons"])
        self.assertIn("rotation_error_increase", rows[1]["drop_reasons"])
        self.assertAlmostEqual(rows[1]["psnr_delta_to_baseline"], -1.84, places=2)
        self.assertEqual(result["worst_quality_drop_interval"], "100-200")

    def test_groups_runtime_intervals_by_sequence_order_before_file_index(self):
        result = build_interval_metric_evaluation(
            [
                {
                    "sequence_order": "205",
                    "original_frame_idx": "11",
                    "psnr": "16.0",
                    "ssim": "0.40",
                    "lpips": "0.45",
                }
            ],
            interval_size=100,
        )

        self.assertEqual([row["interval"] for row in result["interval_rows"]], ["200-300"])

    def test_groups_runtime_intervals_by_stream_frame_index_before_keyframe_order(self):
        result = build_interval_metric_evaluation(
            [
                {
                    "stream_frame_idx": "240",
                    "sequence_order": "12",
                    "original_frame_idx": "15",
                    "psnr": "16.0",
                    "ssim": "0.40",
                    "lpips": "0.45",
                }
            ],
            interval_size=100,
        )

        self.assertEqual([row["interval"] for row in result["interval_rows"]], ["200-300"])


if __name__ == "__main__":
    unittest.main()
