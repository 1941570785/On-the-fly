from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.baseline_compatible_eval_report import align_metric_rows, draw_combined_quality_svg


class BaselineCompatibleReportTests(unittest.TestCase):
    def test_aligns_rows_by_official_eval_name_order_and_skips_missing_method_frames(self):
        eval_names = ["1.jpg", "11.jpg", "21.jpg"]
        baseline_rows = [
            {"original_image_name": "21.jpg", "psnr": "23", "is_test_view": "True"},
            {"original_image_name": "1.jpg", "psnr": "21", "is_test_view": "True"},
            {"original_image_name": "11.jpg", "psnr": "22", "is_test_view": "True"},
        ]
        method_rows = [
            {"original_image_name": "21.jpg", "psnr": "20", "is_test_view": "True"},
            {"original_image_name": "1.jpg", "psnr": "19", "is_test_view": "True"},
        ]

        aligned = align_metric_rows(eval_names, baseline_rows, method_rows)

        self.assertEqual([row["image_name"] for row in aligned], ["1.jpg", "21.jpg"])
        self.assertEqual(aligned[0]["baseline"]["psnr"], "21")
        self.assertEqual(aligned[0]["method"]["psnr"], "19")

    def test_draws_combined_quality_svg_with_both_methods_and_frame_labels(self):
        baseline_series = [
            {"frame_label": "000", "psnr": 20.0, "ssim": 0.80, "lpips": 0.30},
            {"frame_label": "010", "psnr": 22.0, "ssim": 0.82, "lpips": 0.25},
        ]
        method_series = [
            {"frame_label": "000", "psnr": 21.0, "ssim": 0.81, "lpips": 0.28},
            {"frame_label": "010", "psnr": 23.0, "ssim": 0.84, "lpips": 0.22},
        ]

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "combined.svg"
            draw_combined_quality_svg(
                baseline_series,
                method_series,
                path,
                title="forest1 baseline vs method",
            )

            svg = path.read_text(encoding="utf-8")

        self.assertIn("forest1 baseline vs method", svg)
        self.assertIn("Baseline", svg)
        self.assertIn("Method", svg)
        self.assertIn('data-series="baseline-psnr"', svg)
        self.assertIn('data-series="method-psnr"', svg)
        self.assertIn(">000<", svg)
        self.assertIn(">010<", svg)


if __name__ == "__main__":
    unittest.main()
