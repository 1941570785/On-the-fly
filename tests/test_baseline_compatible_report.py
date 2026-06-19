from __future__ import annotations

import unittest

from tools.baseline_compatible_eval_report import align_metric_rows


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


if __name__ == "__main__":
    unittest.main()
