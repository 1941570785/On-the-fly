import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.make_two_roi_b_internal_tables import (
    METHODS,
    export_roi_table,
    summarize_roi,
)


def id_map(unique_count: int) -> np.ndarray:
    values = np.arange(16, dtype=np.int32) % unique_count
    return values.reshape(4, 4)


class TwoRoiBInternalTableTests(unittest.TestCase):
    def setUp(self):
        ground_truth = np.full((4, 4, 3), 200, dtype=np.uint8)
        self.arrays = {"ground_truth": ground_truth}
        errors = {
            "base": 20,
            "r": 15,
            "r_e": 10,
            "r_d": 12,
            "r_e_d": 5,
        }
        counts = {
            "base": (3, 4),
            "r": (4, 4),
            "r_e": (5, 5),
            "r_d": (4, 5),
            "r_e_d": (6, 6),
        }
        for method in METHODS:
            render = np.full(
                (4, 4, 3),
                200 - errors[method],
                dtype=np.uint8,
            )
            for repetition in (1, 2):
                prefix = f"{method}_repeat_{repetition}"
                self.arrays[f"{prefix}_render"] = render
                self.arrays[f"{prefix}_ids"] = id_map(
                    counts[method][repetition - 1]
                )

    def test_summary_uses_floor_of_mean_and_base_deltas(self):
        rows = summarize_roi(
            self.arrays,
            box=(0, 0, 4, 4),
            repeat_count=2,
        )
        self.assertEqual([row["variant"] for row in rows], [
            "Base",
            "Photo.",
            "Photo.+Struct.",
            "Photo.+Cov.",
            "Full",
        ])
        self.assertEqual(rows[0]["gaussian_numbers"], 3)
        self.assertEqual(rows[1]["gaussian_numbers"], 4)
        self.assertEqual(rows[1]["delta_gaussian_numbers"], 1)
        self.assertAlmostEqual(rows[0]["delta_psnr"], 0.0)
        self.assertGreater(rows[-1]["delta_psnr"], 0.0)

    def test_export_contains_requested_component_and_metric_labels(self):
        rows = summarize_roi(
            self.arrays,
            box=(0, 0, 4, 4),
            repeat_count=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / "red_roi_b_internal_ablation"
            export_roi_table(
                rows,
                roi_name="Red ROI",
                accent="#B52D2D",
                output_stem=stem,
                repeat_count=2,
            )
            for suffix in (".csv", ".tex", ".png", ".pdf", ".svg"):
                self.assertTrue(stem.with_suffix(suffix).is_file())
            svg = stem.with_suffix(".svg").read_text(encoding="utf-8")
            for label in (
                "Photo.",
                "Struct.",
                "Cov.",
                "#Gauss.",
                "Δ#G",
                "PSNR",
                "ΔPSNR",
            ):
                self.assertIn(label, svg)


if __name__ == "__main__":
    unittest.main()
