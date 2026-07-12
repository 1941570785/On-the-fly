from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.summarize_official_pose_repeats import aggregate_records  # noqa: E402


class OfficialPoseRepeatSummaryTest(unittest.TestCase):
    def test_aggregate_reports_paired_mean_std_and_win_count(self) -> None:
        records = []
        for repeat, v31, v31_a in (
            (1, 1.0, 0.8),
            (2, 2.0, 1.5),
            (3, 3.0, 3.2),
        ):
            records.append({"scene": "forest1", "repeat": repeat, "method": "v31", "ATE_RMSE": v31})
            records.append({"scene": "forest1", "repeat": repeat, "method": "v31_a", "ATE_RMSE": v31_a})

        rows = aggregate_records(records, {"ATE_RMSE": "lower"})

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["n"], 3)
        self.assertAlmostEqual(row["v31_mean"], 2.0)
        self.assertAlmostEqual(row["v31_a_mean"], (0.8 + 1.5 + 3.2) / 3.0)
        self.assertAlmostEqual(row["delta_mean"], (-0.2 - 0.5 + 0.2) / 3.0)
        self.assertAlmostEqual(row["delta_std"], 0.3511884584284246)
        self.assertEqual(row["a_better_count"], 2)

    def test_higher_is_better_metric_uses_positive_delta_as_win(self) -> None:
        records = [
            {"scene": "bonsai", "repeat": 1, "method": "v31", "PSNR": 20.0},
            {"scene": "bonsai", "repeat": 1, "method": "v31_a", "PSNR": 20.1},
            {"scene": "bonsai", "repeat": 2, "method": "v31", "PSNR": 20.2},
            {"scene": "bonsai", "repeat": 2, "method": "v31_a", "PSNR": 20.0},
        ]

        row = aggregate_records(records, {"PSNR": "higher"})[0]

        self.assertEqual(row["a_better_count"], 1)
        self.assertEqual(row["n"], 2)


if __name__ == "__main__":
    unittest.main()
