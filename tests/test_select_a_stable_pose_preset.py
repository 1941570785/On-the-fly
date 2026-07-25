import unittest

from tools.select_a_stable_pose_preset import summarize


class StablePosePresetSelectionTests(unittest.TestCase):
    def test_reports_paired_ratios_against_control(self):
        rows = []
        for repeat in (1, 2):
            rows.append(
                {
                    "repeat": repeat,
                    "scene": "counter",
                    "variant": "a_off",
                    "T_APE": 10.0,
                    "R_APE": 10.0,
                    "T_RPE": 10.0,
                    "R_RPE": 10.0,
                }
            )
            rows.append(
                {
                    "repeat": repeat,
                    "scene": "counter",
                    "variant": "candidate",
                    "T_APE": 9.0,
                    "R_APE": 9.0,
                    "T_RPE": 9.0,
                    "R_RPE": 9.0,
                }
            )

        report = summarize(rows, baseline="a_off")[0]

        self.assertEqual(report["preset"], "candidate")
        self.assertEqual(report["paired_observations"], 8)
        self.assertAlmostEqual(report["mean_ratio"], 0.9)
        self.assertEqual(report["mean_cell_wins"], 4)


if __name__ == "__main__":
    unittest.main()
