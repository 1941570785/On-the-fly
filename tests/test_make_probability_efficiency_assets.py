import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.make_probability_efficiency_assets import (
    choose_redistribution_rois,
    roi_expected_samples,
    save_efficiency_bubble_chart,
)


class ProbabilityEfficiencyAssetTests(unittest.TestCase):
    def test_roi_expected_samples_sums_bernoulli_probabilities(self):
        probability = np.arange(24, dtype=np.float64).reshape(4, 6) / 24
        self.assertAlmostEqual(
            roi_expected_samples(probability, (1, 1, 5, 3)),
            float(probability[1:3, 1:5].sum()),
        )

    def test_roi_selection_requires_complementary_probability_changes(self):
        candidates = [
            {
                "box": (0, 0, 10, 10),
                "psnr_gain": 0.4,
                "positive_psnr_repeats": 3,
                "mu_delta": 12.0,
                "redistribution_l1": 20.0,
                "gaussian_reduction": -2.0,
                "lower_gaussian_repeats": 0,
            },
            {
                "box": (40, 0, 50, 10),
                "psnr_gain": 0.3,
                "positive_psnr_repeats": 3,
                "mu_delta": -8.0,
                "redistribution_l1": 16.0,
                "gaussian_reduction": 7.0,
                "lower_gaussian_repeats": 3,
            },
            {
                "box": (80, 0, 90, 10),
                "psnr_gain": 0.8,
                "positive_psnr_repeats": 3,
                "mu_delta": -10.0,
                "redistribution_l1": 30.0,
                "gaussian_reduction": -4.0,
                "lower_gaussian_repeats": 0,
            },
        ]
        red, blue = choose_redistribution_rois(
            candidates,
            minimum_center_distance=20.0,
            minimum_psnr_gain=0.05,
            minimum_positive_repeats=2,
            minimum_mu_change=3.0,
            minimum_gaussian_reduction=5.0,
            minimum_lower_gaussian_repeats=2,
        )
        self.assertEqual(red["box"], (0, 0, 10, 10))
        self.assertEqual(blue["box"], (40, 0, 50, 10))

    def test_bubble_chart_uses_base_and_ours_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / "bubble"
            save_efficiency_bubble_chart(
                gaussian_numbers=np.array([320.0, 300.0]),
                local_psnr=np.array([25.0, 25.4]),
                expected_samples=np.array([180.0, 160.0]),
                output_stem=stem,
            )
            self.assertTrue(stem.with_suffix(".png").is_file())
            self.assertTrue(stem.with_suffix(".pdf").is_file())
            self.assertTrue(stem.with_suffix(".svg").is_file())


if __name__ == "__main__":
    unittest.main()
