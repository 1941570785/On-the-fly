import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.make_probability_efficiency_assets import (
    OURS_SAMPLING_COLOR,
    choose_redistribution_rois,
    contrast_enhanced_bubble_areas,
    deterministic_probability_points,
    mean_normalized_probability,
    normalized_probability_change,
    roi_expected_samples,
    roi_sampling_share,
    sampling_share_bubble_areas,
    save_combined_roi_efficiency_chart,
    save_efficiency_bubble_chart,
    save_global_probability_redistribution_maps,
)


class ProbabilityEfficiencyAssetTests(unittest.TestCase):
    def test_ours_sampling_color_uses_light_green(self):
        self.assertEqual(OURS_SAMPLING_COLOR, "#66A866")

    def test_roi_expected_samples_sums_bernoulli_probabilities(self):
        probability = np.arange(24, dtype=np.float64).reshape(4, 6) / 24
        self.assertAlmostEqual(
            roi_expected_samples(probability, (1, 1, 5, 3)),
            float(probability[1:3, 1:5].sum()),
        )

    def test_roi_sampling_share_reports_percentage_of_frame_mass(self):
        probability = np.arange(1, 25, dtype=np.float64).reshape(4, 6)
        box = (1, 1, 5, 3)
        expected = 100.0 * probability[1:3, 1:5].sum() / probability.sum()
        self.assertAlmostEqual(
            roi_sampling_share(probability, box),
            expected,
        )

    def test_normalized_probability_change_conserves_frame_mass(self):
        arrays = {
            "base_repeat_1_final_probability": np.array(
                [[1.0, 3.0], [2.0, 4.0]]
            ),
            "r_e_d_repeat_1_final_probability": np.array(
                [[2.0, 2.0], [4.0, 2.0]]
            ),
        }
        change = normalized_probability_change(arrays, repeat=1)
        self.assertEqual(change.shape, (2, 2))
        self.assertAlmostEqual(float(change.sum()), 0.0)
        self.assertGreater(change[1, 0], 0.0)
        self.assertLess(change[1, 1], 0.0)

    def test_mean_normalized_probability_has_unit_mass(self):
        arrays = {
            "base_repeat_1_final_probability": np.array(
                [[1.0, 3.0], [2.0, 4.0]]
            ),
            "base_repeat_2_final_probability": np.array(
                [[2.0, 2.0], [4.0, 2.0]]
            ),
        }
        probability = mean_normalized_probability(
            arrays,
            method="base",
            repeat=2,
        )
        self.assertAlmostEqual(float(probability.sum()), 1.0)

    def test_probability_points_are_deterministic_and_unique(self):
        probability = np.arange(1, 101, dtype=np.float64).reshape(10, 10)
        first = deterministic_probability_points(
            probability,
            count=20,
            seed=17,
        )
        second = deterministic_probability_points(
            probability,
            count=20,
            seed=17,
        )
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        coordinates = set(zip(first[0].tolist(), first[1].tolist()))
        self.assertEqual(len(coordinates), 20)

    def test_roi_selection_requires_complementary_probability_changes(self):
        candidates = [
            {
                "box": (0, 0, 10, 10),
                "psnr_gain": 0.4,
                "positive_psnr_repeats": 3,
                "mu_delta": 12.0,
                "mu_relative_change": 0.06,
                "redistribution_l1": 20.0,
                "gaussian_reduction": -2.0,
                "lower_gaussian_repeats": 0,
            },
            {
                "box": (30, 0, 40, 10),
                "psnr_gain": 0.3,
                "positive_psnr_repeats": 3,
                "mu_delta": 8.0,
                "mu_relative_change": 0.20,
                "redistribution_l1": 15.0,
                "gaussian_reduction": -1.0,
                "lower_gaussian_repeats": 0,
            },
            {
                "box": (80, 0, 90, 10),
                "psnr_gain": 0.3,
                "positive_psnr_repeats": 3,
                "mu_delta": -8.0,
                "mu_relative_change": -0.15,
                "redistribution_l1": 16.0,
                "gaussian_reduction": 7.0,
                "lower_gaussian_repeats": 3,
            },
            {
                "box": (120, 0, 130, 10),
                "psnr_gain": 0.8,
                "positive_psnr_repeats": 3,
                "mu_delta": -10.0,
                "mu_relative_change": -0.10,
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
        self.assertEqual(red["box"], (30, 0, 40, 10))
        self.assertEqual(blue["box"], (80, 0, 90, 10))

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

    def test_bubble_areas_make_normalized_mu_changes_visible(self):
        increased = contrast_enhanced_bubble_areas(
            np.array([100.0, 110.0]),
            base_area=1500.0,
            exponent=4.0,
        )
        decreased = contrast_enhanced_bubble_areas(
            np.array([100.0, 80.0]),
            base_area=1500.0,
            exponent=4.0,
        )
        np.testing.assert_allclose(
            increased,
            np.array([1500.0, 1500.0 * 1.1**4]),
        )
        np.testing.assert_allclose(
            decreased,
            np.array([1500.0, 1500.0 * 0.8**4]),
        )

    def test_combined_bubble_area_is_linear_in_sampling_share(self):
        shares = np.array([3.0, 1.5, 0.75], dtype=np.float64)
        areas = sampling_share_bubble_areas(
            shares,
            maximum_area=600.0,
        )
        np.testing.assert_allclose(areas, np.array([600.0, 300.0, 150.0]))

    def test_combined_chart_exports_both_regions(self):
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / "combined"
            save_combined_roi_efficiency_chart(
                red_gaussian_numbers=np.array([409.0, 419.0]),
                red_local_psnr=np.array([22.99, 23.72]),
                red_sampling_share=np.array([3.03, 3.48]),
                blue_gaussian_numbers=np.array([537.0, 532.0]),
                blue_local_psnr=np.array([18.48, 21.12]),
                blue_sampling_share=np.array([2.40, 1.37]),
                output_stem=stem,
            )
            self.assertTrue(stem.with_suffix(".png").is_file())
            self.assertTrue(stem.with_suffix(".pdf").is_file())
            self.assertTrue(stem.with_suffix(".svg").is_file())
            svg = stem.with_suffix(".svg").read_text(encoding="utf-8")
            self.assertIn(
                "Local Gaussian Sampling Redistribution",
                svg,
            )
            self.assertIn(
                "Bubble area indicates ROI sampling share (%)",
                svg,
            )
            self.assertNotIn("Red ROI", svg)
            self.assertNotIn("Blue ROI", svg)
            self.assertIn("409, 22.99 dB", svg)
            self.assertIn("419, 23.72 dB", svg)
            self.assertIn("537, 18.48 dB", svg)
            self.assertIn("532, 21.12 dB", svg)
            self.assertNotIn(" G,", svg)

    def test_global_probability_maps_export_base_and_ours_only(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            base = np.ones((24, 32), dtype=np.float64)
            ours = base.copy()
            ours[3:10, 3:12] *= 2.0
            ours[12:21, 18:29] *= 0.5
            base /= base.sum()
            ours /= ours.sum()
            legacy_stems = (
                "full_frame_probability_redistribution_points",
                "full_frame_probability_change_points",
            )
            for stem in legacy_stems:
                for extension in (".png", ".pdf", ".svg"):
                    (output_dir / f"{stem}{extension}").write_text(
                        "legacy",
                        encoding="utf-8",
                    )
            save_global_probability_redistribution_maps(
                base_probability=base,
                ours_probability=ours,
                valid_mask=np.ones((24, 32), dtype=bool),
                red_box=(3, 3, 12, 10),
                blue_box=(18, 12, 29, 21),
                output_dir=output_dir,
                display_budget=40,
                dpi=100,
            )
            for stem in (
                "full_frame_sampling_probability_comparison",
                "full_frame_base_sampling_points",
                "full_frame_ours_sampling_points",
            ):
                for extension in (".png", ".pdf", ".svg"):
                    self.assertTrue(
                        (output_dir / f"{stem}{extension}").is_file()
                    )
            ours_image = np.asarray(
                Image.open(
                    output_dir / "full_frame_ours_sampling_points.png"
                ).convert("RGB")
            )
            red = ours_image[:, :, 0].astype(np.int16)
            green = ours_image[:, :, 1].astype(np.int16)
            blue = ours_image[:, :, 2].astype(np.int16)
            green_points = (
                (green > 90)
                & ((green - red) > 35)
                & ((green - blue) > 35)
            )
            self.assertGreater(
                int(np.count_nonzero(green_points)),
                20,
            )
            for stem in legacy_stems:
                for extension in (".png", ".pdf", ".svg"):
                    self.assertFalse(
                        (output_dir / f"{stem}{extension}").exists()
                    )


if __name__ == "__main__":
    unittest.main()
