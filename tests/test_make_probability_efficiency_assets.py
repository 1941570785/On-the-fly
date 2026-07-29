import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import tools.make_probability_efficiency_assets as probability_assets
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
    def test_box_image_with_rois_preserves_the_selected_render(self):
        self.assertTrue(
            hasattr(probability_assets, "box_image_with_rois")
        )
        if not hasattr(probability_assets, "box_image_with_rois"):
            return
        source = Image.fromarray(
            np.full((24, 32, 3), 17, dtype=np.uint8),
            mode="RGB",
        )
        boxed = probability_assets.box_image_with_rois(
            source,
            red_box=(2, 2, 14, 14),
            blue_box=(16, 4, 28, 16),
            width=4,
        )
        self.assertEqual(source.getpixel((2, 2)), (17, 17, 17))
        self.assertEqual(boxed.getpixel((2, 2)), (214, 39, 40))
        self.assertEqual(boxed.getpixel((16, 4)), (31, 119, 180))
        self.assertEqual(boxed.getpixel((8, 8)), (17, 17, 17))

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

    def test_additional_roi_selection_keeps_complementary_local_evidence(self):
        self.assertTrue(
            hasattr(
                probability_assets,
                "choose_additional_redistribution_rois",
            )
        )
        if not hasattr(
            probability_assets,
            "choose_additional_redistribution_rois",
        ):
            return

        def candidate(
            box,
            *,
            psnr_gain,
            mu_delta,
            mu_relative_change,
            base_share,
            ours_share,
            base_gaussians,
            ours_gaussians,
            lower_repeats,
        ):
            return {
                "box": box,
                "psnr_gain": psnr_gain,
                "positive_psnr_repeats": 3,
                "mu_delta": mu_delta,
                "mu_relative_change": mu_relative_change,
                "redistribution_l1": 10.0,
                "gaussian_reduction": (
                    base_gaussians - ours_gaussians
                ),
                "lower_gaussian_repeats": lower_repeats,
                "sampling_share": {
                    "base": [base_share] * 3,
                    "r_e_d": [ours_share] * 3,
                },
                "gaussian_numbers": {
                    "base": [base_gaussians] * 3,
                    "r_e_d": [ours_gaussians] * 3,
                },
            }

        candidates = [
            candidate(
                (30, 0, 40, 10),
                psnr_gain=0.5,
                mu_delta=5.0,
                mu_relative_change=0.08,
                base_share=2.0,
                ours_share=2.2,
                base_gaussians=100,
                ours_gaussians=101,
                lower_repeats=0,
            ),
            candidate(
                (60, 0, 70, 10),
                psnr_gain=0.8,
                mu_delta=12.0,
                mu_relative_change=0.20,
                base_share=7.0,
                ours_share=8.0,
                base_gaussians=100,
                ours_gaussians=120,
                lower_repeats=0,
            ),
            candidate(
                (90, 0, 100, 10),
                psnr_gain=1.0,
                mu_delta=-8.0,
                mu_relative_change=-0.25,
                base_share=3.0,
                ours_share=2.3,
                base_gaussians=100,
                ours_gaussians=82,
                lower_repeats=3,
            ),
        ]
        orange, purple = (
            probability_assets.choose_additional_redistribution_rois(
                candidates,
                protected_boxes=((0, 0, 10, 10),),
                minimum_center_distance=15.0,
                minimum_psnr_gain=0.05,
                minimum_negative_psnr_gain=0.5,
                minimum_positive_repeats=2,
                minimum_mu_change=3.0,
                minimum_sampling_share=1.5,
                maximum_sampling_share=4.0,
                maximum_positive_gaussian_change=5.0,
                minimum_gaussian_reduction=5.0,
                minimum_lower_gaussian_repeats=2,
            )
        )
        self.assertEqual(orange["box"], (30, 0, 40, 10))
        self.assertEqual(purple["box"], (90, 0, 100, 10))

    def test_combined_chart_accepts_two_additional_regions(self):
        parameters = inspect.signature(
            save_combined_roi_efficiency_chart
        ).parameters
        self.assertIn("orange_gaussian_numbers", parameters)
        self.assertIn("purple_gaussian_numbers", parameters)
        self.assertIn("relative_share_area", parameters)

    def test_relative_share_bubble_areas_use_each_roi_base_as_reference(self):
        self.assertTrue(
            hasattr(
                probability_assets,
                "relative_sampling_share_bubble_areas",
            )
        )
        if not hasattr(
            probability_assets,
            "relative_sampling_share_bubble_areas",
        ):
            return
        areas = (
            probability_assets.relative_sampling_share_bubble_areas(
                np.array([2.0, 2.2], dtype=np.float64),
                base_area=360.0,
                contrast_exponent=2.5,
            )
        )
        np.testing.assert_allclose(
            areas,
            np.array([360.0, 360.0 * 1.1**2.5]),
        )

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

    def test_combined_bubble_area_contrast_enhances_sampling_share(self):
        shares = np.array([3.0, 1.5, 0.75], dtype=np.float64)
        areas = sampling_share_bubble_areas(
            shares,
            maximum_area=600.0,
            contrast_exponent=2.5,
        )
        np.testing.assert_allclose(
            areas,
            600.0 * np.power(shares / shares.max(), 2.5),
        )

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
