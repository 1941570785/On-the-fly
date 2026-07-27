import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools import make_xyz2521_b_positive_assets as xyz_assets
from tools.make_xyz2521_b_positive_assets import (
    _save_sparse_scatter,
    select_spatially_separated_positive_candidates,
    stratified_display_subset,
)


class XYZ2521PositiveAssetTests(unittest.TestCase):
    def test_selection_requires_positive_full_gain_and_spatial_separation(self):
        candidates = [
            {
                "box": (2, 2, 8, 8),
                "response_score": 4.0,
                "full_gain": 0.2,
                "full_is_best": True,
                "positive_repeats": 2,
                "gaussian_ratio": 1.0,
            },
            {
                "box": (4, 3, 10, 9),
                "response_score": 3.9,
                "full_gain": 0.3,
                "full_is_best": True,
                "positive_repeats": 3,
                "gaussian_ratio": 1.0,
            },
            {
                "box": (20, 20, 26, 26),
                "response_score": 3.0,
                "full_gain": 0.1,
                "full_is_best": True,
                "positive_repeats": 3,
                "gaussian_ratio": 0.98,
            },
            {
                "box": (30, 2, 36, 8),
                "response_score": 5.0,
                "full_gain": -0.1,
                "full_is_best": False,
                "positive_repeats": 0,
                "gaussian_ratio": 1.0,
            },
        ]

        selected = select_spatially_separated_positive_candidates(
            candidates,
            count=2,
            minimum_center_distance=12.0,
            minimum_gain=0.02,
            minimum_positive_repeats=2,
            minimum_gaussian_ratio=0.9,
            maximum_gaussian_ratio=1.1,
        )

        self.assertEqual(
            [record["box"] for record in selected],
            [(2, 2, 8, 8), (20, 20, 26, 26)],
        )

    def test_display_subset_is_deterministic_and_contains_real_points(self):
        x, y = np.meshgrid(np.arange(40), np.arange(30))
        points = np.column_stack((x.ravel(), y.ravel())).astype(float)

        first = stratified_display_subset(
            points,
            shape=(30, 40),
            target_count=120,
            grid_shape=(12, 10),
        )
        second = stratified_display_subset(
            points,
            shape=(30, 40),
            target_count=120,
            grid_shape=(12, 10),
        )

        self.assertEqual(first.shape, (120, 2))
        np.testing.assert_array_equal(first, second)
        source = {tuple(point) for point in points}
        self.assertTrue(all(tuple(point) in source for point in first))

    def test_sparse_scatter_uses_bold_nine_pixel_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "scatter.png"
            _save_sparse_scatter(
                np.asarray([[5.0, 5.0]]),
                shape=(12, 12),
                output_path=output_path,
            )
            pixels = np.asarray(Image.open(output_path).convert("RGB"))

        red = np.all(pixels == np.asarray([255, 59, 92]), axis=2)
        y_coordinates, x_coordinates = np.nonzero(red)
        self.assertEqual(x_coordinates.max() - x_coordinates.min() + 1, 9)
        self.assertEqual(y_coordinates.max() - y_coordinates.min() + 1, 9)

    def test_display_targets_follow_complete_gaussian_count_ratios(self):
        function = getattr(
            xyz_assets,
            "proportional_display_targets",
            None,
        )
        self.assertIsNotNone(function)
        targets = function(
            {
                "base": 251.333,
                "r": 237.667,
                "r_e": 248.667,
                "r_d": 251.0,
                "r_e_d": 261.667,
            },
            maximum_points=100,
        )

        self.assertEqual(
            targets,
            {
                "base": 96,
                "r": 91,
                "r_e": 95,
                "r_d": 96,
                "r_e_d": 100,
            },
        )


if __name__ == "__main__":
    unittest.main()
