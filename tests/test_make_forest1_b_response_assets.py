import unittest

import numpy as np

from tools.make_forest1_b_response_assets import (
    boxes_overlap,
    compute_response_guide,
    footprint_density,
    local_gaussian_count,
    select_response_rois,
)


class Forest1BResponseAssetTests(unittest.TestCase):
    def test_response_guide_ranks_a_photometric_error_patch_higher(self):
        ground_truth = np.full((32, 32, 3), 128, dtype=np.uint8)
        render = ground_truth.copy()
        render[8:20, 10:22] = 32

        guide = compute_response_guide(ground_truth, render)

        self.assertEqual(guide.shape, (32, 32))
        self.assertGreaterEqual(float(guide.min()), 0.25)
        self.assertLessEqual(float(guide.max()), 4.0)
        self.assertGreater(
            float(guide[9:19, 11:21].mean()),
            float(guide[:6, :6].mean()),
        )

    def test_roi_selection_returns_two_non_overlapping_hotspots(self):
        guide = np.ones((32, 32), dtype=np.float64)
        guide[2:8, 3:9] = 10.0
        guide[20:26, 21:27] = 8.0
        support = np.ones_like(guide)

        selected = select_response_rois(
            guide,
            support,
            window_width=6,
            window_height=6,
            stride=1,
            margin=0,
            minimum_support=0.5,
            count=2,
        )

        self.assertEqual(len(selected), 2)
        self.assertFalse(
            boxes_overlap(selected[0]["box"], selected[1]["box"])
        )
        self.assertGreaterEqual(selected[0]["score"], selected[1]["score"])
        self.assertEqual(selected[0]["box"], (3, 2, 9, 8))
        self.assertEqual(selected[1]["box"], (21, 20, 27, 26))

    def test_density_mass_equals_the_number_of_unique_local_gaussians(self):
        identifiers = np.asarray(
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [3, 3, -1, -1],
                [3, 3, -1, -1],
            ],
            dtype=np.int32,
        )
        box = (0, 0, 4, 4)

        density = footprint_density(identifiers, box, sigma=0.8)

        self.assertEqual(local_gaussian_count(identifiers, box), 3)
        self.assertAlmostEqual(float(density.sum()), 3.0, places=6)


if __name__ == "__main__":
    unittest.main()
