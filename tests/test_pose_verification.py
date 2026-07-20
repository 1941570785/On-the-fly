import math
import unittest

import torch

from poses.pose_verification import (
    choose_verified_pose,
    compute_reprojection_errors,
    decide_pose_refinement,
    pose_correction_magnitude,
    select_robust_correspondences,
    summarize_reprojection_errors,
)


def _pose(*, tx: float = 0.0, rz_deg: float = 0.0) -> torch.Tensor:
    angle = math.radians(rz_deg)
    c = math.cos(angle)
    s = math.sin(angle)
    return torch.tensor(
        [
            [c, -s, 0.0, tx],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


class PoseVerificationGeometryTests(unittest.TestCase):
    def test_reprojection_uses_world_to_camera_pose_and_rejects_negative_depth(self):
        xyz = torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 2.0], [0.0, 0.0, -1.0]],
            dtype=torch.float32,
        )
        observed = torch.tensor(
            [[50.0, 50.0], [101.0, 50.0], [50.0, 50.0]],
            dtype=torch.float32,
        )

        errors, valid = compute_reprojection_errors(
            _pose(),
            xyz,
            observed,
            focal=torch.tensor([100.0]),
            centre=torch.tensor([50.0, 50.0]),
        )

        self.assertTrue(torch.equal(valid, torch.tensor([True, True, False])))
        self.assertAlmostEqual(float(errors[0]), 0.0, places=5)
        self.assertAlmostEqual(float(errors[1]), 1.0, places=5)
        self.assertTrue(torch.isinf(errors[2]))

    def test_mad_filter_removes_local_outlier(self):
        errors = torch.tensor([0.2, 0.3, 0.4, 0.5, 12.0])
        uv = torch.tensor(
            [[5.0, 5.0], [8.0, 6.0], [12.0, 8.0], [14.0, 10.0], [10.0, 10.0]]
        )

        mask, debug = select_robust_correspondences(
            errors,
            uv,
            torch.ones(5, dtype=torch.bool),
            width=100,
            height=80,
            mad_scale=2.5,
            max_cutoff=10.0,
            min_support=4,
            grid_rows=4,
            grid_cols=5,
        )

        self.assertTrue(torch.equal(mask, torch.tensor([True, True, True, True, False])))
        self.assertEqual(debug["selected_count"], 4)
        self.assertGreater(debug["rejected_count"], 0)

    def test_filter_preserves_best_correspondence_in_each_occupied_cell(self):
        errors = torch.tensor([0.1, 0.2, 8.0])
        uv = torch.tensor([[5.0, 5.0], [8.0, 8.0], [95.0, 75.0]])

        mask, debug = select_robust_correspondences(
            errors,
            uv,
            torch.ones(3, dtype=torch.bool),
            width=100,
            height=80,
            mad_scale=1.0,
            max_cutoff=10.0,
            min_support=2,
            grid_rows=4,
            grid_cols=5,
        )

        self.assertTrue(mask[2])
        self.assertEqual(debug["occupied_cells"], 2)
        self.assertEqual(debug["selected_cells"], 2)

    def test_filter_restores_lowest_errors_to_minimum_support(self):
        errors = torch.arange(1.0, 11.0)
        uv = torch.stack([torch.arange(10.0), torch.zeros(10)], dim=-1)

        mask, debug = select_robust_correspondences(
            errors,
            uv,
            torch.ones(10, dtype=torch.bool),
            width=100,
            height=80,
            mad_scale=0.0,
            max_cutoff=10.0,
            min_support=7,
            grid_rows=1,
            grid_cols=1,
        )

        self.assertEqual(int(mask.sum()), 7)
        self.assertTrue(torch.equal(torch.where(mask)[0], torch.arange(7)))
        self.assertEqual(debug["restored_for_min_support"], 2)

    def test_summary_reports_robust_distribution(self):
        stats = summarize_reprojection_errors(
            torch.tensor([1.0, 2.0, 3.0, float("inf")]),
            torch.tensor([True, True, True, False]),
        )

        self.assertEqual(stats["valid_count"], 3)
        self.assertAlmostEqual(stats["median"], 2.0)
        self.assertGreater(stats["p90"], 2.0)


class PoseVerificationAcceptanceTests(unittest.TestCase):
    def test_accepts_lower_residual_with_bounded_pose_correction(self):
        initial = _pose()
        refined = _pose(tx=0.01, rz_deg=1.0)
        correction = pose_correction_magnitude(initial, refined)

        decision = decide_pose_refinement(
            pre={"median": 2.0, "p90": 4.0, "valid_count": 100},
            post={"median": 1.5, "p90": 3.5, "valid_count": 96},
            correction=correction,
            max_rotation_deg=3.0,
            max_translation=0.05,
            min_relative_median_improvement=0.05,
            min_support_ratio=0.80,
        )

        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["reason"], "accepted")
        self.assertGreater(decision["relative_median_improvement"], 0.20)

    def test_default_accepts_two_percent_median_gain_when_tail_also_improves(self):
        decision = decide_pose_refinement(
            pre={"median": 2.0, "p90": 4.0, "valid_count": 100},
            post={"median": 1.94, "p90": 3.95, "valid_count": 100},
            correction={"rotation_deg": 1.0, "translation": 0.01},
            max_rotation_deg=3.0,
            max_translation=0.05,
        )

        self.assertTrue(decision["accepted"])

    def test_accepts_small_p90_tolerance_when_mean_and_median_improve(self):
        decision = decide_pose_refinement(
            pre={"mean": 2.2, "median": 2.0, "p90": 4.0, "valid_count": 100},
            post={"mean": 2.0, "median": 1.9, "p90": 4.02, "valid_count": 100},
            correction={"rotation_deg": 1.0, "translation": 0.01},
            max_rotation_deg=3.0,
            max_translation=0.05,
        )

        self.assertTrue(decision["accepted"])

    def test_rejects_mean_degradation_even_when_median_improves(self):
        decision = decide_pose_refinement(
            pre={"mean": 2.0, "median": 2.0, "p90": 4.0, "valid_count": 100},
            post={"mean": 2.1, "median": 1.8, "p90": 4.0, "valid_count": 100},
            correction={"rotation_deg": 1.0, "translation": 0.01},
            max_rotation_deg=3.0,
            max_translation=0.05,
        )

        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["reason"], "mean_degraded")

    def test_rejects_tail_degradation_even_when_median_improves(self):
        decision = decide_pose_refinement(
            pre={"median": 2.0, "p90": 4.0, "valid_count": 100},
            post={"median": 1.0, "p90": 4.5, "valid_count": 100},
            correction={"rotation_deg": 1.0, "translation": 0.01},
            max_rotation_deg=3.0,
            max_translation=0.05,
        )

        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["reason"], "p90_degraded")

    def test_rejects_support_collapse(self):
        decision = decide_pose_refinement(
            pre={"median": 2.0, "p90": 4.0, "valid_count": 100},
            post={"median": 1.0, "p90": 3.0, "valid_count": 40},
            correction={"rotation_deg": 1.0, "translation": 0.01},
            max_rotation_deg=3.0,
            max_translation=0.05,
            min_support_ratio=0.80,
        )

        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["reason"], "support_collapsed")

    def test_rejects_implausible_rotation_and_returns_exact_initial_pose(self):
        initial = _pose(tx=0.2)
        refined = _pose(tx=0.2, rz_deg=8.0)
        decision = decide_pose_refinement(
            pre={"median": 2.0, "p90": 4.0, "valid_count": 100},
            post={"median": 1.0, "p90": 3.0, "valid_count": 100},
            correction=pose_correction_magnitude(initial, refined),
            max_rotation_deg=3.0,
            max_translation=0.05,
        )

        selected = choose_verified_pose(initial, refined, decision)

        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["reason"], "rotation_correction_too_large")
        self.assertTrue(torch.equal(selected, initial))
        self.assertNotEqual(selected.data_ptr(), initial.data_ptr())


if __name__ == "__main__":
    unittest.main()
