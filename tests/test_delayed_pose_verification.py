import unittest
from types import SimpleNamespace

import torch

from poses.delayed_pose_verification import (
    merge_global_reference_groups,
    select_delayed_reference_groups,
    summarize_candidate_validation,
)


def _keyframe(index, *, is_test=False, points=32, quarantined=False):
    info = {"_pose_reference_quarantined": quarantined}
    desc = SimpleNamespace(has_pt3d=torch.ones(points, dtype=torch.bool))
    return SimpleNamespace(
        index=index,
        is_test=is_test,
        info=info,
        desc_kpts=desc,
    )


class DelayedReferenceSelectionTests(unittest.TestCase):
    def test_groups_are_disjoint_and_exclude_test_or_quarantined_refs(self):
        frames = [_keyframe(i) for i in range(12)]
        frames[3].is_test = True
        frames[7].info["_pose_reference_quarantined"] = True

        solve, validation = select_delayed_reference_groups(
            frames,
            5,
            solve_count=3,
            validation_count=2,
        )

        self.assertEqual(len(solve), 3)
        self.assertEqual(len(validation), 2)
        self.assertTrue(set(map(id, solve)).isdisjoint(set(map(id, validation))))
        self.assertNotIn(frames[3], solve + validation)
        self.assertNotIn(frames[7], solve + validation)
        self.assertNotIn(frames[5], solve + validation)

    def test_global_mix_remains_disjoint(self):
        frames = [_keyframe(i) for i in range(20)]
        solve, validation = merge_global_reference_groups(
            frames[1:8],
            frames[8:14],
            [frames[15], frames[16], frames[17], frames[18], frames[19]],
            solve_count=6,
            validation_count=4,
            global_solve_count=3,
            global_validation_count=2,
        )

        self.assertEqual(len(solve), 6)
        self.assertEqual(len(validation), 4)
        self.assertTrue(set(map(id, solve)).isdisjoint(set(map(id, validation))))
        self.assertEqual(solve[:3], [frames[15], frames[16], frames[17]])


class DelayedCandidateDecisionTests(unittest.TestCase):
    def test_accepts_consistent_held_out_improvement(self):
        pre = torch.tensor([5.0, 4.0, 6.0, 5.5, 4.5, 5.2])
        post = pre * 0.8
        valid = torch.ones_like(pre, dtype=torch.bool)

        result = summarize_candidate_validation(
            pre,
            post,
            valid,
            valid,
            max_error=10.0,
            min_support=6,
            min_relative_improvement=0.05,
            max_mean_ratio=0.98,
            max_p90_ratio=1.0,
            correction_translation=0.02,
            correction_rotation_deg=0.5,
            max_translation=0.1,
            max_rotation_deg=3.0,
        )

        self.assertTrue(result["accepted"])

    def test_rejects_tail_regression_even_when_median_improves(self):
        pre = torch.tensor([4.0, 4.5, 5.0, 5.5, 6.0, 6.5])
        post = torch.tensor([3.0, 3.5, 4.0, 4.5, 9.0, 12.0])
        valid = torch.ones_like(pre, dtype=torch.bool)

        result = summarize_candidate_validation(
            pre,
            post,
            valid,
            valid,
            max_error=20.0,
            min_support=6,
            min_relative_improvement=0.05,
            max_mean_ratio=1.2,
            max_p90_ratio=1.02,
            correction_translation=0.02,
            correction_rotation_deg=0.5,
            max_translation=0.1,
            max_rotation_deg=3.0,
        )

        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "p90_regressed")


if __name__ == "__main__":
    unittest.main()
