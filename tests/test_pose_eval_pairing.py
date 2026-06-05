from __future__ import annotations

import unittest

import torch

from scene.pose_eval_utils import select_pose_eval_pairs


class PoseEvalPairingTests(unittest.TestCase):
    def test_selects_predicted_poses_matching_valid_gt_mask_when_lengths_differ(self):
        rts = torch.arange(5 * 4 * 4, dtype=torch.float32).reshape(5, 4, 4)
        gt_rts = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4) + 1000
        gt_mask = torch.tensor([True, False, True, False, True])

        selected_rts, selected_gt = select_pose_eval_pairs(rts, gt_rts, gt_mask)

        self.assertEqual(selected_rts.shape[0], 3)
        self.assertTrue(torch.equal(selected_rts, rts[[0, 2, 4]]))
        self.assertTrue(torch.equal(selected_gt, gt_rts))

    def test_truncates_to_available_gt_when_mask_contains_more_valid_entries(self):
        rts = torch.arange(4 * 4 * 4, dtype=torch.float32).reshape(4, 4, 4)
        gt_rts = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4) + 1000
        gt_mask = torch.tensor([True, True, True, False])

        selected_rts, selected_gt = select_pose_eval_pairs(rts, gt_rts, gt_mask)

        self.assertEqual(selected_rts.shape[0], 2)
        self.assertTrue(torch.equal(selected_rts, rts[[0, 1]]))
        self.assertTrue(torch.equal(selected_gt, gt_rts))


if __name__ == "__main__":
    unittest.main()
