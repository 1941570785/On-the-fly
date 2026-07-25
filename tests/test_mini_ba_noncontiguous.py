from __future__ import annotations

import unittest

import torch

from poses.mini_ba import MiniBAInternal


class MiniBANoncontiguousInputTests(unittest.TestCase):
    def test_projection_preparation_accepts_noncontiguous_rotation_tensor(self):
        optimizer = MiniBAInternal(
            batch=1,
            n_opt_cams=1,
            n_fixed_cams=0,
            npts=4,
            optimize_focal=False,
            optimize_3Dpts=False,
            huber_delta=1,
            outlier_mad_scale=4,
            lm=1e-5,
            ep=1e-2,
            k=2,
            iters=1,
        )
        rotations = torch.randn(1, 1, 2, 3).transpose(-1, -2)
        translations = torch.randn(1, 3)

        _, camera_parameters, _, _ = optimizer.prepare_for_proj(
            torch.randn(4, 3),
            rotations,
            translations,
            torch.randn(1),
            torch.randn(2),
        )

        self.assertFalse(rotations.is_contiguous())
        self.assertEqual(camera_parameters.shape, (4, 1, 9))


if __name__ == "__main__":
    unittest.main()
