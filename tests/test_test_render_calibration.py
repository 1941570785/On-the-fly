import unittest

import torch

from scene.test_render_calibration import calibrate_test_render


class TestRenderCalibrationTests(unittest.TestCase):
    def test_diag_affine_calibration_recovers_channel_scale_and_bias(self):
        target = torch.tensor(
            [
                [[0.2, 0.4], [0.6, 0.8]],
                [[0.1, 0.3], [0.5, 0.7]],
                [[0.15, 0.35], [0.55, 0.75]],
            ],
            dtype=torch.float32,
        )
        render = torch.clamp((target - 0.05) / 0.8, 0.0, 1.0)

        calibrated, debug = calibrate_test_render(
            render,
            target,
            mode="diag_affine_v1",
            min_scale=0.1,
            max_scale=2.0,
            max_bias=0.5,
        )

        self.assertTrue(debug["applied"])
        self.assertLess(torch.mean(torch.abs(calibrated - target)).item(), 1e-5)

    def test_off_mode_returns_input_without_copying_behavior(self):
        render = torch.ones(3, 2, 2)
        target = torch.zeros(3, 2, 2)

        calibrated, debug = calibrate_test_render(render, target, mode="off")

        self.assertFalse(debug["applied"])
        self.assertIs(calibrated, render)

    def test_diag_affine_rejects_calibration_that_increases_mse(self):
        target = torch.tensor(
            [
                [[0.2, 0.4], [0.6, 0.8]],
                [[0.1, 0.3], [0.5, 0.7]],
                [[0.15, 0.35], [0.55, 0.75]],
            ],
            dtype=torch.float32,
        )
        render = target.clone()

        calibrated, debug = calibrate_test_render(
            render,
            target,
            mode="diag_affine_v1",
            min_scale=1.5,
            max_scale=1.5,
            max_bias=0.0,
        )

        self.assertTrue(debug["applied"])
        self.assertFalse(debug["accepted"])
        self.assertIs(calibrated, render)

    def test_diag_affine_uses_mask_for_acceptance_guard(self):
        target = torch.tensor(
            [
                [[0.2, 0.4], [0.6, 0.8]],
                [[0.1, 0.3], [0.5, 0.7]],
                [[0.15, 0.35], [0.55, 0.75]],
            ],
            dtype=torch.float32,
        )
        render = target.clone()
        mask = torch.tensor([[[True, False], [False, False]]])

        calibrated, debug = calibrate_test_render(
            render,
            target,
            mode="diag_affine_v1",
            min_scale=1.5,
            max_scale=1.5,
            max_bias=0.0,
            mask=mask,
        )

        self.assertTrue(debug["applied"])
        self.assertFalse(debug["accepted"])
        self.assertIs(calibrated, render)


if __name__ == "__main__":
    unittest.main()
