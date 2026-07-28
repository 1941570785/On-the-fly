import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image

from tools.extract_b_internal_frame import (
    _load_render,
    _scene_path,
    load_ground_truth,
)
from tools.run_b_internal_ablation import FOREST1_SCENE


class ExtractBInternalFrameTests(unittest.TestCase):
    def test_scene_path_uses_the_selected_dataset_and_scene(self):
        self.assertEqual(
            _scene_path(
                Path("/experiment"),
                "r_e_d",
                3,
                FOREST1_SCENE,
            ),
            Path(
                "/experiment/r_e_d/repeat_3/StaticHikes/forest1"
            ),
        )

    def test_missing_saved_render_falls_back_to_rasterizer_output(self):
        package = {
            "render": torch.tensor(
                [
                    [[0.0, 0.5], [1.0, 0.25]],
                    [[0.2, 0.4], [0.6, 0.8]],
                    [[1.0, 0.0], [0.5, 0.1]],
                ],
                dtype=torch.float32,
            )
        }
        with TemporaryDirectory() as directory:
            render, source = _load_render(
                Path(directory) / "missing.png",
                package,
            )

        self.assertEqual(source, "rasterizer")
        self.assertEqual(render.shape, (2, 2, 3))
        self.assertEqual(render.dtype, np.uint8)
        np.testing.assert_array_equal(
            render[0, 1],
            np.array([128, 102, 0], dtype=np.uint8),
        )

    def test_ground_truth_preserves_alpha_validity_mask(self):
        rgba = np.array(
            [
                [[10, 20, 30, 255], [0, 0, 0, 255]],
                [[40, 50, 60, 0], [70, 80, 90, 128]],
            ],
            dtype=np.uint8,
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ground_truth.png"
            Image.fromarray(rgba, mode="RGBA").save(path)
            rgb, valid = load_ground_truth(path)

        np.testing.assert_array_equal(rgb, rgba[..., :3])
        np.testing.assert_array_equal(
            valid,
            np.array([[True, True], [False, True]]),
        )


if __name__ == "__main__":
    unittest.main()
