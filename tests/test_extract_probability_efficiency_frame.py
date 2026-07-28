import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.extract_probability_efficiency_frame import (
    _load_sampling_trace,
    _scene_path,
    load_ground_truth,
)
from tools.run_b_internal_ablation import XYZ_SCENE


class ExtractProbabilityEfficiencyFrameTests(unittest.TestCase):
    def test_scene_path_uses_base_or_complete_b_mode(self):
        self.assertEqual(
            _scene_path(
                Path("/experiment"),
                "r_e_d",
                2,
                XYZ_SCENE,
            ),
            Path(
                "/experiment/r_e_d/repeat_2/TUM_RGB-D/xyz"
            ),
        )

    def test_load_sampling_trace_requires_matching_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            scene_path = Path(directory)
            trace_dir = scene_path / "sampling_trace"
            trace_dir.mkdir()
            np.savez_compressed(
                trace_dir / "002882.npz",
                base_probability=np.zeros((3, 4), dtype=np.float32),
                final_probability=np.ones((3, 4), dtype=np.float32),
                sample_mask=np.zeros((3, 4), dtype=bool),
            )
            trace = _load_sampling_trace(scene_path, "002882.png")
            self.assertEqual(trace["final_probability"].shape, (3, 4))
            self.assertAlmostEqual(
                float(trace["final_probability"].sum()),
                12.0,
            )

    def test_ground_truth_loader_preserves_alpha_validity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            rgba = np.zeros((2, 3, 4), dtype=np.uint8)
            rgba[..., :3] = 128
            rgba[..., 3] = np.array(
                [[255, 0, 255], [255, 255, 0]],
                dtype=np.uint8,
            )
            Image.fromarray(rgba, mode="RGBA").save(path)
            rgb, valid = load_ground_truth(path)
            self.assertEqual(rgb.shape, (2, 3, 3))
            self.assertEqual(valid.dtype, np.bool_)
            self.assertEqual(int(valid.sum()), 4)


if __name__ == "__main__":
    unittest.main()
