import unittest
from pathlib import Path

from tools.extract_b_internal_frame import _scene_path
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


if __name__ == "__main__":
    unittest.main()
