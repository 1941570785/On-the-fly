import unittest

from tools.evaluate_a_v2_baseline_pose_experiment import (
    OFFICIAL_NUMERIC_REFERENCE_ROOT,
    REFERENCE_ROOT,
    reference_root_for_scene,
)


class AV2PoseEvaluationReferenceTests(unittest.TestCase):
    def test_colmap_scenes_use_recovered_official_pose_bindings(self):
        self.assertEqual(
            reference_root_for_scene("counter"),
            OFFICIAL_NUMERIC_REFERENCE_ROOT,
        )
        self.assertEqual(
            reference_root_for_scene("forest1"),
            OFFICIAL_NUMERIC_REFERENCE_ROOT,
        )

    def test_tum_scenes_keep_official_mocap_reference(self):
        self.assertEqual(reference_root_for_scene("desk"), REFERENCE_ROOT)


if __name__ == "__main__":
    unittest.main()
