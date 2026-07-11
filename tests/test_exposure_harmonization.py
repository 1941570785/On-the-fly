from types import SimpleNamespace
import unittest

from scene.exposure_harmonization import (
    adaptive_source_time_exposure_scene_decision,
    dark_scene_test_exposure_scene_decision,
    harmonized_test_exposure,
)


class ExposureHarmonizationTests(unittest.TestCase):
    def test_source_time_interp_weights_neighbor_exposure_by_source_frame_id(self):
        keyframes = [
            SimpleNamespace(
                exposure=0.0,
                info={"is_test": False, "_paper_aligned_source_frame_id": 10},
            ),
            SimpleNamespace(
                exposure=-1.0,
                info={"is_test": True, "_paper_aligned_source_frame_id": 12},
            ),
            SimpleNamespace(
                exposure=10.0,
                info={"is_test": False, "_paper_aligned_source_frame_id": 20},
            ),
        ]

        exposure = harmonized_test_exposure(
            keyframes,
            1,
            mode="source_time_interp_v1",
        )

        self.assertAlmostEqual(exposure, 2.0)

    def test_neighbor_average_mode_preserves_baseline_midpoint_behavior(self):
        keyframes = [
            SimpleNamespace(exposure=0.0, info={"is_test": False}),
            SimpleNamespace(exposure=-1.0, info={"is_test": True}),
            SimpleNamespace(exposure=10.0, info={"is_test": False}),
        ]

        exposure = harmonized_test_exposure(
            keyframes,
            1,
            mode="neighbor_average_v1",
        )

        self.assertAlmostEqual(exposure, 5.0)

    def test_off_mode_preserves_test_frame_exposure(self):
        keyframes = [
            SimpleNamespace(exposure=0.0, info={"is_test": False}),
            SimpleNamespace(exposure=-1.0, info={"is_test": True}),
            SimpleNamespace(exposure=10.0, info={"is_test": False}),
        ]

        exposure = harmonized_test_exposure(
            keyframes,
            1,
            mode="off",
        )

        self.assertAlmostEqual(exposure, -1.0)

    def test_source_time_guarded_interpolates_when_neighbor_exposure_span_is_small(self):
        keyframes = [
            SimpleNamespace(
                exposure=0.0,
                info={"is_test": False, "_paper_aligned_source_frame_id": 10},
            ),
            SimpleNamespace(
                exposure=-1.0,
                info={"is_test": True, "_paper_aligned_source_frame_id": 12},
            ),
            SimpleNamespace(
                exposure=3.0,
                info={"is_test": False, "_paper_aligned_source_frame_id": 20},
            ),
        ]

        exposure = harmonized_test_exposure(
            keyframes,
            1,
            mode="source_time_guarded_v1",
            max_neighbor_exposure_delta=4.0,
        )

        self.assertAlmostEqual(exposure, 0.6)

    def test_source_time_guarded_falls_back_when_neighbor_exposure_span_is_large(self):
        keyframes = [
            SimpleNamespace(
                exposure=0.0,
                info={"is_test": False, "_paper_aligned_source_frame_id": 10},
            ),
            SimpleNamespace(
                exposure=-1.0,
                info={"is_test": True, "_paper_aligned_source_frame_id": 12},
            ),
            SimpleNamespace(
                exposure=10.0,
                info={"is_test": False, "_paper_aligned_source_frame_id": 20},
            ),
        ]

        exposure = harmonized_test_exposure(
            keyframes,
            1,
            mode="source_time_guarded_v1",
            max_neighbor_exposure_delta=4.0,
        )

        self.assertAlmostEqual(exposure, 5.0)

    def test_adaptive_source_time_decision_selects_only_moderate_gap_low_pressure_scene(self):
        long_office_like = adaptive_source_time_exposure_scene_decision(
            train_keyframes=209,
            test_keyframes=87,
            texture_sampling_events=208,
            texture_sampling_applied=0,
            coverage_deficit_mean=0.059,
        )
        forest2_like = adaptive_source_time_exposure_scene_decision(
            train_keyframes=151,
            test_keyframes=100,
            texture_sampling_events=150,
            texture_sampling_applied=0,
            coverage_deficit_mean=0.0007,
        )
        desk_like = adaptive_source_time_exposure_scene_decision(
            train_keyframes=134,
            test_keyframes=21,
            texture_sampling_events=133,
            texture_sampling_applied=47,
            coverage_deficit_mean=0.0776,
        )
        xyz_like = adaptive_source_time_exposure_scene_decision(
            train_keyframes=69,
            test_keyframes=123,
            texture_sampling_events=68,
            texture_sampling_applied=1,
            coverage_deficit_mean=0.0784,
        )

        self.assertEqual(long_office_like["selected_mode"], "source_time_interp_v1")
        self.assertEqual(forest2_like["selected_mode"], "neighbor_average_v1")
        self.assertEqual(desk_like["selected_mode"], "neighbor_average_v1")
        self.assertEqual(xyz_like["selected_mode"], "neighbor_average_v1")
        self.assertEqual(long_office_like["reason"], "moderate_density_coverage_gap")

    def test_dark_scene_exposure_decision_disables_harmonization_for_dim_unmasked_scene(self):
        decision = dark_scene_test_exposure_scene_decision(
            {
                "mode": "dark_scene_fixed_black_v1",
                "dark_scene_running_mean": 0.38,
                "dark_scene_observations": 1,
                "dark_scene_mask_blocked": 0,
            }
        )

        self.assertEqual(decision["selected_mode"], "off")
        self.assertEqual(decision["reason"], "dim_unmasked_scene")

    def test_dark_scene_exposure_decision_keeps_neighbor_average_for_masked_scene(self):
        decision = dark_scene_test_exposure_scene_decision(
            {
                "mode": "dark_scene_fixed_black_v1",
                "dark_scene_running_mean": 0.20,
                "dark_scene_observations": 0,
                "dark_scene_mask_blocked": 4440,
            }
        )

        self.assertEqual(decision["selected_mode"], "neighbor_average_v1")
        self.assertEqual(decision["reason"], "masked_scene")

    def test_dark_scene_exposure_decision_keeps_neighbor_average_for_bright_scene(self):
        decision = dark_scene_test_exposure_scene_decision(
            {
                "mode": "dark_scene_fixed_black_v1",
                "dark_scene_running_mean": 0.43,
                "dark_scene_observations": 1,
                "dark_scene_mask_blocked": 0,
            }
        )

        self.assertEqual(decision["selected_mode"], "neighbor_average_v1")
        self.assertEqual(decision["reason"], "brightness_out_of_range")


if __name__ == "__main__":
    unittest.main()
