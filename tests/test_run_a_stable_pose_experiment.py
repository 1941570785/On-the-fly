import tempfile
import unittest
from pathlib import Path

from tools.run_a_stable_pose_experiment import (
    build_command,
    build_specs,
    paired_gpu_assignments,
    validate_gpus,
)


class StablePoseExperimentRunnerTests(unittest.TestCase):
    def test_gpu_validation_requires_unique_set_of_at_most_four(self):
        self.assertEqual(validate_gpus(["1", "2", "3", "5"]), ["1", "2", "3", "5"])
        with self.assertRaises(ValueError):
            validate_gpus(["1", "1"])
        with self.assertRaises(ValueError):
            validate_gpus(["0", "1", "2", "3", "4"])

    def test_build_specs_crosses_repeats_presets_and_scenes(self):
        with tempfile.TemporaryDirectory() as td:
            specs = build_specs(
                output_root=Path(td),
                presets=["a_off", "photo_i2_g001"],
                scene_names=["counter", "forest1"],
                repeats=[1, 2],
            )

        self.assertEqual(len(specs), 8)
        self.assertEqual(len({spec.job_id for spec in specs}), 8)

    def test_paired_gpu_assignment_keeps_scene_repeat_on_one_physical_gpu(self):
        with tempfile.TemporaryDirectory() as td:
            specs = build_specs(
                output_root=Path(td),
                presets=["a_off", "photo_raw_i4_g002", "photo_raw_i8_lr5_g005"],
                scene_names=["counter", "forest1"],
                repeats=[1, 2],
            )

        assignments = paired_gpu_assignments(specs, ["1", "2", "3", "5"])
        grouped = {}
        for spec in specs:
            grouped.setdefault((spec.repeat, spec.scene.name), set()).add(
                assignments[spec.job_id]
            )

        self.assertTrue(all(len(gpus) == 1 for gpus in grouped.values()))
        self.assertEqual(set(assignments.values()), {"1", "2", "3", "5"})

    def test_photo_command_preserves_verify_v2_and_enables_only_pose_review(self):
        with tempfile.TemporaryDirectory() as td:
            spec = build_specs(
                output_root=Path(td),
                presets=["photo_i4_g002"],
                scene_names=["forest1"],
                repeats=[1],
            )[0]
            command = build_command(spec)

        joined = " ".join(command)
        self.assertIn("--pose_initialization_risk_mode verify_v2", joined)
        self.assertIn("--pose_verification_photometric_review", command)
        self.assertIn("--pose_verification_photometric_iterations 4", joined)
        self.assertIn(
            "--pose_verification_photometric_min_relative_improvement 0.002",
            joined,
        )
        self.assertIn("--pose_risk_utility_admission_mode off", joined)

    def test_a_off_command_does_not_enable_photometric_review(self):
        with tempfile.TemporaryDirectory() as td:
            spec = build_specs(
                output_root=Path(td),
                presets=["a_off"],
                scene_names=["counter"],
                repeats=[1],
            )[0]
            command = build_command(spec)

        self.assertIn("off", command)
        self.assertNotIn("--pose_verification_photometric_review", command)

    def test_a_observe_records_risk_without_applying_any_review(self):
        with tempfile.TemporaryDirectory() as td:
            spec = build_specs(
                output_root=Path(td),
                presets=["a_observe"],
                scene_names=["counter"],
                repeats=[1],
            )[0]
            command = build_command(spec)

        joined = " ".join(command)
        self.assertIn("--pose_initialization_risk_mode observe_v1", joined)
        self.assertNotIn("--pose_verification_photometric_review", command)

    def test_isolation_presets_keep_pose_fixed_and_only_tune_admission(self):
        with tempfile.TemporaryDirectory() as td:
            default_spec, conservative_spec, severe_spec = build_specs(
                output_root=Path(td),
                presets=[
                    "a_isolate",
                    "a_isolate_conservative",
                    "a_isolate_severe",
                ],
                scene_names=["forest1"],
                repeats=[1],
            )

        default_command = " ".join(build_command(default_spec))
        conservative_command = " ".join(build_command(conservative_spec))
        severe_command = " ".join(build_command(severe_spec))
        self.assertIn("--pose_initialization_risk_mode isolate_v1", default_command)
        self.assertNotIn("--pose_verification_photometric_review", default_command)
        self.assertIn(
            "--pose_initialization_risk_absolute_threshold 0.12",
            conservative_command,
        )
        self.assertIn(
            "--pose_initialization_risk_cooldown_frames 20",
            conservative_command,
        )
        self.assertIn(
            "--pose_initialization_risk_absolute_threshold 0.15",
            severe_command,
        )
        self.assertIn(
            "--pose_initialization_risk_cooldown_frames 20",
            severe_command,
        )

    def test_quarantine_keeps_keyframe_and_excludes_only_pose_reference(self):
        with tempfile.TemporaryDirectory() as td:
            spec = build_specs(
                output_root=Path(td),
                presets=["a_quarantine"],
                scene_names=["forest1"],
                repeats=[1],
            )[0]

        command = " ".join(build_command(spec))
        self.assertIn("--pose_initialization_risk_mode observe_v1", command)
        self.assertIn(
            "--pose_risk_utility_admission_mode pose_quarantine_v1",
            command,
        )
        self.assertNotIn("--pose_verification_photometric_review", command)

    def test_utility_quarantine_uses_low_utility_risk_admission(self):
        with tempfile.TemporaryDirectory() as td:
            spec = build_specs(
                output_root=Path(td),
                presets=["a_quarantine_utility"],
                scene_names=["forest1"],
                repeats=[1],
            )[0]

        command = " ".join(build_command(spec))
        self.assertIn("--pose_initialization_risk_mode observe_v1", command)
        self.assertIn(
            "--pose_risk_utility_admission_mode pose_quarantine_utility_v1",
            command,
        )
        self.assertIn(
            "--pose_risk_utility_quarantine_risk_margin 0.04",
            command,
        )
        self.assertNotIn("--pose_verification_photometric_review", command)

    def test_utility_observe_matches_probe_cost_without_quarantine(self):
        with tempfile.TemporaryDirectory() as td:
            spec = build_specs(
                output_root=Path(td),
                presets=["a_utility_observe"],
                scene_names=["forest1"],
                repeats=[1],
            )[0]

        command = " ".join(build_command(spec))
        self.assertIn("--pose_initialization_risk_mode observe_v1", command)
        self.assertIn(
            "--pose_risk_utility_admission_mode observe_v1",
            command,
        )
        self.assertNotIn("pose_quarantine", command)
        self.assertNotIn("--pose_verification_photometric_review", command)

    def test_utility_observe_can_probe_existing_verification_candidates(self):
        with tempfile.TemporaryDirectory() as td:
            spec = build_specs(
                output_root=Path(td),
                presets=["a_utility_observe_verify"],
                scene_names=["counter"],
                repeats=[1],
            )[0]

        command = " ".join(build_command(spec))
        self.assertIn(
            "--pose_risk_utility_admission_mode observe_v1",
            command,
        )
        self.assertIn(
            "--pose_risk_utility_use_verification_candidates",
            command,
        )
        self.assertNotIn("pose_quarantine", command)

    def test_severe_quarantine_uses_existing_severe_flag_and_short_cooldown(self):
        with tempfile.TemporaryDirectory() as td:
            spec = build_specs(
                output_root=Path(td),
                presets=["a_quarantine_severe"],
                scene_names=["forest1"],
                repeats=[1],
            )[0]

        command = " ".join(build_command(spec))
        self.assertIn(
            "--pose_risk_utility_admission_mode pose_quarantine_severe_v1",
            command,
        )
        self.assertIn(
            "--pose_risk_utility_use_verification_candidates",
            command,
        )
        self.assertIn(
            "--pose_risk_utility_quarantine_cooldown_frames 12",
            command,
        )

    def test_test_only_preset_is_explicit_in_command(self):
        with tempfile.TemporaryDirectory() as td:
            spec = build_specs(
                output_root=Path(td),
                presets=["photo_test_i2_g001"],
                scene_names=["forest1"],
                repeats=[1],
            )[0]
            command = build_command(spec)

        self.assertIn(
            "--pose_verification_photometric_scope test_only",
            " ".join(command),
        )

    def test_raw_seed_preset_observes_risk_without_geometric_refinement(self):
        with tempfile.TemporaryDirectory() as td:
            spec = build_specs(
                output_root=Path(td),
                presets=["photo_raw_i4_g002"],
                scene_names=["forest1"],
                repeats=[1],
            )[0]
            command = build_command(spec)

        joined = " ".join(command)
        self.assertIn("--pose_initialization_risk_mode observe_v1", joined)
        self.assertIn("--pose_verification_photometric_seed raw", joined)
        self.assertIn("--pose_verification_photometric_review", command)

    def test_strong_raw_preset_scales_only_bounded_pose_review(self):
        with tempfile.TemporaryDirectory() as td:
            spec = build_specs(
                output_root=Path(td),
                presets=["photo_raw_i8_lr5_g005"],
                scene_names=["forest1"],
                repeats=[1],
            )[0]
            command = build_command(spec)

        joined = " ".join(command)
        self.assertIn("--pose_verification_photometric_iterations 8", joined)
        self.assertIn("--pose_verification_photometric_lr_scale 5.0", joined)
        self.assertIn(
            "--pose_verification_photometric_min_relative_improvement 0.005",
            joined,
        )


if __name__ == "__main__":
    unittest.main()
