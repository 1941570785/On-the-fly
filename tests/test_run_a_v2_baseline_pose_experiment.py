import tempfile
import unittest
from pathlib import Path

from tools.run_a_v2_baseline_pose_experiment import (
    SCENES,
    VARIANTS,
    build_command,
    build_specs,
    validate_single_gpu,
)


class AV2BaselinePoseRunnerTests(unittest.TestCase):
    def test_requires_exactly_one_physical_gpu_identifier(self):
        self.assertEqual(validate_single_gpu("7"), "7")
        with self.assertRaises(ValueError):
            validate_single_gpu("")
        with self.assertRaises(ValueError):
            validate_single_gpu("0,1")

    def test_default_matrix_contains_three_variants_five_repeats_and_nine_scenes(self):
        with tempfile.TemporaryDirectory() as directory:
            specs = build_specs(
                output_root=Path(directory),
                variants=list(VARIANTS),
                scene_names=list(SCENES),
                repeats=[1, 2, 3, 4, 5],
            )

        self.assertEqual(len(specs), 3 * 5 * 9)
        self.assertEqual({spec.repeat for spec in specs}, {1, 2, 3, 4, 5})
        self.assertEqual({spec.variant for spec in specs}, set(VARIANTS))
        self.assertEqual({spec.scene.name for spec in specs}, set(SCENES))
        self.assertTrue(
            all(f"repeat_{spec.repeat:02d}" in str(spec.run_dir) for spec in specs)
        )

    def test_variants_only_change_a_mode_and_keep_final_k16_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            specs = build_specs(
                output_root=Path(directory),
                variants=list(VARIANTS),
                scene_names=["bonsai"],
                repeats=[1],
            )
        commands = {spec.variant: build_command(spec) for spec in specs}

        for variant, mode in VARIANTS.items():
            command = commands[variant]
            mode_index = command.index("--pose_initialization_risk_mode") + 1
            self.assertEqual(command[mode_index], mode)
            self.assertIn("baseline_render_lock_intra_frame_v31", command)
            self.assertIn("--pose_risk_utility_admission_mode", command)
            self.assertIn("--paper_aligned_pose_render_extra_optimization_max_extra", command)
            budget_index = command.index(
                "--paper_aligned_pose_render_extra_optimization_max_extra"
            ) + 1
            self.assertEqual(command[budget_index], "16")

        normalized = []
        for command in commands.values():
            values = list(command)
            values[values.index("-m") + 1] = "MODEL_DIR"
            values[values.index("--pose_initialization_risk_mode") + 1] = "A_MODE"
            normalized.append(values)
        self.assertTrue(all(command == normalized[0] for command in normalized[1:]))

    def test_each_scene_keeps_all_variants_adjacent_with_rotated_repeat_order(self):
        with tempfile.TemporaryDirectory() as directory:
            specs = build_specs(
                output_root=Path(directory),
                variants=list(VARIANTS),
                scene_names=["bonsai"],
                repeats=[1, 2, 3],
            )

        by_repeat = {
            repeat: [spec.variant for spec in specs if spec.repeat == repeat]
            for repeat in (1, 2, 3)
        }
        self.assertEqual(by_repeat[1], ["a_off", "a_current", "a_v2"])
        self.assertEqual(by_repeat[2], ["a_current", "a_v2", "a_off"])
        self.assertEqual(by_repeat[3], ["a_v2", "a_off", "a_current"])


if __name__ == "__main__":
    unittest.main()
