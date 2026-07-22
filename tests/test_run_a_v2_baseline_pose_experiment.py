import json
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path

from tools.run_a_v2_baseline_pose_experiment import (
    SCENES,
    VARIANTS,
    _summary,
    _write_manifest,
    build_command,
    build_specs,
    validate_single_gpu,
)


class AV2BaselinePoseRunnerTests(unittest.TestCase):
    def test_manifest_preserves_jobs_from_previous_batched_invocations(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            _write_manifest(
                output_root,
                [{"job_id": "repeat01:bonsai:a_off", "returncode": 0}],
            )
            _write_manifest(
                output_root,
                [{"job_id": "repeat02:bonsai:a_off", "returncode": 0}],
            )
            rows = json.loads((output_root / "manifest.json").read_text())

        self.assertEqual(
            [row["job_id"] for row in rows],
            ["repeat01:bonsai:a_off", "repeat02:bonsai:a_off"],
        )

    def test_summary_reads_the_trace_verification_accepted_key(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = build_specs(
                output_root=Path(directory),
                variants=["a_v2"],
                scene_names=["bonsai"],
                repeats=[1],
            )[0]
            spec.model_dir.mkdir(parents=True)
            (spec.model_dir / "pose_initialization_risk_trace.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "verification_attempts": 37,
                            "verification_accepted": 3,
                        }
                    }
                ),
                encoding="utf-8",
            )

            summary = _summary(spec, gpu="6", returncode=0)

        self.assertEqual(summary["a_attempts"], 37)
        self.assertEqual(summary["a_accepts"], 3)

    def test_direct_script_entrypoint_can_resolve_repo_modules(self):
        root = Path(__file__).resolve().parents[1]
        process = subprocess.run(
            [
                sys.executable,
                str(root / "tools" / "run_a_v2_baseline_pose_experiment.py"),
                "--help",
            ],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("--gpu", process.stdout)

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
            v2_threshold_index = command.index(
                "--pose_verification_v2_min_improvement"
            ) + 1
            self.assertEqual(command[v2_threshold_index], "0.0")
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
