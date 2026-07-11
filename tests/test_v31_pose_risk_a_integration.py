import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from args import get_args
from tools.run_v31_pose_risk_a_experiment import (
    SCENE_SOURCES,
    SUMMARY_FIELDS,
    VARIANTS,
    build_command,
    build_specs,
    parse_args,
    run_one,
)


class V31PoseRiskAIntegrationTests(unittest.TestCase):
    def test_cli_accepts_pose_initialization_risk_options(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "train.py",
                "-s",
                td,
                "-m",
                str(Path(td) / "out"),
                "--pose_initialization_risk_mode",
                "isolate_v1",
                "--pose_initialization_risk_absolute_threshold",
                "0.58",
                "--pose_initialization_risk_adaptive_sigma",
                "2.2",
                "--pose_initialization_risk_warmup",
                "10",
                "--pose_initialization_risk_history_size",
                "48",
                "--pose_initialization_risk_cooldown_frames",
                "7",
            ]
            with patch.object(sys, "argv", argv):
                args = get_args()

        self.assertEqual(args.pose_initialization_risk_mode, "isolate_v1")
        self.assertAlmostEqual(args.pose_initialization_risk_absolute_threshold, 0.58)
        self.assertAlmostEqual(args.pose_initialization_risk_adaptive_sigma, 2.2)
        self.assertEqual(args.pose_initialization_risk_warmup, 10)
        self.assertEqual(args.pose_initialization_risk_history_size, 48)
        self.assertEqual(args.pose_initialization_risk_cooldown_frames, 7)

    def test_training_loop_contains_post_pose_gate_and_trace(self):
        train_source = Path("train.py").read_text(encoding="utf-8")

        self.assertIn("PoseInitializationRiskGate", train_source)
        self.assertIn("pose_initialization_risk_gate.evaluate", train_source)
        self.assertIn("pose_initialization_risk_trace.json", train_source)
        self.assertIn('pose_initialization_risk_decision["isolated"]', train_source)

    def test_isolation_restores_temporary_pose_match_state(self):
        train_source = Path("train.py").read_text(encoding="utf-8")

        self.assertIn("pose_initialization_risk_match_before", train_source)
        self.assertIn(
            "_restore_pose_match_state(\n"
            "                                desc_kpts,\n"
            "                                prev_keyframes_for_pose,",
            train_source,
        )

    def test_runner_keeps_v31_fixed_and_varies_only_a_module(self):
        args = parse_args(["--dry_run"])
        specs = build_specs(Path("/tmp/out"), ["bonsai", "forest1"])

        self.assertEqual(args.test_hold, 8)
        self.assertEqual(set(SCENE_SOURCES), {"bonsai", "forest1"})
        self.assertEqual(
            [name for name, _ in VARIANTS],
            ["V31_control", "V31_A_observe", "V31_A_isolate"],
        )
        self.assertEqual(len(specs), 6)

        commands = {spec.variant: build_command(spec, args) for spec in specs[:3]}
        for command in commands.values():
            joined = " ".join(command)
            self.assertIn("baseline_render_lock_intra_frame_v31", joined)
            self.assertIn("on_the_fly_innovation_v1", joined)
        self.assertIn(" off", " ".join(commands["V31_control"]))
        self.assertIn("observe_v1", " ".join(commands["V31_A_observe"]))
        self.assertIn("isolate_v1", " ".join(commands["V31_A_isolate"]))
        isolate_command = " ".join(commands["V31_A_isolate"])
        self.assertIn("--pose_initialization_risk_absolute_threshold 0.10", isolate_command)
        self.assertIn("--pose_initialization_risk_adaptive_sigma 2.0", isolate_command)
        self.assertIn("--pose_initialization_risk_cooldown_frames 12", isolate_command)
        self.assertIn("delta_psnr_to_observe", SUMMARY_FIELDS)

    def test_skip_existing_preserves_original_command_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td)
            spec = build_specs(output_root, ["bonsai"])[0]
            spec.model_dir.mkdir(parents=True)
            original_command = ["python", "train.py", "--original-run"]
            (spec.model_dir / "command.json").write_text(
                json.dumps(original_command), encoding="utf-8"
            )
            (spec.model_dir / "metadata.json").write_text(
                json.dumps({"PSNR": 1.0}), encoding="utf-8"
            )

            row = run_one(spec, parse_args(["--skip_existing"]))

            self.assertEqual(row["PSNR"], 1.0)
            self.assertEqual(
                json.loads((spec.model_dir / "command.json").read_text(encoding="utf-8")),
                original_command,
            )


if __name__ == "__main__":
    unittest.main()
