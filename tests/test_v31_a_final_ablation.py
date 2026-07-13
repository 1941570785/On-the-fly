from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from args import get_args  # noqa: E402
from paper_aligned_policy.config import apply_coupled_innovation_defaults  # noqa: E402
from tools.evaluate_v31_a_final_ablation import (  # noqa: E402
    METHODS,
    aggregate_rendering,
    extract_trigger_summary,
)
from tools.run_v31_a_final_ablation import (  # noqa: E402
    VARIANTS,
    build_command,
    build_specs,
)


def parse_training_args(component_ablation: str):
    with tempfile.TemporaryDirectory() as tmp:
        argv = [
            "train.py",
            "-s",
            tmp,
            "-m",
            str(Path(tmp) / "out"),
            "--risk_admission_mode",
            "on_the_fly_innovation_v1",
            "--paper_aligned_pose_render_assimilation_profile",
            "baseline_render_lock_intra_frame_v31",
            "--paper_aligned_v31_component_ablation",
            component_ablation,
        ]
        with patch.object(sys, "argv", argv):
            return get_args()


class V31FinalComponentAblationTest(unittest.TestCase):
    def test_runner_is_executable_as_a_direct_script(self) -> None:
        process = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "run_v31_a_final_ablation.py"), "--help"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(process.returncode, 0, process.stderr)

    def test_evaluator_is_executable_as_a_direct_script(self) -> None:
        process = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "evaluate_v31_a_final_ablation.py"),
                "--help",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(process.returncode, 0, process.stderr)

    def test_disable_response_sampling_preserves_extra_optimization(self) -> None:
        args = parse_training_args("disable_response_sampling")
        cfg = apply_coupled_innovation_defaults(args)

        self.assertEqual(cfg.pose_render_texture_sampling, "off")
        self.assertEqual(cfg.pose_render_extra_optimization, "render_response_v3")
        self.assertEqual(cfg.render_frame_policy, "baseline_keyframe_lock_v1")

    def test_disable_extra_optimization_preserves_response_sampling(self) -> None:
        args = parse_training_args("disable_extra_optimization")
        cfg = apply_coupled_innovation_defaults(args)

        self.assertEqual(
            cfg.pose_render_texture_sampling,
            "residual_edge_response_guard_v2",
        )
        self.assertEqual(cfg.pose_render_extra_optimization, "off")
        self.assertEqual(cfg.render_frame_policy, "baseline_keyframe_lock_v1")

    def test_runner_defines_leave_one_out_matrix(self) -> None:
        self.assertEqual(
            tuple(VARIANTS),
            (
                "w_o_pose_risk_a",
                "w_o_response_sampling",
                "w_o_extra_optimization",
                "full",
            ),
        )

    def test_runner_builds_36_unique_official_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            specs = build_specs(Path(tmp), list(VARIANTS), None)

        self.assertEqual(len(specs), 36)
        self.assertEqual(len({spec.model_dir for spec in specs}), 36)
        self.assertTrue(all(spec.scene.test_hold in {8, 10, 30} for spec in specs))

    def test_commands_remove_only_the_named_component(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            specs = build_specs(Path(tmp), list(VARIANTS), ["bonsai"])
        commands = {
            spec.variant: build_command(spec, python=Path("python"))
            for spec in specs
        }

        for command in commands.values():
            self.assertIn("baseline_render_lock_intra_frame_v31", command)
            self.assertIn("--enable_reboot", command)
            self.assertEqual(command[command.index("--test_hold") + 1], "8")

        no_a = commands["w_o_pose_risk_a"]
        self.assertEqual(no_a[no_a.index("--pose_initialization_risk_mode") + 1], "off")
        self.assertNotIn("--paper_aligned_v31_component_ablation", no_a)

        no_sampling = commands["w_o_response_sampling"]
        self.assertIn("pose_quarantine_v1", no_sampling)
        self.assertEqual(
            no_sampling[
                no_sampling.index("--paper_aligned_v31_component_ablation") + 1
            ],
            "disable_response_sampling",
        )

        no_extra = commands["w_o_extra_optimization"]
        self.assertIn("pose_quarantine_v1", no_extra)
        self.assertEqual(
            no_extra[
                no_extra.index("--paper_aligned_v31_component_ablation") + 1
            ],
            "disable_extra_optimization",
        )

        full = commands["full"]
        self.assertIn("pose_quarantine_v1", full)
        self.assertNotIn("--paper_aligned_v31_component_ablation", full)

    def test_evaluator_uses_frozen_baseline_and_four_ablation_methods(self) -> None:
        self.assertEqual(
            METHODS,
            (
                "baseline",
                "w_o_pose_risk_a",
                "w_o_response_sampling",
                "w_o_extra_optimization",
                "full",
            ),
        )

    def test_trigger_summary_reports_actual_component_activity(self) -> None:
        metadata = {
            "pose_render_texture_sampling": {
                "mode": "residual_edge_response_guard_v2",
                "events": 20,
                "applied": 3,
                "budget_shift_abs_sum": 0.25,
            },
            "pose_render_extra_optimization": {
                "mode": "render_response_v3",
                "events": 18,
                "applied": 2,
                "extra_iterations_sum": 16,
            },
        }
        risk_trace = {
            "events": [
                {"risk_candidate": True, "pose_reference_quarantined": True},
                {"risk_candidate": False, "pose_reference_quarantined": False},
            ]
        }

        summary = extract_trigger_summary(metadata, risk_trace)

        self.assertEqual(summary["sampling_applied"], 3)
        self.assertEqual(summary["extra_optimization_applied"], 2)
        self.assertEqual(summary["extra_iterations"], 16)
        self.assertEqual(summary["risk_candidates"], 1)
        self.assertEqual(summary["quarantined"], 1)

    def test_rendering_aggregate_is_macro_average_per_method(self) -> None:
        rows = [
            {"method": "full", "PSNR": 20.0, "SSIM": 0.6, "LPIPS": 0.3, "time": 10.0},
            {"method": "full", "PSNR": 22.0, "SSIM": 0.8, "LPIPS": 0.2, "time": 14.0},
        ]

        aggregate = aggregate_rendering(rows)

        self.assertEqual(len(aggregate), 1)
        self.assertEqual(aggregate[0]["method"], "full")
        self.assertAlmostEqual(aggregate[0]["PSNR"], 21.0)
        self.assertAlmostEqual(aggregate[0]["SSIM"], 0.7)
        self.assertAlmostEqual(aggregate[0]["LPIPS"], 0.25)
        self.assertAlmostEqual(aggregate[0]["time"], 12.0)


if __name__ == "__main__":
    unittest.main()
