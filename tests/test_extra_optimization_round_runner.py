import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tools.run_extra_optimization_round_ablation import (
    ACTIVE_SCENES,
    DEFAULT_BUDGETS,
    RoundSpec,
    budget_args,
    build_command,
    build_specs,
    build_worker_queues,
    is_complete,
    run_specs,
    summarize,
    validate_budgets,
    validate_gpus,
)
from tools.run_v31_a_official_pose_benchmark import SCENES


class ExtraOptimizationRoundRunnerTests(unittest.TestCase):
    def make_spec(self, root: Path, budget: int = 8, seed: int = 0) -> RoundSpec:
        run_dir = root / "sweep" / f"k{budget:02d}" / f"seed{seed}" / "desk"
        return RoundSpec(
            phase="sweep",
            budget=budget,
            seed=seed,
            scene=SCENES["desk"],
            run_dir=run_dir,
            model_dir=run_dir / "model",
        )

    def test_zero_budget_disables_only_extra_optimization(self):
        with tempfile.TemporaryDirectory() as directory:
            command = build_command(self.make_spec(Path(directory), budget=0, seed=1))

        self.assertIn("disable_extra_optimization", command)
        self.assertIn("--experiment_seed", command)
        self.assertEqual(command[command.index("--experiment_seed") + 1], "1")
        self.assertNotIn(
            "--paper_aligned_pose_render_extra_optimization_max_extra", command
        )

    def test_positive_budget_maps_fraction_and_cap(self):
        self.assertEqual(
            budget_args(12),
            [
                "--paper_aligned_pose_render_extra_optimization_fraction",
                "0.383333333333",
                "--paper_aligned_pose_render_extra_optimization_max_extra",
                "12",
            ],
        )
        self.assertEqual(budget_args(8)[1], "0.25")

    def test_gpu_and_budget_validation_reject_invalid_values(self):
        self.assertEqual(validate_gpus(["5", "6", "7"]), ("5", "6", "7"))
        self.assertEqual(validate_budgets([0, 2, 8]), (0, 2, 8))
        for invalid in ([], ["0", "0"], ["0", "1", "2", "3"]):
            with self.subTest(gpus=invalid), self.assertRaises(ValueError):
                validate_gpus(invalid)
        for invalid in ([], [0, 0], [-1, 2], [1, 2]):
            with self.subTest(budgets=invalid), self.assertRaises(ValueError):
                validate_budgets(invalid)

    def test_sweep_has_120_unique_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            specs = build_specs(
                output_root=Path(directory),
                phase="sweep",
                scene_names=ACTIVE_SCENES,
                budgets=DEFAULT_BUDGETS,
                seeds=(0, 1, 2),
            )

        self.assertEqual(len(specs), 120)
        self.assertEqual(len({spec.job_id for spec in specs}), 120)

    def test_worker_queues_keep_paired_blocks_on_one_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            specs = build_specs(
                output_root=Path(directory),
                phase="sweep",
                scene_names=("bonsai", "desk"),
                budgets=(0, 2, 4, 8),
                seeds=(0, 1, 2),
            )
        queues = build_worker_queues(specs, ("5", "6", "7"))
        assignments = {
            (spec.scene.name, spec.seed): gpu
            for gpu, jobs in queues.items()
            for spec in jobs
        }

        for scene_name in ("bonsai", "desk"):
            for seed in (0, 1, 2):
                gpu = assignments[(scene_name, seed)]
                budgets = {
                    spec.budget
                    for spec in queues[gpu]
                    if spec.scene.name == scene_name and spec.seed == seed
                }
                self.assertEqual(budgets, {0, 2, 4, 8})

    def test_completion_requires_success_status_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = self.make_spec(Path(directory))
            spec.run_dir.mkdir(parents=True)
            spec.model_dir.mkdir(parents=True)
            (spec.run_dir / "run_status.json").write_text(
                json.dumps({"returncode": 0}), encoding="utf-8"
            )
            self.assertFalse(is_complete(spec))

            (spec.model_dir / "metadata.json").write_text("{}", encoding="utf-8")
            self.assertTrue(is_complete(spec))

            (spec.model_dir / "metadata.json").write_text("[]", encoding="utf-8")
            self.assertFalse(is_complete(spec))

    def test_summary_reports_requested_and_realized_iterations(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = self.make_spec(Path(directory), budget=8, seed=2)
            spec.model_dir.mkdir(parents=True)
            (spec.model_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "PSNR": 24.5,
                        "SSIM": 0.8,
                        "LPIPS": 0.2,
                        "time": 12.0,
                        "pose_render_extra_optimization": {
                            "mode": "render_response_v3",
                            "events": 10,
                            "applied": 4,
                            "extra_iterations_sum": 31,
                            "extra_iterations_mean": 7.75,
                        },
                    }
                ),
                encoding="utf-8",
            )

            row = summarize(spec, gpu="7", returncode=0)

        self.assertEqual(row["budget"], 8)
        self.assertEqual(row["requested_fraction"], 0.25)
        self.assertEqual(row["extra_iterations"], 31)
        self.assertEqual(row["extra_iterations_mean"], 7.75)

    def test_run_specs_rejects_empty_job_list(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "at least one job"):
                run_specs(
                    [],
                    output_root=Path(directory),
                    python=Path("/tmp/python"),
                    gpus=("0",),
                    skip_existing=False,
                    dry_run=True,
                )

    def test_dry_run_writes_a_complete_manifest_without_launching(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = build_specs(
                output_root=root,
                phase="smoke",
                scene_names=("desk",),
                budgets=(0, 8),
                seeds=(0,),
            )

            with redirect_stdout(io.StringIO()):
                rows = run_specs(
                    specs,
                    output_root=root,
                    python=Path("/does/not/run/python"),
                    gpus=("5",),
                    skip_existing=False,
                    dry_run=True,
                )

            manifest = json.loads(
                (root / "smoke" / "manifest.json").read_text(encoding="utf-8")
            )
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(manifest), 2)
        self.assertTrue(all(row["dry_run"] for row in manifest))


if __name__ == "__main__":
    unittest.main()
