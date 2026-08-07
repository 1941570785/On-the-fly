from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.run_asr_gs_benchmark import (
    ROOT,
    SCENES,
    aggregate_dataset_rows,
    build_train_command,
    git_state,
    prepare_output_directory,
    validate_gpus,
)


class BenchmarkContractTests(unittest.TestCase):
    def test_protocol_has_three_scenes_per_dataset(self):
        counts = {}
        for scene in SCENES:
            counts[scene.dataset] = counts.get(scene.dataset, 0) + 1
        self.assertEqual(
            counts,
            {"Mip-NeRF360": 3, "StaticHikes": 3, "TUM RGB-D": 3},
        )

    def test_gpu_limit_is_enforced(self):
        self.assertEqual(validate_gpus(["0", "1", "2"]), ("0", "1", "2"))
        with self.assertRaises(ValueError):
            validate_gpus(["0", "1", "2", "3"])

    def test_command_exposes_only_public_ablation_flags(self):
        command = build_train_command(
            scene=SCENES[0],
            source_path=Path("/data/bonsai"),
            output_path=Path("/results/bonsai"),
            method="asr-gs",
            ablations=("a", "c"),
            seed=3,
            deterministic=True,
        )
        self.assertIn("--ablate-a", command)
        self.assertIn("--ablate-c", command)
        self.assertNotIn("--ablate-b", command)
        self.assertNotIn("max_extra_iterations", " ".join(command))

    def test_dataset_macro_is_arithmetic_mean(self):
        rows = [
            {
                "dataset": "D",
                "repeat": 1,
                "PSNR": value,
                "SSIM": value / 10,
                "LPIPS": value / 20,
                "time_seconds": value * 2,
            }
            for value in (1.0, 2.0, 3.0)
        ]
        result = aggregate_dataset_rows(rows)[0]
        self.assertEqual(result["PSNR"], 2.0)
        self.assertEqual(result["time_seconds"], 4.0)

    def test_force_recreates_only_a_nested_output_directory(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "results"
            target = root / "asr-gs" / "bonsai"
            target.mkdir(parents=True)
            marker = target / "stale.txt"
            marker.write_text("stale", encoding="utf-8")

            prepare_output_directory(target, root, force=True)

            self.assertTrue(target.is_dir())
            self.assertFalse(marker.exists())
            with self.assertRaises(ValueError):
                prepare_output_directory(root, root, force=True)

    def test_manifest_git_state_identifies_the_exact_working_tree(self):
        state = git_state(ROOT)
        self.assertEqual(len(state["revision"]), 40)
        self.assertIsInstance(state["dirty"], bool)
        self.assertEqual(len(state["working_tree_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
