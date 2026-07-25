import unittest
from pathlib import Path

from tools.run_b_internal_ablation import (
    B_SIGNAL_MODES,
    XYZ_SCENE,
    build_jobs,
    build_train_command,
    validate_gpus,
)


class BInternalRunnerTests(unittest.TestCase):
    def test_runner_defines_only_the_five_internal_b_variants(self):
        self.assertEqual(
            B_SIGNAL_MODES,
            ("base", "r", "r_e", "r_d", "r_e_d"),
        )

    def test_each_repeat_uses_the_same_seed_for_every_variant(self):
        jobs = build_jobs(repeat=3, base_seed=17)
        self.assertEqual(len(jobs), 15)
        for repeat in range(1, 4):
            selected = [job for job in jobs if job.repeat == repeat]
            self.assertEqual(len(selected), 5)
            self.assertEqual({job.seed for job in selected}, {16 + repeat})

    def test_command_freezes_a_and_disables_c(self):
        command = build_train_command(
            scene=XYZ_SCENE,
            source_path=Path("/data/xyz"),
            output_path=Path("/results/r_e"),
            signal_mode="r_e",
            seed=4,
            deterministic=True,
        )
        self.assertIn("--method", command)
        self.assertIn("asr-gs", command)
        self.assertIn("--ablate-c", command)
        self.assertIn("--b-signal-mode", command)
        self.assertIn("r_e", command)
        self.assertNotIn("--ablate-a", command)
        self.assertNotIn("--ablate-b", command)
        self.assertIn("--deterministic", command)

    def test_runner_allows_at_most_three_unique_gpus(self):
        self.assertEqual(validate_gpus(["0", "1", "2"]), ("0", "1", "2"))
        with self.assertRaises(ValueError):
            validate_gpus(["0", "1", "2", "3"])
        with self.assertRaises(ValueError):
            validate_gpus(["0", "0"])


if __name__ == "__main__":
    unittest.main()
