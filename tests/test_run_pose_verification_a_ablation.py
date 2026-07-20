import tempfile
import unittest
from pathlib import Path

from tools.run_pose_verification_a_ablation import (
    VARIANTS,
    build_command,
    build_specs,
    distribute_specs,
    validate_gpus,
)
from tools.prewarm_model_caches import target_image_size


class PoseVerificationAblationRunnerTest(unittest.TestCase):
    def test_variants_only_change_pose_a_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            specs = build_specs(
                Path(directory),
                list(VARIANTS),
                ["bonsai"],
            )
        commands = {spec.variant: build_command(spec) for spec in specs}
        joined = {name: " ".join(command) for name, command in commands.items()}

        self.assertIn("--pose_initialization_risk_mode off", joined["v31_control"])
        self.assertIn("--pose_initialization_risk_mode observe_v1", joined["a_observe"])
        self.assertIn("--pose_initialization_risk_mode verify_v1", joined["a_full"])
        for command in joined.values():
            self.assertIn("--pose_risk_utility_admission_mode off", command)
            self.assertIn("baseline_render_lock_intra_frame_v31", command)
            self.assertIn("--enable_reboot", command)

    def test_scene_variants_stay_on_the_same_gpu_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            specs = build_specs(
                Path(directory),
                list(VARIANTS),
                ["bonsai", "counter", "garden"],
            )
        groups = distribute_specs(specs, ["1", "3"])
        scene_to_gpus = {}
        for gpu, worker_specs in groups.items():
            for spec in worker_specs:
                scene_to_gpus.setdefault(spec.scene.name, set()).add(gpu)
        self.assertTrue(all(len(gpus) == 1 for gpus in scene_to_gpus.values()))
        self.assertTrue(all(len(worker_specs) in (3, 6) for worker_specs in groups.values()))

    def test_gpu_validation_limits_parallelism_to_three(self):
        self.assertEqual(validate_gpus(["1", "3", "5"]), ["1", "3", "5"])
        with self.assertRaises(ValueError):
            validate_gpus([])
        with self.assertRaises(ValueError):
            validate_gpus(["0", "1", "2", "3"])
        with self.assertRaises(ValueError):
            validate_gpus(["1", "1"])

    def test_cache_prewarm_uses_the_training_resolution_rule(self):
        self.assertEqual(target_image_size(640, 480), (640, 480))
        self.assertEqual(target_image_size(3000, 2000), (1500, 1000))


if __name__ == "__main__":
    unittest.main()
