from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_v31_a_official_pose_benchmark import (  # noqa: E402
    SCENES,
    build_command,
    build_specs,
    validate_single_gpu,
)


class OfficialProtocolRunnerTest(unittest.TestCase):
    def test_official_holdouts_are_dataset_specific(self) -> None:
        self.assertEqual(SCENES["bonsai"].test_hold, 8)
        self.assertEqual(SCENES["forest1"].test_hold, 10)
        self.assertEqual(SCENES["desk"].test_hold, 30)

    def test_a_command_preserves_v31_and_enables_quarantine_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec = build_specs(Path(tmp), ["v31_a"], ["bonsai"])[0]
            command = build_command(spec, python=Path("python"))

        self.assertIn("baseline_render_lock_intra_frame_v31", command)
        self.assertIn("pose_quarantine_v1", command)
        self.assertEqual(command.count("--pose_risk_utility_review_iterations"), 1)
        review_index = command.index("--pose_risk_utility_review_iterations")
        self.assertEqual(command[review_index + 1], "0")
        hold_index = command.index("--test_hold")
        self.assertEqual(command[hold_index + 1], "8")

    def test_v31_command_disables_a_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec = build_specs(Path(tmp), ["v31"], ["forest1"])[0]
            command = build_command(spec, python=Path("python"))

        risk_index = command.index("--pose_initialization_risk_mode")
        utility_index = command.index("--pose_risk_utility_admission_mode")
        self.assertEqual(command[risk_index + 1], "off")
        self.assertEqual(command[utility_index + 1], "off")
        self.assertEqual(command[command.index("--test_hold") + 1], "10")

    def test_build_specs_are_serial_and_have_unique_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            specs = build_specs(
                Path(tmp),
                ["baseline", "v31", "v31_a"],
                ["bonsai", "forest1", "desk"],
            )

        self.assertEqual(len(specs), 9)
        self.assertEqual(len({spec.model_dir for spec in specs}), 9)
        self.assertEqual(
            [(spec.variant, spec.scene.name) for spec in specs[:3]],
            [("baseline", "bonsai"), ("baseline", "forest1"), ("baseline", "desk")],
        )

    def test_gpu_guard_accepts_exactly_one_visible_device(self) -> None:
        self.assertEqual(validate_single_gpu("7"), "7")
        self.assertEqual(validate_single_gpu(" GPU-abc "), "GPU-abc")
        for value in ("", "0,1", "0, 1"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_single_gpu(value)


if __name__ == "__main__":
    unittest.main()
