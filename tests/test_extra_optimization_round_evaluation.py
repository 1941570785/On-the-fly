import json
import tempfile
import unittest
from pathlib import Path

from tools.evaluate_extra_optimization_round_ablation import (
    active_macro_summary,
    evaluate_rows,
    pair_against_zero,
    paired_budget_summary,
    scene_summary,
    select_budget,
)


class ExtraOptimizationRoundEvaluationTests(unittest.TestCase):
    @staticmethod
    def synthetic_rows():
        rows = []
        scene_offsets = {"bonsai": 0.0, "desk": 1.0}
        gains = {
            0: [0.0, 0.0, 0.0],
            4: [0.19, 0.20, 0.21],
            8: [0.21, 0.201, 0.20],
        }
        for scene, offset in scene_offsets.items():
            dataset = "MipNeRF360" if scene == "bonsai" else "TUM"
            for seed in range(3):
                for budget in (0, 4, 8):
                    gain = gains[budget][seed]
                    if scene == "desk" and budget == 4 and seed == 2:
                        gain += 0.02
                    rows.append(
                        {
                            "job_id": f"sweep:{scene}:seed{seed}:k{budget}",
                            "phase": "sweep",
                            "dataset": dataset,
                            "scene": scene,
                            "seed": seed,
                            "budget": budget,
                            "returncode": 0,
                            "dry_run": False,
                            "PSNR": 20.0 + offset + gain,
                            "SSIM": 0.70 + gain * 0.05 + (0.0001 if budget == 4 else 0.0),
                            "LPIPS": 0.30 - gain * 0.05,
                            "time": 10.0 + budget * 0.1,
                            "wall_time_seconds": 11.0 + budget * 0.1,
                            "extra_events": 10,
                            "extra_applied": 0 if budget == 0 else 5,
                            "extra_iterations": budget * 5,
                            "extra_iterations_mean": float(budget),
                            "num_keyframes": 100,
                            "num_anchors": 2,
                        }
                    )
        return rows

    def test_pairing_uses_positive_gain_for_better_lpips(self):
        paired = pair_against_zero(self.synthetic_rows())
        summary = paired_budget_summary(paired, bootstrap_replicates=200)

        self.assertEqual(len(paired), 12)
        self.assertGreater(summary[8]["PSNR_gain_mean"], 0.0)
        self.assertGreater(summary[8]["LPIPS_gain_mean"], 0.0)

    def test_selection_chooses_smallest_statistically_equivalent_budget(self):
        rows = self.synthetic_rows()
        scenes = ("bonsai", "desk")
        curve = active_macro_summary(scene_summary(rows), active_scenes=scenes)

        selection = select_budget(
            rows,
            curve,
            active_scenes=scenes,
            bootstrap_replicates=500,
        )

        self.assertEqual(selection["max_psnr_budget"], 8)
        self.assertEqual(selection["selected_budget"], 4)
        evidence = {item["budget"]: item for item in selection["candidates"]}
        self.assertTrue(evidence[4]["equivalent"])

    def test_evaluate_rows_writes_required_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            selection = evaluate_rows(
                self.synthetic_rows(),
                output_dir=output_dir,
                active_scenes=("bonsai", "desk"),
                bootstrap_replicates=200,
            )

            required = (
                "round_scene_summary.csv",
                "round_active_macro_summary.csv",
                "round_paired_vs_zero.csv",
                "round_paired_vs_zero_summary.csv",
                "full9_selected_comparison.csv",
                "selection.json",
                "extra_round_quality_curve.png",
                "extra_round_quality_curve.pdf",
                "extra_round_quality_time_pareto.png",
                "extra_round_quality_time_pareto.pdf",
                "results.md",
            )
            for filename in required:
                with self.subTest(filename=filename):
                    path = output_dir / filename
                    self.assertTrue(path.is_file())
                    self.assertGreater(path.stat().st_size, 0)
            persisted = json.loads(
                (output_dir / "selection.json").read_text(encoding="utf-8")
            )

        self.assertEqual(selection["selected_budget"], 4)
        self.assertEqual(persisted["selected_budget"], 4)


if __name__ == "__main__":
    unittest.main()
