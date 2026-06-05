from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.run_forest1_module_ablation_v1 import AblationSpec, build_ablation_specs, summarize


class AblationRunnerSpecTests(unittest.TestCase):
    def test_builds_ordered_module_specs_with_expected_switches(self):
        specs = build_ablation_specs(Path("out"), max_frames=300)
        self.assertEqual(
            [s.name for s in specs],
            [
                "A0_baseline_passthrough",
                "A1_risk_only_no_pool",
                "A2_risk_plus_pool_no_commit",
                "A3_true_source_commit_no_control",
                "A4_commit_control_no_density",
                "A5_optional_commit_control_plus_density",
            ],
        )
        args_by_name = {s.name: s.extra_args for s in specs}
        self.assertIn("paper_aligned_baseline_passthrough", args_by_name["A0_baseline_passthrough"])
        self.assertIn("--paper_aligned_semantic_recovery_max_attempts", args_by_name["A1_risk_only_no_pool"])
        self.assertIn("0", args_by_name["A1_risk_only_no_pool"])
        self.assertIn("--paper_aligned_recovery_commit_bridge", args_by_name["A2_risk_plus_pool_no_commit"])
        self.assertIn("off", args_by_name["A2_risk_plus_pool_no_commit"])
        self.assertIn("true_source_commit", args_by_name["A3_true_source_commit_no_control"])
        self.assertIn("recovery_commit_early_seed_v7", args_by_name["A4_commit_control_no_density"])
        self.assertIn("target_band_v2_2_2_1", args_by_name["A5_optional_commit_control_plus_density"])
        self.assertTrue(all(s.model_dir.parts[0] == "out" for s in specs))

    def test_full_coupled_spec_accepts_explicit_density_band(self):
        specs = build_ablation_specs(
            Path("out"),
            max_frames=300,
            direct_density_upper_per_100=70.0,
            direct_density_hard_upper_per_100=85.0,
        )

        a5_args = {s.name: s.extra_args for s in specs}["A5_optional_commit_control_plus_density"]

        self.assertIn("--paper_aligned_direct_density_upper_per_100", a5_args)
        self.assertIn("70.0", a5_args)
        self.assertIn("--paper_aligned_direct_density_hard_upper_per_100", a5_args)
        self.assertIn("85.0", a5_args)

    def test_summary_falls_back_to_frame_metrics_for_pose_means(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "A5_full_coupled" / "short300"
            model_dir.mkdir(parents=True)
            (model_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "num keyframes": 2,
                        "PSNR": 20.0,
                        "SSIM": 0.6,
                        "LPIPS": 0.3,
                    }
                ),
                encoding="utf-8",
            )
            (model_dir / "frame_metrics.csv").write_text(
                "\n".join(
                    [
                        "psnr,ssim,lpips,abs_trans_error,abs_rot_error_deg",
                        "20.0,0.60,0.30,0.10,1.00",
                        "22.0,0.70,0.20,0.30,3.00",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            row = summarize(
                AblationSpec("A5_optional_commit_control_plus_density", model_dir, []),
                SimpleNamespace(max_frames=300),
                {"PSNR": 18.0, "SSIM": 0.5, "LPIPS": 0.4, "R°": 4.0, "t": 1.0},
                returncode=0,
            )

        self.assertEqual(row["R°"], 2.0)
        self.assertEqual(row["t"], 0.2)
        self.assertEqual(row["r_delta_to_baseline"], -2.0)
        self.assertEqual(row["t_delta_to_baseline"], -0.8)


if __name__ == "__main__":
    unittest.main()
