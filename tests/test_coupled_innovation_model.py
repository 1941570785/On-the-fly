from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from args import get_args
from paper_aligned_policy import config as policy_config
from paper_aligned_policy.runtime_gate import PaperAlignedRuntimeGate


def _args(**overrides):
    base = {
        "risk_admission_mode": "on_the_fly_innovation_v1",
        "paper_aligned_contract_trace_path": "",
        "paper_aligned_recovery_commit_bridge": None,
        "paper_aligned_defer_recovery_support_bridge": None,
        "paper_aligned_recovery_commit_control": None,
        "paper_aligned_direct_density_control": None,
        "paper_aligned_direct_update_prev_desc_on_hold": None,
        "paper_aligned_tau_R_low": None,
        "paper_aligned_tau_R_high": None,
        "paper_aligned_tau_V": None,
        "paper_aligned_tau_V_min": None,
        "paper_aligned_tau_B": None,
        "paper_aligned_tau_Q": None,
        "paper_aligned_recovery_delay_frames": None,
        "paper_aligned_semantic_recovery_max_attempts": None,
        "paper_aligned_semantic_recovery_attempts_per_tick": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class CoupledInnovationModelTests(unittest.TestCase):
    def test_coupled_preset_resolves_complete_model_without_online_quality_metrics(self):
        resolver = getattr(policy_config, "resolve_coupled_innovation_config", None)
        self.assertIsNotNone(resolver)

        cfg = resolver(_args())

        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.requested_mode, "on_the_fly_innovation_v1")
        self.assertEqual(cfg.runtime_mode, "paper_aligned_semantic_v1")
        self.assertEqual(cfg.recovery_commit_bridge, "true_source_commit")
        self.assertEqual(cfg.defer_recovery_support_bridge, "v1")
        self.assertEqual(cfg.recovery_commit_control, "recovery_commit_early_seed_v7")
        self.assertEqual(cfg.direct_density_control, "target_band_v2_2_2_1")
        self.assertEqual(cfg.direct_update_prev_desc_on_hold, "off")
        self.assertEqual(cfg.thresholds.tau_V_min, 0.45)
        self.assertEqual(cfg.thresholds.tau_B, 0.12)
        self.assertEqual(cfg.online_quality_metric_fields, ())
        self.assertIn("PSNR", cfg.offline_stage_metric_fields)
        self.assertIn("absolute_relative_translation_error", cfg.offline_stage_metric_fields)

    def test_explicit_module_and_parameter_overrides_are_preserved_for_tuning(self):
        resolver = getattr(policy_config, "resolve_coupled_innovation_config", None)
        self.assertIsNotNone(resolver)

        cfg = resolver(
            _args(
                paper_aligned_defer_recovery_support_bridge="off",
                paper_aligned_recovery_commit_control="adaptive",
                paper_aligned_direct_density_control="target_band_v2_2",
                paper_aligned_tau_R_low=0.33,
                paper_aligned_tau_R_high=0.81,
                paper_aligned_tau_V=0.61,
                paper_aligned_tau_V_min=0.29,
                paper_aligned_tau_B=0.44,
                paper_aligned_tau_Q=0.17,
                paper_aligned_recovery_delay_frames=5,
                paper_aligned_semantic_recovery_max_attempts=7,
                paper_aligned_semantic_recovery_attempts_per_tick=2,
            )
        )

        self.assertEqual(cfg.defer_recovery_support_bridge, "off")
        self.assertEqual(cfg.recovery_commit_control, "adaptive")
        self.assertEqual(cfg.direct_density_control, "target_band_v2_2")
        self.assertEqual(cfg.thresholds.tau_R_low, 0.33)
        self.assertEqual(cfg.thresholds.tau_R_high, 0.81)
        self.assertEqual(cfg.thresholds.tau_V, 0.61)
        self.assertEqual(cfg.thresholds.tau_V_min, 0.29)
        self.assertEqual(cfg.thresholds.tau_B, 0.44)
        self.assertEqual(cfg.thresholds.tau_Q, 0.17)
        self.assertEqual(cfg.recovery_delay_frames, 5)
        self.assertEqual(cfg.recovery_max_attempts, 7)
        self.assertEqual(cfg.recovery_attempts_per_tick, 2)

    def test_cli_accepts_coupled_mode_and_tuning_knobs(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "train.py",
                "-s",
                td,
                "-m",
                str(Path(td) / "out"),
                "--risk_admission_mode",
                "on_the_fly_innovation_v1",
                "--paper_aligned_tau_R_low",
                "0.35",
                "--paper_aligned_recovery_delay_frames",
                "5",
                "--paper_aligned_tau_B",
                "0.41",
                "--paper_aligned_semantic_recovery_max_attempts",
                "6",
                "--paper_aligned_semantic_recovery_attempts_per_tick",
                "2",
            ]
            with patch.object(sys, "argv", argv):
                try:
                    args = get_args()
                except SystemExit as exc:
                    self.fail(f"coupled CLI should parse without SystemExit, got {exc}")

        self.assertEqual(args.risk_admission_mode, "on_the_fly_innovation_v1")
        self.assertEqual(args.paper_aligned_tau_R_low, 0.35)
        self.assertEqual(args.paper_aligned_tau_B, 0.41)
        self.assertEqual(args.paper_aligned_recovery_delay_frames, 5)
        self.assertEqual(args.paper_aligned_semantic_recovery_max_attempts, 6)
        self.assertEqual(args.paper_aligned_semantic_recovery_attempts_per_tick, 2)

    def test_runtime_gate_applies_coupled_preset_to_args_before_subsystems_use_them(self):
        args = _args(paper_aligned_tau_R_low=0.31, paper_aligned_recovery_delay_frames=4)

        gate = PaperAlignedRuntimeGate(args)

        self.assertEqual(getattr(gate, "requested_mode", None), "on_the_fly_innovation_v1")
        self.assertEqual(gate.mode, "paper_aligned_semantic_v1")
        self.assertEqual(gate.training_risk_mode, "paper_aligned_semantic_v1")
        self.assertEqual(args.risk_admission_mode, gate.training_risk_mode)
        self.assertIsNotNone(gate.semantic_policy)
        self.assertEqual(args.paper_aligned_defer_recovery_support_bridge, "v1")
        self.assertEqual(args.paper_aligned_recovery_commit_control, "recovery_commit_early_seed_v7")
        self.assertEqual(args.paper_aligned_direct_density_control, "target_band_v2_2_2_1")
        self.assertEqual(gate.semantic_policy.th.tau_R_low, 0.31)
        self.assertEqual(gate.semantic_policy.recovery_delay_frames, 4)

        _, action = gate.decide(
            1,
            {"image_name": "frame_001"},
            baseline_should_add=True,
            evidence={"num_matches": 300, "min_num_inliers_threshold": 100, "median_displacement": 0.8},
        )
        event = gate.trace_events[-1]
        self.assertIn(action, {"direct_admit", "defer_recoverable", "discard"})
        self.assertEqual(event["requested_mode"], "on_the_fly_innovation_v1")
        self.assertEqual(event["mode"], "paper_aligned_semantic_v1")
        self.assertIn("coupled_model", event["decision_meta"])
        for forbidden in ("PSNR", "SSIM", "LPIPS", "RPE", "APE"):
            self.assertNotIn(forbidden, event["decision_meta"])

    def test_trace_flush_records_coupled_contract_and_stage_metric_contract(self):
        with tempfile.TemporaryDirectory() as td:
            trace_path = Path(td) / "semantic_trace.json"
            args = _args(paper_aligned_contract_trace_path=str(trace_path))
            gate = PaperAlignedRuntimeGate(args)
            gate.decide(
                1,
                {"image_name": "frame_001"},
                baseline_should_add=False,
                evidence={"num_matches": 100, "min_num_inliers_threshold": 100, "recent_pose_fail_rate": 0.2},
            )

            gate.flush_trace()

            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("requested_mode"), "on_the_fly_innovation_v1")
            self.assertEqual(payload["mode"], "paper_aligned_semantic_v1")
            self.assertEqual(payload.get("coupled_innovation_config", {}).get("defer_recovery_support_bridge"), "v1")
            self.assertEqual(payload.get("stage_metric_contract", {}).get("online_quality_metric_fields"), [])
            self.assertIn("absolute_relative_rotation_error", payload.get("stage_metric_contract", {}).get("offline_stage_metric_fields", []))


if __name__ == "__main__":
    unittest.main()
