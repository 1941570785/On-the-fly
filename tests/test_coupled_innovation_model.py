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
from paper_aligned_policy.direct_density_control import DirectDensityController
from paper_aligned_policy.recovery_commit_control import RecoveryCommitController
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
        "paper_aligned_direct_density_upper_per_100": None,
        "paper_aligned_direct_density_hard_upper_per_100": None,
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
        self.assertEqual(cfg.recovery_commit_control, "off")
        self.assertEqual(cfg.direct_density_control, "off")
        self.assertEqual(cfg.direct_update_prev_desc_on_hold, "off")
        self.assertEqual(cfg.direct_density_upper_per_100, 75.0)
        self.assertEqual(cfg.direct_density_hard_upper_per_100, 90.0)
        self.assertEqual(cfg.thresholds.tau_R_low, 0.10)
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
                paper_aligned_direct_density_upper_per_100=70.0,
                paper_aligned_direct_density_hard_upper_per_100=80.0,
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
        self.assertEqual(cfg.direct_density_upper_per_100, 70.0)
        self.assertEqual(cfg.direct_density_hard_upper_per_100, 80.0)

    def test_coupled_preset_promotes_legacy_density_parser_defaults(self):
        args = _args(
            paper_aligned_direct_density_upper_per_100=45.0,
            paper_aligned_direct_density_hard_upper_per_100=50.0,
        )

        cfg = policy_config.resolve_coupled_innovation_config(args)

        self.assertEqual(cfg.direct_density_upper_per_100, 75.0)
        self.assertEqual(cfg.direct_density_hard_upper_per_100, 90.0)

        PaperAlignedRuntimeGate(args)

        self.assertEqual(args.paper_aligned_direct_density_upper_per_100, 75.0)
        self.assertEqual(args.paper_aligned_direct_density_hard_upper_per_100, 90.0)

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
                "--paper_aligned_direct_density_control",
                "pose_rep_value_decouple_v3",
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
        self.assertEqual(args.paper_aligned_direct_density_control, "pose_rep_value_decouple_v3")

    def test_cli_accepts_long_video_active_memory_direct_density_mode(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "train.py",
                "-s",
                td,
                "-m",
                str(Path(td) / "out"),
                "--paper_aligned_direct_density_control",
                "pose_rep_active_memory_v1",
            ]

            with patch.object(sys, "argv", argv):
                args = get_args()

        self.assertEqual(args.paper_aligned_direct_density_control, "pose_rep_active_memory_v1")

    def test_runtime_gate_applies_coupled_preset_to_args_before_subsystems_use_them(self):
        args = _args(paper_aligned_tau_R_low=0.31, paper_aligned_recovery_delay_frames=4)

        gate = PaperAlignedRuntimeGate(args)

        self.assertEqual(getattr(gate, "requested_mode", None), "on_the_fly_innovation_v1")
        self.assertEqual(gate.mode, "paper_aligned_semantic_v1")
        self.assertEqual(gate.training_risk_mode, "paper_aligned_semantic_v1")
        self.assertEqual(args.risk_admission_mode, gate.training_risk_mode)
        self.assertIsNotNone(gate.semantic_policy)
        self.assertEqual(args.paper_aligned_defer_recovery_support_bridge, "v1")
        self.assertEqual(args.paper_aligned_recovery_commit_control, "off")
        self.assertEqual(args.paper_aligned_direct_density_control, "off")
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

    def test_semantic_mode_preserves_explicit_runtime_tuning_knobs(self):
        args = _args(
            risk_admission_mode="paper_aligned_semantic_v1",
            paper_aligned_tau_R_low=0.27,
            paper_aligned_recovery_delay_frames=5,
            paper_aligned_semantic_recovery_max_attempts=4,
            paper_aligned_semantic_recovery_attempts_per_tick=0,
        )

        gate = PaperAlignedRuntimeGate(args)

        self.assertEqual(gate.mode, "paper_aligned_semantic_v1")
        self.assertIsNotNone(gate.semantic_policy)
        self.assertEqual(gate.semantic_policy.th.tau_R_low, 0.27)
        self.assertEqual(gate.semantic_policy.recovery_delay_frames, 5)
        self.assertEqual(gate.semantic_policy.recovery_max_attempts, 4)
        self.assertEqual(gate.semantic_policy.recovery_attempts_per_tick, 0)

    def test_recovery_commit_control_allows_strong_context_candidate_above_density_band(self):
        controller = RecoveryCommitController(
            _args(paper_aligned_recovery_commit_control="recovery_commit_early_seed_v7")
        )
        recovered = {
            "source_frame_id": 7,
            "source_input_index": 7,
            "scores": {"R_t": 0.0, "V_t": 0.63, "Q_t": 0.89},
            "source_payload": {"inlier_evidence": {"num_matches": 3106}},
        }
        context = {
            "current_tick_frame_id": 15,
            "source_num_inliers": 1670,
            "keyframe_density_per_100": 46.67,
            "source_gap_to_last_committed": 4,
            "predicted_gap_if_hold": 12,
            "keyframe_growth_recent": 7,
            "recent_materialization_rate": 0.0,
            "recent_pose_fail_rate": 0.0,
            "recent_runtime_attempt_count": 0,
            "recent_materialized_count": 0,
            "recent_failed_no_materialization_count": 0,
            "source_pose_fail_count": 0,
            "keyframes_since_last_pose_fail": 999,
            "source_already_committed": False,
            "is_surrogate": False,
            "is_contamination_risk": False,
            "anchor_count_available": False,
            "anchor_count_before": None,
        }

        decision = controller.decide(recovered, context)

        self.assertEqual(decision.action, "commit")
        self.assertIn("coverage", decision.reason)
        self.assertEqual(decision.debug["rescue_channel"], "coverage_rescue")

    def test_recovery_commit_control_holds_low_inlier_early_seed_candidates(self):
        controller = RecoveryCommitController(
            _args(paper_aligned_recovery_commit_control="recovery_commit_early_seed_v7")
        )
        recovered = {
            "source_frame_id": 270,
            "source_input_index": 270,
            "scores": {"R_t": 0.105, "V_t": 1.0, "Q_t": 0.9685},
            "source_payload": {"inlier_evidence": {"num_matches": 373}},
        }
        context = {
            "current_tick_frame_id": 278,
            "source_num_inliers": 8,
            "source_gap_to_last_committed": 10,
            "predicted_gap_if_hold": 18,
            "source_pose_fail_count": 0,
            "source_already_committed": False,
            "is_surrogate": False,
            "is_contamination_risk": False,
        }

        decision = controller.decide(recovered, context)

        self.assertEqual(decision.action, "hold")
        self.assertEqual(decision.reason, "v7_hold_low_seed_inliers")
        self.assertTrue(decision.debug["early_seed_low_inlier_guard"])

    def test_density_held_recovery_candidate_does_not_bypass_density_with_early_seed(self):
        controller = RecoveryCommitController(
            _args(paper_aligned_recovery_commit_control="recovery_commit_early_seed_v7")
        )
        recovered = {
            "source_frame_id": 160,
            "source_input_index": 160,
            "scores": {"R_t": 0.0, "V_t": 1.0, "Q_t": 1.0},
            "source_payload": {
                "inlier_evidence": {"num_matches": 1200},
                "density_hold_context": {"hold_reason": "at_or_above_upper_default_hold_gap_safe"},
            },
        }
        context = {
            "current_tick_frame_id": 168,
            "source_num_inliers": 900,
            "keyframe_density_per_100": 72.0,
            "source_gap_to_last_committed": 1,
            "predicted_gap_if_hold": 9,
            "keyframe_growth_recent": 30,
            "recent_materialization_rate": 1.0,
            "recent_pose_fail_rate": 0.0,
            "recent_runtime_attempt_count": 0,
            "recent_materialized_count": 0,
            "recent_failed_no_materialization_count": 0,
            "source_pose_fail_count": 0,
            "keyframes_since_last_pose_fail": 999,
            "source_already_committed": False,
            "is_surrogate": False,
            "is_contamination_risk": False,
            "anchor_count_available": False,
            "anchor_count_before": None,
        }

        decision = controller.decide(recovered, context)

        self.assertEqual(decision.action, "hold")
        self.assertTrue(decision.reason.startswith("v6_hold_"))
        self.assertFalse(decision.debug["can_be_early_seed_candidate"])
        self.assertTrue(decision.debug["early_seed_blocked_by_density_hold"])
        self.assertEqual(decision.debug["density_state"], "above_hard")

    def test_pose_rep_decouple_mode_holds_dense_redundant_direct_frame(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_rep_decouple_v1")
        )

        self.assertEqual(controller.prev_desc_update_on_hold, "light")
        self.assertLessEqual(controller.density_upper, 42.0)
        decision = controller.decide(
            frame_id=220,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=60.0,
            local_density_before=60.0,
            local_window_density=60.0,
            local_window_keyframes=60,
            local_window_gap_max=2.0,
            local_window_gap_after_if_hold=2.0,
            keyframe_growth_recent=30,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=1,
            main_chain_gap_before=1.0,
            main_chain_gap_after_if_hold=2.0,
            anchor_changed=False,
            support_triggered=True,
            median_displacement=31.0,
            displacement_threshold=30.0,
            num_matches=2400,
            min_num_inliers=100,
            pose_inliers=700,
            novelty_proxy=0.35,
            current_keyframe_count=132,
        )

        self.assertFalse(decision.finalize)
        self.assertEqual(decision.decision, "hold_density_high")
        self.assertTrue(
            controller.should_update_prev_desc_on_hold(decision.decision, decision.debug["density_state"])
        )
        self.assertFalse(controller.should_enqueue_hold_recovery())
        regular_controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="target_band_v2_2_2_1")
        )
        self.assertTrue(regular_controller.should_enqueue_hold_recovery())

    def test_pose_rep_decouple_mode_preserves_sparse_gap_frame(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_rep_decouple_v1")
        )

        decision = controller.decide(
            frame_id=220,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=18.0,
            local_density_before=18.0,
            local_window_density=18.0,
            local_window_keyframes=18,
            local_window_gap_max=15.0,
            local_window_gap_after_if_hold=16.0,
            keyframe_growth_recent=8,
            baseline_relative_density=0.7,
            source_gap_to_last_keyframe=16,
            main_chain_gap_before=15.0,
            main_chain_gap_after_if_hold=16.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=31.0,
            displacement_threshold=30.0,
            num_matches=2400,
            min_num_inliers=100,
            pose_inliers=700,
            novelty_proxy=0.35,
            current_keyframe_count=40,
        )

        self.assertTrue(decision.finalize)
        self.assertIn("gap", decision.reason)

    def test_direct_density_control_holds_redundant_baseline_direct_frames_above_band(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="target_band_v2_2_2_1")
        )

        decision = controller.decide(
            frame_id=200,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=92.0,
            local_density_before=92.0,
            local_window_density=92.0,
            local_window_keyframes=92,
            local_window_gap_max=2.0,
            local_window_gap_after_if_hold=2.0,
            keyframe_growth_recent=30,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=2,
            main_chain_gap_before=1.0,
            main_chain_gap_after_if_hold=2.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=31.0,
            displacement_threshold=30.0,
            num_matches=2400,
            min_num_inliers=100,
            pose_inliers=700,
            novelty_proxy=0.35,
            current_keyframe_count=184,
        )

        self.assertFalse(decision.finalize)
        self.assertEqual(decision.decision, "hold_density_high")
        self.assertIn("upper", decision.reason)

    def test_coupled_direct_density_finalizes_moderate_safe_gap_direct_candidate(self):
        args = _args(paper_aligned_direct_density_control="target_band_v2_2_2_1")
        gate = PaperAlignedRuntimeGate(args)
        controller = gate.direct_density_controller

        decision = controller.decide(
            frame_id=300,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=48.0,
            local_density_before=48.0,
            local_window_density=48.0,
            local_window_keyframes=48,
            local_window_gap_max=5.0,
            local_window_gap_after_if_hold=5.0,
            keyframe_growth_recent=30,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=2,
            main_chain_gap_before=2.0,
            main_chain_gap_after_if_hold=3.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=31.0,
            displacement_threshold=30.0,
            num_matches=2400,
            min_num_inliers=100,
            pose_inliers=700,
            novelty_proxy=0.35,
            current_keyframe_count=144,
        )

        self.assertTrue(decision.finalize)
        self.assertEqual(decision.decision, "finalize_baseline_skeleton_reserve")
        self.assertEqual(decision.reason, "baseline_skeleton_reserve")

    def test_density_held_direct_candidate_enters_recovery_pool(self):
        gate = PaperAlignedRuntimeGate(_args())
        info = {
            "image_name": "252.jpg",
            "image_path": "/tmp/252.jpg",
            "is_test": False,
            "_inlier_evidence": {"num_matches": 2400},
            "_local_context": {"phase": "incremental"},
        }
        evidence = {
            "median_displacement": 31.0,
            "displacement_threshold": 30.0,
            "num_matches": 2400,
            "min_num_inliers_threshold": 100,
            "is_test": False,
            "recent_pose_fail_rate": 0.0,
        }

        admit, action = gate.decide(
            252,
            info,
            baseline_should_add=True,
            phase="incremental",
            evidence=evidence,
        )

        self.assertTrue(admit)
        self.assertEqual(action, "direct_admit")
        self.assertEqual(len(gate.semantic_policy.recovery_pool), 0)

        enqueued = gate.enqueue_density_hold_recovery_candidate(
            frame_id=252,
            info=info,
            evidence=evidence,
            hold_decision="hold_density_high",
            hold_reason="at_or_above_upper_default_hold_gap_safe",
            density_debug={"density_before": 53.17, "local_window_density": 46.0},
        )

        event = gate.trace_events[-1]
        self.assertTrue(enqueued)
        self.assertEqual(len(gate.semantic_policy.recovery_pool), 1)
        self.assertEqual(gate.semantic_policy.recovery_pool[0]["frame_id"], 252)
        self.assertEqual(
            gate.semantic_policy.recovery_pool[0]["defer_reason"],
            "direct_density_hold",
        )
        self.assertTrue(event["density_hold_recovery_enqueued"])
        self.assertEqual(
            event["decision_meta"]["density_hold_recovery_bridge_tag"],
            "density_hold_recoverable",
        )
        self.assertTrue(gate.is_recovery_pose_path_candidate(252))

        admit, action = gate.decide(
            253,
            {**info, "image_name": "253.jpg"},
            baseline_should_add=True,
            phase="incremental",
            evidence=evidence,
        )
        self.assertTrue(admit)
        self.assertEqual(action, "direct_admit")
        self.assertFalse(gate.is_recovery_pose_path_candidate(253))

    def test_direct_density_preserves_baseline_skeleton_when_gap_budget_is_exhausted(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="target_band_v2_2_2_1")
        )
        controller._sync_budget_window(248)
        controller._budget.gap_rescue_used = controller.gap_rescue_budget_per_100

        decision = controller.decide(
            frame_id=248,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=45.2,
            local_density_before=45.0,
            local_window_density=45.0,
            local_window_keyframes=45,
            local_window_gap_max=10.0,
            local_window_gap_after_if_hold=1.0,
            keyframe_growth_recent=30,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=1,
            main_chain_gap_before=3.0,
            main_chain_gap_after_if_hold=4.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=31.0,
            displacement_threshold=30.0,
            num_matches=2400,
            min_num_inliers=100,
            pose_inliers=700,
            novelty_proxy=0.35,
            current_keyframe_count=112,
        )

        self.assertTrue(decision.finalize)
        self.assertEqual(decision.decision, "finalize_baseline_skeleton_reserve")
        self.assertEqual(decision.reason, "baseline_skeleton_reserve")

    def test_direct_density_control_keeps_gap_critical_baseline_direct_frames(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="target_band_v2_2_2_1")
        )

        decision = controller.decide(
            frame_id=200,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=62.0,
            local_density_before=62.0,
            local_window_density=62.0,
            local_window_keyframes=62,
            local_window_gap_max=19.0,
            local_window_gap_after_if_hold=21.0,
            keyframe_growth_recent=30,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=21,
            main_chain_gap_before=19.0,
            main_chain_gap_after_if_hold=21.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=31.0,
            displacement_threshold=30.0,
            num_matches=2400,
            min_num_inliers=100,
            pose_inliers=700,
            novelty_proxy=0.35,
            current_keyframe_count=124,
        )

        self.assertTrue(decision.finalize)
        self.assertIn("gap", decision.reason)

    def test_direct_density_control_holds_noncritical_anchor_boundary_in_dense_chain(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="target_band_v2_2_2_1")
        )

        decision = controller.decide(
            frame_id=200,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=62.0,
            local_density_before=62.0,
            local_window_density=62.0,
            local_window_keyframes=62,
            local_window_gap_max=2.0,
            local_window_gap_after_if_hold=2.0,
            keyframe_growth_recent=30,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=2,
            main_chain_gap_before=1.0,
            main_chain_gap_after_if_hold=2.0,
            anchor_changed=True,
            support_triggered=False,
            median_displacement=31.0,
            displacement_threshold=30.0,
            num_matches=2400,
            min_num_inliers=100,
            pose_inliers=700,
            novelty_proxy=0.35,
            current_keyframe_count=124,
        )

        self.assertFalse(decision.finalize)
        self.assertEqual(decision.decision, "hold_density_high")
        self.assertIn("anchor_boundary", decision.reason)

    def test_direct_density_control_holds_high_novelty_anchor_boundary_for_recovery_context(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="target_band_v2_2_2_1")
        )

        decision = controller.decide(
            frame_id=200,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=80.0,
            local_density_before=80.0,
            local_window_density=80.0,
            local_window_keyframes=80,
            local_window_gap_max=2.0,
            local_window_gap_after_if_hold=2.0,
            keyframe_growth_recent=30,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=1,
            main_chain_gap_before=1.0,
            main_chain_gap_after_if_hold=1.0,
            anchor_changed=True,
            support_triggered=False,
            median_displacement=60.0,
            displacement_threshold=30.0,
            num_matches=2600,
            min_num_inliers=100,
            pose_inliers=1600,
            novelty_proxy=0.95,
            current_keyframe_count=160,
        )

        self.assertFalse(decision.finalize)
        self.assertEqual(decision.decision, "hold_density_high")
        self.assertIn("anchor_boundary", decision.reason)

    def test_value_aware_pose_rep_decouple_holds_low_representation_value_pose_reference(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_rep_value_decouple_v2")
        )

        decision = controller.decide(
            frame_id=420,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=46.0,
            local_density_before=44.0,
            local_window_density=44.0,
            local_window_keyframes=44,
            local_window_gap_max=3.0,
            local_window_gap_after_if_hold=3.0,
            keyframe_growth_recent=18,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=2,
            main_chain_gap_before=2.0,
            main_chain_gap_after_if_hold=3.0,
            anchor_changed=False,
            support_triggered=True,
            median_displacement=21.0,
            displacement_threshold=30.0,
            num_matches=150,
            min_num_inliers=100,
            pose_inliers=170,
            novelty_proxy=0.22,
            current_keyframe_count=193,
            semantic_scores={"R_t": 0.12, "V_t": 0.20, "Q_t": 0.82, "C_t": 0.35, "B_R_t": 1.0},
        )

        self.assertFalse(decision.finalize)
        self.assertEqual(decision.decision, "hold_low_representation_value")
        self.assertIn("pose_reference_only", decision.reason)
        self.assertLess(decision.debug["representation_value_score"], 0.42)
        self.assertGreaterEqual(decision.debug["pose_reference_value_score"], 0.55)
        self.assertTrue(decision.debug["value_hold_allowed"])
        self.assertTrue(
            controller.should_update_prev_desc_on_hold(decision.decision, decision.debug["density_state"])
        )
        self.assertFalse(controller.should_enqueue_hold_recovery())

    def test_value_aware_pose_rep_decouple_finalizes_high_representation_value(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_rep_value_decouple_v2")
        )

        decision = controller.decide(
            frame_id=420,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=46.0,
            local_density_before=44.0,
            local_window_density=44.0,
            local_window_keyframes=44,
            local_window_gap_max=3.0,
            local_window_gap_after_if_hold=3.0,
            keyframe_growth_recent=18,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=5,
            main_chain_gap_before=4.0,
            main_chain_gap_after_if_hold=5.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=48.0,
            displacement_threshold=30.0,
            num_matches=520,
            min_num_inliers=100,
            pose_inliers=220,
            novelty_proxy=0.76,
            current_keyframe_count=193,
            semantic_scores={"R_t": 0.10, "V_t": 0.78, "Q_t": 0.86, "C_t": 0.68, "B_R_t": 1.0},
        )

        self.assertTrue(decision.finalize)
        self.assertEqual(decision.decision, "finalize_high_representation_value")
        self.assertGreaterEqual(decision.debug["representation_value_score"], 0.42)
        self.assertFalse(decision.debug["value_hold_allowed"])
        self.assertEqual(decision.debug["value_hold_block_reason"], "representation_value_high")

    def test_value_aware_pose_rep_decouple_finalizes_high_risk_pose_even_if_representation_is_low(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_rep_value_decouple_v2")
        )

        decision = controller.decide(
            frame_id=420,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=46.0,
            local_density_before=44.0,
            local_window_density=44.0,
            local_window_keyframes=44,
            local_window_gap_max=3.0,
            local_window_gap_after_if_hold=3.0,
            keyframe_growth_recent=18,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=5,
            main_chain_gap_before=4.0,
            main_chain_gap_after_if_hold=5.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=18.0,
            displacement_threshold=30.0,
            num_matches=150,
            min_num_inliers=100,
            pose_inliers=115,
            novelty_proxy=0.20,
            current_keyframe_count=193,
            semantic_scores={"R_t": 0.66, "V_t": 0.20, "Q_t": 0.36, "C_t": 0.35, "B_R_t": 0.45},
        )

        self.assertTrue(decision.finalize)
        self.assertEqual(decision.decision, "finalize_pose_risk_reference")
        self.assertEqual(decision.debug["value_hold_block_reason"], "pose_risk_high")

    def test_value_risk_decouple_v3_preserves_small_scene_high_growth_representation(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_rep_value_decouple_v3")
        )

        decision = controller.decide(
            frame_id=220,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=91.0,
            local_density_before=89.0,
            local_window_density=89.0,
            local_window_keyframes=89,
            local_window_gap_max=2.0,
            local_window_gap_after_if_hold=2.0,
            keyframe_growth_recent=42,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=1,
            main_chain_gap_before=1.0,
            main_chain_gap_after_if_hold=2.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=60.0,
            displacement_threshold=30.0,
            num_matches=2400,
            min_num_inliers=100,
            pose_inliers=1600,
            novelty_proxy=0.0,
            current_keyframe_count=205,
            semantic_scores={"R_t": 0.0, "V_t": 0.99, "Q_t": 0.997, "C_t": 1.0, "B_R_t": 1.0},
        )

        self.assertTrue(decision.finalize)
        self.assertNotEqual(decision.decision, "hold_low_representation_value")
        self.assertEqual(decision.debug["value_hold_block_reason"], "high_recent_growth_representation_guard")
        self.assertTrue(decision.debug["high_recent_growth_representation_guard"])

    def test_value_risk_decouple_v3_allows_long_stream_low_growth_pose_only_hold(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_rep_value_decouple_v3")
        )

        decision = controller.decide(
            frame_id=420,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=91.0,
            local_density_before=91.0,
            local_window_density=91.0,
            local_window_keyframes=91,
            local_window_gap_max=2.0,
            local_window_gap_after_if_hold=2.0,
            keyframe_growth_recent=-1,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=1,
            main_chain_gap_before=1.0,
            main_chain_gap_after_if_hold=2.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=60.0,
            displacement_threshold=30.0,
            num_matches=2400,
            min_num_inliers=100,
            pose_inliers=1600,
            novelty_proxy=0.0,
            current_keyframe_count=380,
            semantic_scores={"R_t": 0.0, "V_t": 0.99, "Q_t": 0.997, "C_t": 1.0, "B_R_t": 1.0},
        )

        self.assertFalse(decision.finalize)
        self.assertEqual(decision.decision, "hold_low_representation_value")
        self.assertTrue(decision.debug["long_stream_low_growth_context"])
        self.assertGreaterEqual(decision.debug["value_hold_budget_per_100"], 40)

    def test_value_risk_decouple_v3_waits_for_long_sequence_maturity(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_rep_value_decouple_v3")
        )

        decision = controller.decide(
            frame_id=240,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=86.2,
            local_density_before=69.0,
            local_window_density=69.0,
            local_window_keyframes=69,
            local_window_gap_max=2.0,
            local_window_gap_after_if_hold=2.0,
            keyframe_growth_recent=-31,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=1,
            main_chain_gap_before=1.0,
            main_chain_gap_after_if_hold=1.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=60.0,
            displacement_threshold=30.0,
            num_matches=2400,
            min_num_inliers=100,
            pose_inliers=1600,
            novelty_proxy=0.0,
            current_keyframe_count=207,
            semantic_scores={"R_t": 0.0, "V_t": 0.99, "Q_t": 0.997, "C_t": 1.0, "B_R_t": 1.0},
        )

        self.assertTrue(decision.finalize)
        self.assertNotEqual(decision.decision, "hold_low_representation_value")
        self.assertFalse(decision.debug["long_stream_low_growth_context"])
        self.assertTrue(decision.debug["long_sequence_maturity_guard"])
        self.assertEqual(decision.debug["value_hold_block_reason"], "long_sequence_maturity_guard")

    def test_value_risk_decouple_v3_does_not_hold_from_density_alone(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_rep_value_decouple_v3")
        )

        decision = controller.decide(
            frame_id=420,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=96.0,
            local_density_before=96.0,
            local_window_density=96.0,
            local_window_keyframes=96,
            local_window_gap_max=2.0,
            local_window_gap_after_if_hold=2.0,
            keyframe_growth_recent=30,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=1,
            main_chain_gap_before=1.0,
            main_chain_gap_after_if_hold=2.0,
            anchor_changed=True,
            support_triggered=False,
            median_displacement=60.0,
            displacement_threshold=30.0,
            num_matches=2400,
            min_num_inliers=100,
            pose_inliers=1600,
            novelty_proxy=0.95,
            current_keyframe_count=390,
            semantic_scores={"R_t": 0.0, "V_t": 0.99, "Q_t": 0.997, "C_t": 1.0, "B_R_t": 1.0},
        )

        self.assertTrue(decision.finalize)
        self.assertFalse(decision.debug["hold_density_high"])
        self.assertEqual(decision.decision, "finalize_high_representation_value")

    def test_active_memory_v1_holds_low_parallax_redundant_tum_frame(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v1")
        )

        decision = controller.decide(
            frame_id=900,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=72.0,
            local_density_before=72.0,
            local_window_density=72.0,
            local_window_keyframes=72,
            local_window_gap_max=2.0,
            local_window_gap_after_if_hold=2.0,
            keyframe_growth_recent=28,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=1,
            main_chain_gap_before=1.0,
            main_chain_gap_after_if_hold=2.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=18.0,
            displacement_threshold=30.0,
            num_matches=2600,
            min_num_inliers=100,
            pose_inliers=1800,
            novelty_proxy=0.08,
            current_keyframe_count=650,
            semantic_scores={
                "R_t": 0.05,
                "V_t": 0.92,
                "Q_t": 0.96,
                "C_t": 0.96,
                "B_R_t": 0.95,
            },
        )

        self.assertFalse(decision.finalize)
        self.assertEqual(decision.decision, "hold_low_representation_value")
        self.assertEqual(decision.reason, "pose_reference_only_low_representation_value")
        self.assertEqual(decision.debug["active_memory_frame_role"], "tracking_only")
        self.assertTrue(decision.debug["active_memory_context"])
        self.assertLess(decision.debug["active_memory_marginal_value"], 0.35)

    def test_active_memory_v1_holds_high_motion_low_marginal_representation_frame(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v1")
        )

        decision = controller.decide(
            frame_id=520,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=68.0,
            local_density_before=88.0,
            local_window_density=88.0,
            local_window_keyframes=88,
            local_window_gap_max=2.0,
            local_window_gap_after_if_hold=2.0,
            keyframe_growth_recent=42,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=1,
            main_chain_gap_before=1.0,
            main_chain_gap_after_if_hold=2.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=60.0,
            displacement_threshold=30.0,
            num_matches=2600,
            min_num_inliers=100,
            pose_inliers=1800,
            novelty_proxy=0.0,
            current_keyframe_count=350,
            semantic_scores={
                "R_t": 0.02,
                "V_t": 0.96,
                "Q_t": 0.997,
                "C_t": 0.99,
                "B_R_t": 1.0,
            },
        )

        self.assertFalse(decision.finalize)
        self.assertEqual(decision.decision, "hold_low_representation_value")
        self.assertEqual(decision.debug["active_memory_frame_role"], "tracking_only")
        self.assertTrue(decision.debug["active_memory_context"])
        self.assertTrue(decision.debug["active_memory_low_marginal_representation"])
        self.assertFalse(decision.debug["active_memory_low_parallax"])

    def test_active_memory_v1_trusts_composite_marginal_value_over_raw_novelty_proxy(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v1")
        )

        decision = controller.decide(
            frame_id=540,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=72.0,
            local_density_before=92.0,
            local_window_density=92.0,
            local_window_keyframes=92,
            local_window_gap_max=2.0,
            local_window_gap_after_if_hold=2.0,
            keyframe_growth_recent=36,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=1,
            main_chain_gap_before=1.0,
            main_chain_gap_after_if_hold=2.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=60.0,
            displacement_threshold=30.0,
            num_matches=2600,
            min_num_inliers=100,
            pose_inliers=1800,
            novelty_proxy=0.62,
            current_keyframe_count=390,
            semantic_scores={
                "R_t": 0.01,
                "V_t": 0.96,
                "Q_t": 0.997,
                "C_t": 0.99,
                "B_R_t": 1.0,
            },
        )

        self.assertFalse(decision.finalize)
        self.assertEqual(decision.decision, "hold_low_representation_value")
        self.assertLess(decision.debug["representation_value_score"], 0.38)
        self.assertGreater(decision.debug["novelty_value_score"], 0.25)
        self.assertTrue(decision.debug["active_memory_low_representation_value"])

    def test_active_memory_v1_keeps_high_novelty_gap_frame_as_representation(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v1")
        )

        decision = controller.decide(
            frame_id=900,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=72.0,
            local_density_before=72.0,
            local_window_density=72.0,
            local_window_keyframes=72,
            local_window_gap_max=2.0,
            local_window_gap_after_if_hold=8.0,
            keyframe_growth_recent=28,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=6,
            main_chain_gap_before=4.0,
            main_chain_gap_after_if_hold=8.0,
            anchor_changed=False,
            support_triggered=True,
            median_displacement=62.0,
            displacement_threshold=30.0,
            num_matches=1400,
            min_num_inliers=100,
            pose_inliers=800,
            novelty_proxy=0.78,
            current_keyframe_count=650,
            semantic_scores={
                "R_t": 0.18,
                "V_t": 0.88,
                "Q_t": 0.82,
                "C_t": 0.72,
                "B_R_t": 0.80,
            },
        )

        self.assertTrue(decision.finalize)
        self.assertEqual(decision.debug["active_memory_frame_role"], "representation")
        self.assertFalse(decision.debug["active_memory_context"])
        self.assertGreaterEqual(decision.debug["active_memory_marginal_value"], 0.35)

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
