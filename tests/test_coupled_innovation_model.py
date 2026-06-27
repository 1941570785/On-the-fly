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
from poses.feature_detector import DescribedKeypoints
from poses.pose_initializer import PoseInitializer


def _pose_pool_desc(num_keypoints=768, support_count=512, match_score=220):
    import torch

    desc = DescribedKeypoints(
        torch.zeros(num_keypoints, 2),
        torch.ones(num_keypoints, 8),
    )
    idx = torch.arange(min(support_count, num_keypoints))
    desc.update_3D_pts(
        torch.ones(len(idx), 3),
        torch.ones(len(idx)),
        torch.ones(len(idx)),
        idx,
    )
    desc._test_match_score = match_score
    return desc


class _PosePoolMatcher:
    def evaluate_match(self, ref_desc, _curr_desc):
        return float(getattr(ref_desc, "_test_match_score", 0.0))


def _args(**overrides):
    base = {
        "risk_admission_mode": "on_the_fly_innovation_v1",
        "paper_aligned_contract_trace_path": "",
        "paper_aligned_recovery_commit_bridge": None,
        "paper_aligned_defer_recovery_support_bridge": None,
        "paper_aligned_recovery_commit_control": None,
        "paper_aligned_direct_density_control": None,
        "paper_aligned_pose_memory_geometry_context": None,
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

    def test_cli_accepts_active_memory_v2_direct_density_mode(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "train.py",
                "-s",
                td,
                "-m",
                str(Path(td) / "out"),
                "--paper_aligned_direct_density_control",
                "pose_rep_active_memory_v2",
            ]

            with patch.object(sys, "argv", argv):
                args = get_args()

        self.assertEqual(args.paper_aligned_direct_density_control, "pose_rep_active_memory_v2")

    def test_cli_accepts_active_memory_v4_direct_density_mode(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "train.py",
                "-s",
                td,
                "-m",
                str(Path(td) / "out"),
                "--paper_aligned_direct_density_control",
                "pose_rep_active_memory_v4",
            ]

            with patch.object(sys, "argv", argv):
                args = get_args()

        self.assertEqual(args.paper_aligned_direct_density_control, "pose_rep_active_memory_v4")

    def test_cli_accepts_active_memory_v6_direct_density_mode(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "train.py",
                "-s",
                td,
                "-m",
                str(Path(td) / "out"),
                "--paper_aligned_direct_density_control",
                "pose_rep_active_memory_v6",
            ]

            with patch.object(sys, "argv", argv):
                args = get_args()

        self.assertEqual(args.paper_aligned_direct_density_control, "pose_rep_active_memory_v6")

    def test_cli_accepts_active_memory_v8_direct_density_mode(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "train.py",
                "-s",
                td,
                "-m",
                str(Path(td) / "out"),
                "--paper_aligned_direct_density_control",
                "pose_rep_active_memory_v8",
            ]

            with patch.object(sys, "argv", argv):
                args = get_args()

        self.assertEqual(args.paper_aligned_direct_density_control, "pose_rep_active_memory_v8")

    def test_cli_accepts_active_memory_v9_direct_density_mode(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "train.py",
                "-s",
                td,
                "-m",
                str(Path(td) / "out"),
                "--paper_aligned_direct_density_control",
                "pose_rep_active_memory_v9",
            ]

            with patch.object(sys, "argv", argv):
                args = get_args()

        self.assertEqual(args.paper_aligned_direct_density_control, "pose_rep_active_memory_v9")

    def test_cli_accepts_active_memory_v24_direct_density_mode(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "train.py",
                "-s",
                td,
                "-m",
                str(Path(td) / "out"),
                "--paper_aligned_direct_density_control",
                "pose_rep_active_memory_v24",
            ]

            with patch.object(sys, "argv", argv):
                args = get_args()

        self.assertEqual(args.paper_aligned_direct_density_control, "pose_rep_active_memory_v24")

    def test_cli_accepts_active_memory_v25_direct_density_mode(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "train.py",
                "-s",
                td,
                "-m",
                str(Path(td) / "out"),
                "--paper_aligned_direct_density_control",
                "pose_rep_active_memory_v25",
            ]

            with patch.object(sys, "argv", argv):
                args = get_args()

        self.assertEqual(args.paper_aligned_direct_density_control, "pose_rep_active_memory_v25")

    def test_cli_accepts_active_memory_v26_direct_density_mode(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "train.py",
                "-s",
                td,
                "-m",
                str(Path(td) / "out"),
                "--paper_aligned_direct_density_control",
                "pose_rep_active_memory_v26",
            ]

            with patch.object(sys, "argv", argv):
                args = get_args()

        self.assertEqual(args.paper_aligned_direct_density_control, "pose_rep_active_memory_v26")

    def test_cli_accepts_pose_memory_geometry_context_knob(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "train.py",
                "-s",
                td,
                "-m",
                str(Path(td) / "out"),
                "--paper_aligned_direct_density_control",
                "pose_rep_active_memory_v33",
                "--paper_aligned_pose_memory_geometry_context",
                "v1",
            ]

            with patch.object(sys, "argv", argv):
                args = get_args()

        self.assertEqual(args.paper_aligned_pose_memory_geometry_context, "v1")

    def test_cli_accepts_streaming_memory_controller_mode(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "train.py",
                "-s",
                td,
                "-m",
                str(Path(td) / "out"),
                "--paper_aligned_direct_density_control",
                "pose_rep_streaming_memory_v1",
            ]

            with patch.object(sys, "argv", argv):
                args = get_args()

        self.assertEqual(
            args.paper_aligned_direct_density_control,
            "pose_rep_streaming_memory_v1",
        )

    def test_cli_accepts_pose_safe_streaming_memory_mode(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "train.py",
                "-s",
                td,
                "-m",
                str(Path(td) / "out"),
                "--paper_aligned_direct_density_control",
                "pose_safe_streaming_memory_v1",
                "--paper_aligned_pose_memory_geometry_context",
                "v1",
            ]

            with patch.object(sys, "argv", argv):
                args = get_args()

        self.assertEqual(
            args.paper_aligned_direct_density_control,
            "pose_safe_streaming_memory_v1",
        )
        self.assertEqual(args.paper_aligned_pose_memory_geometry_context, "v1")

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

    def test_active_memory_v1_allows_tracking_support_without_forcing_representation(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v1")
        )

        decision = controller.decide(
            frame_id=560,
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
            support_triggered=True,
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
        self.assertEqual(decision.debug["active_memory_frame_role"], "tracking_only")
        self.assertGreater(decision.debug["support_needed_score"], 0.0)

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

    def test_active_memory_v2_holds_low_turn_dense_context_frame(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v2")
        )

        decision = controller.decide(
            frame_id=392,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=75.5,
            local_density_before=44.0,
            local_window_density=44.0,
            local_window_keyframes=44,
            local_window_gap_max=3.0,
            local_window_gap_after_if_hold=3.0,
            keyframe_growth_recent=36,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=2,
            main_chain_gap_before=2.0,
            main_chain_gap_after_if_hold=3.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=18.0,
            displacement_threshold=30.0,
            num_matches=2600,
            min_num_inliers=100,
            pose_inliers=1800,
            novelty_proxy=0.08,
            current_keyframe_count=296,
            semantic_scores={
                "R_t": 0.0,
                "V_t": 0.92,
                "Q_t": 0.997,
                "C_t": 1.0,
                "B_R_t": 1.0,
            },
            viewpoint_scores={
                "viewpoint_rotation_deg_window_max": 2.0,
                "inlier_grid_coverage": 1.0,
            },
        )

        self.assertFalse(decision.finalize)
        self.assertEqual(decision.decision, "hold_low_representation_value")
        self.assertTrue(decision.debug["active_memory_low_turn_dense_context"])
        self.assertTrue(decision.debug["active_memory_context"])
        self.assertEqual(decision.debug["active_memory_frame_role"], "tracking_only")

    def test_pose_only_reference_pool_registers_and_selects_tracking_only_frames(self):
        import torch

        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v1")
        )
        curr_desc = _pose_pool_desc(match_score=0)
        rt = torch.eye(4)

        rejected = gate.register_pose_only_reference(
            frame_id=310,
            info={"is_test": False, "image_name": "rejected"},
            desc_kpts=_pose_pool_desc(support_count=600, match_score=500),
            Rt=rt,
            density_debug={
                "active_memory_context": False,
                "active_memory_frame_role": "representation",
            },
            pose_debug={"num_pnp_inliers": 100, "num_miniba_inliers": 100},
        )
        accepted = gate.register_pose_only_reference(
            frame_id=320,
            info={"is_test": False, "image_name": "accepted"},
            desc_kpts=_pose_pool_desc(support_count=600, match_score=500),
            Rt=rt,
            density_debug={
                "active_memory_context": True,
                "active_memory_frame_role": "tracking_only",
            },
            pose_debug={"num_pnp_inliers": 100, "num_miniba_inliers": 100},
        )

        immediate = gate.select_pose_only_references(
            frame_id=321,
            curr_desc_kpts=curr_desc,
            matcher=_PosePoolMatcher(),
        )
        selected = gate.select_pose_only_references(
            frame_id=330,
            curr_desc_kpts=curr_desc,
            matcher=_PosePoolMatcher(),
        )

        self.assertFalse(rejected)
        self.assertTrue(accepted)
        self.assertEqual(immediate, [])
        self.assertEqual(len(selected), 1)
        self.assertGreaterEqual(selected[0].index, 0)
        self.assertEqual(selected[0].info["_paper_aligned_source_frame_id"], 320)
        self.assertEqual(selected[0].info["_paper_aligned_commit_origin"], "pose_only_reference")
        self.assertEqual(gate.pose_only_reference_pool_summary()["pool_size"], 1)

    def test_pose_only_reference_pool_v2_uses_register_and_selection_cooldown(self):
        import torch

        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v2")
        )
        rt = torch.eye(4)
        density_debug = {
            "active_memory_context": True,
            "active_memory_frame_role": "tracking_only",
        }

        first = gate.register_pose_only_reference(
            frame_id=320,
            info={"is_test": False, "image_name": "first"},
            desc_kpts=_pose_pool_desc(support_count=700, match_score=480),
            Rt=rt,
            density_debug=density_debug,
            pose_debug={"num_pnp_inliers": 300, "num_miniba_inliers": 300},
        )
        cooldown = gate.register_pose_only_reference(
            frame_id=325,
            info={"is_test": False, "image_name": "cooldown"},
            desc_kpts=_pose_pool_desc(support_count=700, match_score=520),
            Rt=rt,
            density_debug=density_debug,
            pose_debug={"num_pnp_inliers": 300, "num_miniba_inliers": 300},
        )
        second = gate.register_pose_only_reference(
            frame_id=333,
            info={"is_test": False, "image_name": "second"},
            desc_kpts=_pose_pool_desc(support_count=700, match_score=520),
            Rt=rt,
            density_debug=density_debug,
            pose_debug={"num_pnp_inliers": 300, "num_miniba_inliers": 300},
        )

        selected_first = gate.select_pose_only_references(
            frame_id=352,
            curr_desc_kpts=_pose_pool_desc(match_score=0),
            matcher=_PosePoolMatcher(),
        )
        selected_second = gate.select_pose_only_references(
            frame_id=353,
            curr_desc_kpts=_pose_pool_desc(match_score=0),
            matcher=_PosePoolMatcher(),
        )

        self.assertTrue(first)
        self.assertFalse(cooldown)
        self.assertTrue(second)
        self.assertEqual(
            gate.pose_reference_pool_events[1]["block_reason"],
            "pose_only_register_cooldown",
        )
        self.assertEqual(
            [ref.info["_paper_aligned_source_frame_id"] for ref in selected_first],
            [333],
        )
        self.assertEqual(
            [ref.info["_paper_aligned_source_frame_id"] for ref in selected_second],
            [320],
        )
        summary = gate.pose_only_reference_pool_summary()
        self.assertEqual(summary["min_age_frames"], 18)
        self.assertEqual(summary["register_min_interval_frames"], 12)
        self.assertEqual(summary["selection_cooldown_frames"], 8)

    def test_pose_only_reference_pool_v4_gates_repetitive_low_diversity_refs(self):
        import torch

        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v4")
        )
        density_debug = {
            "active_memory_context": True,
            "active_memory_frame_role": "tracking_only",
            "support_concentration": 0.04,
            "new_view_event_score": 0.14,
            "anchor_health_score": 0.60,
            "keyframe_growth_recent": -10,
        }

        accepted = gate.register_pose_only_reference(
            frame_id=320,
            info={"is_test": False, "image_name": "forest_like"},
            desc_kpts=_pose_pool_desc(support_count=700, match_score=480),
            Rt=torch.eye(4),
            density_debug=density_debug,
            pose_debug={"num_pnp_inliers": 300, "num_miniba_inliers": 300},
        )

        self.assertFalse(accepted)
        self.assertEqual(
            gate.pose_reference_pool_events[-1]["block_reason"],
            "pose_only_low_support_diversity",
        )

    def test_pose_only_reference_pool_v4_accepts_growth_stall_diverse_refs(self):
        import torch

        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v4")
        )
        density_debug = {
            "active_memory_context": True,
            "active_memory_frame_role": "tracking_only",
            "support_concentration": 0.09,
            "new_view_event_score": 0.15,
            "anchor_health_score": 0.60,
            "keyframe_growth_recent": -8,
        }

        accepted = gate.register_pose_only_reference(
            frame_id=320,
            info={"is_test": False, "image_name": "university_like"},
            desc_kpts=_pose_pool_desc(support_count=700, match_score=480),
            Rt=torch.eye(4),
            density_debug=density_debug,
            pose_debug={"num_pnp_inliers": 300, "num_miniba_inliers": 300},
        )

        self.assertTrue(accepted)
        self.assertEqual(gate.pose_only_reference_pool_summary()["pool_size"], 1)

    def test_pose_only_reference_pool_v4_blocks_healthy_anchor_high_new_view_refs(self):
        import torch

        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v4")
        )
        density_debug = {
            "active_memory_context": True,
            "active_memory_frame_role": "tracking_only",
            "support_concentration": 0.14,
            "new_view_event_score": 0.30,
            "anchor_health_score": 0.74,
            "keyframe_growth_recent": -6,
        }

        accepted = gate.register_pose_only_reference(
            frame_id=320,
            info={"is_test": False, "image_name": "desk1_like"},
            desc_kpts=_pose_pool_desc(support_count=700, match_score=480),
            Rt=torch.eye(4),
            density_debug=density_debug,
            pose_debug={"num_pnp_inliers": 300, "num_miniba_inliers": 300},
        )

        self.assertFalse(accepted)
        self.assertEqual(
            gate.pose_reference_pool_events[-1]["block_reason"],
            "pose_only_anchor_healthy_high_new_view",
        )

    def test_pose_only_reference_pool_v5_blocks_borderline_low_diversity_refs(self):
        import torch

        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v5")
        )
        accepted = gate.register_pose_only_reference(
            frame_id=320,
            info={"is_test": False, "image_name": "forest_borderline"},
            desc_kpts=_pose_pool_desc(support_count=700, match_score=480),
            Rt=torch.eye(4),
            density_debug={
                "active_memory_context": True,
                "active_memory_frame_role": "tracking_only",
                "support_concentration": 0.075,
                "new_view_event_score": 0.14,
                "anchor_health_score": 0.60,
                "keyframe_growth_recent": -10,
            },
            pose_debug={"num_pnp_inliers": 300, "num_miniba_inliers": 300},
        )

        self.assertFalse(accepted)
        self.assertEqual(
            gate.pose_reference_pool_events[-1]["block_reason"],
            "pose_only_low_support_diversity",
        )

    def test_pose_only_reference_pool_v5_accepts_strong_low_new_view_refs(self):
        import torch

        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v5")
        )
        accepted = gate.register_pose_only_reference(
            frame_id=320,
            info={"is_test": False, "image_name": "university_strong"},
            desc_kpts=_pose_pool_desc(support_count=700, match_score=480),
            Rt=torch.eye(4),
            density_debug={
                "active_memory_context": True,
                "active_memory_frame_role": "tracking_only",
                "support_concentration": 0.09,
                "new_view_event_score": 0.15,
                "anchor_health_score": 0.60,
                "keyframe_growth_recent": -8,
            },
            pose_debug={"num_pnp_inliers": 300, "num_miniba_inliers": 300},
        )

        self.assertTrue(accepted)
        summary = gate.pose_only_reference_pool_summary()
        self.assertEqual(summary["pool_size"], 1)
        self.assertEqual(summary["min_support_concentration"], 0.08)
        self.assertEqual(summary["low_new_view_max"], 0.165)

    def test_pose_only_reference_pool_v5_blocks_mid_new_view_refs(self):
        import torch

        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v5")
        )
        accepted = gate.register_pose_only_reference(
            frame_id=320,
            info={"is_test": False, "image_name": "forest2_mid_view"},
            desc_kpts=_pose_pool_desc(support_count=700, match_score=480),
            Rt=torch.eye(4),
            density_debug={
                "active_memory_context": True,
                "active_memory_frame_role": "tracking_only",
                "support_concentration": 0.10,
                "new_view_event_score": 0.18,
                "anchor_health_score": 0.60,
                "keyframe_growth_recent": -8,
            },
            pose_debug={"num_pnp_inliers": 300, "num_miniba_inliers": 300},
        )

        self.assertFalse(accepted)
        self.assertEqual(
            gate.pose_reference_pool_events[-1]["block_reason"],
            "pose_only_risk_gate_not_met",
        )

    def test_pose_only_reference_pool_v6_blocks_zero_growth_refs(self):
        import torch

        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v6")
        )
        accepted = gate.register_pose_only_reference(
            frame_id=320,
            info={"is_test": False, "image_name": "zero_growth_reference"},
            desc_kpts=_pose_pool_desc(support_count=700, match_score=480),
            Rt=torch.eye(4),
            density_debug={
                "active_memory_context": True,
                "active_memory_frame_role": "tracking_only",
                "support_concentration": 0.10,
                "new_view_event_score": 0.15,
                "anchor_health_score": 0.60,
                "keyframe_growth_recent": 0,
                "inlier_grid_entropy": 0.92,
            },
            pose_debug={"num_pnp_inliers": 300, "num_miniba_inliers": 300},
        )

        self.assertFalse(accepted)
        self.assertEqual(
            gate.pose_reference_pool_events[-1]["block_reason"],
            "pose_only_growth_not_stalled",
        )

    def test_pose_only_reference_pool_v6_blocks_repetitive_entropy_low_support_refs(self):
        import torch

        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v6")
        )
        accepted = gate.register_pose_only_reference(
            frame_id=320,
            info={"is_test": False, "image_name": "forest_repetitive_reference"},
            desc_kpts=_pose_pool_desc(support_count=700, match_score=480),
            Rt=torch.eye(4),
            density_debug={
                "active_memory_context": True,
                "active_memory_frame_role": "tracking_only",
                "support_concentration": 0.075,
                "new_view_event_score": 0.14,
                "anchor_health_score": 0.60,
                "keyframe_growth_recent": -8,
                "inlier_grid_entropy": 0.955,
            },
            pose_debug={"num_pnp_inliers": 300, "num_miniba_inliers": 300},
        )

        self.assertFalse(accepted)
        self.assertEqual(
            gate.pose_reference_pool_events[-1]["block_reason"],
            "pose_only_repetitive_entropy_low_support",
        )

    def test_pose_only_reference_pool_v6_accepts_strict_university_refs(self):
        import torch

        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v6")
        )
        accepted = gate.register_pose_only_reference(
            frame_id=320,
            info={"is_test": False, "image_name": "university_low_growth_reference"},
            desc_kpts=_pose_pool_desc(support_count=700, match_score=480),
            Rt=torch.eye(4),
            density_debug={
                "active_memory_context": True,
                "active_memory_frame_role": "tracking_only",
                "support_concentration": 0.09,
                "new_view_event_score": 0.15,
                "anchor_health_score": 0.60,
                "keyframe_growth_recent": -8,
                "inlier_grid_entropy": 0.92,
            },
            pose_debug={"num_pnp_inliers": 300, "num_miniba_inliers": 300},
        )

        self.assertTrue(accepted)
        summary = gate.pose_only_reference_pool_summary()
        self.assertEqual(summary["pool_size"], 1)
        self.assertEqual(summary["min_support_concentration"], 0.08)
        self.assertTrue(summary["requires_negative_growth"])

    def test_pose_only_reference_pool_v6_blocks_high_new_view_pnp_rescue(self):
        import torch

        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v6")
        )
        accepted = gate.register_pose_only_reference(
            frame_id=320,
            info={"is_test": False, "image_name": "high_new_view_pnp_reference"},
            desc_kpts=_pose_pool_desc(support_count=700, match_score=480),
            Rt=torch.eye(4),
            density_debug={
                "active_memory_context": True,
                "active_memory_frame_role": "tracking_only",
                "support_concentration": 0.13,
                "new_view_event_score": 0.25,
                "anchor_health_score": 0.60,
                "keyframe_growth_recent": -8,
                "inlier_grid_entropy": 0.88,
            },
            pose_debug={"num_pnp_inliers": 300, "num_miniba_inliers": 300},
        )

        self.assertFalse(accepted)
        self.assertEqual(
            gate.pose_reference_pool_events[-1]["block_reason"],
            "pose_only_high_new_view_pnp_disabled",
        )

    def test_active_memory_v8_debug_carries_ssm_reference_gate_metrics(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v8")
        )

        decision = controller.decide(
            frame_id=420,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=86.0,
            local_density_before=72.0,
            local_window_density=72.0,
            local_window_keyframes=72,
            local_window_gap_max=2.0,
            local_window_gap_after_if_hold=2.0,
            keyframe_growth_recent=-8,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=1,
            main_chain_gap_before=1.0,
            main_chain_gap_after_if_hold=2.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=20.0,
            displacement_threshold=30.0,
            num_matches=2400,
            min_num_inliers=100,
            pose_inliers=1600,
            novelty_proxy=0.02,
            current_keyframe_count=360,
            semantic_scores={"R_t": 0.05, "V_t": 0.99, "Q_t": 0.98, "C_t": 0.99, "B_R_t": 0.95},
            viewpoint_scores={
                "viewpoint_rotation_deg_window_max": 4.0,
                "inlier_grid_coverage": 0.98,
                "inlier_grid_entropy": 0.955,
                "support_concentration": 0.075,
                "anchor_health_score": 0.60,
                "new_view_event_score": 0.14,
            },
        )

        self.assertFalse(decision.finalize)
        self.assertEqual(decision.decision, "hold_low_representation_value")
        self.assertEqual(decision.debug["support_concentration"], 0.075)
        self.assertEqual(decision.debug["inlier_grid_entropy"], 0.955)
        self.assertEqual(decision.debug["anchor_health_score"], 0.60)
        self.assertEqual(decision.debug["new_view_event_score"], 0.14)

    def test_pose_only_reference_pool_v8_uses_real_ssm_metrics_to_block_forest_like_refs(self):
        import torch

        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v8")
        )
        controller = gate.direct_density_controller
        decision = controller.decide(
            frame_id=420,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=86.0,
            local_density_before=72.0,
            local_window_density=72.0,
            local_window_keyframes=72,
            local_window_gap_max=2.0,
            local_window_gap_after_if_hold=2.0,
            keyframe_growth_recent=-8,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=1,
            main_chain_gap_before=1.0,
            main_chain_gap_after_if_hold=2.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=20.0,
            displacement_threshold=30.0,
            num_matches=2400,
            min_num_inliers=100,
            pose_inliers=1600,
            novelty_proxy=0.02,
            current_keyframe_count=360,
            semantic_scores={"R_t": 0.05, "V_t": 0.99, "Q_t": 0.98, "C_t": 0.99, "B_R_t": 0.95},
            viewpoint_scores={
                "viewpoint_rotation_deg_window_max": 4.0,
                "inlier_grid_coverage": 0.98,
                "inlier_grid_entropy": 0.955,
                "support_concentration": 0.075,
                "anchor_health_score": 0.60,
                "new_view_event_score": 0.14,
            },
        )

        accepted = gate.register_pose_only_reference(
            frame_id=420,
            info={"is_test": False, "image_name": "forest_metric_block"},
            desc_kpts=_pose_pool_desc(support_count=700, match_score=480),
            Rt=torch.eye(4),
            density_debug=decision.debug,
            pose_debug={"num_pnp_inliers": 300, "num_miniba_inliers": 300},
        )

        self.assertFalse(accepted)
        self.assertEqual(
            gate.pose_reference_pool_events[-1]["block_reason"],
            "pose_only_repetitive_entropy_low_support",
        )
        self.assertAlmostEqual(
            gate.pose_reference_pool_events[-1]["pose_only_support_concentration"],
            0.075,
        )

    def test_pose_only_reference_pool_v8_accepts_real_ssm_university_like_refs(self):
        import torch

        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v8")
        )
        controller = gate.direct_density_controller
        decision = controller.decide(
            frame_id=420,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=86.0,
            local_density_before=72.0,
            local_window_density=72.0,
            local_window_keyframes=72,
            local_window_gap_max=2.0,
            local_window_gap_after_if_hold=2.0,
            keyframe_growth_recent=-8,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=1,
            main_chain_gap_before=1.0,
            main_chain_gap_after_if_hold=2.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=20.0,
            displacement_threshold=30.0,
            num_matches=2400,
            min_num_inliers=100,
            pose_inliers=1600,
            novelty_proxy=0.02,
            current_keyframe_count=360,
            semantic_scores={"R_t": 0.05, "V_t": 0.99, "Q_t": 0.98, "C_t": 0.99, "B_R_t": 0.95},
            viewpoint_scores={
                "viewpoint_rotation_deg_window_max": 4.0,
                "inlier_grid_coverage": 0.98,
                "inlier_grid_entropy": 0.91,
                "support_concentration": 0.09,
                "anchor_health_score": 0.60,
                "new_view_event_score": 0.15,
            },
        )

        accepted = gate.register_pose_only_reference(
            frame_id=420,
            info={"is_test": False, "image_name": "university_metric_accept"},
            desc_kpts=_pose_pool_desc(support_count=700, match_score=480),
            Rt=torch.eye(4),
            density_debug=decision.debug,
            pose_debug={"num_pnp_inliers": 300, "num_miniba_inliers": 300},
        )

        self.assertTrue(accepted)
        self.assertEqual(gate.pose_only_reference_pool_summary()["pool_size"], 1)

    def test_pose_only_reference_pool_v9_accepts_supported_high_new_view_refs(self):
        import torch

        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v9")
        )
        accepted = gate.register_pose_only_reference(
            frame_id=420,
            info={"is_test": False, "image_name": "long_supported_high_new_view"},
            desc_kpts=_pose_pool_desc(support_count=700, match_score=480),
            Rt=torch.eye(4),
            density_debug={
                "active_memory_context": True,
                "active_memory_frame_role": "tracking_only",
                "support_concentration": 0.105,
                "new_view_event_score": 0.25,
                "anchor_health_score": 0.66,
                "keyframe_growth_recent": -8,
                "inlier_grid_entropy": 0.895,
            },
            pose_debug={"num_pnp_inliers": 300, "num_miniba_inliers": 300},
        )

        self.assertTrue(accepted)
        summary = gate.pose_only_reference_pool_summary()
        self.assertTrue(summary["allow_high_new_view_rescue"])
        self.assertEqual(summary["high_new_view_entropy_max"], 0.92)

    def test_pose_only_reference_pool_v9_blocks_healthy_anchor_high_new_view_refs(self):
        import torch

        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v9")
        )
        accepted = gate.register_pose_only_reference(
            frame_id=420,
            info={"is_test": False, "image_name": "desk_healthy_high_new_view"},
            desc_kpts=_pose_pool_desc(support_count=700, match_score=480),
            Rt=torch.eye(4),
            density_debug={
                "active_memory_context": True,
                "active_memory_frame_role": "tracking_only",
                "support_concentration": 0.13,
                "new_view_event_score": 0.27,
                "anchor_health_score": 0.76,
                "keyframe_growth_recent": -8,
                "inlier_grid_entropy": 0.87,
            },
            pose_debug={"num_pnp_inliers": 300, "num_miniba_inliers": 300},
        )

        self.assertFalse(accepted)
        self.assertEqual(
            gate.pose_reference_pool_events[-1]["block_reason"],
            "pose_only_anchor_healthy_high_new_view",
        )

    def test_pose_only_reference_pool_v9_blocks_high_entropy_high_new_view_refs(self):
        import torch

        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v9")
        )
        accepted = gate.register_pose_only_reference(
            frame_id=420,
            info={"is_test": False, "image_name": "forest_entropy_high_new_view"},
            desc_kpts=_pose_pool_desc(support_count=700, match_score=480),
            Rt=torch.eye(4),
            density_debug={
                "active_memory_context": True,
                "active_memory_frame_role": "tracking_only",
                "support_concentration": 0.105,
                "new_view_event_score": 0.25,
                "anchor_health_score": 0.66,
                "keyframe_growth_recent": -8,
                "inlier_grid_entropy": 0.945,
            },
            pose_debug={"num_pnp_inliers": 300, "num_miniba_inliers": 300},
        )

        self.assertFalse(accepted)
        self.assertEqual(
            gate.pose_reference_pool_events[-1]["block_reason"],
            "pose_only_risk_gate_not_met",
        )

    def test_pose_only_reference_pool_v24_uses_risk_aware_selection_score(self):
        import torch

        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v24")
        )
        rt = torch.eye(4)
        accepted = []
        for frame_id, name, match_score, support, new_view, anchor, entropy in [
            (420, "raw_match_only", 540, 0.081, 0.15, 0.62, 0.910),
            (448, "turn_bridge", 500, 0.130, 0.24, 0.64, 0.875),
            (476, "recent_raw_match", 560, 0.082, 0.14, 0.61, 0.905),
        ]:
            accepted.append(
                gate.register_pose_only_reference(
                    frame_id=frame_id,
                    info={"is_test": False, "image_name": name},
                    desc_kpts=_pose_pool_desc(support_count=700, match_score=match_score),
                    Rt=rt,
                    density_debug={
                        "active_memory_context": True,
                        "active_memory_frame_role": "tracking_only",
                        "support_concentration": support,
                        "new_view_event_score": new_view,
                        "anchor_health_score": anchor,
                        "keyframe_growth_recent": -8,
                        "inlier_grid_entropy": entropy,
                    },
                    pose_debug={"num_pnp_inliers": 320, "num_miniba_inliers": 300},
                )
            )

        selected = gate.select_pose_only_references(
            frame_id=520,
            curr_desc_kpts=_pose_pool_desc(match_score=0),
            matcher=_PosePoolMatcher(),
        )

        self.assertEqual(accepted, [True, True, True])
        self.assertEqual(
            [ref.info["_paper_aligned_source_frame_id"] for ref in selected],
            [448, 476],
        )
        summary = gate.pose_only_reference_pool_summary()
        self.assertEqual(summary["max_per_query"], 2)
        self.assertEqual(summary["selection_strategy"], "risk_aware")
        select_events = [
            event for event in gate.pose_reference_pool_events
            if event.get("event_type") == "pose_only_select_reference"
        ]
        self.assertEqual(select_events[0]["pose_only_selection_strategy"], "risk_aware")
        self.assertGreater(
            select_events[0]["pose_only_geometry_score"],
            select_events[1]["pose_only_geometry_score"],
        )

    def test_pose_only_reference_pool_v25_keeps_risk_aware_scoring_single_reference(self):
        import torch

        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v25")
        )
        rt = torch.eye(4)
        for frame_id, name, match_score, support, new_view, anchor, entropy in [
            (420, "raw_match_only", 540, 0.081, 0.15, 0.62, 0.910),
            (448, "turn_bridge", 500, 0.130, 0.24, 0.64, 0.875),
            (476, "recent_raw_match", 560, 0.082, 0.14, 0.61, 0.905),
        ]:
            self.assertTrue(
                gate.register_pose_only_reference(
                    frame_id=frame_id,
                    info={"is_test": False, "image_name": name},
                    desc_kpts=_pose_pool_desc(support_count=700, match_score=match_score),
                    Rt=rt,
                    density_debug={
                        "active_memory_context": True,
                        "active_memory_frame_role": "tracking_only",
                        "support_concentration": support,
                        "new_view_event_score": new_view,
                        "anchor_health_score": anchor,
                        "keyframe_growth_recent": -8,
                        "inlier_grid_entropy": entropy,
                    },
                    pose_debug={"num_pnp_inliers": 320, "num_miniba_inliers": 300},
                )
            )

        selected = gate.select_pose_only_references(
            frame_id=520,
            curr_desc_kpts=_pose_pool_desc(match_score=0),
            matcher=_PosePoolMatcher(),
        )

        self.assertEqual(
            [ref.info["_paper_aligned_source_frame_id"] for ref in selected],
            [448],
        )
        summary = gate.pose_only_reference_pool_summary()
        self.assertEqual(summary["max_per_query"], 1)
        self.assertEqual(summary["selection_strategy"], "risk_aware")

    def test_pose_only_reference_pool_v26_blocks_only_late_extreme_new_view_thin_support(self):
        import torch

        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v26")
        )
        accepted = gate.register_pose_only_reference(
            frame_id=1272,
            info={"is_test": False, "image_name": "long_late_extreme_new_view"},
            desc_kpts=_pose_pool_desc(support_count=700, match_score=480),
            Rt=torch.eye(4),
            density_debug={
                "active_memory_context": True,
                "active_memory_frame_role": "tracking_only",
                "support_concentration": 0.102,
                "new_view_event_score": 0.454,
                "anchor_health_score": 0.665,
                "keyframe_growth_recent": -1,
                "inlier_grid_entropy": 0.898,
                "viewpoint_rotation_deg_window_max": 74.4,
            },
            pose_debug={"num_pnp_inliers": 300, "num_miniba_inliers": 300},
        )

        self.assertFalse(accepted)
        self.assertEqual(
            gate.pose_reference_pool_events[-1]["block_reason"],
            "pose_only_late_extreme_new_view_thin_support",
        )
        summary = gate.pose_only_reference_pool_summary()
        self.assertEqual(summary["late_extreme_new_view_frame_min"], 1200)
        self.assertEqual(summary["late_extreme_new_view_support_min"], 0.15)

    def test_pose_only_reference_pool_v26_does_not_enable_v10_early_turn_bridge(self):
        import torch

        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v26")
        )
        accepted = gate.register_pose_only_reference(
            frame_id=752,
            info={"is_test": False, "image_name": "early_turn_not_bridged"},
            desc_kpts=_pose_pool_desc(support_count=700, match_score=480),
            Rt=torch.eye(4),
            density_debug={
                "active_memory_context": True,
                "active_memory_frame_role": "tracking_only",
                "support_concentration": 0.104,
                "new_view_event_score": 0.30,
                "anchor_health_score": 0.715,
                "keyframe_growth_recent": -15,
                "inlier_grid_entropy": 0.896,
                "viewpoint_rotation_deg_window_max": 45.6,
            },
            pose_debug={"num_pnp_inliers": 300, "num_miniba_inliers": 300},
        )

        self.assertFalse(accepted)
        self.assertEqual(
            gate.pose_reference_pool_events[-1]["block_reason"],
            "pose_only_anchor_healthy_high_new_view",
        )
        self.assertEqual(
            gate.pose_only_reference_pool_summary()["early_turn_bridge_frame_max"],
            0,
        )

    def test_pose_only_reference_pool_v30_inherits_strict_health_gate(self):
        import torch

        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v30")
        )

        summary = gate.pose_only_reference_pool_summary()
        self.assertEqual(summary["max_size"], 12)
        self.assertEqual(summary["ttl_frames"], 100)
        self.assertEqual(summary["min_age_frames"], 28)
        self.assertEqual(summary["min_match_score"], 280.0)
        self.assertEqual(summary["register_min_interval_frames"], 24)
        self.assertEqual(summary["selection_cooldown_frames"], 12)
        self.assertTrue(summary["risk_gate_enabled"])
        self.assertTrue(summary["requires_negative_growth"])
        self.assertTrue(summary["allow_high_new_view_rescue"])
        self.assertEqual(summary["high_new_view_entropy_max"], 0.92)
        self.assertEqual(summary["late_extreme_new_view_frame_min"], 1200)
        self.assertEqual(summary["late_extreme_new_view_support_min"], 0.15)
        self.assertEqual(summary["selection_strategy"], "risk_aware")

        accepted = gate.register_pose_only_reference(
            frame_id=520,
            info={"is_test": False, "image_name": "default_pool_would_accept"},
            desc_kpts=_pose_pool_desc(support_count=700, match_score=500),
            Rt=torch.eye(4),
            density_debug={
                "active_memory_context": True,
                "active_memory_frame_role": "tracking_only",
                "support_concentration": 0.04,
                "new_view_event_score": 0.14,
                "anchor_health_score": 0.60,
                "keyframe_growth_recent": -8,
                "inlier_grid_entropy": 0.955,
            },
            pose_debug={"num_pnp_inliers": 300, "num_miniba_inliers": 300},
        )

        self.assertFalse(accepted)
        self.assertEqual(
            gate.pose_reference_pool_events[-1]["block_reason"],
            "pose_only_repetitive_entropy_low_support",
        )

    def test_active_memory_v30_keeps_v29_hard_window_without_utility_hold(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v30")
        )

        decision = controller.decide(
            frame_id=620,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=86.0,
            local_density_before=82.0,
            local_window_density=82.0,
            local_window_keyframes=82,
            local_window_gap_max=5.0,
            local_window_gap_after_if_hold=5.0,
            keyframe_growth_recent=16,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=1,
            main_chain_gap_before=2.0,
            main_chain_gap_after_if_hold=3.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=20.0,
            displacement_threshold=30.0,
            num_matches=2200,
            min_num_inliers=100,
            pose_inliers=1600,
            novelty_proxy=0.10,
            current_keyframe_count=520,
            semantic_scores={
                "R_t": 0.04,
                "V_t": 0.92,
                "Q_t": 0.94,
                "C_t": 0.92,
                "B_R_t": 0.91,
            },
            viewpoint_scores={
                "viewpoint_rotation_deg_window_max": 24.0,
                "inlier_grid_coverage": 0.91,
                "support_concentration": 0.12,
                "anchor_health_score": 0.52,
                "new_view_event_score": 0.32,
            },
        )

        self.assertTrue(decision.finalize)
        self.assertEqual(decision.decision, "finalize_high_representation_value")
        self.assertEqual(decision.reason, "utility_hard_window_representation_guard")
        self.assertTrue(decision.debug["utility_hard_window_guard"])
        self.assertTrue(decision.debug["utility_representation_role"])
        self.assertFalse(decision.debug["utility_tracking_only_role"])

    def test_active_memory_v31_ignores_short_window_turns_for_hard_guard(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v31")
        )

        decision = controller.decide(
            frame_id=620,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=86.0,
            local_density_before=82.0,
            local_window_density=82.0,
            local_window_keyframes=82,
            local_window_gap_max=5.0,
            local_window_gap_after_if_hold=5.0,
            keyframe_growth_recent=16,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=1,
            main_chain_gap_before=2.0,
            main_chain_gap_after_if_hold=3.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=20.0,
            displacement_threshold=30.0,
            num_matches=2200,
            min_num_inliers=100,
            pose_inliers=1600,
            novelty_proxy=0.10,
            current_keyframe_count=520,
            semantic_scores={
                "R_t": 0.04,
                "V_t": 0.92,
                "Q_t": 0.94,
                "C_t": 0.92,
                "B_R_t": 0.91,
            },
            viewpoint_scores={
                "viewpoint_rotation_deg_window_20": 24.0,
                "viewpoint_rotation_deg_window_50": 6.0,
                "viewpoint_rotation_deg_window_100": 4.0,
                "viewpoint_rotation_deg_window_max": 24.0,
                "viewpoint_rotation_window_max_size": 20,
                "inlier_grid_coverage": 0.91,
                "support_concentration": 0.12,
                "anchor_health_score": 0.52,
                "new_view_event_score": 0.32,
            },
        )

        self.assertFalse(decision.debug["utility_hard_window_guard"])
        self.assertFalse(decision.debug["utility_representation_role"])
        self.assertFalse(decision.debug["utility_tracking_only_role"])
        self.assertNotEqual(decision.reason, "utility_hard_window_representation_guard")

    def test_active_memory_v31_preserves_persistent_long_turn_hard_guard(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v31")
        )

        decision = controller.decide(
            frame_id=1220,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=86.0,
            local_density_before=82.0,
            local_window_density=82.0,
            local_window_keyframes=82,
            local_window_gap_max=5.0,
            local_window_gap_after_if_hold=5.0,
            keyframe_growth_recent=16,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=1,
            main_chain_gap_before=2.0,
            main_chain_gap_after_if_hold=3.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=20.0,
            displacement_threshold=30.0,
            num_matches=2200,
            min_num_inliers=100,
            pose_inliers=1600,
            novelty_proxy=0.10,
            current_keyframe_count=720,
            semantic_scores={
                "R_t": 0.04,
                "V_t": 0.92,
                "Q_t": 0.94,
                "C_t": 0.92,
                "B_R_t": 0.91,
            },
            viewpoint_scores={
                "viewpoint_rotation_deg_window_20": 18.0,
                "viewpoint_rotation_deg_window_50": 26.0,
                "viewpoint_rotation_deg_window_100": 32.0,
                "viewpoint_rotation_deg_window_max": 32.0,
                "viewpoint_rotation_window_max_size": 100,
                "inlier_grid_coverage": 0.91,
                "support_concentration": 0.12,
                "anchor_health_score": 0.52,
                "new_view_event_score": 0.42,
            },
        )

        self.assertTrue(decision.finalize)
        self.assertEqual(decision.decision, "finalize_high_representation_value")
        self.assertEqual(decision.reason, "utility_hard_window_representation_guard")
        self.assertTrue(decision.debug["utility_hard_window_guard"])
        self.assertTrue(decision.debug["utility_persistent_long_turn_context"])
        self.assertTrue(decision.debug["utility_representation_role"])
        self.assertFalse(decision.debug["utility_tracking_only_role"])

    def test_active_memory_v33_ignores_forest_scale_late_turns(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v33")
        )

        decision = controller.decide(
            frame_id=1180,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=86.0,
            local_density_before=82.0,
            local_window_density=82.0,
            local_window_keyframes=82,
            local_window_gap_max=5.0,
            local_window_gap_after_if_hold=5.0,
            keyframe_growth_recent=16,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=1,
            main_chain_gap_before=2.0,
            main_chain_gap_after_if_hold=3.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=18.0,
            displacement_threshold=32.0,
            num_matches=2400,
            min_num_inliers=100,
            pose_inliers=1800,
            novelty_proxy=0.08,
            current_keyframe_count=740,
            semantic_scores={
                "R_t": 0.03,
                "V_t": 0.24,
                "Q_t": 0.94,
                "C_t": 0.93,
                "B_R_t": 0.92,
            },
            viewpoint_scores={
                "viewpoint_rotation_deg_window_20": 18.0,
                "viewpoint_rotation_deg_window_50": 26.0,
                "viewpoint_rotation_deg_window_100": 32.0,
                "viewpoint_rotation_deg_window_max": 32.0,
                "viewpoint_rotation_window_max_size": 100,
                "inlier_grid_coverage": 0.96,
                "support_concentration": 0.10,
                "anchor_health_score": 0.72,
                "new_view_event_score": 0.24,
            },
        )

        self.assertFalse(decision.debug["utility_late_long_turn_context"])
        self.assertFalse(decision.debug["active_memory_late_long_turn_tracking_context"])
        self.assertFalse(decision.debug["utility_hard_window_guard"])
        self.assertFalse(decision.debug["utility_representation_role"])
        self.assertNotEqual(decision.reason, "utility_hard_window_representation_guard")

    def test_pose_memory_geometry_context_holds_forest_scale_low_representation_frames(self):
        controller = DirectDensityController(
            _args(
                paper_aligned_direct_density_control="pose_rep_active_memory_v33",
                paper_aligned_pose_memory_geometry_context="v1",
            )
        )

        decision = controller.decide(
            frame_id=1180,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=86.0,
            local_density_before=82.0,
            local_window_density=82.0,
            local_window_keyframes=82,
            local_window_gap_max=5.0,
            local_window_gap_after_if_hold=5.0,
            keyframe_growth_recent=16,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=1,
            main_chain_gap_before=2.0,
            main_chain_gap_after_if_hold=3.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=18.0,
            displacement_threshold=32.0,
            num_matches=2400,
            min_num_inliers=100,
            pose_inliers=1800,
            novelty_proxy=0.08,
            current_keyframe_count=740,
            semantic_scores={
                "R_t": 0.03,
                "V_t": 0.24,
                "Q_t": 0.94,
                "C_t": 0.93,
                "B_R_t": 0.92,
            },
            viewpoint_scores={
                "viewpoint_rotation_deg_window_20": 18.0,
                "viewpoint_rotation_deg_window_50": 26.0,
                "viewpoint_rotation_deg_window_100": 32.0,
                "viewpoint_rotation_deg_window_max": 32.0,
                "viewpoint_rotation_window_max_size": 100,
                "inlier_grid_coverage": 0.96,
                "support_concentration": 0.10,
                "anchor_health_score": 0.72,
                "new_view_event_score": 0.24,
                "pose_memory_reference_count": 1,
                "pose_memory_pool_size": 3,
                "pose_memory_candidate_pool_size": 0,
            },
        )

        self.assertFalse(decision.finalize)
        self.assertEqual(decision.decision, "hold_low_representation_value")
        self.assertEqual(decision.reason, "pose_memory_geometry_tracking_context")
        self.assertTrue(decision.debug["pose_memory_geometry_context_enabled"])
        self.assertTrue(decision.debug["pose_memory_geometry_tracking_context"])
        self.assertTrue(decision.debug["pose_memory_context_without_candidate_pool"])
        self.assertTrue(decision.debug["active_memory_context"])
        self.assertEqual(decision.debug["active_memory_frame_role"], "tracking_only")

    def test_pose_memory_geometry_context_blocks_weak_geometry_tracking_hold(self):
        controller = DirectDensityController(
            _args(
                paper_aligned_direct_density_control="pose_rep_active_memory_v33",
                paper_aligned_pose_memory_geometry_context="v1",
            )
        )

        decision = controller.decide(
            frame_id=1180,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=86.0,
            local_density_before=82.0,
            local_window_density=82.0,
            local_window_keyframes=82,
            local_window_gap_max=5.0,
            local_window_gap_after_if_hold=5.0,
            keyframe_growth_recent=16,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=1,
            main_chain_gap_before=2.0,
            main_chain_gap_after_if_hold=3.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=18.0,
            displacement_threshold=32.0,
            num_matches=900,
            min_num_inliers=100,
            pose_inliers=650,
            novelty_proxy=0.08,
            current_keyframe_count=740,
            semantic_scores={
                "R_t": 0.03,
                "V_t": 0.24,
                "Q_t": 0.94,
                "C_t": 0.93,
                "B_R_t": 0.92,
            },
            viewpoint_scores={
                "viewpoint_rotation_deg_window_20": 18.0,
                "viewpoint_rotation_deg_window_50": 26.0,
                "viewpoint_rotation_deg_window_100": 32.0,
                "viewpoint_rotation_deg_window_max": 32.0,
                "viewpoint_rotation_window_max_size": 100,
                "inlier_grid_coverage": 0.55,
                "support_concentration": 0.38,
                "anchor_health_score": 0.46,
                "new_view_event_score": 0.54,
                "pose_memory_reference_count": 0,
                "pose_memory_pool_size": 0,
                "pose_memory_candidate_pool_size": 0,
            },
        )

        self.assertTrue(decision.finalize)
        self.assertEqual(decision.decision, "finalize_high_representation_value")
        self.assertEqual(decision.reason, "pose_memory_geometry_guard")
        self.assertTrue(decision.debug["pose_memory_geometry_context_enabled"])
        self.assertFalse(decision.debug["pose_memory_geometry_tracking_context"])
        self.assertTrue(decision.debug["pose_memory_geometry_guard"])

    def test_streaming_memory_defers_high_value_geometry_unsafe_frames(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_rep_streaming_memory_v1")
        )

        decision = controller.decide(
            frame_id=820,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=62.0,
            local_density_before=58.0,
            local_window_density=58.0,
            local_window_keyframes=58,
            local_window_gap_max=6.0,
            local_window_gap_after_if_hold=4.0,
            keyframe_growth_recent=36,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=2,
            main_chain_gap_before=2.0,
            main_chain_gap_after_if_hold=4.0,
            anchor_changed=False,
            support_triggered=True,
            median_displacement=54.0,
            displacement_threshold=30.0,
            num_matches=720,
            min_num_inliers=100,
            pose_inliers=520,
            novelty_proxy=0.82,
            current_keyframe_count=508,
            semantic_scores={
                "R_t": 0.30,
                "V_t": 0.86,
                "Q_t": 0.64,
                "C_t": 0.58,
                "B_R_t": 0.55,
                "recovery_pool_size": 5,
            },
            viewpoint_scores={
                "viewpoint_rotation_deg_window_20": 26.0,
                "viewpoint_rotation_deg_window_50": 38.0,
                "viewpoint_rotation_deg_window_100": 44.0,
                "viewpoint_rotation_deg_window_max": 48.0,
                "viewpoint_rotation_window_max_size": 100,
                "inlier_grid_coverage": 0.58,
                "inlier_grid_entropy": 0.62,
                "support_concentration": 0.34,
                "anchor_health_score": 0.38,
                "new_view_event_score": 0.72,
                "pose_memory_reference_count": 0,
                "pose_memory_pool_size": 1,
                "pose_memory_candidate_pool_size": 0,
            },
        )

        self.assertFalse(decision.finalize)
        self.assertEqual(decision.decision, "hold_candidate_verification")
        self.assertEqual(decision.debug["stream_memory_frame_identity"], "defer")
        self.assertEqual(decision.debug["stream_memory_memory_identity"], "candidate")
        self.assertEqual(decision.debug["stream_memory_write_action"], "candidate_verify")
        self.assertTrue(decision.debug["stream_memory_candidate_verification_required"])
        self.assertGreater(decision.debug["stream_memory_representation_need"], 0.45)
        self.assertLess(decision.debug["stream_memory_geometry_safety"], 0.60)

    def test_streaming_memory_holds_stable_low_representation_frames_as_pose_only(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_rep_streaming_memory_v1")
        )

        decision = controller.decide(
            frame_id=620,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=68.0,
            local_density_before=64.0,
            local_window_density=64.0,
            local_window_keyframes=64,
            local_window_gap_max=5.0,
            local_window_gap_after_if_hold=3.0,
            keyframe_growth_recent=18,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=1,
            main_chain_gap_before=2.0,
            main_chain_gap_after_if_hold=3.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=16.0,
            displacement_threshold=30.0,
            num_matches=2600,
            min_num_inliers=100,
            pose_inliers=2100,
            novelty_proxy=0.06,
            current_keyframe_count=420,
            semantic_scores={
                "R_t": 0.03,
                "V_t": 0.18,
                "Q_t": 0.95,
                "C_t": 0.94,
                "B_R_t": 0.93,
            },
            viewpoint_scores={
                "viewpoint_rotation_deg_window_20": 12.0,
                "viewpoint_rotation_deg_window_50": 22.0,
                "viewpoint_rotation_deg_window_100": 28.0,
                "viewpoint_rotation_deg_window_max": 30.0,
                "viewpoint_rotation_window_max_size": 100,
                "inlier_grid_coverage": 0.97,
                "inlier_grid_entropy": 0.96,
                "support_concentration": 0.08,
                "anchor_health_score": 0.68,
                "new_view_event_score": 0.20,
                "pose_memory_reference_count": 1,
                "pose_memory_pool_size": 4,
                "pose_memory_candidate_pool_size": 0,
            },
        )

        self.assertFalse(decision.finalize)
        self.assertEqual(decision.decision, "hold_low_representation_value")
        self.assertEqual(decision.debug["stream_memory_frame_identity"], "pose-only")
        self.assertEqual(decision.debug["stream_memory_memory_identity"], "local")
        self.assertEqual(decision.debug["stream_memory_write_action"], "pose_only_write")
        self.assertFalse(decision.debug["stream_memory_candidate_verification_required"])
        self.assertTrue(decision.debug["stream_memory_sparse_write"])

    def test_streaming_memory_only_enqueues_deferred_candidates_for_recovery(self):
        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_rep_streaming_memory_v1")
        )

        self.assertTrue(
            gate.direct_density_controller.should_enqueue_hold_recovery(
                "hold_candidate_verification"
            )
        )
        self.assertFalse(
            gate.direct_density_controller.should_enqueue_hold_recovery(
                "hold_low_representation_value"
            )
        )

    def test_pose_safe_streaming_memory_keeps_baseline_representation_contract(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_safe_streaming_memory_v1")
        )

        baseline_keyframe = controller.decide(
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
        pose_only_frame = controller.decide(
            frame_id=421,
            runtime_action="direct_admit",
            baseline_should_add=False,
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

        self.assertTrue(controller.is_pose_safe_streaming_memory_v1)
        self.assertTrue(controller.is_pose_rep_active_memory)
        self.assertTrue(baseline_keyframe.finalize)
        self.assertTrue(baseline_keyframe.debug["representation_updated"])
        self.assertEqual(baseline_keyframe.decision, "finalize_baseline_repr")
        self.assertFalse(pose_only_frame.finalize)
        self.assertEqual(pose_only_frame.decision, "hold_pose_only_baseline_repr")
        self.assertEqual(pose_only_frame.debug["active_memory_frame_role"], "tracking_only")
        self.assertTrue(
            controller.should_update_prev_desc_on_hold(
                pose_only_frame.decision,
                pose_only_frame.debug["density_state"],
            )
        )
        self.assertFalse(controller.should_enqueue_hold_recovery(pose_only_frame.decision))

    def test_pose_safe_streaming_memory_tracks_only_strong_deferred_frames(self):
        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_safe_streaming_memory_v1")
        )

        allowed = gate.should_pose_safe_track_deferred(
            frame_id=128,
            action="defer_recoverable",
            phase="incremental",
            evidence={
                "num_matches": 360,
                "min_num_inliers_threshold": 100,
                "median_displacement": 0.020,
                "displacement_threshold": 0.030,
                "recent_pose_fail_rate": 0.0,
            },
        )
        weak = gate.should_pose_safe_track_deferred(
            frame_id=129,
            action="defer_recoverable",
            phase="incremental",
            evidence={
                "num_matches": 120,
                "min_num_inliers_threshold": 100,
                "median_displacement": 0.020,
                "displacement_threshold": 0.030,
                "recent_pose_fail_rate": 0.0,
            },
        )
        discard = gate.should_pose_safe_track_deferred(
            frame_id=130,
            action="discard",
            phase="incremental",
            evidence={
                "num_matches": 500,
                "min_num_inliers_threshold": 100,
                "median_displacement": 0.020,
                "displacement_threshold": 0.030,
                "recent_pose_fail_rate": 0.0,
            },
        )

        self.assertTrue(allowed)
        self.assertFalse(weak)
        self.assertFalse(discard)

    def test_pose_safe_streaming_memory_preserves_baseline_keyframe_skeleton(self):
        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_safe_streaming_memory_v1")
        )

        self.assertTrue(
            gate.should_pose_safe_preserve_baseline_keyframe(
                action="defer_recoverable",
                baseline_should_add=True,
                phase="incremental",
            )
        )
        self.assertFalse(
            gate.should_pose_safe_preserve_baseline_keyframe(
                action="defer_recoverable",
                baseline_should_add=False,
                phase="incremental",
            )
        )
        self.assertFalse(
            gate.should_pose_safe_preserve_baseline_keyframe(
                action="direct_admit",
                baseline_should_add=True,
                phase="incremental",
            )
        )

    def test_pose_safe_streaming_memory_budgets_deferred_tracking_per_window(self):
        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_safe_streaming_memory_v1")
        )
        evidence = {
            "num_matches": 500,
            "min_num_inliers_threshold": 100,
            "median_displacement": 0.025,
            "displacement_threshold": 0.030,
            "recent_pose_fail_rate": 0.0,
        }
        decisions = [
            gate.should_pose_safe_track_deferred(
                frame_id=200 + i,
                action="defer_recoverable",
                phase="incremental",
                evidence=evidence,
            )
            for i in range(gate.pose_safe_tracking_budget_per_100 + 2)
        ]

        self.assertEqual(
            sum(1 for ok in decisions if ok),
            gate.pose_safe_tracking_budget_per_100,
        )
        self.assertFalse(decisions[-1])

    def test_pose_safe_streaming_memory_uses_strict_reference_pool_defaults(self):
        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_safe_streaming_memory_v1")
        )
        summary = gate.pose_only_reference_pool_summary()

        self.assertEqual(summary["selection_strategy"], "pose_safe")
        self.assertTrue(summary["risk_gate_enabled"])
        self.assertEqual(summary["max_per_query"], 1)
        self.assertLessEqual(summary["ttl_frames"], 120)
        self.assertGreaterEqual(summary["min_3d_points"], 500)
        self.assertGreaterEqual(summary["min_match_score"], 240.0)
        self.assertGreater(summary["min_geometry_score"], 0.0)

    def test_pose_safe_streaming_memory_blocks_forest_like_repetitive_refs(self):
        import torch

        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_safe_streaming_memory_v1")
        )
        accepted = gate.register_pose_only_reference(
            frame_id=640,
            info={"is_test": False, "image_name": "forest_like"},
            desc_kpts=_pose_pool_desc(support_count=850, match_score=520),
            Rt=torch.eye(4),
            density_debug={
                "active_memory_context": True,
                "active_memory_frame_role": "tracking_only",
                "support_concentration": 0.07,
                "new_view_event_score": 0.28,
                "anchor_health_score": 0.74,
                "keyframe_growth_recent": 8,
                "inlier_grid_entropy": 0.97,
                "pose_support_score": 0.90,
                "match_support_score": 0.92,
            },
            pose_debug={"num_pnp_inliers": 420, "num_miniba_inliers": 360},
        )

        self.assertFalse(accepted)
        self.assertEqual(
            gate.pose_reference_pool_events[-1]["block_reason"],
            "pose_only_repetitive_entropy_low_support",
        )

    def test_pose_safe_streaming_memory_selects_geometry_safe_ref_over_raw_match(self):
        import torch

        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_safe_streaming_memory_v1")
        )
        rt = torch.eye(4)
        for frame_id, name, match_score, support, new_view, anchor, entropy, pnp, miniba in [
            (420, "raw_match_unsafe", 650, 0.091, 0.18, 0.68, 0.91, 260, 240),
            (448, "geometry_safe_turn", 520, 0.180, 0.24, 0.58, 0.84, 450, 420),
        ]:
            self.assertTrue(
                gate.register_pose_only_reference(
                    frame_id=frame_id,
                    info={"is_test": False, "image_name": name},
                    desc_kpts=_pose_pool_desc(support_count=850, match_score=match_score),
                    Rt=rt,
                    density_debug={
                        "active_memory_context": True,
                        "active_memory_frame_role": "tracking_only",
                        "support_concentration": support,
                        "new_view_event_score": new_view,
                        "anchor_health_score": anchor,
                        "keyframe_growth_recent": 4,
                        "inlier_grid_entropy": entropy,
                        "pose_support_score": 0.90,
                        "match_support_score": 0.92,
                    },
                    pose_debug={"num_pnp_inliers": pnp, "num_miniba_inliers": miniba},
                )
            )

        selected = gate.select_pose_only_references(
            frame_id=540,
            curr_desc_kpts=_pose_pool_desc(match_score=0),
            matcher=_PosePoolMatcher(),
        )

        self.assertEqual(
            [ref.info["_paper_aligned_source_frame_id"] for ref in selected],
            [448],
        )
        select_events = [
            event for event in gate.pose_reference_pool_events
            if event.get("event_type") == "pose_only_select_reference"
        ]
        self.assertEqual(select_events[0]["pose_only_selection_strategy"], "pose_safe")
        self.assertGreater(select_events[0]["pose_only_geometry_score"], 100.0)

    def test_pose_memory_geometry_context_seeds_distributed_stable_pose_memory(self):
        gate = PaperAlignedRuntimeGate(
            _args(
                paper_aligned_direct_density_control="pose_rep_active_memory_v33",
                paper_aligned_pose_memory_geometry_context="v1",
            )
        )

        ok, reason, metrics = gate._pose_only_reference_risk_gate_decision(
            {
                "pose_memory_geometry_context_enabled": True,
                "active_memory_stable_pose_reference": True,
                "active_memory_low_marginal_representation": True,
                "pose_support_score": 0.90,
                "match_support_score": 0.92,
                "viewpoint_grid_coverage": 0.98,
                "support_concentration": 0.04,
                "inlier_grid_entropy": 0.96,
                "anchor_health_score": 0.62,
                "new_view_event_score": 0.18,
                "keyframe_growth_recent": 12,
                "viewpoint_rotation_deg_window_max": 24.0,
            },
            frame_id=420,
        )

        self.assertTrue(ok)
        self.assertEqual(reason, "")
        self.assertEqual(metrics["pose_memory_geometry_seed_context"], 1.0)

    def test_active_memory_v33_without_geometry_keeps_original_pose_memory_growth_gate(self):
        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v33")
        )

        ok, reason, metrics = gate._pose_only_reference_risk_gate_decision(
            {
                "pose_memory_geometry_context_enabled": False,
                "active_memory_stable_pose_reference": True,
                "active_memory_low_marginal_representation": True,
                "pose_support_score": 0.90,
                "match_support_score": 0.92,
                "viewpoint_grid_coverage": 0.98,
                "support_concentration": 0.04,
                "inlier_grid_entropy": 0.96,
                "anchor_health_score": 0.62,
                "new_view_event_score": 0.18,
                "keyframe_growth_recent": 12,
                "viewpoint_rotation_deg_window_max": 24.0,
            },
            frame_id=420,
        )

        self.assertFalse(ok)
        self.assertEqual(reason, "pose_only_growth_not_stalled")
        self.assertEqual(metrics["pose_memory_geometry_seed_context"], 0.0)

    def test_active_memory_v33_holds_late_long_turn_low_representation_frames(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v33")
        )

        decision = controller.decide(
            frame_id=1600,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=86.0,
            local_density_before=82.0,
            local_window_density=82.0,
            local_window_keyframes=82,
            local_window_gap_max=5.0,
            local_window_gap_after_if_hold=5.0,
            keyframe_growth_recent=16,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=1,
            main_chain_gap_before=2.0,
            main_chain_gap_after_if_hold=3.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=18.0,
            displacement_threshold=32.0,
            num_matches=2400,
            min_num_inliers=100,
            pose_inliers=1800,
            novelty_proxy=0.08,
            current_keyframe_count=900,
            semantic_scores={
                "R_t": 0.03,
                "V_t": 0.24,
                "Q_t": 0.94,
                "C_t": 0.93,
                "B_R_t": 0.92,
            },
            viewpoint_scores={
                "viewpoint_rotation_deg_window_20": 18.0,
                "viewpoint_rotation_deg_window_50": 26.0,
                "viewpoint_rotation_deg_window_100": 32.0,
                "viewpoint_rotation_deg_window_max": 32.0,
                "viewpoint_rotation_window_max_size": 100,
                "inlier_grid_coverage": 0.96,
                "support_concentration": 0.10,
                "anchor_health_score": 0.72,
                "new_view_event_score": 0.24,
            },
        )

        self.assertFalse(decision.finalize)
        self.assertEqual(decision.decision, "hold_low_representation_value")
        self.assertTrue(decision.debug["utility_late_long_turn_context"])
        self.assertTrue(decision.debug["active_memory_late_long_turn_tracking_context"])
        self.assertTrue(decision.debug["active_memory_context"])
        self.assertFalse(decision.debug["utility_hard_window_guard"])
        self.assertFalse(decision.debug["utility_representation_role"])

    def test_active_memory_v33_finalizes_late_long_turn_unsafe_new_view_frames(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v33")
        )

        decision = controller.decide(
            frame_id=1600,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=86.0,
            local_density_before=82.0,
            local_window_density=82.0,
            local_window_keyframes=82,
            local_window_gap_max=5.0,
            local_window_gap_after_if_hold=5.0,
            keyframe_growth_recent=16,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=1,
            main_chain_gap_before=2.0,
            main_chain_gap_after_if_hold=3.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=20.0,
            displacement_threshold=30.0,
            num_matches=1800,
            min_num_inliers=100,
            pose_inliers=1200,
            novelty_proxy=0.18,
            current_keyframe_count=900,
            semantic_scores={
                "R_t": 0.04,
                "V_t": 0.78,
                "Q_t": 0.90,
                "C_t": 0.78,
                "B_R_t": 0.86,
            },
            viewpoint_scores={
                "viewpoint_rotation_deg_window_20": 18.0,
                "viewpoint_rotation_deg_window_50": 26.0,
                "viewpoint_rotation_deg_window_100": 32.0,
                "viewpoint_rotation_deg_window_max": 32.0,
                "viewpoint_rotation_window_max_size": 100,
                "inlier_grid_coverage": 0.78,
                "support_concentration": 0.22,
                "anchor_health_score": 0.54,
                "new_view_event_score": 0.48,
            },
        )

        self.assertTrue(decision.finalize)
        self.assertEqual(decision.decision, "finalize_high_representation_value")
        self.assertEqual(decision.reason, "utility_hard_window_representation_guard")
        self.assertTrue(decision.debug["utility_late_long_turn_context"])
        self.assertTrue(decision.debug["utility_hard_window_guard"])
        self.assertTrue(decision.debug["v33_late_long_turn_unsafe_representation_context"])
        self.assertTrue(decision.debug["utility_representation_role"])

    def test_training_loop_registers_pose_only_references_for_active_memory_modes(self):
        train_source = (Path(__file__).resolve().parents[1] / "train.py").read_text(
            encoding="utf-8-sig"
        )
        register_call = "runtime_gate.register_pose_only_reference"
        register_pos = train_source.index(register_call)
        guard_window = train_source[max(0, register_pos - 360) : register_pos]

        self.assertIn(
            "runtime_gate.direct_density_controller.is_pose_rep_active_memory",
            guard_window,
        )
        self.assertNotIn(
            "runtime_gate.direct_density_controller.is_pose_rep_active_memory_v1",
            guard_window,
        )

    def test_training_loop_routes_pose_safe_deferred_frames_to_tracking_only_path(self):
        train_source = (Path(__file__).resolve().parents[1] / "train.py").read_text(
            encoding="utf-8-sig"
        )

        self.assertIn("should_pose_safe_track_deferred", train_source)
        self.assertIn("should_pose_safe_preserve_baseline_keyframe", train_source)
        self.assertIn("pose_safe_tracking_only", train_source)
        self.assertIn("pose_safe_baseline_skeleton_forced", train_source)
        self.assertIn("and not pose_safe_tracking_only", train_source)
        self.assertIn("is_pose_only_baseline_repr_family", train_source)

    def test_pose_safe_memory_pose_choice_rejects_weaker_or_unused_memory(self):
        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_safe_streaming_memory_v1")
        )

        weaker = gate.choose_pose_safe_memory_pose(
            baseline_pose_success=True,
            memory_pose_success=True,
            baseline_debug={
                "num_2d3d_correspondences": 1000,
                "num_pnp_inliers": 120,
                "num_miniba_inliers": 96,
                "pnp_ref_keyframe_ids": [1, 2, 3],
                "miniba_ref_keyframe_ids": [1, 2, 3],
            },
            memory_debug={
                "num_2d3d_correspondences": 980,
                "num_pnp_inliers": 110,
                "num_miniba_inliers": 92,
                "pnp_ref_keyframe_ids": [1, 2, 3, 1001],
                "miniba_ref_keyframe_ids": [1, 2, 3, 1001],
            },
            pose_only_reference_ids=[1001],
        )

        self.assertFalse(weaker["use_memory_pose"])
        self.assertEqual(weaker["decision"], "baseline_pose")
        self.assertEqual(weaker["reason"], "memory_quality_not_better")

        unused = gate.choose_pose_safe_memory_pose(
            baseline_pose_success=True,
            memory_pose_success=True,
            baseline_debug={
                "num_2d3d_correspondences": 1000,
                "num_pnp_inliers": 120,
                "num_miniba_inliers": 96,
                "pnp_ref_keyframe_ids": [1, 2, 3],
                "miniba_ref_keyframe_ids": [1, 2, 3],
            },
            memory_debug={
                "num_2d3d_correspondences": 1000,
                "num_pnp_inliers": 150,
                "num_miniba_inliers": 120,
                "pnp_ref_keyframe_ids": [1, 2, 3],
                "miniba_ref_keyframe_ids": [1, 2, 3],
            },
            pose_only_reference_ids=[1001],
        )

        self.assertFalse(unused["use_memory_pose"])
        self.assertEqual(unused["reason"], "memory_reference_not_used")

    def test_pose_safe_memory_pose_choice_accepts_stronger_memory(self):
        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_safe_streaming_memory_v1")
        )

        decision = gate.choose_pose_safe_memory_pose(
            baseline_pose_success=True,
            memory_pose_success=True,
            baseline_debug={
                "num_2d3d_correspondences": 900,
                "num_pnp_inliers": 90,
                "num_miniba_inliers": 70,
                "pnp_ref_keyframe_ids": [1, 2, 3],
                "miniba_ref_keyframe_ids": [1, 2, 3],
            },
            memory_debug={
                "num_2d3d_correspondences": 980,
                "num_pnp_inliers": 126,
                "num_miniba_inliers": 92,
                "pnp_ref_keyframe_ids": [1, 2, 3, 1001],
                "miniba_ref_keyframe_ids": [1, 2, 3, 1001],
            },
            pose_only_reference_ids=[1001],
        )

        self.assertTrue(decision["use_memory_pose"])
        self.assertEqual(decision["decision"], "memory_pose")
        self.assertEqual(decision["reason"], "memory_quality_improved")
        self.assertGreater(decision["memory_score"], decision["baseline_score"])

    def test_training_loop_dual_checks_pose_safe_memory_candidates(self):
        train_source = (Path(__file__).resolve().parents[1] / "train.py").read_text(
            encoding="utf-8-sig"
        )

        self.assertIn("_snapshot_pose_match_state", train_source)
        self.assertIn("_restore_pose_match_state", train_source)
        self.assertIn("_snapshot_torch_rng_state", train_source)
        self.assertIn("_restore_torch_rng_state", train_source)
        self.assertIn("_clone_pose_support", train_source)
        self.assertIn("choose_pose_safe_memory_pose", train_source)
        self.assertIn("pose_safe_dual_candidate", train_source)
        self.assertIn("pose_safe_pose_rng_before", train_source)
        self.assertIn("pose_safe_pose_match_before", train_source)
        self.assertIn("baseline_rng_after", train_source)
        self.assertIn("memory_rng_after", train_source)
        self.assertIn("pose_safe_match_restored_on_hold", train_source)

    def test_training_loop_exports_pose_memory_geometry_context_to_viewpoint_scores(self):
        train_source = (Path(__file__).resolve().parents[1] / "train.py").read_text(
            encoding="utf-8-sig"
        )
        self.assertIn("pose_memory_reference_count", train_source)
        self.assertIn("pose_memory_pool_size", train_source)
        self.assertIn("pose_memory_candidate_pool_size", train_source)

    def test_pose_only_reference_pool_expires_caps_and_ranks_by_match_support(self):
        import torch

        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v1")
        )
        rt = torch.eye(4)
        for frame_id in range(100, 170):
            gate.register_pose_only_reference(
                frame_id=frame_id,
                info={"is_test": False, "image_name": str(frame_id)},
                desc_kpts=_pose_pool_desc(
                    support_count=600,
                    match_score=frame_id + 100,
                ),
                Rt=rt,
                density_debug={
                    "active_memory_context": True,
                    "active_memory_frame_role": "tracking_only",
                },
                pose_debug={"num_pnp_inliers": 100, "num_miniba_inliers": 100},
            )

        selected = gate.select_pose_only_references(
            frame_id=180,
            curr_desc_kpts=_pose_pool_desc(match_score=0),
            matcher=_PosePoolMatcher(),
        )
        expired = gate.select_pose_only_references(
            frame_id=600,
            curr_desc_kpts=_pose_pool_desc(match_score=0),
            matcher=_PosePoolMatcher(),
        )

        summary = gate.pose_only_reference_pool_summary()
        self.assertLessEqual(summary["pool_size"], 32)
        self.assertEqual([ref.info["_paper_aligned_source_frame_id"] for ref in selected], [169])
        self.assertEqual(expired, [])
        self.assertEqual(gate.pose_only_reference_pool_summary()["pool_size"], 0)

    def test_pose_initializer_records_incremental_pose_support_for_pose_only_pool(self):
        import torch

        initializer = object.__new__(PoseInitializer)
        match_indices = torch.tensor([2, 5, 8])
        pts3d = torch.ones(3, 3)
        pts_conf = torch.tensor([0.9, 0.8, 0.7])

        initializer._record_incremental_pose_support(match_indices, pts3d, pts_conf)

        support = initializer.last_incremental_pose_support
        self.assertEqual(support["match_indices"].tolist(), [2, 5, 8])
        self.assertEqual(support["pts3d"].shape, (3, 3))
        self.assertAlmostEqual(float(support["pts_conf"][0]), 0.9, places=5)
        self.assertAlmostEqual(float(support["pts_conf"][1]), 0.8, places=5)
        self.assertAlmostEqual(float(support["pts_conf"][2]), 0.7, places=5)
        pts3d[0, 0] = 42.0
        self.assertNotEqual(float(support["pts3d"][0, 0]), 42.0)

    def test_active_memory_v27_holds_high_pose_low_representation_utility(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v27")
        )

        decision = controller.decide(
            frame_id=520,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=88.0,
            local_density_before=84.0,
            local_window_density=84.0,
            local_window_keyframes=84,
            local_window_gap_max=2.0,
            local_window_gap_after_if_hold=2.0,
            keyframe_growth_recent=18,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=1,
            main_chain_gap_before=1.0,
            main_chain_gap_after_if_hold=2.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=18.0,
            displacement_threshold=30.0,
            num_matches=2100,
            min_num_inliers=100,
            pose_inliers=1500,
            novelty_proxy=0.12,
            current_keyframe_count=430,
            semantic_scores={
                "R_t": 0.06,
                "V_t": 0.22,
                "Q_t": 0.96,
                "C_t": 0.72,
                "B_R_t": 0.94,
                "recovery_pool_size": 1,
            },
            viewpoint_scores={
                "viewpoint_rotation_deg_window_max": 2.0,
                "inlier_grid_coverage": 0.90,
                "inlier_grid_entropy": 0.86,
                "support_concentration": 0.10,
                "anchor_health_score": 0.92,
                "new_view_event_score": 0.08,
            },
        )

        self.assertFalse(decision.finalize)
        self.assertEqual(decision.decision, "hold_low_representation_value")
        self.assertEqual(decision.debug["utility_frame_role"], "tracking_only")
        self.assertGreaterEqual(decision.debug["utility_pose_reference"], 0.58)
        self.assertLess(decision.debug["utility_representation"], 0.38)
        self.assertGreater(decision.debug["utility_compute_cost"], 0.45)
        self.assertEqual(decision.debug["value_hold_block_reason"], "utility_tracking_only_role")

    def test_active_memory_v27_finalizes_new_view_recovery_pressure_representation(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v27")
        )

        decision = controller.decide(
            frame_id=520,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=86.0,
            local_density_before=82.0,
            local_window_density=82.0,
            local_window_keyframes=82,
            local_window_gap_max=4.0,
            local_window_gap_after_if_hold=4.0,
            keyframe_growth_recent=16,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=4,
            main_chain_gap_before=3.0,
            main_chain_gap_after_if_hold=4.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=68.0,
            displacement_threshold=30.0,
            num_matches=1600,
            min_num_inliers=100,
            pose_inliers=980,
            novelty_proxy=0.86,
            current_keyframe_count=420,
            semantic_scores={
                "R_t": 0.08,
                "V_t": 0.92,
                "Q_t": 0.91,
                "C_t": 0.63,
                "B_R_t": 0.86,
                "recovery_pool_size": 9,
            },
            viewpoint_scores={
                "viewpoint_rotation_deg_window_max": 34.0,
                "inlier_grid_coverage": 0.56,
                "inlier_grid_entropy": 0.58,
                "support_concentration": 0.42,
                "anchor_health_score": 0.62,
                "new_view_event_score": 0.82,
            },
        )

        self.assertTrue(decision.finalize)
        self.assertEqual(decision.decision, "finalize_high_representation_value")
        self.assertEqual(decision.reason, "utility_representation_gain")
        self.assertEqual(decision.debug["utility_frame_role"], "representation")
        self.assertGreaterEqual(decision.debug["utility_representation"], 0.38)
        self.assertTrue(decision.debug["utility_recovery_pressure_context"])

    def test_active_memory_v28_blocks_tracking_only_on_hard_new_view_window(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v28")
        )

        decision = controller.decide(
            frame_id=620,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=86.0,
            local_density_before=82.0,
            local_window_density=82.0,
            local_window_keyframes=82,
            local_window_gap_max=5.0,
            local_window_gap_after_if_hold=5.0,
            keyframe_growth_recent=16,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=1,
            main_chain_gap_before=2.0,
            main_chain_gap_after_if_hold=3.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=20.0,
            displacement_threshold=30.0,
            num_matches=2200,
            min_num_inliers=100,
            pose_inliers=1600,
            novelty_proxy=0.10,
            current_keyframe_count=500,
            semantic_scores={
                "R_t": 0.05,
                "V_t": 0.25,
                "Q_t": 0.96,
                "C_t": 0.74,
                "B_R_t": 0.94,
                "recovery_pool_size": 1,
            },
            viewpoint_scores={
                "viewpoint_rotation_deg_window_max": 36.0,
                "inlier_grid_coverage": 0.58,
                "inlier_grid_entropy": 0.60,
                "support_concentration": 0.36,
                "anchor_health_score": 0.62,
                "new_view_event_score": 0.78,
            },
        )

        self.assertTrue(decision.finalize)
        self.assertEqual(decision.decision, "finalize_high_representation_value")
        self.assertEqual(decision.reason, "utility_hard_window_representation_guard")
        self.assertEqual(decision.debug["utility_frame_role"], "representation")
        self.assertTrue(decision.debug["utility_hard_window_guard"])
        self.assertFalse(decision.debug["utility_tracking_only_role"])

    def test_active_memory_v28_keeps_low_turn_redundant_frame_tracking_only(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v28")
        )

        decision = controller.decide(
            frame_id=620,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=88.0,
            local_density_before=84.0,
            local_window_density=84.0,
            local_window_keyframes=84,
            local_window_gap_max=2.0,
            local_window_gap_after_if_hold=2.0,
            keyframe_growth_recent=18,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=1,
            main_chain_gap_before=1.0,
            main_chain_gap_after_if_hold=2.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=18.0,
            displacement_threshold=30.0,
            num_matches=2100,
            min_num_inliers=100,
            pose_inliers=1500,
            novelty_proxy=0.12,
            current_keyframe_count=430,
            semantic_scores={
                "R_t": 0.06,
                "V_t": 0.22,
                "Q_t": 0.96,
                "C_t": 0.92,
                "B_R_t": 0.94,
                "recovery_pool_size": 1,
            },
            viewpoint_scores={
                "viewpoint_rotation_deg_window_max": 2.0,
                "inlier_grid_coverage": 0.95,
                "inlier_grid_entropy": 0.86,
                "support_concentration": 0.10,
                "anchor_health_score": 0.92,
                "new_view_event_score": 0.08,
            },
        )

        self.assertFalse(decision.finalize)
        self.assertEqual(decision.decision, "hold_low_representation_value")
        self.assertEqual(decision.debug["utility_frame_role"], "tracking_only")
        self.assertTrue(decision.debug["utility_tracking_safe_context"])
        self.assertFalse(decision.debug["utility_hard_window_guard"])

    def test_active_memory_v29_preserves_hard_window_representation_guard(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v29")
        )

        decision = controller.decide(
            frame_id=620,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=86.0,
            local_density_before=82.0,
            local_window_density=82.0,
            local_window_keyframes=82,
            local_window_gap_max=5.0,
            local_window_gap_after_if_hold=5.0,
            keyframe_growth_recent=16,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=1,
            main_chain_gap_before=2.0,
            main_chain_gap_after_if_hold=3.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=20.0,
            displacement_threshold=30.0,
            num_matches=2200,
            min_num_inliers=100,
            pose_inliers=1600,
            novelty_proxy=0.10,
            current_keyframe_count=500,
            semantic_scores={
                "R_t": 0.05,
                "V_t": 0.25,
                "Q_t": 0.96,
                "C_t": 0.74,
                "B_R_t": 0.94,
                "recovery_pool_size": 1,
            },
            viewpoint_scores={
                "viewpoint_rotation_deg_window_max": 36.0,
                "inlier_grid_coverage": 0.58,
                "inlier_grid_entropy": 0.60,
                "support_concentration": 0.36,
                "anchor_health_score": 0.62,
                "new_view_event_score": 0.78,
            },
        )

        self.assertTrue(decision.finalize)
        self.assertEqual(decision.decision, "finalize_high_representation_value")
        self.assertEqual(decision.reason, "utility_hard_window_representation_guard")
        self.assertTrue(decision.debug["utility_hard_window_guard"])
        self.assertFalse(decision.debug["utility_tracking_only_role"])

    def test_active_memory_v29_does_not_use_utility_tracking_hold(self):
        controller = DirectDensityController(
            _args(paper_aligned_direct_density_control="pose_rep_active_memory_v29")
        )

        decision = controller.decide(
            frame_id=520,
            runtime_action="direct_admit",
            baseline_should_add=True,
            is_test=False,
            is_bootstrap_phase=False,
            density_before=88.0,
            local_density_before=84.0,
            local_window_density=84.0,
            local_window_keyframes=84,
            local_window_gap_max=2.0,
            local_window_gap_after_if_hold=2.0,
            keyframe_growth_recent=18,
            baseline_relative_density=1.0,
            source_gap_to_last_keyframe=1,
            main_chain_gap_before=1.0,
            main_chain_gap_after_if_hold=2.0,
            anchor_changed=False,
            support_triggered=False,
            median_displacement=18.0,
            displacement_threshold=30.0,
            num_matches=2100,
            min_num_inliers=100,
            pose_inliers=1500,
            novelty_proxy=0.12,
            current_keyframe_count=430,
            semantic_scores={
                "R_t": 0.06,
                "V_t": 0.22,
                "Q_t": 0.96,
                "C_t": 0.72,
                "B_R_t": 0.94,
                "recovery_pool_size": 1,
            },
            viewpoint_scores={
                "viewpoint_rotation_deg_window_max": 2.0,
                "inlier_grid_coverage": 0.90,
                "inlier_grid_entropy": 0.86,
                "support_concentration": 0.10,
                "anchor_health_score": 0.92,
                "new_view_event_score": 0.08,
            },
        )

        self.assertFalse(decision.debug["utility_tracking_only_role"])
        self.assertNotEqual(decision.debug["value_hold_block_reason"], "utility_tracking_only_role")

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

    def test_runtime_gate_flushes_viewpoint_coverage_events(self):
        with tempfile.TemporaryDirectory() as td:
            trace_path = Path(td) / "semantic_trace.json"
            gate = PaperAlignedRuntimeGate(
                _args(paper_aligned_contract_trace_path=str(trace_path))
            )

            gate.append_viewpoint_coverage_event(
                {
                    "frame_id": 42,
                    "viewpoint_rotation_deg_to_last_keyframe": 37.5,
                    "inlier_grid_coverage": 0.5,
                    "anchor_health_score": 0.75,
                }
            )
            gate.flush_trace()

            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["viewpoint_coverage_events"]), 1)
            self.assertEqual(payload["viewpoint_coverage_events"][0]["frame_id"], 42)

    def test_viewpoint_coverage_helper_reports_rotation_and_grid_support(self):
        import torch

        from paper_aligned_policy.viewpoint_coverage import (
            build_viewpoint_coverage_event,
            grid_coverage,
            grid_entropy,
            rotation_degrees_between,
        )

        eye = torch.eye(4)
        rot_z_90 = torch.eye(4)
        rot_z_90[:3, :3] = torch.tensor(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        spread = torch.tensor(
            [
                [10.0, 10.0],
                [90.0, 10.0],
                [10.0, 90.0],
                [90.0, 90.0],
            ]
        )
        concentrated = torch.tensor(
            [
                [10.0, 10.0],
                [11.0, 11.0],
                [12.0, 12.0],
                [13.0, 13.0],
            ]
        )

        self.assertAlmostEqual(rotation_degrees_between(eye, eye), 0.0, places=4)
        self.assertAlmostEqual(rotation_degrees_between(eye, rot_z_90), 90.0, places=3)
        self.assertAlmostEqual(grid_coverage(spread, 100, 100, grid_size=4), 0.25, places=4)
        self.assertLess(grid_entropy(concentrated, 100, 100, grid_size=4), grid_entropy(spread, 100, 100, grid_size=4))

        event = build_viewpoint_coverage_event(
            frame_id=42,
            current_Rt=rot_z_90,
            last_keyframe_Rt=eye,
            active_anchor_Rt=eye,
            inlier_kpts=spread,
            image_width=100,
            image_height=100,
            pose_debug={"num_miniba_inliers": 80, "match_count_total": 100},
            active_anchor_keyframe_count=4,
            selected_reference_count=2,
        )
        for key in (
            "viewpoint_rotation_deg_to_last_keyframe",
            "viewpoint_rotation_deg_to_active_anchor",
            "inlier_grid_coverage",
            "inlier_grid_entropy",
            "support_concentration",
            "anchor_health_score",
            "new_view_event_score",
        ):
            self.assertIn(key, event)
        self.assertEqual(event["frame_id"], 42)
        self.assertGreater(event["new_view_event_score"], 0.0)


    def test_viewpoint_coverage_reports_windowed_rotation_when_keyframes_are_dense(self):
        import math
        import torch

        from paper_aligned_policy.viewpoint_coverage import (
            build_viewpoint_coverage_event,
            windowed_rotation_degrees,
        )

        def rot_z(degrees: float):
            radians = math.radians(degrees)
            rt = torch.eye(4)
            rt[:3, :3] = torch.tensor(
                [
                    [math.cos(radians), -math.sin(radians), 0.0],
                    [math.sin(radians), math.cos(radians), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            return rt

        current = rot_z(90.0)
        nearly_adjacent = rot_z(89.0)
        history = [
            (0, torch.eye(4)),
            (50, torch.eye(4)),
            (80, torch.eye(4)),
            (99, nearly_adjacent),
        ]

        windowed = windowed_rotation_degrees(
            current_Rt=current,
            pose_history=history,
            current_frame_id=100,
            windows=(20, 50, 100),
        )

        self.assertAlmostEqual(windowed["viewpoint_rotation_deg_window_20"], 90.0, places=3)
        self.assertAlmostEqual(windowed["viewpoint_rotation_deg_window_50"], 90.0, places=3)
        self.assertAlmostEqual(windowed["viewpoint_rotation_deg_window_100"], 90.0, places=3)
        self.assertEqual(windowed["viewpoint_rotation_window_source_20"], 80)
        self.assertGreater(windowed["viewpoint_rotation_deg_window_max"], 80.0)

        event = build_viewpoint_coverage_event(
            frame_id=100,
            current_Rt=current,
            last_keyframe_Rt=nearly_adjacent,
            active_anchor_Rt=nearly_adjacent,
            pose_history=history,
            inlier_kpts=torch.tensor([[10.0, 10.0], [90.0, 90.0]]),
            image_width=100,
            image_height=100,
            pose_debug={"num_miniba_inliers": 80, "match_count_total": 100},
            active_anchor_keyframe_count=4,
            selected_reference_count=2,
        )

        self.assertLess(event["viewpoint_rotation_deg_to_last_keyframe"], 2.0)
        self.assertGreater(event["viewpoint_rotation_deg_window_max"], 80.0)
        self.assertGreater(event["new_view_event_score"], 0.35)

    def test_train_records_viewpoint_coverage_as_ssm_only_trace(self):
        train_source = Path("train.py").read_text(encoding="utf-8")

        self.assertIn("build_viewpoint_coverage_event", train_source)
        self.assertIn("append_viewpoint_coverage_event", train_source)
        self.assertIn("viewpoint_coverage_event", train_source)
        self.assertIn("viewpoint_pose_history", train_source)
        self.assertIn("viewpoint_rotation_deg_window_max", train_source)
        self.assertNotIn("new_view_event_score >= ", train_source)


if __name__ == "__main__":
    unittest.main()
