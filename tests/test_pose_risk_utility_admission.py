import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from scene.pose_risk_utility_admission import (
    PoseRiskUtilityAdmissionGate,
    filter_pose_reference_indices,
    pose_reference_quarantine_enabled,
    pose_review_candidate,
    pose_review_acceptance,
    representation_utility_score,
    restore_optimizer_parameter_state,
    restore_optimizer_parameter_learning_rates,
    scale_optimizer_parameter_learning_rates,
    snapshot_optimizer_parameter_state,
)


def risk_event(**overrides):
    event = {
        "eligible": True,
        "warmed_up": True,
        "risk_score": 0.42,
        "risk_threshold": 0.18,
        "multi_signal_risk": True,
        "severe_pose_risk": False,
        "pose_uncertainty": 0.24,
    }
    event.update(overrides)
    return event


class PoseRiskUtilityAdmissionTests(unittest.TestCase):
    def test_pose_review_requires_configured_relative_loss_improvement(self):
        accepted, reason = pose_review_acceptance(
            start_loss=1.0,
            end_loss=0.995,
            rotation_delta_deg=0.1,
            translation_delta=0.001,
            max_rotation_delta_deg=1.5,
            max_translation_delta=0.05,
            min_relative_loss_improvement=0.01,
        )

        self.assertFalse(accepted)
        self.assertEqual(reason, "loss_improvement_too_small")

    def test_pose_review_accepts_sufficient_relative_loss_improvement(self):
        accepted, reason = pose_review_acceptance(
            start_loss=1.0,
            end_loss=0.98,
            rotation_delta_deg=0.1,
            translation_delta=0.001,
            max_rotation_delta_deg=1.5,
            max_translation_delta=0.05,
            min_relative_loss_improvement=0.01,
        )

        self.assertTrue(accepted)
        self.assertEqual(reason, "accepted")

    def test_pose_review_rejects_validation_support_loss(self):
        accepted, reason = pose_review_acceptance(
            start_loss=1.0,
            end_loss=0.8,
            rotation_delta_deg=0.1,
            translation_delta=0.001,
            max_rotation_delta_deg=1.5,
            max_translation_delta=0.05,
            validation_support_ratio=0.85,
            min_validation_support_ratio=0.95,
        )

        self.assertFalse(accepted)
        self.assertEqual(reason, "validation_support_lost")

    def test_pose_review_rejects_non_finite_validation_loss(self):
        accepted, reason = pose_review_acceptance(
            start_loss=1.0,
            end_loss=float("nan"),
            rotation_delta_deg=0.1,
            translation_delta=0.001,
            max_rotation_delta_deg=1.5,
            max_translation_delta=0.05,
        )

        self.assertFalse(accepted)
        self.assertEqual(reason, "non_finite_loss")

    def test_optimizer_snapshot_restores_pose_values_and_moments(self):
        optimizer = SimpleNamespace(
            params={
                "rW2C": {
                    "val": torch.tensor([1.0]),
                    "exp_avg": torch.tensor([2.0]),
                    "exp_avg_sq": torch.tensor([3.0]),
                },
                "exposure": {
                    "val": torch.tensor([4.0]),
                    "exp_avg": torch.tensor([5.0]),
                    "exp_avg_sq": torch.tensor([6.0]),
                },
            }
        )
        snapshot = snapshot_optimizer_parameter_state(
            optimizer,
            {"rW2C", "tW2C"},
        )
        optimizer.params["rW2C"]["val"].fill_(11.0)
        optimizer.params["rW2C"]["exp_avg"].fill_(12.0)
        optimizer.params["rW2C"]["exp_avg_sq"].fill_(13.0)

        restore_optimizer_parameter_state(optimizer, snapshot)

        self.assertEqual(optimizer.params["rW2C"]["val"].item(), 1.0)
        self.assertEqual(optimizer.params["rW2C"]["exp_avg"].item(), 2.0)
        self.assertEqual(optimizer.params["rW2C"]["exp_avg_sq"].item(), 3.0)
        self.assertNotIn("exposure", snapshot)

    def test_optimizer_snapshot_can_restore_moments_without_pose_value(self):
        optimizer = SimpleNamespace(
            params={
                "rW2C": {
                    "val": torch.tensor([1.0]),
                    "exp_avg": torch.tensor([2.0]),
                    "exp_avg_sq": torch.tensor([3.0]),
                },
            }
        )
        snapshot = snapshot_optimizer_parameter_state(optimizer, {"rW2C"})
        optimizer.params["rW2C"]["val"].fill_(11.0)
        optimizer.params["rW2C"]["exp_avg"].fill_(12.0)
        optimizer.params["rW2C"]["exp_avg_sq"].fill_(13.0)

        restore_optimizer_parameter_state(
            optimizer,
            snapshot,
            restore_values=False,
        )

        self.assertEqual(optimizer.params["rW2C"]["val"].item(), 11.0)
        self.assertEqual(optimizer.params["rW2C"]["exp_avg"].item(), 2.0)
        self.assertEqual(optimizer.params["rW2C"]["exp_avg_sq"].item(), 3.0)

    def test_pose_review_learning_rate_scale_is_temporary(self):
        optimizer = SimpleNamespace(
            params={
                "rW2C": {"lr": 1e-4},
                "tW2C": {"lr": 2e-4},
                "exposure": {"lr": 3e-4},
            }
        )

        snapshot = scale_optimizer_parameter_learning_rates(
            optimizer,
            {"rW2C", "tW2C"},
            scale=5.0,
        )

        self.assertAlmostEqual(optimizer.params["rW2C"]["lr"], 5e-4)
        self.assertAlmostEqual(optimizer.params["tW2C"]["lr"], 1e-3)
        self.assertAlmostEqual(optimizer.params["exposure"]["lr"], 3e-4)
        restore_optimizer_parameter_learning_rates(optimizer, snapshot)
        self.assertAlmostEqual(optimizer.params["rW2C"]["lr"], 1e-4)
        self.assertAlmostEqual(optimizer.params["tW2C"]["lr"], 2e-4)

    def test_representation_utility_combines_bounded_signals(self):
        score, debug = representation_utility_score(
            coverage_deficit=0.40,
            residual_selectivity=1.80,
            new_view_event_score=0.50,
            selectivity_reference=1.80,
        )

        self.assertAlmostEqual(score, 0.60)
        self.assertAlmostEqual(debug["selectivity_value"], 1.0)

    def test_non_risk_frame_preserves_v31_admission(self):
        gate = PoseRiskUtilityAdmissionGate(mode="active_v1")
        decision = gate.evaluate(
            frame_id=10,
            risk_event=risk_event(risk_score=0.10, risk_threshold=0.18),
            render_probe=None,
            baseline_selected=True,
            is_test=False,
            is_bootstrap=False,
        )

        self.assertEqual(decision["decision"], "admit")
        self.assertFalse(decision["risk_candidate"])
        self.assertFalse(decision["probe_required"])

    def test_high_risk_high_utility_frame_is_reviewed_and_retained(self):
        gate = PoseRiskUtilityAdmissionGate(
            mode="active_v1", utility_threshold=0.25
        )
        decision = gate.evaluate(
            frame_id=20,
            risk_event=risk_event(),
            render_probe={
                "coverage_deficit": 0.35,
                "residual_selectivity": 2.0,
                "new_view_event_score": 0.55,
            },
            baseline_selected=True,
            is_test=False,
            is_bootstrap=False,
        )

        self.assertEqual(decision["decision"], "review_admit")
        self.assertTrue(decision["review"])
        self.assertFalse(decision["isolated"])
        self.assertGreaterEqual(decision["utility_score"], 0.25)

    def test_high_risk_low_utility_frame_is_isolated(self):
        gate = PoseRiskUtilityAdmissionGate(
            mode="active_v1", utility_threshold=0.25
        )
        decision = gate.evaluate(
            frame_id=30,
            risk_event=risk_event(),
            render_probe={
                "coverage_deficit": 0.02,
                "residual_selectivity": 0.10,
                "new_view_event_score": 0.05,
            },
            baseline_selected=True,
            is_test=False,
            is_bootstrap=False,
        )

        self.assertEqual(decision["decision"], "isolate_low_utility")
        self.assertTrue(decision["isolated"])
        self.assertFalse(decision["review"])

    def test_moderate_risk_low_utility_frame_keeps_baseline_admission(self):
        gate = PoseRiskUtilityAdmissionGate(
            mode="active_v1",
            utility_threshold=0.24,
            isolation_risk_margin=0.04,
        )
        decision = gate.evaluate(
            frame_id=31,
            risk_event=risk_event(risk_score=0.12, risk_threshold=0.10),
            render_probe={
                "coverage_deficit": 0.0,
                "residual_selectivity": 0.10,
                "new_view_event_score": 0.05,
            },
            baseline_selected=True,
            is_test=False,
            is_bootstrap=False,
        )

        self.assertEqual(decision["decision"], "admit_conservative")
        self.assertFalse(decision["isolated"])

    def test_isolation_cooldown_prevents_burst_frame_removal(self):
        gate = PoseRiskUtilityAdmissionGate(
            mode="active_v1",
            utility_threshold=0.24,
            isolation_risk_margin=0.04,
            isolation_cooldown_frames=24,
        )
        probe = {
            "coverage_deficit": 0.0,
            "residual_selectivity": 0.10,
            "new_view_event_score": 0.05,
        }
        first = gate.evaluate(
            frame_id=100,
            risk_event=risk_event(risk_score=0.16, risk_threshold=0.10),
            render_probe=probe,
            baseline_selected=True,
            is_test=False,
            is_bootstrap=False,
        )
        second = gate.evaluate(
            frame_id=110,
            risk_event=risk_event(risk_score=0.17, risk_threshold=0.10),
            render_probe=probe,
            baseline_selected=True,
            is_test=False,
            is_bootstrap=False,
        )

        self.assertEqual(first["decision"], "isolate_low_utility")
        self.assertEqual(second["decision"], "cooldown_admit")
        self.assertFalse(second["isolated"])

    def test_observe_mode_records_suggestion_without_changing_admission(self):
        gate = PoseRiskUtilityAdmissionGate(
            mode="observe_v1", utility_threshold=0.25
        )
        decision = gate.evaluate(
            frame_id=40,
            risk_event=risk_event(),
            render_probe={
                "coverage_deficit": 0.01,
                "residual_selectivity": 0.10,
                "new_view_event_score": 0.02,
            },
            baseline_selected=True,
            is_test=False,
            is_bootstrap=False,
        )

        self.assertEqual(decision["decision"], "observe")
        self.assertEqual(decision["suggested_decision"], "isolate_low_utility")
        self.assertFalse(decision["isolated"])

    def test_test_and_bootstrap_frames_bypass_joint_policy(self):
        gate = PoseRiskUtilityAdmissionGate(mode="active_v1")
        test_decision = gate.evaluate(
            frame_id=50,
            risk_event=risk_event(),
            render_probe={"coverage_deficit": 0.0},
            baseline_selected=True,
            is_test=True,
            is_bootstrap=False,
        )
        bootstrap_decision = gate.evaluate(
            frame_id=51,
            risk_event=risk_event(),
            render_probe={"coverage_deficit": 0.0},
            baseline_selected=True,
            is_test=False,
            is_bootstrap=True,
        )

        self.assertEqual(test_decision["decision"], "bypass_test")
        self.assertEqual(bootstrap_decision["decision"], "bypass_bootstrap")
        self.assertFalse(test_decision["isolated"])
        self.assertFalse(bootstrap_decision["isolated"])

    def test_opt_in_routes_verification_candidate_to_review(self):
        gate = PoseRiskUtilityAdmissionGate(
            mode="active_v1",
            utility_threshold=0.0,
            use_verification_candidates=True,
        )
        decision = gate.evaluate(
            frame_id=52,
            risk_event=risk_event(
                risk_score=0.10,
                risk_threshold=0.18,
                verification_candidate=True,
            ),
            render_probe={
                "coverage_deficit": 0.0,
                "residual_selectivity": 0.0,
                "new_view_event_score": 0.0,
            },
            baseline_selected=True,
            is_test=False,
            is_bootstrap=False,
        )

        self.assertTrue(decision["risk_candidate"])
        self.assertTrue(decision["verification_candidate_routed"])
        self.assertEqual(decision["decision"], "review_admit")

    def test_review_candidate_helper_matches_gate_routing(self):
        candidate_event = risk_event(
            risk_score=0.10,
            risk_threshold=0.18,
            multi_signal_risk=False,
            verification_candidate=True,
        )

        self.assertFalse(pose_review_candidate(candidate_event))
        self.assertTrue(
            pose_review_candidate(
                candidate_event,
                use_verification_candidates=True,
            )
        )
        self.assertFalse(
            pose_review_candidate(
                {**candidate_event, "warmed_up": False},
                use_verification_candidates=True,
            )
        )

    def test_opt_in_allows_pose_only_review_for_test_candidate(self):
        gate = PoseRiskUtilityAdmissionGate(
            mode="active_v1",
            utility_threshold=0.0,
            use_verification_candidates=True,
            review_test_candidates=True,
        )
        decision = gate.evaluate(
            frame_id=53,
            risk_event=risk_event(verification_candidate=True),
            render_probe={
                "coverage_deficit": 0.0,
                "residual_selectivity": 0.0,
                "new_view_event_score": 0.0,
            },
            baseline_selected=True,
            is_test=True,
            is_bootstrap=False,
        )

        self.assertEqual(decision["decision"], "review_admit")
        self.assertTrue(decision["review"])
        self.assertFalse(decision["isolated"])

    def test_flush_records_policy_summary(self):
        gate = PoseRiskUtilityAdmissionGate(mode="active_v1")
        gate.evaluate(
            frame_id=60,
            risk_event=risk_event(),
            render_probe={
                "coverage_deficit": 0.30,
                "residual_selectivity": 2.0,
                "new_view_event_score": 0.50,
            },
            baseline_selected=True,
            is_test=False,
            is_bootstrap=False,
        )

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "trace.json"
            gate.flush(path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["summary"]["review_admit"], 1)
        self.assertEqual(payload["config"]["mode"], "active_v1")

    def test_trace_preserves_pose_identity_for_common_frame_evaluation(self):
        gate = PoseRiskUtilityAdmissionGate(mode="active_v1")
        event = risk_event(
            image_name="000021.jpg",
            source_frame_id=21,
            estimated_Rt=[[1.0, 0.0], [0.0, 1.0]],
            gt_Rt=[[1.0, 0.0], [0.0, 1.0]],
        )
        decision = gate.evaluate(
            frame_id=21,
            risk_event=event,
            render_probe={
                "coverage_deficit": 0.30,
                "residual_selectivity": 2.0,
                "new_view_event_score": 0.50,
            },
            baseline_selected=True,
            is_test=False,
            is_bootstrap=False,
        )

        self.assertEqual(decision["image_name"], "000021.jpg")
        self.assertEqual(decision["source_frame_id"], 21)
        self.assertEqual(decision["estimated_Rt"], event["estimated_Rt"])
        self.assertEqual(decision["gt_Rt"], event["gt_Rt"])

    def test_pose_quarantine_retains_render_frame_without_future_pose_use(self):
        gate = PoseRiskUtilityAdmissionGate(
            mode="pose_quarantine_v1",
            utility_threshold=0.24,
            quarantine_risk_margin=0.08,
            quarantine_cooldown_frames=64,
        )
        decision = gate.evaluate(
            frame_id=70,
            risk_event=risk_event(),
            render_probe={
                "coverage_deficit": 0.0,
                "residual_selectivity": 1.8,
                "new_view_event_score": 0.50,
            },
            baseline_selected=True,
            is_test=False,
            is_bootstrap=False,
        )

        self.assertEqual(decision["decision"], "render_admit_pose_quarantine")
        self.assertTrue(decision["pose_reference_quarantined"])
        self.assertFalse(decision["isolated"])
        self.assertFalse(decision["review"])

    def test_pose_quarantine_requires_extreme_risk_and_has_cooldown(self):
        gate = PoseRiskUtilityAdmissionGate(
            mode="pose_quarantine_v1",
            utility_threshold=0.24,
            quarantine_risk_margin=0.08,
            quarantine_cooldown_frames=64,
        )
        probe = {
            "coverage_deficit": 0.0,
            "residual_selectivity": 1.8,
            "new_view_event_score": 0.50,
        }
        moderate = gate.evaluate(
            frame_id=80,
            risk_event=risk_event(risk_score=0.15, risk_threshold=0.10),
            render_probe=probe,
            baseline_selected=True,
            is_test=False,
            is_bootstrap=False,
        )
        extreme = gate.evaluate(
            frame_id=100,
            risk_event=risk_event(risk_score=0.20, risk_threshold=0.10),
            render_probe=probe,
            baseline_selected=True,
            is_test=False,
            is_bootstrap=False,
        )
        cooldown = gate.evaluate(
            frame_id=120,
            risk_event=risk_event(risk_score=0.21, risk_threshold=0.10),
            render_probe=probe,
            baseline_selected=True,
            is_test=False,
            is_bootstrap=False,
        )

        self.assertEqual(moderate["decision"], "admit_conservative")
        self.assertEqual(extreme["decision"], "render_admit_pose_quarantine")
        self.assertEqual(cooldown["decision"], "quarantine_cooldown_admit")

    def test_utility_quarantine_preserves_high_utility_pose_reference(self):
        gate = PoseRiskUtilityAdmissionGate(
            mode="pose_quarantine_utility_v1",
            utility_threshold=0.24,
            quarantine_risk_margin=0.04,
        )

        decision = gate.evaluate(
            frame_id=100,
            risk_event=risk_event(risk_score=0.16, risk_threshold=0.10),
            render_probe={
                "coverage_deficit": 0.30,
                "residual_selectivity": 1.8,
                "new_view_event_score": 0.50,
            },
            baseline_selected=True,
            is_test=False,
            is_bootstrap=False,
        )

        self.assertEqual(decision["decision"], "render_admit_high_utility")
        self.assertFalse(decision["pose_reference_quarantined"])

    def test_utility_quarantine_excludes_low_utility_high_risk_pose_reference(self):
        gate = PoseRiskUtilityAdmissionGate(
            mode="pose_quarantine_utility_v1",
            utility_threshold=0.24,
            quarantine_risk_margin=0.04,
        )

        decision = gate.evaluate(
            frame_id=100,
            risk_event=risk_event(risk_score=0.16, risk_threshold=0.10),
            render_probe={
                "coverage_deficit": 0.0,
                "residual_selectivity": 0.0,
                "new_view_event_score": 0.10,
            },
            baseline_selected=True,
            is_test=False,
            is_bootstrap=False,
        )

        self.assertEqual(
            decision["decision"],
            "conservative_render_pose_quarantine",
        )
        self.assertTrue(decision["pose_reference_quarantined"])
        self.assertFalse(decision["isolated"])

    def test_pose_reference_quarantine_modes_are_explicit(self):
        self.assertTrue(pose_reference_quarantine_enabled("pose_quarantine_v1"))
        self.assertTrue(
            pose_reference_quarantine_enabled("pose_quarantine_utility_v1")
        )
        self.assertTrue(
            pose_reference_quarantine_enabled("pose_quarantine_severe_v1")
        )
        self.assertFalse(pose_reference_quarantine_enabled("active_v1"))

    def test_severe_quarantine_routes_only_existing_severe_pose_risk(self):
        gate = PoseRiskUtilityAdmissionGate(
            mode="pose_quarantine_severe_v1",
            quarantine_cooldown_frames=12,
            use_verification_candidates=True,
        )
        verification_event = risk_event(
            risk_score=0.08,
            risk_threshold=0.10,
            multi_signal_risk=False,
            verification_candidate=True,
        )

        moderate = gate.evaluate(
            frame_id=100,
            risk_event={
                **verification_event,
                "severe_pose_risk": False,
                "pose_uncertainty": 0.29,
            },
            render_probe={},
            baseline_selected=True,
            is_test=False,
            is_bootstrap=False,
        )
        severe = gate.evaluate(
            frame_id=120,
            risk_event={
                **verification_event,
                "severe_pose_risk": True,
                "pose_uncertainty": 0.30,
            },
            render_probe={},
            baseline_selected=True,
            is_test=False,
            is_bootstrap=False,
        )

        self.assertEqual(moderate["decision"], "admit_conservative")
        self.assertFalse(moderate["pose_reference_quarantined"])
        self.assertEqual(
            severe["decision"],
            "conservative_render_pose_quarantine",
        )
        self.assertTrue(severe["pose_reference_quarantined"])

    def test_pose_reference_filter_keeps_render_keyframes_but_excludes_quarantine(self):
        class Frame:
            def __init__(self, quarantined):
                self.info = {"_pose_reference_quarantined": quarantined}

        keyframes = [Frame(False), Frame(True), Frame(False)]

        self.assertEqual(
            filter_pose_reference_indices(keyframes, [0, 1, 2], enabled=True),
            [0, 2],
        )
        self.assertEqual(
            filter_pose_reference_indices(keyframes, [1], enabled=True),
            [1],
        )
        self.assertEqual(
            filter_pose_reference_indices(keyframes, [0, 1], enabled=False),
            [0, 1],
        )


if __name__ == "__main__":
    unittest.main()
