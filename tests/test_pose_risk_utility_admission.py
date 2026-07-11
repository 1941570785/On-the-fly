import json
import tempfile
import unittest
from pathlib import Path

from scene.pose_risk_utility_admission import (
    PoseRiskUtilityAdmissionGate,
    filter_pose_reference_indices,
    representation_utility_score,
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
            mode="pose_quarantine_v1", utility_threshold=0.24
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
