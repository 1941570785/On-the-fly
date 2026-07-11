import json
import tempfile
import unittest
from pathlib import Path

import torch

from scene.pose_initialization_risk import PoseInitializationRiskGate


def _rt(rotation_deg: float = 0.0) -> torch.Tensor:
    radians = torch.tensor(rotation_deg * 3.141592653589793 / 180.0)
    c = torch.cos(radians)
    s = torch.sin(radians)
    return torch.tensor(
        [
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )


def _strong_pose_debug() -> dict[str, int]:
    return {
        "match_count_total": 420,
        "num_2d3d_correspondences": 400,
        "num_pnp_inliers": 250,
        "num_miniba_inliers": 230,
    }


def _strong_viewpoint() -> dict[str, float | int]:
    return {
        "inlier_grid_coverage": 0.88,
        "inlier_grid_entropy": 0.90,
        "anchor_health_score": 0.82,
        "selected_reference_count": 3,
    }


class PoseInitializationRiskGateTests(unittest.TestCase):
    def test_default_absolute_floor_keeps_online_tail_detection_reachable(self):
        gate = PoseInitializationRiskGate(mode="isolate_v1")

        self.assertAlmostEqual(gate.absolute_threshold, 0.10)
        self.assertAlmostEqual(gate.adaptive_sigma, 2.0)

    def test_strong_pose_support_has_low_risk(self):
        gate = PoseInitializationRiskGate(mode="isolate_v1", warmup=0)

        decision = gate.evaluate(
            frame_id=10,
            pose_debug=_strong_pose_debug(),
            viewpoint_scores=_strong_viewpoint(),
            min_num_inliers=100,
            recent_pose_fail_rate=0.0,
            current_Rt=_rt(2.0),
            pose_history=[(8, _rt(0.0)), (9, _rt(1.0))],
            baseline_selected=True,
            is_test=False,
            is_bootstrap=False,
        )

        self.assertLess(decision["pose_uncertainty"], 0.25)
        self.assertLess(decision["state_support_gap"], 0.30)
        self.assertLess(decision["risk_score"], 0.30)
        self.assertFalse(decision["isolated"])
        self.assertEqual(decision["decision"], "admit")

    def test_jointly_weak_signals_isolate_after_warmup(self):
        gate = PoseInitializationRiskGate(
            mode="isolate_v1",
            absolute_threshold=0.50,
            warmup=3,
        )
        history = [(1, _rt(0.0)), (2, _rt(1.0)), (3, _rt(2.0))]
        for frame_id in range(4, 7):
            gate.evaluate(
                frame_id=frame_id,
                pose_debug=_strong_pose_debug(),
                viewpoint_scores=_strong_viewpoint(),
                min_num_inliers=100,
                recent_pose_fail_rate=0.0,
                current_Rt=_rt(float(frame_id - 1)),
                pose_history=history,
                baseline_selected=True,
                is_test=False,
                is_bootstrap=False,
            )

        decision = gate.evaluate(
            frame_id=7,
            pose_debug={
                "match_count_total": 500,
                "num_2d3d_correspondences": 480,
                "num_pnp_inliers": 70,
                "num_miniba_inliers": 45,
            },
            viewpoint_scores={
                "inlier_grid_coverage": 0.25,
                "inlier_grid_entropy": 0.35,
                "anchor_health_score": 0.20,
                "selected_reference_count": 1,
            },
            min_num_inliers=100,
            recent_pose_fail_rate=0.60,
            current_Rt=_rt(35.0),
            pose_history=history,
            baseline_selected=True,
            is_test=False,
            is_bootstrap=False,
        )

        self.assertGreater(decision["pose_uncertainty"], 0.65)
        self.assertGreater(decision["state_support_gap"], 0.60)
        self.assertGreater(decision["temporal_degradation"], 0.50)
        self.assertGreaterEqual(decision["risk_score"], decision["risk_threshold"])
        self.assertTrue(decision["isolated"])
        self.assertEqual(decision["decision"], "isolate")

    def test_moderate_joint_tail_isolated_relative_to_stable_history(self):
        gate = PoseInitializationRiskGate(
            mode="isolate_v1",
            absolute_threshold=0.10,
            adaptive_sigma=2.0,
            warmup=3,
        )
        history = [(1, _rt(0.0)), (2, _rt(1.0)), (3, _rt(2.0))]
        for frame_id in range(4, 7):
            gate.evaluate(
                frame_id=frame_id,
                pose_debug=_strong_pose_debug(),
                viewpoint_scores=_strong_viewpoint(),
                min_num_inliers=100,
                recent_pose_fail_rate=0.0,
                current_Rt=_rt(float(frame_id - 1)),
                pose_history=history,
                baseline_selected=True,
                is_test=False,
                is_bootstrap=False,
            )

        decision = gate.evaluate(
            frame_id=7,
            pose_debug={
                "match_count_total": 400,
                "num_2d3d_correspondences": 400,
                "num_pnp_inliers": 150,
                "num_miniba_inliers": 110,
            },
            viewpoint_scores={
                "inlier_grid_coverage": 0.55,
                "inlier_grid_entropy": 0.65,
                "anchor_health_score": 0.45,
                "selected_reference_count": 1,
            },
            min_num_inliers=100,
            recent_pose_fail_rate=0.0,
            current_Rt=_rt(4.0),
            pose_history=history,
            baseline_selected=True,
            is_test=False,
            is_bootstrap=False,
        )

        self.assertLess(decision["pose_uncertainty"], 0.45)
        self.assertTrue(decision["multi_signal_risk"])
        self.assertTrue(decision["isolated"])

    def test_cooldown_keeps_a_risk_burst_from_repeated_isolation(self):
        gate = PoseInitializationRiskGate(
            mode="isolate_v1",
            absolute_threshold=0.10,
            adaptive_sigma=0.0,
            warmup=0,
            cooldown_frames=12,
        )
        weak = {
            "match_count_total": 500,
            "num_2d3d_correspondences": 480,
            "num_pnp_inliers": 70,
            "num_miniba_inliers": 45,
        }
        weak_viewpoint = {
            "inlier_grid_coverage": 0.25,
            "inlier_grid_entropy": 0.35,
            "anchor_health_score": 0.20,
            "selected_reference_count": 1,
        }
        kwargs = {
            "pose_debug": weak,
            "viewpoint_scores": weak_viewpoint,
            "min_num_inliers": 100,
            "recent_pose_fail_rate": 0.60,
            "current_Rt": _rt(35.0),
            "pose_history": [(8, _rt(0.0)), (9, _rt(1.0))],
            "baseline_selected": True,
            "is_test": False,
            "is_bootstrap": False,
        }

        first = gate.evaluate(frame_id=10, **kwargs)
        second = gate.evaluate(frame_id=11, **kwargs)
        after_cooldown = gate.evaluate(frame_id=22, **kwargs)

        self.assertTrue(first["isolated"])
        self.assertFalse(second["isolated"])
        self.assertTrue(second["cooldown_active"])
        self.assertEqual(second["decision"], "cooldown_admit")
        self.assertTrue(after_cooldown["isolated"])

    def test_observe_mode_records_risk_without_isolating(self):
        gate = PoseInitializationRiskGate(
            mode="observe_v1",
            absolute_threshold=0.10,
            warmup=0,
        )

        decision = gate.evaluate(
            frame_id=11,
            pose_debug={},
            viewpoint_scores={},
            min_num_inliers=100,
            recent_pose_fail_rate=1.0,
            current_Rt=_rt(80.0),
            pose_history=[(9, _rt(0.0)), (10, _rt(1.0))],
            baseline_selected=True,
            is_test=False,
            is_bootstrap=False,
        )

        self.assertGreater(decision["risk_score"], 0.50)
        self.assertFalse(decision["isolated"])
        self.assertEqual(decision["decision"], "observe")

    def test_test_and_nonbaseline_frames_are_never_isolated(self):
        gate = PoseInitializationRiskGate(
            mode="isolate_v1",
            absolute_threshold=0.10,
            warmup=0,
        )
        kwargs = {
            "pose_debug": {},
            "viewpoint_scores": {},
            "min_num_inliers": 100,
            "recent_pose_fail_rate": 1.0,
            "current_Rt": _rt(80.0),
            "pose_history": [(1, _rt(0.0)), (2, _rt(1.0))],
            "is_bootstrap": False,
        }

        test_decision = gate.evaluate(
            frame_id=3,
            baseline_selected=True,
            is_test=True,
            **kwargs,
        )
        nonbaseline_decision = gate.evaluate(
            frame_id=4,
            baseline_selected=False,
            is_test=False,
            **kwargs,
        )

        self.assertFalse(test_decision["isolated"])
        self.assertEqual(test_decision["decision"], "bypass_test")
        self.assertFalse(nonbaseline_decision["isolated"])
        self.assertEqual(nonbaseline_decision["decision"], "bypass_not_selected")

    def test_flush_writes_reproducible_summary_and_events(self):
        gate = PoseInitializationRiskGate(mode="observe_v1", warmup=0)
        gate.evaluate(
            frame_id=5,
            pose_debug=_strong_pose_debug(),
            viewpoint_scores=_strong_viewpoint(),
            min_num_inliers=100,
            recent_pose_fail_rate=0.0,
            current_Rt=_rt(2.0),
            pose_history=[(3, _rt(0.0)), (4, _rt(1.0))],
            baseline_selected=True,
            is_test=False,
            is_bootstrap=False,
        )

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pose_initialization_risk_trace.json"
            gate.flush(path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["summary"]["events"], 1)
        self.assertEqual(payload["summary"]["isolated"], 0)
        self.assertEqual(payload["events"][0]["frame_id"], 5)
        self.assertEqual(payload["config"]["mode"], "observe_v1")


if __name__ == "__main__":
    unittest.main()
