import unittest
from pathlib import Path

from scene.pose_render_keyframe_sampling import (
    POSE_CONFIDENCE_TEMPORAL_SAMPLING_MODE,
    choose_pose_render_keyframe_id,
    pose_render_keyframe_sampling_weights,
)


def _stable_info():
    return {
        "is_test": False,
        "_paper_aligned_pose_render_coupling": {
            "pose_render_risk_score": 0.08,
            "utility_drift_risk": 0.05,
            "pose_support_score": 0.92,
            "match_support_score": 0.90,
        },
    }


def _risky_info():
    return {
        "is_test": False,
        "_paper_aligned_pose_render_coupling": {
            "pose_render_risk_score": 0.62,
            "pose_render_risk_high": True,
            "utility_drift_risk": 0.05,
            "pose_support_score": 0.91,
            "match_support_score": 0.88,
        },
    }


class _RecordingRng:
    def __init__(self):
        self.last_values = None
        self.last_p = None

    def choice(self, values, p=None):
        self.last_values = list(values)
        self.last_p = None if p is None else list(p)
        return values[0]


class PoseRenderKeyframeSamplingTests(unittest.TestCase):
    def test_disabled_sampling_returns_uniform_weights(self):
        weights, debug = pose_render_keyframe_sampling_weights(
            mode="off",
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_infos=[_stable_info(), _risky_info()],
        )

        self.assertEqual(weights, [1.0, 1.0])
        self.assertFalse(debug["applied"])
        self.assertEqual(debug["reason"], "off")

    def test_pose_confidence_temporal_downweights_risky_keyframes(self):
        weights, debug = pose_render_keyframe_sampling_weights(
            mode=POSE_CONFIDENCE_TEMPORAL_SAMPLING_MODE,
            direct_density_mode="pose_safe_streaming_memory_v1",
            keyframe_infos=[_stable_info(), _risky_info(), {}],
            min_weight=0.35,
            max_pose_risk=0.38,
        )

        self.assertTrue(debug["applied"])
        self.assertEqual(debug["reason"], "pose_confidence_temporal")
        self.assertEqual(weights[0], 1.0)
        self.assertGreater(weights[0], weights[1])
        self.assertAlmostEqual(weights[1], 0.35)
        self.assertEqual(weights[2], 1.0)
        self.assertEqual(debug["high_risk_candidates"], 1)

    def test_choose_keyframe_normalizes_weighted_distribution(self):
        rng = _RecordingRng()

        chosen, debug = choose_pose_render_keyframe_id(
            candidate_ids=[7, 8],
            keyframe_infos=[_stable_info(), _risky_info()],
            mode=POSE_CONFIDENCE_TEMPORAL_SAMPLING_MODE,
            direct_density_mode="pose_safe_streaming_memory_v1",
            rng=rng,
            min_weight=0.35,
            max_pose_risk=0.38,
        )

        self.assertEqual(chosen, 7)
        self.assertEqual(rng.last_values, [7, 8])
        self.assertIsNotNone(rng.last_p)
        self.assertAlmostEqual(sum(rng.last_p), 1.0)
        self.assertGreater(rng.last_p[0], rng.last_p[1])
        self.assertEqual(debug["chosen_keyframe_id"], 7)

    def test_uniform_weights_preserve_baseline_rng_choice(self):
        rng = _RecordingRng()

        chosen, debug = choose_pose_render_keyframe_id(
            candidate_ids=[7, 8],
            keyframe_infos=[_stable_info(), _stable_info()],
            mode=POSE_CONFIDENCE_TEMPORAL_SAMPLING_MODE,
            direct_density_mode="pose_safe_streaming_memory_v1",
            rng=rng,
            min_weight=0.35,
            max_pose_risk=0.38,
        )

        self.assertEqual(chosen, 7)
        self.assertIsNone(rng.last_p)
        self.assertFalse(debug["used_weighted_probabilities"])

    def test_scene_model_integrates_keyframe_sampling_in_random_branch(self):
        source = Path("scene/scene_model.py").read_text(encoding="utf-8")

        self.assertIn("choose_pose_render_keyframe_id", source)
        self.assertIn("self.pose_render_keyframe_sampling", source)
        self.assertIn("_record_pose_render_keyframe_sampling", source)

    def test_args_exposes_pose_render_keyframe_sampling(self):
        source = Path("args.py").read_text(encoding="utf-8")

        self.assertIn("--paper_aligned_pose_render_keyframe_sampling", source)
        self.assertIn("pose_confidence_temporal_v1", source)
        self.assertIn("--paper_aligned_pose_render_keyframe_sampling_min_weight", source)


if __name__ == "__main__":
    unittest.main()
