import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.evaluate_pose_verification_a_ablation import (
    METHODS,
    evaluate_a_trace,
    evaluate_scene,
)


def _pose(tx=0.0, ty=0.0, tz=0.0):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [tx, ty, tz]
    return pose


class PoseVerificationAblationEvaluatorTest(unittest.TestCase):
    def test_scene_uses_one_common_frame_set_for_all_variants(self):
        reference = {str(i): _pose(float(i), float(i % 2), 0.0) for i in range(5)}
        methods = {
            "v31_control": dict(reference),
            "a_observe": dict(reference),
            "a_full": {key: value for key, value in reference.items() if key != "4"},
        }
        result = evaluate_scene(reference, methods)
        self.assertEqual(result["common_frames"], 4)
        self.assertEqual(set(result["methods"]), set(METHODS))
        for metrics in result["methods"].values():
            self.assertAlmostEqual(metrics["ape_trans_rmse"], 0.0, places=8)

    def test_trace_reports_detection_refinement_safety_and_overhead(self):
        reference = {str(i): _pose(float(i), float(i % 2), 0.1 * i) for i in range(5)}
        trajectory = dict(reference)
        events = []
        for i in range(4):
            initial = reference[str(i)].copy()
            final = initial.copy()
            candidate = i == 2
            attempted = candidate
            accepted = candidate
            if candidate:
                initial[0, 3] += 0.40
                final[0, 3] += 0.05
            else:
                initial[0, 3] += 0.01
                final[0, 3] += 0.01
            events.append(
                {
                    "image_name": f"{i}.png",
                    "risk_score": 0.9 if candidate else 0.1,
                    "verification_candidate": candidate,
                    "verification_attempted": attempted,
                    "verification_accepted": accepted,
                    "initial_estimated_Rt": initial.tolist(),
                    "estimated_Rt": final.tolist(),
                    "verification": {
                        "pre_reprojection_mean": 3.0,
                        "post_reprojection_mean": 2.0,
                        "pre_reprojection_median": 2.8,
                        "post_reprojection_median": 1.9,
                        "pre_reprojection_p90": 4.0,
                        "post_reprojection_p90": 3.5,
                        "runtime_seconds": 0.02,
                    },
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.json"
            trace_path.write_text(json.dumps({"events": events}), encoding="utf-8")
            result = evaluate_a_trace(
                trace_path,
                reference,
                trajectory,
                list(reference),
            )
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["attempt_count"], 1)
        self.assertEqual(result["accepted_count"], 1)
        self.assertEqual(result["accepted_translation_improved_count"], 1)
        self.assertLess(result["accepted_final_translation_error_mean"], result["accepted_initial_translation_error_mean"])
        self.assertGreater(result["candidate_translation_enrichment"], 1.0)
        self.assertAlmostEqual(result["reprojection_mean_reduction"], 1.0)
        self.assertAlmostEqual(result["runtime_seconds_total"], 0.02)


if __name__ == "__main__":
    unittest.main()
