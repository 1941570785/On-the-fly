from __future__ import annotations

import math
import unittest

import numpy as np

from tools.standard_pose_eval_report import (
    evaluate_metadata_pair,
    fit_similarity_w2c,
    pose_errors,
    relative_pose_errors,
)


def _c2w(tx: float, ty: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    c = math.cos(yaw)
    s = math.sin(yaw)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.asarray(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    out[:3, 3] = [tx, ty, 0.0]
    return out


def _w2c(tx: float, ty: float = 0.0, yaw_deg: float = 0.0) -> list[list[float]]:
    return np.linalg.inv(_c2w(tx, ty, yaw_deg)).tolist()


def _metadata(prefix: str, poses: list[list[list[float]]]) -> dict:
    return {
        "num keyframes": len(poses),
        "time": 1.0,
        "keyframes": [
            {
                "info": {
                    "name": f"{idx:03d}.png",
                    "is_test": True,
                    "gt_Rt": _w2c(float(idx), 0.0, 0.0),
                },
                "Rt": pose,
            }
            for idx, pose in enumerate(poses)
        ],
        "label": prefix,
    }


class StandardPoseEvalReportTest(unittest.TestCase):
    def test_similarity_alignment_removes_global_scale_rotation_translation(self):
        gt = np.asarray([_w2c(0, 0), _w2c(1, 0), _w2c(0, 1)], dtype=np.float64)
        transformed_c2w = []
        rot = _c2w(0, 0, 30.0)[:3, :3]
        for pose_w2c in gt:
            pose_c2w = np.linalg.inv(pose_w2c)
            out = np.eye(4, dtype=np.float64)
            out[:3, :3] = rot @ pose_c2w[:3, :3]
            out[:3, 3] = 2.0 * (rot @ pose_c2w[:3, 3]) + np.asarray([4.0, -3.0, 1.0])
            transformed_c2w.append(out)
        estimated = np.linalg.inv(np.asarray(transformed_c2w))

        similarity = fit_similarity_w2c(estimated, gt)
        errors = pose_errors(similarity.apply(estimated), gt)

        self.assertLess(float(errors["trans_mean"]), 1e-9)
        self.assertLess(float(errors["rot_rad_mean"]), 1e-7)

    def test_relative_pose_errors_report_translation_and_rotation_drift(self):
        gt = np.asarray([_w2c(0, 0, 0), _w2c(1, 0, 0), _w2c(2, 0, 0)], dtype=np.float64)
        estimated = np.asarray(
            [_w2c(0, 0, 0), _w2c(1.25, 0, 10), _w2c(2.25, 0, 10)],
            dtype=np.float64,
        )

        errors = relative_pose_errors(estimated, gt, delta=1)

        self.assertEqual(errors["count"], 2)
        self.assertGreater(errors["trans_mean"], 0.1)
        self.assertGreater(errors["rot_deg_mean"], 4.0)

    def test_metadata_pair_uses_common_gt_frames_and_reports_ape_rpe(self):
        baseline = _metadata("baseline", [_w2c(0), _w2c(1), _w2c(2), _w2c(3)])
        method = _metadata("method", [_w2c(0), _w2c(1.05), _w2c(2.05), _w2c(3.10)])
        method["keyframes"].append(
            {
                "info": {"name": "999.png", "is_test": True, "gt_Rt": _w2c(999)},
                "Rt": _w2c(999),
            }
        )

        result = evaluate_metadata_pair(baseline, method)

        self.assertEqual(result["pose_eval_frames"], 4)
        self.assertEqual(result["baseline"]["ape_count"], 4)
        self.assertEqual(result["method"]["rpe_count"], 3)
        self.assertIn("ape_rot_rad_mean", result["method"])
        self.assertIn("rpe_trans_mean", result["method"])


if __name__ == "__main__":
    unittest.main()
