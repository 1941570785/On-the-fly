from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.evaluate_official_pose_benchmark import (  # noqa: E402
    canonical_frame_id,
    evaluate_risk_trace,
    evaluate_scene,
    load_metadata_trajectory,
    load_tum_reference,
)


def w2c_at(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 3] = [x, y, z]
    return np.linalg.inv(c2w)


class OfficialPoseEvaluatorTest(unittest.TestCase):
    def test_canonical_frame_id_matches_padded_names(self) -> None:
        self.assertEqual(canonical_frame_id("000031.png"), "31")
        self.assertEqual(canonical_frame_id("31.png"), "31")
        self.assertEqual(canonical_frame_id("Frame-A.JPG"), "frame-a")

    def test_tum_reference_loader_uses_only_valid_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gt_dir = Path(tmp)
            poses = np.stack([w2c_at(0), w2c_at(1), w2c_at(2), w2c_at(3)])
            np.save(gt_dir / "poses_w2c.npy", poses)
            np.save(gt_dir / "valid_mask.npy", np.asarray([False, True, True, True]))
            with (gt_dir / "frame_map.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["image_name"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"image_name": "000001.png"},
                        {"image_name": "000002.png"},
                        {"image_name": "000003.png"},
                        {"image_name": "000004.png"},
                    ]
                )

            references = load_tum_reference(gt_dir)

        self.assertEqual(set(references), {"2", "3", "4"})
        np.testing.assert_allclose(references["2"], poses[1])

    def test_metadata_loader_rejects_invalid_pose_and_canonicalizes_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            metadata = {
                "keyframes": [
                    {"info": {"name": "000001.jpg"}, "Rt": w2c_at(0).tolist()},
                    {"info": {"name": "000002.jpg"}, "Rt": [[1.0, 0.0]]},
                    {"info": {"name": "000003.jpg"}, "Rt": w2c_at(1).tolist()},
                    {"info": {"name": "000004.jpg"}, "Rt": w2c_at(2).tolist()},
                ]
            }
            (model_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            trajectory = load_metadata_trajectory(model_dir)

        self.assertEqual(set(trajectory), {"1", "3", "4"})

    def test_three_way_evaluation_uses_one_common_frame_set(self) -> None:
        reference = {
            "1": w2c_at(0, 0),
            "2": w2c_at(1, 0),
            "3": w2c_at(0, 1),
            "4": w2c_at(1, 1),
            "5": w2c_at(2, 1),
        }
        methods = {
            "baseline": {key: value.copy() for key, value in reference.items()},
            "v31": {key: value.copy() for key, value in reference.items() if key != "5"},
            "v31_a": {key: value.copy() for key, value in reference.items() if key != "4"},
        }

        report = evaluate_scene(reference, methods, rpe_delta=1)

        self.assertEqual(report["three_way_common_frame_ids"], ["1", "2", "3"])
        self.assertEqual(report["three_way_common_frames"], 3)
        self.assertEqual(report["methods"]["baseline"]["ape_count"], 3)
        self.assertEqual(report["v31_vs_a"]["common_frame_ids"], ["1", "2", "3"])
        self.assertAlmostEqual(report["methods"]["v31_a"]["coverage"], 0.8)
        self.assertLess(report["methods"]["baseline"]["ape_trans_rmse"], 1e-9)

    def test_risk_diagnostics_use_external_reference_and_report_quarantine(self) -> None:
        reference = {
            "1": w2c_at(0, 0),
            "2": w2c_at(1, 0),
            "3": w2c_at(0, 1),
            "4": w2c_at(1, 1),
        }
        events = [
            {
                "image_name": f"{index:06d}.jpg",
                "estimated_Rt": reference[str(index)].tolist(),
                "risk_score": float(index) / 10.0,
                "risk_candidate": index >= 3,
                "pose_reference_quarantined": index == 4,
            }
            for index in range(1, 5)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "pose_risk_utility_trace.json"
            trace_path.write_text(json.dumps({"events": events}), encoding="utf-8")
            diagnostics = evaluate_risk_trace(
                trace_path,
                reference,
                reference,
                ["1", "2", "3", "4"],
            )

        self.assertEqual(diagnostics["evaluated_events"], 4)
        self.assertEqual(diagnostics["quarantined_count"], 1)
        self.assertEqual(diagnostics["quarantined"][0]["frame_id"], "4")
        self.assertGreaterEqual(diagnostics["quarantined"][0]["rotation_percentile"], 0.0)


if __name__ == "__main__":
    unittest.main()
