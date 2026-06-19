from __future__ import annotations

import unittest
from pathlib import Path

from dataloaders.baseline_eval import baseline_eval_metadata


class BaselineEvalAlignmentTests(unittest.TestCase):
    def test_matches_official_test_hold_rule_after_start_at(self):
        row = baseline_eval_metadata(
            sequence_index=30,
            image_name="frame_00042.png",
            test_hold=30,
            start_at=12,
        )

        self.assertTrue(row["is_test"])
        self.assertTrue(row["_baseline_eval_frame"])
        self.assertEqual(row["_baseline_eval_hold"], 30)
        self.assertEqual(row["_baseline_eval_sequence_index"], 30)
        self.assertEqual(row["_baseline_eval_original_index"], 42)
        self.assertEqual(row["image_name"], "frame_00042.png")

    def test_non_eval_frames_keep_baseline_metadata_without_being_test(self):
        row = baseline_eval_metadata(
            sequence_index=29,
            image_name="frame_00041.png",
            test_hold=30,
            start_at=12,
        )

        self.assertFalse(row["is_test"])
        self.assertFalse(row["_baseline_eval_frame"])
        self.assertEqual(row["_baseline_eval_sequence_index"], -1)
        self.assertEqual(row["_baseline_eval_original_index"], -1)

    def test_disabled_test_hold_matches_baseline_no_eval_frames(self):
        row = baseline_eval_metadata(
            sequence_index=0,
            image_name="00000.jpg",
            test_hold=-1,
            start_at=0,
        )

        self.assertFalse(row["is_test"])
        self.assertFalse(row["_baseline_eval_frame"])

    def test_train_loop_preserves_baseline_eval_frames_after_runtime_gate(self):
        train_py = Path("train.py").read_text(encoding="utf-8")

        self.assertIn("_baseline_eval_frame", train_py)
        self.assertIn("baseline_eval_frame_forced", train_py)


if __name__ == "__main__":
    unittest.main()
