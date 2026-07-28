import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from asr_gs.sampling_trace import (
    effective_bernoulli_probability,
    sampling_trace_matches,
    write_sampling_trace,
)


class SamplingTraceTests(unittest.TestCase):
    def test_effective_bernoulli_probability_clips_raw_intensity(self):
        raw = np.array([[-0.2, 0.4, 1.3]], dtype=np.float32)
        np.testing.assert_array_equal(
            effective_bernoulli_probability(raw),
            np.array([[0.0, 0.4, 1.0]], dtype=np.float32),
        )

    def test_trace_target_supports_exact_frame_and_applied_event(self):
        self.assertTrue(
            sampling_trace_matches(
                target="000123.png",
                frame_name="/dataset/images/000123.png",
                applied=False,
            )
        )
        self.assertTrue(
            sampling_trace_matches(
                target="applied",
                frame_name="000456.png",
                applied=True,
            )
        )
        self.assertFalse(
            sampling_trace_matches(
                target="applied",
                frame_name="000456.png",
                applied=False,
            )
        )
        self.assertFalse(
            sampling_trace_matches(
                target="",
                frame_name="000123.png",
                applied=True,
            )
        )

    def test_trace_writes_exact_probabilities_and_bernoulli_draw(self):
        base = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        final = np.array([[0.05, 0.25], [0.35, 0.4]], dtype=np.float32)
        sampled = np.array([[False, True], [True, False]])
        with tempfile.TemporaryDirectory() as directory:
            output = write_sampling_trace(
                output_root=Path(directory),
                target="000123.png",
                frame_name="/dataset/images/000123.png",
                base_probability=base,
                final_probability=final,
                sample_mask=sampled,
                debug={"applied": True, "reason": "response_guided"},
            )
            self.assertIsNotNone(output)
            with np.load(output) as archive:
                np.testing.assert_array_equal(
                    archive["base_probability"],
                    base,
                )
                np.testing.assert_array_equal(
                    archive["final_probability"],
                    final,
                )
                np.testing.assert_array_equal(
                    archive["sample_mask"],
                    sampled,
                )
            self.assertGreaterEqual(float(final.min()), 0.0)
            self.assertLessEqual(float(final.max()), 1.0)
            metadata_path = output.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["frame_name"], "000123.png")
            self.assertTrue(metadata["debug"]["applied"])

    def test_non_matching_frame_does_not_create_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            output = write_sampling_trace(
                output_root=Path(directory),
                target="000123.png",
                frame_name="000456.png",
                base_probability=np.zeros((2, 2), dtype=np.float32),
                final_probability=np.zeros((2, 2), dtype=np.float32),
                sample_mask=np.zeros((2, 2), dtype=bool),
                debug={"applied": False},
            )
            self.assertIsNone(output)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
