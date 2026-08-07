from __future__ import annotations

import unittest
from contextlib import redirect_stderr
from io import StringIO

from args import get_args


class ReleaseCliTests(unittest.TestCase):
    def test_final_method_is_default(self):
        args = get_args(["-s", "/tmp/scene", "-m", "/tmp/model"])
        self.assertEqual(args.method, "asr-gs")
        self.assertEqual(args.asr_gs_config.refinement.max_extra_iterations, 8)

    def test_paper_ablation_flags_are_public(self):
        args = get_args(
            [
                "-s",
                "/tmp/scene",
                "-m",
                "/tmp/model",
                "--ablate-a",
                "--ablate-c",
            ]
        )
        self.assertFalse(args.asr_gs_config.pose.enabled)
        self.assertTrue(args.asr_gs_config.sampling.enabled)
        self.assertFalse(args.asr_gs_config.refinement.enabled)

    def test_historical_profile_switch_is_rejected(self):
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                get_args(
                    [
                        "-s",
                        "/tmp/scene",
                        "-m",
                        "/tmp/model",
                        "--paper_aligned_pose_render_assimilation_profile",
                        "baseline_render_lock_intra_frame_v31",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
