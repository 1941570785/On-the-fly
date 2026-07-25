from __future__ import annotations

import random
import unittest

import cv2
import numpy as np
import torch

from experiment_reproducibility import (
    configure_experiment_reproducibility,
    experiment_cuda_graphs_enabled,
)


class ExperimentReproducibilityTests(unittest.TestCase):
    def test_cuda_graphs_are_disabled_only_for_deterministic_experiments(self):
        self.assertTrue(experiment_cuda_graphs_enabled(False))
        self.assertFalse(experiment_cuda_graphs_enabled(True))

    def test_deterministic_mode_reseeds_libraries_and_disables_fast_math(self):
        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        previous_benchmark = torch.backends.cudnn.benchmark
        previous_cudnn_deterministic = torch.backends.cudnn.deterministic
        previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        try:
            configure_experiment_reproducibility(17, deterministic=True)
            first = (
                random.random(),
                float(np.random.rand()),
                float(torch.rand(())),
            )
            configure_experiment_reproducibility(17, deterministic=True)
            second = (
                random.random(),
                float(np.random.rand()),
                float(torch.rand(())),
            )

            self.assertEqual(first, second)
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertTrue(torch.is_deterministic_algorithms_warn_only_enabled())
            self.assertFalse(torch.backends.cudnn.benchmark)
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertEqual(cv2.getNumThreads(), 1)
        finally:
            torch.use_deterministic_algorithms(
                previous_deterministic,
                warn_only=previous_warn_only,
            )
            torch.backends.cudnn.benchmark = previous_benchmark
            torch.backends.cudnn.deterministic = previous_cudnn_deterministic
            torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
            torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32


if __name__ == "__main__":
    unittest.main()
