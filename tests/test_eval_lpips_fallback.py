import unittest
from pathlib import Path

from scene.scene_model import _lpips_cpu_fallback_needed


class EvalLpipsFallbackTest(unittest.TestCase):
    def test_low_free_cuda_memory_uses_cpu_lpips(self):
        self.assertTrue(_lpips_cpu_fallback_needed(128 * 1024 * 1024))

    def test_sufficient_free_cuda_memory_keeps_gpu_lpips(self):
        self.assertFalse(_lpips_cpu_fallback_needed(2 * 1024 * 1024 * 1024))

    def test_evaluate_accumulates_lpips_metric_by_name(self):
        source = Path("scene/scene_model.py").read_text()
        self.assertIn('metrics["LPIPS"] += lpips_value', source)
        self.assertNotIn('metrics[LPIPS] += lpips_value', source)


if __name__ == "__main__":
    unittest.main()
