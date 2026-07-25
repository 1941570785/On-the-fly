from __future__ import annotations

import unittest

from scene.scene_model import SceneModel


class TransactionalRefinementStatsTests(unittest.TestCase):
    def test_committed_transaction_is_counted_once(self):
        model = object.__new__(SceneModel)
        model.pose_render_extra_optimization_stats = {
            "events": 0,
            "applied": 0,
            "extra_iterations_sum": 0,
            "confidence_sum": 0.0,
            "transaction_attempted": 0,
            "transaction_committed": 0,
            "transaction_rolled_back": 0,
            "transaction_early_stopped": 0,
            "transaction_budget_exhausted": 0,
            "transaction_best_iteration_sum": 0,
            "transaction_runtime_seconds": 0.0,
        }

        model._record_pose_render_extra_optimization(
            {
                "reason": "transaction_committed",
                "applied": True,
                "extra_iterations": 2,
                "confidence": 1.0,
                "transaction_attempted": True,
                "committed": True,
                "best_iteration": 2,
                "refinement_runtime_seconds": 0.1,
            }
        )

        stats = model.pose_render_extra_optimization_stats
        self.assertEqual(stats["transaction_attempted"], 1)
        self.assertEqual(stats["transaction_committed"], 1)
        self.assertEqual(stats["transaction_best_iteration_sum"], 2)


if __name__ == "__main__":
    unittest.main()
