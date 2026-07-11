from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scene.scene_model import SceneModel


def _model(mode="dark_scene_fixed_black_v1"):
    model = SceneModel.__new__(SceneModel)
    model.paper_aligned_training_background_mode = mode
    model._training_background_step = 0
    return model


def _keyframe(value, mask=None):
    return SimpleNamespace(
        index=7,
        image_pyr=[torch.full((3, 4, 4), float(value))],
        mask_pyr=None if mask is None else [mask],
    )


class TrainingBackgroundGuardTests(unittest.TestCase):
    def test_dark_scene_without_mask_uses_fixed_black_background(self):
        model = _model()
        background = model._training_background_for_keyframe(_keyframe(0.20), 0)

        self.assertTrue(torch.equal(background, torch.zeros_like(background)))
        stats = model.training_background_stats
        self.assertEqual(stats["dark_scene_fixed_black_applied"], 1)
        self.assertEqual(stats["dark_scene_mask_blocked"], 0)
        self.assertLessEqual(stats["dark_scene_running_mean"], 0.34)

    def test_bright_scene_without_mask_falls_back_to_random_background(self):
        model = _model()

        with patch("torch.rand", return_value=torch.tensor([0.25, 0.50, 0.75])):
            background = model._training_background_for_keyframe(_keyframe(0.52), 0)

        self.assertTrue(torch.equal(background, torch.tensor([0.25, 0.50, 0.75])))
        stats = model.training_background_stats
        self.assertEqual(stats["dark_scene_fixed_black_applied"], 0)
        self.assertEqual(stats["dark_scene_random_fallback"], 1)
        self.assertGreater(stats["dark_scene_running_mean"], 0.34)

    def test_masked_frame_never_uses_fixed_black_even_when_dark(self):
        model = _model()
        mask = torch.ones(4, 4, dtype=torch.bool)

        with patch("torch.rand", return_value=torch.tensor([0.10, 0.20, 0.30])):
            background = model._training_background_for_keyframe(
                _keyframe(0.18, mask=mask),
                0,
            )

        self.assertTrue(torch.equal(background, torch.tensor([0.10, 0.20, 0.30])))
        stats = model.training_background_stats
        self.assertEqual(stats["dark_scene_fixed_black_applied"], 0)
        self.assertEqual(stats["dark_scene_mask_blocked"], 1)

    def test_dark_scene_decision_latches_after_first_unmasked_probe(self):
        model = _model()

        first = model._training_background_for_keyframe(_keyframe(0.20), 0)
        second = model._training_background_for_keyframe(_keyframe(0.80), 0)

        self.assertTrue(torch.equal(first, torch.zeros_like(first)))
        self.assertTrue(torch.equal(second, torch.zeros_like(second)))
        stats = model.training_background_stats
        self.assertEqual(stats["dark_scene_observations"], 1)
        self.assertEqual(stats["dark_scene_fixed_black_applied"], 2)

    def test_bright_scene_decision_latches_random_background(self):
        model = _model()

        with patch("torch.rand", side_effect=[
            torch.tensor([0.25, 0.50, 0.75]),
            torch.tensor([0.10, 0.20, 0.30]),
        ]):
            first = model._training_background_for_keyframe(_keyframe(0.52), 0)
            second = model._training_background_for_keyframe(_keyframe(0.18), 0)

        self.assertTrue(torch.equal(first, torch.tensor([0.25, 0.50, 0.75])))
        self.assertTrue(torch.equal(second, torch.tensor([0.10, 0.20, 0.30])))
        stats = model.training_background_stats
        self.assertEqual(stats["dark_scene_observations"], 1)
        self.assertEqual(stats["dark_scene_fixed_black_applied"], 0)
        self.assertEqual(stats["dark_scene_random_fallback"], 2)


if __name__ == "__main__":
    unittest.main()
