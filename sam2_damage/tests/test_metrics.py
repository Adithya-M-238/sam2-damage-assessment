"""Known-answer tests for the xView2 scoring functions."""

import unittest

import torch

from sam2_damage.metrics import compute_xview2_metrics


class TestXView2Metrics(unittest.TestCase):
    """Test perfect and completely wrong predictions."""

    def test_perfect_prediction_scores_one(self):
        # One background pixel and one pixel from every damage class.
        damage_target = torch.tensor(
            [[[0, 1, 2],
              [3, 4, 1]]]
        )
        building_target = (damage_target > 0).long()

        # Very high score for the correct class at every pixel.
        building_logits = torch.where(
            building_target.unsqueeze(1) == 1,
            torch.tensor(10.0),
            torch.tensor(-10.0),
        )

        damage_logits = torch.full((1, 5, 2, 3), -10.0)
        damage_logits.scatter_(1, damage_target.unsqueeze(1), 10.0)

        metrics = compute_xview2_metrics(
            building_logits,
            damage_logits,
            building_target,
            damage_target,
        )

        for name, score in metrics.items():
            self.assertAlmostEqual(score, 1.0, places=6, msg=name)

    def test_all_wrong_prediction_scores_zero(self):
        damage_target = torch.tensor(
            [[[1, 2],
              [3, 4]]]
        )
        building_target = torch.ones_like(damage_target)

        # Model predicts no buildings and only background damage.
        building_logits = torch.full((1, 1, 2, 2), -10.0)
        damage_logits = torch.full((1, 5, 2, 2), -10.0)
        damage_logits[:, 0] = 10.0

        metrics = compute_xview2_metrics(
            building_logits,
            damage_logits,
            building_target,
            damage_target,
        )

        for name, score in metrics.items():
            self.assertAlmostEqual(score, 0.0, places=6, msg=name)


if __name__ == "__main__":
    unittest.main(verbosity=2)