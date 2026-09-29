"""Tests for the random fake dataset."""

import unittest

import torch

from sam2_damage.dataset import FakeDamageDataset


class TestFakeDamageDataset(unittest.TestCase):
    """Check shapes, labels, and repeatability."""

    def test_sample_has_correct_shapes_and_types(self) -> None:
        dataset = FakeDamageDataset(num_samples=3, image_size=(32, 48))
        sample = dataset[0]

        self.assertEqual(sample["pre_image"].shape, (3, 32, 48))
        self.assertEqual(sample["post_image"].shape, (3, 32, 48))
        self.assertEqual(sample["building_mask"].shape, (32, 48))
        self.assertEqual(sample["damage_mask"].shape, (32, 48))

        self.assertEqual(sample["pre_image"].dtype, torch.float32)
        self.assertEqual(sample["building_mask"].dtype, torch.long)
        self.assertEqual(sample["damage_mask"].dtype, torch.long)

    def test_damage_is_background_outside_buildings(self) -> None:
        dataset = FakeDamageDataset(num_samples=1, image_size=64)
        sample = dataset[0]

        background_damage = sample["damage_mask"][sample["building_mask"] == 0]
        self.assertTrue(torch.all(background_damage == 0))

        self.assertTrue(torch.all((sample["building_mask"] >= 0) & (sample["building_mask"] <= 1)))
        self.assertTrue(torch.all((sample["damage_mask"] >= 0) & (sample["damage_mask"] <= 4)))

    def test_same_index_is_repeatable(self) -> None:
        dataset = FakeDamageDataset(num_samples=2, image_size=32, seed=123)

        first_read = dataset[0]
        second_read = dataset[0]

        self.assertTrue(torch.equal(first_read["pre_image"], second_read["pre_image"]))
        self.assertTrue(torch.equal(first_read["damage_mask"], second_read["damage_mask"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)