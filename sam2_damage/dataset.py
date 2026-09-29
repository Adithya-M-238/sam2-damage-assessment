"""
Dataset interfaces for post-disaster damage assessment.

The fake dataset lets us test the model and training loop before connecting
the real xBD/xView2 data loader.
"""

from typing import Dict, Tuple, Union

import torch
from torch.utils.data import Dataset


Sample = Dict[str, torch.Tensor]


class FakeDamageDataset(Dataset):
    """
    Small random dataset with the same interface as the real xBD dataset.

    Each sample contains:
        pre_image:     (3, H, W), float32
        post_image:    (3, H, W), float32
        building_mask: (H, W),    0=background, 1=building
        damage_mask:   (H, W),    0=background, 1..4=damage classes
    """

    def __init__(
        self,
        num_samples: int = 8,
        image_size: Union[int, Tuple[int, int]] = 128,
        seed: int = 42,
        building_probability: float = 0.20,
    ) -> None:
        """Save the settings used to generate samples."""
        self.num_samples = num_samples
        self.seed = seed
        self.building_probability = building_probability

        # Allow either image_size=128 or image_size=(128, 256).
        if isinstance(image_size, int):
            self.height = image_size
            self.width = image_size
        else:
            self.height, self.width = image_size

    def __len__(self) -> int:
        """Return how many samples the dataset contains."""
        return self.num_samples

    def __getitem__(self, index: int) -> Sample:
        """
        Create one fake sample.

        A fixed seed plus the sample index makes each sample repeatable:
        asking for dataset[0] twice gives the same random sample.
        """
        if index < 0 or index >= self.num_samples:
            raise IndexError(f"Index {index} is outside this dataset.")

        generator = torch.Generator().manual_seed(self.seed + index)

        # Random normal values imitate images after standard normalization.
        pre_image = torch.randn(3, self.height, self.width, generator=generator)
        post_image = torch.randn(3, self.height, self.width, generator=generator)

        # Randomly select some pixels as buildings.
        building_mask = (
            torch.rand(self.height, self.width, generator=generator)
            < self.building_probability
        ).long()

        # Background remains 0. Every building gets a random class from 1 to 4.
        damage_mask = torch.zeros(self.height, self.width, dtype=torch.long)
        number_of_building_pixels = int(building_mask.sum().item())

        if number_of_building_pixels > 0:
            damage_mask[building_mask.bool()] = torch.randint(
                low=1,
                high=5,  # 5 is excluded, so possible labels are 1, 2, 3, 4.
                size=(number_of_building_pixels,),
                generator=generator,
            )

        return {
            "pre_image": pre_image.float(),
            "post_image": post_image.float(),
            "building_mask": building_mask.long(),
            "damage_mask": damage_mask.long(),
        }


if __name__ == "__main__":
    # Small manual smoke test when this file is run directly.
    dataset = FakeDamageDataset(num_samples=2, image_size=64)
    sample = dataset[0]

    for name, tensor in sample.items():
        print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")

    print("Building labels:", torch.unique(sample["building_mask"]).tolist())
    print("Damage labels:", torch.unique(sample["damage_mask"]).tolist())