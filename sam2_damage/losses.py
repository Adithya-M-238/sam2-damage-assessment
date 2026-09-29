"""
Loss functions for building localization and damage classification.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# These are the xBD frequencies among building damage labels:
# no-damage, minor-damage, major-damage, destroyed.
DAMAGE_CLASS_FREQUENCIES = (0.721, 0.092, 0.087, 0.081)


def make_inverse_frequency_weights(
    damage_frequencies: Sequence[float] = DAMAGE_CLASS_FREQUENCIES,
    background_weight: float = 1.0,
) -> torch.Tensor:
    """
    Create five Cross-Entropy weights: background plus four damage classes.

    The supplied xBD statistics describe only building classes, not the
    background pixel frequency. Therefore background receives a neutral
    weight of 1.0 by default. Later, when real dataset pixel counts are known,
    we can replace all five weights with exact values.
    """
    frequencies = torch.tensor(damage_frequencies, dtype=torch.float32)

    # Rare classes have larger inverse-frequency values.
    inverse = 1.0 / frequencies

    # Keep the average damage-class weight near 1 for stable loss size.
    inverse = inverse / inverse.mean()

    return torch.cat(
        [
            torch.tensor([background_weight], dtype=torch.float32),
            inverse,
        ]
    )


class LocalizationLoss(nn.Module):
    """Binary cross-entropy plus Dice loss for building localization."""

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(
        self,
        building_logits: torch.Tensor,
        building_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate localization loss.

        building_logits: (B, 1, H, W), raw model scores
        building_target: (B, H, W) or (B, 1, H, W), values 0 or 1
        """
        if building_target.ndim == 3:
            building_target = building_target.unsqueeze(1)

        building_target = building_target.float()

        # BCE checks every pixel individually.
        bce = F.binary_cross_entropy_with_logits(
            building_logits,
            building_target,
        )

        # Dice checks how well predicted and true building regions overlap.
        probabilities = torch.sigmoid(building_logits)
        probabilities = probabilities.flatten(start_dim=1)
        target_flat = building_target.flatten(start_dim=1)

        intersection = (probabilities * target_flat).sum(dim=1)
        denominator = probabilities.sum(dim=1) + target_flat.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (
            denominator + self.smooth
        )
        dice_loss = 1.0 - dice_score.mean()

        return self.bce_weight * bce + self.dice_weight * dice_loss


class FocalCrossEntropyLoss(nn.Module):
    """
    Class-weighted focal loss for 5-class damage segmentation.

    Focal loss reduces the contribution of already-easy pixels and focuses
    more learning on difficult or rare examples.
    """

    def __init__(
        self,
        class_weights: torch.Tensor,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights.float())
        self.gamma = gamma

    def forward(
        self,
        damage_logits: torch.Tensor,
        damage_target: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate class-weighted focal cross-entropy."""
        cross_entropy = F.cross_entropy(
            damage_logits,
            damage_target.long(),
            weight=self.class_weights,
            reduction="none",
        )

        probabilities = F.softmax(damage_logits, dim=1)
        target_probability = probabilities.gather(
            dim=1,
            index=damage_target.long().unsqueeze(1),
        ).squeeze(1)

        focal_multiplier = (1.0 - target_probability) ** self.gamma

        return (focal_multiplier * cross_entropy).mean()


class DamageAssessmentLoss(nn.Module):
    """
    Combined loss for the two model outputs.

    Total = localization_weight * localization_loss
          + damage_weight * damage_loss
    """

    def __init__(
        self,
        localization_weight: float = 1.0,
        damage_weight: float = 1.0,
        class_weights: Optional[Sequence[float]] = None,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()

        if class_weights is None:
            class_weights_tensor = make_inverse_frequency_weights()
        else:
            class_weights_tensor = torch.tensor(
                class_weights,
                dtype=torch.float32,
            )

        if len(class_weights_tensor) != 5:
            raise ValueError(
                "class_weights must contain 5 values: background plus 4 classes."
            )

        self.localization_weight = localization_weight
        self.damage_weight = damage_weight
        self.use_focal_loss = use_focal_loss

        self.localization_loss = LocalizationLoss()
        self.register_buffer("class_weights", class_weights_tensor)

        self.focal_loss = FocalCrossEntropyLoss(
            class_weights_tensor,
            gamma=focal_gamma,
        )

    def forward(
        self,
        building_logits: torch.Tensor,
        damage_logits: torch.Tensor,
        building_target: torch.Tensor,
        damage_target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Return each loss separately, plus the final total loss.

        Training will call:
            losses["total_loss"].backward()
        """
        localization = self.localization_loss(
            building_logits,
            building_target,
        )

        if self.use_focal_loss:
            damage = self.focal_loss(damage_logits, damage_target)
        else:
            damage = F.cross_entropy(
                damage_logits,
                damage_target.long(),
                weight=self.class_weights,
            )

        total = (
            self.localization_weight * localization
            + self.damage_weight * damage
        )

        return {
            "total_loss": total,
            "localization_loss": localization,
            "damage_loss": damage,
        }


if __name__ == "__main__":
    # Small CPU-only smoke test.
    batch_size, height, width = 2, 32, 32

    fake_building_logits = torch.randn(batch_size, 1, height, width)
    fake_damage_logits = torch.randn(batch_size, 5, height, width)

    fake_building_target = torch.randint(0, 2, (batch_size, height, width))
    fake_damage_target = torch.randint(0, 5, (batch_size, height, width))

    criterion = DamageAssessmentLoss(use_focal_loss=False)
    losses = criterion(
        fake_building_logits,
        fake_damage_logits,
        fake_building_target,
        fake_damage_target,
    )

    print("Class weights:", criterion.class_weights.tolist())

    for name, value in losses.items():
        print(f"{name}: {value.item():.4f}")