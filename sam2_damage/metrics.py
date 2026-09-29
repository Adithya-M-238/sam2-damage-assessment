"""Metrics for xView2-style building localization and damage assessment."""

from typing import Dict

import torch


DAMAGE_CLASSES = (1, 2, 3, 4)


def _squeeze_mask(mask: torch.Tensor) -> torch.Tensor:
    """Convert (B, 1, H, W) into (B, H, W) when needed."""
    if mask.ndim == 4 and mask.shape[1] == 1:
        return mask[:, 0]
    return mask


def building_logits_to_mask(
    building_output: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Convert model building logits into a binary mask.

    The model returns one raw score (logit) per pixel. Sigmoid converts that
    score to a probability from 0 to 1; 0.5 or above means building.
    """
    if building_output.ndim == 4:
        if building_output.shape[1] != 1:
            raise ValueError("Building logits must have exactly one channel.")

        probabilities = torch.sigmoid(building_output[:, 0])
        return (probabilities >= threshold).long()

    if building_output.ndim == 3:
        return building_output.long()  # Already a 0/1 label mask.

    raise ValueError(
        "Building output must be (B, 1, H, W) logits or (B, H, W) labels."
    )


def damage_logits_to_mask(damage_output: torch.Tensor) -> torch.Tensor:
    """Convert 5-class damage logits into one class label per pixel."""
    if damage_output.ndim == 4:
        if damage_output.shape[1] != 5:
            raise ValueError("Damage logits must have 5 channels.")
        return damage_output.argmax(dim=1)

    if damage_output.ndim == 3:
        return damage_output.long()  # Already an integer label mask.

    raise ValueError(
        "Damage output must be (B, 5, H, W) logits or (B, H, W) labels."
    )


def binary_f1(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate F1 = 2TP / (2TP + FP + FN) for binary masks."""
    prediction = prediction.bool()
    target = target.bool()

    true_positive = (prediction & target).sum().float()
    false_positive = (prediction & ~target).sum().float()
    false_negative = (~prediction & target).sum().float()

    return (
        (2 * true_positive)
        / (2 * true_positive + false_positive + false_negative + 1e-8)
    ).item()


def harmonic_mean(values: list[float]) -> float:
    """Combine class F1 scores; a zero class score makes the result zero."""
    if not values or any(score <= 0.0 for score in values):
        return 0.0

    return len(values) / sum(1.0 / score for score in values)


def compute_xview2_metrics(
    building_output: torch.Tensor,
    damage_output: torch.Tensor,
    building_target: torch.Tensor,
    damage_target: torch.Tensor,
) -> Dict[str, float]:
    """
    Calculate all requested metrics.

    Overall score = 0.3 × Localization F1 + 0.7 × Damage F1.
    """
    predicted_buildings = building_logits_to_mask(building_output)
    predicted_damage = damage_logits_to_mask(damage_output)

    building_target = _squeeze_mask(building_target).long()
    damage_target = _squeeze_mask(damage_target).long()

    localization = binary_f1(predicted_buildings, building_target)

    # Evaluate damage only where the ground truth says a building exists.
    building_pixels = building_target.bool()
    class_scores = {}

    for class_id in DAMAGE_CLASSES:
        predicted_class = (predicted_damage == class_id) & building_pixels
        target_class = (damage_target == class_id) & building_pixels
        class_scores[class_id] = binary_f1(predicted_class, target_class)

    damage = harmonic_mean(list(class_scores.values()))
    overall = 0.3 * localization + 0.7 * damage

    return {
        "localization_f1": localization,
        "no_damage_f1": class_scores[1],
        "minor_damage_f1": class_scores[2],
        "major_damage_f1": class_scores[3],
        "destroyed_f1": class_scores[4],
        "damage_f1": damage,
        "overall_score": overall,
    }