"""
Evaluate a trained SAM2 damage-assessment checkpoint.

Use --dry_run to evaluate the saved checkpoint on FakeDamageDataset.
Later, use --dataset_factory with the teammate's real holdout loader.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from torch.utils.data import DataLoader, Dataset

from sam2_damage.dataset import FakeDamageDataset
from sam2_damage.metrics import (
    building_logits_to_mask,
    compute_xview2_metrics,
    damage_logits_to_mask,
)
from sam2_damage.model import SAM2DamageModel


# Background, no-damage, minor, major, destroyed.
CLASS_COLORS = np.array(
    [
        [0, 0, 0],        # black
        [34, 139, 34],    # green
        [255, 215, 0],    # yellow
        [255, 140, 0],    # orange
        [220, 20, 60],    # red
    ],
    dtype=np.uint8,
)

CLASS_NAMES = [
    "Background",
    "No damage",
    "Minor damage",
    "Major damage",
    "Destroyed",
]


def choose_device(requested_device: str) -> str:
    """Choose CPU or CUDA."""
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    return requested_device


def load_real_dataset(factory_path: str, split: str) -> Dataset:
    """
    Load a real dataset through a factory function.

    Example:
        --dataset_factory my_xbd_loader:make_dataset
    """
    module_name, function_name = factory_path.split(":")
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    return factory(split=split)


def make_evaluation_dataset(args: argparse.Namespace) -> Dataset:
    """Use fake data for dry run, otherwise request the real holdout dataset."""
    if args.dry_run:
        return FakeDamageDataset(
            num_samples=4,
            image_size=args.image_size,
            seed=args.seed + 20_000,
        )

    if args.dataset_factory is None:
        raise ValueError(
            "Real evaluation needs --dataset_factory module_name:function_name. "
            "Use --dry_run until the real holdout loader is available."
        )

    return load_real_dataset(args.dataset_factory, split="holdout")


def image_for_display(image: torch.Tensor) -> np.ndarray:
    """
    Make a normalized tensor viewable as an RGB image.

    Fake images are random normalized values, so this rescales them to 0..1
    only for visualization. It does not change data used by the model.
    """
    image = image.detach().cpu().float()
    image = image.permute(1, 2, 0).numpy()

    minimum = image.min()
    maximum = image.max()

    if maximum > minimum:
        image = (image - minimum) / (maximum - minimum)

    return image


def damage_mask_to_rgb(damage_mask: torch.Tensor) -> np.ndarray:
    """Convert labels 0..4 into an RGB image using the class palette."""
    labels = damage_mask.detach().cpu().long().numpy()
    labels = np.clip(labels, 0, 4)
    return CLASS_COLORS[labels]


def save_visualization(
    pre_image: torch.Tensor,
    post_image: torch.Tensor,
    target_damage: torch.Tensor,
    predicted_damage: torch.Tensor,
    output_path: Path,
) -> None:
    """Save pre-image, post-image, ground truth, and prediction side by side."""
    figure, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(image_for_display(pre_image))
    axes[0].set_title("Pre-disaster image")

    axes[1].imshow(image_for_display(post_image))
    axes[1].set_title("Post-disaster image")

    axes[2].imshow(damage_mask_to_rgb(target_damage))
    axes[2].set_title("Ground-truth damage")

    axes[3].imshow(damage_mask_to_rgb(predicted_damage))
    axes[3].set_title("Predicted damage")

    for axis in axes:
        axis.axis("off")

    legend = [
        Patch(color=CLASS_COLORS[class_id] / 255.0, label=CLASS_NAMES[class_id])
        for class_id in range(5)
    ]

    figure.legend(
        handles=legend,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.02),
    )

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def evaluate(
    model: SAM2DamageModel,
    data_loader: DataLoader,
    device: str,
    visualization_dir: Path,
    number_of_visualizations: int,
) -> Dict[str, float]:
    """Run predictions, calculate global metrics, and save example images."""
    model.eval()

    all_building_predictions = []
    all_damage_predictions = []
    all_building_targets = []
    all_damage_targets = []

    saved_visualizations = 0

    with torch.no_grad():
        for batch in data_loader:
            pre_image = batch["pre_image"].to(device)
            post_image = batch["post_image"].to(device)
            building_target = batch["building_mask"].to(device)
            damage_target = batch["damage_mask"].to(device)

            building_logits, damage_logits = model(pre_image, post_image)

            building_prediction = building_logits_to_mask(building_logits)
            damage_prediction = damage_logits_to_mask(damage_logits)

            all_building_predictions.append(building_prediction.cpu())
            all_damage_predictions.append(damage_prediction.cpu())
            all_building_targets.append(building_target.cpu())
            all_damage_targets.append(damage_target.cpu())

            # Save a few visual examples. We gate damage with localization here
            # so the displayed output is one combined 5-class damage map.
            for item_index in range(pre_image.shape[0]):
                if saved_visualizations >= number_of_visualizations:
                    break

                combined_damage_prediction = damage_prediction[item_index].clone()
                combined_damage_prediction[
                    building_prediction[item_index] == 0
                ] = 0

                output_path = (
                    visualization_dir
                    / f"prediction_{saved_visualizations:02d}.png"
                )

                save_visualization(
                    pre_image[item_index],
                    post_image[item_index],
                    damage_target[item_index],
                    combined_damage_prediction,
                    output_path,
                )

                saved_visualizations += 1

    metrics = compute_xview2_metrics(
        torch.cat(all_building_predictions),
        torch.cat(all_damage_predictions),
        torch.cat(all_building_targets),
        torch.cat(all_damage_targets),
    )

    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """Print metrics in the format used for the project report."""
    print("\nEvaluation results")
    print(f"Localization F1: {metrics['localization_f1']:.4f}")
    print(f"Localization Dice: {metrics['localization_f1']:.4f}")
    print(f"No-damage F1: {metrics['no_damage_f1']:.4f}")
    print(f"Minor-damage F1: {metrics['minor_damage_f1']:.4f}")
    print(f"Major-damage F1: {metrics['major_damage_f1']:.4f}")
    print(f"Destroyed F1: {metrics['destroyed_f1']:.4f}")
    print(f"Damage F1: {metrics['damage_f1']:.4f}")
    print(f"Overall score: {metrics['overall_score']:.4f}")


def parse_arguments() -> argparse.Namespace:
    """Read command-line settings."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default="outputs/dry_run/best_checkpoint.pt",
    )
    parser.add_argument(
        "--config",
        default="configs/sam2.1/sam2.1_hiera_t.yaml",
    )
    parser.add_argument("--output_dir", default="outputs/evaluation")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_visualizations", type=int, default=3)
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument(
        "--use_attention_gates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--freeze_encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dataset_factory",
        default=None,
        help="Example: my_xbd_loader:make_dataset",
    )

    return parser.parse_args()


def main() -> None:
    """Load checkpoint, evaluate data, print metrics, and save output files."""
    args = parse_arguments()
    device = choose_device(args.device)

    output_dir = Path(args.output_dir)
    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    model = SAM2DamageModel(
        model_cfg=args.config,
        checkpoint_path="sam2/checkpoints/sam2.1_hiera_tiny.pt",
        device=device,
        freeze_encoder=args.freeze_encoder,
        use_attention_gates=args.use_attention_gates,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = make_evaluation_dataset(args)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Evaluation samples: {len(dataset)}")

    metrics = evaluate(
        model,
        data_loader,
        device,
        visualization_dir,
        args.num_visualizations,
    )

    print_metrics(metrics)

    metrics_path = output_dir / "metrics.json"

    with metrics_path.open("w") as file:
        json.dump(metrics, file, indent=2)

    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved visualizations: {visualization_dir}")


if __name__ == "__main__":
    main()