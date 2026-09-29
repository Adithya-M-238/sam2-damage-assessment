"""
Train the SAM2 damage-assessment model.

Use --dry_run first. It trains on FakeDamageDataset so we can verify the
complete pipeline on CPU before using the real xBD dataset.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import random
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from sam2_damage.dataset import FakeDamageDataset
from sam2_damage.losses import DamageAssessmentLoss
from sam2_damage.metrics import (
    building_logits_to_mask,
    compute_xview2_metrics,
    damage_logits_to_mask,
)
from sam2_damage.model import SAM2DamageModel


def set_seed(seed: int) -> None:
    """Make the small dry run repeatable."""
    random.seed(seed)
    torch.manual_seed(seed)


def choose_device(requested_device: str) -> str:
    """Use CUDA when available, otherwise use CPU."""
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    return requested_device


def load_real_dataset(factory_path: str, split: str) -> Dataset:
    """
    Load a real dataset through a user-supplied factory function.

    Example:
        --dataset_factory my_xbd_loader:make_dataset

    The function must accept `split="train"` or `split="val"` and return a
    PyTorch Dataset whose samples have pre_image, post_image, building_mask,
    and damage_mask.
    """
    module_name, function_name = factory_path.split(":")
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    return factory(split=split)


def make_datasets(args: argparse.Namespace) -> tuple[Dataset, Dataset]:
    """Choose fake data for dry runs, otherwise use the supplied real loader."""
    if args.dry_run:
        return (
            FakeDamageDataset(
                num_samples=8,
                image_size=args.image_size,
                seed=args.seed,
            ),
            FakeDamageDataset(
                num_samples=4,
                image_size=args.image_size,
                seed=args.seed + 10_000,
            ),
        )

    if args.dataset_factory is None:
        raise ValueError(
            "Real training needs --dataset_factory module_name:function_name. "
            "Use --dry_run until the real loader factory is available."
        )

    return (
        load_real_dataset(args.dataset_factory, split="train"),
        load_real_dataset(args.dataset_factory, split="val"),
    )


def move_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: str,
) -> Dict[str, torch.Tensor]:
    """Move every tensor in one DataLoader batch to CPU or CUDA."""
    return {
        name: tensor.to(device, non_blocking=True)
        for name, tensor in batch.items()
    }


def run_epoch(
    model: SAM2DamageModel,
    data_loader: DataLoader,
    criterion: DamageAssessmentLoss,
    device: str,
    optimizer: Optional[torch.optim.Optimizer],
) -> Dict[str, float]:
    """
    Run one training or validation epoch.

    optimizer=None means validation: no backward pass and no weight updates.
    """
    is_training = optimizer is not None
    model.train(is_training)

    total_loss_sum = 0.0
    localization_loss_sum = 0.0
    damage_loss_sum = 0.0
    number_of_samples = 0

    # Store whole-epoch predictions so F1 is calculated globally, not batch-wise.
    all_building_predictions = []
    all_damage_predictions = []
    all_building_targets = []
    all_damage_targets = []

    for batch in data_loader:
        batch = move_batch_to_device(batch, device)
        batch_size = batch["pre_image"].shape[0]

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            building_logits, damage_logits = model(
                batch["pre_image"],
                batch["post_image"],
            )

            losses = criterion(
                building_logits,
                damage_logits,
                batch["building_mask"],
                batch["damage_mask"],
            )

            if is_training:
                losses["total_loss"].backward()
                optimizer.step()

        total_loss_sum += losses["total_loss"].item() * batch_size
        localization_loss_sum += losses["localization_loss"].item() * batch_size
        damage_loss_sum += losses["damage_loss"].item() * batch_size
        number_of_samples += batch_size

        # Save label masks on CPU for epoch-level xView2 metrics.
        all_building_predictions.append(
            building_logits_to_mask(building_logits.detach()).cpu()
        )
        all_damage_predictions.append(
            damage_logits_to_mask(damage_logits.detach()).cpu()
        )
        all_building_targets.append(batch["building_mask"].detach().cpu())
        all_damage_targets.append(batch["damage_mask"].detach().cpu())

    metrics = compute_xview2_metrics(
        torch.cat(all_building_predictions),
        torch.cat(all_damage_predictions),
        torch.cat(all_building_targets),
        torch.cat(all_damage_targets),
    )

    metrics["total_loss"] = total_loss_sum / number_of_samples
    metrics["localization_loss"] = localization_loss_sum / number_of_samples
    metrics["damage_loss"] = damage_loss_sum / number_of_samples

    # For binary masks, hard Dice score and F1 are the same formula.
    metrics["localization_dice"] = metrics["localization_f1"]

    return metrics


def print_epoch_summary(
    epoch: int,
    epochs: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
) -> None:
    """Print a compact, readable summary after each epoch."""
    print(f"\nEpoch {epoch}/{epochs}")
    print(
        f"  Train loss: {train_metrics['total_loss']:.4f} | "
        f"Val loss: {val_metrics['total_loss']:.4f}"
    )
    print(
        f"  Val localization F1/Dice: {val_metrics['localization_f1']:.4f} | "
        f"Val damage F1: {val_metrics['damage_f1']:.4f} | "
        f"Val overall: {val_metrics['overall_score']:.4f}"
    )


def save_checkpoint(
    path: Path,
    epoch: int,
    model: SAM2DamageModel,
    optimizer: torch.optim.Optimizer,
    best_score: float,
    args: argparse.Namespace,
) -> None:
    """Save all information needed to continue training later."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_score": best_score,
            "arguments": vars(args),
        },
        path,
    )


def append_csv_row(
    csv_path: Path,
    epoch: int,
    learning_rate: float,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
) -> None:
    """Append one epoch of losses and metrics to a CSV file."""
    row = {
        "epoch": epoch,
        "learning_rate": learning_rate,
    }

    row.update({f"train_{name}": value for name, value in train_metrics.items()})
    row.update({f"val_{name}": value for name, value in val_metrics.items()})

    write_header = not csv_path.exists()

    with csv_path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def parse_arguments() -> argparse.Namespace:
    """Read command-line settings."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default="sam2/checkpoints/sam2.1_hiera_tiny.pt",
    )
    parser.add_argument(
        "--config",
        default="configs/sam2.1/sam2.1_hiera_t.yaml",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)

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
    parser.add_argument("--use_focal_loss", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    # This is used later when the teammate's real loader is available.
    parser.add_argument(
        "--dataset_factory",
        default=None,
        help="Example: my_xbd_loader:make_dataset",
    )

    return parser.parse_args()


def main() -> None:
    """Build everything, train, validate, save checkpoints, and log CSV rows."""
    args = parse_arguments()
    set_seed(args.seed)

    if args.dry_run:
        # Tiny CPU-safe run: 8 train images, 4 validation images, 2 epochs.
        args.epochs = min(args.epochs, 2)
        args.image_size = min(args.image_size, 128)

    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Dry run: {args.dry_run}")
    print(f"Output folder: {output_dir.resolve()}")

    train_dataset, val_dataset = make_datasets(args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = SAM2DamageModel(
        model_cfg=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
        freeze_encoder=args.freeze_encoder,
        use_attention_gates=args.use_attention_gates,
    )

    criterion = DamageAssessmentLoss(
        use_focal_loss=args.use_focal_loss,
    ).to(device)

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )

    start_epoch = 1
    best_score = float("-inf")

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        print(f"Resumed from epoch {checkpoint['epoch']}.")

    csv_path = output_dir / "training_log.csv"
    latest_path = output_dir / "latest_checkpoint.pt"
    best_path = output_dir / "best_checkpoint.pt"

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
        )

        print_epoch_summary(epoch, args.epochs, train_metrics, val_metrics)

        current_learning_rate = optimizer.param_groups[0]["lr"]

        append_csv_row(
            csv_path,
            epoch,
            current_learning_rate,
            train_metrics,
            val_metrics,
        )

        save_checkpoint(
            latest_path,
            epoch,
            model,
            optimizer,
            best_score,
            args,
        )

        if val_metrics["overall_score"] > best_score:
            best_score = val_metrics["overall_score"]

            save_checkpoint(
                best_path,
                epoch,
                model,
                optimizer,
                best_score,
                args,
            )

            print("  Saved new best checkpoint.")

    print("\nTraining finished.")
    print(f"Best validation overall score: {best_score:.4f}")


if __name__ == "__main__":
    main()