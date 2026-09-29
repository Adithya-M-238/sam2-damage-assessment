"""
SAM2-based model for building localization and damage classification.

Input:
    pre_image, post_image: (B, 3, H, W)

Output:
    building_logits: (B, 1, H, W)
    damage_logits:   (B, 5, H, W)
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# SAM2 protects against importing its repository folder as the Python package.
# Our project root contains the source folder B:\Mini Project\sam2, so remove
# the project root from Python's import search list before importing SAM2.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path[:] = [
    entry
    for entry in sys.path
    if Path(entry or ".").resolve() != PROJECT_ROOT
]

from sam2.build_sam import build_sam2



def make_group_norm(channels: int) -> nn.GroupNorm:
    """Make a GroupNorm layer whose group count divides the channel count."""
    groups = min(8, channels)

    while channels % groups != 0:
        groups -= 1

    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Module):
    """Two simple convolution layers used inside the decoder."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            make_group_norm(output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            make_group_norm(output_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class BottleneckAdapter(nn.Module):
    """
    A small trainable residual adapter.

    SAM2 remains mostly frozen. This module learns a small correction that
    adapts SAM2's general image features to satellite imagery.
    """

    def __init__(self, channels: int, bottleneck_channels: int = 64) -> None:
        super().__init__()

        self.down_project = nn.Conv2d(channels, bottleneck_channels, kernel_size=1)
        self.activation = nn.GELU()
        self.up_project = nn.Conv2d(bottleneck_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        correction = self.up_project(self.activation(self.down_project(x)))
        return x + correction


class FeatureFusion(nn.Module):
    """
    Combine matching pre- and post-disaster features.

    It receives pre-image features, post-image features, and their absolute
    difference. The difference helps the model notice damage-related change.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),
            make_group_norm(channels),
            nn.GELU(),
        )

    def forward(
        self,
        pre_features: torch.Tensor,
        post_features: torch.Tensor,
    ) -> torch.Tensor:
        difference = torch.abs(post_features - pre_features)

        combined = torch.cat(
            [pre_features, post_features, difference],
            dim=1,
        )
        return self.fuse(combined)


class AttentionGate(nn.Module):
    """
    Additive attention gate from Attention U-Net.

    `skip_features` are fine-detail features. `gating_features` come from a
    coarser decoder layer and tell the gate which areas are useful. The output
    is the skip features multiplied by learned values between 0 and 1.
    """

    def __init__(
        self,
        skip_channels: int,
        gating_channels: int,
        intermediate_channels: int,
    ) -> None:
        super().__init__()

        self.skip_projection = nn.Conv2d(
            skip_channels,
            intermediate_channels,
            kernel_size=1,
            bias=False,
        )
        self.gating_projection = nn.Conv2d(
            gating_channels,
            intermediate_channels,
            kernel_size=1,
            bias=False,
        )
        self.attention_score = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(intermediate_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(
        self,
        skip_features: torch.Tensor,
        gating_features: torch.Tensor,
    ) -> torch.Tensor:
        projected_skip = self.skip_projection(skip_features)
        projected_gate = self.gating_projection(gating_features)

        # Coarse decoder features are resized to match the skip connection.
        projected_gate = F.interpolate(
            projected_gate,
            size=projected_skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        coefficients = self.attention_score(projected_skip + projected_gate)
        return skip_features * coefficients


class MultiScaleDecoder(nn.Module):
    """
    Decode SAM2's three fused feature maps into a full-resolution prediction.

    Features arrive in this order:
        high resolution:   approximately H/4 by W/4
        middle resolution: approximately H/8 by W/8
        low resolution:    approximately H/16 by W/16
    """

    def __init__(
        self,
        feature_channels: int,
        decoder_channels: int,
        output_channels: int,
        use_attention_gates: bool,
    ) -> None:
        super().__init__()

        self.use_attention_gates = use_attention_gates

        self.coarse_block = ConvBlock(feature_channels, decoder_channels)
        self.middle_block = ConvBlock(
            decoder_channels + feature_channels,
            decoder_channels,
        )
        self.high_block = ConvBlock(
            decoder_channels + feature_channels,
            decoder_channels,
        )

        if use_attention_gates:
            self.middle_gate = AttentionGate(
                skip_channels=feature_channels,
                gating_channels=decoder_channels,
                intermediate_channels=decoder_channels // 2,
            )
            self.high_gate = AttentionGate(
                skip_channels=feature_channels,
                gating_channels=decoder_channels,
                intermediate_channels=decoder_channels // 2,
            )

        self.output_layer = nn.Conv2d(
            decoder_channels,
            output_channels,
            kernel_size=1,
        )

    def forward(
        self,
        features: List[torch.Tensor],
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        if len(features) != 3:
            raise ValueError("The decoder expects exactly three feature maps.")

        high_features, middle_features, low_features = features

        # Start from the coarsest map and gradually add finer detail.
        x = self.coarse_block(low_features)

        x = F.interpolate(
            x,
            size=middle_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        middle_skip = middle_features

        if self.use_attention_gates:
            middle_skip = self.middle_gate(middle_skip, x)

        x = self.middle_block(torch.cat([x, middle_skip], dim=1))

        x = F.interpolate(
            x,
            size=high_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        high_skip = high_features

        if self.use_attention_gates:
            high_skip = self.high_gate(high_skip, x)

        x = self.high_block(torch.cat([x, high_skip], dim=1))

        # Return logits at exactly the same H and W as the input image.
        x = F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        return self.output_layer(x)


class SAM2DamageModel(nn.Module):
    """
    Full pre/post disaster model.

    The same SAM2 encoder processes both images. Adapters and fusion layers
    learn the satellite-specific changes, then two decoders predict buildings
    and damage classes.
    """

    def __init__(
        self,
        model_cfg: str,
        checkpoint_path: str,
        device: str,
        freeze_encoder: bool = True,
        use_attention_gates: bool = True,
        adapter_bottleneck_channels: int = 64,
        decoder_channels: int = 128,
    ) -> None:
        super().__init__()

        self.freeze_encoder = freeze_encoder

        # Build SAM2 correctly, but keep only its image encoder.
        full_sam2_model = build_sam2(
            model_cfg,
            checkpoint_path,
            device=device,
        )
        self.image_encoder = full_sam2_model.image_encoder
        del full_sam2_model

        # The actual SAM2 source says neck.d_model is the FPN channel count.
        self.feature_channels = self.image_encoder.neck.d_model

        if freeze_encoder:
            for parameter in self.image_encoder.parameters():
                parameter.requires_grad = False

        # One adapter and fusion module for each of SAM2's three FPN scales.
        self.adapters = nn.ModuleList(
            [
                BottleneckAdapter(
                    self.feature_channels,
                    adapter_bottleneck_channels,
                )
                for _ in range(3)
            ]
        )
        self.fusion_layers = nn.ModuleList(
            [FeatureFusion(self.feature_channels) for _ in range(3)]
        )

        # Localization does not use attention gates. Damage can use them.
        self.localization_decoder = MultiScaleDecoder(
            feature_channels=self.feature_channels,
            decoder_channels=decoder_channels,
            output_channels=1,
            use_attention_gates=False,
        )
        self.damage_decoder = MultiScaleDecoder(
            feature_channels=self.feature_channels,
            decoder_channels=decoder_channels,
            output_channels=5,
            use_attention_gates=use_attention_gates,
        )

        # This also moves newly-created adapters and decoders to the device.
        self.to(device)

        self.print_parameter_summary()

    def train(self, mode: bool = True) -> "SAM2DamageModel":
        """
        Keep a frozen encoder in evaluation mode during training.

        This stops frozen BatchNorm-like layers from changing their statistics.
        """
        super().train(mode)

        if self.freeze_encoder:
            self.image_encoder.eval()

        return self

    def encode_image(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Run SAM2 and return its three FPN feature maps."""
        encoder_output = self.image_encoder(image)
        features = encoder_output["backbone_fpn"]

        if len(features) != 3:
            raise ValueError(
                f"Expected 3 SAM2 feature maps, but received {len(features)}."
            )

        return features

    def forward(
        self,
        pre_image: torch.Tensor,
        post_image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return building logits and 5-class damage logits."""
        if pre_image.shape != post_image.shape:
            raise ValueError("Pre- and post-disaster images must have the same shape.")

        if pre_image.ndim != 4 or pre_image.shape[1] != 3:
            raise ValueError("Images must have shape (B, 3, H, W).")

        output_size = pre_image.shape[-2:]

        pre_features = self.encode_image(pre_image)
        post_features = self.encode_image(post_image)

        fused_features = []

        for pre, post, adapter, fusion in zip(
            pre_features,
            post_features,
            self.adapters,
            self.fusion_layers,
        ):
            adapted_pre = adapter(pre)
            adapted_post = adapter(post)
            fused_features.append(fusion(adapted_pre, adapted_post))

        building_logits = self.localization_decoder(
            fused_features,
            output_size,
        )
        damage_logits = self.damage_decoder(
            fused_features,
            output_size,
        )

        return building_logits, damage_logits

    def print_parameter_summary(self) -> None:
        """Print how much of the model is trainable."""
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Frozen parameters: {total - trainable:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2 damage-model shape test")
    parser.add_argument(
        "--checkpoint",
        default="sam2/checkpoints/sam2.1_hiera_tiny.pt",
    )
    parser.add_argument(
        "--config",
        default="configs/sam2.1/sam2.1_hiera_t.yaml",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
        help="Use 128 for a quick CPU smoke test; real images are 1024.",
    )
    parser.add_argument(
        "--unfreeze_encoder",
        action="store_true",
        help="Train SAM2's encoder too (normally leave it frozen).",
    )
    parser.add_argument(
        "--disable_attention_gates",
        action="store_true",
        help="Use this later for an attention-gate ablation.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SAM2DamageModel(
        model_cfg=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
        freeze_encoder=not args.unfreeze_encoder,
        use_attention_gates=not args.disable_attention_gates,
    )
    model.eval()

    dummy_pre_image = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    dummy_post_image = torch.randn(1, 3, args.image_size, args.image_size, device=device)

    with torch.no_grad():
        building_logits, damage_logits = model(
            dummy_pre_image,
            dummy_post_image,
        )

    print("Input shape:", tuple(dummy_pre_image.shape))
    print("Building logits shape:", tuple(building_logits.shape))
    print("Damage logits shape:", tuple(damage_logits.shape))