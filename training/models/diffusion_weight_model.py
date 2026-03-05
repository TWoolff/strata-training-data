"""Diffusion-enhanced weight prediction model for Strata (Model 4).

Dual-input variant of :class:`WeightPredictionModel` that fuses standard
31-dim per-vertex features with encoder features from the segmentation
model's backbone. The visual context helps predict better skinning weights
for characters with unusual proportions (chibi, elongated limbs, loose clothing).

ONNX contract (from ``strata/src-tauri/src/ai/weights.rs``):

- Input ``"features"``: ``[1, 31, 2048, 1]`` float32
- Input ``"diffusion_features"``: ``[1, C, 2048, 1]`` float32
- Output ``"weights"``: ``[1, 20, 2048, 1]`` raw logits (softmax at inference)
- Output ``"confidence"``: ``[1, 1, 2048, 1]`` raw logits (sigmoid at inference)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

NUM_FEATURES: int = 31
NUM_BONES: int = 20
DEFAULT_ENCODER_CHANNELS: int = 960  # MobileNetV3-Large backbone output


class DiffusionWeightPredictionModel(nn.Module):
    """Dual-input per-vertex weight prediction via 1x1 Conv2d MLP.

    Concatenates standard vertex features with encoder features from the
    segmentation model, then processes through a shared MLP.

    Args:
        num_features: Standard input feature dimension per vertex (default 31).
        num_bones: Number of bones to predict weights for (default 20).
        encoder_channels: Channel count of encoder features (default 960).
    """

    def __init__(
        self,
        num_features: int = NUM_FEATURES,
        num_bones: int = NUM_BONES,
        encoder_channels: int = DEFAULT_ENCODER_CHANNELS,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_bones = num_bones
        self.encoder_channels = encoder_channels

        fused_dim = num_features + encoder_channels

        # Weight prediction head: fused MLP via 1x1 Conv2d
        self.weight_mlp = nn.Sequential(
            nn.Conv2d(fused_dim, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_bones, kernel_size=1),  # [B, 20, N, 1]
        )

        # Confidence prediction head
        self.confidence_mlp = nn.Sequential(
            nn.Conv2d(fused_dim, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),  # [B, 1, N, 1]
        )

        logger.info(
            "DiffusionWeightPredictionModel: %d + %d features → %d bones",
            num_features,
            encoder_channels,
            num_bones,
        )

    def forward(
        self,
        features: torch.Tensor,
        diffusion_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with dual inputs.

        Args:
            features: Standard vertex features ``[B, 31, N, 1]``.
            diffusion_features: Encoder features ``[B, C, N, 1]``.

        Returns:
            Dict with keys:
            - ``"weights"``: ``[B, 20, N, 1]`` raw logits (softmax at inference)
            - ``"confidence"``: ``[B, 1, N, 1]`` raw logits (sigmoid at inference)
        """
        # Fuse along channel dimension
        x = torch.cat([features, diffusion_features], dim=1)  # [B, 31+C, N, 1]

        weights = self.weight_mlp(x)  # [B, 20, N, 1]
        confidence = self.confidence_mlp(x)  # [B, 1, N, 1]

        return {
            "weights": weights,
            "confidence": confidence,
        }
