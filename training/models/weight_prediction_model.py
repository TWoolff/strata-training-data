"""Per-vertex weight prediction model for Strata.

Predicts per-vertex bone weights and confidence from a 31-dimensional
per-vertex feature tensor, using 1x1 Conv2d layers as a shared-weight MLP.

Unlike the image-based :class:`WeightModel`, this model processes vertices
directly — each vertex is treated independently with shared weights across
the vertex dimension.

Using 1x1 convolutions instead of linear layers makes the model naturally
handle variable vertex counts (the N dimension) and matches the
``[B, C, N, 1]`` tensor layout the Rust runtime expects.

ONNX contract (from ``strata/src-tauri/src/ai/weights.rs``):

- Input ``"input"``: ``[1, 31, 2048, 1]`` float32
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


class WeightPredictionModel(nn.Module):
    """Per-vertex bone weight prediction via 1x1 Conv2d MLP.

    Processes each vertex independently with shared weights. The 1x1 convolution
    treats each vertex as a single "pixel" in the spatial dimension.

    Args:
        num_features: Input feature dimension per vertex (default 31).
        num_bones: Number of bones to predict weights for (default 20).
    """

    def __init__(
        self,
        num_features: int = NUM_FEATURES,
        num_bones: int = NUM_BONES,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_bones = num_bones

        # Weight prediction head: per-vertex MLP via 1x1 Conv2d
        self.weight_mlp = nn.Sequential(
            nn.Conv2d(num_features, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
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
            nn.Conv2d(num_features, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),  # [B, 1, N, 1]
        )

        logger.info(
            "WeightPredictionModel: %d features → %d bones",
            num_features,
            num_bones,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass producing per-vertex weight logits and confidence.

        Args:
            x: Input tensor ``[B, 31, N, 1]``.

        Returns:
            Dict with keys:
            - ``"weights"``: ``[B, 20, N, 1]`` raw logits (softmax at inference)
            - ``"confidence"``: ``[B, 1, N, 1]`` raw logits (sigmoid at inference)
        """
        weights = self.weight_mlp(x)  # [B, 20, N, 1]
        confidence = self.confidence_mlp(x)  # [B, 1, N, 1]

        return {
            "weights": weights,
            "confidence": confidence,
        }
