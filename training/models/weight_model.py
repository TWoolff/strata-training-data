"""Weight prediction model for Strata.

Predicts per-vertex bone weights and confidence from an RGB image input,
using a MobileNetV3-Large backbone with a dense prediction head.

The model outputs per-pixel bone weights at the input resolution, which are
then sampled at vertex positions by the Rust runtime.

ONNX contract (from ``strata/src-tauri/src/ai/weights.rs``):
- Input ``"input"``: ``[1, 3, 512, 512]`` float32
- Output ``"weights"``: ``[1, 20, N, 1]`` softmax bone weights (N = vertex count, dynamic)
- Output ``"confidence"``: ``[1, 1, N, 1]`` per-vertex confidence
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

logger = logging.getLogger(__name__)

NUM_BONES: int = 20


def _make_head(in_channels: int, out_channels: int) -> nn.Sequential:
    """Build a lightweight 2-conv head."""
    return nn.Sequential(
        nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, out_channels, kernel_size=1),
    )


class WeightModel(nn.Module):
    """Per-pixel bone weight prediction with MobileNetV3-Large backbone.

    Args:
        num_bones: Number of bones to predict weights for (default 20).
        pretrained_backbone: Use ImageNet-pretrained MobileNetV3-Large weights.
    """

    def __init__(self, num_bones: int = NUM_BONES, pretrained_backbone: bool = True) -> None:
        super().__init__()
        self.num_bones = num_bones

        weights_backbone = "IMAGENET1K_V1" if pretrained_backbone else None
        base = deeplabv3_mobilenet_v3_large(
            weights=None,
            weights_backbone=weights_backbone,
            num_classes=num_bones,
        )

        self.backbone = base.backbone

        backbone_channels = self._detect_backbone_channels()
        logger.info("Weight model backbone channels: %d", backbone_channels)

        # Weight head: softmax over bones per pixel
        self.weight_head = _make_head(backbone_channels, num_bones)

        # Confidence head: per-pixel confidence
        self.confidence_head = _make_head(backbone_channels, 1)

    def _detect_backbone_channels(self) -> int:
        """Run a dummy forward pass to detect backbone output channel count."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64)
            features = self.backbone(dummy)
            out = features["out"]
        channels = out.shape[1]
        if channels != 960:
            logger.warning(
                "Expected 960 backbone channels (MobileNetV3-Large), got %d.",
                channels,
            )
        return channels

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass producing per-pixel bone weights and confidence.

        Args:
            x: Input tensor ``[B, 3, H, W]``.

        Returns:
            Dict with keys ``"weights"`` ``[B, num_bones, H, W]`` (softmax over bones)
            and ``"confidence"`` ``[B, 1, H, W]`` (sigmoid).
        """
        input_shape = x.shape[-2:]

        features = self.backbone(x)
        backbone_out = features["out"]  # [B, C, H/16, W/16]

        # Weight head — softmax over bone dimension
        weights = self.weight_head(backbone_out)
        weights = F.interpolate(weights, size=input_shape, mode="bilinear", align_corners=False)
        weights = F.softmax(weights, dim=1)  # [B, num_bones, H, W]

        # Confidence head
        confidence = self.confidence_head(backbone_out)
        confidence = F.interpolate(
            confidence, size=input_shape, mode="bilinear", align_corners=False
        )
        confidence = torch.sigmoid(confidence)  # [B, 1, H, W]

        return {
            "weights": weights,
            "confidence": confidence,
        }
