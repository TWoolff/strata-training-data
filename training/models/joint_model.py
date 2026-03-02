"""Joint refinement model for Strata.

Predicts 2D joint offsets, per-joint confidence, and joint presence/visibility
from an RGB image input, using a MobileNetV3-Large backbone with regression heads.

ONNX contract (from ``strata/src-tauri/src/ai/joints.rs``):
- Input ``"input"``: ``[1, 3, 512, 512]`` float32
- Output ``"offsets"``: ``[40]`` (20 joints × 2 xy offsets)
- Output ``"confidence"``: ``[20]`` per-joint confidence
- Output ``"present"``: ``[20]`` joint visibility logits
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large

logger = logging.getLogger(__name__)

NUM_JOINTS: int = 20


class JointModel(nn.Module):
    """Joint refinement model with MobileNetV3-Large backbone.

    Args:
        num_joints: Number of joints to predict (default 20, matching BONE_ORDER).
        pretrained_backbone: Use ImageNet-pretrained MobileNetV3-Large weights.
    """

    def __init__(self, num_joints: int = NUM_JOINTS, pretrained_backbone: bool = True) -> None:
        super().__init__()
        self.num_joints = num_joints

        weights = "IMAGENET1K_V1" if pretrained_backbone else None
        backbone = mobilenet_v3_large(weights=weights)

        # Use all feature layers, drop the classifier
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Detect feature channels
        backbone_channels = self._detect_backbone_channels()
        logger.info("Joint model backbone channels: %d", backbone_channels)

        # Offset head: predicts (x, y) for each joint → flat [num_joints * 2]
        self.offset_head = nn.Sequential(
            nn.Linear(backbone_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_joints * 2),
        )

        # Confidence head: per-joint confidence score
        self.confidence_head = nn.Sequential(
            nn.Linear(backbone_channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_joints),
            nn.Sigmoid(),
        )

        # Presence head: per-joint visibility logit
        self.presence_head = nn.Sequential(
            nn.Linear(backbone_channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_joints),
        )

    def _detect_backbone_channels(self) -> int:
        """Run a dummy forward pass to detect backbone output channel count."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64)
            features = self.features(dummy)
            pooled = self.pool(features)
        return pooled.shape[1]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass producing joint offsets, confidence, and presence.

        Args:
            x: Input tensor ``[B, 3, H, W]``.

        Returns:
            Dict with keys ``"offsets"`` ``[B, num_joints * 2]``,
            ``"confidence"`` ``[B, num_joints]``,
            ``"present"`` ``[B, num_joints]``.
        """
        features = self.features(x)
        pooled = self.pool(features).flatten(1)  # [B, C]

        offsets = self.offset_head(pooled)  # [B, num_joints * 2]
        confidence = self.confidence_head(pooled)  # [B, num_joints]
        present = self.presence_head(pooled)  # [B, num_joints]

        return {
            "offsets": offsets,
            "confidence": confidence,
            "present": present,
        }
