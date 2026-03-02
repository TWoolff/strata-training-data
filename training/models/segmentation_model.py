"""Multi-head DeepLabV3+ segmentation model for Strata.

Three output heads sharing a MobileNetV3-Large backbone:
- **segmentation**: 22-class raw logits ``[B, 22, H, W]``
- **draw_order**: depth in [0, 1] via sigmoid ``[B, 1, H, W]``
- **confidence**: foreground confidence via sigmoid ``[B, 1, H, W]``

ONNX contract (from ``strata/src-tauri/src/ai/segmentation.rs``):
- Input ``"input"``: ``[1, 3, 512, 512]`` float32
- Output ``"segmentation"``: ``[1, 22, 512, 512]`` raw logits
- Output ``"draw_order"``: ``[1, 1, 512, 512]`` sigmoid
- Output ``"confidence"``: ``[1, 1, 512, 512]`` sigmoid
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

logger = logging.getLogger(__name__)

NUM_CLASSES: int = 22  # 20 body regions + background + accessory


def _make_aux_head(in_channels: int) -> nn.Sequential:
    """Build a lightweight 2-conv auxiliary head (draw_order or confidence)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 1, kernel_size=1),
    )


class SegmentationModel(nn.Module):
    """Multi-head DeepLabV3+ with MobileNetV3-Large backbone.

    Args:
        num_classes: Number of segmentation classes (default 22).
        pretrained_backbone: Use ImageNet-pretrained MobileNetV3-Large weights.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained_backbone: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Load the torchvision DeepLabV3 model to extract its components
        weights_backbone = "IMAGENET1K_V1" if pretrained_backbone else None
        base = deeplabv3_mobilenet_v3_large(
            weights=None,
            weights_backbone=weights_backbone,
            num_classes=num_classes,
        )

        self.backbone = base.backbone
        self.classifier = base.classifier  # ASPP → segmentation logits

        # Determine backbone output channels via dry run
        backbone_channels = self._detect_backbone_channels()
        logger.info("Backbone output channels: %d", backbone_channels)

        # Auxiliary heads branch from backbone features
        self.draw_order_head = _make_aux_head(backbone_channels)
        self.confidence_head = _make_aux_head(backbone_channels)

    def _detect_backbone_channels(self) -> int:
        """Run a dummy forward pass to detect backbone output channel count."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64)
            features = self.backbone(dummy)
            out = features["out"]
        channels = out.shape[1]
        if channels != 960:
            logger.warning(
                "Expected 960 backbone channels (MobileNetV3-Large), got %d. "
                "Auxiliary heads will adapt automatically.",
                channels,
            )
        return channels

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass producing all three output heads.

        Args:
            x: Input tensor ``[B, 3, H, W]``.

        Returns:
            Dict with keys ``"segmentation"``, ``"draw_order"``, ``"confidence"``.
            All outputs are upsampled to the input spatial resolution.
        """
        input_shape = x.shape[-2:]

        # Shared backbone
        features = self.backbone(x)
        backbone_out = features["out"]  # [B, C, H/16, W/16]

        # Segmentation head (ASPP classifier, produces num_classes channels)
        seg = self.classifier(backbone_out)
        seg = F.interpolate(seg, size=input_shape, mode="bilinear", align_corners=False)

        # Draw order head
        draw_order = self.draw_order_head(backbone_out)
        draw_order = F.interpolate(
            draw_order, size=input_shape, mode="bilinear", align_corners=False
        )
        draw_order = torch.sigmoid(draw_order)

        # Confidence head
        confidence = self.confidence_head(backbone_out)
        confidence = F.interpolate(
            confidence, size=input_shape, mode="bilinear", align_corners=False
        )
        confidence = torch.sigmoid(confidence)

        return {
            "segmentation": seg,
            "draw_order": draw_order,
            "confidence": confidence,
        }
