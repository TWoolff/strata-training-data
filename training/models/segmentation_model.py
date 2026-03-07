"""Multi-head DeepLabV3+ segmentation model for Strata.

Five output heads sharing a MobileNetV3-Large backbone:
- **segmentation**: 22-class raw logits ``[B, 22, H, W]``
- **depth**: monocular depth in [0, 1] via sigmoid ``[B, 1, H, W]``
- **normals**: surface normals in [-1, 1] via tanh ``[B, 3, H, W]``
- **confidence**: foreground confidence via sigmoid ``[B, 1, H, W]``
- **encoder_features**: backbone activations for weight prediction ``[B, C, H/8, W/8]``

The depth head is trained with Marigold LCM depth labels. The normals head is
trained with Marigold LCM normal labels. Both are knowledge-distilled from the
Marigold diffusion model into this lightweight MobileNetV3 student — one forward
pass produces segmentation + depth + normals.

ONNX contract:
- Input ``"input"``: ``[1, 3, 512, 512]`` float32
- Output ``"segmentation"``: ``[1, 22, 512, 512]`` raw logits
- Output ``"depth"``: ``[1, 1, 512, 512]`` sigmoid
- Output ``"normals"``: ``[1, 3, 512, 512]`` tanh
- Output ``"confidence"``: ``[1, 1, 512, 512]`` sigmoid
- Output ``"encoder_features"``: ``[1, C, 64, 64]`` raw activations
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

logger = logging.getLogger(__name__)

NUM_CLASSES: int = 22  # 20 body regions + background + accessory
ENCODER_FEATURE_SIZE: int = 64  # Downsampled feature map resolution


def _make_aux_head(in_channels: int, out_channels: int = 1) -> nn.Sequential:
    """Build a lightweight 2-conv auxiliary head."""
    return nn.Sequential(
        nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, out_channels, kernel_size=1),
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
        self.depth_head = _make_aux_head(backbone_channels, out_channels=1)
        self.normals_head = _make_aux_head(backbone_channels, out_channels=3)
        self.confidence_head = _make_aux_head(backbone_channels, out_channels=1)

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
        """Forward pass producing all output heads.

        Args:
            x: Input tensor ``[B, 3, H, W]``.

        Returns:
            Dict with keys ``"segmentation"``, ``"depth"``, ``"normals"``,
            ``"confidence"``, ``"encoder_features"``.
        """
        input_shape = x.shape[-2:]

        # Shared backbone
        features = self.backbone(x)
        backbone_out = features["out"]  # [B, C, H/16, W/16]

        # Segmentation head (ASPP classifier, produces num_classes channels)
        seg = self.classifier(backbone_out)
        seg = F.interpolate(seg, size=input_shape, mode="bilinear", align_corners=False)

        # Depth head (Marigold-distilled)
        depth = self.depth_head(backbone_out)
        depth = F.interpolate(depth, size=input_shape, mode="bilinear", align_corners=False)
        depth = torch.sigmoid(depth)

        # Normals head (Marigold-distilled, 3-channel)
        normals = self.normals_head(backbone_out)
        normals = F.interpolate(normals, size=input_shape, mode="bilinear", align_corners=False)
        normals = torch.tanh(normals)

        # Confidence head
        confidence = self.confidence_head(backbone_out)
        confidence = F.interpolate(
            confidence, size=input_shape, mode="bilinear", align_corners=False
        )
        confidence = torch.sigmoid(confidence)

        # Encoder features (downsampled for weight prediction)
        encoder_features = F.interpolate(
            backbone_out,
            size=(ENCODER_FEATURE_SIZE, ENCODER_FEATURE_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        return {
            "segmentation": seg,
            "depth": depth,
            "normals": normals,
            "confidence": confidence,
            "encoder_features": encoder_features,
        }
