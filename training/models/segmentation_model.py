"""Multi-head DeepLabV3+ segmentation model for Strata.

Five output heads sharing a configurable backbone:
- **segmentation**: 22-class raw logits ``[B, 22, H, W]``
- **depth**: monocular depth in [0, 1] via sigmoid ``[B, 1, H, W]``
- **normals**: surface normals in [-1, 1] via tanh ``[B, 3, H, W]``
- **confidence**: foreground confidence via sigmoid ``[B, 1, H, W]``
- **encoder_features**: backbone activations for weight prediction ``[B, C, H/8, W/8]``

Supported backbones:
- ``mobilenet_v3_large`` (default, ~5M params) — via torchvision DeepLabV3
- ``resnet50`` (~24M params) — via torchvision DeepLabV3
- ``resnet101`` (~43M params) — via torchvision DeepLabV3
- ``efficientnet_b3`` (~12M params) — custom DeepLabV3-style with ASPP

The depth head is trained with Marigold LCM depth labels. The normals head is
trained with Marigold LCM normal labels. Both are knowledge-distilled from the
Marigold diffusion model — one forward pass produces segmentation + depth + normals.

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
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, deeplabv3_resnet50, deeplabv3_resnet101

logger = logging.getLogger(__name__)

NUM_CLASSES: int = 22  # 20 body regions + background + accessory
ENCODER_FEATURE_SIZE: int = 64  # Downsampled feature map resolution

SUPPORTED_BACKBONES = {
    "mobilenet_v3_large",
    "resnet50",
    "resnet101",
    "efficientnet_b3",
}


def _make_aux_head(in_channels: int, out_channels: int = 1) -> nn.Sequential:
    """Build a lightweight 2-conv auxiliary head."""
    return nn.Sequential(
        nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, out_channels, kernel_size=1),
    )


class _ASPPConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, dilation: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _ASPPPooling(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class _ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (same design as torchvision DeepLabV3)."""

    def __init__(self, in_ch: int, atrous_rates: tuple[int, ...] = (12, 24, 36), out_ch: int = 256) -> None:
        super().__init__()
        modules: list[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        ]
        for rate in atrous_rates:
            modules.append(_ASPPConv(in_ch, out_ch, rate))
        modules.append(_ASPPPooling(in_ch, out_ch))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(atrous_rates) + 2), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = [conv(x) for conv in self.convs]
        return self.project(torch.cat(res, dim=1))


class _EfficientNetBackbone(nn.Module):
    """Wrap an EfficientNet as a backbone returning dict with 'out' key."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        eff = efficientnet_b3(weights=weights)
        self.features = eff.features  # Sequential of blocks

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.features(x)
        return {"out": out}


def _build_backbone_and_classifier(
    backbone_name: str, num_classes: int, pretrained_backbone: bool,
) -> tuple[nn.Module, nn.Module]:
    """Build backbone + segmentation classifier for the given backbone name."""
    weights_backbone = "IMAGENET1K_V1" if pretrained_backbone else None

    if backbone_name == "mobilenet_v3_large":
        base = deeplabv3_mobilenet_v3_large(
            weights=None, weights_backbone=weights_backbone, num_classes=num_classes,
        )
        return base.backbone, base.classifier

    if backbone_name == "resnet50":
        base = deeplabv3_resnet50(
            weights=None, weights_backbone=weights_backbone, num_classes=num_classes,
        )
        return base.backbone, base.classifier

    if backbone_name == "resnet101":
        base = deeplabv3_resnet101(
            weights=None, weights_backbone=weights_backbone, num_classes=num_classes,
        )
        return base.backbone, base.classifier

    if backbone_name == "efficientnet_b3":
        backbone = _EfficientNetBackbone(pretrained=pretrained_backbone)
        # Detect output channels
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64)
            ch = backbone(dummy)["out"].shape[1]
        # Build ASPP classifier
        classifier = nn.Sequential(
            OrderedDict([
                ("aspp", _ASPP(ch, atrous_rates=(12, 24, 36), out_ch=256)),
                ("conv", nn.Conv2d(256, num_classes, 1)),
            ])
        )
        return backbone, classifier

    raise ValueError(f"Unsupported backbone: {backbone_name!r}. Choose from {SUPPORTED_BACKBONES}")


class SegmentationModel(nn.Module):
    """Multi-head DeepLabV3+ with configurable backbone.

    Args:
        num_classes: Number of segmentation classes (default 22).
        pretrained_backbone: Use ImageNet-pretrained backbone weights.
        backbone: Backbone architecture name. One of:
            ``mobilenet_v3_large``, ``resnet50``, ``resnet101``, ``efficientnet_b3``.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained_backbone: bool = True,
        backbone: str = "mobilenet_v3_large",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone

        self.backbone, self.classifier = _build_backbone_and_classifier(
            backbone, num_classes, pretrained_backbone,
        )

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
        return out.shape[1]

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
