"""PatchGAN discriminator for segmentation map quality (training only).

Classifies 70×70 overlapping patches of (image, segmentation) pairs as
real or fake.  Discarded at ONNX export — zero inference cost.

Architecture follows pix2pix (Isola et al. 2017).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SegPatchDiscriminator(nn.Module):
    """PatchGAN discriminator for segmentation maps.

    Input: concatenation of RGB image ``[B, 3, H, W]`` + seg softmax
    ``[B, num_classes, H, W]``.

    Output: ``[B, 1, ~H/8, ~W/8]`` patch predictions (real/fake).

    Only used during training — never exported to ONNX.

    Args:
        in_channels: 3 (image) + num_classes (seg).  Default 25 (3+22).
    """

    def __init__(self, in_channels: int = 25) -> None:
        super().__init__()
        self.model = nn.Sequential(
            # Layer 1: no BN
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 4: stride 1
            nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: 1 channel
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
