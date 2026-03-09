"""U-Net for back view generation from front + 3/4 view inputs (Model 6).

Multi-view conditioned encoder-decoder that generates a back view RGBA image
from concatenated front and three-quarter view inputs.

ONNX contract (matches ``src-tauri/src/ai/back_view_gen.rs``):

- Input ``"input"``: ``[1, 8, 512, 512]`` float32 (front RGBA + 3/4 RGBA)
- Output ``"output"``: ``[1, 4, 512, 512]`` float32 (back view RGBA, 0-1)
- Output filename: ``back_view_generation.onnx``

Pure Python + PyTorch (no Blender dependency).
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _DownBlock(nn.Module):
    """Encoder block: Conv(stride=2) -> BN -> LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not use_bn),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UpBlock(nn.Module):
    """Decoder block: ConvTranspose(stride=2) -> BN -> ReLU -> (optional dropout)."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------


class BackViewModel(nn.Module):
    """U-Net for 512x512 back view generation.

    Takes concatenated front RGBA (4ch) + three-quarter RGBA (4ch) = 8 channels.
    Returns back view RGBA (4ch) via sigmoid.

    Args:
        in_channels: Number of input channels (default 8: front + 3/4 RGBA).
        out_channels: Number of output channels (default 4: RGBA).
    """

    def __init__(self, in_channels: int = 8, out_channels: int = 4) -> None:
        super().__init__()

        # Encoder
        self.e1 = _DownBlock(in_channels, 64, use_bn=False)  # 256
        self.e2 = _DownBlock(64, 128)  # 128
        self.e3 = _DownBlock(128, 256)  # 64
        self.e4 = _DownBlock(256, 512)  # 32
        self.e5 = _DownBlock(512, 512)  # 16

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),  # 8
            nn.ReLU(inplace=True),
        )

        # Decoder (input channels doubled by skip connections)
        self.d1 = _UpBlock(512, 512, dropout=0.5)  # 16 (+ e5 skip = 1024)
        self.d2 = _UpBlock(1024, 512)  # 32 (+ e4 skip = 1024)
        self.d3 = _UpBlock(1024, 256)  # 64 (+ e3 skip = 512)
        self.d4 = _UpBlock(512, 128)  # 128 (+ e2 skip = 256)
        self.d5 = _UpBlock(256, 64)  # 256 (+ e1 skip = 128)

        # Final output
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # RGBA in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: ``[B, 8, H, W]`` concatenated front RGBA + 3/4 RGBA.

        Returns:
            Dict with ``"output"``: ``[B, 4, H, W]`` back view RGBA.
        """
        # Encoder
        s1 = self.e1(x)  # [B, 64, 256, 256]
        s2 = self.e2(s1)  # [B, 128, 128, 128]
        s3 = self.e3(s2)  # [B, 256, 64, 64]
        s4 = self.e4(s3)  # [B, 512, 32, 32]
        s5 = self.e5(s4)  # [B, 512, 16, 16]

        # Bottleneck
        b = self.bottleneck(s5)  # [B, 512, 8, 8]

        # Decoder with skip connections
        d = self.d1(b)  # [B, 512, 16, 16]
        d = self.d2(torch.cat([d, s5], 1))  # [B, 512, 32, 32]
        d = self.d3(torch.cat([d, s4], 1))  # [B, 256, 64, 64]
        d = self.d4(torch.cat([d, s3], 1))  # [B, 128, 128, 128]
        d = self.d5(torch.cat([d, s2], 1))  # [B, 64, 256, 256]

        out = self.final(torch.cat([d, s1], 1))  # [B, 4, 512, 512]

        return {"output": out}
