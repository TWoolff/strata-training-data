"""U-Net for unified view synthesis (Model 6 replacement).

Generates any target view RGBA from two source view RGBAs + a target angle.
Replaces the back-view-only model with a general view synthesis model.

ONNX contract:

- Input ``"input"``: ``[1, 9, 512, 512]`` float32
    - Channels 0-3: source view A (RGBA)
    - Channels 4-7: source view B (RGBA)
    - Channel 8: target angle map (constant value broadcast to H×W)
- Output ``"output"``: ``[1, 4, 512, 512]`` float32 (target view RGBA, 0-1)
- Output filename: ``view_synthesis.onnx``

Angle encoding (normalized 0-1):
    0.0 = front
    0.2 = front 3/4
    0.4 = side
    0.6 = back 3/4
    0.8 = back

Pure Python + PyTorch (no Blender dependency).
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Angle constants
# ---------------------------------------------------------------------------

VIEW_ANGLES: dict[str, float] = {
    "front": 0.0,
    "threequarter": 0.2,
    "frontthreequarter": 0.2,  # alias
    "side": 0.4,
    "back_threequarter": 0.6,
    "back": 0.8,
}


# ---------------------------------------------------------------------------
# Building blocks (same as BackViewModel)
# ---------------------------------------------------------------------------


class _DownBlock(nn.Module):
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
# View Synthesis U-Net
# ---------------------------------------------------------------------------


class ViewSynthesisModel(nn.Module):
    """U-Net for 512×512 view synthesis.

    Takes two source view RGBAs (8ch) + target angle map (1ch) = 9 channels.
    Returns target view RGBA (4ch) via sigmoid.

    Args:
        in_channels: Number of input channels (default 9: 2×RGBA + angle).
        out_channels: Number of output channels (default 4: RGBA).
    """

    def __init__(self, in_channels: int = 9, out_channels: int = 4) -> None:
        super().__init__()

        # Encoder
        self.e1 = _DownBlock(in_channels, 64, use_bn=False)  # 256
        self.e2 = _DownBlock(64, 128)   # 128
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
        self.d4 = _UpBlock(512, 128)   # 128 (+ e2 skip = 256)
        self.d5 = _UpBlock(256, 64)    # 256 (+ e1 skip = 128)

        # Final output
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # RGBA in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: ``[B, 9, H, W]`` — source A (4ch) + source B (4ch) + angle map (1ch).

        Returns:
            Dict with ``"output"``: ``[B, 4, H, W]`` target view RGBA.
        """
        s1 = self.e1(x)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        s5 = self.e5(s4)

        b = self.bottleneck(s5)

        d = self.d1(b)
        d = self.d2(torch.cat([d, s5], 1))
        d = self.d3(torch.cat([d, s4], 1))
        d = self.d4(torch.cat([d, s3], 1))
        d = self.d5(torch.cat([d, s2], 1))

        out = self.final(torch.cat([d, s1], 1))

        return {"output": out}
