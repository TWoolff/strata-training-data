"""ControlNet-based UV texture inpainting (Model 5 — texture_inpainting).

Fills unobserved UV texture regions when unwrapping a 2D character painting
onto a 3D mesh.  Uses Stable Diffusion 1.5 Inpainting as the base model
with a ControlNet conditioned on UV position + normal maps for 3D-aware
inpainting.

The ControlNet is fine-tuned while SD weights remain frozen.

Training input:
    - partial UV texture (RGBA, observed regions only)
    - inpainting mask (1=needs inpainting, 0=observed)
    - UV position map (RGB, per-texel 3D world position)
    - UV normal map (RGB, per-texel surface normal)

Output: completed RGBA UV texture (512x512)

At inference, a hard compositing step guarantees pixel preservation::

    final = partial * (1 - mask) + inpainted * mask

Pure Python + PyTorch + diffusers (no Blender dependency).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SD 1.5 Inpainting model ID
SD_INPAINTING_MODEL_ID = "runwayml/stable-diffusion-inpainting"

# Number of ControlNet conditioning channels: position (3) + normal (3)
CONTROLNET_CONDITION_CHANNELS = 6

# Default inference steps (can be overridden)
DEFAULT_NUM_INFERENCE_STEPS = 50


# ---------------------------------------------------------------------------
# ControlNet wrapper for UV geometry conditioning
# ---------------------------------------------------------------------------


class TextureInpaintingControlNet(nn.Module):
    """Wrapper that loads/creates a ControlNet for UV geometry conditioning.

    The ControlNet takes a 6-channel input (UV position map + UV normal map)
    and provides conditioning to the SD inpainting UNet.

    Args:
        pretrained_path: Path to pretrained ControlNet weights, or None to
            initialize from the SD inpainting UNet (recommended for fine-tuning).
        dtype: Model dtype (default: float32, use float16 for A100 training).
    """

    def __init__(
        self,
        pretrained_path: str | Path | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self._controlnet = None
        self._pretrained_path = pretrained_path

    def load(self, sd_model_id: str = SD_INPAINTING_MODEL_ID) -> None:
        """Lazily load the ControlNet model.

        If *pretrained_path* was given, loads from there.  Otherwise,
        initializes from the SD UNet weights (standard ControlNet init).
        """
        from diffusers import ControlNetModel

        if self._pretrained_path is not None:
            logger.info("Loading ControlNet from %s", self._pretrained_path)
            self._controlnet = ControlNetModel.from_pretrained(
                str(self._pretrained_path),
                torch_dtype=self.dtype,
            )
        else:
            logger.info(
                "Initializing ControlNet from SD UNet (%s) with %d condition channels",
                sd_model_id,
                CONTROLNET_CONDITION_CHANNELS,
            )
            self._controlnet = ControlNetModel.from_unet(
                self._load_sd_unet(sd_model_id),
                conditioning_channels=CONTROLNET_CONDITION_CHANNELS,
            )

        # Ensure correct dtype
        self._controlnet = self._controlnet.to(dtype=self.dtype)
        logger.info(
            "ControlNet loaded: %s trainable parameters",
            f"{sum(p.numel() for p in self._controlnet.parameters() if p.requires_grad):,}",
        )

    @staticmethod
    def _load_sd_unet(model_id: str) -> nn.Module:
        """Load the UNet from an SD inpainting model for ControlNet init."""
        from diffusers import UNet2DConditionModel

        return UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
        )

    @property
    def controlnet(self):
        """Access the underlying ControlNet model."""
        if self._controlnet is None:
            raise RuntimeError("Call .load() before accessing the ControlNet")
        return self._controlnet

    def save_pretrained(self, path: str | Path) -> None:
        """Save ControlNet weights to disk."""
        self.controlnet.save_pretrained(str(path))
        logger.info("ControlNet saved to %s", path)


# ---------------------------------------------------------------------------
# Full inpainting pipeline wrapper
# ---------------------------------------------------------------------------


class TextureInpaintingPipeline:
    """Wraps SD Inpainting + ControlNet for UV texture completion.

    This is NOT an nn.Module — it manages the diffusers pipeline for
    inference.  For training, use the ControlNet + SD components directly.

    Args:
        controlnet_path: Path to fine-tuned ControlNet weights.
        sd_model_id: Stable Diffusion inpainting model ID.
        device: Torch device.
        dtype: Model dtype (float16 recommended for inference).
    """

    def __init__(
        self,
        controlnet_path: str | Path,
        sd_model_id: str = SD_INPAINTING_MODEL_ID,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        from diffusers import (
            ControlNetModel,
            StableDiffusionControlNetInpaintPipeline,
        )

        controlnet = ControlNetModel.from_pretrained(
            str(controlnet_path),
            torch_dtype=dtype,
        )
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            sd_model_id,
            controlnet=controlnet,
            torch_dtype=dtype,
        ).to(device)

        self.device = torch.device(device) if isinstance(device, str) else device

    @torch.no_grad()
    def inpaint(
        self,
        partial_texture: torch.Tensor,
        inpainting_mask: torch.Tensor,
        position_map: torch.Tensor,
        normal_map: torch.Tensor,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        guidance_scale: float = 7.5,
    ) -> torch.Tensor:
        """Run inpainting with pixel-preservation compositing.

        Args:
            partial_texture: [B, 4, H, W] float32 RGBA in [0, 1].
            inpainting_mask: [B, 1, H, W] float32 (1=inpaint, 0=keep).
            position_map: [B, 3, H, W] float32 UV position in [0, 1].
            normal_map: [B, 3, H, W] float32 UV normal in [0, 1].
            num_inference_steps: Diffusion steps.
            guidance_scale: Classifier-free guidance scale.

        Returns:
            [B, 4, H, W] float32 completed RGBA with pixel preservation.
        """
        from diffusers.utils import make_image_grid
        from PIL import Image
        import numpy as np

        results = []
        for i in range(partial_texture.shape[0]):
            # Convert to PIL for the diffusers pipeline
            rgb = partial_texture[i, :3].permute(1, 2, 0).cpu().numpy()
            rgb_pil = Image.fromarray((rgb * 255).clip(0, 255).astype(np.uint8))

            mask = inpainting_mask[i, 0].cpu().numpy()
            mask_pil = Image.fromarray((mask * 255).clip(0, 255).astype(np.uint8))

            # ControlNet condition: concat position + normal
            control = torch.cat([position_map[i], normal_map[i]], dim=0)  # [6, H, W]
            control_pil = Image.fromarray(
                (control[:3].permute(1, 2, 0).cpu().numpy() * 255)
                .clip(0, 255)
                .astype(np.uint8)
            )

            output = self.pipe(
                prompt="",
                image=rgb_pil,
                mask_image=mask_pil,
                control_image=control_pil,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]

            # Convert back to tensor
            out_arr = np.array(output).astype(np.float32) / 255.0
            out_tensor = torch.from_numpy(out_arr).permute(2, 0, 1)  # [3, H, W]

            # Add alpha channel (use partial texture alpha where observed,
            # full opacity where inpainted)
            alpha = partial_texture[i, 3:4].cpu()
            mask_t = inpainting_mask[i].cpu()
            alpha_out = alpha * (1 - mask_t) + mask_t  # observed alpha + inpainted = 1.0
            out_rgba = torch.cat([out_tensor, alpha_out], dim=0)  # [4, H, W]

            # Pixel preservation: compositing guarantee
            mask_t_4ch = mask_t.expand(4, -1, -1)
            partial_i = partial_texture[i].cpu()
            final = partial_i * (1 - mask_t_4ch) + out_rgba * mask_t_4ch

            results.append(final)

        return torch.stack(results)


# ---------------------------------------------------------------------------
# Legacy U-Net model (kept for backwards compatibility / distillation)
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


class TextureInpaintingModel(nn.Module):
    """Legacy U-Net for 512x512 UV texture inpainting.

    Kept for backwards compatibility and potential distillation from the
    ControlNet model.  New training should use TextureInpaintingControlNet.

    Args:
        in_channels: Number of input channels (default 5: RGBA + mask).
        out_channels: Number of output channels (default 4: RGBA).
    """

    def __init__(self, in_channels: int = 5, out_channels: int = 4) -> None:
        super().__init__()

        # Encoder
        self.e1 = _DownBlock(in_channels, 64, use_bn=False)
        self.e2 = _DownBlock(64, 128)
        self.e3 = _DownBlock(128, 256)
        self.e4 = _DownBlock(256, 512)
        self.e5 = _DownBlock(512, 512)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # Decoder (input channels doubled by skip connections)
        self.d1 = _UpBlock(512, 512, dropout=0.5)
        self.d2 = _UpBlock(1024, 512)
        self.d3 = _UpBlock(1024, 256)
        self.d4 = _UpBlock(512, 128)
        self.d5 = _UpBlock(256, 64)

        # Final output
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
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

        return {"inpainted": out}
