#!/usr/bin/env python3
"""Apply trained ControlNet texture inpainting to a partial UV texture.

Loads the v3 ControlNet + frozen SD 1.5 Inpainting and runs inpainting on a
partial texture, conditioned on UV position + normal maps. Composites the
inpainted output with the original projection (pixel-preservation).

Usage::

    python3 scripts/apply_texture_inpaint.py \\
        --checkpoint ./checkpoints/texture_inpainting_controlnet_v3/best/controlnet \\
        --partial output/lichtung_test/final/projected_partial.png \\
        --mask output/lichtung_test/final/inpainting_mask.png \\
        --position output/lichtung_test/final/position_map.png \\
        --normal output/lichtung_test/final/normal_map.png \\
        --output output/lichtung_test/final/complete_texture_v3.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

SD_MODEL = "runwayml/stable-diffusion-inpainting"


def load_image(path: Path, mode: str = "RGB", size: int = 512) -> torch.Tensor:
    img = Image.open(path).convert(mode)
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS if mode == "RGB" else Image.NEAREST)
    arr = np.array(img).astype(np.float32) / 255.0
    if mode == "L":
        arr = arr[None, ...]  # (1, H, W)
    else:
        arr = arr.transpose(2, 0, 1)  # (C, H, W)
    return torch.from_numpy(arr)


@torch.no_grad()
def inpaint(
    controlnet, unet, vae, text_encoder, tokenizer, scheduler,
    partial_rgb, mask, control, device, num_inference_steps=30,
):
    """Run manual denoising loop matching the validate() function."""
    weight_dtype = next(vae.parameters()).dtype

    # Add batch dim
    partial_rgb = partial_rgb.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    control = control.unsqueeze(0).to(device)

    # Empty text embedding
    tok = tokenizer("", return_tensors="pt", padding="max_length",
                    max_length=tokenizer.model_max_length, truncation=True)
    encoder_hidden_states = text_encoder(tok.input_ids.to(device))[0]

    # Encode partial image
    partial_norm = (partial_rgb * 2.0 - 1.0).to(dtype=weight_dtype)
    partial_latents = vae.encode(partial_norm).latent_dist.mean * vae.config.scaling_factor
    mask_latent = F.interpolate(mask, size=partial_latents.shape[2:], mode="nearest").to(dtype=weight_dtype)

    # Resize control to image size
    control_resized = F.interpolate(
        control, size=(partial_rgb.shape[2], partial_rgb.shape[3]),
        mode="bilinear", align_corners=False,
    ).to(dtype=weight_dtype)

    # Set up scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)

    # Start from noise
    latents = torch.randn_like(partial_latents)

    for t in scheduler.timesteps:
        t_batch = t.expand(1)
        unet_input = torch.cat([latents, mask_latent, partial_latents], dim=1)

        with torch.amp.autocast(device.type):
            down_samples, mid_sample = controlnet(
                unet_input, t_batch,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=control_resized,
                return_dict=False,
            )
            noise_pred = unet(
                unet_input, t_batch,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_samples,
                mid_block_additional_residual=mid_sample,
            ).sample

        latents = scheduler.step(noise_pred.float(), t, latents).prev_sample

    # Decode
    with torch.amp.autocast(device.type):
        decoded = vae.decode(latents / vae.config.scaling_factor).sample
    pred_rgb = (decoded * 0.5 + 0.5).clamp(0, 1)[0]

    # Pixel-preservation compositing: keep original where mask is 0, use prediction where mask is 1
    mask_3ch = mask[0].expand(3, -1, -1)
    final = partial_rgb[0] * (1 - mask_3ch) + pred_rgb * mask_3ch
    return final.cpu().numpy().transpose(1, 2, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to ControlNet checkpoint (e.g. .../best/controlnet)")
    parser.add_argument("--partial", type=Path, required=True,
                        help="Partial texture PNG (RGBA, transparent where missing)")
    parser.add_argument("--mask", type=Path, required=True,
                        help="Inpainting mask PNG (white = needs inpainting)")
    parser.add_argument("--position", type=Path, required=True,
                        help="UV position map PNG (RGB)")
    parser.add_argument("--normal", type=Path, required=True,
                        help="UV normal map PNG (RGB)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output complete texture PNG")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--keep_alpha", action="store_true",
                        help="Preserve alpha channel from partial texture")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    # Load models
    from diffusers import (
        AutoencoderKL, ControlNetModel, DDIMScheduler, UNet2DConditionModel,
    )
    from transformers import CLIPTextModel, CLIPTokenizer

    weight_dtype = torch.float16 if device.type == "cuda" else torch.float32
    logger.info("Loading SD 1.5 Inpainting (frozen)...")
    tokenizer = CLIPTokenizer.from_pretrained(SD_MODEL, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(SD_MODEL, subfolder="text_encoder", torch_dtype=weight_dtype).to(device)
    vae = AutoencoderKL.from_pretrained(SD_MODEL, subfolder="vae", torch_dtype=weight_dtype).to(device)
    unet = UNet2DConditionModel.from_pretrained(SD_MODEL, subfolder="unet", torch_dtype=weight_dtype).to(device)
    scheduler = DDIMScheduler.from_pretrained(SD_MODEL, subfolder="scheduler")

    logger.info("Loading ControlNet from %s...", args.checkpoint)
    controlnet = ControlNetModel.from_pretrained(str(args.checkpoint)).to(device)
    controlnet.eval()

    text_encoder.eval()
    vae.eval()
    unet.eval()

    # Load inputs
    logger.info("Loading inputs...")
    partial_rgba = Image.open(args.partial).convert("RGBA")
    if partial_rgba.size != (512, 512):
        partial_rgba = partial_rgba.resize((512, 512), Image.LANCZOS)
    partial_alpha = np.array(partial_rgba)[:, :, 3]
    partial_rgb = load_image(args.partial, "RGB", 512)
    mask = load_image(args.mask, "L", 512)
    position = load_image(args.position, "RGB", 512)
    normal = load_image(args.normal, "RGB", 512)

    # Combine position + normal into 6-channel control
    control = torch.cat([position, normal], dim=0)

    logger.info("Running inpainting (%d steps)...", args.steps)
    result = inpaint(
        controlnet, unet, vae, text_encoder, tokenizer, scheduler,
        partial_rgb, mask, control, device, num_inference_steps=args.steps,
    )

    # Save (1024 res for use with mesh)
    result_uint8 = (result * 255).clip(0, 255).astype(np.uint8)
    if args.keep_alpha:
        # Use original alpha — preserves UV coverage info
        rgba = np.zeros((512, 512, 4), dtype=np.uint8)
        rgba[:, :, :3] = result_uint8
        rgba[:, :, 3] = partial_alpha
        out_img = Image.fromarray(rgba, "RGBA")
    else:
        out_img = Image.fromarray(result_uint8, "RGB")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Save at 1024 (upscale) for matching the original texture resolution
    out_img_1024 = out_img.resize((1024, 1024), Image.LANCZOS)
    out_img_1024.save(args.output)
    logger.info("Saved %s", args.output)


if __name__ == "__main__":
    main()
