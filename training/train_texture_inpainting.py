"""Training script for ControlNet-based UV texture inpainting (Model 5).

Fine-tunes a ControlNet conditioned on UV position + normal maps, using
Stable Diffusion 1.5 Inpainting as the frozen base model.  The ControlNet
learns to generate plausible texture for unobserved UV regions while
respecting 3D surface geometry.

Usage::

    # Single GPU
    python -m training.train_texture_inpainting \
        --config training/configs/texture_inpainting_a100.yaml

    # Multi-GPU with accelerate
    accelerate launch -m training.train_texture_inpainting \
        --config training/configs/texture_inpainting_a100.yaml

Pure Python + PyTorch + diffusers (no Blender dependency).
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from training.data.texture_inpainting_dataset import (
    TextureInpaintingDataset,
    TextureInpaintingDatasetConfig,
)
from training.utils.checkpoint import EarlyStopping, save_checkpoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Perceptual loss (VGG feature matching)
# ---------------------------------------------------------------------------


class PerceptualLoss(torch.nn.Module):
    """VGG16 feature-matching loss for perceptual quality."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        from torchvision.models import vgg16

        vgg = vgg16(weights="IMAGENET1K_V1").features[:16].to(device)
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x[:, :3] - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self.vgg(self._normalize(pred)), self.vgg(self._normalize(target)))


# ---------------------------------------------------------------------------
# SSIM metric
# ---------------------------------------------------------------------------


def _ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    """Compute mean SSIM between pred and target (RGB)."""
    x = pred[:, :3]
    y = target[:, :3]
    channels = 3
    c1, c2 = 0.01**2, 0.03**2

    kernel_1d = torch.arange(window_size, dtype=torch.float32, device=x.device)
    kernel_1d = kernel_1d - window_size // 2
    kernel_1d = torch.exp(-0.5 * (kernel_1d / 1.5) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    window = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()

    pad = window_size // 2
    mu_x = F.conv2d(x, window, padding=pad, groups=channels)
    mu_y = F.conv2d(y, window, padding=pad, groups=channels)

    sigma_x_sq = F.conv2d(x * x, window, padding=pad, groups=channels) - mu_x * mu_x
    sigma_y_sq = F.conv2d(y * y, window, padding=pad, groups=channels) - mu_y * mu_y
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=channels) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x_sq + sigma_y_sq + c2)
    )
    return ssim_map.mean()


# ---------------------------------------------------------------------------
# Training step (ControlNet fine-tuning with diffusion loss)
# ---------------------------------------------------------------------------


def train_one_epoch(
    unet: torch.nn.Module,
    controlnet: torch.nn.Module,
    vae: torch.nn.Module,
    text_encoder: torch.nn.Module,
    noise_scheduler: object,
    tokenizer: object,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    weight_dtype: torch.dtype,
    scaler: torch.amp.GradScaler | None,
) -> dict[str, float]:
    """Train ControlNet for one epoch using diffusion denoising loss."""
    controlnet.train()
    unet.eval()

    total_loss = 0.0
    n_batches = 0
    use_amp = scaler is not None

    # Empty text embedding (we don't use text prompts — geometry conditions everything)
    empty_tokens = tokenizer(
        "", padding="max_length", max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(device)
    with torch.no_grad():
        empty_embeds = text_encoder(empty_tokens)[0]

    for batch in loader:
        # Move to device
        target_rgb = batch["target"].to(device, dtype=weight_dtype)   # [B, 3, H, W]
        partial_rgb = batch["image"].to(device, dtype=weight_dtype)   # [B, 3, H, W]
        mask = batch["mask"].to(device, dtype=weight_dtype)           # [B, 1, H, W]
        control = batch["control"].to(device, dtype=weight_dtype)     # [B, 6, H, W]

        with torch.no_grad():
            # Encode target to latent space
            # SD expects images in [-1, 1]
            target_latents = vae.encode(target_rgb * 2 - 1).latent_dist.sample()
            target_latents = target_latents * vae.config.scaling_factor

            # Encode partial texture
            partial_latents = vae.encode(partial_rgb * 2 - 1).latent_dist.sample()
            partial_latents = partial_latents * vae.config.scaling_factor

            # Downsample mask to latent resolution
            mask_latent = F.interpolate(mask, size=target_latents.shape[-2:], mode="nearest")

        # Sample noise and timestep
        noise = torch.randn_like(target_latents)
        bsz = target_latents.shape[0]
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (bsz,), device=device, dtype=torch.long,
        )

        # Add noise to target latents
        noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)

        # SD inpainting UNet expects 9-channel input: noisy_latents (4) + mask (1) + partial_latents (4)
        unet_input = torch.cat([noisy_latents, mask_latent, partial_latents], dim=1)

        # Expand text embeddings to batch size
        encoder_hidden_states = empty_embeds.expand(bsz, -1, -1)

        # Resize control to match expected ControlNet input size
        control_resized = F.interpolate(
            control, size=(target_rgb.shape[2], target_rgb.shape[3]), mode="bilinear",
            align_corners=False,
        )

        with torch.amp.autocast("cuda", enabled=use_amp):
            # ControlNet forward (same 9-channel input as UNet since it was
            # initialized from the inpainting UNet)
            down_block_res_samples, mid_block_res_sample = controlnet(
                unet_input,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=control_resized,
                return_dict=False,
            )

            # UNet forward with ControlNet conditioning
            noise_pred = unet(
                unet_input,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

            # Diffusion denoising loss (predict noise)
            loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {"train/loss": total_loss / max(n_batches, 1)}


# ---------------------------------------------------------------------------
# Validation (generate actual inpainted textures and measure quality)
# ---------------------------------------------------------------------------


@torch.no_grad()
def validate(
    controlnet,
    unet,
    vae,
    text_encoder,
    tokenizer,
    noise_scheduler,
    loader: DataLoader,
    device: torch.device,
    num_inference_steps: int = 20,
) -> dict[str, float]:
    """Validate by running manual denoising loop and measuring reconstruction.

    Uses a manual loop instead of the diffusers pipeline because the SD
    Inpainting ControlNet expects 9-channel input (same as the UNet).
    """
    controlnet.eval()

    # Empty text embedding
    tok = tokenizer("", return_tensors="pt", padding="max_length",
                    max_length=tokenizer.model_max_length, truncation=True)
    empty_embeds = text_encoder(tok.input_ids.to(device))[0]

    # Set up scheduler for inference
    from diffusers import DDIMScheduler
    val_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
    val_scheduler.set_timesteps(num_inference_steps, device=device)

    total_l1 = 0.0
    total_ssim = 0.0
    n_batches = 0

    weight_dtype = next(vae.parameters()).dtype

    for batch in loader:
        target_rgb = batch["target"].to(device)
        partial_rgb = batch["image"].to(device)
        mask = batch["mask"].to(device)
        control = batch["control"].to(device)

        bsz = target_rgb.shape[0]
        encoder_hidden_states = empty_embeds.expand(bsz, -1, -1)

        # Encode partial image (cast to model dtype for fp16 VAE)
        partial_norm = (partial_rgb * 2.0 - 1.0).to(dtype=weight_dtype)
        partial_latents = vae.encode(partial_norm).latent_dist.mean * vae.config.scaling_factor
        mask_latent = F.interpolate(mask, size=partial_latents.shape[2:], mode="nearest")

        # Resize control and cast to model dtype
        control_resized = F.interpolate(
            control, size=(target_rgb.shape[2], target_rgb.shape[3]),
            mode="bilinear", align_corners=False,
        ).to(dtype=weight_dtype)
        mask_latent = mask_latent.to(dtype=weight_dtype)

        # Start from random noise
        latents = torch.randn_like(partial_latents)

        for t in val_scheduler.timesteps:
            t_batch = t.expand(bsz)
            unet_input = torch.cat([latents, mask_latent, partial_latents], dim=1)

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

            latents = val_scheduler.step(noise_pred, t, latents).prev_sample

        # Decode
        decoded = vae.decode(latents / vae.config.scaling_factor).sample
        pred_rgb = (decoded * 0.5 + 0.5).clamp(0, 1)

        # Pixel preservation compositing
        mask_3ch = mask.expand(-1, 3, -1, -1)
        final = partial_rgb * (1 - mask_3ch) + pred_rgb * mask_3ch

        # Metrics on inpainted regions only
        masked_pred = final * mask_3ch
        masked_target = target_rgb * mask_3ch
        mask_sum = mask_3ch.sum().clamp(min=1.0)

        l1 = (masked_pred - masked_target).abs().sum() / mask_sum
        ssim_val = _ssim(final, target_rgb)

        total_l1 += l1.item()
        total_ssim += ssim_val.item()
        n_batches += 1

    controlnet.train()
    d = max(n_batches, 1)
    return {
        "val/l1": total_l1 / d,
        "val/ssim": total_ssim / d,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ControlNet for UV texture inpainting (Model 5)"
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from ControlNet checkpoint dir")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    ckpt_cfg = cfg.get("checkpointing", {})

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info("Using device: %s", device)

    weight_dtype = torch.float16 if train_cfg.get("mixed_precision") == "fp16" else torch.float32
    use_amp = weight_dtype == torch.float16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # --- Load SD Inpainting components (frozen) ---
    from diffusers import (
        AutoencoderKL,
        ControlNetModel,
        DDPMScheduler,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextModel, CLIPTokenizer

    sd_model_id = model_cfg.get("base_model", "runwayml/stable-diffusion-inpainting")
    logger.info("Loading SD Inpainting from %s", sd_model_id)

    tokenizer = CLIPTokenizer.from_pretrained(sd_model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_model_id, subfolder="text_encoder").to(device, dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(sd_model_id, subfolder="vae").to(device, dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(sd_model_id, subfolder="unet").to(device, dtype=weight_dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(sd_model_id, subfolder="scheduler")

    # Freeze SD components
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # --- ControlNet (trainable) ---
    controlnet_channels = model_cfg.get("controlnet_channels", 6)

    if args.resume:
        logger.info("Resuming ControlNet from %s", args.resume)
        controlnet = ControlNetModel.from_pretrained(str(args.resume)).to(device)
    else:
        logger.info("Initializing ControlNet from SD UNet with %d condition channels", controlnet_channels)
        controlnet = ControlNetModel.from_unet(
            unet,
            conditioning_channels=controlnet_channels,
        ).to(device)

    # ControlNet trains in fp32 for stability, even when SD runs in fp16
    controlnet.train()
    trainable_params = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
    logger.info("ControlNet trainable parameters: %s", f"{trainable_params:,}")

    # --- Dataset ---
    ds_config = TextureInpaintingDatasetConfig.from_dict(cfg)

    train_ds = TextureInpaintingDataset(
        TextureInpaintingDatasetConfig(**{**ds_config.__dict__, "split": "train"})
    )
    val_ds = TextureInpaintingDataset(
        TextureInpaintingDatasetConfig(**{**ds_config.__dict__, "split": "val"})
    )

    batch_size = train_cfg.get("batch_size", 1)
    num_workers = train_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    logger.info("Train: %d examples, Val: %d examples", len(train_ds), len(val_ds))

    # --- Optimizer ---
    base_lr = train_cfg.get("learning_rate", 1e-5)
    grad_accum = train_cfg.get("gradient_accumulation", 1)
    optimizer = torch.optim.AdamW(
        controlnet.parameters(),
        lr=base_lr,
        weight_decay=train_cfg.get("weight_decay", 1e-2),
    )

    # --- Checkpointing ---
    save_dir = Path(ckpt_cfg.get("save_dir", "./checkpoints/texture_inpainting_controlnet"))
    save_dir.mkdir(parents=True, exist_ok=True)

    es_patience = ckpt_cfg.get("early_stopping_patience", 20)
    early_stopping = EarlyStopping(patience=es_patience, metric_name="val/l1", mode="min")

    # --- TensorBoard ---
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=str(save_dir / "tb_logs"))
    except ImportError:
        tb_writer = None

    # --- Training loop ---
    epochs = train_cfg.get("epochs", 100)
    validate_every = train_cfg.get("validate_every", 5)
    val_steps = train_cfg.get("val_inference_steps", 20)
    best_val_l1 = float("inf")

    for epoch in range(epochs):
        # Cosine LR schedule
        progress = epoch / max(epochs - 1, 1)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        train_metrics = train_one_epoch(
            unet=unet,
            controlnet=controlnet,
            vae=vae,
            text_encoder=text_encoder,
            noise_scheduler=noise_scheduler,
            tokenizer=tokenizer,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            weight_dtype=weight_dtype,
            scaler=scaler,
        )

        logger.info(
            "Epoch %d/%d — lr=%.2e | train/loss=%.6f",
            epoch + 1, epochs, lr, train_metrics["train/loss"],
        )

        # Validate periodically (full inference is slow)
        val_metrics = {}
        if (epoch + 1) % validate_every == 0 and len(val_ds) > 0:
            val_metrics = validate(
                controlnet, unet, vae, text_encoder, tokenizer,
                noise_scheduler, val_loader, device,
                num_inference_steps=val_steps,
            )

            logger.info(
                "  val/l1=%.4f | val/ssim=%.4f",
                val_metrics["val/l1"],
                val_metrics["val/ssim"],
            )

            # Save best
            if val_metrics["val/l1"] < best_val_l1:
                best_val_l1 = val_metrics["val/l1"]
                controlnet.save_pretrained(str(save_dir / "best"))
                logger.info("  New best val/l1: %.4f — saved to %s", best_val_l1, save_dir / "best")

            # Clean up validation pipeline
            del val_pipe, val_controlnet
            torch.cuda.empty_cache() if device.type == "cuda" else None

        # TensorBoard
        if tb_writer is not None:
            all_metrics = {**train_metrics, **val_metrics, "lr": lr}
            for key, value in all_metrics.items():
                tb_writer.add_scalar(key, value, epoch)

        # Save latest ControlNet
        controlnet.save_pretrained(str(save_dir / "latest"))

        # Early stopping (only check when we have val metrics)
        if val_metrics and early_stopping.step({**train_metrics, **val_metrics}):
            logger.info("Early stopping triggered at epoch %d", epoch + 1)
            break

    if tb_writer is not None:
        tb_writer.close()

    logger.info("Training complete. Best val/l1: %.4f", best_val_l1)
    logger.info("Best checkpoint: %s", save_dir / "best")


if __name__ == "__main__":
    main()
