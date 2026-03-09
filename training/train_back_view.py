"""Training script for back view generation model (Model 6).

Trains a U-Net to generate back view RGBA from concatenated front + 3/4 view
inputs.  Loss: L1 reconstruction + perceptual (VGG) + palette consistency.

Usage::

    python -m training.train_back_view --config training/configs/back_view.yaml

Pure Python + PyTorch (no Blender dependency).
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from training.data.back_view_dataset import BackViewDataset, BackViewDatasetConfig
from training.models.back_view_model import BackViewModel
from training.utils.checkpoint import EarlyStopping, save_checkpoint

logger = logging.getLogger(__name__)

HISTOGRAM_BINS: int = 8


# ---------------------------------------------------------------------------
# Perceptual loss (VGG feature matching)
# ---------------------------------------------------------------------------


class PerceptualLoss(nn.Module):
    """VGG16 feature-matching loss for perceptual quality."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        from torchvision.models import vgg16

        vgg = vgg16(weights="IMAGENET1K_V1").features[:16].to(device)
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize RGB channels (drop alpha)."""
        rgb = x[:, :3]
        return (rgb - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_f = self.vgg(self._normalize(pred))
        target_f = self.vgg(self._normalize(target))
        return nn.functional.l1_loss(pred_f, target_f)


# ---------------------------------------------------------------------------
# Palette consistency loss
# ---------------------------------------------------------------------------


def _color_histogram(rgb: torch.Tensor, alpha: torch.Tensor, bins: int) -> torch.Tensor:
    """Compute per-channel color histogram for visible pixels.

    Args:
        rgb: ``[B, 3, H, W]`` float in [0, 1].
        alpha: ``[B, 1, H, W]`` float in [0, 1].
        bins: Number of bins per channel.

    Returns:
        ``[B, 3*bins]`` normalized histogram.
    """
    b = rgb.shape[0]
    mask = (alpha > 0.5).squeeze(1)  # [B, H, W]
    histograms = []

    for i in range(b):
        vis = mask[i]  # [H, W]
        sample_hists = []
        for c in range(3):
            channel = rgb[i, c]  # [H, W]
            vis_pixels = channel[vis]  # [N]
            if vis_pixels.numel() == 0:
                sample_hists.append(torch.zeros(bins, device=rgb.device))
                continue
            # Bin edges: [0, 1/bins, 2/bins, ..., 1]
            binned = (vis_pixels * bins).clamp(0, bins - 1).long()
            hist = torch.zeros(bins, device=rgb.device)
            hist.scatter_add_(0, binned, torch.ones_like(vis_pixels))
            hist = hist / hist.sum().clamp(min=1.0)
            sample_hists.append(hist)
        histograms.append(torch.cat(sample_hists))

    return torch.stack(histograms)  # [B, 3*bins]


def palette_consistency_loss(
    pred_back: torch.Tensor,
    three_quarter: torch.Tensor,
    bins: int = HISTOGRAM_BINS,
) -> torch.Tensor:
    """Palette consistency between predicted back view and 3/4 view's back-facing surface.

    Samples colors from the rightmost 20% of the 3/4 view (back-facing region)
    and compares histogram with predicted back view.

    Args:
        pred_back: ``[B, 4, H, W]`` predicted RGBA.
        three_quarter: ``[B, 4, H, W]`` input 3/4 view RGBA.
        bins: Histogram bins per channel.

    Returns:
        Scalar L1 loss between histograms.
    """
    w = three_quarter.shape[3]
    rightmost_col = int(w * 0.8)

    # 3/4 view rightmost 20%
    tq_crop = three_quarter[:, :, :, rightmost_col:]
    tq_rgb = tq_crop[:, :3]
    tq_alpha = tq_crop[:, 3:4]
    tq_hist = _color_histogram(tq_rgb, tq_alpha, bins)

    # Predicted back view (full)
    pred_rgb = pred_back[:, :3]
    pred_alpha = pred_back[:, 3:4]
    pred_hist = _color_histogram(pred_rgb, pred_alpha, bins)

    return nn.functional.l1_loss(pred_hist, tq_hist)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def _adjust_lr(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    base_lr: float,
) -> float:
    """Linear warmup + cosine annealing schedule."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: BackViewModel,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    l1_weight: float,
    perceptual_loss: PerceptualLoss | None,
    perceptual_weight: float,
    palette_weight: float,
) -> dict[str, float]:
    """Train for one epoch. Returns average losses."""
    model.train()
    total_l1 = 0.0
    total_perc = 0.0
    total_palette = 0.0
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        image = batch["image"].to(device)  # [B, 8, H, W]
        target = batch["target"].to(device)  # [B, 4, H, W]

        outputs = model(image)
        pred = outputs["output"]

        # Alpha-weighted L1: only penalize non-transparent pixels
        alpha = target[:, 3:4]  # [B, 1, H, W]
        l1 = (torch.abs(pred - target) * alpha).sum() / alpha.sum().clamp(min=1.0)
        loss = l1_weight * l1

        # Perceptual loss
        perc_val = 0.0
        if perceptual_loss is not None and perceptual_weight > 0:
            perc = perceptual_loss(pred, target)
            loss = loss + perceptual_weight * perc
            perc_val = perc.item()

        # Palette consistency loss
        palette_val = 0.0
        if palette_weight > 0:
            three_quarter = image[:, 4:8]  # second 4 channels = 3/4 view
            pal = palette_consistency_loss(pred, three_quarter)
            loss = loss + palette_weight * pal
            palette_val = pal.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_l1 += l1.item()
        total_perc += perc_val
        total_palette += palette_val
        total_loss += loss.item()
        n_batches += 1

    d = max(n_batches, 1)
    return {
        "train/loss": total_loss / d,
        "train/l1": total_l1 / d,
        "train/perceptual": total_perc / d,
        "train/palette": total_palette / d,
    }


@torch.no_grad()
def validate(
    model: BackViewModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    l1_weight: float,
    perceptual_loss: PerceptualLoss | None,
    perceptual_weight: float,
    palette_weight: float,
) -> dict[str, float]:
    """Validate. Returns average losses."""
    model.eval()
    total_l1 = 0.0
    total_perc = 0.0
    total_palette = 0.0
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        image = batch["image"].to(device)
        target = batch["target"].to(device)

        outputs = model(image)
        pred = outputs["output"]

        alpha = target[:, 3:4]
        l1 = (torch.abs(pred - target) * alpha).sum() / alpha.sum().clamp(min=1.0)
        loss = l1_weight * l1

        perc_val = 0.0
        if perceptual_loss is not None and perceptual_weight > 0:
            perc = perceptual_loss(pred, target)
            loss = loss + perceptual_weight * perc
            perc_val = perc.item()

        palette_val = 0.0
        if palette_weight > 0:
            three_quarter = image[:, 4:8]
            pal = palette_consistency_loss(pred, three_quarter)
            loss = loss + palette_weight * pal
            palette_val = pal.item()

        total_l1 += l1.item()
        total_perc += perc_val
        total_palette += palette_val
        total_loss += loss.item()
        n_batches += 1

    d = max(n_batches, 1)
    return {
        "val/loss": total_loss / d,
        "val/l1": total_l1 / d,
        "val/perceptual": total_perc / d,
        "val/palette": total_palette / d,
    }


# ---------------------------------------------------------------------------
# TensorBoard visual logging
# ---------------------------------------------------------------------------


def _log_visual_comparison(
    writer: object,
    model: BackViewModel,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    max_images: int = 4,
) -> None:
    """Log front | 3/4 | predicted | GT comparison to TensorBoard."""
    model.eval()
    batch = next(iter(val_loader))
    image = batch["image"][:max_images].to(device)
    target = batch["target"][:max_images]

    with torch.no_grad():
        pred = model(image)["output"].cpu()

    front = image[:, :4].cpu()  # [N, 4, H, W]
    tq = image[:, 4:8].cpu()  # [N, 4, H, W]

    # Build comparison grid: front | 3/4 | predicted | GT (use RGB only)
    grid_rows = []
    for i in range(pred.shape[0]):
        row = torch.cat([front[i, :3], tq[i, :3], pred[i, :3], target[i, :3]], dim=2)
        grid_rows.append(row)

    grid = torch.cat(grid_rows, dim=1)  # [3, N*H, 4*W]
    writer.add_image("val/comparison", grid.clamp(0, 1), epoch)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train back view generation model (Model 6)")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    loss_cfg = cfg.get("loss", {})
    ckpt_cfg = cfg.get("checkpointing", {})
    model_cfg = cfg.get("model", {})
    aug_cfg = cfg.get("augmentation", {})

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info("Using device: %s", device)

    # --- Dataset ---
    dataset_dirs = [Path(p) for p in data_cfg.get("dataset_dirs", [])]
    split_ratios = data_cfg.get("split_ratios", {"train": 0.8, "val": 0.1, "test": 0.1})

    train_ds_config = BackViewDatasetConfig(
        dataset_dirs=dataset_dirs,
        resolution=data_cfg.get("resolution", 512),
        split="train",
        split_seed=data_cfg.get("split_seed", 42),
        split_ratios=(split_ratios["train"], split_ratios["val"], split_ratios["test"]),
        horizontal_flip=aug_cfg.get("horizontal_flip", True),
        color_jitter=aug_cfg.get("color_jitter", {}),
    )
    val_ds_config = BackViewDatasetConfig(
        dataset_dirs=dataset_dirs,
        resolution=data_cfg.get("resolution", 512),
        split="val",
        split_seed=data_cfg.get("split_seed", 42),
        split_ratios=(split_ratios["train"], split_ratios["val"], split_ratios["test"]),
        horizontal_flip=False,
        color_jitter={},
    )

    train_ds = BackViewDataset(train_ds_config)
    val_ds = BackViewDataset(val_ds_config)

    logger.info("Train: %d pairs, Val: %d pairs", len(train_ds), len(val_ds))

    batch_size = train_cfg.get("batch_size", 4)
    num_workers = train_cfg.get("num_workers", 4)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # --- Model ---
    model = BackViewModel(
        in_channels=model_cfg.get("in_channels", 8),
        out_channels=model_cfg.get("out_channels", 4),
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s", f"{param_count:,}")

    # --- Optimizer ---
    base_lr = train_cfg.get("learning_rate", 2e-4)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=base_lr,
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )

    # --- Loss ---
    l1_weight = loss_cfg.get("l1_weight", 1.0)
    perceptual_weight = loss_cfg.get("perceptual_weight", 0.1)
    palette_weight = loss_cfg.get("palette_weight", 0.05)

    perceptual_loss: PerceptualLoss | None = None
    if perceptual_weight > 0:
        try:
            perceptual_loss = PerceptualLoss(device)
            logger.info("Perceptual loss enabled (weight=%.3f)", perceptual_weight)
        except Exception as e:
            logger.warning("Could not load VGG for perceptual loss: %s", e)
            perceptual_loss = None

    # --- Checkpointing ---
    save_dir = Path(ckpt_cfg.get("save_dir", "./checkpoints/back_view"))
    save_dir.mkdir(parents=True, exist_ok=True)

    es_metric = ckpt_cfg.get("early_stopping_metric", "val/l1")
    es_patience = ckpt_cfg.get("early_stopping_patience", 20)
    early_stopping = EarlyStopping(patience=es_patience, metric_name=es_metric, mode="min")

    # --- TensorBoard ---
    try:
        from torch.utils.tensorboard import SummaryWriter

        tb_writer = SummaryWriter(log_dir=str(save_dir / "tb_logs"))
        logger.info("TensorBoard logging to %s", save_dir / "tb_logs")
    except ImportError:
        tb_writer = None
        logger.warning("TensorBoard not available — skipping visual logging")

    # --- Resume from checkpoint ---
    start_epoch = 0
    resume_path = save_dir / "latest.pt"
    if resume_path.exists():
        from training.utils.checkpoint import load_checkpoint

        info = load_checkpoint(resume_path, model, optimizer)
        start_epoch = info["epoch"] + 1
        logger.info("Resumed from epoch %d", start_epoch)

    # --- Training loop ---
    epochs = train_cfg.get("epochs", 150)
    warmup = train_cfg.get("warmup_epochs", 5)
    best_val_l1 = float("inf")
    log_images_every = 5  # Log visual comparison every N epochs

    for epoch in range(start_epoch, epochs):
        lr = _adjust_lr(optimizer, epoch, epochs, warmup, base_lr)

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            l1_weight,
            perceptual_loss,
            perceptual_weight,
            palette_weight,
        )
        val_metrics = validate(
            model,
            val_loader,
            device,
            l1_weight,
            perceptual_loss,
            perceptual_weight,
            palette_weight,
        )

        all_metrics = {**train_metrics, **val_metrics, "lr": lr}

        logger.info(
            "Epoch %d/%d — lr=%.2e | train/l1=%.4f | val/l1=%.4f | val/loss=%.4f",
            epoch + 1,
            epochs,
            lr,
            train_metrics["train/l1"],
            val_metrics["val/l1"],
            val_metrics["val/loss"],
        )

        # TensorBoard logging
        if tb_writer is not None:
            for key, value in all_metrics.items():
                tb_writer.add_scalar(key, value, epoch)
            if epoch % log_images_every == 0 and len(val_ds) > 0:
                _log_visual_comparison(tb_writer, model, val_loader, device, epoch)

        # Save latest
        save_checkpoint(model, optimizer, None, epoch, all_metrics, save_dir / "latest.pt")

        # Save best
        if val_metrics["val/l1"] < best_val_l1:
            best_val_l1 = val_metrics["val/l1"]
            save_checkpoint(model, optimizer, None, epoch, all_metrics, save_dir / "best.pt")
            logger.info("  New best val/l1: %.4f", best_val_l1)

        # Early stopping
        if early_stopping.step(all_metrics):
            break

    if tb_writer is not None:
        tb_writer.close()

    logger.info("Training complete. Best val/l1: %.4f", best_val_l1)


if __name__ == "__main__":
    main()
