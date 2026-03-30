"""Training script for unified view synthesis model (Model 6 replacement).

Trains a U-Net to generate any target view RGBA from two source views + target angle.
Loss: L1 reconstruction + perceptual (VGG) + palette consistency.

Usage::

    python -m training.train_view_synthesis --config training/configs/view_synthesis.yaml

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

from training.data.view_synthesis_dataset import ViewSynthesisConfig, ViewSynthesisDataset
from training.models.view_synthesis_model import ViewSynthesisModel
from training.utils.checkpoint import EarlyStopping, save_checkpoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Perceptual loss (VGG feature matching) — same as back view
# ---------------------------------------------------------------------------


class PerceptualLoss(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        from torchvision.models import vgg16

        vgg = vgg16(weights="IMAGENET1K_V1").features[:16].to(device)
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x[:, :3] - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.l1_loss(self.vgg(self._normalize(pred)), self.vgg(self._normalize(target)))


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def _adjust_lr(optimizer, epoch, total_epochs, warmup_epochs, base_lr) -> float:
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
    model: ViewSynthesisModel,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    l1_weight: float,
    perceptual_loss: PerceptualLoss | None,
    perceptual_weight: float,
) -> dict[str, float]:
    model.train()
    total_l1 = 0.0
    total_perc = 0.0
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        image = batch["image"].to(device)   # [B, 9, H, W]
        target = batch["target"].to(device)  # [B, 4, H, W]

        outputs = model(image)
        pred = outputs["output"]

        # Alpha-weighted L1
        alpha = target[:, 3:4]
        l1 = (torch.abs(pred - target) * alpha).sum() / alpha.sum().clamp(min=1.0)
        loss = l1_weight * l1

        # Perceptual loss
        perc_val = 0.0
        if perceptual_loss is not None and perceptual_weight > 0:
            perc = perceptual_loss(pred, target)
            loss = loss + perceptual_weight * perc
            perc_val = perc.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_l1 += l1.item()
        total_perc += perc_val
        total_loss += loss.item()
        n_batches += 1

    d = max(n_batches, 1)
    return {
        "train/loss": total_loss / d,
        "train/l1": total_l1 / d,
        "train/perceptual": total_perc / d,
    }


@torch.no_grad()
def validate(
    model: ViewSynthesisModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    l1_weight: float,
    perceptual_loss: PerceptualLoss | None,
    perceptual_weight: float,
) -> dict[str, float]:
    model.eval()
    total_l1 = 0.0
    total_perc = 0.0
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

        total_l1 += l1.item()
        total_perc += perc_val
        total_loss += loss.item()
        n_batches += 1

    d = max(n_batches, 1)
    return {
        "val/loss": total_loss / d,
        "val/l1": total_l1 / d,
        "val/perceptual": total_perc / d,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train view synthesis model")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    loss_cfg = cfg.get("loss", {})
    ckpt_cfg = cfg.get("checkpointing", {})
    model_cfg = cfg.get("model", {})

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info("Using device: %s", device)

    # --- Dataset ---
    ds_config = ViewSynthesisConfig.from_dict(cfg)

    train_ds = ViewSynthesisDataset(ds_config, split="train")
    val_ds = ViewSynthesisDataset(ds_config, split="val")

    logger.info("Train: %d triplets, Val: %d triplets", len(train_ds), len(val_ds))

    batch_size = train_cfg.get("batch_size", 16)
    num_workers = train_cfg.get("num_workers", 4)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # --- Model ---
    model = ViewSynthesisModel(
        in_channels=model_cfg.get("in_channels", 9),
        out_channels=model_cfg.get("out_channels", 4),
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s", f"{param_count:,}")

    # --- Optimizer ---
    base_lr = train_cfg.get("learning_rate", 2e-4)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=base_lr,
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )

    # --- Loss ---
    l1_weight = loss_cfg.get("l1_weight", 1.0)
    perceptual_weight = loss_cfg.get("perceptual_weight", 0.1)

    perceptual_loss: PerceptualLoss | None = None
    if perceptual_weight > 0:
        try:
            perceptual_loss = PerceptualLoss(device)
            logger.info("Perceptual loss enabled (weight=%.3f)", perceptual_weight)
        except Exception as e:
            logger.warning("Could not load VGG: %s", e)

    # --- Checkpointing ---
    save_dir = Path(ckpt_cfg.get("save_dir", "./checkpoints/view_synthesis"))
    save_dir.mkdir(parents=True, exist_ok=True)

    es_metric = ckpt_cfg.get("early_stopping_metric", "val/l1")
    es_patience = ckpt_cfg.get("early_stopping_patience", 30)
    early_stopping = EarlyStopping(patience=es_patience, metric_name=es_metric, mode="min")

    # --- TensorBoard ---
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=str(save_dir / "tb_logs"))
        logger.info("TensorBoard logging to %s", save_dir / "tb_logs")
    except ImportError:
        tb_writer = None

    # --- Resume ---
    start_epoch = 0
    resume_path = save_dir / "latest.pt"
    if resume_path.exists():
        from training.utils.checkpoint import load_checkpoint
        info = load_checkpoint(resume_path, model, optimizer)
        start_epoch = info["epoch"] + 1
        logger.info("Resumed from epoch %d", start_epoch)

    # --- Training loop ---
    epochs = train_cfg.get("epochs", 200)
    warmup = train_cfg.get("warmup_epochs", 5)
    best_val_l1 = float("inf")

    for epoch in range(start_epoch, epochs):
        lr = _adjust_lr(optimizer, epoch, epochs, warmup, base_lr)

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            l1_weight, perceptual_loss, perceptual_weight,
        )
        val_metrics = validate(
            model, val_loader, device,
            l1_weight, perceptual_loss, perceptual_weight,
        )

        all_metrics = {**train_metrics, **val_metrics, "lr": lr}

        logger.info(
            "Epoch %d/%d — lr=%.2e | train/l1=%.4f | val/l1=%.4f | val/loss=%.4f",
            epoch + 1, epochs, lr,
            train_metrics["train/l1"], val_metrics["val/l1"], val_metrics["val/loss"],
        )

        if tb_writer is not None:
            for key, value in all_metrics.items():
                tb_writer.add_scalar(key, value, epoch)

        save_checkpoint(model, optimizer, None, epoch, all_metrics, save_dir / "latest.pt")

        if val_metrics["val/l1"] < best_val_l1:
            best_val_l1 = val_metrics["val/l1"]
            save_checkpoint(model, optimizer, None, epoch, all_metrics, save_dir / "best.pt")
            logger.info("  New best val/l1: %.4f", best_val_l1)

        if early_stopping.step(all_metrics):
            break

    if tb_writer is not None:
        tb_writer.close()

    logger.info("Training complete. Best val/l1: %.4f", best_val_l1)


if __name__ == "__main__":
    main()
