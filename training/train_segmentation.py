"""Train the multi-head DeepLabV3+ segmentation model.

CLI entry point::

    python training/train_segmentation.py --config training/configs/segmentation.yaml
    python training/train_segmentation.py --config training/configs/segmentation.yaml \\
        --resume checkpoints/segmentation/best.pt

Trains on pipeline output with combined segmentation CE + depth L1 +
normals L1 + confidence BCE loss. Logs to TensorBoard and saves best/latest
checkpoints.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import sys
from pathlib import Path

# Python 3.14+ changed default to 'forkserver' which requires pickling.
# Our dataset has unpicklable module refs, so use 'fork' instead.
try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass  # already set

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from training.data.segmentation_dataset import DatasetConfig, SegmentationDataset
from training.models.segmentation_model import SegmentationModel
from training.utils.checkpoint import EarlyStopping, load_checkpoint, save_checkpoint
from training.utils.metrics import SegmentationMetrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class weight computation
# ---------------------------------------------------------------------------


def compute_class_weights(
    dataset: SegmentationDataset,
    num_classes: int = 22,
    max_samples: int = 2000,
) -> torch.Tensor:
    """Compute median frequency balancing weights from training labels.

    Samples up to ``max_samples`` masks (uniformly spaced) to estimate
    per-class pixel frequency, then returns ``median_freq / class_freq``
    for each class. Classes with zero pixels get weight 0.

    Args:
        dataset: Training dataset to scan.
        num_classes: Total number of classes.
        max_samples: Maximum examples to scan (0 = all).

    Returns:
        Tensor of shape ``[num_classes]`` with class weights.
    """
    n = len(dataset)
    if max_samples > 0 and n > max_samples:
        # Uniformly spaced indices for representative sampling
        indices = np.linspace(0, n - 1, max_samples, dtype=int)
        logger.info("Sampling %d/%d examples for class weight estimation", max_samples, n)
    else:
        indices = range(n)

    counts = np.zeros(num_classes, dtype=np.int64)
    for i in indices:
        mask = dataset[i]["segmentation"].numpy().ravel()
        valid = mask[mask >= 0]  # exclude ignore_index (-1) pixels
        if len(valid) > 0:
            counts += np.bincount(valid, minlength=num_classes)[:num_classes]

    total = counts.sum()
    if total == 0:
        logger.warning("No labeled pixels found — using uniform class weights")
        return torch.ones(num_classes, dtype=torch.float32)

    freq = counts.astype(np.float64) / total
    nonzero = freq > 0
    if not nonzero.any():
        return torch.ones(num_classes, dtype=torch.float32)

    median_freq = float(np.median(freq[nonzero]))
    weights = np.where(nonzero, median_freq / freq, 0.0)
    logger.info(
        "Class weights computed (median_freq=%.4f, %d active classes, %d samples scanned)",
        median_freq,
        int(nonzero.sum()),
        len(list(indices)),
    )
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Multi-head loss
# ---------------------------------------------------------------------------


def compute_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    class_weights: torch.Tensor,
    loss_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute combined multi-head loss.

    Args:
        outputs: Model output dict with ``segmentation``, ``depth``,
            ``normals``, ``confidence`` tensors.
        targets: Target dict with ``segmentation``, ``depth``, ``has_depth``,
            ``normals``, ``has_normals``, ``confidence_target`` tensors.
        class_weights: Per-class CE weights, shape ``[num_classes]``.
        loss_weights: Dict with weight scalars for each loss component.

    Returns:
        ``(total_loss, component_dict)`` where ``component_dict`` maps loss
        names to their scalar values (for logging).
    """
    device = outputs["segmentation"].device
    sample_weights = targets["dataset_weight"].to(device)  # [B]

    # Segmentation: CrossEntropyLoss with class weights + optional label smoothing
    # Use soft targets when available (boundary softening), else hard CE
    label_smoothing = loss_weights.get("label_smoothing", 0.0)

    if "soft_segmentation" in targets:
        # Soft cross-entropy: -sum(soft_target * log_softmax(logits))
        soft_targets = targets["soft_segmentation"].to(device)  # [B, C, H, W]
        log_probs = F.log_softmax(outputs["segmentation"], dim=1)  # [B, C, H, W]
        # Apply class weights via soft_targets
        cw = class_weights.to(device).view(1, -1, 1, 1)  # [1, C, 1, 1]
        seg_loss_per_pixel = -(soft_targets * log_probs * cw).sum(dim=1)  # [B, H, W]
        # Mask out ignore pixels (soft_targets sum to 0 there, so loss is already 0)
        per_sample_seg = seg_loss_per_pixel.mean(dim=(1, 2))  # [B]
    else:
        seg_loss_per_pixel = F.cross_entropy(
            outputs["segmentation"],
            targets["segmentation"],
            weight=class_weights.to(device),
            ignore_index=-1,
            label_smoothing=label_smoothing,
            reduction="none",
        )  # [B, H, W]
        per_sample_seg = seg_loss_per_pixel.mean(dim=(1, 2))  # [B]

    seg_loss = (per_sample_seg * sample_weights).mean()

    # Depth: L1 loss, only for examples that have Marigold depth labels
    has_depth = targets["has_depth"]  # [B] bool
    if has_depth.any():
        depth_pred = outputs["depth"][has_depth]  # [N, 1, H, W]
        depth_target = targets["depth"][has_depth]
        depth_loss = F.l1_loss(depth_pred, depth_target)
    else:
        depth_loss = torch.tensor(0.0, device=device)

    # Normals: L1 loss, only for examples that have Marigold normal labels
    has_normals = targets["has_normals"]  # [B] bool
    if has_normals.any():
        normals_pred = outputs["normals"][has_normals]  # [N, 3, H, W]
        normals_target = targets["normals"][has_normals]
        normals_loss = F.l1_loss(normals_pred, normals_target)
    else:
        normals_loss = torch.tensor(0.0, device=device)

    # Confidence: BCE loss with per-sample weighting
    conf_loss_per_pixel = F.binary_cross_entropy(
        outputs["confidence"],
        targets["confidence_target"],
        reduction="none",
    )  # [B, 1, H, W]
    per_sample_conf = conf_loss_per_pixel.mean(dim=(1, 2, 3))  # [B]
    conf_loss = (per_sample_conf * sample_weights).mean()

    # Weighted sum
    w_seg = loss_weights.get("segmentation_weight", 1.0)
    w_depth = loss_weights.get("depth_weight", 0.5)
    w_normals = loss_weights.get("normals_weight", 0.5)
    w_conf = loss_weights.get("confidence_weight", 0.1)

    total = w_seg * seg_loss + w_depth * depth_loss + w_normals * normals_loss + w_conf * conf_loss

    components = {
        "loss/segmentation": float(seg_loss),
        "loss/depth": float(depth_loss),
        "loss/normals": float(normals_loss),
        "loss/confidence": float(conf_loss),
        "loss/total": float(total),
    }
    return total, components


# ---------------------------------------------------------------------------
# Learning rate with warmup
# ---------------------------------------------------------------------------


def adjust_lr(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    base_lr: float,
    warmup_epochs: int,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> float:
    """Set optimizer LR with linear warmup, then delegate to scheduler.

    Returns the current learning rate.
    """
    if epoch < warmup_epochs and warmup_epochs > 0:
        lr = base_lr * (epoch + 1) / warmup_epochs
        for pg in optimizer.param_groups:
            pg["lr"] = lr
    else:
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
    return lr


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Custom collate that stacks tensors and converts booleans to bool tensors."""
    result = {
        "image": torch.stack([b["image"] for b in batch]),
        "segmentation": torch.stack([b["segmentation"] for b in batch]),
        "depth": torch.stack([b["depth"] for b in batch]),
        "has_depth": torch.tensor([b["has_depth"] for b in batch], dtype=torch.bool),
        "normals": torch.stack([b["normals"] for b in batch]),
        "has_normals": torch.tensor([b["has_normals"] for b in batch], dtype=torch.bool),
        "confidence_target": torch.stack([b["confidence_target"] for b in batch]),
        "dataset_weight": torch.tensor([b["dataset_weight"] for b in batch], dtype=torch.float32),
    }
    # Soft segmentation targets (boundary softening, train only)
    if "soft_segmentation" in batch[0]:
        result["soft_segmentation"] = torch.stack([b["soft_segmentation"] for b in batch])
    return result


# ---------------------------------------------------------------------------
# Train & validate one epoch
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor,
    loss_weights: dict[str, float],
    device: torch.device,
    writer: SummaryWriter | None,
    global_step: int,
    discriminator: torch.nn.Module | None = None,
    optimizer_d: torch.optim.Optimizer | None = None,
    adversarial_weight: float = 0.0,
) -> tuple[float, int]:
    """Run one training epoch.

    Returns:
        ``(avg_loss, updated_global_step)``.
    """
    model.train()
    if discriminator is not None:
        discriminator.train()
    total_loss = 0.0
    num_batches = 0

    num_classes = class_weights.shape[0]

    for batch in loader:
        images = batch["image"].to(device)
        targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        outputs = model(images)
        loss, components = compute_loss(outputs, targets, class_weights, loss_weights)

        # --- Discriminator step ---
        if discriminator is not None and optimizer_d is not None and adversarial_weight > 0:
            optimizer_d.zero_grad()
            seg_logits = outputs["segmentation"].detach()
            seg_soft = torch.softmax(seg_logits, dim=1)  # [B, C, H, W]

            # De-normalize image for discriminator (approx: just use raw tensor)
            img_raw = images

            # Real: image + one-hot GT
            gt_seg = targets["segmentation"].long()  # [B, H, W]
            gt_onehot = torch.zeros(gt_seg.shape[0], num_classes, *gt_seg.shape[1:], device=device)
            gt_valid = gt_seg.clamp(0, num_classes - 1)
            gt_onehot.scatter_(1, gt_valid.unsqueeze(1), 1.0)
            # Zero out ignore pixels
            ignore = (gt_seg < 0) | (gt_seg >= num_classes)
            gt_onehot[:, :, ignore.squeeze() if ignore.dim() == 2 else True] = 0.0

            real_input = torch.cat([img_raw, gt_onehot], dim=1)
            pred_real = discriminator(real_input)
            loss_d_real = F.mse_loss(pred_real, torch.ones_like(pred_real) * 0.9)  # label smoothing

            # Fake: image + softmax prediction
            fake_input = torch.cat([img_raw, seg_soft], dim=1)
            pred_fake = discriminator(fake_input)
            loss_d_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake) + 0.1)

            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            optimizer_d.step()

            components["loss/disc"] = float(loss_d)

            # --- Generator adversarial loss ---
            seg_soft_g = torch.softmax(outputs["segmentation"], dim=1)
            fake_input_g = torch.cat([img_raw, seg_soft_g], dim=1)
            pred_fake_g = discriminator(fake_input_g)
            gan_g_loss = F.mse_loss(pred_fake_g, torch.ones_like(pred_fake_g))
            loss = loss + adversarial_weight * gan_g_loss
            components["loss/gan_g"] = float(gan_g_loss)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss)
        num_batches += 1
        global_step += 1

        if writer is not None:
            for name, value in components.items():
                writer.add_scalar(f"train/{name}", value, global_step)

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, global_step


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    class_weights: torch.Tensor,
    loss_weights: dict[str, float],
    device: torch.device,
    metrics: SegmentationMetrics,
) -> tuple[float, float, dict[str, float]]:
    """Run validation and compute metrics.

    Returns:
        ``(avg_loss, miou, per_class_iou)``.
    """
    model.eval()
    metrics.reset()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        outputs = model(images)
        loss, _ = compute_loss(outputs, targets, class_weights, loss_weights)
        total_loss += float(loss)
        num_batches += 1

        # Compute segmentation predictions for metrics
        seg_pred = outputs["segmentation"].argmax(dim=1).cpu().numpy()
        seg_target = batch["segmentation"].numpy()
        metrics.update(seg_pred, seg_target)

    avg_loss = total_loss / max(num_batches, 1)
    miou = metrics.miou()
    per_class = metrics.per_class_iou()
    return avg_loss, miou, per_class


# ---------------------------------------------------------------------------
# Sample overlay logging
# ---------------------------------------------------------------------------


def log_sample_overlays(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    num_samples: int = 4,
) -> None:
    """Log side-by-side prediction overlays to TensorBoard."""
    model.eval()
    batch = next(iter(loader))
    images = batch["image"][:num_samples].to(device)
    gt_masks = batch["segmentation"][:num_samples].numpy()

    with torch.no_grad():
        outputs = model(images)
    pred_masks = outputs["segmentation"][:num_samples].argmax(dim=1).cpu().numpy()

    # Normalize masks to [0, 1] for visualization (scale by num_classes)
    num_classes = outputs["segmentation"].shape[1]
    for i in range(min(num_samples, len(images))):
        gt_vis = torch.tensor(gt_masks[i], dtype=torch.float32) / num_classes
        pred_vis = torch.tensor(pred_masks[i], dtype=torch.float32) / num_classes
        writer.add_image(f"samples/{i}/gt_mask", gt_vis.unsqueeze(0), epoch)
        writer.add_image(f"samples/{i}/pred_mask", pred_vis.unsqueeze(0), epoch)

        # Log depth prediction
        depth_pred = outputs["depth"][i].cpu()  # [1, H, W]
        writer.add_image(f"samples/{i}/depth_pred", depth_pred, epoch)

        # Log normals prediction (shift from [-1,1] to [0,1] for visualization)
        normals_pred = (outputs["normals"][i].cpu() + 1.0) / 2.0  # [3, H, W]
        writer.add_image(f"samples/{i}/normals_pred", normals_pred, epoch)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def select_device() -> torch.device:
    """Auto-detect the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(config: dict, resume_path: str | None = None, args: argparse.Namespace | None = None) -> None:
    """Run the full training loop.

    Args:
        config: Parsed YAML config dict.
        resume_path: Optional path to checkpoint for resuming training.
        args: Parsed CLI args (for --reset-epochs flag).
    """
    device = select_device()
    logger.info("Using device: %s", device)

    # ---- Config sections ----
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    loss_cfg = config.get("loss", {})
    ckpt_cfg = config.get("checkpointing", {})

    # ---- Datasets ----
    dataset_dirs = [Path(d) for d in data_cfg.get("dataset_dirs", ["../output/segmentation"])]
    ds_config = DatasetConfig.from_dict(config)

    train_dataset = SegmentationDataset(dataset_dirs, split="train", augment=True, config=ds_config)
    val_dataset = SegmentationDataset(dataset_dirs, split="val", augment=False, config=ds_config)

    if len(train_dataset) == 0:
        logger.error(
            "Training dataset is empty — cannot train. Check dataset_dirs: %s", dataset_dirs
        )
        return
    if len(val_dataset) == 0:
        logger.warning("Validation dataset is empty — metrics will not be computed")

    batch_size = train_cfg.get("batch_size", 8)
    num_workers = train_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ---- Model ----
    num_classes = model_cfg.get("num_classes", 22)
    pretrained = model_cfg.get("pretrained_backbone", True)
    backbone = model_cfg.get("backbone", "mobilenet_v3_large")
    backbone_weights_path = model_cfg.get("backbone_weights_path")
    # CLI flag overrides config
    if getattr(args, "backbone_weights", None):
        backbone_weights_path = args.backbone_weights
    model = SegmentationModel(
        num_classes=num_classes,
        pretrained_backbone=pretrained,
        backbone=backbone,
        backbone_weights_path=backbone_weights_path,
    )
    model = model.to(device)

    # ---- Class weights ----
    logger.info("Computing class weights from training set (%d examples)...", len(train_dataset))
    class_weights = compute_class_weights(train_dataset, num_classes=num_classes).to(device)

    # ---- Optimizer ----
    lr = train_cfg.get("learning_rate", 1e-4)
    weight_decay = train_cfg.get("weight_decay", 1e-5)
    opt_name = train_cfg.get("optimizer", "adam").lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---- Scheduler + Resume ----
    epochs = train_cfg.get("epochs", 200)
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    start_epoch = 0
    global_step = 0

    if resume_path is not None:
        # Load model weights and optimizer state (skip scheduler — we'll create a fresh one)
        info = load_checkpoint(resume_path, model, optimizer, scheduler=None)
        if getattr(args, "reset_epochs", False):
            # Fresh training run from checkpoint weights — reset epoch counter
            start_epoch = 0
            logger.info("Loaded weights from %s (epoch %d) — resetting epoch counter to 0", resume_path, info["epoch"])
        else:
            start_epoch = info["epoch"] + 1
            global_step = start_epoch * len(train_loader)
            if start_epoch >= epochs:
                # Treat configured epochs as "additional epochs from resume point"
                total_new_epochs = epochs
                epochs = start_epoch + total_new_epochs
                logger.info("Resuming from epoch %d — training %d more epochs (to %d)", start_epoch, total_new_epochs, epochs)
            else:
                logger.info("Resuming from epoch %d", start_epoch)
        # Reset optimizer LR to configured value (checkpoint may have decayed LR)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

    # Create scheduler spanning only the remaining training window
    remaining_epochs = epochs - start_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(remaining_epochs - warmup_epochs, 1), eta_min=0
    )

    # ---- Early stopping ----
    patience = ckpt_cfg.get("early_stopping_patience", 20)
    es_metric = ckpt_cfg.get("early_stopping_metric", "val/miou")
    early_stopping = EarlyStopping(patience=patience, metric_name=es_metric, mode="max")

    # ---- TensorBoard ----
    save_dir = Path(ckpt_cfg.get("save_dir", "./checkpoints/segmentation"))
    writer = SummaryWriter(log_dir=str(save_dir / "logs"))

    # ---- Metrics ----
    val_metrics = SegmentationMetrics(num_classes=num_classes)

    # ---- GAN discriminator (optional) ----
    adversarial_weight = loss_cfg.get("adversarial_weight", 0.0)
    discriminator = None
    optimizer_d = None
    if adversarial_weight > 0:
        from training.models.seg_discriminator import SegPatchDiscriminator

        disc_in_ch = 3 + num_classes  # image RGB + seg one-hot/softmax
        discriminator = SegPatchDiscriminator(in_channels=disc_in_ch).to(device)
        disc_lr = loss_cfg.get("discriminator_lr", 2e-4)
        optimizer_d = torch.optim.Adam(
            discriminator.parameters(), lr=disc_lr, betas=(0.5, 0.999),
        )
        d_params = sum(p.numel() for p in discriminator.parameters())
        logger.info("SegPatchDiscriminator enabled (weight=%.3f, %s params)", adversarial_weight, f"{d_params:,}")

    # ---- Training loop ----
    best_miou = 0.0
    sample_overlay_interval = 10  # log overlays every N epochs

    logger.info(
        "Starting training: %d epochs, batch_size=%d, lr=%.1e, device=%s",
        epochs,
        batch_size,
        lr,
        device,
    )

    for epoch in range(start_epoch, epochs):
        # Adjust learning rate (warmup then cosine)
        current_lr = adjust_lr(optimizer, epoch, lr, warmup_epochs, scheduler)
        writer.add_scalar("train/lr", current_lr, epoch)

        # Train
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, class_weights, loss_cfg, device, writer, global_step,
            discriminator=discriminator, optimizer_d=optimizer_d,
            adversarial_weight=adversarial_weight,
        )
        writer.add_scalar("train/epoch_loss", train_loss, epoch)

        # Validate
        val_loss, miou, per_class_iou = validate(
            model, val_loader, class_weights, loss_cfg, device, val_metrics
        )
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/miou", miou, epoch)
        for cls_name, iou_val in per_class_iou.items():
            writer.add_scalar(f"val/iou/{cls_name}", iou_val, epoch)

        logger.info(
            "Epoch %d/%d — train_loss=%.4f, val_loss=%.4f, mIoU=%.4f, lr=%.1e",
            epoch + 1,
            epochs,
            train_loss,
            val_loss,
            miou,
            current_lr,
        )

        # Sample overlays
        if epoch % sample_overlay_interval == 0 and len(val_loader) > 0:
            log_sample_overlays(model, val_loader, device, writer, epoch)

        # Checkpoint: save latest
        epoch_metrics = {"val/miou": miou, "val/loss": val_loss, "train/loss": train_loss}
        save_checkpoint(model, optimizer, scheduler, epoch, epoch_metrics, save_dir / "latest.pt")

        # Checkpoint: save best
        if miou > best_miou:
            best_miou = miou
            save_checkpoint(model, optimizer, scheduler, epoch, epoch_metrics, save_dir / "best.pt")
            logger.info("New best mIoU: %.4f (epoch %d)", best_miou, epoch + 1)

        # Early stopping
        if early_stopping.step(epoch_metrics):
            logger.info("Early stopping triggered at epoch %d", epoch + 1)
            break

    writer.close()
    logger.info("Training complete. Best mIoU: %.4f", best_miou)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for segmentation training."""
    parser = argparse.ArgumentParser(description="Train Strata segmentation model")
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/segmentation.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--reset-epochs",
        action="store_true",
        help="Reset epoch counter to 0 when resuming (treat as fresh run with pretrained weights)",
    )
    parser.add_argument(
        "--backbone-weights",
        type=str,
        default=None,
        help="Path to a backbone state_dict to load after construction (overrides config.model.backbone_weights_path)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    train(config, resume_path=args.resume, args=args)


if __name__ == "__main__":
    main()
