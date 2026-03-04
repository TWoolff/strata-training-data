"""Train the joint refinement model.

CLI entry point::

    python training/train_joints.py --config training/configs/joints.yaml
    python training/train_joints.py --config training/configs/joints.yaml \\
        --resume checkpoints/joints/best.pt

Trains on pipeline output with combined offset SmoothL1 + presence BCE +
confidence BCE loss. Logs to TensorBoard and saves best/latest checkpoints.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from training.data.joint_dataset import JointDataset, JointDatasetConfig
from training.models.joint_model import JointModel
from training.utils.checkpoint import EarlyStopping, load_checkpoint, save_checkpoint
from training.utils.metrics import JointMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------


def compute_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    loss_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute combined joint refinement loss.

    Args:
        outputs: Model output dict with ``offsets`` ``[B, 2, 20]``,
            ``confidence`` ``[B, 20]``, ``present`` ``[B, 20]``.
        targets: Target dict with ``gt_offsets`` ``[B, 2, 20]``,
            ``gt_visible`` ``[B, 20]``.
        loss_weights: Dict with ``offset_weight``, ``presence_weight``,
            ``confidence_weight`` scalars.

    Returns:
        ``(total_loss, component_dict)`` where ``component_dict`` maps loss
        names to their scalar values (for logging).
    """
    gt_offsets = targets["gt_offsets"]  # [B, 2, 20]
    gt_visible = targets["gt_visible"]  # [B, 20]

    pred_offsets = outputs["offsets"]  # [B, 2, 20]
    pred_present = outputs["present"]  # [B, 20]
    pred_confidence = outputs["confidence"]  # [B, 20]

    # Offset loss: SmoothL1 for visible joints only
    vis_mask = gt_visible.bool().unsqueeze(1).expand_as(pred_offsets)  # [B, 2, 20]
    if vis_mask.any():
        offset_loss = F.smooth_l1_loss(
            pred_offsets[vis_mask], gt_offsets[vis_mask], reduction="mean"
        )
    else:
        offset_loss = torch.tensor(0.0, device=pred_offsets.device)

    # Presence loss: BCE for all joints
    presence_loss = F.binary_cross_entropy_with_logits(pred_present, gt_visible, reduction="mean")

    # Confidence target: high (1.0) when offset error is small, low (0.0) otherwise
    # Use per-joint L2 offset error; threshold at noise std level
    with torch.no_grad():
        per_joint_err = (pred_offsets - gt_offsets).pow(2).sum(dim=1).sqrt()  # [B, 20]
        conf_target = (per_joint_err < 0.03).float() * gt_visible

    confidence_loss = F.binary_cross_entropy_with_logits(
        pred_confidence, conf_target, reduction="mean"
    )

    # Weighted sum
    w_off = loss_weights.get("offset_weight", 1.0)
    w_pres = loss_weights.get("presence_weight", 1.0)
    w_conf = loss_weights.get("confidence_weight", 0.5)

    total = w_off * offset_loss + w_pres * presence_loss + w_conf * confidence_loss

    components = {
        "loss/offset": float(offset_loss),
        "loss/presence": float(presence_loss),
        "loss/confidence": float(confidence_loss),
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
    """Custom collate that stacks all tensor fields."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "gt_positions": torch.stack([b["gt_positions"] for b in batch]),
        "gt_visible": torch.stack([b["gt_visible"] for b in batch]),
        "geo_positions": torch.stack([b["geo_positions"] for b in batch]),
        "gt_offsets": torch.stack([b["gt_offsets"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Train & validate one epoch
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_weights: dict[str, float],
    device: torch.device,
    writer: SummaryWriter | None,
    global_step: int,
) -> tuple[float, int]:
    """Run one training epoch.

    Returns:
        ``(avg_loss, updated_global_step)``.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        targets = {k: v.to(device) for k, v in batch.items() if k != "image"}

        outputs = model(images)
        loss, components = compute_loss(outputs, targets, loss_weights)

        optimizer.zero_grad()
        loss.backward()
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
    loss_weights: dict[str, float],
    device: torch.device,
    metrics: JointMetrics,
) -> tuple[float, float, float]:
    """Run validation and compute metrics.

    Returns:
        ``(avg_loss, mean_offset_error, presence_accuracy)``.
    """
    model.eval()
    metrics.reset()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        targets = {k: v.to(device) for k, v in batch.items() if k != "image"}

        outputs = model(images)
        loss, _ = compute_loss(outputs, targets, loss_weights)
        total_loss += float(loss)
        num_batches += 1

        # Compute metrics on CPU numpy
        pred_offsets = outputs["offsets"].cpu().numpy()  # [B, 2, 20]
        gt_offsets = batch["gt_offsets"].numpy()  # [B, 2, 20]
        pred_present = (torch.sigmoid(outputs["present"]) > 0.5).cpu().numpy()  # [B, 20]
        gt_visible = batch["gt_visible"].numpy()  # [B, 20]

        # JointMetrics expects [B, J, 2] layout
        pred_offsets_jxy = np.transpose(pred_offsets, (0, 2, 1))  # [B, 20, 2]
        gt_offsets_jxy = np.transpose(gt_offsets, (0, 2, 1))  # [B, 20, 2]
        metrics.update(pred_offsets_jxy, gt_offsets_jxy, pred_present, gt_visible)

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, metrics.mean_offset_error(), metrics.presence_accuracy()


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


def train(config: dict, resume_path: str | None = None) -> None:
    """Run the full training loop.

    Args:
        config: Parsed YAML config dict.
        resume_path: Optional path to checkpoint for resuming training.
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
    ds_config = JointDatasetConfig.from_dict(config)

    train_dataset = JointDataset(dataset_dirs, split="train", augment=True, config=ds_config)
    val_dataset = JointDataset(dataset_dirs, split="val", augment=False, config=ds_config)

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
    num_joints = model_cfg.get("num_joints", 20)
    pretrained = model_cfg.get("pretrained_backbone", True)
    model = JointModel(num_joints=num_joints, pretrained_backbone=pretrained)
    model = model.to(device)

    # ---- Optimizer ----
    lr = train_cfg.get("learning_rate", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---- Scheduler (cosine annealing after warmup) ----
    epochs = train_cfg.get("epochs", 50)
    warmup_epochs = train_cfg.get("warmup_epochs", 3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=0
    )

    # ---- Early stopping ----
    patience = ckpt_cfg.get("early_stopping_patience", 15)
    es_metric = ckpt_cfg.get("early_stopping_metric", "val/mean_offset_error")
    early_stopping = EarlyStopping(patience=patience, metric_name=es_metric, mode="min")

    # ---- Resume from checkpoint ----
    start_epoch = 0
    global_step = 0
    if resume_path is not None:
        info = load_checkpoint(resume_path, model, optimizer, scheduler)
        start_epoch = info["epoch"] + 1
        global_step = start_epoch * len(train_loader)
        logger.info("Resuming from epoch %d", start_epoch)

    # ---- TensorBoard ----
    save_dir = Path(ckpt_cfg.get("save_dir", "./checkpoints/joints"))
    writer = SummaryWriter(log_dir=str(save_dir / "logs"))

    # ---- Metrics ----
    val_metrics = JointMetrics(num_joints=num_joints)

    # ---- Training loop ----
    best_error = float("inf")

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
            model, train_loader, optimizer, loss_cfg, device, writer, global_step
        )
        writer.add_scalar("train/epoch_loss", train_loss, epoch)

        # Validate
        val_loss, mean_err, pres_acc = validate(model, val_loader, loss_cfg, device, val_metrics)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/mean_offset_error", mean_err, epoch)
        writer.add_scalar("val/presence_accuracy", pres_acc, epoch)

        per_joint = val_metrics.per_joint_error()
        for joint_name, error_val in per_joint.items():
            writer.add_scalar(f"val/joint_error/{joint_name}", error_val, epoch)

        logger.info(
            "Epoch %d/%d — train_loss=%.4f, val_loss=%.4f, mean_err=%.6f, pres_acc=%.4f, lr=%.1e",
            epoch + 1,
            epochs,
            train_loss,
            val_loss,
            mean_err,
            pres_acc,
            current_lr,
        )

        # Checkpoint: save latest
        epoch_metrics = {
            "val/mean_offset_error": mean_err,
            "val/presence_accuracy": pres_acc,
            "val/loss": val_loss,
            "train/loss": train_loss,
        }
        save_checkpoint(model, optimizer, scheduler, epoch, epoch_metrics, save_dir / "latest.pt")

        # Checkpoint: save best
        if mean_err < best_error:
            best_error = mean_err
            save_checkpoint(model, optimizer, scheduler, epoch, epoch_metrics, save_dir / "best.pt")
            logger.info("New best mean_offset_error: %.6f (epoch %d)", best_error, epoch + 1)

        # Early stopping
        if early_stopping.step(epoch_metrics):
            logger.info("Early stopping triggered at epoch %d", epoch + 1)
            break

    writer.close()
    logger.info("Training complete. Best mean_offset_error: %.6f", best_error)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for joint refinement training."""
    parser = argparse.ArgumentParser(description="Train Strata joint refinement model")
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/joints.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
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

    train(config, resume_path=args.resume)


if __name__ == "__main__":
    main()
