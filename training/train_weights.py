"""Train the per-vertex weight prediction model.

CLI entry point::

    python training/train_weights.py --config training/configs/weights.yaml
    python training/train_weights.py --config training/configs/weights.yaml \\
        --resume checkpoints/weights/best.pt

Trains on pipeline weight JSON output with combined KL divergence (soft
weight labels) + BCE (confidence) loss. Logs to TensorBoard and saves
best/latest checkpoints.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from training.data.weight_dataset import WeightDataset, WeightDatasetConfig
from training.models.weight_prediction_model import WeightPredictionModel
from training.utils.checkpoint import EarlyStopping, load_checkpoint, save_checkpoint
from training.utils.metrics import WeightMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------


def compute_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    loss_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute combined weight prediction loss.

    Uses KL divergence between predicted weight distribution and ground-truth
    soft labels (GT weights are not one-hot — they represent smooth weight
    distributions across bones).

    Args:
        outputs: Model output dict with ``weights`` ``[B, 20, N, 1]`` raw logits
            and ``confidence`` ``[B, 1, N, 1]`` raw logits.
        targets: Target dict with ``weights_target`` ``[B, 20, N]``,
            ``confidence_target`` ``[B, N]``, ``num_vertices`` ``[B]``.
        loss_weights: Dict with ``weight_loss_weight`` and ``confidence_weight``.

    Returns:
        ``(total_loss, component_dict)`` where ``component_dict`` maps loss
        names to their scalar values (for logging).
    """
    pred_weights = outputs["weights"].squeeze(-1)  # [B, 20, N]
    pred_confidence = outputs["confidence"].squeeze(-1).squeeze(1)  # [B, N]

    gt_weights = targets["weights_target"]  # [B, 20, N]
    gt_confidence = targets["confidence_target"]  # [B, N]
    num_vertices = targets["num_vertices"]  # [B]

    batch_size = gt_weights.shape[0]
    device = pred_weights.device

    # Build per-vertex mask: only compute loss on real vertices with GT data
    n = gt_weights.shape[2]
    vertex_mask = torch.zeros(batch_size, n, device=device, dtype=torch.bool)
    for b in range(batch_size):
        nv = int(num_vertices[b])
        vertex_mask[b, :nv] = gt_confidence[b, :nv] > 0.5

    # Weight loss: KL divergence on vertices with GT data
    if vertex_mask.any():
        # Log-softmax of predictions, softmax of targets (already normalized)
        pred_log_softmax = F.log_softmax(pred_weights, dim=1)  # [B, 20, N]

        # KL(target || pred) = sum(target * (log(target) - log_softmax(pred)))
        # Use F.kl_div which expects log-predictions and targets
        # We need to transpose to [B, N, 20] for per-vertex loss, then mask
        pred_log_sm = pred_log_softmax.permute(0, 2, 1)  # [B, N, 20]
        gt_w = gt_weights.permute(0, 2, 1)  # [B, N, 20]

        # Compute per-vertex KL divergence
        kl_per_vertex = F.kl_div(pred_log_sm, gt_w, reduction="none", log_target=False).sum(
            dim=-1
        )  # [B, N]

        weight_loss = kl_per_vertex[vertex_mask].mean()
    else:
        weight_loss = torch.tensor(0.0, device=device)

    # Confidence loss: BCE on all real vertices (not just those with GT)
    real_mask = torch.zeros(batch_size, n, device=device, dtype=torch.bool)
    for b in range(batch_size):
        nv = int(num_vertices[b])
        real_mask[b, :nv] = True

    if real_mask.any():
        confidence_loss = F.binary_cross_entropy_with_logits(
            pred_confidence[real_mask],
            gt_confidence[real_mask],
            reduction="mean",
        )
    else:
        confidence_loss = torch.tensor(0.0, device=device)

    # Weighted sum
    w_weight = loss_weights.get("weight_loss_weight", 1.0)
    w_conf = loss_weights.get("confidence_weight", 0.5)

    total = w_weight * weight_loss + w_conf * confidence_loss

    components = {
        "loss/weight_kl": float(weight_loss),
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
    """Custom collate that stacks feature tensors and targets."""
    return {
        "features": torch.stack([b["features"] for b in batch]),
        "weights_target": torch.stack([b["weights_target"] for b in batch]),
        "confidence_target": torch.stack([b["confidence_target"] for b in batch]),
        "num_vertices": torch.tensor([b["num_vertices"] for b in batch], dtype=torch.long),
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
        features = batch["features"].to(device)
        targets = {k: v.to(device) for k, v in batch.items() if k != "features"}

        outputs = model(features)
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
    metrics: WeightMetrics,
) -> tuple[float, float, float]:
    """Run validation and compute metrics.

    Returns:
        ``(avg_loss, mae, confidence_accuracy)``.
    """
    model.eval()
    metrics.reset()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        features = batch["features"].to(device)
        targets = {k: v.to(device) for k, v in batch.items() if k != "features"}

        outputs = model(features)
        loss, _ = compute_loss(outputs, targets, loss_weights)
        total_loss += float(loss)
        num_batches += 1

        # Compute metrics on CPU numpy
        pred_weights_logits = outputs["weights"].squeeze(-1).cpu()  # [B, 20, N]
        pred_weights = F.softmax(pred_weights_logits, dim=1).numpy()
        gt_weights = batch["weights_target"].numpy()  # [B, 20, N]

        pred_conf = torch.sigmoid(outputs["confidence"].squeeze(-1).squeeze(1)) > 0.5
        pred_conf = pred_conf.cpu().numpy()  # [B, N]
        gt_conf = batch["confidence_target"].numpy()  # [B, N]
        num_verts = batch["num_vertices"].numpy()  # [B]

        metrics.update(pred_weights, gt_weights, pred_conf, gt_conf, num_verts)

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, metrics.mae(), metrics.confidence_accuracy()


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
    ds_config = WeightDatasetConfig.from_dict(config)

    train_dataset = WeightDataset(dataset_dirs, split="train", config=ds_config)
    val_dataset = WeightDataset(dataset_dirs, split="val", config=ds_config)

    if len(train_dataset) == 0:
        logger.error(
            "Training dataset is empty — cannot train. Check dataset_dirs: %s", dataset_dirs
        )
        return
    if len(val_dataset) == 0:
        logger.warning("Validation dataset is empty — metrics will not be computed")

    batch_size = train_cfg.get("batch_size", 16)
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
    num_features = model_cfg.get("num_features", 31)
    num_bones = model_cfg.get("num_bones", 20)
    model = WeightPredictionModel(num_features=num_features, num_bones=num_bones)
    model = model.to(device)

    # ---- Optimizer ----
    lr = train_cfg.get("learning_rate", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---- Scheduler (cosine annealing after warmup) ----
    epochs = train_cfg.get("epochs", 100)
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=0
    )

    # ---- Early stopping ----
    patience = ckpt_cfg.get("early_stopping_patience", 20)
    es_metric = ckpt_cfg.get("early_stopping_metric", "val/mae")
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
    save_dir = Path(ckpt_cfg.get("save_dir", "./checkpoints/weights"))
    writer = SummaryWriter(log_dir=str(save_dir / "logs"))

    # ---- Metrics ----
    val_metrics = WeightMetrics(num_bones=num_bones)

    # ---- Training loop ----
    best_mae = float("inf")

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
        val_loss, mae, conf_acc = validate(model, val_loader, loss_cfg, device, val_metrics)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/mae", mae, epoch)
        writer.add_scalar("val/confidence_accuracy", conf_acc, epoch)

        per_bone = val_metrics.per_bone_mae()
        for bone_name, error_val in per_bone.items():
            writer.add_scalar(f"val/bone_mae/{bone_name}", error_val, epoch)

        logger.info(
            "Epoch %d/%d — train_loss=%.4f, val_loss=%.4f, mae=%.6f, conf_acc=%.4f, lr=%.1e",
            epoch + 1,
            epochs,
            train_loss,
            val_loss,
            mae,
            conf_acc,
            current_lr,
        )

        # Checkpoint: save latest
        epoch_metrics = {
            "val/mae": mae,
            "val/confidence_accuracy": conf_acc,
            "val/loss": val_loss,
            "train/loss": train_loss,
        }
        save_checkpoint(model, optimizer, scheduler, epoch, epoch_metrics, save_dir / "latest.pt")

        # Checkpoint: save best
        if mae < best_mae:
            best_mae = mae
            save_checkpoint(model, optimizer, scheduler, epoch, epoch_metrics, save_dir / "best.pt")
            logger.info("New best MAE: %.6f (epoch %d)", best_mae, epoch + 1)

        # Early stopping
        if early_stopping.step(epoch_metrics):
            logger.info("Early stopping triggered at epoch %d", epoch + 1)
            break

    writer.close()
    logger.info("Training complete. Best MAE: %.6f", best_mae)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for weight prediction training."""
    parser = argparse.ArgumentParser(description="Train Strata per-vertex weight prediction model")
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/weights.yaml",
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
