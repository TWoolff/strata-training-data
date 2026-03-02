"""Checkpoint save/load and early stopping for training loops.

Provides:
- :func:`save_checkpoint` — serialize model, optimizer, scheduler, and metrics
- :func:`load_checkpoint` — restore training state from a ``.pt`` file
- :class:`EarlyStopping` — patience-based stopping on a monitored metric
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    epoch: int,
    metrics: dict,
    path: str | Path,
) -> None:
    """Save training checkpoint to disk.

    Args:
        model: The model to save.
        optimizer: Optimizer state.
        scheduler: LR scheduler state (must have ``state_dict()``).
        epoch: Current epoch number.
        metrics: Dict of metric values at this checkpoint.
        path: File path for the checkpoint.
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        state["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(state, path)
    logger.info("Saved checkpoint to %s (epoch %d)", path, epoch)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
) -> dict:
    """Load training checkpoint from disk.

    Args:
        path: Path to the ``.pt`` checkpoint file.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore state.
        scheduler: Optional scheduler to restore state.

    Returns:
        Dict with ``epoch`` and ``metrics`` from the checkpoint.
    """
    import torch

    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Loaded model weights from %s", path)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
    }


class EarlyStopping:
    """Patience-based early stopping on a monitored metric.

    Args:
        patience: Number of epochs without improvement before stopping.
        metric_name: Name of the metric to monitor (for logging).
        mode: ``"max"`` if higher is better, ``"min"`` if lower is better.
    """

    def __init__(self, patience: int, metric_name: str = "val/miou", mode: str = "max") -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.patience = patience
        self.metric_name = metric_name
        self.mode = mode
        self.best_value: float | None = None
        self.epochs_without_improvement = 0

    def step(self, metrics: dict) -> bool:
        """Check if training should stop.

        Args:
            metrics: Dict containing the monitored metric.

        Returns:
            ``True`` if training should stop (patience exhausted).
        """
        value = metrics.get(self.metric_name)
        if value is None:
            logger.warning("Metric %r not found in metrics dict — skipping early stopping check")
            return False

        improved = (
            self.best_value is None
            or (self.mode == "max" and value > self.best_value)
            or (self.mode == "min" and value < self.best_value)
        )

        if improved:
            self.best_value = value
            self.epochs_without_improvement = 0
            return False

        self.epochs_without_improvement += 1
        if self.epochs_without_improvement >= self.patience:
            logger.info(
                "Early stopping: %s did not improve for %d epochs (best=%.4f)",
                self.metric_name,
                self.patience,
                self.best_value,
            )
            return True

        return False
