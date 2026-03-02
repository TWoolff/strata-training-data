"""Training utilities — metrics, checkpointing, logging helpers."""

from training.utils.checkpoint import EarlyStopping
from training.utils.metrics import JointMetrics, SegmentationMetrics

__all__ = [
    "EarlyStopping",
    "JointMetrics",
    "SegmentationMetrics",
]


def __getattr__(name: str):
    """Lazy imports for torch-dependent checkpoint functions."""
    if name in ("load_checkpoint", "save_checkpoint"):
        from training.utils.checkpoint import load_checkpoint, save_checkpoint

        return load_checkpoint if name == "load_checkpoint" else save_checkpoint
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
