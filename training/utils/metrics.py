"""Streaming training metrics for segmentation and joint prediction.

Provides memory-efficient metric classes that accumulate statistics across
batches without storing all predictions:

- :class:`SegmentationMetrics` — confusion matrix → mIoU, per-class IoU, accuracy
- :class:`JointMetrics` — per-joint MSE and presence accuracy

All metrics operate on numpy arrays (decoupled from PyTorch tensors).

Pure Python + NumPy (no Blender or PyTorch dependency).
"""

from __future__ import annotations

import numpy as np

from pipeline.config import REGION_NAMES
from training.data.transforms import BONE_ORDER

# ---------------------------------------------------------------------------
# Extended region names (22 classes matching model output)
# ---------------------------------------------------------------------------

_EXTENDED_REGION_NAMES: dict[int, str] = {
    **REGION_NAMES,
    20: "unused",
    21: "accessory",
}


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------


class SegmentationMetrics:
    """Streaming confusion-matrix metrics for semantic segmentation.

    Accumulates a ``(num_classes, num_classes)`` confusion matrix across
    batches and derives mIoU, per-class IoU, per-class accuracy, and
    overall pixel accuracy without storing all predictions in memory.

    Args:
        num_classes: Number of segmentation classes (default 22).
        ignore_index: Label value to ignore when updating (default -1).
    """

    def __init__(self, num_classes: int = 22, ignore_index: int = -1) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        """Accumulate predictions into the confusion matrix.

        Args:
            pred: Predicted class IDs, shape ``[B, H, W]`` or ``[H, W]``.
            target: Ground-truth class IDs, same shape as *pred*.
        """
        pred_flat = pred.ravel()
        target_flat = target.ravel()

        # Filter out ignore_index pixels.
        mask = target_flat != self.ignore_index
        pred_flat = pred_flat[mask]
        target_flat = target_flat[mask]

        # np.bincount trick: linearize (target, pred) pairs into a 1-D index,
        # count occurrences, then reshape into the confusion matrix.
        indices = target_flat * self.num_classes + pred_flat
        counts = np.bincount(indices, minlength=self.num_classes**2)
        self.confusion += counts.reshape(self.num_classes, self.num_classes)

    def reset(self) -> None:
        """Zero the confusion matrix."""
        self.confusion[:] = 0

    def miou(self) -> float:
        """Mean Intersection-over-Union across classes with > 0 GT pixels."""
        ious = self._class_ious()
        # Exclude classes with no ground-truth pixels.
        gt_per_class = self.confusion.sum(axis=1)
        present = gt_per_class > 0
        if not present.any():
            return 0.0
        return float(ious[present].mean())

    def per_class_iou(self) -> dict[str, float]:
        """Return ``{region_name: IoU}`` for all classes."""
        ious = self._class_ious()
        return {
            _EXTENDED_REGION_NAMES.get(i, f"class_{i}"): float(ious[i])
            for i in range(self.num_classes)
        }

    def per_class_accuracy(self) -> dict[str, float]:
        """Return ``{region_name: accuracy}`` for all classes.

        Per-class accuracy = TP / (TP + FN) = diagonal / row sum.
        """
        row_sums = self.confusion.sum(axis=1)
        diag = self.confusion.diagonal()
        with np.errstate(divide="ignore", invalid="ignore"):
            acc = np.where(row_sums > 0, diag / row_sums, 0.0)
        return {
            _EXTENDED_REGION_NAMES.get(i, f"class_{i}"): float(acc[i])
            for i in range(self.num_classes)
        }

    def overall_accuracy(self) -> float:
        """Pixel accuracy across all classes."""
        total = self.confusion.sum()
        if total == 0:
            return 0.0
        return float(self.confusion.diagonal().sum() / total)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _class_ious(self) -> np.ndarray:
        """Compute IoU for each class from the confusion matrix.

        IoU_i = C[i,i] / (row_i + col_i - C[i,i])
        """
        diag = self.confusion.diagonal()
        row_sums = self.confusion.sum(axis=1)
        col_sums = self.confusion.sum(axis=0)
        union = row_sums + col_sums - diag
        with np.errstate(divide="ignore", invalid="ignore"):
            ious = np.where(union > 0, diag / union, 0.0)
        return ious


# ---------------------------------------------------------------------------
# Joint metrics
# ---------------------------------------------------------------------------


class JointMetrics:
    """Streaming metrics for joint offset prediction and presence detection.

    Tracks per-joint mean squared error (for visible joints only) and
    binary presence/visibility accuracy.

    Args:
        num_joints: Number of joints (default 20, matching BONE_ORDER).
    """

    def __init__(self, num_joints: int = 20) -> None:
        self.num_joints = num_joints
        self._squared_errors = np.zeros(num_joints, dtype=np.float64)
        self._joint_counts = np.zeros(num_joints, dtype=np.int64)
        self._presence_correct = 0
        self._presence_total = 0

    def update(
        self,
        pred_offsets: np.ndarray,
        gt_offsets: np.ndarray,
        pred_present: np.ndarray,
        gt_visible: np.ndarray,
    ) -> None:
        """Accumulate a batch of joint predictions.

        Args:
            pred_offsets: Predicted 2-D offsets, shape ``[B, J, 2]``.
            gt_offsets: Ground-truth offsets, shape ``[B, J, 2]``.
            pred_present: Predicted visibility, shape ``[B, J]`` (bool).
            gt_visible: Ground-truth visibility, shape ``[B, J]`` (bool).
        """
        gt_vis = gt_visible.astype(bool)

        # Per-joint squared error (only for visible GT joints).
        diff = pred_offsets - gt_offsets  # [B, J, 2]
        sq_err = (diff**2).sum(axis=-1)  # [B, J]
        for j in range(self.num_joints):
            vis_mask = gt_vis[:, j]
            if vis_mask.any():
                self._squared_errors[j] += sq_err[:, j][vis_mask].sum()
                self._joint_counts[j] += int(vis_mask.sum())

        # Presence accuracy.
        self._presence_correct += int((pred_present.astype(bool) == gt_vis).sum())
        self._presence_total += int(gt_vis.size)

    def reset(self) -> None:
        """Zero all accumulators."""
        self._squared_errors[:] = 0.0
        self._joint_counts[:] = 0
        self._presence_correct = 0
        self._presence_total = 0

    def mean_offset_error(self) -> float:
        """Mean squared error averaged across all joints with observations."""
        active = self._joint_counts > 0
        if not active.any():
            return 0.0
        per_joint = self._squared_errors[active] / self._joint_counts[active]
        return float(per_joint.mean())

    def per_joint_error(self) -> dict[str, float]:
        """Return ``{joint_name: MSE}`` for each joint."""
        result: dict[str, float] = {}
        for j in range(self.num_joints):
            name = BONE_ORDER[j] if j < len(BONE_ORDER) else f"joint_{j}"
            if self._joint_counts[j] > 0:
                result[name] = float(self._squared_errors[j] / self._joint_counts[j])
            else:
                result[name] = 0.0
        return result

    def presence_accuracy(self) -> float:
        """Binary accuracy of visibility/presence prediction."""
        if self._presence_total == 0:
            return 0.0
        return float(self._presence_correct / self._presence_total)
