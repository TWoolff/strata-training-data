"""Visualization utilities for model evaluation.

Functions for overlaying segmentation masks, plotting confusion matrices,
generating prediction grids, and visualizing joint predictions.

Uses REGION_COLORS from pipeline/config.py for consistent color coding.

Pure Python + NumPy + matplotlib (no PyTorch or Blender dependency).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from pipeline.config import REGION_COLORS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color LUT from REGION_COLORS
# ---------------------------------------------------------------------------

_COLOR_LUT: np.ndarray | None = None


def _get_color_lut() -> np.ndarray:
    """Build a 256×3 uint8 lookup table from REGION_COLORS."""
    global _COLOR_LUT
    if _COLOR_LUT is None:
        lut = np.zeros((256, 3), dtype=np.uint8)
        for region_id, rgb in REGION_COLORS.items():
            if 0 <= region_id < 256:
                lut[region_id] = rgb
        _COLOR_LUT = lut
    return _COLOR_LUT


# ---------------------------------------------------------------------------
# Segmentation overlays
# ---------------------------------------------------------------------------


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a grayscale region-ID mask to an RGB color image.

    Args:
        mask: 2D uint8 array where pixel values are region IDs (0–21).

    Returns:
        3D uint8 array of shape ``[H, W, 3]`` with RGB colors.
    """
    lut = _get_color_lut()
    return lut[mask]


def overlay_segmentation(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay color-coded segmentation on an image.

    Args:
        image: RGB image, shape ``[H, W, 3]``, uint8.
        mask: Region-ID mask, shape ``[H, W]``, uint8.
        alpha: Blend factor for the overlay (0 = image only, 1 = mask only).

    Returns:
        Blended RGB image, shape ``[H, W, 3]``, uint8.
    """
    colored = colorize_mask(mask)
    blended = (1 - alpha) * image.astype(np.float32) + alpha * colored.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def save_prediction_grid(
    images: list[np.ndarray],
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
    output_path: str | Path,
    n: int = 8,
) -> None:
    """Save a grid of image | GT mask | predicted mask triples.

    Args:
        images: List of RGB images, each ``[H, W, 3]`` uint8.
        gt_masks: List of ground-truth masks, each ``[H, W]`` uint8.
        pred_masks: List of predicted masks, each ``[H, W]`` uint8.
        output_path: Path to save the output PNG.
        n: Maximum number of rows in the grid.
    """
    import matplotlib.pyplot as plt

    n = min(n, len(images))
    if n == 0:
        logger.warning("No examples to plot — skipping prediction grid")
        return

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(colorize_mask(gt_masks[i]))
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(colorize_mask(pred_masks[i]))
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved prediction grid to %s", output_path)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    confusion: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
    normalize: bool = True,
) -> None:
    """Save a confusion matrix heatmap as PNG.

    Args:
        confusion: Square confusion matrix ``[C, C]``.
        class_names: List of class name strings.
        output_path: Path to save the output PNG.
        normalize: If True, normalize each row to sum to 1.
    """
    import matplotlib.pyplot as plt

    n = len(class_names)
    matrix = confusion[:n, :n].astype(np.float64)

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            matrix = np.where(row_sums > 0, matrix / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.6), max(8, n * 0.5)))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1.0 if normalize else None)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))

    # Annotate cells with values > 0.01
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if val > 0.01:
                text_color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=text_color)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", output_path)


# ---------------------------------------------------------------------------
# Per-class IoU bar chart
# ---------------------------------------------------------------------------


def plot_per_class_iou(
    iou_dict: dict[str, float],
    output_path: str | Path,
) -> None:
    """Save a horizontal bar chart of per-class IoU values.

    Args:
        iou_dict: Mapping of class name to IoU value.
        output_path: Path to save the output PNG.
    """
    import matplotlib.pyplot as plt

    names = list(iou_dict.keys())
    values = list(iou_dict.values())

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.35)))
    y_pos = range(len(names))
    bars = ax.barh(y_pos, values, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("IoU")
    ax.set_title("Per-Class IoU")

    for bar, val in zip(bars, values, strict=True):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=7,
        )

    ax.invert_yaxis()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved per-class IoU chart to %s", output_path)


# ---------------------------------------------------------------------------
# Joint overlays
# ---------------------------------------------------------------------------


def overlay_joints(
    image: np.ndarray,
    joints: dict[str, tuple[float, float]],
    color: tuple[int, int, int] = (0, 255, 0),
    radius: int = 3,
) -> np.ndarray:
    """Draw joint positions on an image.

    Args:
        image: RGB image, shape ``[H, W, 3]``, uint8.
        joints: Mapping of joint name to ``(x, y)`` pixel coordinates.
        color: RGB color for the joint circles.
        radius: Radius in pixels.

    Returns:
        Copy of the image with joint circles drawn.
    """
    import cv2

    out = image.copy()
    for _name, (x, y) in joints.items():
        cx, cy = round(x), round(y)
        cv2.circle(out, (cx, cy), radius, color, -1)
    return out


def save_joint_comparison(
    image: np.ndarray,
    gt_joints: dict[str, tuple[float, float]],
    pred_joints: dict[str, tuple[float, float]],
    output_path: str | Path,
) -> None:
    """Save an overlay of GT vs predicted joints on the image.

    GT joints are drawn in green, predicted joints in red.

    Args:
        image: RGB image, ``[H, W, 3]`` uint8.
        gt_joints: Ground-truth joint positions ``{name: (x, y)}``.
        pred_joints: Predicted joint positions ``{name: (x, y)}``.
        output_path: Path to save the output PNG.
    """
    import cv2

    out = overlay_joints(image, gt_joints, color=(0, 255, 0), radius=4)
    out = overlay_joints(out, pred_joints, color=(255, 0, 0), radius=3)

    # Draw lines connecting GT to predicted for each joint
    for name in gt_joints:
        if name in pred_joints:
            gt_pt = (round(gt_joints[name][0]), round(gt_joints[name][1]))
            pred_pt = (round(pred_joints[name][0]), round(pred_joints[name][1]))
            cv2.line(out, gt_pt, pred_pt, (255, 255, 0), 1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.fromarray(out).save(output_path)
    logger.info("Saved joint comparison to %s", output_path)


# ---------------------------------------------------------------------------
# Joint scatter plot
# ---------------------------------------------------------------------------


def plot_joint_scatter(
    gt_positions: np.ndarray,
    pred_positions: np.ndarray,
    output_path: str | Path,
) -> None:
    """Save a scatter plot of predicted vs ground-truth joint positions.

    Args:
        gt_positions: Ground-truth positions, shape ``[N, 2]``.
        pred_positions: Predicted positions, shape ``[N, 2]``.
        output_path: Path to save the output PNG.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].scatter(gt_positions[:, 0], pred_positions[:, 0], alpha=0.3, s=4)
    axes[0].plot([0, 1], [0, 1], "r--", linewidth=1)
    axes[0].set_xlabel("GT x")
    axes[0].set_ylabel("Pred x")
    axes[0].set_title("X coordinates")
    axes[0].set_aspect("equal")

    axes[1].scatter(gt_positions[:, 1], pred_positions[:, 1], alpha=0.3, s=4)
    axes[1].plot([0, 1], [0, 1], "r--", linewidth=1)
    axes[1].set_xlabel("GT y")
    axes[1].set_ylabel("Pred y")
    axes[1].set_title("Y coordinates")
    axes[1].set_aspect("equal")

    fig.suptitle("Predicted vs Ground Truth Joint Positions")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved joint scatter plot to %s", output_path)


# ---------------------------------------------------------------------------
# Per-joint error bar chart
# ---------------------------------------------------------------------------


def plot_per_joint_error(
    error_dict: dict[str, float],
    output_path: str | Path,
) -> None:
    """Save a horizontal bar chart of per-joint MSE values.

    Args:
        error_dict: Mapping of joint name to MSE value.
        output_path: Path to save the output PNG.
    """
    import matplotlib.pyplot as plt

    names = list(error_dict.keys())
    values = list(error_dict.values())

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.35)))
    y_pos = range(len(names))
    bars = ax.barh(y_pos, values, color="coral")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("MSE")
    ax.set_title("Per-Joint Mean Squared Error")

    for bar, val in zip(bars, values, strict=True):
        ax.text(
            bar.get_width() + 0.0001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=7,
        )

    ax.invert_yaxis()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved per-joint error chart to %s", output_path)
