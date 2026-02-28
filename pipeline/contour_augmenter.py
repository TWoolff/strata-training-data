"""Contour style augmentation for contour-removal training data.

Takes a ``(without_contours, contour_mask)`` pair and produces 5 visually
distinct contour style variants by dilating, coloring, and compositing the
contour lines onto the clean image.

Pure Python (PIL / OpenCV / NumPy) — no Blender dependency.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import (
    CONTOUR_REGION_COLORS,
    CONTOUR_STYLES,
    CONTOUR_WOBBLE_RANGE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dilate_mask(mask: np.ndarray, target_width: int) -> np.ndarray:
    """Dilate a binary mask to approximate the desired line width.

    Args:
        mask: 2-D uint8 array (0 or 255).
        target_width: Desired line width in pixels.

    Returns:
        Dilated mask (same shape, 0/255 values).
    """
    if target_width <= 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (target_width, target_width))
    return cv2.dilate(mask, kernel, iterations=1)


def _apply_wobble(mask: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Displace contour pixels for a hand-drawn wobble effect.

    Uses random per-pixel displacement via ``cv2.remap`` (same technique
    as ``style_augmentor.apply_sketch``).

    Args:
        mask: 2-D uint8 array (0 or 255).
        seed: Random seed for deterministic wobble.

    Returns:
        Wobbled mask (same shape).
    """
    if CONTOUR_WOBBLE_RANGE <= 0:
        return mask

    rng = np.random.default_rng(seed)
    h, w = mask.shape
    dx = rng.integers(-CONTOUR_WOBBLE_RANGE, CONTOUR_WOBBLE_RANGE + 1, size=(h, w)).astype(
        np.float32
    )
    dy = rng.integers(-CONTOUR_WOBBLE_RANGE, CONTOUR_WOBBLE_RANGE + 1, size=(h, w)).astype(
        np.float32
    )
    map_x = (np.arange(w)[np.newaxis, :] + dx).astype(np.float32)
    map_y = (np.arange(h)[:, np.newaxis] + dy).astype(np.float32)
    return cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderValue=0)


def _composite_contours(
    base_image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
    opacity: float,
) -> Image.Image:
    """Composite solid-color contour lines onto a base image.

    Args:
        base_image: RGBA base image (without contours).
        mask: 2-D uint8 contour mask (0 or 255).
        color: RGB color for the contour lines.
        opacity: Blend opacity (0.0–1.0).

    Returns:
        New RGBA image with contours composited.
    """
    result = np.array(base_image.convert("RGBA"), dtype=np.float32)
    contour_pixels = mask > 0

    for c in range(3):
        result[:, :, c] = np.where(
            contour_pixels,
            result[:, :, c] * (1.0 - opacity) + color[c] * opacity,
            result[:, :, c],
        )

    return Image.fromarray(result.astype(np.uint8), mode="RGBA")


def _composite_per_region(
    base_image: Image.Image,
    mask: np.ndarray,
    seg_mask: np.ndarray,
    opacity: float,
) -> Image.Image:
    """Composite contour lines colored per body region.

    Each contour pixel is colored according to the underlying segmentation
    region ID.

    Args:
        base_image: RGBA base image (without contours).
        mask: 2-D uint8 contour mask (0 or 255).
        seg_mask: 2-D uint8 segmentation mask (pixel value = region ID).
        opacity: Blend opacity (0.0–1.0).

    Returns:
        New RGBA image with per-region-colored contours composited.
    """
    result = np.array(base_image.convert("RGBA"), dtype=np.float32)
    contour_pixels = mask > 0

    # Build per-pixel color lookup from segmentation mask.
    h, w = seg_mask.shape
    color_map = np.zeros((h, w, 3), dtype=np.float32)
    for region_id, rgb in CONTOUR_REGION_COLORS.items():
        region_mask = seg_mask == region_id
        for c in range(3):
            color_map[:, :, c] = np.where(region_mask, rgb[c], color_map[:, :, c])

    for c in range(3):
        result[:, :, c] = np.where(
            contour_pixels,
            result[:, :, c] * (1.0 - opacity) + color_map[:, :, c] * opacity,
            result[:, :, c],
        )

    return Image.fromarray(result.astype(np.uint8), mode="RGBA")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def augment_contour_style(
    without_contours: Image.Image,
    contour_mask: np.ndarray,
    style: dict[str, object],
    *,
    seg_mask: np.ndarray | None = None,
    seed: int = 0,
) -> Image.Image:
    """Generate a single contour style variant.

    Args:
        without_contours: Clean RGBA image (no contour lines).
        contour_mask: 2-D uint8 binary mask (0/255) of contour pixels.
        style: Style definition dict from ``config.CONTOUR_STYLES``.
        seg_mask: Segmentation mask (required for ``"per_region"`` color mode).
        seed: Random seed for wobble.

    Returns:
        RGBA image with the specified contour style composited.
    """
    line_width: int = int(style["line_width"])  # type: ignore[arg-type]
    color = style["color"]
    opacity: float = float(style["opacity"])  # type: ignore[arg-type]
    wobble: bool = bool(style["wobble"])

    # Dilate mask to target width.
    styled_mask = _dilate_mask(contour_mask, line_width)

    # Apply wobble if requested.
    if wobble:
        styled_mask = _apply_wobble(styled_mask, seed=seed)

    # Composite with appropriate coloring.
    if color == "per_region":
        if seg_mask is None:
            logger.warning("per_region style requires seg_mask; falling back to black")
            return _composite_contours(without_contours, styled_mask, (0, 0, 0), opacity)
        return _composite_per_region(without_contours, styled_mask, seg_mask, opacity)

    return _composite_contours(without_contours, styled_mask, color, opacity)  # type: ignore[arg-type]


def augment_all_styles(
    without_contours: Image.Image,
    contour_mask: np.ndarray,
    output_dir: Path,
    file_prefix: str,
    *,
    seg_mask: np.ndarray | None = None,
    seed: int = 0,
) -> list[Path]:
    """Generate all contour style variants and save to disk.

    Args:
        without_contours: Clean RGBA image (no contour lines).
        contour_mask: 2-D uint8 binary mask (0/255) of contour pixels.
        output_dir: Root output directory (writes to ``contours/`` sub-dir).
        file_prefix: Filename prefix, e.g. ``"char01_pose_00_front"``.
        seg_mask: Segmentation mask (needed for per-region style).
        seed: Base random seed.

    Returns:
        List of paths to saved variant images.
    """
    contour_dir = output_dir / "contours"
    contour_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for idx, style in enumerate(CONTOUR_STYLES):
        style_name: str = str(style["name"])
        variant = augment_contour_style(
            without_contours,
            contour_mask,
            style,
            seg_mask=seg_mask,
            seed=seed + idx,
        )
        out_path = contour_dir / f"{file_prefix}_contour_{style_name}.png"
        variant.save(out_path, format="PNG", compress_level=9)
        paths.append(out_path)
        logger.debug("Saved contour variant: %s", out_path.name)

    logger.info(
        "Contour augmentation: %d styles for %s",
        len(CONTOUR_STYLES),
        file_prefix,
    )
    return paths
