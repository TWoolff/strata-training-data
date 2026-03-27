#!/usr/bin/env python3
"""Precompute boundary-softened segmentation targets and save to disk.

For each example with a segmentation.png, computes the soft one-hot target
array and saves it as soft_segmentation.npy. The training dataset loader
can then load these directly instead of running scipy per sample.

Usage::

    # All datasets
    python scripts/precompute_soft_targets.py \
        --data-dir ./data_cloud/humanrig \
        --data-dir ./data_cloud/sora_diverse \
        --radius 2 --sigma 1.0

    # Skip already computed
    python scripts/precompute_soft_targets.py \
        --data-dir ./data_cloud/sora_diverse \
        --radius 2 --sigma 1.0 --only-missing

    # With excluded classes (neck=2, hair_back=21)
    python scripts/precompute_soft_targets.py \
        --data-dir ./data_cloud/sora_diverse \
        --radius 2 --sigma 1.0 --exclude-classes 2 21
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter

logger = logging.getLogger(__name__)

NUM_CLASSES = 22


def soften_boundaries(
    mask: np.ndarray,
    radius: int = 2,
    sigma: float = 1.0,
    exclude_classes: set[int] | None = None,
) -> np.ndarray:
    """Create soft one-hot targets with Gaussian-blurred boundaries.

    Args:
        mask: [H, W] int64 region IDs (0=bg, 1-21=body parts, -1=ignore).
        radius: Dilation radius for boundary detection.
        sigma: Gaussian sigma for softening.
        exclude_classes: Class IDs to keep as hard labels even at boundaries.

    Returns:
        [num_classes, H, W] float32 soft target distribution.
    """
    if exclude_classes is None:
        exclude_classes = set()

    h, w = mask.shape

    # Build one-hot encoding
    one_hot = np.zeros((NUM_CLASSES, h, w), dtype=np.float32)
    for c in range(NUM_CLASSES):
        one_hot[c] = (mask == c).astype(np.float32)

    # Gaussian blur each class channel
    soft = np.zeros_like(one_hot)
    for c in range(NUM_CLASSES):
        if one_hot[c].any():
            soft[c] = gaussian_filter(one_hot[c], sigma=sigma)

    # Normalize
    total = soft.sum(axis=0, keepdims=True).clip(min=1e-8)
    soft = soft / total

    # Detect boundaries
    fg_mask = (mask > 0) & (mask < NUM_CLASSES)
    dilated = maximum_filter(mask, size=2 * radius + 1)
    eroded = minimum_filter(mask, size=2 * radius + 1)
    boundary = (dilated != eroded) & fg_mask

    # Exclude specified classes
    for c in exclude_classes:
        boundary = boundary & (mask != c)

    # Revert non-boundary to hard one-hot
    for c in range(NUM_CLASSES):
        soft[c] = np.where(boundary, soft[c], one_hot[c])

    # Re-normalize boundary pixels
    if boundary.any():
        boundary_total = soft[:, boundary].sum(axis=0, keepdims=True).clip(min=1e-8)
        soft[:, boundary] = soft[:, boundary] / boundary_total

    # Ignore index pixels → all zeros
    ignore_mask = (mask < 0) | (mask >= NUM_CLASSES)
    soft[:, ignore_mask] = 0.0

    return soft


def process_directory(
    data_dir: Path,
    radius: int,
    sigma: float,
    exclude_classes: set[int],
    only_missing: bool,
) -> tuple[int, int]:
    """Process all examples in a dataset directory.

    Returns (processed, skipped) counts.
    """
    processed = 0
    skipped = 0

    # Find all segmentation masks
    candidates = sorted(data_dir.iterdir()) if data_dir.is_dir() else []
    examples = []
    for child in candidates:
        if not child.is_dir():
            continue
        seg_path = child / "segmentation.png"
        if seg_path.exists():
            examples.append(child)

    if not examples:
        logger.warning("No examples found in %s", data_dir)
        return 0, 0

    logger.info("Found %d examples in %s", len(examples), data_dir.name)

    for i, ex_dir in enumerate(examples):
        soft_path = ex_dir / "soft_segmentation.npz"

        if only_missing and soft_path.exists():
            skipped += 1
            continue

        seg_path = ex_dir / "segmentation.png"
        mask = np.array(Image.open(seg_path).convert("L"), dtype=np.int64)

        soft = soften_boundaries(mask, radius=radius, sigma=sigma, exclude_classes=exclude_classes)

        # Save as compressed npz (~50-200KB per example vs 11MB uncompressed)
        np.savez_compressed(soft_path, soft=soft.astype(np.float16))

        processed += 1

        if (processed + skipped) % 500 == 0:
            logger.info("  %s: %d/%d done (%d skipped)", data_dir.name, processed + skipped, len(examples), skipped)

    return processed, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute boundary-softened seg targets")
    parser.add_argument("--data-dir", type=Path, action="append", required=True, help="Dataset directory (repeatable)")
    parser.add_argument("--radius", type=int, default=2, help="Boundary detection radius (default: 2)")
    parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian sigma (default: 1.0)")
    parser.add_argument("--exclude-classes", type=int, nargs="*", default=[2, 21], help="Classes to exclude from softening (default: 2=neck, 21=hair_back)")
    parser.add_argument("--only-missing", action="store_true", help="Skip examples that already have soft_segmentation.npy")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    exclude = set(args.exclude_classes) if args.exclude_classes else set()
    logger.info("Radius=%d, sigma=%.1f, exclude_classes=%s, only_missing=%s",
                args.radius, args.sigma, exclude, args.only_missing)

    total_processed = 0
    total_skipped = 0
    t0 = time.time()

    for data_dir in args.data_dir:
        if not data_dir.exists():
            logger.warning("Directory not found: %s", data_dir)
            continue
        p, s = process_directory(data_dir, args.radius, args.sigma, exclude, args.only_missing)
        total_processed += p
        total_skipped += s
        logger.info("  %s: %d processed, %d skipped", data_dir.name, p, s)

    elapsed = time.time() - t0
    rate = total_processed / max(elapsed, 1)
    logger.info("Done: %d processed, %d skipped in %.0fs (%.1f/s)", total_processed, total_skipped, elapsed, rate)


if __name__ == "__main__":
    main()
