#!/usr/bin/env python3
"""Split a character turnaround sheet into individual view images.

Detects 5 characters in a row by finding vertical gaps (low-content columns),
crops each view, removes background, and saves as individual PNGs.

Usage::

    # Split a single turnaround sheet
    python scripts/split_turnaround.py \
        --input /path/to/turnaround_sheet.png \
        --output-dir /path/to/output/ \
        --name bear_chef

    # Process all sheets in a directory
    python scripts/split_turnaround.py \
        --input-dir /path/to/sheets/ \
        --output-dir /path/to/output/

Output files:
    {name}_front.png
    {name}_threequarter.png
    {name}_side.png
    {name}_back_threequarter.png
    {name}_back.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

VIEW_NAMES = ["front", "threequarter", "side", "back_threequarter", "back"]


def find_split_columns(img_array: np.ndarray, num_splits: int = 5) -> list[tuple[int, int]]:
    """Find column ranges for each character view by detecting vertical gaps.

    Args:
        img_array: [H, W, 4] RGBA or [H, W, 3] RGB uint8 image.
        num_splits: Expected number of views (default 5).

    Returns:
        List of (col_start, col_end) for each view.
    """
    h, w = img_array.shape[:2]

    # Compute content density per column (ignore bottom 15% where labels are)
    crop_h = int(h * 0.85)
    if img_array.shape[2] == 4:
        # Use alpha channel for content detection
        col_density = (img_array[:crop_h, :, 3] > 30).sum(axis=0).astype(float)
    else:
        # Use brightness for RGB images (white bg = high brightness)
        gray = np.mean(img_array[:crop_h, :, :3], axis=2)
        col_density = (gray < 240).sum(axis=0).astype(float)

    # Smooth the density to avoid noise
    kernel_size = max(w // 100, 3)
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(col_density, kernel, mode="same")

    # Find content threshold
    threshold = smoothed.max() * 0.05

    # Find runs of content (above threshold)
    is_content = smoothed > threshold
    regions = []
    in_region = False
    start = 0

    for col in range(w):
        if is_content[col] and not in_region:
            start = col
            in_region = True
        elif not is_content[col] and in_region:
            regions.append((start, col))
            in_region = False
    if in_region:
        regions.append((start, w))

    # Merge regions that are very close (< 2% of image width)
    merge_gap = int(w * 0.02)
    merged = [regions[0]]
    for start, end in regions[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end < merge_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    # If we have more regions than expected, keep the N largest
    if len(merged) > num_splits:
        merged.sort(key=lambda r: r[1] - r[0], reverse=True)
        merged = sorted(merged[:num_splits], key=lambda r: r[0])

    # If we have fewer, try equal-width splitting as fallback
    if len(merged) < num_splits:
        logger.warning(
            "Found %d regions, expected %d — falling back to equal-width split",
            len(merged), num_splits,
        )
        view_width = w // num_splits
        merged = [(i * view_width, (i + 1) * view_width) for i in range(num_splits)]

    # Add padding to each region (5% on each side)
    pad = int(w * 0.01)
    padded = [(max(0, s - pad), min(w, e + pad)) for s, e in merged]

    return padded


def find_content_rows(img_array: np.ndarray) -> tuple[int, int]:
    """Find the top and bottom rows containing content (excluding labels)."""
    h, w = img_array.shape[:2]

    if img_array.shape[2] == 4:
        row_density = (img_array[:, :, 3] > 30).sum(axis=1).astype(float)
    else:
        gray = np.mean(img_array[:, :, :3], axis=2)
        row_density = (gray < 240).sum(axis=1).astype(float)

    threshold = row_density.max() * 0.05
    content_rows = np.where(row_density > threshold)[0]

    if len(content_rows) == 0:
        return 0, h

    # Crop to content area, leaving a small margin
    top = max(0, content_rows[0] - 5)
    # Exclude bottom 12% to remove labels
    bottom = min(h, int(h * 0.88))

    return top, bottom


def split_turnaround(
    img_path: Path,
    output_dir: Path,
    name: str,
    num_views: int = 5,
) -> list[Path]:
    """Split a turnaround sheet into individual view images.

    Args:
        img_path: Path to the turnaround sheet image.
        output_dir: Directory to save individual views.
        name: Character name prefix for output files.
        num_views: Expected number of views (default 5).

    Returns:
        List of saved file paths.
    """
    img = Image.open(img_path)
    img_array = np.array(img)

    # Ensure RGBA
    if img_array.shape[2] == 3:
        # Add alpha channel (white = transparent for white-bg images)
        gray = np.mean(img_array[:, :, :3], axis=2)
        alpha = np.where(gray < 250, 255, 0).astype(np.uint8)
        img_array = np.concatenate([img_array, alpha[:, :, np.newaxis]], axis=2)

    # Find content area (rows)
    top, bottom = find_content_rows(img_array)
    logger.info("  Content rows: %d to %d (of %d)", top, bottom, img_array.shape[0])

    # Find split columns
    columns = find_split_columns(img_array, num_views)
    logger.info("  Found %d view regions: %s", len(columns), columns)

    if len(columns) != num_views:
        logger.error("Expected %d views, found %d — skipping", num_views, len(columns))
        return []

    view_names = VIEW_NAMES[:num_views]
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    try:
        from rembg import remove
        has_rembg = True
    except ImportError:
        has_rembg = False
        logger.warning("rembg not available — skipping background removal")

    for i, ((col_start, col_end), view_name) in enumerate(zip(columns, view_names)):
        # Crop the view
        crop = img_array[top:bottom, col_start:col_end]
        crop_img = Image.fromarray(crop, "RGBA")

        # Remove background
        if has_rembg:
            crop_rgb = crop_img.convert("RGB")
            crop_img = remove(crop_rgb)

        # Resize to 512x512 with padding
        w, h = crop_img.size
        scale = min(512 / w, 512 / h) * 0.9
        new_w, new_h = int(w * scale), int(h * scale)
        crop_img = crop_img.resize((new_w, new_h), Image.LANCZOS)

        canvas = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        x = (512 - new_w) // 2
        y = (512 - new_h) // 2
        canvas.paste(crop_img, (x, y))

        out_path = output_dir / f"{name}_{view_name}.png"
        canvas.save(out_path)
        saved.append(out_path)
        logger.info("  Saved %s (%dx%d crop)", out_path.name, col_end - col_start, bottom - top)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Split turnaround sheets into individual views")
    parser.add_argument("--input", type=Path, help="Single turnaround sheet image")
    parser.add_argument("--input-dir", type=Path, help="Directory of turnaround sheets")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--name", type=str, help="Character name (for single input)")
    parser.add_argument("--num-views", type=int, default=5, help="Expected views per sheet")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.input:
        name = args.name or args.input.stem
        logger.info("Processing: %s → %s", args.input.name, name)
        saved = split_turnaround(args.input, args.output_dir, name, args.num_views)
        logger.info("Saved %d views", len(saved))

    elif args.input_dir:
        sheets = sorted(
            p for p in args.input_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
            and not p.name.startswith("._")
        )
        logger.info("Found %d turnaround sheets in %s", len(sheets), args.input_dir)

        total = 0
        for sheet in sheets:
            name = sheet.stem
            logger.info("Processing: %s", sheet.name)
            saved = split_turnaround(sheet, args.output_dir, name, args.num_views)
            total += len(saved)

        logger.info("Done: %d total views from %d sheets", total, len(sheets))
    else:
        parser.error("Provide --input or --input-dir")


if __name__ == "__main__":
    main()
