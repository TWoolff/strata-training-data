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

    Uses rembg alpha channel for reliable content detection, then finds
    the deepest valleys in column density to split between characters.

    Args:
        img_array: [H, W, 4] RGBA uint8 image (should be rembg-processed).
        num_splits: Expected number of views (default 5).

    Returns:
        List of (col_start, col_end) for each view.
    """
    h, w = img_array.shape[:2]

    # Use alpha channel — ignore bottom 15% where labels are
    crop_h = int(h * 0.85)
    alpha = img_array[:crop_h, :, 3]
    col_density = (alpha > 30).sum(axis=0).astype(float)

    # Smooth heavily to find broad character regions
    kernel_size = max(w // 50, 5)
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(col_density, kernel, mode="same")

    # We need to find (num_splits - 1) split points between characters.
    # Strategy: find the valleys (local minima) in the smoothed density,
    # then pick the (num_splits - 1) deepest ones.

    # Expected character width
    expected_width = w / num_splits
    min_char_width = int(expected_width * 0.4)

    # Find all local minima
    valleys = []
    for col in range(min_char_width, w - min_char_width):
        # Check if this is a local minimum in a window
        window = int(expected_width * 0.15)
        local_slice = smoothed[max(0, col - window):min(w, col + window + 1)]
        if smoothed[col] <= local_slice.min() + 1e-6:
            valleys.append((col, smoothed[col]))

    # Deduplicate: keep only the deepest valley in each window
    if valleys:
        deduped = [valleys[0]]
        for col, val in valleys[1:]:
            if col - deduped[-1][0] < min_char_width:
                # Same valley — keep the deeper one
                if val < deduped[-1][1]:
                    deduped[-1] = (col, val)
            else:
                deduped.append((col, val))
        valleys = deduped

    # Sort by depth (ascending density = deepest valley first)
    valleys.sort(key=lambda x: x[1])

    # Pick the best (num_splits - 1) split points
    split_cols = sorted([v[0] for v in valleys[:num_splits - 1]])

    if len(split_cols) == num_splits - 1:
        # Build regions from split points
        boundaries = [0] + split_cols + [w]
        regions = [(boundaries[i], boundaries[i + 1]) for i in range(num_splits)]
    else:
        # Fallback: equal-width split
        logger.warning(
            "Found %d valleys, need %d — falling back to equal-width split",
            len(split_cols), num_splits - 1,
        )
        view_width = w // num_splits
        regions = [(i * view_width, (i + 1) * view_width) for i in range(num_splits)]

    return regions


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

    # Find the bottom of character content (before label text).
    # Labels are typically a thin band of content at the very bottom
    # with a gap above them. Look for a gap in content density from
    # the bottom up.
    last_content = content_rows[-1]
    # Search from 95% down for a gap (low density row)
    search_start = int(h * 0.75)
    gap_threshold = row_density.max() * 0.15
    bottom = last_content + 5  # default: use all content

    # Walk from bottom up — find where character content ends and label begins
    for row in range(last_content, search_start, -1):
        if row_density[row] < gap_threshold:
            # Found a gap — this is between feet and labels
            bottom = row
            break

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

    # Run rembg on the FULL sheet first for clean alpha-based splitting
    try:
        from rembg import remove
        logger.info("  Running rembg on full sheet for split detection...")
        img_rgba = remove(img.convert("RGB"))
        img_array = np.array(img_rgba)
    except ImportError:
        logger.warning("  rembg not available — using brightness-based detection")
        img_array = np.array(img.convert("RGBA") if img.mode == "RGBA" else img)
        if img_array.shape[2] == 3:
            gray = np.mean(img_array[:, :, :3], axis=2)
            alpha = np.where(gray < 250, 255, 0).astype(np.uint8)
            img_array = np.concatenate([img_array, alpha[:, :, np.newaxis]], axis=2)

    # Find content area (rows) using rembg'd alpha
    top, bottom = find_content_rows(img_array)
    logger.info("  Content rows: %d to %d (of %d)", top, bottom, img_array.shape[0])

    # Find split columns using rembg'd alpha (clean gaps)
    columns = find_split_columns(img_array, num_views)
    logger.info("  Found %d view regions: %s", len(columns), columns)

    if len(columns) != num_views:
        logger.error("Expected %d views, found %d — skipping", num_views, len(columns))
        return []

    # Load the ORIGINAL image for cropping (rembg on full sheet may be imperfect)
    orig_img = Image.open(img_path).convert("RGBA") if Image.open(img_path).mode == "RGBA" else Image.open(img_path).convert("RGB")
    orig_array = np.array(orig_img)

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
        # Crop from ORIGINAL image (not rembg'd sheet)
        crop = orig_array[top:bottom, col_start:col_end]
        crop_img = Image.fromarray(crop)

        # Run rembg on each individual crop (more accurate than full sheet)
        if has_rembg:
            crop_img = remove(crop_img.convert("RGB"))

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
