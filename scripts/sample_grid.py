#!/usr/bin/env python3
"""Quick visual sample of a preprocessed dataset — random N examples in one grid PNG.

Picks N random example dirs (containing ``image.png`` + ``segmentation.png``),
composites each as image-|-overlay-|-mask and tiles them into a single output
PNG for fast eyeballing. Useful for auditing label quality without clicking
through an interactive tool.

Usage::

    python scripts/sample_grid.py \\
        --data-dir /Volumes/TAMWoolff/data/preprocessed/sora_diverse \\
        --n 20 \\
        --output /tmp/sora_diverse_sample.png

    # Open the PNG with your image viewer of choice
    open /tmp/sora_diverse_sample.png    # macOS
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# 22-class anatomy colors (same as pipeline/config.py REGION_COLORS)
REGION_COLORS = [
    (  0,   0,   0),  # 0  bg
    (255,  99,  99),  # 1  head
    (255, 180,  99),  # 2  neck
    (255, 255,  99),  # 3  chest
    (180, 255,  99),  # 4  spine
    ( 99, 255,  99),  # 5  hips
    ( 99, 255, 180),  # 6  shoulder_l
    ( 99, 255, 255),  # 7  upper_arm_l
    ( 99, 180, 255),  # 8  forearm_l
    ( 99,  99, 255),  # 9  hand_l
    (180,  99, 255),  # 10 shoulder_r
    (255,  99, 255),  # 11 upper_arm_r
    (255,  99, 180),  # 12 forearm_r
    (180, 180, 180),  # 13 hand_r
    (255, 128,   0),  # 14 upper_leg_l
    (  0, 255, 128),  # 15 lower_leg_l
    (128,   0, 255),  # 16 foot_l
    (255,   0, 128),  # 17 upper_leg_r
    (  0, 128, 255),  # 18 lower_leg_r
    (128, 255,   0),  # 19 foot_r
    (200, 200, 200),  # 20 accessory
    (255, 200, 150),  # 21 hair_back
]

CELL = 256  # per-panel size (image|overlay|mask)
GAP = 4


def colorize(mask: np.ndarray) -> np.ndarray:
    """Map (H, W) class IDs to (H, W, 3) uint8 RGB."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, color in enumerate(REGION_COLORS):
        out[mask == cls] = color
    return out


def make_row(example_dir: Path) -> Image.Image | None:
    img_p = example_dir / "image.png"
    seg_p = example_dir / "segmentation.png"
    if not (img_p.exists() and seg_p.exists()):
        return None
    img = Image.open(img_p).convert("RGBA").resize((CELL, CELL), Image.LANCZOS)
    seg = np.array(Image.open(seg_p).resize((CELL, CELL), Image.NEAREST))
    colored = colorize(seg)

    # Composite image + overlay
    img_rgb = Image.new("RGB", (CELL, CELL), (40, 40, 40))
    img_rgb.paste(img, (0, 0), img.split()[-1] if img.mode == "RGBA" else None)

    overlay = Image.blend(img_rgb, Image.fromarray(colored), alpha=0.55)
    mask_img = Image.fromarray(colored)

    row = Image.new("RGB", (CELL * 3 + GAP * 2, CELL), (25, 25, 25))
    row.paste(img_rgb, (0, 0))
    row.paste(overlay, (CELL + GAP, 0))
    row.paste(mask_img, (CELL * 2 + GAP * 2, 0))
    return row


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Preprocessed dataset dir (contains example subdirs).")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--output", type=Path, default=Path("/tmp/sample_grid.png"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.data_dir.is_dir():
        print(f"not a dir: {args.data_dir}", file=sys.stderr)
        return 1

    examples = sorted(
        d for d in args.data_dir.iterdir()
        if d.is_dir() and (d / "image.png").exists() and (d / "segmentation.png").exists()
    )
    if not examples:
        print(f"no examples with image.png+segmentation.png in {args.data_dir}", file=sys.stderr)
        return 1

    random.seed(args.seed)
    picks = random.sample(examples, min(args.n, len(examples)))

    rows = []
    for ex in picks:
        r = make_row(ex)
        if r is not None:
            rows.append((ex.name, r))

    if not rows:
        print("no valid examples found", file=sys.stderr)
        return 1

    # Tile: one row per example, label on left
    from PIL import ImageDraw, ImageFont
    label_w = 220
    row_w = CELL * 3 + GAP * 2
    grid_w = label_w + row_w
    grid_h = CELL * len(rows) + GAP * (len(rows) - 1)
    grid = Image.new("RGB", (grid_w, grid_h), (15, 15, 15))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        font = ImageFont.load_default()

    # Header
    header = Image.new("RGB", (row_w, 20), (50, 50, 50))
    hdraw = ImageDraw.Draw(header)
    hdraw.text((4, 4), "image   |   overlay   |   mask",
               fill=(230, 230, 230), font=font)
    # (skipping header in grid to keep simple — put titles inside each cell if needed)

    for i, (name, r) in enumerate(rows):
        y = i * (CELL + GAP)
        draw.text((4, y + 4), name[:28], fill=(230, 230, 230), font=font)
        if len(name) > 28:
            draw.text((4, y + 20), name[28:56], fill=(180, 180, 180), font=font)
        grid.paste(r, (label_w, y))

    grid.save(args.output)
    print(f"saved {args.output}  —  {len(rows)} samples from {args.data_dir.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
