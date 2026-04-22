"""Filter segmentation masks by quality criteria.

Scans dataset directories for ``segmentation.png`` files and checks each mask
against configurable quality thresholds.  Outputs a ``quality_filter.json``
with passed/rejected lists and per-dataset stats, plus an optional visual
grid of rejected samples.

Supports two directory layouts:
  - Per-example:  ``{dataset_dir}/{example_id}/segmentation.png``
  - Nested view:  ``{dataset_dir}/{example_id}/{view}/segmentation.png``

Masks are 8-bit single-channel PNGs where pixel value = region ID (0-21).

Usage::

    python scripts/filter_seg_quality.py \\
      --data-dirs ./data_cloud/humanrig ./data_cloud/meshy_cc0 \\
      --output-dir ./output/quality_filter/ \\
      --min-regions 4 \\
      --max-single-region 0.70 \\
      --min-foreground 0.05
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Region names for readable rejection reasons.
REGION_NAMES = {
    0: "background", 1: "head", 2: "neck", 3: "chest", 4: "spine",
    5: "hips", 6: "shoulder_l", 7: "upper_arm_l", 8: "forearm_l",
    9: "hand_l", 10: "shoulder_r", 11: "upper_arm_r", 12: "forearm_r",
    13: "hand_r", 14: "upper_leg_l", 15: "lower_leg_l", 16: "foot_l",
    17: "upper_leg_r", 18: "lower_leg_r", 19: "foot_r", 20: "accessory",
    21: "hair_back",
}

TORSO_REGIONS = {3, 4, 5}  # chest, spine, hips
HEAD_REGION = 1

# Paired L/R region IDs for asymmetry check
LR_PAIRS = [
    (6, 10),   # shoulder_l/r
    (7, 11),   # upper_arm_l/r
    (8, 12),   # forearm_l/r
    (9, 13),   # hand_l/r
    (14, 17),  # upper_leg_l/r
    (15, 18),  # lower_leg_l/r
    (16, 19),  # foot_l/r
]


# ---------------------------------------------------------------------------
# Mask discovery
# ---------------------------------------------------------------------------

def discover_masks(data_dir: Path) -> list[tuple[str, Path]]:
    """Find all segmentation.png files under *data_dir*.

    Returns a list of ``(example_key, mask_path)`` tuples where
    *example_key* encodes the example (and optional view) identity.
    """
    results: list[tuple[str, Path]] = []

    if not data_dir.is_dir():
        logger.warning("Directory does not exist: %s", data_dir)
        return results

    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue

        # Layout 1: {example_id}/segmentation.png
        seg = child / "segmentation.png"
        if seg.is_file():
            results.append((child.name, seg))
            continue

        # Layout 2: {example_id}/{view}/segmentation.png
        for view_dir in sorted(child.iterdir()):
            if not view_dir.is_dir():
                continue
            seg = view_dir / "segmentation.png"
            if seg.is_file():
                key = f"{child.name}/{view_dir.name}"
                results.append((key, seg))

    return results


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

def check_mask(
    mask_path: Path,
    *,
    min_regions: int = 4,
    max_single_region: float = 0.70,
    min_foreground: float = 0.05,
    skip_anatomy: bool = False,
    drop_head_below_torso: bool = False,
    max_lr_asymmetry: float | None = None,
) -> list[str]:
    """Check a single segmentation mask against quality criteria.

    Returns a list of rejection reasons (empty list = passed).

    Args:
        skip_anatomy: If True, skip missing_head/missing_torso checks.
            Use for GT data where masks are known-correct but poses may
            legitimately hide the head or torso from certain angles.
        drop_head_below_torso: If True, reject examples where the head's
            centroid sits below the torso's vertical median (an anatomically
            impossible label, typically indicates a flipped / bad pseudo-label).
            Audit on flux_diverse_clean (Apr 22) found 15% of examples fail this.
        max_lr_asymmetry: If set, reject examples where any paired L/R class
            has |area_l - area_r| / max(area_l, area_r) > this value. Use 0.85
            or higher — 92% of hand labels pass 0.70 because many characters
            are drawn in 3/4 view with one side mostly hidden.
    """
    reasons: list[str] = []

    try:
        img = Image.open(mask_path)
        arr = np.asarray(img, dtype=np.uint8)
    except Exception as exc:
        return [f"read_error({exc})"]

    # Flatten to 2D if needed (some masks may be saved as RGB).
    if arr.ndim == 3:
        arr = arr[:, :, 0]

    total_pixels = arr.size
    fg_mask = arr > 0
    fg_pixels = int(fg_mask.sum())

    # --- Min foreground ---
    fg_ratio = fg_pixels / total_pixels if total_pixels > 0 else 0.0
    if fg_ratio < min_foreground:
        reasons.append(f"low_foreground({fg_ratio:.1%})")

    if fg_pixels == 0:
        reasons.append("no_foreground")
        return reasons

    # --- Unique non-background regions ---
    fg_values = arr[fg_mask]
    unique_regions = np.unique(fg_values)
    n_regions = len(unique_regions)

    if n_regions < min_regions:
        reasons.append(f"too_few_regions({n_regions})")

    # --- Required regions (skip for GT posed data) ---
    unique_set = set(unique_regions.tolist())
    if not skip_anatomy:
        if HEAD_REGION not in unique_set:
            reasons.append("missing_head")
        if not unique_set & TORSO_REGIONS:
            reasons.append("missing_torso")

    # --- Max single region dominance ---
    counts = np.bincount(fg_values, minlength=22)
    for rid in range(1, 22):
        if counts[rid] == 0:
            continue
        region_ratio = counts[rid] / fg_pixels
        if region_ratio > max_single_region:
            name = REGION_NAMES.get(rid, str(rid))
            reasons.append(f"single_region_dominates({name}={region_ratio:.0%})")

    # --- Head-below-torso (anatomically impossible, strong signal of bad label) ---
    if drop_head_below_torso and HEAD_REGION in unique_set and (unique_set & TORSO_REGIONS):
        head_ys, _ = np.where(arr == HEAD_REGION)
        torso_ys = np.concatenate([
            np.where(arr == c)[0] for c in TORSO_REGIONS if c in unique_set
        ])
        if len(head_ys) > 0 and len(torso_ys) > 0:
            head_cy = float(head_ys.mean())
            torso_median_y = float(np.median(torso_ys))
            if head_cy >= torso_median_y:
                reasons.append(f"head_below_torso(head_y={head_cy:.0f},torso_med={torso_median_y:.0f})")

    # --- L/R asymmetry (one side of the body vastly larger than the other) ---
    # Only fires when BOTH sides are labeled — one-sided characters (3/4 view,
    # occluded limbs) are a separate "missing class" issue, not asymmetry.
    if max_lr_asymmetry is not None:
        for left_id, right_id in LR_PAIRS:
            al = int(counts[left_id])
            ar = int(counts[right_id])
            if al == 0 or ar == 0:
                continue  # one side missing — not an asymmetry issue
            m = max(al, ar)
            asym = abs(al - ar) / m
            if asym > max_lr_asymmetry:
                ln = REGION_NAMES.get(left_id, str(left_id))
                rn = REGION_NAMES.get(right_id, str(right_id))
                reasons.append(f"lr_asymmetry({ln}/{rn}={asym:.0%})")
                break  # One pair is enough to reject

    return reasons


# ---------------------------------------------------------------------------
# Rejected samples grid
# ---------------------------------------------------------------------------

def _load_thumbnail(path: Path, size: int = 128) -> Image.Image:
    """Load and resize an image to a square thumbnail."""
    try:
        img = Image.open(path).convert("RGBA")
        img.thumbnail((size, size), Image.NEAREST)
    except Exception:
        img = Image.new("RGBA", (size, size), (128, 128, 128, 255))
    # Paste onto white background.
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    return bg


def _colorize_mask(mask_path: Path, size: int = 128) -> Image.Image:
    """Load a segmentation mask and colorize it for visualization."""
    palette = [
        (0, 0, 0),       # 0 background
        (255, 0, 0),     # 1 head
        (255, 128, 0),   # 2 neck
        (0, 255, 0),     # 3 chest
        (0, 200, 0),     # 4 spine
        (0, 150, 0),     # 5 hips
        (0, 0, 255),     # 6 shoulder_l
        (0, 100, 255),   # 7 upper_arm_l
        (0, 200, 255),   # 8 forearm_l
        (100, 255, 255), # 9 hand_l
        (128, 0, 255),   # 10 shoulder_r
        (180, 0, 255),   # 11 upper_arm_r
        (220, 0, 255),   # 12 forearm_r
        (255, 0, 255),   # 13 hand_r
        (255, 255, 0),   # 14 upper_leg_l
        (200, 200, 0),   # 15 lower_leg_l
        (150, 150, 0),   # 16 foot_l
        (255, 200, 0),   # 17 upper_leg_r
        (200, 150, 0),   # 18 lower_leg_r
        (150, 100, 0),   # 19 foot_r
        (128, 128, 128), # 20 accessory
        (180, 100, 60),  # 21 hair_back
    ]
    try:
        arr = np.asarray(Image.open(mask_path), dtype=np.uint8)
    except Exception:
        return Image.new("RGB", (size, size), (64, 64, 64))

    if arr.ndim == 3:
        arr = arr[:, :, 0]

    color = np.zeros((*arr.shape, 3), dtype=np.uint8)
    for rid, c in enumerate(palette):
        color[arr == rid] = c

    img = Image.fromarray(color)
    img.thumbnail((size, size), Image.NEAREST)
    return img


def generate_rejected_grid(
    rejected_items: list[tuple[str, Path, list[str]]],
    output_path: Path,
    *,
    max_samples: int = 50,
    thumb_size: int = 128,
) -> None:
    """Generate a grid image showing rejected masks with their source images."""
    items = rejected_items[:max_samples]
    if not items:
        logger.info("No rejected samples — skipping grid generation.")
        return

    cols = 5  # pairs per row (image + mask = 2 cells per pair, but we group)
    cell_w = thumb_size * 2 + 4  # image + mask + gap
    cell_h = thumb_size + 20     # thumbnail + label
    rows = (len(items) + cols - 1) // cols
    grid_w = cols * cell_w + 4
    grid_h = rows * cell_h + 4

    canvas = Image.new("RGB", (grid_w, grid_h), (240, 240, 240))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for idx, (key, mask_path, reasons) in enumerate(items):
        col = idx % cols
        row = idx // cols
        x = col * cell_w + 2
        y = row * cell_h + 2

        # Try to find source image alongside mask.
        image_path = mask_path.parent / "image.png"
        if image_path.is_file():
            thumb = _load_thumbnail(image_path, thumb_size)
            canvas.paste(thumb, (x, y))

        mask_vis = _colorize_mask(mask_path, thumb_size)
        canvas.paste(mask_vis, (x + thumb_size + 4, y))

        # Draw label.
        label = f"{key[:30]}: {', '.join(reasons)}"
        label = label[:60]
        draw.text((x, y + thumb_size + 2), label, fill=(0, 0, 0), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    logger.info("Saved rejected grid (%d samples) to %s", len(items), output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Filter segmentation masks by quality criteria.",
    )
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="One or more dataset directories to scan.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/quality_filter"),
        help="Directory for output files (default: output/quality_filter/).",
    )
    parser.add_argument(
        "--min-regions",
        type=int,
        default=4,
        help="Minimum number of unique non-background region IDs (default: 4).",
    )
    parser.add_argument(
        "--max-single-region",
        type=float,
        default=0.70,
        help="Max fraction of foreground a single region may cover (default: 0.70).",
    )
    parser.add_argument(
        "--min-foreground",
        type=float,
        default=0.05,
        help="Min fraction of image that must be non-background (default: 0.05).",
    )
    parser.add_argument(
        "--skip-anatomy",
        action="store_true",
        help="Skip missing_head/missing_torso checks (for GT posed data).",
    )
    parser.add_argument(
        "--drop-head-below-torso",
        action="store_true",
        help="Reject examples where the head centroid sits below the torso median "
             "y-position (anatomically impossible; ~15%% of flux_diverse_clean fails this).",
    )
    parser.add_argument(
        "--max-lr-asymmetry",
        type=float,
        default=None,
        help="Reject examples where any paired L/R class has "
             "|area_l - area_r| / max(area_l, area_r) > this value. Suggested: 0.90 "
             "(many illustrated characters are drawn in 3/4 view so 0.70 is too tight).",
    )
    parser.add_argument(
        "--max-rejected-grid",
        type=int,
        default=50,
        help="Max rejected samples to show in the grid image (default: 50).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    passed: list[str] = []
    rejected: dict[str, list[str]] = {}
    rejected_for_grid: list[tuple[str, Path, list[str]]] = []
    per_dataset: dict[str, dict[str, int]] = {}

    t0 = time.monotonic()

    for data_dir in args.data_dirs:
        dataset_name = data_dir.name
        masks = discover_masks(data_dir)
        logger.info("Found %d masks in %s", len(masks), data_dir)

        ds_passed = 0
        ds_rejected = 0

        for key, mask_path in masks:
            full_key = f"{dataset_name}/{key}"
            reasons = check_mask(
                mask_path,
                min_regions=args.min_regions,
                max_single_region=args.max_single_region,
                min_foreground=args.min_foreground,
                skip_anatomy=args.skip_anatomy,
                drop_head_below_torso=args.drop_head_below_torso,
                max_lr_asymmetry=args.max_lr_asymmetry,
            )
            if reasons:
                rejected[full_key] = reasons
                ds_rejected += 1
                if len(rejected_for_grid) < args.max_rejected_grid:
                    rejected_for_grid.append((full_key, mask_path, reasons))
            else:
                passed.append(full_key)
                ds_passed += 1

        per_dataset[dataset_name] = {
            "total": ds_passed + ds_rejected,
            "passed": ds_passed,
            "rejected": ds_rejected,
        }

    elapsed = time.monotonic() - t0
    total = len(passed) + len(rejected)
    reject_rate = len(rejected) / total if total > 0 else 0.0

    stats = {
        "total": total,
        "passed": len(passed),
        "rejected": len(rejected),
        "reject_rate": round(reject_rate, 4),
        "per_dataset": per_dataset,
    }

    result = {
        "passed": passed,
        "rejected": rejected,
        "stats": stats,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "quality_filter.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Wrote %s", json_path)

    # Print summary.
    logger.info(
        "Done in %.1fs — %d total, %d passed, %d rejected (%.1f%%)",
        elapsed, total, len(passed), len(rejected), reject_rate * 100,
    )
    for ds_name, ds_stats in per_dataset.items():
        ds_rate = ds_stats["rejected"] / ds_stats["total"] if ds_stats["total"] > 0 else 0.0
        logger.info(
            "  %-20s  total=%d  passed=%d  rejected=%d (%.1f%%)",
            ds_name, ds_stats["total"], ds_stats["passed"], ds_stats["rejected"],
            ds_rate * 100,
        )

    # Generate rejected samples grid.
    if rejected_for_grid:
        grid_path = args.output_dir / "rejected_samples.png"
        generate_rejected_grid(
            rejected_for_grid,
            grid_path,
            max_samples=args.max_rejected_grid,
        )


if __name__ == "__main__":
    main()
