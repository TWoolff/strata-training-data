#!/usr/bin/env python3
"""Score each example in a preprocessed dataset for label quality, surface the worst.

For each example dir (containing ``image.png`` + ``segmentation.png`` [+ optional
``confidence.png``]), compute a handful of lightweight sanity-check metrics:

- ``mean_conf``          mean pseudo-label confidence over foreground pixels (if confidence.png present)
- ``largest_frac``       fraction of foreground pixels held by the single largest non-bg class
- ``n_classes``          number of non-bg classes present (≥4 is the filter default)
- ``disconn_count``      total number of disconnected components across all body-part classes
                         (one blob per class is normal; > n_classes + 2 is suspicious)
- ``head_above_torso``   1 if head centroid is above max(chest/spine/hips) centroid, else 0
- ``lr_asymmetry``       max |area_l − area_r| / max(area_l, area_r) over paired L/R classes

Composite ``badness`` score combines these so the worst labels float to the top.

Usage::

    python scripts/audit_labels.py \\
        --data-dir /Volumes/TAMWoolff/data/preprocessed/sora_diverse \\
        --csv-out /tmp/sora_audit.csv \\
        --grid-out /tmp/sora_worst30.png \\
        --n-worst 30

Outputs:
- CSV with per-example metrics and rank
- Grid PNG of the N worst examples (same layout as scripts/sample_grid.py)
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image

# Lightweight: avoid scipy dependency by rolling our own connected-component count
# via a simple 4-connected flood fill on small images (512×512 is tiny).

# Paired L/R classes
LR_PAIRS = [
    (6, 10),   # shoulder
    (7, 11),   # upper_arm
    (8, 12),   # forearm
    (9, 13),   # hand
    (14, 17),  # upper_leg
    (15, 18),  # lower_leg
    (16, 19),  # foot
]

HEAD = 1
TORSO_CLASSES = {3, 4, 5}  # chest, spine, hips

REGION_NAMES = [
    "background", "head", "neck", "chest", "spine", "hips",
    "shoulder_l", "upper_arm_l", "forearm_l", "hand_l",
    "shoulder_r", "upper_arm_r", "forearm_r", "hand_r",
    "upper_leg_l", "lower_leg_l", "foot_l",
    "upper_leg_r", "lower_leg_r", "foot_r",
    "accessory", "hair_back",
]

REGION_COLORS = [
    (  0,   0,   0), (255,  99,  99), (255, 180,  99), (255, 255,  99),
    (180, 255,  99), ( 99, 255,  99), ( 99, 255, 180), ( 99, 255, 255),
    ( 99, 180, 255), ( 99,  99, 255), (180,  99, 255), (255,  99, 255),
    (255,  99, 180), (180, 180, 180), (255, 128,   0), (  0, 255, 128),
    (128,   0, 255), (255,   0, 128), (  0, 128, 255), (128, 255,   0),
    (200, 200, 200), (255, 200, 150),
]


@dataclass
class AuditRow:
    name: str
    mean_conf: float
    largest_frac: float
    n_classes: int
    disconn_count: int
    head_above_torso: int
    lr_asymmetry: float
    badness: float


def count_connected_components(mask: np.ndarray) -> int:
    """Count 4-connected components in a 2D bool mask using iterative flood fill."""
    if not mask.any():
        return 0
    seen = np.zeros_like(mask, dtype=bool)
    H, W = mask.shape
    count = 0
    ys, xs = np.where(mask & ~seen)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        count += 1
        # Iterative stack-based flood fill
        stack = [(y0, x0)]
        while stack:
            y, x = stack.pop()
            if y < 0 or y >= H or x < 0 or x >= W:
                continue
            if seen[y, x] or not mask[y, x]:
                continue
            seen[y, x] = True
            stack.append((y + 1, x))
            stack.append((y - 1, x))
            stack.append((y, x + 1))
            stack.append((y, x - 1))
    return count


def audit_example(example_dir: Path) -> AuditRow | None:
    seg_p = example_dir / "segmentation.png"
    if not seg_p.exists():
        return None
    seg = np.array(Image.open(seg_p))

    fg = seg > 0
    fg_count = int(fg.sum())
    if fg_count == 0:
        return AuditRow(example_dir.name, 0, 0, 0, 0, 0, 0, 1.0)

    # Confidence if available
    conf_p = example_dir / "confidence.png"
    if conf_p.exists():
        conf = np.array(Image.open(conf_p).convert("L")).astype(np.float32) / 255.0
        mean_conf = float(conf[fg].mean())
    else:
        mean_conf = -1.0  # missing

    # Class areas
    class_areas: dict[int, int] = {}
    for cls in range(1, 22):
        a = int((seg == cls).sum())
        if a > 0:
            class_areas[cls] = a
    n_classes = len(class_areas)

    # Largest class fraction
    if class_areas:
        largest_frac = max(class_areas.values()) / fg_count
    else:
        largest_frac = 0.0

    # Connected components
    disconn_count = 0
    for cls in class_areas.keys():
        disconn_count += count_connected_components(seg == cls)

    # Head above torso
    head_above_torso = 0
    if HEAD in class_areas and any(c in class_areas for c in TORSO_CLASSES):
        head_ys, _ = np.where(seg == HEAD)
        head_cy = float(head_ys.mean())
        torso_ys_all = []
        for c in TORSO_CLASSES:
            if c in class_areas:
                ys, _ = np.where(seg == c)
                torso_ys_all.extend(ys.tolist())
        if torso_ys_all:
            torso_max_cy = max(torso_ys_all) / 1.0  # lowest pixel of any torso class
            # expect head_cy < torso lower-half centroid; loose check: head above torso median
            torso_median = float(np.median(torso_ys_all))
            head_above_torso = 1 if head_cy < torso_median else 0

    # L/R asymmetry across paired classes
    asym_scores = []
    for left_id, right_id in LR_PAIRS:
        al = class_areas.get(left_id, 0)
        ar = class_areas.get(right_id, 0)
        m = max(al, ar)
        if m > 0:
            asym_scores.append(abs(al - ar) / m)
    lr_asymmetry = max(asym_scores) if asym_scores else 0.0

    # Composite badness score — tuned to surface obvious problems
    badness = 0.0
    if mean_conf >= 0:
        badness += (1.0 - mean_conf) * 0.6                   # low confidence
    badness += max(0, largest_frac - 0.50) * 1.0              # one class dominating
    badness += max(0, 6 - n_classes) * 0.15                   # too few classes
    # Disconnected components: expect ~n_classes for a clean char; penalise the excess
    badness += max(0, disconn_count - n_classes - 2) * 0.05
    if HEAD in class_areas and any(c in class_areas for c in TORSO_CLASSES):
        badness += (1 - head_above_torso) * 0.6               # head below torso = obvious flip
    badness += max(0, lr_asymmetry - 0.70) * 0.5              # one-sided labels

    return AuditRow(
        name=example_dir.name,
        mean_conf=mean_conf,
        largest_frac=largest_frac,
        n_classes=n_classes,
        disconn_count=disconn_count,
        head_above_torso=head_above_torso,
        lr_asymmetry=lr_asymmetry,
        badness=badness,
    )


def colorize(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, color in enumerate(REGION_COLORS):
        out[mask == cls] = color
    return out


def make_row(example_dir: Path, cell: int = 256, gap: int = 4) -> Image.Image | None:
    img_p = example_dir / "image.png"
    seg_p = example_dir / "segmentation.png"
    if not (img_p.exists() and seg_p.exists()):
        return None
    img = Image.open(img_p).convert("RGBA").resize((cell, cell), Image.LANCZOS)
    seg = np.array(Image.open(seg_p).resize((cell, cell), Image.NEAREST))
    colored = colorize(seg)
    img_rgb = Image.new("RGB", (cell, cell), (40, 40, 40))
    img_rgb.paste(img, (0, 0), img.split()[-1] if img.mode == "RGBA" else None)
    overlay = Image.blend(img_rgb, Image.fromarray(colored), alpha=0.55)
    mask_img = Image.fromarray(colored)
    row = Image.new("RGB", (cell * 3 + gap * 2, cell), (25, 25, 25))
    row.paste(img_rgb, (0, 0))
    row.paste(overlay, (cell + gap, 0))
    row.paste(mask_img, (cell * 2 + gap * 2, 0))
    return row


def save_worst_grid(
    worst: list[tuple[AuditRow, Path]],
    output: Path,
    cell: int = 256,
    gap: int = 4,
) -> None:
    from PIL import ImageDraw, ImageFont
    if not worst:
        return
    label_w = 340
    row_w = cell * 3 + gap * 2
    grid_w = label_w + row_w
    grid_h = cell * len(worst) + gap * (len(worst) - 1)
    grid = Image.new("RGB", (grid_w, grid_h), (15, 15, 15))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except Exception:
        font = ImageFont.load_default()
    for i, (row, example_dir) in enumerate(worst):
        r = make_row(example_dir, cell=cell, gap=gap)
        if r is None:
            continue
        y = i * (cell + gap)
        grid.paste(r, (label_w, y))
        txt_lines = [
            row.name[:44],
            f"badness {row.badness:.2f}",
            f"conf {row.mean_conf:.2f}" if row.mean_conf >= 0 else "conf n/a",
            f"n_classes {row.n_classes}  largest {row.largest_frac:.2f}",
            f"disconn {row.disconn_count}  lr_asym {row.lr_asymmetry:.2f}",
            f"head_above_torso {row.head_above_torso}",
        ]
        for li, t in enumerate(txt_lines):
            draw.text((6, y + 6 + li * 14), t, fill=(230, 230, 230), font=font)
    grid.save(output)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--csv-out", type=Path, default=None)
    p.add_argument("--grid-out", type=Path, default=None,
                   help="Save N-worst examples as a grid PNG here.")
    p.add_argument("--n-worst", type=int, default=30)
    p.add_argument("--limit", type=int, default=None,
                   help="Only audit the first N examples (debug)")
    args = p.parse_args()

    if not args.data_dir.is_dir():
        print(f"not a dir: {args.data_dir}", file=sys.stderr)
        return 1

    examples = sorted(
        d for d in args.data_dir.iterdir()
        if d.is_dir() and (d / "segmentation.png").exists()
    )
    if args.limit:
        examples = examples[: args.limit]

    rows: list[tuple[AuditRow, Path]] = []
    for i, ex in enumerate(examples):
        r = audit_example(ex)
        if r is not None:
            rows.append((r, ex))
        if (i + 1) % 200 == 0:
            print(f"  audited {i + 1}/{len(examples)}")

    if not rows:
        print("no valid examples", file=sys.stderr)
        return 1

    rows.sort(key=lambda t: -t[0].badness)

    # Summary
    total = len(rows)
    mean_badness = float(np.mean([r.badness for r, _ in rows]))
    med_badness = float(np.median([r.badness for r, _ in rows]))
    pct_low_conf = sum(1 for r, _ in rows if 0 <= r.mean_conf < 0.5) / total * 100
    pct_one_class_dom = sum(1 for r, _ in rows if r.largest_frac > 0.7) / total * 100
    pct_few_classes = sum(1 for r, _ in rows if r.n_classes < 4) / total * 100
    pct_head_below = sum(1 for r, _ in rows
                         if r.head_above_torso == 0 and r.n_classes >= 4) / total * 100
    pct_asym = sum(1 for r, _ in rows if r.lr_asymmetry > 0.70) / total * 100

    print(f"\nDataset: {args.data_dir.name}  ({total} examples)")
    print(f"  mean badness:       {mean_badness:.3f}")
    print(f"  median badness:     {med_badness:.3f}")
    print(f"  % low confidence:   {pct_low_conf:.1f}%  (< 0.5)")
    print(f"  % single-class dom: {pct_one_class_dom:.1f}%  (> 70% fg)")
    print(f"  % few classes:      {pct_few_classes:.1f}%  (< 4 non-bg)")
    print(f"  % head not above torso: {pct_head_below:.1f}%")
    print(f"  % L/R asymmetry:    {pct_asym:.1f}%  (> 70% imbalance)")
    print(f"  worst 5:")
    for r, ex in rows[:5]:
        print(f"    {r.badness:.3f}  {ex.name}")

    if args.csv_out:
        with args.csv_out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(asdict(rows[0][0]).keys()) + ["rank"])
            for rank, (r, _) in enumerate(rows, 1):
                w.writerow(list(asdict(r).values()) + [rank])
        print(f"\n  CSV → {args.csv_out}")

    if args.grid_out:
        worst = rows[: args.n_worst]
        save_worst_grid(worst, args.grid_out)
        print(f"  Grid (worst {len(worst)}) → {args.grid_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
