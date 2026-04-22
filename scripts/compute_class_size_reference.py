#!/usr/bin/env python3
"""Compute per-class area distributions from a known-good (hand-labeled) dataset.

Scans a preprocessed dataset directory, computes per-example (area as fraction
of foreground) for each of the 22 anatomy classes, then saves mean + std per
class to a JSON file. ``filter_seg_quality.py --class-size-ref`` uses this to
flag examples whose class areas are statistical outliers.

Intended input: ``gemini_li_converted`` (Dr. Li's 694 hand-labeled illustrated
chars) — small, diverse, and label-space-accurate.

Usage::

    python scripts/compute_class_size_reference.py \\
        --data-dir /Volumes/TAMWoolff/data/preprocessed/gemini_li_converted \\
        --output /Volumes/TAMWoolff/data/class_size_reference.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REGION_NAMES = [
    "background", "head", "neck", "chest", "spine", "hips",
    "shoulder_l", "upper_arm_l", "forearm_l", "hand_l",
    "shoulder_r", "upper_arm_r", "forearm_r", "hand_r",
    "upper_leg_l", "lower_leg_l", "foot_l",
    "upper_leg_r", "lower_leg_r", "foot_r",
    "accessory", "hair_back",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    # Per-class list of (area_fraction) values, across all examples that contain that class
    class_fracs: dict[int, list[float]] = {i: [] for i in range(22)}
    n_examples = 0

    for ex in sorted(args.data_dir.iterdir()):
        seg_p = ex / "segmentation.png"
        if not seg_p.is_file():
            continue
        try:
            arr = np.asarray(Image.open(seg_p), dtype=np.uint8)
        except Exception:
            continue
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        fg = int((arr > 0).sum())
        if fg == 0:
            continue
        n_examples += 1
        counts = np.bincount(arr.ravel(), minlength=22)
        for cls in range(1, 22):
            if counts[cls] > 0:
                class_fracs[cls].append(float(counts[cls]) / fg)

    if n_examples == 0:
        print(f"no examples in {args.data_dir}", file=sys.stderr)
        return 1

    # Compute mean, std, median, counts per class
    classes_out: dict[int, dict] = {}
    for cls in range(1, 22):
        vals = class_fracs[cls]
        if not vals:
            continue
        arr = np.asarray(vals)
        classes_out[cls] = {
            "name": REGION_NAMES[cls],
            "count": int(arr.size),
            "presence_rate": round(arr.size / n_examples, 4),
            "mean": round(float(arr.mean()), 5),
            "std": round(float(arr.std()), 5),
            "median": round(float(np.median(arr)), 5),
            "p10": round(float(np.percentile(arr, 10)), 5),
            "p90": round(float(np.percentile(arr, 90)), 5),
        }

    out = {
        "source": str(args.data_dir),
        "n_examples": n_examples,
        "classes": {str(k): v for k, v in classes_out.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"Reference from {n_examples} examples → {args.output}")
    print()
    print(f"{'class':<16} {'presence':>9} {'mean':>8} {'std':>8} {'p10':>8} {'p90':>8}")
    for cls, info in classes_out.items():
        print(f"{info['name']:<16} {info['presence_rate']:>9.2%} "
              f"{info['mean']:>8.3f} {info['std']:>8.3f} "
              f"{info['p10']:>8.3f} {info['p90']:>8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
