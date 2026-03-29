#!/usr/bin/env python3
"""Score illustrated characters by segmentation quality for demo selection.

Runs the seg model on a directory of preprocessed characters and ranks
them by a composite quality score based on:
- Number of distinct body regions detected
- Foreground coverage (not too small, not too large)
- Region symmetry (L/R balance)
- No missing critical parts (head, chest, at least one arm+leg)

Outputs a ranked CSV and optionally saves a grid of the top-N characters.

Usage::

    python scripts/score_illustrations.py \
        --input-dir /Volumes/TAMWoolff/data/preprocessed/gemini_diverse \
        --checkpoint /Volumes/TAMWoolff/data/checkpoints_run20_seg/run20_best.pt \
        --output scores.csv --top-n 20 --device mps
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

NUM_CLASSES = 22

# Expected body regions for a good full-body character
CRITICAL_REGIONS = {1, 3, 5}  # head, chest, hips
ARM_REGIONS_L = {7, 8, 9}     # upper_arm_l, forearm_l, hand_l
ARM_REGIONS_R = {11, 12, 13}   # upper_arm_r, forearm_r, hand_r
LEG_REGIONS_L = {14, 15, 16}   # upper_leg_l, lower_leg_l, foot_l
LEG_REGIONS_R = {17, 18, 19}   # upper_leg_r, lower_leg_r, foot_r

LR_PAIRS = [
    ({6}, {10}),    # shoulder
    ({7}, {11}),    # upper_arm
    ({8}, {12}),    # forearm
    ({9}, {13}),    # hand
    ({14}, {17}),   # upper_leg
    ({15}, {18}),   # lower_leg
    ({16}, {19}),   # foot
]


def load_model(checkpoint_path: Path, device: str) -> torch.nn.Module:
    from training.models.segmentation_model import SegmentationModel

    model = SegmentationModel(backbone="mobilenet_v3_large", num_classes=NUM_CLASSES)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_segmentation(model: torch.nn.Module, image_path: Path, device: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
    seg = out["segmentation"].argmax(dim=1).squeeze().cpu().numpy()
    return seg


def score_character(seg: np.ndarray) -> dict:
    """Score a segmentation mask for demo quality."""
    h, w = seg.shape
    total_pixels = h * w

    # Foreground pixels
    fg_mask = seg > 0
    fg_pixels = fg_mask.sum()
    fg_ratio = fg_pixels / total_pixels

    # Unique regions
    regions = set(np.unique(seg)) - {0}
    num_regions = len(regions)

    # Critical regions check
    has_head = bool(regions & {1})
    has_chest = bool(regions & {3})
    has_hips = bool(regions & {5})
    has_arm_l = bool(regions & ARM_REGIONS_L)
    has_arm_r = bool(regions & ARM_REGIONS_R)
    has_leg_l = bool(regions & LEG_REGIONS_L)
    has_leg_r = bool(regions & LEG_REGIONS_R)

    critical_score = sum([has_head, has_chest, has_hips, has_arm_l, has_arm_r, has_leg_l, has_leg_r]) / 7.0

    # L/R symmetry score
    symmetry_scores = []
    for l_set, r_set in LR_PAIRS:
        l_pixels = sum((seg == c).sum() for c in l_set)
        r_pixels = sum((seg == c).sum() for c in r_set)
        if l_pixels + r_pixels > 0:
            sym = min(l_pixels, r_pixels) / max(l_pixels, r_pixels)
            symmetry_scores.append(sym)
    symmetry = np.mean(symmetry_scores) if symmetry_scores else 0.0

    # Region size balance (penalize if one region dominates)
    region_sizes = []
    for c in regions:
        region_sizes.append((seg == c).sum() / fg_pixels if fg_pixels > 0 else 0)
    max_region_ratio = max(region_sizes) if region_sizes else 1.0
    balance = 1.0 - max_region_ratio  # higher is better

    # Foreground size penalty (too small or too large)
    fg_score = 1.0
    if fg_ratio < 0.10:
        fg_score = fg_ratio / 0.10
    elif fg_ratio > 0.70:
        fg_score = (1.0 - fg_ratio) / 0.30

    # Composite score
    composite = (
        0.30 * critical_score +
        0.25 * min(num_regions / 15.0, 1.0) +
        0.20 * symmetry +
        0.15 * balance +
        0.10 * fg_score
    )

    return {
        "num_regions": num_regions,
        "fg_ratio": round(fg_ratio, 4),
        "critical_score": round(critical_score, 3),
        "symmetry": round(symmetry, 3),
        "balance": round(balance, 3),
        "composite": round(composite, 4),
        "has_head": has_head,
        "has_chest": has_chest,
        "has_arms": has_arm_l and has_arm_r,
        "has_legs": has_leg_l and has_leg_r,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score illustrations for demo selection")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("scores.csv"))
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--use-existing-masks", action="store_true",
                        help="Use existing segmentation.png instead of running model")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model = None
    if not args.use_existing_masks:
        logger.info("Loading model from %s", args.checkpoint)
        model = load_model(args.checkpoint, args.device)

    # Find all examples
    examples = []
    for child in sorted(args.input_dir.iterdir()):
        if not child.is_dir():
            continue
        img_path = child / "image.png"
        if img_path.exists():
            examples.append(child)

    logger.info("Found %d examples", len(examples))

    results = []
    for i, ex_dir in enumerate(examples):
        if args.use_existing_masks:
            seg_path = ex_dir / "segmentation.png"
            if not seg_path.exists():
                continue
            seg = np.array(Image.open(seg_path).convert("L"), dtype=np.int64)
        else:
            seg = predict_segmentation(model, ex_dir / "image.png", args.device)

        scores = score_character(seg)
        scores["name"] = ex_dir.name
        results.append(scores)

        if (i + 1) % 100 == 0:
            logger.info("  %d/%d scored", i + 1, len(examples))

    # Sort by composite score
    results.sort(key=lambda x: x["composite"], reverse=True)

    # Write CSV
    if results:
        fields = ["name", "composite", "num_regions", "fg_ratio", "critical_score",
                   "symmetry", "balance", "has_head", "has_chest", "has_arms", "has_legs"]
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        logger.info("Wrote %d scores to %s", len(results), args.output)

    # Print top N
    print(f"\nTop {args.top_n} characters for demo:")
    print("-" * 80)
    for r in results[:args.top_n]:
        print(f"  {r['composite']:.3f}  regions={r['num_regions']:2d}  sym={r['symmetry']:.2f}  "
              f"fg={r['fg_ratio']:.2f}  {r['name']}")

    logger.info("Done. Top score: %.4f", results[0]["composite"] if results else 0)


if __name__ == "__main__":
    main()
