#!/usr/bin/env python3
"""Auto-triage pseudo-labeled masks into accept/reject/review buckets.

Analyzes each segmentation mask for anatomical plausibility and confidence,
then updates the review_manifest.json with auto-accept/auto-reject statuses.
Only the uncertain middle band needs manual review.

Heuristics:
  - Expected body parts: head(1), chest(3), at least one arm, at least one leg
  - Region count: too few (<4) or too many regions on small images → suspicious
  - Region proportions: head/chest shouldn't dominate >60% of foreground
  - Foreground ratio: characters should cover 10-90% of image area
  - Confidence: mean confidence over foreground pixels (requires --checkpoint)

Usage::

    # Mask-only triage (no model needed, fast)
    python scripts/auto_triage.py --data-dir ./output/gemini_corrected

    # With confidence scoring (re-runs inference, slower but more accurate)
    python scripts/auto_triage.py --data-dir ./output/gemini_corrected \
        --checkpoint checkpoints/segmentation/run7_best.pt

    # Dry run — just print stats, don't update manifest
    python scripts/auto_triage.py --data-dir ./output/gemini_corrected --dry-run

    # Custom thresholds
    python scripts/auto_triage.py --data-dir ./output/gemini_corrected \
        --accept-threshold 0.7 --reject-threshold 0.3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

logger = logging.getLogger(__name__)

# Body part groups for anatomical checks
CORE_PARTS = {1, 3}  # head, chest — should almost always be present
ARM_PARTS_L = {6, 7, 8, 9}  # shoulder_l, upper_arm_l, forearm_l, hand_l
ARM_PARTS_R = {10, 11, 12, 13}  # shoulder_r, upper_arm_r, forearm_r, hand_r
LEG_PARTS_L = {14, 15, 16}  # upper_leg_l, lower_leg_l, foot_l
LEG_PARTS_R = {17, 18, 19}  # upper_leg_r, lower_leg_r, foot_r
ALL_BODY = CORE_PARTS | ARM_PARTS_L | ARM_PARTS_R | LEG_PARTS_L | LEG_PARTS_R | {2, 4, 5}  # +neck,spine,hips


@dataclass
class TriageResult:
    name: str
    score: float  # 0.0 (bad) to 1.0 (good)
    reasons: list[str]
    decision: str  # "accept", "reject", "review"


def analyze_mask(mask: np.ndarray, alpha: np.ndarray | None = None) -> tuple[float, list[str]]:
    """Score a segmentation mask on anatomical plausibility.

    Returns (score, reasons) where score is 0.0-1.0 and reasons lists any issues.
    """
    h, w = mask.shape
    total_pixels = h * w

    # Foreground = non-zero mask pixels
    fg_mask = mask > 0
    fg_pixels = int(fg_mask.sum())

    # If we have alpha, use it for expected foreground area
    if alpha is not None:
        expected_fg = int((alpha > 10).sum())
    else:
        expected_fg = fg_pixels

    fg_ratio = fg_pixels / total_pixels if total_pixels > 0 else 0
    reasons = []
    penalties = 0.0

    # --- Check 1: Foreground coverage ---
    if fg_ratio < 0.05:
        reasons.append(f"very small foreground ({fg_ratio:.1%})")
        penalties += 0.5
    elif fg_ratio < 0.10:
        reasons.append(f"small foreground ({fg_ratio:.1%})")
        penalties += 0.2

    if fg_ratio > 0.90:
        reasons.append(f"foreground fills image ({fg_ratio:.1%})")
        penalties += 0.3

    # --- Check 2: Unique regions present ---
    unique_regions = set(int(r) for r in np.unique(mask) if r > 0)
    n_regions = len(unique_regions)

    if n_regions < 3:
        reasons.append(f"only {n_regions} body regions detected")
        penalties += 0.6
    elif n_regions < 5:
        reasons.append(f"few body regions ({n_regions})")
        penalties += 0.2

    # --- Check 3: Core body parts ---
    has_head = 1 in unique_regions
    has_chest = 3 in unique_regions
    has_any_arm = bool(unique_regions & (ARM_PARTS_L | ARM_PARTS_R))
    has_any_leg = bool(unique_regions & (LEG_PARTS_L | LEG_PARTS_R))

    if not has_head:
        reasons.append("missing head")
        penalties += 0.3
    if not has_chest:
        reasons.append("missing chest")
        penalties += 0.3
    if not has_any_arm:
        reasons.append("missing arms")
        penalties += 0.2
    if not has_any_leg:
        reasons.append("missing legs")
        penalties += 0.15

    # --- Check 4: Region proportion sanity ---
    if fg_pixels > 0:
        region_counts = {}
        for rid in unique_regions:
            region_counts[rid] = int((mask == rid).sum())

        # No single non-accessory region should dominate >50% of foreground
        for rid, count in region_counts.items():
            if rid in (20, 21):  # accessory/hair_back can be large
                continue
            ratio = count / fg_pixels
            if ratio > 0.50:
                from pipeline.config import REGION_NAMES
                name = REGION_NAMES.get(rid, f"region_{rid}")
                reasons.append(f"{name} too large ({ratio:.0%} of fg)")
                penalties += 0.25

        # Head should be present and not tiny (>1% of fg) if detected
        if has_head:
            head_ratio = region_counts.get(1, 0) / fg_pixels
            if head_ratio < 0.01:
                reasons.append(f"head extremely small ({head_ratio:.1%})")
                penalties += 0.15

    # --- Check 5: Left/right symmetry (soft) ---
    has_left_arm = bool(unique_regions & ARM_PARTS_L)
    has_right_arm = bool(unique_regions & ARM_PARTS_R)
    has_left_leg = bool(unique_regions & LEG_PARTS_L)
    has_right_leg = bool(unique_regions & LEG_PARTS_R)

    # Having one arm but not the other is OK (could be pose), but note it
    if has_left_arm != has_right_arm:
        reasons.append("asymmetric arms (may be occluded)")
        penalties += 0.05
    if has_left_leg != has_right_leg:
        reasons.append("asymmetric legs (may be occluded)")
        penalties += 0.05

    # --- Check 6: Mask vs alpha agreement ---
    if alpha is not None and expected_fg > 0:
        # Foreground pixels with no mask label (should be labeled)
        unlabeled_fg = int(((alpha > 10) & (mask == 0)).sum())
        unlabeled_ratio = unlabeled_fg / expected_fg
        if unlabeled_ratio > 0.30:
            reasons.append(f"large unlabeled foreground ({unlabeled_ratio:.0%})")
            penalties += 0.25
        elif unlabeled_ratio > 0.15:
            reasons.append(f"some unlabeled foreground ({unlabeled_ratio:.0%})")
            penalties += 0.1

    score = max(0.0, 1.0 - penalties)
    return score, reasons


def triage_example(
    example_dir: Path,
    accept_threshold: float,
    reject_threshold: float,
    confidence_map: np.ndarray | None = None,
) -> TriageResult:
    """Analyze one example and return triage decision."""
    name = example_dir.name
    mask_path = example_dir / "segmentation.png"
    image_path = example_dir / "image.png"

    if not mask_path.exists():
        return TriageResult(name, 0.0, ["no segmentation.png"], "reject")

    mask = np.array(Image.open(mask_path).convert("L"))

    # Load alpha channel
    alpha = None
    if image_path.exists():
        img = Image.open(image_path).convert("RGBA")
        if img.size != mask.shape[::-1]:
            img = img.resize(mask.shape[::-1], Image.LANCZOS)
        alpha = np.array(img)[:, :, 3]

    # Mask-based scoring
    score, reasons = analyze_mask(mask, alpha)

    # Confidence-based scoring (if available)
    if confidence_map is not None:
        fg_mask = mask > 0
        if fg_mask.sum() > 0:
            mean_conf = float(confidence_map[fg_mask].mean())
            if mean_conf < 0.3:
                reasons.append(f"low model confidence ({mean_conf:.2f})")
                score = max(0.0, score - 0.3)
            elif mean_conf < 0.5:
                reasons.append(f"moderate model confidence ({mean_conf:.2f})")
                score = max(0.0, score - 0.1)
            elif mean_conf > 0.8:
                score = min(1.0, score + 0.1)  # boost for high confidence

    # Decision
    if score >= accept_threshold:
        decision = "accept"
    elif score <= reject_threshold:
        decision = "reject"
    else:
        decision = "review"

    return TriageResult(name, score, reasons, decision)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-triage pseudo-labeled masks into accept/reject/review buckets."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Seg checkpoint for confidence scoring (optional, slower)")
    parser.add_argument("--accept-threshold", type=float, default=0.7,
                        help="Score >= this → auto-accept (default: 0.7)")
    parser.add_argument("--reject-threshold", type=float, default=0.3,
                        help="Score <= this → auto-reject (default: 0.3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print results without updating manifest")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Discover examples
    examples = sorted(
        d for d in args.data_dir.iterdir()
        if d.is_dir() and (d / "image.png").exists()
        and not d.name.startswith(".")
    )
    logger.info("Found %d examples in %s", len(examples), args.data_dir)

    # Optionally load model for confidence scoring
    model = None
    device = None
    if args.checkpoint:
        import torch
        from run_seg_enrich import load_seg_model, predict_segmentation

        if args.device:
            device = torch.device(args.device)
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        logger.info("Loading model from %s on %s", args.checkpoint, device)
        model = load_seg_model(args.checkpoint, device)

    # Load existing manifest
    manifest_path = args.data_dir / "review_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"total": 0, "reviewed": 0, "rejected": 0, "needs_review": 0, "examples": {}}

    # Triage each example
    results: list[TriageResult] = []
    accepted = rejected = review = 0
    already_reviewed = 0

    for i, example_dir in enumerate(examples):
        name = example_dir.name

        # Skip already manually reviewed examples
        existing_status = manifest.get("examples", {}).get(name, {}).get("status", "needs_review")
        if existing_status in ("reviewed", "rejected"):
            already_reviewed += 1
            continue

        # Get confidence map if model available
        confidence_map = None
        if model is not None:
            from pipeline.config import RENDER_RESOLUTION
            from run_seg_enrich import predict_segmentation
            _, _, confidence_map = predict_segmentation(
                model, example_dir / "image.png", device, RENDER_RESOLUTION
            )

        result = triage_example(
            example_dir, args.accept_threshold, args.reject_threshold, confidence_map
        )
        results.append(result)

        if result.decision == "accept":
            accepted += 1
        elif result.decision == "reject":
            rejected += 1
        else:
            review += 1

        if (i + 1) % 100 == 0:
            logger.info("Progress: %d/%d", i + 1, len(examples))

    # Print summary
    print(f"\n{'='*60}")
    print(f"Auto-triage results for {args.data_dir}")
    print(f"{'='*60}")
    print(f"  Already reviewed:  {already_reviewed}")
    print(f"  Auto-accept:       {accepted}  (score >= {args.accept_threshold})")
    print(f"  Auto-reject:       {rejected}  (score <= {args.reject_threshold})")
    print(f"  Needs review:      {review}  (manual)")
    print(f"  Total triaged:     {len(results)}")
    print()

    # Print rejected examples with reasons
    rejected_results = [r for r in results if r.decision == "reject"]
    if rejected_results:
        print(f"--- Auto-rejected ({len(rejected_results)}) ---")
        for r in sorted(rejected_results, key=lambda x: x.score):
            print(f"  {r.name}: score={r.score:.2f} — {', '.join(r.reasons)}")
        print()

    # Print review examples with reasons
    review_results = [r for r in results if r.decision == "review"]
    if review_results:
        print(f"--- Needs manual review ({len(review_results)}) ---")
        for r in sorted(review_results, key=lambda x: x.score):
            print(f"  {r.name}: score={r.score:.2f} — {', '.join(r.reasons)}")
        print()

    # Score distribution
    if results:
        scores = [r.score for r in results]
        print(f"Score distribution:")
        print(f"  Min:    {min(scores):.2f}")
        print(f"  Median: {sorted(scores)[len(scores)//2]:.2f}")
        print(f"  Mean:   {sum(scores)/len(scores):.2f}")
        print(f"  Max:    {max(scores):.2f}")
        print()

    # Update manifest (unless dry run)
    if not args.dry_run:
        for result in results:
            if result.decision == "accept":
                manifest.setdefault("examples", {})[result.name] = {"status": "reviewed"}
            elif result.decision == "reject":
                manifest.setdefault("examples", {})[result.name] = {"status": "rejected"}
            # "review" stays as "needs_review"

        # Recount
        all_examples = manifest.get("examples", {})
        manifest["total"] = len(all_examples)
        manifest["reviewed"] = sum(1 for e in all_examples.values() if e["status"] == "reviewed")
        manifest["rejected"] = sum(1 for e in all_examples.values() if e["status"] == "rejected")
        manifest["needs_review"] = sum(1 for e in all_examples.values() if e["status"] == "needs_review")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        logger.info("Updated manifest: %d accepted, %d rejected, %d needs_review",
                     manifest["reviewed"], manifest["rejected"], manifest["needs_review"])
        print(f"Next: python scripts/review_masks.py --data-dir {args.data_dir} --only-needs-review")
    else:
        print("(dry run — manifest not updated)")


if __name__ == "__main__":
    main()
