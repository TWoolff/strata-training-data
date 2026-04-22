#!/usr/bin/env python3
"""Ingest Gemini-generated character images into Strata training format.

End-to-end pipeline:
1. Background removal + resize to 512x512 (gemini_diverse_adapter)
2. 22-class segmentation pseudo-labels (trained Model 1)
3. Confidence-based quality report (flags low-quality examples for review)

Usage:
    # Full pipeline: ingest + pseudo-label
    python3 scripts/ingest_gemini.py \
        --input-dir /Volumes/TAMWoolff/data/raw/gemini_diverse \
        --output-dir ./output/gemini_diverse \
        --checkpoint checkpoints/segmentation/best.pt

    # Skip already-processed images
    python3 scripts/ingest_gemini.py \
        --input-dir /Volumes/TAMWoolff/data/raw/gemini_diverse \
        --output-dir ./output/gemini_diverse \
        --checkpoint checkpoints/segmentation/best.pt \
        --only-new

    # Ingest only (no pseudo-labels, no checkpoint needed)
    python3 scripts/ingest_gemini.py \
        --input-dir /Volumes/TAMWoolff/data/raw/gemini_diverse \
        --output-dir ./output/gemini_diverse \
        --no-seg
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

logger = logging.getLogger(__name__)

REGION_NAMES = [
    "background", "head", "neck", "chest", "spine", "hips",
    "shoulder_l", "upper_arm_l", "forearm_l", "hand_l",
    "shoulder_r", "upper_arm_r", "forearm_r", "hand_r",
    "upper_leg_l", "lower_leg_l", "foot_l",
    "upper_leg_r", "lower_leg_r", "foot_r",
    "accessory", "hair_back",
]


def run_ingest(input_dir: Path, output_dir: Path, *, only_new: bool = False) -> int:
    """Step 1: Run gemini_diverse_adapter to convert raw images."""
    from ingest.gemini_diverse_adapter import convert_directory

    result = convert_directory(
        input_dir, output_dir, only_new=only_new,
    )
    logger.info(
        "Ingest: %d processed, %d skipped, %d errors",
        result.images_processed, result.images_skipped, len(result.errors),
    )
    for err in result.errors:
        logger.warning("  %s", err)
    return result.images_processed


def run_pseudo_labels(
    output_dir: Path,
    checkpoint: Path,
    *,
    only_new: bool = False,
) -> dict[str, float]:
    """Step 2: Run seg model on all examples, save 22-class masks + confidence."""
    import torch
    from run_seg_enrich import load_seg_model, predict_segmentation

    # Auto-detect device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Pseudo-label device: %s", device)

    model = load_seg_model(checkpoint, device)

    examples = sorted(
        d for d in output_dir.iterdir()
        if d.is_dir() and (d / "image.png").exists()
    )

    confidence_scores: dict[str, float] = {}
    enriched = 0
    skipped = 0
    start = time.monotonic()

    for i, example_dir in enumerate(examples):
        # Skip if already has model-generated seg
        if only_new:
            meta_path = example_dir / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("segmentation_source", "").startswith("model_"):
                    skipped += 1
                    continue

        image_path = example_dir / "image.png"
        seg_mask, draw_order, confidence = predict_segmentation(
            model, image_path, device, 512,
        )

        # Zero out background using alpha
        img = Image.open(image_path).convert("RGBA")
        alpha = np.array(img)[:, :, 3]
        fg_mask = alpha >= 10
        seg_mask[~fg_mask] = 0
        draw_order[~fg_mask] = 0
        confidence[~fg_mask] = 0.0

        # Compute mean confidence over foreground pixels
        if fg_mask.any():
            mean_conf = float(confidence[fg_mask].mean())
        else:
            mean_conf = 0.0
        confidence_scores[example_dir.name] = mean_conf

        # Count unique regions (excluding background)
        unique_regions = set(int(v) for v in np.unique(seg_mask) if v > 0)
        region_names = [REGION_NAMES[r] for r in sorted(unique_regions)]

        # Save outputs
        Image.fromarray(seg_mask).save(example_dir / "segmentation.png")
        Image.fromarray(draw_order).save(example_dir / "draw_order.png")

        # Save confidence map as uint8 for visualization
        conf_uint8 = (confidence * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(conf_uint8).save(example_dir / "confidence.png")

        # Update metadata
        meta_path = example_dir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            meta = {"id": example_dir.name, "source": "gemini_diverse"}

        meta["has_segmentation_mask"] = True
        meta["has_draw_order"] = True
        meta["segmentation_source"] = "model_v3"
        meta["seg_mean_confidence"] = round(mean_conf, 4)
        meta["seg_num_regions"] = len(unique_regions)
        meta["seg_regions"] = region_names
        if "missing_annotations" in meta:
            meta["missing_annotations"] = [
                a for a in meta["missing_annotations"]
                if a != "strata_segmentation"
            ]

        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        enriched += 1

        if (i + 1) % 20 == 0 or (i + 1) == len(examples):
            elapsed = time.monotonic() - start
            speed = enriched / elapsed if elapsed > 0 else 0
            logger.info(
                "Pseudo-labels: %d/%d (%.1f img/s, %d skipped)",
                i + 1, len(examples), speed, skipped,
            )

    logger.info(
        "Pseudo-labeling complete: %d enriched, %d skipped",
        enriched, skipped,
    )
    return confidence_scores


def run_quality_filter(
    output_dir: Path,
    *,
    drop_rejected: bool = False,
    drop_head_below_torso: bool = True,
    max_bg_bleed: float = 0.10,
    min_silhouette_coverage: float = 0.50,
) -> dict[str, list[str]]:
    """Step 3: Apply validated quality filters to pseudo-labeled examples.

    Uses the same checks as ``scripts/filter_seg_quality.check_mask`` — the
    combination validated on April 22 against the hand-labeled control
    (gemini_li_converted rejects ~4.5%, bad pseudo-labels 10-18%):

    - default: min_regions=4, max_single_region=0.70, min_foreground=0.05,
      missing_head, missing_torso
    - head_below_torso: anatomically impossible (15% of flux_diverse_clean)
    - bg_bleed: labels painted outside character silhouette
    - silhouette_coverage: labels cover too little of character

    If drop_rejected, failing examples are removed entirely from output_dir.
    Otherwise they stay on disk but have `quality_filter_reasons` appended
    to their metadata.json (and `quality_filter_passed: false`).
    """
    from scripts.filter_seg_quality import check_mask

    examples = sorted(
        d for d in output_dir.iterdir()
        if d.is_dir() and (d / "segmentation.png").exists()
    )

    rejected: dict[str, list[str]] = {}
    passed = 0
    for ex in examples:
        seg_path = ex / "segmentation.png"
        reasons = check_mask(
            seg_path,
            min_regions=4,
            max_single_region=0.70,
            min_foreground=0.05,
            skip_anatomy=False,
            drop_head_below_torso=drop_head_below_torso,
            max_bg_bleed=max_bg_bleed,
            min_silhouette_coverage=min_silhouette_coverage,
        )

        # Always stamp metadata with filter outcome
        meta_path = ex / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            meta = {"id": ex.name, "source": "gemini_diverse"}
        meta["quality_filter_passed"] = not reasons
        if reasons:
            meta["quality_filter_reasons"] = reasons
        else:
            meta.pop("quality_filter_reasons", None)
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        if reasons:
            rejected[ex.name] = reasons
            if drop_rejected:
                import shutil
                shutil.rmtree(ex)
        else:
            passed += 1

    total = passed + len(rejected)
    logger.info(
        "Quality filter: %d passed, %d rejected (%.1f%%)%s",
        passed, len(rejected),
        (len(rejected) / total * 100) if total > 0 else 0.0,
        " — deleted from disk" if drop_rejected else " — marked in metadata",
    )

    # Save summary for downstream consumers
    summary = {
        "total": total,
        "passed": passed,
        "rejected": len(rejected),
        "drop_rejected": drop_rejected,
        "thresholds": {
            "drop_head_below_torso": drop_head_below_torso,
            "max_bg_bleed": max_bg_bleed,
            "min_silhouette_coverage": min_silhouette_coverage,
        },
        "rejected_examples": rejected,
    }
    (output_dir / "ingest_quality_filter.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )

    return rejected


def print_quality_report(
    confidence_scores: dict[str, float],
    output_dir: Path,
) -> None:
    """Step 3: Print confidence report and flag low-quality examples."""
    if not confidence_scores:
        print("\nNo examples to report on.")
        return

    scores = list(confidence_scores.values())
    scores_arr = np.array(scores)

    print("\n" + "=" * 60)
    print("  Quality Report — Gemini Pseudo-Labels")
    print("=" * 60)
    print(f"  Total examples:     {len(scores)}")
    print(f"  Mean confidence:    {scores_arr.mean():.4f}")
    print(f"  Median confidence:  {float(np.median(scores_arr)):.4f}")
    print(f"  Min confidence:     {scores_arr.min():.4f}")
    print(f"  Max confidence:     {scores_arr.max():.4f}")
    print()

    # Flag low-confidence examples
    LOW_THRESHOLD = 0.5
    VERY_LOW_THRESHOLD = 0.3

    low = {k: v for k, v in confidence_scores.items() if v < LOW_THRESHOLD}
    very_low = {k: v for k, v in confidence_scores.items() if v < VERY_LOW_THRESHOLD}

    if very_low:
        print(f"  VERY LOW confidence (<{VERY_LOW_THRESHOLD}) — likely bad labels:")
        for name, score in sorted(very_low.items(), key=lambda x: x[1]):
            print(f"    {score:.4f}  {name}")
        print()

    if low and len(low) > len(very_low):
        remaining_low = {k: v for k, v in low.items() if k not in very_low}
        print(f"  LOW confidence (<{LOW_THRESHOLD}) — review recommended:")
        for name, score in sorted(remaining_low.items(), key=lambda x: x[1]):
            print(f"    {score:.4f}  {name}")
        print()

    good = len(scores) - len(low)
    print(f"  Summary: {good} good, {len(low) - len(very_low)} review, {len(very_low)} likely bad")
    print()

    # Save report as JSON
    report = {
        "total": len(scores),
        "mean_confidence": round(float(scores_arr.mean()), 4),
        "median_confidence": round(float(np.median(scores_arr)), 4),
        "low_confidence_examples": {
            k: round(v, 4) for k, v in sorted(low.items(), key=lambda x: x[1])
        },
        "very_low_confidence_examples": {
            k: round(v, 4) for k, v in sorted(very_low.items(), key=lambda x: x[1])
        },
    }
    report_path = output_dir / "quality_report.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(f"  Report saved to {report_path}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Gemini images + pseudo-label with trained seg model",
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Directory with raw Gemini PNG images",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for Strata-formatted examples",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("checkpoints/segmentation/best.pt"),
        help="Path to trained segmentation checkpoint",
    )
    parser.add_argument(
        "--only-new", action="store_true",
        help="Skip already-processed images",
    )
    parser.add_argument(
        "--no-seg", action="store_true",
        help="Skip pseudo-labeling (ingest only, no checkpoint needed)",
    )
    parser.add_argument(
        "--no-quality-filter", action="store_true",
        help="Skip the post-pseudo-label quality filter step. By default, ingested "
             "examples go through the same validated filter combo as Run 29 "
             "(drop-head-below-torso + max-bg-bleed + min-silhouette-coverage). "
             "Failing examples get quality_filter_passed=false in metadata; add "
             "--drop-rejected to delete them from disk instead.",
    )
    parser.add_argument(
        "--drop-rejected", action="store_true",
        help="When quality filter flags an example, delete its directory entirely. "
             "Without this flag, failing examples stay on disk with a metadata marker "
             "so you can audit / recover them.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    start = time.monotonic()

    # Step 1: Ingest
    print("\n[1/2] Ingesting Gemini images...")
    count = run_ingest(args.input_dir, args.output_dir, only_new=args.only_new)

    if args.no_seg:
        print(f"\nDone. {count} images ingested (pseudo-labeling skipped).")
        return

    # Step 2: Pseudo-label
    if not args.checkpoint.exists():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        logger.error("Run with --no-seg to skip pseudo-labeling, or provide a valid checkpoint.")
        sys.exit(1)

    print("\n[2/3] Running segmentation pseudo-labels...")
    confidence_scores = run_pseudo_labels(
        args.output_dir, args.checkpoint, only_new=args.only_new,
    )

    # Step 3: Confidence-based quality report (printed + saved)
    print_quality_report(confidence_scores, args.output_dir)

    # Step 4: Anatomy/spatial quality filter (validated April 22 combo)
    if not args.no_quality_filter:
        print("\n[3/3] Running anatomy/spatial quality filter...")
        run_quality_filter(
            args.output_dir,
            drop_rejected=args.drop_rejected,
        )
    else:
        print("\n[3/3] Quality filter skipped (--no-quality-filter).")

    elapsed = time.monotonic() - start
    print(f"\nTotal elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
