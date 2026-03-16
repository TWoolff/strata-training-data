#!/usr/bin/env python3
"""Batch pseudo-label images with the segmentation model for manual correction.

Runs the trained seg model on a directory of images, produces per-example
subdirectories with image.png + segmentation.png + metadata.json, and writes
a review_manifest.json for the correction UI.

Usage::

    # From per-example subdirs (e.g. gemini preprocessed)
    python scripts/batch_pseudo_label.py \
        --input-dir /Volumes/TAMWoolff/data/preprocessed/gemini/ \
        --output-dir ./output/gemini_corrected \
        --checkpoint checkpoints/segmentation/best.pt

    # From flat directory of loose images
    python scripts/batch_pseudo_label.py \
        --input-dir ./data/images/ \
        --output-dir ./output/labeled \
        --checkpoint checkpoints/segmentation/best.pt \
        --flat

    # Skip already-processed examples
    python scripts/batch_pseudo_label.py \
        --input-dir ... --output-dir ... --checkpoint ... --only-missing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pipeline.config import RENDER_RESOLUTION
from run_seg_enrich import load_seg_model, predict_segmentation

logger = logging.getLogger(__name__)


def load_manifest(output_dir: Path) -> dict:
    """Load or create review_manifest.json."""
    manifest_path = output_dir / "review_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"total": 0, "reviewed": 0, "rejected": 0, "needs_review": 0, "examples": {}}


def save_manifest(output_dir: Path, manifest: dict) -> None:
    """Save review_manifest.json with updated counts."""
    manifest["total"] = len(manifest["examples"])
    manifest["reviewed"] = sum(1 for e in manifest["examples"].values() if e["status"] == "reviewed")
    manifest["rejected"] = sum(1 for e in manifest["examples"].values() if e["status"] == "rejected")
    manifest["needs_review"] = sum(1 for e in manifest["examples"].values() if e["status"] == "needs_review")
    path = output_dir / "review_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def process_image(
    model,
    image_path: Path,
    example_dir: Path,
    device: torch.device,
    resolution: int,
) -> bool:
    """Run inference on one image and save to example_dir."""
    example_dir.mkdir(parents=True, exist_ok=True)

    # Copy/resize source image
    img = Image.open(image_path).convert("RGBA")
    img_resized = img.resize((resolution, resolution), Image.LANCZOS)
    img_resized.save(example_dir / "image.png")

    # Run segmentation
    seg_mask, draw_order, confidence = predict_segmentation(
        model, example_dir / "image.png", device, resolution
    )

    # Zero out background where alpha is transparent
    alpha = np.array(img_resized)[:, :, 3]
    seg_mask[alpha < 10] = 0

    Image.fromarray(seg_mask).save(example_dir / "segmentation.png")

    # Metadata
    meta = {
        "id": example_dir.name,
        "source": "pseudo_label",
        "source_filename": image_path.name,
        "resolution": resolution,
        "has_segmentation_mask": True,
        "segmentation_source": "pseudo_label",
        "review_status": "needs_review",
        "regions": sorted(int(r) for r in np.unique(seg_mask) if r > 0),
    }
    (example_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch pseudo-label images with seg model for manual correction."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("checkpoints/segmentation/best.pt"),
    )
    parser.add_argument("--resolution", type=int, default=RENDER_RESOLUTION)
    parser.add_argument("--flat", action="store_true",
                        help="Input is a flat directory of images.")
    parser.add_argument("--only-missing", action="store_true",
                        help="Skip examples that already have segmentation.")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Auto-detect device
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    model = load_seg_model(args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.output_dir)

    # Collect images to process
    if args.flat:
        images = sorted(
            p for p in args.input_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
            and not p.name.startswith(".")
        )
        pairs = [(img, args.output_dir / img.stem) for img in images]
    else:
        examples = sorted(
            d for d in args.input_dir.iterdir()
            if d.is_dir() and (d / "image.png").exists()
            and not d.name.startswith(".")
        )
        pairs = [(d / "image.png", args.output_dir / d.name) for d in examples]

    total = len(pairs)
    logger.info("Found %d images to process", total)

    start = time.monotonic()
    processed = 0
    skipped = 0

    for i, (image_path, example_dir) in enumerate(pairs):
        if args.only_missing and (example_dir / "segmentation.png").exists():
            skipped += 1
            continue

        ok = process_image(model, image_path, example_dir, device, args.resolution)
        if ok:
            manifest["examples"][example_dir.name] = {"status": "needs_review"}
            processed += 1

        if (i + 1) % 50 == 0 or (i + 1) == total:
            elapsed = time.monotonic() - start
            speed = processed / elapsed if elapsed > 0 else 0
            logger.info(
                "Progress: %d/%d — %d processed, %d skipped (%.1f img/s)",
                i + 1, total, processed, skipped, speed,
            )

    save_manifest(args.output_dir, manifest)

    elapsed = time.monotonic() - start
    print(f"\nBatch pseudo-labeling complete:")
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Total:     {total}")
    print(f"  Elapsed:   {elapsed:.1f}s")
    print(f"  Manifest:  {args.output_dir / 'review_manifest.json'}")
    print(f"\nNext: python scripts/review_masks.py --data-dir {args.output_dir}")


if __name__ == "__main__":
    main()
