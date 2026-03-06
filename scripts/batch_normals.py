#!/usr/bin/env python3
"""Batch normal map generation using StableNormal.

Downloads 2D-origin dataset images from Hetzner bucket, runs StableNormal
inference, filters out low-quality results, and uploads normal maps + metadata
back to the bucket.

Usage:
    python scripts/batch_normals.py --dataset curated_diverse --batch-size 4
    python scripts/batch_normals.py --dataset all --batch-size 8
    python scripts/batch_normals.py --dataset anime_seg --max-examples 100  # test run

Requires: GPU with CUDA, StableNormal installed
    pip install git+https://github.com/Stable-X/StableNormal.git
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Datasets that are 2D-origin and contain character images worth processing
DATASETS_2D = [
    "curated_diverse",
    "anime_seg",
    "anime_instance_seg",
    "animerun",
    "conr",
    "fbanimehq",
    "instaorder",
    "live2d",
]

BUCKET = "hetzner:strata-training-data"
RCLONE_FLAGS = ["--transfers", "16", "--checkers", "32", "--fast-list", "--size-only"]

# Quality filter thresholds
# Normal map where the dominant color covers >80% of foreground is likely failed
# (e.g. flat purple = [128, 128, 255] means "everything faces camera" = no 3D info)
MAX_DOMINANT_RATIO = 0.80
# Minimum standard deviation across normal channels in foreground pixels
# Low std = flat/uniform normals = model didn't detect 3D structure
MIN_NORMAL_STD = 15.0
# Minimum foreground pixel count (skip tiny characters)
MIN_FG_PIXELS = 500


def rclone_cmd(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["rclone"] + args
    logger.debug("Running: %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def list_examples(dataset: str) -> list[str]:
    """List all example subdirectories in a dataset prefix."""
    result = rclone_cmd(["lsf", "--dirs-only", f"{BUCKET}/{dataset}/"])
    dirs = [d.rstrip("/") for d in result.stdout.strip().split("\n") if d.strip()]
    return dirs


def already_has_normals(dataset: str, example_id: str) -> bool:
    """Check if an example already has a normals.png."""
    result = rclone_cmd(
        ["ls", f"{BUCKET}/{dataset}/{example_id}/normals.png"], check=False
    )
    return result.returncode == 0 and "normals.png" in result.stdout


def download_image(dataset: str, example_id: str, local_dir: Path) -> Path | None:
    """Download just the image.png for one example."""
    local_path = local_dir / f"{example_id}_image.png"
    result = rclone_cmd(
        ["copyto", f"{BUCKET}/{dataset}/{example_id}/image.png", str(local_path)],
        check=False,
    )
    if result.returncode != 0 or not local_path.exists():
        return None
    return local_path


def upload_files(dataset: str, example_id: str, files: dict[str, Path]) -> bool:
    """Upload multiple files to a bucket example directory."""
    ok = True
    for remote_name, local_path in files.items():
        result = rclone_cmd(
            ["copyto", str(local_path), f"{BUCKET}/{dataset}/{example_id}/{remote_name}"],
            check=False,
        )
        if result.returncode != 0:
            ok = False
    return ok


def load_model(turbo: bool = True):
    """Load StableNormal predictor."""
    variant = "StableNormal_turbo" if turbo else "StableNormal"
    logger.info("Loading %s model...", variant)
    predictor = torch.hub.load("Stable-X/StableNormal", variant, trust_repo=True)
    logger.info("Model loaded.")
    return predictor


def assess_normal_quality(
    normal_arr: np.ndarray, source_img: Image.Image
) -> dict:
    """Assess normal map quality and extract metadata.

    Returns a dict with quality metrics and derived data. The 'pass' key
    indicates whether the normal map meets quality thresholds.
    """
    # Build foreground mask from source image alpha channel
    if source_img.mode == "RGBA":
        alpha = np.array(source_img)[:, :, 3]
        fg_mask = alpha > 10
    else:
        # No alpha — use normal map background detection
        # Background in StableNormal output is typically [128, 128, 255] (flat blue)
        bg_color = np.array([128, 128, 255])
        diff = np.abs(normal_arr.astype(float) - bg_color.astype(float)).sum(axis=2)
        fg_mask = diff > 30

    fg_pixels = fg_mask.sum()
    total_pixels = fg_mask.size

    if fg_pixels < MIN_FG_PIXELS:
        return {
            "pass": False,
            "reject_reason": "too_few_fg_pixels",
            "fg_pixels": int(fg_pixels),
        }

    # Extract foreground normals
    fg_normals = normal_arr[fg_mask]  # shape (N, 3)

    # Check dominant color ratio — quantize to 4-bit per channel to group similar colors
    quantized = (fg_normals // 16).astype(np.uint32)
    keys = quantized[:, 0] * 256 + quantized[:, 1] * 16 + quantized[:, 2]
    _, counts = np.unique(keys, return_counts=True)
    dominant_ratio = float(counts.max()) / fg_pixels

    # Channel-wise standard deviation across foreground
    std_r = float(fg_normals[:, 0].std())
    std_g = float(fg_normals[:, 1].std())
    std_b = float(fg_normals[:, 2].std())
    mean_std = (std_r + std_g + std_b) / 3.0

    # Mean normal direction (normalized to [-1, 1] from [0, 255])
    mean_normal = fg_normals.mean(axis=0).tolist()
    mean_normal_normalized = [round((v / 127.5) - 1.0, 4) for v in mean_normal]

    # Facing ratio: how much of the character faces the camera (Z component > 0.5)
    z_channel = fg_normals[:, 2].astype(float) / 255.0
    front_facing_ratio = float((z_channel > 0.6).sum()) / fg_pixels

    passed = dominant_ratio <= MAX_DOMINANT_RATIO and mean_std >= MIN_NORMAL_STD
    reject_reason = ""
    if not passed:
        reasons = []
        if dominant_ratio > MAX_DOMINANT_RATIO:
            reasons.append(f"dominant_color={dominant_ratio:.2f}")
        if mean_std < MIN_NORMAL_STD:
            reasons.append(f"low_std={mean_std:.1f}")
        reject_reason = "; ".join(reasons)

    return {
        "pass": passed,
        "reject_reason": reject_reason,
        "fg_pixels": int(fg_pixels),
        "fg_ratio": round(fg_pixels / total_pixels, 4),
        "dominant_color_ratio": round(dominant_ratio, 4),
        "std_rgb": [round(std_r, 2), round(std_g, 2), round(std_b, 2)],
        "mean_std": round(mean_std, 2),
        "mean_normal": mean_normal_normalized,
        "front_facing_ratio": round(front_facing_ratio, 4),
    }


def predict_and_filter(
    predictor,
    image_path: Path,
    output_dir: Path,
    example_id: str,
) -> tuple[Path | None, Path | None, dict]:
    """Run normal prediction, assess quality, save if passes.

    Returns (normals_path, meta_path, quality_dict).
    normals_path is None if quality check failed.
    """
    img = Image.open(image_path)
    img_rgb = img.convert("RGB")

    normal_img = predictor(img_rgb)
    normal_arr = np.array(normal_img)

    quality = assess_normal_quality(normal_arr, img)
    quality["example_id"] = example_id

    if not quality["pass"]:
        return None, None, quality

    # Save normal map
    normals_path = output_dir / f"{example_id}_normals.png"
    normal_img.save(normals_path, format="PNG")

    # Save metadata
    meta_path = output_dir / f"{example_id}_normals_meta.json"
    with open(meta_path, "w") as f:
        json.dump(quality, f, indent=2)

    return normals_path, meta_path, quality


def process_dataset(
    predictor,
    dataset: str,
    max_examples: int | None = None,
    batch_size: int = 4,
    skip_existing: bool = True,
):
    """Process all examples in a dataset."""
    logger.info("=== Processing dataset: %s ===", dataset)

    examples = list_examples(dataset)
    logger.info("Found %d examples in %s", len(examples), dataset)

    if max_examples:
        examples = examples[:max_examples]
        logger.info("Limiting to %d examples", max_examples)

    processed = 0
    skipped = 0
    filtered = 0
    failed = 0

    with tempfile.TemporaryDirectory(prefix=f"normals_{dataset}_") as tmpdir:
        tmp = Path(tmpdir)
        dl_dir = tmp / "downloads"
        out_dir = tmp / "normals"
        dl_dir.mkdir()
        out_dir.mkdir()

        for batch_start in range(0, len(examples), batch_size):
            batch_ids = examples[batch_start : batch_start + batch_size]

            # Filter out examples that already have normals
            if skip_existing:
                todo_ids = []
                for eid in batch_ids:
                    if already_has_normals(dataset, eid):
                        skipped += 1
                    else:
                        todo_ids.append(eid)
                batch_ids = todo_ids

            if not batch_ids:
                continue

            for eid in batch_ids:
                # Download
                img_path = download_image(dataset, eid, dl_dir)
                if not img_path:
                    logger.warning("Could not download image for %s/%s", dataset, eid)
                    failed += 1
                    continue

                # Predict + filter
                try:
                    normals_path, meta_path, quality = predict_and_filter(
                        predictor, img_path, out_dir, eid
                    )
                except Exception as e:
                    logger.warning("Inference failed for %s/%s: %s", dataset, eid, e)
                    failed += 1
                    img_path.unlink(missing_ok=True)
                    continue

                if not quality["pass"]:
                    logger.debug(
                        "Filtered %s/%s: %s", dataset, eid, quality["reject_reason"]
                    )
                    filtered += 1
                    img_path.unlink(missing_ok=True)
                    continue

                # Upload normals.png + normals_meta.json
                to_upload = {"normals.png": normals_path}
                if meta_path:
                    to_upload["normals_meta.json"] = meta_path

                if upload_files(dataset, eid, to_upload):
                    processed += 1
                else:
                    logger.warning("Failed to upload for %s/%s", dataset, eid)
                    failed += 1

                # Clean up
                img_path.unlink(missing_ok=True)
                if normals_path:
                    normals_path.unlink(missing_ok=True)
                if meta_path:
                    meta_path.unlink(missing_ok=True)

            total_done = processed + skipped + filtered + failed
            if total_done % 100 < batch_size:
                logger.info(
                    "[%s] Progress: %d processed, %d filtered, %d skipped, %d failed / %d total",
                    dataset,
                    processed,
                    filtered,
                    skipped,
                    failed,
                    len(examples),
                )

    logger.info(
        "[%s] Done: %d processed, %d filtered out, %d skipped, %d failed",
        dataset,
        processed,
        filtered,
        skipped,
        failed,
    )
    return processed, filtered, skipped, failed


def main():
    parser = argparse.ArgumentParser(description="Batch normal map generation with StableNormal")
    parser.add_argument(
        "--dataset",
        default="all",
        help=f"Dataset to process, or 'all' for all 2D datasets. Options: {', '.join(DATASETS_2D)}",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Images per batch")
    parser.add_argument("--max-examples", type=int, default=None, help="Max examples per dataset (for testing)")
    parser.add_argument("--no-skip-existing", action="store_true", help="Re-process examples that already have normals")
    parser.add_argument("--no-turbo", action="store_true", help="Use full StableNormal instead of turbo variant")
    args = parser.parse_args()

    datasets = DATASETS_2D if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        if ds not in DATASETS_2D:
            logger.error("Unknown dataset: %s. Options: %s", ds, ", ".join(DATASETS_2D))
            sys.exit(1)

    predictor = load_model(turbo=not args.no_turbo)

    total_processed = 0
    total_filtered = 0
    total_skipped = 0
    total_failed = 0

    for ds in datasets:
        p, f, s, e = process_dataset(
            predictor,
            ds,
            max_examples=args.max_examples,
            batch_size=args.batch_size,
            skip_existing=not args.no_skip_existing,
        )
        total_processed += p
        total_filtered += f
        total_skipped += s
        total_failed += e

    logger.info(
        "=== ALL DONE === %d processed, %d filtered out, %d skipped, %d failed",
        total_processed,
        total_filtered,
        total_skipped,
        total_failed,
    )


if __name__ == "__main__":
    main()
