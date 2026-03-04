"""Validate Mixamo renders in the Hetzner bucket for broken/stretched meshes.

Downloads each image from segmentation/images/, checks for corruption
indicators (low fill ratio, extreme aspect ratio, tiny foreground), and
outputs a report of bad renders to remove.

Usage::

    python scripts/validate_renders.py
    python scripts/validate_renders.py --delete  # actually delete bad renders from bucket
    python scripts/validate_renders.py --resume   # skip already-validated images

Requires: boto3, Pillow, python-dotenv
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from PIL import Image

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0  # seconds, doubles each retry

# ---------------------------------------------------------------------------
# Thresholds (calibrated from known good/bad examples)
# ---------------------------------------------------------------------------

MIN_FG_PERCENT = 3.0       # Minimum foreground pixels as % of image
MIN_FILL_RATIO = 0.15      # Minimum fg_pixels / bounding_box_area
MAX_ASPECT_RATIO = 3.5     # Maximum bounding box height/width ratio
MIN_FG_PIXELS = 5000       # Absolute minimum foreground pixel count


def check_image(img_bytes: bytes) -> dict:
    """Analyze a render image for corruption indicators.

    Args:
        img_bytes: Raw PNG bytes.

    Returns:
        Dict with metrics and ``is_bad`` flag.
    """
    img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGBA"))
    alpha = img[:, :, 3]
    fg_mask = alpha > 0
    fg_count = int(fg_mask.sum())
    total = fg_mask.size
    fg_percent = 100 * fg_count / total

    result = {
        "fg_pixels": fg_count,
        "fg_percent": round(fg_percent, 2),
        "is_bad": False,
        "reasons": [],
    }

    if fg_count < MIN_FG_PIXELS:
        result["is_bad"] = True
        result["reasons"].append(f"fg_pixels={fg_count} < {MIN_FG_PIXELS}")
        return result

    if fg_percent < MIN_FG_PERCENT:
        result["is_bad"] = True
        result["reasons"].append(f"fg_percent={fg_percent:.1f}% < {MIN_FG_PERCENT}%")

    # Bounding box analysis
    rows = np.any(fg_mask, axis=1)
    cols = np.any(fg_mask, axis=0)
    rmin, rmax = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    cmin, cmax = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    bb_h = rmax - rmin + 1
    bb_w = cmax - cmin + 1
    aspect = bb_h / max(bb_w, 1)
    fill = fg_count / max(bb_h * bb_w, 1)

    result["bbox"] = [cmin, rmin, cmax, rmax]
    result["bbox_size"] = [bb_w, bb_h]
    result["aspect_ratio"] = round(aspect, 2)
    result["fill_ratio"] = round(fill, 3)

    if fill < MIN_FILL_RATIO:
        result["is_bad"] = True
        result["reasons"].append(f"fill_ratio={fill:.3f} < {MIN_FILL_RATIO}")

    if aspect > MAX_ASPECT_RATIO:
        result["is_bad"] = True
        result["reasons"].append(f"aspect_ratio={aspect:.2f} > {MAX_ASPECT_RATIO}")

    return result


def s3_get_with_retry(s3, bucket: str, key: str) -> bytes:
    """Download an object from S3 with exponential backoff retry."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = s3.get_object(Bucket=bucket, Key=key)
            return resp["Body"].read()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning("S3 error on %s (attempt %d/%d): %s — retrying in %.0fs",
                           key.split("/")[-1], attempt + 1, MAX_RETRIES, e, delay)
            time.sleep(delay)
    return b""  # unreachable


def format_eta(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def print_progress(current: int, total: int, bad_count: int, start_time: float) -> None:
    """Print a single-line progress bar to stderr."""
    elapsed = time.time() - start_time
    pct = current / total
    rate = current / elapsed if elapsed > 0 else 0
    eta = (total - current) / rate if rate > 0 else 0
    bar_len = 30
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)
    line = f"\r  {bar} {current}/{total} ({pct:.0%}) | {bad_count} bad | {rate:.1f} img/s | ETA {format_eta(eta)}  "
    sys.stderr.write(line)
    sys.stderr.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Mixamo renders in bucket")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete bad renders from bucket (images + masks + joints + draw_order)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images already present in existing report",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="validation_report.json",
        help="Path to save validation report JSON",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler("validate_renders.log"),
            logging.StreamHandler(sys.stderr),
        ],
    )

    load_dotenv()
    import boto3
    from botocore.config import Config

    s3 = boto3.client(
        "s3",
        endpoint_url="https://fsn1.your-objectstorage.com",
        aws_access_key_id=os.getenv("BUCKET_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("BUCKET_SECRET"),
        config=Config(retries={"max_attempts": 3, "mode": "adaptive"}),
    )
    bucket = "strata-training-data"

    # List all images
    logger.info("Listing segmentation/images/ ...")
    paginator = s3.get_paginator("list_objects_v2")
    image_keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix="segmentation/images/"):
        for obj in page.get("Contents", []):
            image_keys.append(obj["Key"])

    logger.info("Found %d images to validate", len(image_keys))

    # Load existing report for resume
    already_checked: set[str] = set()
    report_path = Path(args.output)
    prior_good: list[dict] = []
    prior_bad: list[dict] = []
    if args.resume and report_path.exists():
        prior = json.loads(report_path.read_text())
        for entry in prior.get("bad_files", []):
            already_checked.add(entry["filename"])
            prior_bad.append(entry)
        for entry in prior.get("good_files", []):
            already_checked.add(entry["filename"])
            prior_good.append(entry)
        logger.info("Resuming: %d already validated, %d remaining",
                     len(already_checked), len(image_keys) - len(already_checked))

    good = list(prior_good)
    bad = list(prior_bad)
    start_time = time.time()
    checked = 0
    to_check = [k for k in image_keys if k.split("/")[-1] not in already_checked]
    total = len(to_check)

    if total == 0 and not (args.delete and bad):
        logger.info("Nothing to validate — all %d images already checked.", len(image_keys))
        return
    elif total == 0:
        logger.info("Nothing to validate — all %d images already checked. Proceeding to delete %d bad.",
                     len(image_keys), len(bad))

    if total > 0:
        logger.info("Validating %d images ...", total)

        for i, key in enumerate(to_check):
            checked = i + 1
            if checked % 10 == 0 or checked == total:
                print_progress(checked, total, len(bad), start_time)

            img_bytes = s3_get_with_retry(s3, bucket, key)
            result = check_image(img_bytes)
            result["key"] = key
            result["filename"] = key.split("/")[-1]

            if result["is_bad"]:
                bad.append(result)
                logger.warning("BAD: %s — %s", result["filename"], ", ".join(result["reasons"]))
            else:
                good.append(result)

            # Save incremental report every 200 images
            if checked % 200 == 0:
                _save_report(report_path, image_keys, good, bad)

        sys.stderr.write("\n")  # newline after progress bar

    # Final report
    logger.info("=" * 60)
    logger.info("Validation complete: %d good, %d bad out of %d total",
                len(good), len(bad), len(image_keys))

    # Group bad by character
    bad_chars: dict[str, list[str]] = {}
    for r in bad:
        stem = r["filename"].replace(".png", "").replace("_flat", "")
        parts = stem.split("_pose_")
        char = parts[0] if parts else stem
        bad_chars.setdefault(char, []).append(r["filename"])

    if bad_chars:
        logger.info("Bad renders by character:")
        for char in sorted(bad_chars):
            logger.info("  %s: %d bad poses", char, len(bad_chars[char]))

    _save_report(report_path, image_keys, good, bad)
    logger.info("Report saved to %s", report_path)

    # Delete if requested
    if args.delete and bad:
        logger.info("Deleting %d bad renders + associated masks/joints/draw_order ...", len(bad))
        deleted = 0
        for r in bad:
            stem = r["filename"].replace("_flat.png", "")
            for subdir, ext in [
                ("images", "_flat.png"),
                ("masks", ".png"),
                ("joints", ".json"),
                ("draw_order", ".png"),
            ]:
                del_key = f"segmentation/{subdir}/{stem}{ext}"
                try:
                    s3.delete_object(Bucket=bucket, Key=del_key)
                    deleted += 1
                except Exception:
                    pass  # File may not exist in all subdirs
        logger.info("Deleted %d objects from bucket", deleted)


def _save_report(
    report_path: Path,
    image_keys: list[str],
    good: list[dict],
    bad: list[dict],
) -> None:
    """Save validation report to disk."""
    bad_chars: dict[str, list[str]] = {}
    for r in bad:
        stem = r["filename"].replace(".png", "").replace("_flat", "")
        parts = stem.split("_pose_")
        char = parts[0] if parts else stem
        bad_chars.setdefault(char, []).append(r["filename"])

    report = {
        "total": len(image_keys),
        "good": len(good),
        "bad": len(bad),
        "bad_characters": {k: len(v) for k, v in sorted(bad_chars.items())},
        "bad_files": [{"filename": r["filename"], "reasons": r["reasons"]} for r in bad],
        "good_files": [{"filename": r["filename"]} for r in good],
    }
    report_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
