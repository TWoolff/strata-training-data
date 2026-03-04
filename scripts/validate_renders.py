"""Validate Mixamo renders in the Hetzner bucket for broken/stretched meshes.

Downloads each image from segmentation/images/, checks for corruption
indicators (low fill ratio, extreme aspect ratio, tiny foreground), and
outputs a report of bad renders to remove.

Usage::

    python scripts/validate_renders.py
    python scripts/validate_renders.py --delete  # actually delete bad renders from bucket

Requires: boto3, Pillow, python-dotenv
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from PIL import Image

logger = logging.getLogger(__name__)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Mixamo renders in bucket")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete bad renders from bucket (images + masks + joints + draw_order)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="validation_report.json",
        help="Path to save validation report JSON",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    load_dotenv()
    import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url="https://fsn1.your-objectstorage.com",
        aws_access_key_id=os.getenv("BUCKET_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("BUCKET_SECRET"),
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

    good = []
    bad = []

    for i, key in enumerate(image_keys):
        if (i + 1) % 100 == 0:
            logger.info("Progress: %d/%d (%d bad so far)", i + 1, len(image_keys), len(bad))

        resp = s3.get_object(Bucket=bucket, Key=key)
        img_bytes = resp["Body"].read()
        result = check_image(img_bytes)
        result["key"] = key
        result["filename"] = key.split("/")[-1]

        if result["is_bad"]:
            bad.append(result)
            logger.warning("BAD: %s — %s", result["filename"], ", ".join(result["reasons"]))
        else:
            good.append(result)

    # Report
    logger.info("=" * 60)
    logger.info("Validation complete: %d good, %d bad out of %d total", len(good), len(bad), len(image_keys))

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

    # Save report
    report = {
        "total": len(image_keys),
        "good": len(good),
        "bad": len(bad),
        "bad_characters": {k: len(v) for k, v in sorted(bad_chars.items())},
        "bad_files": [{"filename": r["filename"], "reasons": r["reasons"]} for r in bad],
    }
    report_path = Path(args.output)
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Report saved to %s", report_path)

    # Delete if requested
    if args.delete and bad:
        logger.info("Deleting %d bad renders + associated masks/joints/draw_order ...", len(bad))
        deleted = 0
        for r in bad:
            stem = r["filename"].replace("_flat.png", "")
            # Delete from all subdirectories
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


if __name__ == "__main__":
    main()
