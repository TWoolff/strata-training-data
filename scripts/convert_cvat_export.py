#!/usr/bin/env python3
"""Convert CVAT Segmentation Mask 1.1 export to Strata training format.

CVAT exports:
    export.zip/
    ├── labelmap.txt           # label_name:R,G,B::
    ├── SegmentationClass/     # RGB color masks (one per image)
    │   ├── image1.png
    │   └── ...
    └── (other files we ignore)

Strata training format (per-example dirs):
    output_dir/
    ├── {example_id}/
    │   ├── image.png          # 512×512 RGBA
    │   └── segmentation.png   # 8-bit grayscale, pixel value = region ID (0-21)
    └── ...

Usage:
    # After exporting from CVAT as "Segmentation mask 1.1":
    python scripts/convert_cvat_export.py \
        --cvat-export ./cvat_export.zip \
        --source-images /tmp/cvat_annotation_batch/ \
        --output-dir /Volumes/TAMWoolff/data/preprocessed/cvat_annotated/

    # Or if already unzipped:
    python scripts/convert_cvat_export.py \
        --cvat-export ./cvat_export_dir/ \
        --source-images /tmp/cvat_annotation_batch/ \
        --output-dir /Volumes/TAMWoolff/data/preprocessed/cvat_annotated/
"""

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

# Strata region name → ID mapping (from pipeline/config.py)
REGION_NAME_TO_ID = {
    "background": 0,
    "head": 1,
    "neck": 2,
    "chest": 3,
    "spine": 4,
    "hips": 5,
    "shoulder_l": 6,
    "upper_arm_l": 7,
    "forearm_l": 8,
    "hand_l": 9,
    "shoulder_r": 10,
    "upper_arm_r": 11,
    "forearm_r": 12,
    "hand_r": 13,
    "upper_leg_l": 14,
    "lower_leg_l": 15,
    "foot_l": 16,
    "upper_leg_r": 17,
    "lower_leg_r": 18,
    "foot_r": 19,
    "accessory": 20,
    "hair_back": 21,
}


def parse_labelmap(labelmap_path: Path) -> dict[tuple[int, int, int], str]:
    """Parse CVAT labelmap.txt → {(R,G,B): label_name}."""
    color_to_name = {}
    with open(labelmap_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Format: label_name:R,G,B::'part':'action'
            # or:     label_name:R,G,B::
            parts = line.split(":")
            if len(parts) < 2:
                continue
            name = parts[0].strip()
            color_str = parts[1].strip()
            if not color_str:
                continue
            rgb = tuple(int(c.strip()) for c in color_str.split(","))
            if len(rgb) == 3:
                color_to_name[rgb] = name
    return color_to_name


def rgb_mask_to_region_ids(
    mask_rgb: np.ndarray,
    color_to_region_id: dict[tuple[int, int, int], int],
) -> np.ndarray:
    """Convert RGB color mask to grayscale region ID mask.

    Args:
        mask_rgb: [H, W, 3] uint8 array (RGB color mask from CVAT)
        color_to_region_id: mapping from (R,G,B) → region ID (0-21)

    Returns:
        [H, W] uint8 array with pixel values = region IDs
    """
    h, w = mask_rgb.shape[:2]
    region_mask = np.zeros((h, w), dtype=np.uint8)  # default: background (0)

    for color, region_id in color_to_region_id.items():
        r, g, b = color
        match = (
            (mask_rgb[:, :, 0] == r)
            & (mask_rgb[:, :, 1] == g)
            & (mask_rgb[:, :, 2] == b)
        )
        region_mask[match] = region_id

    return region_mask


def convert_cvat_export(
    cvat_dir: Path,
    source_images_dir: Path,
    output_dir: Path,
    resolution: int = 512,
) -> dict:
    """Convert CVAT export to Strata per-example training format.

    Returns stats dict with counts.
    """
    # Parse labelmap
    labelmap_path = cvat_dir / "labelmap.txt"
    if not labelmap_path.exists():
        print(f"ERROR: {labelmap_path} not found in CVAT export")
        sys.exit(1)

    color_to_name = parse_labelmap(labelmap_path)
    print(f"  Parsed labelmap: {len(color_to_name)} labels")

    # Build color → region ID mapping
    color_to_region_id = {}
    unmapped = []
    for color, name in color_to_name.items():
        if name in REGION_NAME_TO_ID:
            color_to_region_id[color] = REGION_NAME_TO_ID[name]
        else:
            unmapped.append(name)

    if unmapped:
        print(f"  WARNING: unmapped labels (skipped): {unmapped}")

    print(f"  Mapped {len(color_to_region_id)} colors to region IDs")

    # Find mask files
    seg_class_dir = cvat_dir / "SegmentationClass"
    if not seg_class_dir.exists():
        print(f"ERROR: {seg_class_dir} not found")
        sys.exit(1)

    mask_files = sorted(seg_class_dir.glob("*.png"))
    print(f"  Found {len(mask_files)} mask files in SegmentationClass/")

    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"converted": 0, "skipped_no_source": 0, "skipped_empty": 0}

    for mask_path in mask_files:
        stem = mask_path.stem  # e.g., "gemini_diverse_gemini_1sy54t..."

        # Find matching source image
        source_path = source_images_dir / f"{stem}.png"
        if not source_path.exists():
            # Try other extensions
            for ext in [".jpg", ".jpeg", ".webp"]:
                candidate = source_images_dir / f"{stem}{ext}"
                if candidate.exists():
                    source_path = candidate
                    break
            else:
                print(f"  SKIP: no source image for {stem}")
                stats["skipped_no_source"] += 1
                continue

        # Load and convert mask
        mask_img = Image.open(mask_path).convert("RGB")
        mask_rgb = np.array(mask_img)
        region_mask = rgb_mask_to_region_ids(mask_rgb, color_to_region_id)

        # Check if mask has any foreground
        unique_regions = np.unique(region_mask)
        fg_regions = [r for r in unique_regions if r > 0]
        if len(fg_regions) < 2:
            print(f"  SKIP: {stem} has only {len(fg_regions)} foreground regions")
            stats["skipped_empty"] += 1
            continue

        # Load source image
        source_img = Image.open(source_path).convert("RGBA")

        # Resize both to target resolution
        source_img = source_img.resize((resolution, resolution), Image.LANCZOS)
        # Use NEAREST for mask to preserve exact region IDs
        region_mask_img = Image.fromarray(region_mask, mode="L")
        region_mask_img = region_mask_img.resize(
            (resolution, resolution), Image.NEAREST
        )

        # Create per-example directory
        example_dir = output_dir / stem
        example_dir.mkdir(parents=True, exist_ok=True)

        # Save
        source_img.save(example_dir / "image.png")
        region_mask_img.save(example_dir / "segmentation.png")

        # Write metadata
        metadata = {
            "source_type": "cvat_manual",
            "source_file": stem,
            "regions": sorted(int(r) for r in np.unique(np.array(region_mask_img))),
            "annotation_quality": "manual_gt",
        }
        with open(example_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        stats["converted"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert CVAT Segmentation Mask export to Strata training format"
    )
    parser.add_argument(
        "--cvat-export",
        required=True,
        help="Path to CVAT export (.zip file or extracted directory)",
    )
    parser.add_argument(
        "--source-images",
        required=True,
        help="Directory containing the original source images",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for per-example training data",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Output resolution (default: 512)",
    )
    args = parser.parse_args()

    cvat_path = Path(args.cvat_export)
    source_dir = Path(args.source_images)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("  CVAT Export → Strata Training Format Converter")
    print("=" * 60)

    # Handle zip or directory
    if cvat_path.suffix == ".zip":
        tmp_dir = tempfile.mkdtemp(prefix="cvat_export_")
        print(f"  Extracting {cvat_path} to {tmp_dir}...")
        with zipfile.ZipFile(cvat_path, "r") as zf:
            zf.extractall(tmp_dir)
        cvat_dir = Path(tmp_dir)
        # CVAT sometimes nests in a subdirectory
        if not (cvat_dir / "labelmap.txt").exists():
            subdirs = [d for d in cvat_dir.iterdir() if d.is_dir()]
            for sd in subdirs:
                if (sd / "labelmap.txt").exists():
                    cvat_dir = sd
                    break
    else:
        cvat_dir = cvat_path

    print(f"  CVAT export: {cvat_dir}")
    print(f"  Source images: {source_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Resolution: {args.resolution}×{args.resolution}")
    print()

    stats = convert_cvat_export(cvat_dir, source_dir, output_dir, args.resolution)

    print()
    print(f"  Converted: {stats['converted']}")
    print(f"  Skipped (no source): {stats['skipped_no_source']}")
    print(f"  Skipped (empty mask): {stats['skipped_empty']}")
    print()

    if stats["converted"] > 0:
        print(f"  Output ready at: {output_dir}")
        print(f"  To upload: tar cf cvat_annotated.tar -C {output_dir.parent} {output_dir.name}")
        print(f"             rclone copy cvat_annotated.tar hetzner:strata-training-data/tars/ -P")
    else:
        print("  WARNING: No images converted!")

    print("=" * 60)


if __name__ == "__main__":
    main()
