#!/usr/bin/env python3
"""Convert Strata GT segmentation data to COCO JSON format for SAM 3 fine-tuning.

Reads our standard format (image.png + segmentation.png per example) and produces
COCO instance annotations with RLE-encoded masks. Each connected component of
each class becomes a separate annotation.

Usage::

    python scripts/convert_gt_to_coco.py \
        --data-dirs ./data_cloud/humanrig ./data_cloud/vroid_cc0 \
        --output-dir ./data_cloud/sam3_coco \
        --split-file ./data_cloud/frozen_val_test.json

    # Quick test on small sample
    python scripts/convert_gt_to_coco.py \
        --data-dirs ./data_cloud/humanrig \
        --output-dir /tmp/sam3_coco_test \
        --max-images 100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label as connected_components

logger = logging.getLogger(__name__)

# Strata 22-class names (matching pipeline/config.py)
CLASS_NAMES = [
    "background", "head", "neck", "chest", "spine", "hips",
    "shoulder_l", "upper_arm_l", "forearm_l", "hand_l",
    "shoulder_r", "upper_arm_r", "forearm_r", "hand_r",
    "upper_leg_l", "lower_leg_l", "foot_l",
    "upper_leg_r", "lower_leg_r", "foot_r",
    "accessory", "hair_back",
]

# Text prompts for SAM 3 (more descriptive than raw class names)
CLASS_PROMPTS = [
    "background",
    "head",
    "neck",
    "chest",
    "spine",
    "hips",
    "left shoulder",
    "left upper arm",
    "left forearm",
    "left hand",
    "right shoulder",
    "right upper arm",
    "right forearm",
    "right hand",
    "left upper leg",
    "left lower leg",
    "left foot",
    "right upper leg",
    "right lower leg",
    "right foot",
    "accessory",
    "back hair",
]

NUM_CLASSES = 22


def mask_to_rle(binary_mask: np.ndarray) -> dict:
    """Encode a binary mask as COCO RLE format.

    Args:
        binary_mask: [H, W] uint8 array with 0/1 values.

    Returns:
        RLE dict with 'size' and 'counts' keys.
    """
    from pycocotools import mask as mask_util

    # pycocotools expects Fortran-order array
    rle = mask_util.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def process_example(
    image_path: Path,
    seg_path: Path,
    image_id: int,
    ann_id_start: int,
    images_dir: Path,
) -> tuple[dict, list[dict], int]:
    """Process one example into COCO format.

    Returns:
        image_info: dict for COCO images list
        annotations: list of annotation dicts
        next_ann_id: next available annotation ID
    """
    img = Image.open(image_path)
    w, h = img.size

    # Copy image to output images dir
    dst = images_dir / f"{image_id:06d}.png"
    if not dst.exists():
        shutil.copy2(image_path, dst)

    image_info = {
        "id": image_id,
        "file_name": f"{image_id:06d}.png",
        "width": w,
        "height": h,
    }

    # Load segmentation mask
    seg = np.array(Image.open(seg_path).convert("L"))
    if seg.shape != (h, w):
        seg = np.array(Image.open(seg_path).convert("L").resize((w, h), Image.NEAREST))

    annotations = []
    ann_id = ann_id_start

    # Skip class 0 (background) — SAM 3 doesn't need background annotations
    for class_id in range(1, NUM_CLASSES):
        class_mask = (seg == class_id).astype(np.uint8)
        if class_mask.sum() == 0:
            continue

        # Find connected components (instances)
        labeled, n_components = connected_components(class_mask)

        for comp_id in range(1, n_components + 1):
            inst_mask = (labeled == comp_id).astype(np.uint8)
            area = int(inst_mask.sum())

            # Skip tiny components (< 4 pixels)
            if area < 4:
                continue

            # Bounding box [x, y, width, height]
            rows, cols = np.where(inst_mask)
            x_min, x_max = int(cols.min()), int(cols.max())
            y_min, y_max = int(rows.min()), int(rows.max())
            bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]

            # RLE encoding
            rle = mask_to_rle(inst_mask)

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "segmentation": rle,
            })
            ann_id += 1

    return image_info, annotations, ann_id


def main():
    parser = argparse.ArgumentParser(description="Convert Strata GT to COCO JSON for SAM 3")
    parser.add_argument("--data-dirs", type=str, nargs="+", required=True,
                        help="Directories with GT data (image.png + segmentation.png)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for COCO format data")
    parser.add_argument("--split-file", type=str, default=None,
                        help="Frozen val/test split JSON (optional)")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Max images to process (0 = all)")
    parser.add_argument("--quality-filter", action="store_true",
                        help="Only include examples that pass quality filter")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output_dir = Path(args.output_dir)

    # Load frozen splits if provided
    frozen_splits = None
    if args.split_file and Path(args.split_file).exists():
        with open(args.split_file) as f:
            frozen_splits = json.load(f)
        logger.info("Loaded frozen splits: %d val, %d test chars",
                    len(frozen_splits.get("val", [])), len(frozen_splits.get("test", [])))

    # Discover examples
    examples = []
    for data_dir in args.data_dirs:
        data_dir = Path(data_dir)
        if not data_dir.exists():
            logger.warning("Directory not found: %s", data_dir)
            continue

        for ex_dir in sorted(data_dir.iterdir()):
            if not ex_dir.is_dir():
                continue
            image_path = ex_dir / "image.png"
            seg_path = ex_dir / "segmentation.png"
            if not image_path.exists() or not seg_path.exists():
                continue

            # Quality filter
            if args.quality_filter:
                qf_path = data_dir / "quality_filter.json"
                if qf_path.exists():
                    with open(qf_path) as f:
                        qf = json.load(f)
                    if ex_dir.name in qf.get("rejected", {}):
                        continue

            examples.append((image_path, seg_path, ex_dir.name, data_dir.name))

    logger.info("Found %d examples across %d directories", len(examples), len(args.data_dirs))

    if args.max_images > 0:
        examples = examples[:args.max_images]
        logger.info("Capped to %d examples", len(examples))

    # Split into train/val/test
    splits = {"train": [], "val": [], "test": []}
    for img_path, seg_path, ex_name, ds_name in examples:
        char_id = f"{ds_name}/{ex_name}"
        if frozen_splits:
            if char_id in frozen_splits.get("val", []):
                splits["val"].append((img_path, seg_path))
            elif char_id in frozen_splits.get("test", []):
                splits["test"].append((img_path, seg_path))
            else:
                splits["train"].append((img_path, seg_path))
        else:
            # Simple 85/10/5 split by index
            idx = hash(char_id) % 100
            if idx < 85:
                splits["train"].append((img_path, seg_path))
            elif idx < 95:
                splits["val"].append((img_path, seg_path))
            else:
                splits["test"].append((img_path, seg_path))

    logger.info("Split: train=%d, val=%d, test=%d",
                len(splits["train"]), len(splits["val"]), len(splits["test"]))

    # Build COCO categories (skip background = class 0)
    categories = [
        {"id": i, "name": CLASS_PROMPTS[i]}
        for i in range(1, NUM_CLASSES)
    ]

    # Process each split
    for split_name, split_examples in splits.items():
        if not split_examples:
            continue

        split_dir = output_dir / split_name
        images_dir = split_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        coco_data = {
            "images": [],
            "categories": categories,
            "annotations": [],
        }

        ann_id = 0
        for image_id, (img_path, seg_path) in enumerate(split_examples):
            if (image_id + 1) % 1000 == 0:
                logger.info("  %s: %d/%d", split_name, image_id + 1, len(split_examples))

            try:
                image_info, annotations, ann_id = process_example(
                    img_path, seg_path, image_id, ann_id, images_dir,
                )
                coco_data["images"].append(image_info)
                coco_data["annotations"].extend(annotations)
            except Exception as e:
                logger.warning("Error on %s: %s", img_path, e)

        # Write COCO JSON
        json_path = split_dir / "_annotations.coco.json"
        with open(json_path, "w") as f:
            json.dump(coco_data, f)

        logger.info("%s: %d images, %d annotations → %s",
                    split_name, len(coco_data["images"]),
                    len(coco_data["annotations"]), json_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
