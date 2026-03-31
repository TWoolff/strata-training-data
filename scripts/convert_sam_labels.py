#!/usr/bin/env python3
"""Convert Dr. Li's SAM 19-class body parsing labels to Strata 22-class schema.

Reads sam_segmentation.npz (19 binary masks) and produces segmentation.png
(8-bit grayscale, pixel value = region ID 0-21).

Uses image centerline for L/R splitting of symmetric body parts.

Usage::

    python scripts/convert_sam_labels.py \
        --input-dir /Volumes/TAMWoolff/data/sam_labels/data_cloud/sora_diverse \
        --output-dir /Volumes/TAMWoolff/data/preprocessed/gemini_diverse \
        --only-missing
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Dr. Li's SAM 19 classes (index → name)
SAM_CLASSES = [
    "hair", "headwear", "face", "eyes", "eyewear", "ears", "earwear",
    "nose", "mouth", "neck", "neckwear", "topwear", "handwear",
    "bottomwear", "legwear", "footwear", "tail", "wings", "objects",
]

# Strata 22-class IDs
S = {
    "background": 0, "head": 1, "neck": 2, "chest": 3, "spine": 4,
    "hips": 5, "shoulder_l": 6, "upper_arm_l": 7, "forearm_l": 8,
    "hand_l": 9, "shoulder_r": 10, "upper_arm_r": 11, "forearm_r": 12,
    "hand_r": 13, "upper_leg_l": 14, "lower_leg_l": 15, "foot_l": 16,
    "upper_leg_r": 17, "lower_leg_r": 18, "foot_r": 19,
    "accessory": 20, "hair_back": 21,
}


def convert_sam_to_strata(
    masks: np.ndarray,
    image_width: int = 512,
) -> np.ndarray:
    """Convert 19-class binary masks to single-channel 22-class mask.

    Args:
        masks: [19, H, W] binary masks from SAM body parsing.
        image_width: Width for determining L/R center line.

    Returns:
        [H, W] uint8 with Strata region IDs (0-21).
    """
    h, w = masks.shape[1], masks.shape[2]
    result = np.zeros((h, w), dtype=np.uint8)  # background = 0

    center_x = w // 2

    # Create left/right masks (from character's perspective: left side of image = character's right)
    # We use image coordinates: left half = character's right side
    left_half = np.zeros((h, w), dtype=bool)
    left_half[:, :center_x] = True
    right_half = ~left_half

    # --- Head region (merge face-related classes) ---
    head_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # hair, headwear, face, eyes, eyewear, ears, earwear, nose, mouth
    head_mask = np.any(masks[head_classes], axis=0)
    result[head_mask] = S["head"]

    # Hair gets special treatment — pixels in lower portion might be hair_back
    hair_mask = masks[0].astype(bool)  # "hair" class
    if hair_mask.any():
        # Hair below the face center → hair_back
        face_mask = masks[2].astype(bool)
        if face_mask.any():
            face_rows = np.where(face_mask.any(axis=1))[0]
            face_bottom = face_rows[-1] if len(face_rows) > 0 else h // 2
            hair_back_region = hair_mask.copy()
            hair_back_region[:face_bottom, :] = False  # Only below face
            # Only assign hair_back if it's behind the head (near edges)
            result[hair_back_region] = S["hair_back"]

    # --- Neck ---
    neck_classes = [9, 10]  # neck, neckwear
    neck_mask = np.any(masks[neck_classes], axis=0)
    result[neck_mask] = S["neck"]

    # --- Topwear → chest/spine + shoulders + upper arms ---
    topwear_mask = masks[11].astype(bool)
    if topwear_mask.any():
        # Find topwear bounding box
        rows = np.where(topwear_mask.any(axis=1))[0]
        cols = np.where(topwear_mask.any(axis=0))[0]
        if len(rows) > 0 and len(cols) > 0:
            top, bottom = rows[0], rows[-1]
            left, right = cols[0], cols[-1]
            height = bottom - top
            width = right - left

            # Central 60% width → chest/spine
            chest_left = left + int(width * 0.2)
            chest_right = right - int(width * 0.2)

            # Upper 60% height → chest, lower 40% → spine
            chest_bottom = top + int(height * 0.6)

            # Assign regions
            for y in range(top, bottom + 1):
                for x in range(left, right + 1):
                    if not topwear_mask[y, x]:
                        continue

                    if chest_left <= x <= chest_right:
                        if y <= chest_bottom:
                            result[y, x] = S["chest"]
                        else:
                            result[y, x] = S["spine"]
                    elif x < chest_left:
                        # Left side of image = character's right
                        if y < top + int(height * 0.3):
                            result[y, x] = S["shoulder_r"]
                        else:
                            result[y, x] = S["upper_arm_r"]
                    else:
                        # Right side = character's left
                        if y < top + int(height * 0.3):
                            result[y, x] = S["shoulder_l"]
                        else:
                            result[y, x] = S["upper_arm_l"]

    # --- Handwear → hand_l / hand_r ---
    handwear_mask = masks[12].astype(bool)
    if handwear_mask.any():
        result[handwear_mask & left_half] = S["hand_r"]   # image left = char right
        result[handwear_mask & right_half] = S["hand_l"]  # image right = char left

    # --- Bottomwear → hips + upper legs ---
    bottomwear_mask = masks[13].astype(bool)
    if bottomwear_mask.any():
        rows = np.where(bottomwear_mask.any(axis=1))[0]
        if len(rows) > 0:
            top, bottom = rows[0], rows[-1]
            height = bottom - top
            hip_bottom = top + int(height * 0.3)

            hip_region = bottomwear_mask.copy()
            hip_region[hip_bottom:, :] = False
            result[hip_region] = S["hips"]

            leg_region = bottomwear_mask.copy()
            leg_region[:hip_bottom, :] = False
            result[leg_region & left_half] = S["upper_leg_r"]
            result[leg_region & right_half] = S["upper_leg_l"]

    # --- Legwear → lower legs ---
    legwear_mask = masks[14].astype(bool)
    if legwear_mask.any():
        result[legwear_mask & left_half] = S["lower_leg_r"]
        result[legwear_mask & right_half] = S["lower_leg_l"]

    # --- Footwear → feet ---
    footwear_mask = masks[15].astype(bool)
    if footwear_mask.any():
        result[footwear_mask & left_half] = S["foot_r"]
        result[footwear_mask & right_half] = S["foot_l"]

    # --- Tail, wings, objects → accessory ---
    accessory_classes = [16, 17, 18]
    acc_mask = np.any(masks[accessory_classes], axis=0)
    result[acc_mask] = S["accessory"]

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SAM 19-class to Strata 22-class")
    parser.add_argument("--input-dir", type=Path, required=True, help="Dir with sam_segmentation.npz files")
    parser.add_argument("--output-dir", type=Path, help="Dir to save segmentation.png (default: same as input)")
    parser.add_argument("--only-missing", action="store_true", help="Skip if segmentation.png exists")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output_dir = args.output_dir or args.input_dir

    # Find all examples with SAM labels
    examples = sorted([
        d for d in args.input_dir.iterdir()
        if d.is_dir() and (d / "sam_segmentation.npz").exists()
    ])
    logger.info("Found %d examples with SAM labels", len(examples))

    processed = 0
    skipped = 0

    for i, ex_dir in enumerate(examples):
        # Determine output path
        if args.output_dir:
            out_dir = output_dir / ex_dir.name
        else:
            out_dir = ex_dir

        out_path = out_dir / "segmentation.png"

        if args.only_missing and out_path.exists():
            skipped += 1
            continue

        try:
            data = np.load(ex_dir / "sam_segmentation.npz")
            masks = data["masks"]  # [19, H, W]

            strata_mask = convert_sam_to_strata(masks)

            out_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(strata_mask, mode="L").save(out_path)
            processed += 1

        except Exception as e:
            logger.warning("Error on %s: %s", ex_dir.name, e)

        if (i + 1) % 500 == 0:
            logger.info("Progress: %d/%d (%d processed, %d skipped)", i + 1, len(examples), processed, skipped)

    logger.info("Done: %d processed, %d skipped", processed, skipped)


if __name__ == "__main__":
    main()
