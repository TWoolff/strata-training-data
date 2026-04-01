#!/usr/bin/env python3
"""Convert Dr. Li's SAM 19-class body parsing labels to Strata 22-class schema.

Reads sam_segmentation.npz (19 binary masks) and produces segmentation.png
(8-bit grayscale, pixel value = region ID 0-21).

Uses mask geometry and connected components for anatomy-aware region assignment.

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
from scipy.ndimage import label as connected_components

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


def _find_lr_center(masks: np.ndarray) -> int:
    """Find the character's center of mass X coordinate using torso masks.

    Uses neck + topwear + bottomwear center to determine L/R split,
    which is more robust than image center for off-center characters.
    """
    h, w = masks.shape[1], masks.shape[2]
    # Combine torso-area masks: neck(9), topwear(11), bottomwear(13)
    torso = masks[9].astype(bool) | masks[11].astype(bool) | masks[13].astype(bool)
    if torso.any():
        cols = np.where(torso.any(axis=0))[0]
        return int((cols[0] + cols[-1]) / 2)
    # Fallback: use any foreground
    fg = np.any(masks, axis=0)
    if fg.any():
        cols = np.where(fg.any(axis=0))[0]
        return int((cols[0] + cols[-1]) / 2)
    return w // 2


def _split_connected_lr(
    mask: np.ndarray, center_x: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split a binary mask into left and right components using connected components.

    For each connected component, assigns based on its center of mass relative
    to center_x. This handles non-frontal views better than a hard vertical split
    because limbs on the same side stay together.
    """
    if not mask.any():
        return np.zeros_like(mask, dtype=bool), np.zeros_like(mask, dtype=bool)

    labeled, n_components = connected_components(mask)
    left_mask = np.zeros_like(mask, dtype=bool)  # image left = char right
    right_mask = np.zeros_like(mask, dtype=bool)  # image right = char left

    for i in range(1, n_components + 1):
        comp = labeled == i
        cols = np.where(comp.any(axis=0))[0]
        comp_center_x = (cols[0] + cols[-1]) / 2
        if comp_center_x < center_x:
            left_mask |= comp
        else:
            right_mask |= comp

    return left_mask, right_mask


def _split_topwear(
    topwear_mask: np.ndarray,
    neck_mask: np.ndarray,
    center_x: int,
) -> dict[str, np.ndarray]:
    """Split topwear into chest, spine, shoulders, upper arms, and forearms.

    Uses the neck position and topwear geometry to determine anatomical regions:
    - Torso core (central mass) → chest (upper) + spine (lower)
    - Shoulder caps (wide area near neck) → shoulders
    - Arm extensions (narrow protrusions from torso) → upper arm + forearm

    The key insight: arms are lateral protrusions from the torso. We detect them
    by finding pixels that extend beyond the torso core width.
    """
    if not topwear_mask.any():
        return {}

    h, w = topwear_mask.shape
    result = {}

    # Find topwear bounding box
    rows = np.where(topwear_mask.any(axis=1))[0]
    top, bottom = rows[0], rows[-1]
    height = bottom - top + 1

    # Find torso core width per row (the widest contiguous region near center)
    # The torso is the central mass; arms are narrower lateral extensions
    row_widths = np.zeros(h)
    row_left = np.zeros(h, dtype=int)
    row_right = np.zeros(h, dtype=int)

    for y in range(top, bottom + 1):
        row_pixels = np.where(topwear_mask[y, :])[0]
        if len(row_pixels) == 0:
            continue
        row_left[y] = row_pixels[0]
        row_right[y] = row_pixels[-1]
        row_widths[y] = row_pixels[-1] - row_pixels[0]

    # Torso core width: use the median width of the middle rows (less affected by arms)
    mid_start = top + int(height * 0.3)
    mid_end = top + int(height * 0.7)
    mid_widths = row_widths[mid_start:mid_end]
    mid_widths = mid_widths[mid_widths > 0]
    if len(mid_widths) == 0:
        # Fallback: treat everything as chest
        result["chest"] = topwear_mask.copy()
        return result

    torso_width = np.median(mid_widths)

    # Torso core boundaries (centered on center_x)
    torso_half = torso_width / 2
    torso_left_bound = center_x - torso_half
    torso_right_bound = center_x + torso_half

    # Build masks
    chest_mask = np.zeros((h, w), dtype=bool)
    spine_mask = np.zeros((h, w), dtype=bool)
    shoulder_l_mask = np.zeros((h, w), dtype=bool)
    shoulder_r_mask = np.zeros((h, w), dtype=bool)
    upper_arm_l_mask = np.zeros((h, w), dtype=bool)
    upper_arm_r_mask = np.zeros((h, w), dtype=bool)
    forearm_l_mask = np.zeros((h, w), dtype=bool)
    forearm_r_mask = np.zeros((h, w), dtype=bool)

    # Shoulder zone: top 15% of topwear height (deltoid cap area)
    shoulder_bottom = top + int(height * 0.15)
    # Chest/spine split: chest is upper 55%, spine is lower 45%
    chest_spine_split = top + int(height * 0.55)

    for y in range(top, bottom + 1):
        row_pixels = np.where(topwear_mask[y, :])[0]
        if len(row_pixels) == 0:
            continue

        for x in row_pixels:
            # Is this pixel in the torso core?
            in_core = torso_left_bound <= x <= torso_right_bound

            if in_core:
                # Torso core → chest or spine
                if y <= chest_spine_split:
                    chest_mask[y, x] = True
                else:
                    spine_mask[y, x] = True
            else:
                # Lateral extension → shoulder, upper arm, or forearm
                is_left = x < center_x  # image left = char right

                # Determine arm sub-region by vertical position within the arm
                # The arm extends from shoulder (top) to wherever topwear ends
                arm_top = top  # arms start at topwear top
                arm_length = bottom - arm_top + 1

                arm_frac = (y - arm_top) / max(arm_length, 1)

                if y <= shoulder_bottom:
                    # Shoulder zone
                    if is_left:
                        shoulder_r_mask[y, x] = True
                    else:
                        shoulder_l_mask[y, x] = True
                elif arm_frac < 0.55:
                    # Upper arm (middle section)
                    if is_left:
                        upper_arm_r_mask[y, x] = True
                    else:
                        upper_arm_l_mask[y, x] = True
                else:
                    # Forearm (lower section of arm extension)
                    if is_left:
                        forearm_r_mask[y, x] = True
                    else:
                        forearm_l_mask[y, x] = True

    result["chest"] = chest_mask
    result["spine"] = spine_mask
    result["shoulder_l"] = shoulder_l_mask
    result["shoulder_r"] = shoulder_r_mask
    result["upper_arm_l"] = upper_arm_l_mask
    result["upper_arm_r"] = upper_arm_r_mask
    result["forearm_l"] = forearm_l_mask
    result["forearm_r"] = forearm_r_mask

    return result


def _split_bottomwear(
    bottomwear_mask: np.ndarray,
    center_x: int,
) -> dict[str, np.ndarray]:
    """Split bottomwear into hips and upper legs.

    Uses connected components to separate individual leg sections, then assigns
    hips (top portion) and upper_legs (lower portions, L/R by component center).
    """
    if not bottomwear_mask.any():
        return {}

    h, w = bottomwear_mask.shape
    rows = np.where(bottomwear_mask.any(axis=1))[0]
    top, bottom = rows[0], rows[-1]
    height = bottom - top + 1

    # Top 25% → hips
    hip_bottom = top + int(height * 0.25)

    hips_mask = bottomwear_mask.copy()
    hips_mask[hip_bottom:, :] = False

    # Below hips → upper legs (use connected components for L/R)
    leg_mask = bottomwear_mask.copy()
    leg_mask[:hip_bottom, :] = False

    left_legs, right_legs = _split_connected_lr(leg_mask, center_x)

    return {
        "hips": hips_mask,
        "upper_leg_r": left_legs,   # image left = char right
        "upper_leg_l": right_legs,  # image right = char left
    }


def _find_bare_skin(
    masks: np.ndarray,
    alpha: np.ndarray | None,
) -> np.ndarray:
    """Find foreground pixels not covered by any SAM class (bare skin).

    These are typically bare arms, hands, or legs not covered by clothing.
    """
    h, w = masks.shape[1], masks.shape[2]
    any_class = np.any(masks, axis=0)

    if alpha is not None:
        foreground = alpha > 128
    else:
        foreground = np.ones((h, w), dtype=bool)

    return foreground & ~any_class


def _assign_bare_skin(
    result: np.ndarray,
    bare_skin: np.ndarray,
    masks: np.ndarray,
    center_x: int,
) -> np.ndarray:
    """Assign bare skin pixels to the nearest body region.

    For each connected component of bare skin, determines the most likely
    body region based on:
    - Adjacency to labeled regions (what's above/beside it)
    - Vertical position relative to known landmarks
    - L/R based on component center of mass
    """
    if not bare_skin.any():
        return result

    h, w = result.shape
    labeled, n_components = connected_components(bare_skin)

    # Find key vertical landmarks from existing labels
    neck_mask = masks[9].astype(bool) | masks[10].astype(bool)
    topwear_mask = masks[11].astype(bool)
    bottomwear_mask = masks[13].astype(bool)
    legwear_mask = masks[14].astype(bool)
    footwear_mask = masks[15].astype(bool)

    # Vertical extents
    neck_bottom = 0
    if neck_mask.any():
        neck_bottom = np.where(neck_mask.any(axis=1))[0][-1]

    topwear_bottom = 0
    if topwear_mask.any():
        topwear_bottom = np.where(topwear_mask.any(axis=1))[0][-1]

    bottomwear_top = h
    bottomwear_bottom = h
    if bottomwear_mask.any():
        bw_rows = np.where(bottomwear_mask.any(axis=1))[0]
        bottomwear_top = bw_rows[0]
        bottomwear_bottom = bw_rows[-1]

    legwear_top = h
    if legwear_mask.any():
        legwear_top = np.where(legwear_mask.any(axis=1))[0][0]

    # Torso horizontal extent
    torso_left, torso_right = 0, w
    if topwear_mask.any():
        cols = np.where(topwear_mask.any(axis=0))[0]
        # Use narrower estimate (middle rows)
        rows = np.where(topwear_mask.any(axis=1))[0]
        mid = rows[len(rows) // 2]
        mid_cols = np.where(topwear_mask[mid, :])[0]
        if len(mid_cols) > 0:
            torso_left = mid_cols[0]
            torso_right = mid_cols[-1]

    for i in range(1, n_components + 1):
        comp = labeled == i
        comp_rows = np.where(comp.any(axis=1))[0]
        comp_cols = np.where(comp.any(axis=0))[0]
        if len(comp_rows) == 0 or len(comp_cols) == 0:
            continue

        cy = (comp_rows[0] + comp_rows[-1]) / 2
        cx = (comp_cols[0] + comp_cols[-1]) / 2
        is_left = cx < center_x  # image left = char right

        # Is this component lateral (outside torso width)?
        is_lateral = cx < torso_left or cx > torso_right

        # Determine region by vertical position and context
        if cy < neck_bottom + 10 and not is_lateral:
            # Near neck, central → neck
            result[comp] = S["neck"]
        elif cy <= topwear_bottom and is_lateral:
            # Beside topwear, lateral → arm region
            arm_frac = (cy - neck_bottom) / max(topwear_bottom - neck_bottom, 1)
            if arm_frac < 0.15:
                result[comp] = S["shoulder_r"] if is_left else S["shoulder_l"]
            elif arm_frac < 0.55:
                result[comp] = S["upper_arm_r"] if is_left else S["upper_arm_l"]
            else:
                result[comp] = S["forearm_r"] if is_left else S["forearm_l"]
        elif cy > topwear_bottom and cy < bottomwear_top and is_lateral:
            # Between topwear and bottomwear, lateral → forearm or hand
            result[comp] = S["hand_r"] if is_left else S["hand_l"]
        elif cy > bottomwear_bottom and cy < legwear_top:
            # Between bottomwear and legwear → lower part of upper leg or knee
            result[comp] = S["upper_leg_r"] if is_left else S["upper_leg_l"]
        elif cy > topwear_bottom and cy <= bottomwear_top and not is_lateral:
            # Central, below topwear but above bottomwear → exposed midriff (spine)
            result[comp] = S["spine"]
        elif cy > bottomwear_bottom and is_lateral:
            # Below bottomwear, lateral → lower leg
            result[comp] = S["lower_leg_r"] if is_left else S["lower_leg_l"]
        # else: leave as background (too ambiguous)

    return result


def convert_sam_to_strata(
    masks: np.ndarray,
    image_width: int = 512,
    alpha: np.ndarray | None = None,
) -> np.ndarray:
    """Convert 19-class binary masks to single-channel 22-class mask.

    Args:
        masks: [19, H, W] binary masks from SAM body parsing.
        image_width: Width for determining L/R center line.
        alpha: Optional [H, W] alpha channel for finding bare skin.

    Returns:
        [H, W] uint8 with Strata region IDs (0-21).
    """
    h, w = masks.shape[1], masks.shape[2]
    result = np.zeros((h, w), dtype=np.uint8)  # background = 0

    # Find character center using torso geometry (not image center)
    center_x = _find_lr_center(masks)

    # --- Head region (merge face-related classes) ---
    head_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # hair, headwear, face, eyes, eyewear, ears, earwear, nose, mouth
    head_mask = np.any(masks[head_classes], axis=0)
    result[head_mask] = S["head"]

    # Hair gets special treatment — pixels below face → hair_back
    hair_mask = masks[0].astype(bool)
    if hair_mask.any():
        face_mask = masks[2].astype(bool)
        if face_mask.any():
            face_rows = np.where(face_mask.any(axis=1))[0]
            face_bottom = face_rows[-1] if len(face_rows) > 0 else h // 2
            hair_back_region = hair_mask.copy()
            hair_back_region[:face_bottom, :] = False
            result[hair_back_region] = S["hair_back"]

    # --- Neck ---
    neck_classes = [9, 10]  # neck, neckwear
    neck_mask = np.any(masks[neck_classes], axis=0)
    result[neck_mask] = S["neck"]

    # --- Topwear → chest/spine + shoulders + upper arms + forearms ---
    topwear_mask = masks[11].astype(bool)
    topwear_parts = _split_topwear(topwear_mask, neck_mask, center_x)
    for region_name, region_mask in topwear_parts.items():
        result[region_mask] = S[region_name]

    # --- Handwear → forearm + hand (SAM often labels entire bare arm as handwear) ---
    handwear_mask = masks[12].astype(bool)
    if handwear_mask.any():
        left_hw, right_hw = _split_connected_lr(handwear_mask, center_x)
        # Split each side into forearm (upper portion) and hand (lower portion)
        for hw_side, side_label in [(left_hw, "r"), (right_hw, "l")]:
            if not hw_side.any():
                continue
            rows = np.where(hw_side.any(axis=1))[0]
            if len(rows) == 0:
                continue
            top_hw, bottom_hw = rows[0], rows[-1]
            hw_height = bottom_hw - top_hw + 1
            if hw_height > 30:
                # Large handwear region → split into forearm + hand
                hand_top = bottom_hw - int(hw_height * 0.35)
                forearm_region = hw_side.copy()
                forearm_region[hand_top:, :] = False
                hand_region = hw_side.copy()
                hand_region[:hand_top, :] = False
                result[forearm_region] = S[f"forearm_{side_label}"]
                result[hand_region] = S[f"hand_{side_label}"]
            else:
                # Small → just hand
                result[hw_side] = S[f"hand_{side_label}"]

    # --- Bottomwear → hips + upper legs ---
    bottomwear_mask = masks[13].astype(bool)
    legwear_mask = masks[14].astype(bool)

    if bottomwear_mask.any():
        bottomwear_parts = _split_bottomwear(bottomwear_mask, center_x)
        for region_name, region_mask in bottomwear_parts.items():
            result[region_mask] = S[region_name]

    # --- Legwear handling ---
    # SAM often puts everything below waist into legwear when there's no
    # distinct bottomwear. Split into hips + upper_leg + lower_leg.
    if legwear_mask.any():
        if not bottomwear_mask.any():
            # No bottomwear detected — legwear covers entire lower body.
            # Split into hips (top), upper_leg (middle), lower_leg (bottom).
            rows = np.where(legwear_mask.any(axis=1))[0]
            top_lw, bottom_lw = rows[0], rows[-1]
            lw_height = bottom_lw - top_lw + 1

            # Top 15% → hips, next 35% → upper legs, bottom 50% → lower legs
            hip_bottom = top_lw + int(lw_height * 0.15)
            upper_leg_bottom = top_lw + int(lw_height * 0.50)

            hips_region = legwear_mask.copy()
            hips_region[hip_bottom:, :] = False
            result[hips_region] = S["hips"]

            upper_leg_region = legwear_mask.copy()
            upper_leg_region[:hip_bottom, :] = False
            upper_leg_region[upper_leg_bottom:, :] = False
            left_ul, right_ul = _split_connected_lr(upper_leg_region, center_x)
            result[left_ul] = S["upper_leg_r"]   # image left = char right
            result[right_ul] = S["upper_leg_l"]

            lower_leg_region = legwear_mask.copy()
            lower_leg_region[:upper_leg_bottom, :] = False
            left_ll, right_ll = _split_connected_lr(lower_leg_region, center_x)
            result[left_ll] = S["lower_leg_r"]
            result[right_ll] = S["lower_leg_l"]
        else:
            # Bottomwear exists — legwear is truly just lower legs
            left_legs, right_legs = _split_connected_lr(legwear_mask, center_x)
            result[left_legs] = S["lower_leg_r"]
            result[right_legs] = S["lower_leg_l"]

    # --- Footwear → feet (using connected components for L/R) ---
    footwear_mask = masks[15].astype(bool)
    if footwear_mask.any():
        left_feet, right_feet = _split_connected_lr(footwear_mask, center_x)
        result[left_feet] = S["foot_r"]   # image left = char right
        result[right_feet] = S["foot_l"]  # image right = char left

    # --- Tail, wings, objects → accessory ---
    accessory_classes = [16, 17, 18]
    acc_mask = np.any(masks[accessory_classes], axis=0)
    result[acc_mask] = S["accessory"]

    # --- Bare skin assignment (bare arms, hands, legs) ---
    bare_skin = _find_bare_skin(masks, alpha)
    result = _assign_bare_skin(result, bare_skin, masks, center_x)

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

            # Load alpha channel if available
            alpha = None
            img_path = ex_dir / "image.png"
            if img_path.exists():
                img = Image.open(img_path)
                if img.mode == "RGBA":
                    alpha = np.array(img)[:, :, 3]

            strata_mask = convert_sam_to_strata(masks, alpha=alpha)

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
