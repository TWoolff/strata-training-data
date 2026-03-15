#!/usr/bin/env python3
"""Convert Dr. Li's 19-class clothing-based segmentation labels to Strata 22-class skeleton-based format.

Dr. Li's labels use a clothing/appearance-oriented schema (19 classes, indices * 10):
    hair(0), headwear(10), face(20), eyes(30), eyewear(40), ears(50), earwear(60),
    nose(70), mouth(80), neck(90), neckwear(100), topwear(110), handwear(120),
    bottomwear(130), legwear(140), footwear(150), tail(160), wings(170), objects(180),
    background(255)

Strata needs skeleton-based regions (22 classes, IDs 0-21):
    background(0), head(1), neck(2), chest(3), spine(4), hips(5),
    shoulder_l(6), upper_arm_l(7), forearm_l(8), hand_l(9),
    shoulder_r(10), upper_arm_r(11), forearm_r(12), hand_r(13),
    upper_leg_l(14), lower_leg_l(15), foot_l(16),
    upper_leg_r(17), lower_leg_r(18), foot_r(19),
    accessory(20), hair_back(21)

Direct merges (trivial):
    face + eyes + ears + nose + mouth → head (1)
    neck → neck (2)
    headwear + eyewear + earwear + neckwear + tail + wings + objects → accessory (20)
    background → background (0)

Joint-based splits (need skeleton positions):
    topwear → chest / spine / shoulder_l/r / upper_arm_l/r
    handwear → forearm_l/r / hand_l/r
    bottomwear → hips / upper_leg_l/r
    legwear → upper_leg_l/r / lower_leg_l/r
    footwear → foot_l/r
    hair → head (1) or hair_back (21)

Usage:
    # With joints (best quality — run joints inference first):
    python scripts/convert_li_labels.py \
        --labels /Volumes/TAMWoolff/data/labels/ \
        --gemini-dir /Volumes/TAMWoolff/data/preprocessed/gemini_diverse/ \
        --output-dir /Volumes/TAMWoolff/data/preprocessed/gemini_li_converted/

    # Preview mode (show overlays, don't save):
    python scripts/convert_li_labels.py \
        --labels /Volumes/TAMWoolff/data/labels/ \
        --gemini-dir /Volumes/TAMWoolff/data/preprocessed/gemini_diverse/ \
        --output-dir /tmp/li_preview/ \
        --preview 10
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Dr. Li's 19-class schema (pixel_value → class_name)
LI_CLASSES = {
    0: "hair",
    10: "headwear",
    20: "face",
    30: "eyes",
    40: "eyewear",
    50: "ears",
    60: "earwear",
    70: "nose",
    80: "mouth",
    90: "neck",
    100: "neckwear",
    110: "topwear",
    120: "handwear",
    130: "bottomwear",
    140: "legwear",
    150: "footwear",
    160: "tail",
    170: "wings",
    180: "objects",
    255: "background",
}

# Strata 22-class region IDs
REGION = {
    "background": 0, "head": 1, "neck": 2, "chest": 3, "spine": 4, "hips": 5,
    "shoulder_l": 6, "upper_arm_l": 7, "forearm_l": 8, "hand_l": 9,
    "shoulder_r": 10, "upper_arm_r": 11, "forearm_r": 12, "hand_r": 13,
    "upper_leg_l": 14, "lower_leg_l": 15, "foot_l": 16,
    "upper_leg_r": 17, "lower_leg_r": 18, "foot_r": 19,
    "accessory": 20, "hair_back": 21,
}

# Direct 1:1 or N:1 merges (no spatial reasoning needed)
DIRECT_MERGE = {
    20: REGION["head"],       # face → head
    30: REGION["head"],       # eyes → head
    50: REGION["head"],       # ears → head
    70: REGION["head"],       # nose → head
    80: REGION["head"],       # mouth → head
    90: REGION["neck"],       # neck → neck
    10: REGION["accessory"],  # headwear → accessory
    40: REGION["accessory"],  # eyewear → accessory
    60: REGION["accessory"],  # earwear → accessory
    100: REGION["accessory"], # neckwear → accessory
    160: REGION["accessory"], # tail → accessory
    170: REGION["accessory"], # wings → accessory
    180: REGION["accessory"], # objects → accessory
    255: REGION["background"],
}

# Bone order for joints (matches training/data/transforms.py)
BONE_ORDER = [
    "hips", "spine", "chest", "neck", "head",
    "shoulder_l", "upper_arm_l", "forearm_l", "hand_l",
    "shoulder_r", "upper_arm_r", "forearm_r", "hand_r",
    "upper_leg_l", "lower_leg_l", "foot_l",
    "upper_leg_r", "lower_leg_r", "foot_r",
    "hair_back",
]


_ALPHA_THRESHOLD = 128


def crop_to_foreground_coords(
    img: Image.Image,
    padding_ratio: float = 0.10,
) -> tuple[int, int, int, int]:
    """Compute the square foreground crop box (x_min, y_min, x_max+1, y_max+1).

    Replicates ``ingest.gemini_diverse_adapter._crop_to_foreground`` logic
    so we can apply the identical crop to both the image and the label.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    alpha = np.array(img.split()[-1])
    rows = np.any(alpha >= _ALPHA_THRESHOLD, axis=1)
    cols = np.any(alpha >= _ALPHA_THRESHOLD, axis=0)

    if not rows.any():
        w, h = img.size
        return (0, 0, w, h)

    y_min, y_max = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    x_min, x_max = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])

    w, h = img.size
    fg_w = x_max - x_min + 1
    fg_h = y_max - y_min + 1
    pad = int(max(fg_w, fg_h) * padding_ratio)

    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(w - 1, x_max + pad)
    y_max = min(h - 1, y_max + pad)

    # Make square by expanding the shorter side
    crop_w = x_max - x_min + 1
    crop_h = y_max - y_min + 1
    if crop_w > crop_h:
        diff = crop_w - crop_h
        y_min = max(0, y_min - diff // 2)
        y_max = min(h - 1, y_min + crop_w - 1)
        if y_max - y_min + 1 < crop_w:
            y_min = max(0, y_max - crop_w + 1)
    elif crop_h > crop_w:
        diff = crop_h - crop_w
        x_min = max(0, x_min - diff // 2)
        x_max = min(w - 1, x_min + crop_h - 1)
        if x_max - x_min + 1 < crop_h:
            x_min = max(0, x_max - crop_h + 1)

    return (x_min, y_min, x_max + 1, y_max + 1)


def align_label_to_source(
    raw_img: Image.Image,
    li_label: np.ndarray,
    resolution: int = 512,
) -> tuple[Image.Image, np.ndarray]:
    """Crop and resize both image and label identically.

    1. Run rembg on raw image to get alpha mask
    2. Compute foreground crop box from alpha
    3. Apply same crop to label (resized to raw dims first if needed)
    4. Resize both to *resolution* × *resolution*

    Returns (cropped_resized_image_RGBA, cropped_resized_label).
    """
    from rembg import remove as rembg_remove

    # Step 1: background removal to get alpha
    rgba = rembg_remove(raw_img.convert("RGB"))

    # Step 2: compute crop box
    box = crop_to_foreground_coords(rgba)

    # Step 3: resize label to match raw image dimensions if needed
    raw_w, raw_h = raw_img.size
    lbl_h, lbl_w = li_label.shape
    if (lbl_w, lbl_h) != (raw_w, raw_h):
        li_label_img = Image.fromarray(li_label, mode="L")
        li_label_img = li_label_img.resize((raw_w, raw_h), Image.NEAREST)
        li_label = np.array(li_label_img)

    # Step 4: crop both
    cropped_img = rgba.crop(box)
    cropped_label = li_label[box[1]:box[3], box[0]:box[2]]

    # Step 5: resize to output resolution (aspect-preserving with padding)
    cw, ch = cropped_img.size
    scale = resolution / max(cw, ch)
    new_w = round(cw * scale)
    new_h = round(ch * scale)

    resized_img = cropped_img.resize((new_w, new_h), Image.LANCZOS)
    canvas_img = Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))
    offset_x = (resolution - new_w) // 2
    offset_y = (resolution - new_h) // 2
    canvas_img.paste(resized_img, (offset_x, offset_y))

    resized_lbl = Image.fromarray(cropped_label, mode="L").resize(
        (new_w, new_h), Image.NEAREST,
    )
    canvas_lbl = Image.new("L", (resolution, resolution), 255)  # bg=255
    canvas_lbl.paste(resized_lbl, (offset_x, offset_y))

    return canvas_img, np.array(canvas_lbl)


def label_filename_to_dir(label_name: str) -> str:
    """Convert label filename to gemini_diverse directory name."""
    name = label_name.replace(".png", "")
    if name.startswith("ChatGPT Image"):
        name = name.lower().replace(" ", "_").replace(",", "")
        return name.replace("chatgpt_image", "gemini_diverse_chatgpt")
    elif name.startswith("Gemini_Generated_Image_"):
        suffix = name.replace("Gemini_Generated_Image_", "")
        return f"gemini_diverse_gemini_{suffix}"
    elif name.startswith("sora_"):
        return "gemini_diverse_" + name.replace(" ", "_")
    return name


def load_joints(joints_path: Path) -> dict[str, tuple[int, int]] | None:
    """Load joints.json and return {bone_name: (x, y)} or None."""
    if not joints_path.exists():
        return None
    with open(joints_path) as f:
        data = json.load(f)
    joints = {}
    for name, info in data.get("joints", {}).items():
        pos = info.get("position", [0, 0])
        joints[name] = (int(pos[0]), int(pos[1]))
    return joints


def get_fg_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Get bounding box of all foreground pixels (non-255 in Li's labels).

    Returns (x_min, y_min, x_max, y_max).
    """
    fg = mask != 255
    if not fg.any():
        h, w = mask.shape
        return (0, 0, w, h)
    rows = np.any(fg, axis=1)
    cols = np.any(fg, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return (int(x_min), int(y_min), int(x_max), int(y_max))


def get_face_center(mask: np.ndarray) -> tuple[float, float] | None:
    """Get center of face region (Li value 20) for midline estimation."""
    face_mask = mask == 20
    if not face_mask.any():
        return None
    ys, xs = np.where(face_mask)
    return (float(xs.mean()), float(ys.mean()))


def split_left_right(
    region_mask: np.ndarray,
    li_mask: np.ndarray,
    li_value: int,
    left_id: int,
    right_id: int,
    midline_x: float,
) -> None:
    """Split a Li region into left/right Strata regions using midline.

    Uses character's anatomical left/right: pixels to the LEFT of midline
    in image space are the character's RIGHT side (they face us).
    """
    pixels = li_mask == li_value
    if not pixels.any():
        return
    ys, xs = np.where(pixels)
    # Image left of midline = character's right
    is_char_right = xs < midline_x
    region_mask[ys[is_char_right], xs[is_char_right]] = right_id
    region_mask[ys[~is_char_right], xs[~is_char_right]] = left_id


def convert_with_joints(
    li_mask: np.ndarray,
    joints: dict[str, tuple[int, int]],
) -> np.ndarray:
    """Convert Li labels to Strata 22-class using joint positions for splitting."""
    h, w = li_mask.shape
    result = np.zeros((h, w), dtype=np.uint8)

    # Step 1: Direct merges
    for li_val, strata_id in DIRECT_MERGE.items():
        result[li_mask == li_val] = strata_id

    # Get midline from joints (average of hips, spine, chest, neck, head x-coords)
    midline_joints = ["hips", "spine", "chest", "neck", "head"]
    midline_xs = [joints[j][0] for j in midline_joints if j in joints]
    if midline_xs:
        midline_x = float(np.mean(midline_xs))
    else:
        face_center = get_face_center(li_mask)
        midline_x = face_center[0] if face_center else w / 2.0

    # Step 2: Hair → head(1) or hair_back(21)
    # Hair below the face bottom = likely hair_back (long hair hanging down back)
    hair_pixels = li_mask == 0
    if hair_pixels.any():
        face_pixels = li_mask == 20
        if face_pixels.any():
            face_bottom = np.where(face_pixels)[0].max()
            hair_ys, hair_xs = np.where(hair_pixels)
            # Hair above face bottom → head, hair below → hair_back
            above = hair_ys <= face_bottom
            result[hair_ys[above], hair_xs[above]] = REGION["head"]
            result[hair_ys[~above], hair_xs[~above]] = REGION["hair_back"]
        else:
            result[hair_pixels] = REGION["head"]

    # Step 3: Topwear → chest / spine / shoulder / upper_arm
    topwear = li_mask == 110
    if topwear.any():
        neck_y = joints.get("neck", (0, 0))[1] if "neck" in joints else None
        chest_y = joints.get("chest", (0, 0))[1] if "chest" in joints else None
        spine_y = joints.get("spine", (0, 0))[1] if "spine" in joints else None
        hips_y = joints.get("hips", (0, 0))[1] if "hips" in joints else None

        shoulder_l = joints.get("shoulder_l")
        shoulder_r = joints.get("shoulder_r")
        upper_arm_l = joints.get("upper_arm_l")
        upper_arm_r = joints.get("upper_arm_r")

        tw_ys, tw_xs = np.where(topwear)

        for idx in range(len(tw_ys)):
            y, x = tw_ys[idx], tw_xs[idx]
            is_right_side = x < midline_x  # character's right

            # Check if pixel is in the arm region (outside shoulder x-range)
            in_arm = False
            if shoulder_l and shoulder_r:
                shoulder_inner_l = shoulder_l[0]
                shoulder_inner_r = shoulder_r[0]
                # Character's left shoulder is on image right, and vice versa
                # shoulder_l = character's left = image right side
                # shoulder_r = character's right = image left side
                if x > shoulder_inner_l + 5:  # past character's left shoulder
                    in_arm = True
                    is_right_side = False  # it's on character's left
                elif x < shoulder_inner_r - 5:  # past character's right shoulder
                    in_arm = True
                    is_right_side = True

            if in_arm:
                # Determine shoulder vs upper_arm by vertical position
                ref_shoulder = shoulder_r if is_right_side else shoulder_l
                ref_arm = upper_arm_r if is_right_side else upper_arm_l
                if ref_shoulder and ref_arm:
                    mid_y = (ref_shoulder[1] + ref_arm[1]) / 2.0
                    if y < mid_y:
                        result[y, x] = REGION["shoulder_r"] if is_right_side else REGION["shoulder_l"]
                    else:
                        result[y, x] = REGION["upper_arm_r"] if is_right_side else REGION["upper_arm_l"]
                elif ref_shoulder:
                    result[y, x] = REGION["shoulder_r"] if is_right_side else REGION["shoulder_l"]
                else:
                    result[y, x] = REGION["upper_arm_r"] if is_right_side else REGION["upper_arm_l"]
            else:
                # Torso region: split into chest / spine by y position
                if spine_y and chest_y:
                    mid_torso = (chest_y + spine_y) / 2.0
                    result[y, x] = REGION["chest"] if y < mid_torso else REGION["spine"]
                elif chest_y:
                    result[y, x] = REGION["chest"]
                else:
                    result[y, x] = REGION["chest"]

    # Step 4: Handwear → forearm / hand (with L/R)
    handwear = li_mask == 120
    if handwear.any():
        forearm_l = joints.get("forearm_l")
        forearm_r = joints.get("forearm_r")
        hand_l = joints.get("hand_l")
        hand_r = joints.get("hand_r")

        hw_ys, hw_xs = np.where(handwear)
        for idx in range(len(hw_ys)):
            y, x = hw_ys[idx], hw_xs[idx]

            # Determine L/R by distance to forearm/hand joints
            dist_l = float("inf")
            dist_r = float("inf")
            for jt in [forearm_l, hand_l]:
                if jt:
                    d = (x - jt[0]) ** 2 + (y - jt[1]) ** 2
                    dist_l = min(dist_l, d)
            for jt in [forearm_r, hand_r]:
                if jt:
                    d = (x - jt[0]) ** 2 + (y - jt[1]) ** 2
                    dist_r = min(dist_r, d)

            is_right = dist_r < dist_l

            # Determine forearm vs hand by comparing to joint positions
            ref_forearm = forearm_r if is_right else forearm_l
            ref_hand = hand_r if is_right else hand_l
            if ref_forearm and ref_hand:
                # Project pixel onto forearm→hand axis, split at midpoint
                mid_y = (ref_forearm[1] + ref_hand[1]) / 2.0
                mid_x = (ref_forearm[0] + ref_hand[0]) / 2.0
                # Distance from forearm vs hand
                d_forearm = (x - ref_forearm[0]) ** 2 + (y - ref_forearm[1]) ** 2
                d_hand = (x - ref_hand[0]) ** 2 + (y - ref_hand[1]) ** 2
                if d_hand < d_forearm:
                    result[y, x] = REGION["hand_r"] if is_right else REGION["hand_l"]
                else:
                    result[y, x] = REGION["forearm_r"] if is_right else REGION["forearm_l"]
            elif ref_hand:
                result[y, x] = REGION["hand_r"] if is_right else REGION["hand_l"]
            else:
                result[y, x] = REGION["forearm_r"] if is_right else REGION["forearm_l"]

    # Step 5: Bottomwear → hips / upper_leg (with L/R)
    bottomwear = li_mask == 130
    if bottomwear.any():
        hips_pos = joints.get("hips")
        upper_leg_l = joints.get("upper_leg_l")
        upper_leg_r = joints.get("upper_leg_r")

        bw_ys, bw_xs = np.where(bottomwear)
        if hips_pos:
            hips_bottom_y = hips_pos[1]
            # Everything above hips joint → hips region
            # Everything below → upper_leg with L/R split
            above_hips = bw_ys <= hips_bottom_y
            result[bw_ys[above_hips], bw_xs[above_hips]] = REGION["hips"]
            # Below hips → upper_leg L/R
            below = ~above_hips
            if below.any():
                below_xs = bw_xs[below]
                is_char_right = below_xs < midline_x
                result[bw_ys[below][is_char_right], below_xs[is_char_right]] = REGION["upper_leg_r"]
                result[bw_ys[below][~is_char_right], below_xs[~is_char_right]] = REGION["upper_leg_l"]
        else:
            # No hips joint: top 30% → hips, bottom 70% → upper_leg
            bbox = get_fg_bbox(li_mask)
            split_y = bbox[1] + (bbox[3] - bbox[1]) * 0.5
            above = bw_ys <= split_y
            result[bw_ys[above], bw_xs[above]] = REGION["hips"]
            below = ~above
            if below.any():
                is_char_right = bw_xs[below] < midline_x
                result[bw_ys[below][is_char_right], bw_xs[below][is_char_right]] = REGION["upper_leg_r"]
                result[bw_ys[below][~is_char_right], bw_xs[below][~is_char_right]] = REGION["upper_leg_l"]

    # Step 6: Legwear → lower_leg (with L/R)
    # Some legwear covers upper_leg too, so split at knee joint
    legwear = li_mask == 140
    if legwear.any():
        lower_leg_l_jt = joints.get("lower_leg_l")
        lower_leg_r_jt = joints.get("lower_leg_r")
        upper_leg_l_jt = joints.get("upper_leg_l")
        upper_leg_r_jt = joints.get("upper_leg_r")

        lw_ys, lw_xs = np.where(legwear)

        # Determine knee y positions for upper vs lower split
        knee_y = None
        if lower_leg_l_jt and upper_leg_l_jt:
            knee_y = (lower_leg_l_jt[1] + upper_leg_l_jt[1]) / 2.0
        elif lower_leg_r_jt and upper_leg_r_jt:
            knee_y = (lower_leg_r_jt[1] + upper_leg_r_jt[1]) / 2.0

        for idx in range(len(lw_ys)):
            y, x = lw_ys[idx], lw_xs[idx]
            is_char_right = x < midline_x

            if knee_y and y < knee_y:
                # Above knee → upper_leg
                result[y, x] = REGION["upper_leg_r"] if is_char_right else REGION["upper_leg_l"]
            else:
                # Below knee → lower_leg
                result[y, x] = REGION["lower_leg_r"] if is_char_right else REGION["lower_leg_l"]

    # Step 7: Footwear → foot (with L/R)
    split_left_right(result, li_mask, 150, REGION["foot_l"], REGION["foot_r"], midline_x)

    return result


def convert_with_heuristics(li_mask: np.ndarray) -> np.ndarray:
    """Convert Li labels to Strata 22-class using geometric heuristics only (no joints).

    Uses character bounding box proportions to estimate joint positions.
    Less accurate than joint-based conversion but works without a trained model.
    """
    h, w = li_mask.shape
    result = np.zeros((h, w), dtype=np.uint8)

    # Step 1: Direct merges
    for li_val, strata_id in DIRECT_MERGE.items():
        result[li_mask == li_val] = strata_id

    # Estimate body proportions from bounding box
    bbox = get_fg_bbox(li_mask)
    x_min, y_min, x_max, y_max = bbox
    body_h = y_max - y_min
    body_w = x_max - x_min

    # Midline from face center or bbox center
    face_center = get_face_center(li_mask)
    midline_x = face_center[0] if face_center else (x_min + x_max) / 2.0

    # Estimate vertical proportions (fraction of body height from top)
    neck_y = y_min + body_h * 0.12
    shoulder_y = y_min + body_h * 0.16
    chest_y = y_min + body_h * 0.25
    spine_y = y_min + body_h * 0.38
    hips_y = y_min + body_h * 0.45
    knee_y = y_min + body_h * 0.70

    # Estimate shoulder width from torso pixels at chest height
    # Use the topwear region to measure actual torso width (handles T-pose correctly)
    topwear_pixels = li_mask == 110
    if topwear_pixels.any():
        tw_ys, tw_xs = np.where(topwear_pixels)
        # Sample topwear at chest height (top 30% of topwear) for torso width
        tw_y_min, tw_y_max = tw_ys.min(), tw_ys.max()
        chest_band_top = tw_y_min
        chest_band_bot = tw_y_min + (tw_y_max - tw_y_min) * 0.30
        in_chest = (tw_ys >= chest_band_top) & (tw_ys <= chest_band_bot)
        if in_chest.any():
            torso_x_min = tw_xs[in_chest].min()
            torso_x_max = tw_xs[in_chest].max()
            torso_w = torso_x_max - torso_x_min
            shoulder_half_w = torso_w * 0.45
        else:
            shoulder_half_w = body_w * 0.20
    else:
        shoulder_half_w = body_w * 0.20
    shoulder_l_x = midline_x + shoulder_half_w
    shoulder_r_x = midline_x - shoulder_half_w

    # Step 2: Hair
    hair_pixels = li_mask == 0
    if hair_pixels.any():
        face_pixels = li_mask == 20
        if face_pixels.any():
            face_bottom = np.where(face_pixels)[0].max()
            hair_ys, hair_xs = np.where(hair_pixels)
            above = hair_ys <= face_bottom
            result[hair_ys[above], hair_xs[above]] = REGION["head"]
            result[hair_ys[~above], hair_xs[~above]] = REGION["hair_back"]
        else:
            result[hair_pixels] = REGION["head"]

    # Step 3: Topwear → chest / spine / shoulder / upper_arm
    topwear = li_mask == 110
    if topwear.any():
        tw_ys, tw_xs = np.where(topwear)
        mid_torso = (chest_y + spine_y) / 2.0

        for idx in range(len(tw_ys)):
            y, x = tw_ys[idx], tw_xs[idx]
            if x > shoulder_l_x + 5:
                # Past character's left shoulder → left arm area
                result[y, x] = REGION["shoulder_l"] if y < shoulder_y + body_h * 0.05 else REGION["upper_arm_l"]
            elif x < shoulder_r_x - 5:
                # Past character's right shoulder → right arm area
                result[y, x] = REGION["shoulder_r"] if y < shoulder_y + body_h * 0.05 else REGION["upper_arm_r"]
            else:
                result[y, x] = REGION["chest"] if y < mid_torso else REGION["spine"]

    # Step 4: Handwear → forearm / hand
    handwear = li_mask == 120
    if handwear.any():
        hw_ys, hw_xs = np.where(handwear)
        # Use centroid of each connected component's y for forearm/hand split
        # Simple heuristic: bottom 30% of handwear region = hand, rest = forearm
        hw_y_min, hw_y_max = hw_ys.min(), hw_ys.max()
        hand_threshold = hw_y_min + (hw_y_max - hw_y_min) * 0.65

        for idx in range(len(hw_ys)):
            y, x = hw_ys[idx], hw_xs[idx]
            is_char_right = x < midline_x
            if y > hand_threshold:
                result[y, x] = REGION["hand_r"] if is_char_right else REGION["hand_l"]
            else:
                result[y, x] = REGION["forearm_r"] if is_char_right else REGION["forearm_l"]

    # Step 5: Bottomwear → hips / upper_leg
    bottomwear = li_mask == 130
    if bottomwear.any():
        bw_ys, bw_xs = np.where(bottomwear)
        above_hips = bw_ys <= hips_y
        result[bw_ys[above_hips], bw_xs[above_hips]] = REGION["hips"]
        below = ~above_hips
        if below.any():
            is_char_right = bw_xs[below] < midline_x
            result[bw_ys[below][is_char_right], bw_xs[below][is_char_right]] = REGION["upper_leg_r"]
            result[bw_ys[below][~is_char_right], bw_xs[below][~is_char_right]] = REGION["upper_leg_l"]

    # Step 6: Legwear → upper_leg / lower_leg
    legwear = li_mask == 140
    if legwear.any():
        lw_ys, lw_xs = np.where(legwear)
        for idx in range(len(lw_ys)):
            y, x = lw_ys[idx], lw_xs[idx]
            is_char_right = x < midline_x
            if y < knee_y:
                result[y, x] = REGION["upper_leg_r"] if is_char_right else REGION["upper_leg_l"]
            else:
                result[y, x] = REGION["lower_leg_r"] if is_char_right else REGION["lower_leg_l"]

    # Step 7: Footwear → foot L/R
    split_left_right(result, li_mask, 150, REGION["foot_l"], REGION["foot_r"], midline_x)

    return result


def create_preview(
    source_img: Image.Image,
    li_mask: np.ndarray,
    strata_mask: np.ndarray,
    resolution: int = 512,
) -> Image.Image:
    """Create a side-by-side preview: source | Li colored | Strata colored."""

    # Color palettes
    li_colors = {
        0: (0, 0, 255), 10: (139, 69, 19), 20: (200, 180, 0), 30: (0, 128, 128),
        40: (0, 0, 128), 50: (255, 0, 0), 60: (255, 165, 0), 70: (255, 255, 0),
        80: (192, 255, 0), 90: (0, 128, 0), 100: (0, 255, 255), 110: (128, 0, 0),
        120: (128, 0, 128), 130: (255, 0, 255), 140: (255, 192, 203),
        150: (255, 200, 150), 160: (255, 255, 200), 170: (0, 255, 128),
        180: (200, 200, 255), 255: (40, 40, 40),
    }

    strata_colors = {
        0: (40, 40, 40), 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255),
        4: (255, 255, 0), 5: (255, 0, 255), 6: (0, 255, 255), 7: (128, 0, 0),
        8: (0, 128, 0), 9: (0, 0, 128), 10: (128, 128, 0), 11: (128, 0, 128),
        12: (0, 128, 128), 13: (64, 0, 128), 14: (255, 128, 0), 15: (128, 255, 0),
        16: (0, 128, 255), 17: (255, 0, 128), 18: (0, 255, 128), 19: (128, 0, 255),
        20: (200, 200, 200), 21: (255, 128, 128),
    }

    def colorize(mask, colors):
        colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for val, color in colors.items():
            colored[mask == val] = color
        return Image.fromarray(colored)

    # Resize all to same size
    src_resized = source_img.convert("RGB").resize((resolution, resolution), Image.LANCZOS)
    li_colored = colorize(li_mask, li_colors).resize((resolution, resolution), Image.NEAREST)
    strata_colored = colorize(strata_mask, strata_colors).resize((resolution, resolution), Image.NEAREST)

    gap = 10
    grid = Image.new("RGB", (resolution * 3 + gap * 2, resolution), (255, 255, 255))
    grid.paste(src_resized, (0, 0))
    grid.paste(li_colored, (resolution + gap, 0))
    grid.paste(strata_colored, (resolution * 2 + gap * 2, 0))
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Dr. Li's 19-class labels to Strata 22-class format"
    )
    parser.add_argument(
        "--labels", required=True,
        help="Directory containing Dr. Li's label PNGs",
    )
    parser.add_argument(
        "--gemini-dir", required=True,
        help="gemini_diverse directory (source images + optional joints.json)",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for converted per-example training data",
    )
    parser.add_argument(
        "--resolution", type=int, default=512,
        help="Output resolution (default: 512)",
    )
    parser.add_argument(
        "--preview", type=int, default=0,
        help="Generate N preview images to /tmp/li_conversion_preview/ instead of full conversion",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing segmentation.png files",
    )
    parser.add_argument(
        "--raw-dir",
        help="Directory with raw (pre-ingest) Gemini images. Required for correct "
             "spatial alignment — labels match raw image dimensions, not 512x512.",
    )
    parser.add_argument(
        "--align-only", action="store_true",
        help="Only align raw images and save li_label.png (skip Strata conversion). "
             "Run joints inference on the aligned images, then re-run without this flag.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    labels_dir = Path(args.labels)
    gemini_dir = Path(args.gemini_dir)
    output_dir = Path(args.output_dir)
    resolution = args.resolution
    raw_dir = Path(args.raw_dir) if args.raw_dir else None

    # Discover label files
    label_files = sorted([
        f for f in labels_dir.iterdir()
        if f.suffix == ".png" and not f.name.startswith(".")
    ])
    logger.info("Found %d label files", len(label_files))

    # Match labels to gemini dirs (and optionally raw images)
    matched = []
    for lf in label_files:
        dir_name = label_filename_to_dir(lf.name)
        example_dir = gemini_dir / dir_name
        if example_dir.is_dir() and (example_dir / "image.png").exists():
            # Find matching raw image if raw_dir provided
            raw_path = None
            if raw_dir:
                raw_path = raw_dir / lf.name
                if not raw_path.exists():
                    logger.warning("No raw image for %s", lf.name)
                    raw_path = None
            matched.append((lf, example_dir, dir_name, raw_path))
        else:
            logger.warning("No match for label %s (tried %s)", lf.name, dir_name)

    logger.info("Matched %d/%d labels to source images", len(matched), len(label_files))
    if raw_dir:
        with_raw = sum(1 for _, _, _, rp in matched if rp is not None)
        logger.info("With raw images: %d/%d", with_raw, len(matched))

    # Check how many have joints
    with_joints = sum(1 for _, ed, _, _ in matched if (ed / "joints.json").exists())
    logger.info("Examples with joints.json: %d/%d", with_joints, len(matched))

    if args.preview > 0:
        # Preview mode
        preview_dir = Path("/tmp/li_conversion_preview")
        preview_dir.mkdir(parents=True, exist_ok=True)
        step = max(1, len(matched) // args.preview)
        samples = matched[::step][:args.preview]

        for lf, example_dir, dir_name, raw_path in samples:
            li_mask_raw = np.array(Image.open(lf))

            if raw_path:
                raw_img = Image.open(raw_path)
                source_img, li_mask = align_label_to_source(
                    raw_img, li_mask_raw, resolution,
                )
            else:
                source_img = Image.open(example_dir / "image.png")
                li_mask = li_mask_raw

            joints = load_joints(example_dir / "joints.json")

            if joints:
                # Scale joints to label resolution
                src_w, src_h = source_img.size
                lbl_h, lbl_w = li_mask.shape
                scaled_joints = {
                    name: (int(pos[0] * lbl_w / src_w), int(pos[1] * lbl_h / src_h))
                    for name, pos in joints.items()
                }
                strata_mask = convert_with_joints(li_mask, scaled_joints)
            else:
                strata_mask = convert_with_heuristics(li_mask)

            preview = create_preview(source_img, li_mask, strata_mask, resolution)
            preview.save(preview_dir / f"{dir_name}.png")
            mode = "joints" if joints else "heuristic"
            logger.info("Preview: %s (%s)", dir_name, mode)

        logger.info("Previews saved to %s", preview_dir)
        return

    # Full conversion
    output_dir.mkdir(parents=True, exist_ok=True)
    align_only = args.align_only
    stats = {"converted": 0, "skipped_exists": 0, "errors": 0, "with_joints": 0, "heuristic": 0}

    for i, (lf, example_dir, dir_name, raw_path) in enumerate(matched):
        out_example = output_dir / dir_name

        if align_only:
            # Skip if already aligned
            if not args.overwrite and (out_example / "li_label.png").exists():
                stats["skipped_exists"] += 1
                continue
        else:
            if not args.overwrite and (out_example / "segmentation.png").exists():
                stats["skipped_exists"] += 1
                continue

        try:
            li_mask_raw = np.array(Image.open(lf))

            if raw_path:
                # Align label to source using raw image for correct crop
                raw_img = Image.open(raw_path)
                source_resized, li_mask = align_label_to_source(
                    raw_img, li_mask_raw, resolution,
                )
            else:
                # Fallback: direct resize (may be misaligned if dimensions differ)
                source_img = Image.open(example_dir / "image.png")
                source_resized = source_img.convert("RGBA").resize(
                    (resolution, resolution), Image.LANCZOS,
                )
                li_mask = li_mask_raw

            out_example.mkdir(parents=True, exist_ok=True)

            if align_only:
                # Save aligned image + Li label only (no Strata conversion)
                source_resized.save(out_example / "image.png")
                Image.fromarray(li_mask, mode="L").save(out_example / "li_label.png")
                stats["converted"] += 1

                if (i + 1) % 100 == 0 or i + 1 == len(matched):
                    logger.info("  %d/%d aligned", i + 1, len(matched))
                continue

            # Load joints — prefer from output dir (re-inferred on aligned images)
            joints = load_joints(out_example / "joints.json")
            if joints is None:
                joints = load_joints(example_dir / "joints.json")

            # Load aligned Li label if available (from --align-only pass)
            li_label_path = out_example / "li_label.png"
            if li_label_path.exists():
                li_mask = np.array(Image.open(li_label_path))

            src_w, src_h = source_resized.size
            if joints:
                # Joints are at 512x512, li_mask is also 512x512 after alignment
                lbl_h, lbl_w = li_mask.shape
                scaled_joints = {
                    name: (int(pos[0] * lbl_w / src_w), int(pos[1] * lbl_h / src_h))
                    for name, pos in joints.items()
                }
                strata_mask = convert_with_joints(li_mask, scaled_joints)
                stats["with_joints"] += 1
            else:
                strata_mask = convert_with_heuristics(li_mask)
                stats["heuristic"] += 1

            # Ensure output is at target resolution
            strata_mask_img = Image.fromarray(strata_mask, mode="L")
            if strata_mask_img.size != (resolution, resolution):
                strata_mask_img = strata_mask_img.resize(
                    (resolution, resolution), Image.NEAREST
                )

            if source_resized.size != (resolution, resolution):
                source_resized = source_resized.resize(
                    (resolution, resolution), Image.LANCZOS,
                )

            # Save
            source_resized.save(out_example / "image.png")
            strata_mask_img.save(out_example / "segmentation.png")

            # Copy metadata if exists, otherwise create
            meta_src = example_dir / "metadata.json"
            meta_dst = out_example / "metadata.json"
            if meta_src.exists() and meta_src.resolve() != meta_dst.resolve():
                import shutil
                shutil.copy2(meta_src, meta_dst)
            elif not meta_dst.exists():
                final_mask = np.array(strata_mask_img)
                metadata = {
                    "source_type": "gemini_generated",
                    "source_file": dir_name,
                    "regions": sorted(int(r) for r in np.unique(final_mask)),
                    "annotation_quality": "li_converted",
                    "annotation_source": "dr_li_19class",
                    "conversion_mode": "joints" if joints else "heuristic",
                }
                with open(out_example / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)

            # Copy joints if exists
            joints_src = example_dir / "joints.json"
            joints_dst = out_example / "joints.json"
            if joints_src.exists() and joints_src.resolve() != joints_dst.resolve():
                import shutil
                shutil.copy2(joints_src, joints_dst)

            stats["converted"] += 1

        except Exception:
            logger.exception("Error converting %s", dir_name)
            stats["errors"] += 1

        if (i + 1) % 100 == 0 or i + 1 == len(matched):
            logger.info("  %d/%d processed (%d converted)", i + 1, len(matched), stats["converted"])

    logger.info("Done!")
    logger.info("  Converted: %d", stats["converted"])
    logger.info("  With joints: %d", stats["with_joints"])
    logger.info("  Heuristic: %d", stats["heuristic"])
    logger.info("  Skipped (exists): %d", stats["skipped_exists"])
    logger.info("  Errors: %d", stats["errors"])
    logger.info("  Output: %s", output_dir)


if __name__ == "__main__":
    main()
