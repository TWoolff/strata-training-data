#!/usr/bin/env python3
"""Convert See-Through layer decomposition to Strata 22-class segmentation masks.

Loads the transparent PNG layers output by See-Through and composites their
alpha channels into a single segmentation mask using our 22-class anatomy schema.

See-Through outputs up to ~23 clothing-oriented layers. The mapping to our
anatomy schema is approximate — See-Through doesn't distinguish forearm from
upper_arm (both are under "topwear" or "handwear"). But the *boundaries* are
high quality (trained on 9K illustrated characters), which is what matters most
for improving our seg model.

Usage::

    python scripts/convert_seethrough_to_seg.py \\
        --input_dir output/seethrough_test/ \\
        --output_dir output/seethrough_seg/

Mapping strategy:
- face, nose, mouth → head (1)
- eyes (eyewhite, irides, eyelash, eyebrow) → head (1)
- ears → head (1)
- front hair → head (1)
- back hair → hair_back (21)
- neck → neck (2)
- headwear → accessory (20)
- topwear → chest (3) + spine (4) — split by vertical midpoint
- handwear-l → hand_l (9) + forearm_l (8) + upper_arm_l (7) — split by bbox
- handwear-r → hand_r (13) + forearm_r (12) + upper_arm_r (11)
- bottomwear/legwear → hips (5) + upper_leg (14/17) + lower_leg (15/18) — split by vertical thirds
- footwear → foot_l (16) + foot_r (19) — split by horizontal center
- objects → accessory (20)
- wings/tail → accessory (20)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Strata 22-class IDs
BG = 0
HEAD = 1
NECK = 2
CHEST = 3
SPINE = 4
HIPS = 5
SHOULDER_L = 6
UPPER_ARM_L = 7
FOREARM_L = 8
HAND_L = 9
SHOULDER_R = 10
UPPER_ARM_R = 11
FOREARM_R = 12
HAND_R = 13
UPPER_LEG_L = 14
LOWER_LEG_L = 15
FOOT_L = 16
UPPER_LEG_R = 17
LOWER_LEG_R = 18
FOOT_R = 19
ACCESSORY = 20
HAIR_BACK = 21


def load_layer_alpha(layer_dir: Path, tag: str) -> np.ndarray | None:
    """Load a layer's alpha channel. Returns (H, W) uint8 or None."""
    path = layer_dir / f"{tag}.png"
    if not path.exists():
        return None
    img = np.array(Image.open(path).convert("RGBA"))
    return img[:, :, 3]


def split_left_right(alpha: np.ndarray, bbox: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Split a mask into left and right halves by horizontal center of mass."""
    if alpha.max() == 0:
        return alpha, alpha
    ys, xs = np.where(alpha > 15)
    cx = int(xs.mean())
    left = alpha.copy()
    right = alpha.copy()
    left[:, cx:] = 0
    right[:, :cx] = 0
    return left, right


def split_vertical_thirds(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a mask into top, middle, bottom thirds by vertical extent."""
    if alpha.max() == 0:
        return alpha, alpha, alpha
    ys, xs = np.where(alpha > 15)
    y_min, y_max = ys.min(), ys.max()
    h = y_max - y_min
    t1 = y_min + h // 3
    t2 = y_min + 2 * h // 3

    top = alpha.copy()
    top[t1:, :] = 0
    mid = alpha.copy()
    mid[:t1, :] = 0
    mid[t2:, :] = 0
    bot = alpha.copy()
    bot[:t2, :] = 0
    return top, mid, bot


def split_arm_segments(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split handwear into upper_arm, forearm, hand by vertical extent."""
    if alpha.max() == 0:
        return alpha, alpha, alpha
    ys, xs = np.where(alpha > 15)
    y_min, y_max = ys.min(), ys.max()
    h = y_max - y_min

    # Hand = bottom 25%, forearm = middle 35%, upper_arm = top 40%
    t_upper = y_min + int(h * 0.4)
    t_fore = y_min + int(h * 0.75)

    upper = alpha.copy()
    upper[t_upper:, :] = 0
    fore = alpha.copy()
    fore[:t_upper, :] = 0
    fore[t_fore:, :] = 0
    hand = alpha.copy()
    hand[:t_fore, :] = 0
    return upper, fore, hand


def convert_character(layer_dir: Path, json_path: Path, output_dir: Path) -> dict | None:
    """Convert one character's See-Through layers to a 22-class seg mask."""
    info = json.loads(json_path.read_text())
    parts = info.get("parts", {})
    frame_size = info.get("frame_size", [1280, 1280])
    h, w = frame_size[1], frame_size[0]  # JSON has [w, h]

    seg = np.zeros((h, w), dtype=np.uint8)

    def apply(alpha: np.ndarray | None, class_id: int):
        """Write class_id where alpha > 15 and seg is still background."""
        if alpha is None:
            return
        mask = (alpha > 15) & (seg == BG)
        seg[mask] = class_id

    # Process layers in depth order (front to back) so front layers win

    # Head components
    for tag in ["face", "nose", "mouth", "eyewhite-r", "eyewhite-l",
                "irides-r", "irides-l", "eyelash-r", "eyelash-l",
                "eyebrow-r", "eyebrow-l", "eyewhite", "irides",
                "eyelash", "eyebrow", "ears", "ears-r", "ears-l",
                "front hair"]:
        apply(load_layer_alpha(layer_dir, tag), HEAD)

    # Hair back
    apply(load_layer_alpha(layer_dir, "back hair"), HAIR_BACK)

    # Neck
    apply(load_layer_alpha(layer_dir, "neck"), NECK)

    # Headwear → accessory
    apply(load_layer_alpha(layer_dir, "headwear"), ACCESSORY)

    # Arms (handwear includes full arm in See-Through)
    for side, suffix, ids in [
        ("l", "-l", (UPPER_ARM_L, FOREARM_L, HAND_L, SHOULDER_L)),
        ("r", "-r", (UPPER_ARM_R, FOREARM_R, HAND_R, SHOULDER_R)),
    ]:
        arm_alpha = load_layer_alpha(layer_dir, f"handwear{suffix}")
        if arm_alpha is None:
            # Try unsplit handwear and split by left/right
            full_alpha = load_layer_alpha(layer_dir, "handwear")
            if full_alpha is not None:
                left, right = split_left_right(full_alpha)
                arm_alpha = left if side == "l" else right

        if arm_alpha is not None:
            upper, fore, hand = split_arm_segments(arm_alpha)
            apply(upper, ids[0])   # upper_arm
            apply(fore, ids[1])    # forearm
            apply(hand, ids[2])    # hand

    # Topwear → chest + spine (split at vertical midpoint of topwear)
    topwear_alpha = load_layer_alpha(layer_dir, "topwear")
    if topwear_alpha is not None:
        ys, xs = np.where(topwear_alpha > 15)
        if len(ys) > 0:
            y_mid = int(ys.mean())
            chest_alpha = topwear_alpha.copy()
            chest_alpha[y_mid:, :] = 0
            spine_alpha = topwear_alpha.copy()
            spine_alpha[:y_mid, :] = 0
            apply(chest_alpha, CHEST)
            apply(spine_alpha, SPINE)

    # Bottomwear/legwear → hips + upper_leg + lower_leg
    for tag in ["bottomwear", "legwear"]:
        leg_alpha = load_layer_alpha(layer_dir, tag)
        if leg_alpha is not None:
            top, mid, bot = split_vertical_thirds(leg_alpha)
            # Top third = hips
            apply(top, HIPS)
            # Middle and bottom = legs, split left/right
            mid_l, mid_r = split_left_right(mid)
            bot_l, bot_r = split_left_right(bot)
            apply(mid_l, UPPER_LEG_L)
            apply(mid_r, UPPER_LEG_R)
            apply(bot_l, LOWER_LEG_L)
            apply(bot_r, LOWER_LEG_R)

    # Footwear → feet
    foot_alpha = load_layer_alpha(layer_dir, "footwear")
    if foot_alpha is not None:
        foot_l, foot_r = split_left_right(foot_alpha)
        apply(foot_l, FOOT_L)
        apply(foot_r, FOOT_R)

    # Objects, wings, tail → accessory
    for tag in ["objects", "wings", "tail", "neckwear", "earwear", "eyewear"]:
        apply(load_layer_alpha(layer_dir, tag), ACCESSORY)

    # Count classes present
    unique = set(np.unique(seg)) - {0}
    if len(unique) < 3:
        logger.warning("  Only %d classes found for %s — may be low quality", len(unique), layer_dir.name)

    # Save
    char_name = layer_dir.name
    out_dir = output_dir / char_name
    out_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(seg).save(out_dir / "segmentation.png")

    # Also copy source image if available
    src_img = layer_dir / "src_img.png"
    if src_img.exists():
        Image.open(src_img).save(out_dir / "image.png")

    metadata = {
        "character_id": char_name,
        "source": "seethrough",
        "classes_present": sorted([int(x) for x in unique]),
        "n_classes": len(unique),
        "frame_size": frame_size,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    logger.info("  %s: %d classes, saved to %s", char_name, len(unique), out_dir)
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Convert See-Through layers to Strata 22-class segmentation"
    )
    parser.add_argument("--input_dir", type=Path, required=True,
                        help="See-Through output directory (contains char subdirs with PNGs)")
    parser.add_argument("--output_dir", type=Path, default=Path("output/seethrough_seg"),
                        help="Output directory for segmentation masks")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find character directories (contain PNG layers)
    char_dirs = sorted([
        d for d in args.input_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    logger.info("Found %d character directories in %s", len(char_dirs), args.input_dir)

    n_success = 0
    for char_dir in char_dirs:
        json_path = args.input_dir / f"{char_dir.name}.psd.json"
        if not json_path.exists():
            logger.warning("  No JSON for %s, skipping", char_dir.name)
            continue

        try:
            result = convert_character(char_dir, json_path, args.output_dir)
            if result:
                n_success += 1
        except Exception as e:
            logger.error("  Failed %s: %s", char_dir.name, e)

    logger.info("Done: %d/%d converted", n_success, len(char_dirs))


if __name__ == "__main__":
    main()
