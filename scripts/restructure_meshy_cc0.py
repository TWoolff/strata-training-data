#!/usr/bin/env python3
"""Restructure meshy_cc0_textured from flat dirs to per-example subdirs.

Input (flat):
  images/rigged_XXX_pose_00_back_textured.png
  images/rigged_XXX_pose_00_front_22_textured.png
  masks/rigged_XXX_pose_00.png
  masks/rigged_XXX_pose_00_back.png
  joints/rigged_XXX_pose_00.json

Output (per-example):
  meshy_cc0_XXX_pose_00_front/image.png
  meshy_cc0_XXX_pose_00_front/segmentation.png
  meshy_cc0_XXX_pose_00_front/joints.json
  meshy_cc0_XXX_pose_00_back/image.png
  meshy_cc0_XXX_pose_00_back/segmentation.png

Supports both _textured and _flat suffixes via --suffix flag (default: textured).
"""

import argparse
import json
import re
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Restructure meshy_cc0 flat dirs to per-example subdirs")
    parser.add_argument("--input_dir", type=Path, required=True, help="Path to flat meshy_cc0 dir")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output dir for per-example subdirs")
    parser.add_argument("--suffix", default="textured", help="Image suffix to match: 'textured' or 'flat' (default: textured)")
    parser.add_argument("--dry_run", action="store_true", help="Print what would be done without copying")
    args = parser.parse_args()

    images_dir = args.input_dir / "images"
    masks_dir = args.input_dir / "masks"
    joints_dir = args.input_dir / "joints"
    suffix = f"_{args.suffix}"

    if not images_dir.exists():
        print(f"ERROR: {images_dir} not found")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build mask lookup: stem (without extension) → path
    mask_lookup = {p.stem: p for p in masks_dir.glob("*.png")}

    # Build joints lookup: stem → path
    joints_lookup = {}
    if joints_dir.exists():
        for p in joints_dir.glob("*.json"):
            joints_lookup[p.stem] = p

    # Pattern: rigged_UUID_pose_NN[_angle]_textured.png
    # We need to extract: char_id (rigged_UUID), pose (pose_NN), angle
    pose_pattern = re.compile(r"^(rigged_[0-9a-f-]+)_(pose_\d+)(.*)" + re.escape(suffix) + "$")

    copied = 0
    skipped = 0
    no_mask = 0

    for img_path in sorted(images_dir.glob(f"*{suffix}.png")):
        stem = img_path.stem

        m = pose_pattern.match(stem)
        if not m:
            print(f"  SKIP (no match): {img_path.name}")
            skipped += 1
            continue

        char_id = m.group(1)       # rigged_UUID
        pose = m.group(2)          # pose_00, pose_01, etc.
        angle_suffix = m.group(3)  # "" or "_back" or "_front_22" etc.
        angle = angle_suffix.lstrip("_") if angle_suffix else "front"

        # Mask key: same as image stem but without the suffix
        mask_key = f"{char_id}_{pose}{angle_suffix}"

        if mask_key not in mask_lookup:
            no_mask += 1
            continue

        # Example ID for output dir
        example_id = f"meshy_cc0_{char_id}_{pose}_{angle}"
        example_dir = args.output_dir / example_id

        if args.dry_run:
            print(f"  {img_path.name} → {example_id}/")
            copied += 1
            continue

        example_dir.mkdir(exist_ok=True)

        # Copy image → image.png
        shutil.copy2(img_path, example_dir / "image.png")

        # Copy mask → segmentation.png
        shutil.copy2(mask_lookup[mask_key], example_dir / "segmentation.png")

        # Copy joints if available (per pose, not per angle)
        joints_key = f"{char_id}_{pose}"
        if joints_key in joints_lookup:
            shutil.copy2(joints_lookup[joints_key], example_dir / "joints.json")

        # Write minimal metadata
        metadata = {
            "source": "meshy_cc0_textured",
            "character_id": char_id,
            "pose": pose,
            "angle": angle,
            "license": "CC0",
        }
        with open(example_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        copied += 1
        if copied % 1000 == 0:
            print(f"  {copied} examples copied...")

    print(f"\nDone: {copied} examples copied, {skipped} skipped, {no_mask} missing masks")


if __name__ == "__main__":
    main()
