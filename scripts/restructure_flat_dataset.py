"""Restructure flat-layout datasets into per-example subdirectory format.

Converts datasets with flat directories (images/, masks/, depth/, normals/, joints/)
into the per-example format expected by the segmentation dataset loader:

    {example_id}/image.png
    {example_id}/segmentation.png
    {example_id}/depth.png
    {example_id}/normals.png
    {example_id}/joints.json

Usage::

    python scripts/restructure_flat_dataset.py --input-dir ./data_cloud/meshy_cc0
    python scripts/restructure_flat_dataset.py --input-dir ./data_cloud/meshy_cc0_textured
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def restructure(input_dir: Path) -> None:
    """Restructure a flat-layout dataset in place."""
    images_dir = input_dir / "images"
    masks_dir = input_dir / "masks"
    depth_dir = input_dir / "depth"
    normals_dir = input_dir / "normals"
    joints_dir = input_dir / "joints"

    if not images_dir.exists():
        print(f"  No images/ directory in {input_dir}, skipping.")
        return

    # Discover all images
    image_files = sorted(images_dir.glob("*.png"))
    print(f"  Found {len(image_files)} images in {input_dir.name}")

    moved = 0
    for img_path in image_files:
        # Example: rigged_xxx_pose_00_flat.png -> rigged_xxx_pose_00
        stem = img_path.stem
        # Strip style suffix (_flat, _cel, _sketch, etc.) to get base name
        # The mask/depth/normals use the base name without style suffix
        base_name = stem
        for suffix in ("_flat", "_cel", "_sketch", "_pixel", "_painterly", "_unlit"):
            if base_name.endswith(suffix):
                base_name = base_name[: -len(suffix)]
                break

        # Create per-example subdir using the full stem (including style) as ID
        example_dir = input_dir / stem
        example_dir.mkdir(exist_ok=True)

        # Move image
        shutil.move(str(img_path), str(example_dir / "image.png"))

        # Move mask (base_name.png or base_name_22.png)
        mask_candidates = [
            masks_dir / f"{base_name}_22.png",  # 22-class mask
            masks_dir / f"{base_name}.png",  # generic mask
        ]
        for mask_path in mask_candidates:
            if mask_path.exists():
                shutil.move(str(mask_path), str(example_dir / "segmentation.png"))
                break

        # Move depth
        depth_path = depth_dir / f"{base_name}.png"
        if depth_path.exists():
            shutil.move(str(depth_path), str(example_dir / "depth.png"))

        # Move normals
        normals_path = normals_dir / f"{base_name}.png"
        if normals_path.exists():
            shutil.move(str(normals_path), str(example_dir / "normals.png"))

        # Move joints (JSON, often only exists for _pose_00)
        # Try exact match first, then base character ID
        joints_path = joints_dir / f"{base_name}.json"
        if joints_path.exists():
            shutil.copy2(str(joints_path), str(example_dir / "joints.json"))

        moved += 1

    # Clean up empty flat directories
    for d in [images_dir, masks_dir, depth_dir, normals_dir, joints_dir]:
        if d.exists() and not any(d.iterdir()):
            d.rmdir()

    print(f"  Restructured {moved} examples into per-example subdirs.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restructure flat-layout dataset into per-example subdirs"
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Dataset directory")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Directory not found: {input_dir}")
        return

    restructure(input_dir)


if __name__ == "__main__":
    main()
