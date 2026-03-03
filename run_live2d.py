"""CLI entry point for the Live2D rendering pipeline.

Processes a directory of Live2D models (.moc3 + texture atlas) into
Strata training data: composite images, segmentation masks, draw order
maps, and joint position JSON.

Each model directory must contain:
- A .model3.json entry point
- A .moc3 binary file
- Texture atlas PNG(s) referenced by the .model3.json

Output per model (with augmentation enabled, 4 variants per model):
- images/live2d_{id}_pose_{nn}_flat.png    ← RGBA composite
- masks/live2d_{id}_pose_{nn}.png          ← 8-bit segmentation mask
- draw_order/live2d_{id}_pose_{nn}.png     ← draw order map
- joints/live2d_{id}_pose_{nn}.json        ← 2D joint positions (from region centroids)
- sources/live2d_{id}.json                 ← model metadata

Note: Live2D models are 2D illustrations — true multi-angle views (side, back)
don't exist. Augmentation produces: original + horizontal flip + rotation ±5°
+ scale ±10%, giving 4 training variants per model.

Usage::

    python3 run_live2d.py \\
        --input_dir /Volumes/TAMWoolff/data/live2d \\
        --output_dir ./output/live2d

    # Disable augmentation (1 variant per model):
    python3 run_live2d.py \\
        --input_dir /Volumes/TAMWoolff/data/live2d \\
        --output_dir ./output/live2d \\
        --no_augmentation

    # Process a subset for testing:
    python3 run_live2d.py \\
        --input_dir /Volumes/TAMWoolff/data/live2d \\
        --output_dir ./output/live2d \\
        --max_models 10
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def parse_args() -> argparse.Namespace:
    from pipeline.config import RENDER_RESOLUTION

    parser = argparse.ArgumentParser(
        description="Render Live2D models into Strata training format.",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing Live2D model subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./output/live2d"),
        help="Output directory for Strata-formatted examples (default: ./output/live2d).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=RENDER_RESOLUTION,
        help=f"Output image resolution (default: {RENDER_RESOLUTION}).",
    )
    parser.add_argument(
        "--no_augmentation",
        action="store_true",
        default=False,
        help="Disable augmentation — produce 1 variant per model instead of 4.",
    )
    parser.add_argument(
        "--only_new",
        action="store_true",
        default=False,
        help="Skip models whose output files already exist.",
    )
    parser.add_argument(
        "--max_models",
        type=int,
        default=0,
        help="Maximum number of models to process (0 = all).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()

    if not args.input_dir.is_dir():
        print(f"ERROR: input_dir not found: {args.input_dir}")
        sys.exit(1)

    from pipeline import exporter
    from pipeline.live2d_renderer import (
        _extract_joints_from_mask,
        generate_augmentations,
        process_live2d_model,
    )

    model_dirs = sorted(
        p for p in args.input_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    if args.max_models > 0:
        model_dirs = model_dirs[: args.max_models]

    aug_label = "off (1 variant)" if args.no_augmentation else "on (4 variants)"
    print(f"Live2D pipeline: {len(model_dirs)} models → {args.output_dir}")
    print(f"  augmentation: {aug_label}")
    print(f"  resolution:   {args.resolution}×{args.resolution}")

    # Minimum fraction of fragments that must be mapped for a model to be saved.
    # Models below this threshold have essentially blank masks and are useless.
    MIN_MAPPED_RATIO = 0.05

    exporter.ensure_output_dirs(args.output_dir)
    start = time.monotonic()
    results = []
    failed = 0
    skipped_low_mapping = 0

    for i, model_dir in enumerate(model_dirs):
        result = process_live2d_model(model_dir, resolution=args.resolution)
        if result is None:
            failed += 1
            continue

        mapped_ratio = result.mapped_count / result.fragment_count if result.fragment_count else 0
        if mapped_ratio < MIN_MAPPED_RATIO:
            skipped_low_mapping += 1
            continue

        if not args.no_augmentation:
            variants = generate_augmentations(
                result.image, result.mask, result.draw_order_map, args.resolution
            )
        else:
            variants = [("identity", result.image, result.mask, result.draw_order_map)]

        for pose_index, (_label, aug_image, aug_mask, aug_draw_order) in enumerate(variants):
            exporter.save_mask(aug_mask, args.output_dir, result.char_id, pose_index, only_new=args.only_new)
            joint_data = _extract_joints_from_mask(aug_mask, args.resolution)
            exporter.save_joints(joint_data, args.output_dir, result.char_id, pose_index, only_new=args.only_new)
            exporter.save_draw_order(aug_draw_order, args.output_dir, result.char_id, pose_index, only_new=args.only_new)
            exporter.save_image(aug_image, args.output_dir, result.char_id, pose_index, "flat", only_new=args.only_new)

        exporter.save_source_metadata(
            args.output_dir,
            result.char_id,
            source="live2d",
            name=result.char_id,
            license_="",
            attribution="",
            bone_mapping="auto",
            unmapped_bones=result.unmapped_fragments,
            character_type="humanoid",
            notes=(
                f"Live2D model, {result.fragment_count} fragments, "
                f"{result.mapped_count} mapped, "
                f"{len(result.unmapped_fragments)} unmapped"
            ),
            only_new=args.only_new,
        )

        results.append(result)
        if (i + 1) % 10 == 0 or (i + 1) == len(model_dirs):
            elapsed = time.monotonic() - start
            rate = (i + 1) / elapsed
            eta = (len(model_dirs) - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1}/{len(model_dirs)}] {rate:.1f} models/s "
                f"(ETA {eta/60:.0f}m) — {result.char_id}: "
                f"{result.mapped_count}/{result.fragment_count} frags mapped"
            )

    elapsed = time.monotonic() - start
    total_mapped = sum(r.mapped_count for r in results)
    total_frags = sum(r.fragment_count for r in results)
    variants_per = 1 if args.no_augmentation else 4

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Models succeeded:  {len(results)} / {len(model_dirs)} ({failed} failed, {skipped_low_mapping} skipped low-mapping)")
    print(f"  Fragments total:   {total_frags}")
    print(f"  Fragments mapped:  {total_mapped} ({total_mapped/max(total_frags,1)*100:.0f}%)")
    print(f"  Examples written:  ~{len(results) * variants_per}")
    print(f"  Output:            {args.output_dir}")


if __name__ == "__main__":
    main()
