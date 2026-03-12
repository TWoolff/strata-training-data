"""Render VRoid CC0 GLB characters with ground-truth segmentation masks.

Blender CLI entry point for the VRoid CC0 renderer. Imports GLB files,
renders color images + 22-class segmentation masks + joints at multiple
camera angles.

Usage::

    blender --background --python run_vroid_cc0.py -- \\
        --input_dir /Volumes/TAMWoolff/data/raw/vroid_cc0 \\
        --output_dir /Volumes/TAMWoolff/data/preprocessed/vroid_cc0 \\
        --angles front,three_quarter,side \\
        --only_new

    # With Mixamo poses:
    blender --background --python run_vroid_cc0.py -- \\
        --input_dir /Volumes/TAMWoolff/data/raw/vroid_cc0 \\
        --output_dir /Volumes/TAMWoolff/data/preprocessed/vroid_cc0 \\
        --pose_dir /Volumes/TAMWoolff/data/poses \\
        --angles front,three_quarter,side \\
        --max_characters 2

    # Single character test:
    blender --background --python run_vroid_cc0.py -- \\
        --input_dir /Volumes/TAMWoolff/data/raw/vroid_cc0 \\
        --output_dir ./output/vroid_cc0_test \\
        --max_characters 1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure repo root is on sys.path so ``ingest`` and ``pipeline`` are importable.
repo_root = str(Path(__file__).resolve().parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def _parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Render VRoid CC0 GLB characters with GT segmentation masks"
    )
    parser.add_argument(
        "--input_dir", "--input-dir",
        type=str, required=True,
        help="Directory containing VRoid CC0 .glb files",
    )
    parser.add_argument(
        "--output_dir", "--output-dir",
        type=str, required=True,
        help="Output directory for rendered examples",
    )
    parser.add_argument(
        "--pose_dir", "--pose-dir",
        type=str, default=None,
        help="Directory with Mixamo FBX/BVH poses (optional — T-pose only if omitted)",
    )
    parser.add_argument(
        "--angles",
        type=str, default="front,three_quarter,side",
        help="Comma-separated camera angles (default: front,three_quarter,side)",
    )
    parser.add_argument(
        "--max_characters", "--max-characters",
        type=int, default=0,
        help="Max characters to process (0 = all)",
    )
    parser.add_argument(
        "--only_new", "--only-new",
        action="store_true",
        help="Skip examples that already have image.png",
    )

    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    args = _parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    pose_dir = Path(args.pose_dir) if args.pose_dir else None
    angles = [a.strip() for a in args.angles.split(",")]

    if not input_dir.exists():
        logging.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    logging.info("VRoid CC0 renderer starting")
    logging.info("  Input:  %s", input_dir)
    logging.info("  Output: %s", output_dir)
    logging.info("  Poses:  %s", pose_dir or "T-pose only")
    logging.info("  Angles: %s", angles)

    from ingest.vroid_cc0_renderer import render_directory

    rendered, skipped, errors = render_directory(
        input_dir,
        output_dir,
        pose_dir=pose_dir,
        angles=angles,
        only_new=args.only_new,
        max_characters=args.max_characters,
    )

    logging.info(
        "VRoid CC0 rendering complete: %d rendered, %d skipped, %d errors",
        rendered, skipped, errors,
    )


if __name__ == "__main__":
    main()
