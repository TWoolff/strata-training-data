"""Blender entry point: render HumanRig characters with Mixamo FBX poses.

Run inside Blender's Python environment::

    blender --background --python run_humanrig_posed.py -- \\
        --input_dir /path/to/humanrig_opensource_final \\
        --pose_dir /path/to/poses \\
        --output_dir /path/to/output \\
        --max_samples 100

Arguments after ``--`` are parsed by this script; everything before is
consumed by Blender itself.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path so ``ingest`` is importable inside Blender.
repo_root = str(Path(__file__).resolve().parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

VALID_ANGLES = {"front", "three_quarter", "side", "three_quarter_back", "back"}


def parse_args() -> argparse.Namespace:
    """Parse arguments from the portion after ``--`` in the Blender command."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Render HumanRig characters with Mixamo FBX poses at multiple camera angles.",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Path to humanrig_opensource_final/ directory.",
    )
    parser.add_argument(
        "--pose_dir",
        type=Path,
        default=None,
        help="Directory containing Mixamo FBX/BVH animation files (not needed for --seg_only).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Root output directory for rendered examples.",
    )
    parser.add_argument(
        "--angles",
        type=str,
        default="front,three_quarter,side,three_quarter_back,back",
        help="Comma-separated camera angles (default: all 5 angles).",
    )
    parser.add_argument(
        "--only_new",
        action="store_true",
        default=False,
        help="Skip examples that already have image.png.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum samples to process (0 = all).",
    )
    parser.add_argument(
        "--poses_per_clip",
        type=int,
        default=3,
        help="Keyframes to sample per animation clip (default: 3).",
    )
    parser.add_argument(
        "--seg_only",
        action="store_true",
        default=False,
        help="Only render segmentation masks for existing examples (skip image/joints).",
    )
    return parser.parse_args(argv)


def main() -> None:
    from ingest.humanrig_posed_renderer import render_directory

    args = parse_args()

    if not args.input_dir.is_dir():
        logger.error("Input directory not found: %s", args.input_dir)
        sys.exit(1)

    if not args.seg_only and args.pose_dir is None:
        logger.error("--pose_dir is required unless using --seg_only")
        sys.exit(1)

    if args.pose_dir and not args.pose_dir.is_dir():
        logger.error("Pose directory not found: %s", args.pose_dir)
        sys.exit(1)

    angles = [a.strip() for a in args.angles.split(",") if a.strip()]
    invalid = [a for a in angles if a not in VALID_ANGLES]
    if invalid:
        logger.error("Invalid angle(s): %s. Valid: %s", invalid, sorted(VALID_ANGLES))
        sys.exit(1)

    if args.seg_only:
        logger.info("HumanRig seg-only render starting")
        logger.info("  input_dir:  %s", args.input_dir)
        logger.info("  output_dir: %s", args.output_dir)
        logger.info("  max_samples: %s", args.max_samples or "unlimited")
    else:
        logger.info("HumanRig posed render starting")
        logger.info("  input_dir:      %s", args.input_dir)
        logger.info("  pose_dir:       %s", args.pose_dir)
        logger.info("  output_dir:     %s", args.output_dir)
        logger.info("  angles:         %s", angles)
        logger.info("  only_new:       %s", args.only_new)
        logger.info("  max_samples:    %s", args.max_samples or "unlimited")
        logger.info("  poses_per_clip: %s", args.poses_per_clip)

    start = time.monotonic()

    rendered, skipped, errors = render_directory(
        args.input_dir,
        args.output_dir,
        args.pose_dir,
        angles=angles,
        only_new=args.only_new,
        max_samples=args.max_samples,
        poses_per_clip=args.poses_per_clip,
        seg_only=args.seg_only,
    )

    elapsed = time.monotonic() - start

    print("\nHumanRig posed render complete:")
    print(f"  Images rendered: {rendered}")
    print(f"  Images skipped:  {skipped}")
    print(f"  Errors:          {errors}")
    print(f"  Elapsed:         {elapsed:.1f}s")
    if rendered > 0:
        print(f"  Per image:       {elapsed/rendered:.2f}s")

    sys.exit(0 if errors == 0 else 1)


if __name__ == "__main__":
    main()
