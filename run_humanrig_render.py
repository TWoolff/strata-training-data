"""Blender entry point: render additional camera angles for HumanRig.

Run inside Blender's Python environment::

    blender --background --python run_humanrig_render.py -- \\
        --input_dir /path/to/humanrig_opensource_final \\
        --output_dir ./output/humanrig \\
        --angles three_quarter,side,back \\
        --only_new

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

VALID_ANGLES = {"front", "three_quarter", "side", "back"}


def parse_args() -> argparse.Namespace:
    """Parse arguments from the portion after ``--`` in the Blender command."""
    # Blender passes everything after '--' as sys.argv tail.
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Render HumanRig GLB files from multiple camera angles.",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Path to humanrig_opensource_final/ directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Root output directory (same as used by run_ingest.py --adapter humanrig).",
    )
    parser.add_argument(
        "--angles",
        type=str,
        default="three_quarter,side,back",
        help="Comma-separated camera angles to render (default: three_quarter,side,back).",
    )
    parser.add_argument(
        "--only_new",
        action="store_true",
        default=False,
        help="Skip samples that already have image.png.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum samples to process (0 = all).",
    )
    return parser.parse_args(argv)


def main() -> None:
    from ingest.humanrig_blender_renderer import render_directory

    args = parse_args()

    angles = [a.strip() for a in args.angles.split(",") if a.strip()]
    invalid = [a for a in angles if a not in VALID_ANGLES]
    if invalid:
        logger.error("Invalid angle(s): %s. Valid: %s", invalid, sorted(VALID_ANGLES))
        sys.exit(1)

    if not args.input_dir.is_dir():
        logger.error("Input directory not found: %s", args.input_dir)
        sys.exit(1)

    logger.info("HumanRig render pass starting")
    logger.info("  input_dir:   %s", args.input_dir)
    logger.info("  output_dir:  %s", args.output_dir)
    logger.info("  angles:      %s", angles)
    logger.info("  only_new:    %s", args.only_new)
    logger.info("  max_samples: %s", args.max_samples or "unlimited")

    start = time.monotonic()

    rendered, skipped, errors = render_directory(
        args.input_dir,
        args.output_dir,
        angles=angles,
        only_new=args.only_new,
        max_samples=args.max_samples,
    )

    elapsed = time.monotonic() - start

    print("\nHumanRig render complete:")
    print(f"  Images rendered: {rendered}")
    print(f"  Images skipped:  {skipped}")
    print(f"  Errors:          {errors}")
    print(f"  Elapsed:         {elapsed:.1f}s")
    if rendered > 0:
        print(f"  Per image:       {elapsed/rendered:.2f}s")

    sys.exit(0 if errors == 0 else 1)


if __name__ == "__main__":
    main()
