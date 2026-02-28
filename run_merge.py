"""Standalone entry point for merging multiple dataset sources.

Combines outputs from the 3D synthetic pipeline, Spine parser, Live2D
renderer, manual annotations, and ingest adapters into a single unified
dataset.  See Issue #28 and PRD §10.5.

Usage::

    python run_merge.py \
        --source_dirs ./output/segmentation ./output/spine ./output/live2d \
        --output_dir ./output/merged

    python run_merge.py \
        --source_dirs ./output/segmentation ./output/spine \
        --output_dir ./output/merged \
        --link \
        --no_validate
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so ``pipeline`` is importable.
repo_root = str(Path(__file__).resolve().parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pipeline.config import RENDER_RESOLUTION
from pipeline.dataset_merger import merge_datasets, print_merge_report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge multiple Strata dataset sources into a unified dataset.",
    )
    parser.add_argument(
        "--source_dirs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more source dataset directories to merge.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./output/merged"),
        help="Destination directory for the merged dataset (default: ./output/merged).",
    )
    parser.add_argument(
        "--link",
        action="store_true",
        default=False,
        help="Use symlinks instead of file copies.",
    )
    parser.add_argument(
        "--no_validate",
        action="store_true",
        default=False,
        help="Skip input validation (faster but less safe).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=RENDER_RESOLUTION,
        help=f"Expected image resolution (default: {RENDER_RESOLUTION}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split generation (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    """Run dataset merge and print results."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()

    mode = "link" if args.link else "copy"
    validate = not args.no_validate

    report = merge_datasets(
        source_dirs=args.source_dirs,
        output_dir=args.output_dir,
        mode=mode,
        validate=validate,
        resolution=args.resolution,
        seed=args.seed,
    )

    print_merge_report(report)

    if report.characters_merged == 0:
        print("WARNING: No characters were merged.")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
