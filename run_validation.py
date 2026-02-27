"""Standalone entry point for dataset validation.

Runs all automated validation checks on a generated dataset without
requiring Blender.  See Issue #23 and PRD §11.1.

Usage::

    python run_validation.py --dataset_dir ./output/segmentation/
    python run_validation.py --dataset_dir ./output/segmentation/ --characters mixamo_001,mixamo_002
    python run_validation.py --dataset_dir ./output/segmentation/ --save_report
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
from pipeline.validator import print_report, save_report, validate_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate a Strata dataset for integrity and correctness.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("./output/segmentation"),
        help="Root dataset directory to validate.",
    )
    parser.add_argument(
        "--characters",
        type=str,
        default="",
        help="Comma-separated list of character IDs to validate (default: all).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=RENDER_RESOLUTION,
        help=f"Expected image resolution (default: {RENDER_RESOLUTION}).",
    )
    parser.add_argument(
        "--save_report",
        action="store_true",
        default=False,
        help="Save validation_report.json to the dataset directory.",
    )
    return parser.parse_args()


def main() -> None:
    """Run dataset validation and print results."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()
    dataset_dir: Path = args.dataset_dir
    resolution: int = args.resolution
    save: bool = args.save_report

    characters: list[str] | None = None
    if args.characters:
        characters = [c.strip() for c in args.characters.split(",") if c.strip()]

    if not dataset_dir.is_dir():
        print(f"ERROR: Dataset directory does not exist: {dataset_dir}")
        sys.exit(1)

    report = validate_dataset(
        dataset_dir,
        characters=characters,
        resolution=resolution,
    )

    print_report(report)

    if save:
        report_path = dataset_dir / "validation_report.json"
        save_report(report, report_path)
        print(f"Report saved to {report_path}")

    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
