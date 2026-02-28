"""Standalone entry point for running ingest adapters.

Converts pre-processed external datasets into Strata training format.
Each adapter handles a specific dataset's directory structure and
outputs per-example directories with ``image.png`` and ``metadata.json``.

Usage::

    python run_ingest.py \
        --adapter fbanimehq \
        --input_dir ./data/preprocessed/fbanimehq/data/fbanimehq-00 \
        --output_dir ./output/fbanimehq \
        --max_images 500 \
        --random_sample

    python run_ingest.py \
        --adapter nova_human \
        --input_dir ./data/preprocessed/nova_human \
        --output_dir ./output/nova_human \
        --max_images 100
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure the repo root is on sys.path so ``ingest`` is importable.
repo_root = str(Path(__file__).resolve().parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pipeline.config import RENDER_RESOLUTION


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert a pre-processed dataset into Strata training format.",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        choices=["fbanimehq", "nova_human"],
        help="Which dataset adapter to use.",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Source dataset directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./output/ingest"),
        help="Output directory for Strata-formatted examples (default: ./output/ingest).",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Maximum images to process (0 = unlimited).",
    )
    parser.add_argument(
        "--random_sample",
        action="store_true",
        default=False,
        help="Randomly sample from available images (requires --max_images).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=RENDER_RESOLUTION,
        help=f"Target image resolution (default: {RENDER_RESOLUTION}).",
    )
    parser.add_argument(
        "--only_new",
        action="store_true",
        default=False,
        help="Skip already-converted examples.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Adapter dispatch
# ---------------------------------------------------------------------------


def _run_fbanimehq(args: argparse.Namespace) -> int:
    """Run the FBAnimeHQ adapter."""
    from ingest.fbanimehq_adapter import convert_directory

    result = convert_directory(
        args.input_dir,
        args.output_dir,
        resolution=args.resolution,
        only_new=args.only_new,
        max_images=args.max_images,
        random_sample=args.random_sample,
        seed=args.seed,
    )

    print(f"\nFBAnimeHQ ingestion complete:")
    print(f"  Images processed: {result.images_processed}")
    print(f"  Images skipped:   {result.images_skipped}")
    print(f"  Errors:           {len(result.errors)}")
    print(f"  Output directory:  {args.output_dir}")

    return 0 if result.images_processed > 0 or result.images_skipped > 0 else 1


def _run_nova_human(args: argparse.Namespace) -> int:
    """Run the NOVA-Human adapter."""
    from ingest.nova_human_adapter import convert_directory

    results = convert_directory(
        args.input_dir,
        args.output_dir,
        resolution=args.resolution,
        only_new=args.only_new,
        max_characters=args.max_images,
    )

    total_views = sum(r.views_saved for r in results)
    print(f"\nNOVA-Human ingestion complete:")
    print(f"  Characters processed: {len(results)}")
    print(f"  Views saved:          {total_views}")
    print(f"  Output directory:     {args.output_dir}")

    return 0 if results else 1


_ADAPTERS = {
    "fbanimehq": _run_fbanimehq,
    "nova_human": _run_nova_human,
}


def main() -> None:
    """Run the selected ingest adapter."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()

    start = time.monotonic()
    exit_code = _ADAPTERS[args.adapter](args)
    elapsed = time.monotonic() - start

    print(f"  Elapsed:           {elapsed:.1f}s")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
