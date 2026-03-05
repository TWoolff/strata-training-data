"""Enrich HumanRig output with segmentation masks, draw order, and FG masks.

Reads raw HumanRig data (vertices.json, camera matrices) to generate 22-class
body region annotations from vertex skinning weights.  Writes annotations
alongside existing per-example output (image.png, joints.json, metadata.json).

Usage::

    python3 run_humanrig_enrich.py \
        --raw_dir "/Volumes/TAMWoolff/data/preprocessed/humanrig/data/54T/chuzedong/autorig/preprocess/humanrig_opensource_final" \
        --output_dir "/Volumes/TAMWoolff/data/output/humanrig"

    # Only enrich examples missing segmentation masks:
    python3 run_humanrig_enrich.py \
        --raw_dir "..." \
        --output_dir "..." \
        --only_missing
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

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Enrich HumanRig output with segmentation, draw order, and FG masks.",
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        required=True,
        help="Root directory of raw HumanRig data (humanrig_opensource_final/).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Root directory of existing HumanRig output (per-example dirs).",
    )
    parser.add_argument(
        "--only_missing",
        action="store_true",
        default=False,
        help="Skip examples that already have segmentation.png.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Max samples to process (0 = all).",
    )
    parser.add_argument(
        "--run_adapter",
        action="store_true",
        default=False,
        help="Run the HumanRig adapter first to create base examples if output_dir is empty.",
    )
    return parser.parse_args()


def _discover_sample_pairs(
    raw_dir: Path,
    output_dir: Path,
    *,
    only_missing: bool = False,
) -> list[tuple[Path, Path]]:
    """Match raw sample dirs to output example dirs.

    Returns list of (raw_sample_dir, output_example_dir) pairs.
    """
    pairs: list[tuple[Path, Path]] = []

    for raw_sample in sorted(raw_dir.iterdir(), key=lambda p: _numeric_key(p)):
        if not raw_sample.is_dir() or not raw_sample.name.isdigit():
            continue

        # Required raw files
        if not (raw_sample / "vertices.json").exists():
            continue

        # Match to output dir — adapter names examples "humanrig_NNNNN_front"
        sample_id = int(raw_sample.name)
        example_id = f"humanrig_{sample_id:05d}_front"
        example_dir = output_dir / example_id

        if only_missing and (example_dir / "segmentation.png").exists():
            continue

        pairs.append((raw_sample, example_dir))

    return pairs


def _numeric_key(p: Path) -> int:
    try:
        return int(p.name)
    except ValueError:
        return -1


def main() -> None:
    """Run HumanRig enrichment pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()

    if not args.raw_dir.is_dir():
        print(f"Error: Raw directory not found: {args.raw_dir}")
        sys.exit(1)

    # Optionally run the adapter first to create base examples
    if args.run_adapter:
        if not args.output_dir.exists() or not any(args.output_dir.iterdir()):
            print("Running HumanRig adapter to create base examples...")
            from ingest.humanrig_adapter import convert_directory

            result = convert_directory(
                args.raw_dir,
                args.output_dir,
                angles=["front"],
            )
            print(
                f"Adapter: {result.images_processed} examples created, "
                f"{len(result.errors)} errors"
            )

    # Discover pairs
    pairs = _discover_sample_pairs(
        args.raw_dir,
        args.output_dir,
        only_missing=args.only_missing,
    )

    if args.max_samples > 0:
        pairs = pairs[: args.max_samples]

    total = len(pairs)
    if total == 0:
        print("No samples to enrich.")
        if args.only_missing:
            print("(All examples already have segmentation.png.)")
        sys.exit(0)

    print(f"Found {total} samples to enrich")
    print(f"  Raw:    {args.raw_dir}")
    print(f"  Output: {args.output_dir}")

    from ingest.humanrig_enricher import enrich_sample

    start = time.monotonic()
    enriched = 0
    failed = 0

    for i, (raw_sample, example_dir) in enumerate(pairs):
        try:
            success = enrich_sample(raw_sample, example_dir)
        except Exception as exc:
            logger.warning("Error enriching %s: %s", raw_sample.name, exc)
            success = False

        if success:
            enriched += 1
        else:
            failed += 1

        if (i + 1) % 200 == 0 or (i + 1) == total:
            elapsed = time.monotonic() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            pct = (i + 1) / total * 100
            logger.info(
                "Progress: %d/%d (%.1f%%) — %d enriched, %d failed [%.1f samples/sec]",
                i + 1,
                total,
                pct,
                enriched,
                failed,
                rate,
            )

    elapsed = time.monotonic() - start

    print("\nHumanRig enrichment complete:")
    print(f"  Enriched:   {enriched}")
    print(f"  Failed:     {failed}")
    print(f"  Total:      {total}")
    print(f"  Elapsed:    {elapsed:.1f}s")
    if enriched > 0:
        print(f"  Speed:      {enriched / elapsed:.1f} samples/sec")
    print(f"  Output:     {args.output_dir}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
