"""Standalone entry point for depth estimation enrichment.

Walks ingest output directories, runs Depth Anything v2 inference on each
``image.png``, and writes ``draw_order.png`` alongside.  Updates
``metadata.json`` to reflect the new annotation.

Usage::

    python3 run_depth_enrich.py \
        --input_dir ./output/anime_seg \
        --depth_model ./models/depth_anything_v2_vits.onnx

    python3 run_depth_enrich.py \
        --input_dir ./output/anime_seg \
        --depth_model ./models/depth_anything_v2_vits.onnx \
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

DEFAULT_DEPTH_MODEL = Path("models/depth_anything_v2_vits.onnx")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Enrich datasets with monocular depth estimation (draw order).",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Root directory containing per-example subdirectories with image.png.",
    )
    parser.add_argument(
        "--depth_model",
        type=str,
        default=str(DEFAULT_DEPTH_MODEL),
        help=f"Path to Depth Anything v2 ONNX model (default: {DEFAULT_DEPTH_MODEL}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Inference device (default: cpu).",
    )
    parser.add_argument(
        "--only_missing",
        action="store_true",
        default=False,
        help="Skip examples that already have draw_order.png.",
    )
    return parser.parse_args()


def _discover_examples(input_dir: Path, *, only_missing: bool = False) -> list[Path]:
    """Find all example directories containing image.png."""
    examples = []
    for image_path in sorted(input_dir.rglob("image.png")):
        example_dir = image_path.parent
        if only_missing and (example_dir / "draw_order.png").exists():
            continue
        examples.append(example_dir)
    return examples


def main() -> None:
    """Run depth estimation enrichment."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()

    examples = _discover_examples(args.input_dir, only_missing=args.only_missing)
    total = len(examples)

    if total == 0:
        print("No examples to process.")
        if args.only_missing:
            print("(All examples already have draw_order.png.)")
        sys.exit(0)

    print(f"Found {total} examples to enrich in {args.input_dir}")

    from pipeline.depth_estimator import enrich_example, load_depth_model

    start = time.monotonic()
    session = load_depth_model(args.depth_model, device=args.device)

    enriched = 0
    failed = 0

    for i, example_dir in enumerate(examples):
        success = enrich_example(session, example_dir)

        if success:
            enriched += 1
        else:
            failed += 1

        if (i + 1) % 100 == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            logger.info(
                "Progress: %d/%d (%.1f%%) — %d enriched, %d failed",
                i + 1,
                total,
                pct,
                enriched,
                failed,
            )

    elapsed = time.monotonic() - start

    print("\nDepth enrichment complete:")
    print(f"  Enriched:   {enriched}")
    print(f"  Failed:     {failed}")
    print(f"  Total:      {total}")
    print(f"  Elapsed:    {elapsed:.1f}s")
    if enriched > 0:
        print(f"  Speed:      {enriched / elapsed:.1f} images/sec")
    print(f"  Output:     {args.input_dir}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
