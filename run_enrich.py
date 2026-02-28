"""Standalone entry point for 2D pose estimation enrichment.

Walks ingest output directories, runs RTMPose inference on each
``image.png``, and writes ``joints.json`` alongside.  Updates
``metadata.json`` to reflect the new annotations.

Usage::

    python run_enrich.py \
        --input_dir ./output/fbanimehq \
        --det_model ./models/yolox_m.onnx \
        --pose_model ./models/rtmpose-m.onnx \
        --device cpu

    python run_enrich.py \
        --input_dir ./output/fbanimehq \
        --det_model ./models/yolox_m.onnx \
        --pose_model ./models/rtmpose-m.onnx \
        --device cuda \
        --only_missing
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure the repo root is on sys.path so ``pipeline`` is importable.
repo_root = str(Path(__file__).resolve().parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pipeline.config import RENDER_RESOLUTION

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Enrich image-only datasets with 2D pose estimation joints.",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Root directory containing per-example subdirectories with image.png.",
    )
    parser.add_argument(
        "--det_model",
        type=str,
        required=True,
        help="Path or URL to the detection ONNX model (e.g. YOLOX).",
    )
    parser.add_argument(
        "--pose_model",
        type=str,
        required=True,
        help="Path or URL to the pose estimation ONNX model (e.g. RTMPose-m).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Inference device (default: cpu).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="onnxruntime",
        help="ONNX runtime backend (default: onnxruntime).",
    )
    parser.add_argument(
        "--det_input_size",
        type=int,
        nargs=2,
        default=[640, 640],
        help="Detection model input size (w h). Default: 640 640.",
    )
    parser.add_argument(
        "--pose_input_size",
        type=int,
        nargs=2,
        default=[192, 256],
        help="Pose model input size (w h). Default: 192 256.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=RENDER_RESOLUTION,
        help=f"Image resolution for joint coordinate bounds (default: {RENDER_RESOLUTION}).",
    )
    parser.add_argument(
        "--only_missing",
        action="store_true",
        default=False,
        help="Skip examples that already have joints.json.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.3,
        help="Minimum confidence for a joint to be marked visible (default: 0.3).",
    )
    return parser.parse_args()


def _discover_examples(input_dir: Path, *, only_missing: bool = False) -> list[Path]:
    """Find all example directories containing image.png.

    Args:
        input_dir: Root directory to search.
        only_missing: If True, skip directories that already have joints.json.

    Returns:
        Sorted list of example directory paths.
    """
    examples = []
    for image_path in sorted(input_dir.rglob("image.png")):
        example_dir = image_path.parent
        if only_missing and (example_dir / "joints.json").exists():
            continue
        examples.append(example_dir)
    return examples


def main() -> None:
    """Run pose estimation enrichment."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()

    # Discover examples
    examples = _discover_examples(args.input_dir, only_missing=args.only_missing)
    total = len(examples)

    if total == 0:
        print("No examples to process.")
        if args.only_missing:
            print("(All examples already have joints.json — use without --only_missing to re-run.)")
        sys.exit(0)

    print(f"Found {total} examples to enrich in {args.input_dir}")

    # Load model
    from pipeline.pose_estimator import enrich_example, load_model

    start = time.monotonic()

    model = load_model(
        args.det_model,
        args.pose_model,
        device=args.device,
        backend=args.backend,
        det_input_size=tuple(args.det_input_size),
        pose_input_size=tuple(args.pose_input_size),
    )

    image_size = (args.resolution, args.resolution)
    enriched = 0
    failed = 0

    for i, example_dir in enumerate(examples):
        success = enrich_example(
            model,
            example_dir,
            image_size,
            confidence_threshold=args.confidence_threshold,
        )

        if success:
            enriched += 1
        else:
            failed += 1

        # Progress logging every 100 images
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

    print("\nPose enrichment complete:")
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
