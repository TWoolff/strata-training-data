"""Standalone entry point for running ingest adapters.

Converts pre-processed external datasets into Strata training format.
Each adapter handles a specific dataset's directory structure and
outputs per-example directories with ``image.png`` and ``metadata.json``.

With ``--enrich``, also runs 2D pose estimation (RTMPose) to add
``joints.json`` alongside each example.

Usage::

    python run_ingest.py \
        --adapter fbanimehq \
        --input_dir ./data/preprocessed/fbanimehq/data/fbanimehq-00 \
        --output_dir ./output/fbanimehq \
        --max_images 500 \
        --random_sample

    # Ingest + pose enrichment in one step:
    python run_ingest.py \
        --adapter fbanimehq \
        --input_dir ./data/preprocessed/fbanimehq/data/fbanimehq-00 \
        --output_dir ./output/fbanimehq \
        --max_images 500 \
        --enrich
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

logger = logging.getLogger(__name__)

# Default model paths (relative to repo root).
DEFAULT_DET_MODEL = Path("models/yolox_m_humanart.onnx")
DEFAULT_POSE_MODEL = Path("models/rtmpose_m_body7.onnx")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert a pre-processed dataset into Strata training format.",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        choices=[
            "fbanimehq",
            "nova_human",
            "anime_seg",
            "animerun",
            "animerun_flow",
            "animerun_segment",
            "animerun_correspondence",
        ],
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

    # --- Pose enrichment flags ---
    parser.add_argument(
        "--enrich",
        action="store_true",
        default=False,
        help="Run 2D pose estimation after ingest to add joints.json.",
    )
    parser.add_argument(
        "--det_model",
        type=str,
        default=str(DEFAULT_DET_MODEL),
        help=f"Detection ONNX model path (default: {DEFAULT_DET_MODEL}).",
    )
    parser.add_argument(
        "--pose_model",
        type=str,
        default=str(DEFAULT_POSE_MODEL),
        help=f"Pose estimation ONNX model path (default: {DEFAULT_POSE_MODEL}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Inference device for pose enrichment (default: cpu).",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.3,
        help="Minimum joint confidence for visible=True (default: 0.3).",
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

    print("\nFBAnimeHQ ingestion complete:")
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
    print("\nNOVA-Human ingestion complete:")
    print(f"  Characters processed: {len(results)}")
    print(f"  Views saved:          {total_views}")
    print(f"  Output directory:     {args.output_dir}")

    return 0 if results else 1


def _run_anime_seg(args: argparse.Namespace) -> int:
    """Run the anime-segmentation adapter."""
    from ingest.anime_seg_adapter import convert_directory

    # Determine variant from input path heuristic
    variant = "v2" if "anime_seg_v2" in str(args.input_dir) else "v1"

    result = convert_directory(
        args.input_dir,
        args.output_dir,
        variant=variant,
        resolution=args.resolution,
        only_new=args.only_new,
        max_images=args.max_images,
        random_sample=args.random_sample,
        seed=args.seed,
    )

    print(f"\nanime-segmentation ({variant}) ingestion complete:")
    print(f"  Images processed: {result.images_processed}")
    print(f"  Images skipped:   {result.images_skipped}")
    print(f"  Errors:           {len(result.errors)}")
    print(f"  Output directory:  {args.output_dir}")

    return 0 if result.images_processed > 0 or result.images_skipped > 0 else 1


def _run_animerun(args: argparse.Namespace) -> int:
    """Run the AnimeRun contour adapter."""
    from ingest.animerun_contour_adapter import convert_directory

    results = convert_directory(
        args.input_dir,
        args.output_dir,
        resolution=args.resolution,
        only_new=args.only_new,
        max_frames_per_scene=args.max_images,
    )

    total_saved = sum(r.frames_saved for r in results)
    total_skipped = sum(r.frames_skipped for r in results)
    print("\nAnimeRun ingestion complete:")
    print(f"  Scenes processed: {len(results)}")
    print(f"  Frames saved:     {total_saved}")
    print(f"  Frames skipped:   {total_skipped}")
    print(f"  Output directory:  {args.output_dir}")

    return 0 if total_saved > 0 or total_skipped > 0 else 1


def _run_animerun_flow(args: argparse.Namespace) -> int:
    """Run the AnimeRun optical flow adapter."""
    from ingest.animerun_flow_adapter import convert_directory

    results = convert_directory(
        args.input_dir,
        args.output_dir,
        resolution=args.resolution,
        only_new=args.only_new,
        max_frames_per_scene=args.max_images,
    )

    total_saved = sum(r.frames_saved for r in results)
    total_skipped = sum(r.frames_skipped for r in results)
    print("\nAnimeRun flow ingestion complete:")
    print(f"  Scenes processed: {len(results)}")
    print(f"  Pairs saved:      {total_saved}")
    print(f"  Pairs skipped:    {total_skipped}")
    print(f"  Output directory:  {args.output_dir}")

    return 0 if total_saved > 0 or total_skipped > 0 else 1


def _run_animerun_segment(args: argparse.Namespace) -> int:
    """Run the AnimeRun instance segmentation adapter."""
    from ingest.animerun_segment_adapter import convert_directory

    results = convert_directory(
        args.input_dir,
        args.output_dir,
        resolution=args.resolution,
        only_new=args.only_new,
        max_frames_per_scene=args.max_images,
    )

    total_saved = sum(r.frames_saved for r in results)
    total_skipped = sum(r.frames_skipped for r in results)
    print("\nAnimeRun segment ingestion complete:")
    print(f"  Scenes processed: {len(results)}")
    print(f"  Frames saved:     {total_saved}")
    print(f"  Frames skipped:   {total_skipped}")
    print(f"  Output directory:  {args.output_dir}")

    return 0 if total_saved > 0 or total_skipped > 0 else 1


def _run_animerun_correspondence(args: argparse.Namespace) -> int:
    """Run the AnimeRun temporal correspondence adapter."""
    from ingest.animerun_correspondence_adapter import convert_directory

    results = convert_directory(
        args.input_dir,
        args.output_dir,
        resolution=args.resolution,
        only_new=args.only_new,
        max_frames_per_scene=args.max_images,
    )

    total_saved = sum(r.frames_saved for r in results)
    total_skipped = sum(r.frames_skipped for r in results)
    print("\nAnimeRun correspondence ingestion complete:")
    print(f"  Scenes processed: {len(results)}")
    print(f"  Pairs saved:      {total_saved}")
    print(f"  Pairs skipped:    {total_skipped}")
    print(f"  Output directory:  {args.output_dir}")

    return 0 if total_saved > 0 or total_skipped > 0 else 1


_ADAPTERS = {
    "fbanimehq": _run_fbanimehq,
    "nova_human": _run_nova_human,
    "anime_seg": _run_anime_seg,
    "animerun": _run_animerun,
    "animerun_flow": _run_animerun_flow,
    "animerun_segment": _run_animerun_segment,
    "animerun_correspondence": _run_animerun_correspondence,
}


def _run_enrichment(args: argparse.Namespace) -> int:
    """Run 2D pose estimation on all ingested examples."""
    from pipeline.pose_estimator import enrich_example, load_model

    # Discover examples that need enrichment
    examples = sorted(
        p.parent
        for p in args.output_dir.rglob("image.png")
        if not (p.parent / "joints.json").exists()
    )

    if not examples:
        print("\nPose enrichment: all examples already have joints.json.")
        return 0

    print(f"\nRunning pose enrichment on {len(examples)} examples...")

    model = load_model(args.det_model, args.pose_model, device=args.device)
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

        if (i + 1) % 100 == 0 or (i + 1) == len(examples):
            logger.info(
                "Enrichment: %d/%d (%.1f%%) — %d enriched, %d failed",
                i + 1,
                len(examples),
                (i + 1) / len(examples) * 100,
                enriched,
                failed,
            )

    print(f"  Enriched:          {enriched}")
    print(f"  Failed:            {failed}")

    return 1 if failed > 0 else 0


def main() -> None:
    """Run the selected ingest adapter."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()

    start = time.monotonic()
    exit_code = _ADAPTERS[args.adapter](args)

    # Chain pose enrichment if requested
    if args.enrich and exit_code == 0:
        enrich_code = _run_enrichment(args)
        if enrich_code != 0:
            exit_code = enrich_code

    elapsed = time.monotonic() - start

    print(f"  Elapsed:           {elapsed:.1f}s")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
