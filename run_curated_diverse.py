"""DEPRECATED: curated_diverse removed from training pipeline — ArtStation artwork, no AI training permission.

This script is kept for reference but should NOT be used to generate training data.

Previously ran all three steps:
1. Ingest: rembg background removal → resize → fg mask → metadata
2. RTMPose: 2D joint estimation
3. Depth Anything v2: monocular depth → draw order map

Usage::

    python3 run_curated_diverse.py \
        --input_dir /Volumes/TAMWoolff/data/preprocessed/curated_diverse \
        --output_dir ./output/curated_diverse

    # Skip steps that are already done:
    python3 run_curated_diverse.py \
        --input_dir /Volumes/TAMWoolff/data/preprocessed/curated_diverse \
        --output_dir ./output/curated_diverse \
        --only_new --only_missing

    # Skip background removal (images already have transparent bg):
    python3 run_curated_diverse.py \
        --input_dir /Volumes/TAMWoolff/data/preprocessed/curated_diverse \
        --output_dir ./output/curated_diverse \
        --no_rembg
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DET_MODEL = Path("models/yolox_m_humanart.onnx")
DEFAULT_POSE_MODEL = Path("models/rtmpose_m_body7.onnx")
DEFAULT_DEPTH_MODEL = Path("models/depth_anything_v2_vits.onnx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full pipeline for curated diverse character images.",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing raw curated images.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for Strata-formatted examples.",
    )
    parser.add_argument("--max_images", type=int, default=0, help="Max images (0=all).")
    parser.add_argument("--only_new", action="store_true", help="Skip existing outputs.")
    parser.add_argument("--only_missing", action="store_true", help="Skip existing enrichments.")
    parser.add_argument(
        "--no_rembg",
        action="store_true",
        help="Skip background removal (images already have transparent bg).",
    )

    # Step control
    parser.add_argument("--skip_ingest", action="store_true", help="Skip ingest step.")
    parser.add_argument("--skip_joints", action="store_true", help="Skip RTMPose step.")
    parser.add_argument("--skip_depth", action="store_true", help="Skip depth step.")

    # Model paths
    parser.add_argument("--det_model", type=str, default=str(DEFAULT_DET_MODEL))
    parser.add_argument("--pose_model", type=str, default=str(DEFAULT_POSE_MODEL))
    parser.add_argument("--depth_model", type=str, default=str(DEFAULT_DEPTH_MODEL))
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_start = time.time()

    # --- Step 1: Ingest (rembg + resize + mask + metadata) ---
    if not args.skip_ingest:
        logger.info("=== Step 1: Ingest (rembg + resize) ===")
        t0 = time.time()

        from ingest.curated_diverse_adapter import convert_directory

        result = convert_directory(
            args.input_dir,
            args.output_dir,
            only_new=args.only_new,
            max_images=args.max_images,
            remove_bg=not args.no_rembg,
        )
        elapsed = time.time() - t0
        logger.info(
            "Ingest done: %d processed, %d skipped in %.1f sec",
            result.images_processed,
            result.images_skipped,
            elapsed,
        )
    else:
        logger.info("=== Step 1: Ingest — SKIPPED ===")

    # --- Step 2: RTMPose joint enrichment ---
    if not args.skip_joints:
        logger.info("=== Step 2: RTMPose joint enrichment ===")
        t0 = time.time()

        from pipeline.pose_estimator import enrich_example, load_model

        model = load_model(args.det_model, args.pose_model, device=args.device)
        image_size = (512, 512)

        examples = sorted(
            p.parent
            for p in args.output_dir.rglob("image.png")
            if not args.only_missing or not (p.parent / "joints.json").exists()
        )

        enriched = 0
        for i, ex in enumerate(examples):
            if enrich_example(model, ex, image_size):
                enriched += 1
            if (i + 1) % 50 == 0 or (i + 1) == len(examples):
                logger.info("Joints: %d/%d", i + 1, len(examples))

        elapsed = time.time() - t0
        logger.info("Joints done: %d enriched in %.1f sec", enriched, elapsed)
    else:
        logger.info("=== Step 2: RTMPose — SKIPPED ===")

    # --- Step 3: Depth Anything v2 draw order ---
    if not args.skip_depth:
        depth_model_path = Path(args.depth_model)
        if not depth_model_path.is_file():
            logger.warning(
                "Depth model not found at %s — skipping depth enrichment. "
                "Download from: https://huggingface.co/onnx-community/depth-anything-v2-small",
                depth_model_path,
            )
        else:
            logger.info("=== Step 3: Depth Anything v2 draw order ===")
            t0 = time.time()

            from pipeline.depth_estimator import enrich_example as depth_enrich
            from pipeline.depth_estimator import load_depth_model

            session = load_depth_model(args.depth_model, device=args.device)

            examples = sorted(
                p.parent
                for p in args.output_dir.rglob("image.png")
                if not args.only_missing or not (p.parent / "draw_order.png").exists()
            )

            enriched = 0
            for i, ex in enumerate(examples):
                if depth_enrich(session, ex):
                    enriched += 1
                if (i + 1) % 50 == 0 or (i + 1) == len(examples):
                    logger.info("Depth: %d/%d", i + 1, len(examples))

            elapsed = time.time() - t0
            logger.info("Depth done: %d enriched in %.1f sec", enriched, elapsed)
    else:
        logger.info("=== Step 3: Depth — SKIPPED ===")

    total_elapsed = time.time() - total_start
    logger.info("=== All done in %.1f sec ===", total_elapsed)


if __name__ == "__main__":
    main()
