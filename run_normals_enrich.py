"""Enrich image datasets with surface normals and depth maps using Marigold.

Walks a directory of per-example subdirectories (each containing image.png),
runs Marigold normal + depth estimation, and saves normals.png and depth.png.

Usage::

    # Process existing pipeline output (subdirs with image.png)
    python run_normals_enrich.py --input-dir ./data_cloud/humanrig

    # Skip already-enriched examples
    python run_normals_enrich.py --input-dir ./data_cloud/humanrig --only-missing

    # Normals only (skip depth)
    python run_normals_enrich.py --input-dir ./data_cloud/humanrig --no-depth

    # Depth only (skip normals)
    python run_normals_enrich.py --input-dir ./data_cloud/humanrig --no-normals
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

repo_root = str(Path(__file__).resolve().parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pipeline.config import RENDER_RESOLUTION

logger = logging.getLogger(__name__)

MARIGOLD_NORMALS_MODEL = "prs-eth/marigold-normals-lcm-v0-1"
MARIGOLD_DEPTH_MODEL = "prs-eth/marigold-depth-lcm-v1-0"


def load_pipelines(device: torch.device, *, normals: bool = True, depth: bool = True):
    """Load Marigold pipelines."""
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipes = {}

    if normals:
        from diffusers import MarigoldNormalsPipeline
        pipes["normals"] = MarigoldNormalsPipeline.from_pretrained(
            MARIGOLD_NORMALS_MODEL, torch_dtype=dtype,
        ).to(device)
        logger.info("Loaded Marigold normals pipeline on %s", device)

    if depth:
        from diffusers import MarigoldDepthPipeline
        pipes["depth"] = MarigoldDepthPipeline.from_pretrained(
            MARIGOLD_DEPTH_MODEL, torch_dtype=dtype,
        ).to(device)
        logger.info("Loaded Marigold depth pipeline on %s", device)

    return pipes


def prepare_rgb(image_path: Path) -> tuple[Image.Image, np.ndarray]:
    """Load RGBA image, return (RGB composited, alpha mask)."""
    img = Image.open(image_path).convert("RGBA")
    alpha = np.array(img)[:, :, 3]
    rgb = Image.new("RGB", img.size, (128, 128, 128))
    rgb.paste(img, mask=img.split()[3])
    return rgb, alpha


def predict_normals(pipe, rgb: Image.Image, alpha: np.ndarray) -> np.ndarray:
    """Returns uint8 [H, W, 3] normal map."""
    output = pipe(rgb, num_inference_steps=4)
    normal_np = output.prediction[0]  # [H, W, 3] float32 in [-1, 1]
    normal_uint8 = ((normal_np + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)
    if normal_uint8.ndim == 3 and normal_uint8.shape[0] == 3:
        normal_uint8 = normal_uint8.transpose(1, 2, 0)
    normal_uint8[alpha < 10] = 0
    return normal_uint8


def predict_normals_batch(pipe, rgbs: list[Image.Image], alphas: list[np.ndarray], batch_size: int = 16) -> list[np.ndarray]:
    """Batch normals prediction. Returns list of uint8 [H, W, 3] normal maps."""
    output = pipe(rgbs, num_inference_steps=4, batch_size=batch_size)
    results = []
    for i in range(len(rgbs)):
        normal_np = output.prediction[i]
        normal_uint8 = ((normal_np + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)
        if normal_uint8.ndim == 3 and normal_uint8.shape[0] == 3:
            normal_uint8 = normal_uint8.transpose(1, 2, 0)
        normal_uint8[alphas[i] < 10] = 0
        results.append(normal_uint8)
    return results


def predict_depth(pipe, rgb: Image.Image, alpha: np.ndarray) -> np.ndarray:
    """Returns uint8 [H, W] depth map (0=far, 255=near)."""
    output = pipe(rgb, num_inference_steps=4)
    depth_np = output.prediction[0].squeeze()  # [H, W] float32 in [0, 1]
    depth_uint8 = (depth_np * 255).clip(0, 255).astype(np.uint8)
    depth_uint8[alpha < 10] = 0
    return depth_uint8


def predict_depth_batch(pipe, rgbs: list[Image.Image], alphas: list[np.ndarray], batch_size: int = 16) -> list[np.ndarray]:
    """Batch depth prediction. Returns list of uint8 [H, W] depth maps."""
    output = pipe(rgbs, num_inference_steps=4, batch_size=batch_size)
    results = []
    for i in range(len(rgbs)):
        depth_np = output.prediction[i].squeeze()
        depth_uint8 = (depth_np * 255).clip(0, 255).astype(np.uint8)
        depth_uint8[alphas[i] < 10] = 0
        results.append(depth_uint8)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich datasets with surface normals and depth using Marigold."
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Directory of example subdirs (each containing image.png).",
    )
    parser.add_argument(
        "--only-missing", action="store_true",
        help="Skip examples that already have normals.png and depth.png.",
    )
    parser.add_argument(
        "--no-normals", action="store_true",
        help="Skip normal estimation.",
    )
    parser.add_argument(
        "--no-depth", action="store_true",
        help="Skip depth estimation.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (default: auto-detect cuda/mps/cpu).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for Marigold inference (default 16 for A100).",
    )
    parser.add_argument(
        "--batch-log", type=int, default=50,
        help="Log progress every N images.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    do_normals = not args.no_normals
    do_depth = not args.no_depth

    if not do_normals and not do_depth:
        print("Nothing to do (both --no-normals and --no-depth).")
        return

    # Auto-detect device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    # Discover examples
    examples = sorted(
        d for d in args.input_dir.iterdir()
        if d.is_dir() and (d / "image.png").exists()
    )

    if args.only_missing:
        examples = [
            d for d in examples
            if (do_normals and not (d / "normals.png").exists())
            or (do_depth and not (d / "depth.png").exists())
        ]

    total = len(examples)
    if total == 0:
        print("No examples to process.")
        return

    modes = []
    if do_normals:
        modes.append("normals")
    if do_depth:
        modes.append("depth")
    print(f"Found {total} examples to enrich in {args.input_dir}")
    print(f"  Modes: {' + '.join(modes)}")

    pipes = load_pipelines(device, normals=do_normals, depth=do_depth)

    # Suppress per-inference progress bars (very spammy with batching)
    for p in pipes.values():
        p.set_progress_bar_config(disable=True)

    batch_size = args.batch_size
    start = time.monotonic()
    enriched = 0
    failed = 0

    for batch_start in range(0, total, batch_size):
        batch_dirs = examples[batch_start:batch_start + batch_size]

        # Load all images in this batch
        batch_rgbs = []
        batch_alphas = []
        batch_valid = []  # track which indices loaded OK
        for example_dir in batch_dirs:
            try:
                rgb, alpha = prepare_rgb(example_dir / "image.png")
                batch_rgbs.append(rgb)
                batch_alphas.append(alpha)
                batch_valid.append(True)
            except Exception as e:
                logger.warning("Failed to load %s: %s", example_dir.name, e)
                batch_rgbs.append(None)
                batch_alphas.append(None)
                batch_valid.append(False)
                failed += 1

        # Filter to valid images only
        valid_indices = [i for i, v in enumerate(batch_valid) if v]
        if not valid_indices:
            continue
        valid_rgbs = [batch_rgbs[i] for i in valid_indices]
        valid_alphas = [batch_alphas[i] for i in valid_indices]
        valid_dirs = [batch_dirs[i] for i in valid_indices]

        # Batch normals
        normals_results = None
        if do_normals:
            try:
                normals_results = predict_normals_batch(pipes["normals"], valid_rgbs, valid_alphas, batch_size=batch_size)
            except Exception as e:
                logger.warning("Normals batch failed (batch %d): %s", batch_start, e)
                # Fallback to single-image
                normals_results = []
                for rgb, alpha in zip(valid_rgbs, valid_alphas):
                    try:
                        normals_results.append(predict_normals(pipes["normals"], rgb, alpha))
                    except Exception as e2:
                        logger.warning("Normals single fallback failed: %s", e2)
                        normals_results.append(None)

        # Batch depth
        depth_results = None
        if do_depth:
            try:
                depth_results = predict_depth_batch(pipes["depth"], valid_rgbs, valid_alphas, batch_size=batch_size)
            except Exception as e:
                logger.warning("Depth batch failed (batch %d): %s", batch_start, e)
                depth_results = []
                for rgb, alpha in zip(valid_rgbs, valid_alphas):
                    try:
                        depth_results.append(predict_depth(pipes["depth"], rgb, alpha))
                    except Exception as e2:
                        logger.warning("Depth single fallback failed: %s", e2)
                        depth_results.append(None)

        # Save results
        for j, example_dir in enumerate(valid_dirs):
            try:
                if normals_results and normals_results[j] is not None:
                    if not (args.only_missing and (example_dir / "normals.png").exists()):
                        Image.fromarray(normals_results[j]).save(example_dir / "normals.png")

                if depth_results and depth_results[j] is not None:
                    if not (args.only_missing and (example_dir / "depth.png").exists()):
                        Image.fromarray(depth_results[j], "L").save(example_dir / "depth.png")

                # Update metadata
                meta_path = example_dir / "metadata.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if do_normals:
                        meta["has_normals"] = True
                        meta["normals_source"] = "marigold_lcm_v0.1"
                    if do_depth:
                        meta["has_depth"] = True
                        meta["depth_source"] = "marigold_depth_lcm_v1.0"
                    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

                enriched += 1
            except Exception as e:
                logger.warning("Failed saving %s: %s", example_dir.name, e)
                failed += 1

        processed = min(batch_start + len(batch_dirs), total)
        if processed % args.batch_log < batch_size or processed == total:
            elapsed = time.monotonic() - start
            speed = enriched / elapsed if elapsed > 0 else 0
            logger.info(
                "Progress: %d/%d — %d enriched, %d failed (%.1f img/s)",
                processed, total, enriched, failed, speed,
            )

    elapsed = time.monotonic() - start
    print(f"\nEnrichment complete ({' + '.join(modes)}):")
    print(f"  Enriched:  {enriched}")
    print(f"  Failed:    {failed}")
    print(f"  Total:     {total}")
    print(f"  Elapsed:   {elapsed:.1f}s")
    if enriched > 0:
        print(f"  Speed:     {enriched / elapsed:.1f} img/s")


if __name__ == "__main__":
    main()
