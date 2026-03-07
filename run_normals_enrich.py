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


def predict_depth(pipe, rgb: Image.Image, alpha: np.ndarray) -> np.ndarray:
    """Returns uint8 [H, W] depth map (0=far, 255=near)."""
    output = pipe(rgb, num_inference_steps=4)
    depth_np = output.prediction[0].squeeze()  # [H, W] float32 in [0, 1]
    depth_uint8 = (depth_np * 255).clip(0, 255).astype(np.uint8)
    depth_uint8[alpha < 10] = 0
    return depth_uint8


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

    start = time.monotonic()
    enriched = 0
    failed = 0

    for i, example_dir in enumerate(examples):
        try:
            rgb, alpha = prepare_rgb(example_dir / "image.png")

            if do_normals and not (args.only_missing and (example_dir / "normals.png").exists()):
                normal_map = predict_normals(pipes["normals"], rgb, alpha)
                Image.fromarray(normal_map).save(example_dir / "normals.png")

            if do_depth and not (args.only_missing and (example_dir / "depth.png").exists()):
                depth_map = predict_depth(pipes["depth"], rgb, alpha)
                Image.fromarray(depth_map, "L").save(example_dir / "depth.png")

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
            logger.warning("Failed %s: %s", example_dir.name, e)
            failed += 1

        if (i + 1) % args.batch_log == 0 or (i + 1) == total:
            elapsed = time.monotonic() - start
            speed = enriched / elapsed if elapsed > 0 else 0
            logger.info(
                "Progress: %d/%d — %d enriched, %d failed (%.1f img/s)",
                i + 1, total, enriched, failed, speed,
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
