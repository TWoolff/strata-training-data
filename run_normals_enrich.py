"""Enrich image datasets with surface normal maps using Marigold.

Walks a directory of per-example subdirectories (each containing image.png),
runs Marigold normal estimation, and saves normals.png alongside the original.

Usage::

    # Process existing pipeline output (subdirs with image.png)
    python run_normals_enrich.py --input-dir ./data_cloud/humanrig

    # Skip already-enriched examples
    python run_normals_enrich.py --input-dir ./data_cloud/humanrig --only-missing

    # Use CPU instead of auto-detect
    python run_normals_enrich.py --input-dir ./data_cloud/humanrig --device cpu
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

MARIGOLD_MODEL = "prs-eth/marigold-normals-lcm-v0-1"


def load_normals_pipeline(device: torch.device):
    """Load Marigold normals pipeline."""
    from diffusers import MarigoldNormalsPipeline

    pipe = MarigoldNormalsPipeline.from_pretrained(
        MARIGOLD_MODEL,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    pipe = pipe.to(device)
    logger.info("Loaded Marigold normals pipeline on %s", device)
    return pipe


def predict_normals(pipe, image_path: Path) -> np.ndarray:
    """Run normal estimation on a single image.

    Returns:
        normals: uint8 [H, W, 3] normal map (RGB, standard encoding).
    """
    img = Image.open(image_path).convert("RGBA")
    alpha = np.array(img)[:, :, 3]

    # Composite onto neutral normal background (128, 128, 255) for RGB input
    rgb = Image.new("RGB", img.size, (128, 128, 255))
    rgb.paste(img, mask=img.split()[3])

    output = pipe(rgb, num_inference_steps=4)
    normal_np = output.prediction[0]  # [H, W, 3] float32 in [-1, 1]

    # Map [-1, 1] to [0, 255]
    normal_uint8 = ((normal_np + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)
    if normal_uint8.ndim == 3 and normal_uint8.shape[0] == 3:
        normal_uint8 = normal_uint8.transpose(1, 2, 0)

    # Mask out background
    normal_uint8[alpha < 10] = 0

    return normal_uint8


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich datasets with surface normals using Marigold."
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Directory of example subdirs (each containing image.png).",
    )
    parser.add_argument(
        "--only-missing", action="store_true",
        help="Skip examples that already have normals.png.",
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
        examples = [d for d in examples if not (d / "normals.png").exists()]

    total = len(examples)
    if total == 0:
        print("No examples to process.")
        return

    print(f"Found {total} examples to enrich in {args.input_dir}")

    pipe = load_normals_pipeline(device)

    start = time.monotonic()
    enriched = 0
    failed = 0

    for i, example_dir in enumerate(examples):
        try:
            normal_map = predict_normals(pipe, example_dir / "image.png")
            Image.fromarray(normal_map).save(example_dir / "normals.png")

            # Update metadata
            meta_path = example_dir / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["has_normals"] = True
                meta["normals_source"] = "marigold_lcm_v0.1"
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
    print(f"\nNormals enrichment complete:")
    print(f"  Enriched:  {enriched}")
    print(f"  Failed:    {failed}")
    print(f"  Total:     {total}")
    print(f"  Elapsed:   {elapsed:.1f}s")
    if enriched > 0:
        print(f"  Speed:     {enriched / elapsed:.1f} img/s")


if __name__ == "__main__":
    main()
