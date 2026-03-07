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


def load_pipeline(device: torch.device, mode: str):
    """Load a single Marigold pipeline. mode is 'normals' or 'depth'."""
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    if mode == "normals":
        from diffusers import MarigoldNormalsPipeline
        pipe = MarigoldNormalsPipeline.from_pretrained(
            MARIGOLD_NORMALS_MODEL, torch_dtype=dtype,
        ).to(device)
    else:
        from diffusers import MarigoldDepthPipeline
        pipe = MarigoldDepthPipeline.from_pretrained(
            MARIGOLD_DEPTH_MODEL, torch_dtype=dtype,
        ).to(device)
    pipe.set_progress_bar_config(disable=True)
    logger.info("Loaded Marigold %s pipeline on %s", mode, device)
    return pipe


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
    print(f"  Strategy: sequential model loading (one model in VRAM at a time)")

    batch_size = args.batch_size
    start = time.monotonic()
    failed = 0

    # Process each mode separately so only one model occupies VRAM at a time
    for mode in modes:
        is_normals = mode == "normals"
        out_file = "normals.png" if is_normals else "depth.png"

        # Filter to examples needing this mode
        if args.only_missing:
            mode_examples = [d for d in examples if not (d / out_file).exists()]
        else:
            mode_examples = examples
        mode_total = len(mode_examples)

        if mode_total == 0:
            print(f"\n  {mode}: all done, skipping.")
            continue

        print(f"\n  {mode}: {mode_total} images to process...")
        pipe = load_pipeline(device, mode)

        mode_done = 0
        mode_start = time.monotonic()

        for batch_start in range(0, mode_total, batch_size):
            batch_dirs = mode_examples[batch_start:batch_start + batch_size]

            # Load batch images
            valid = []
            for d in batch_dirs:
                try:
                    rgb, alpha = prepare_rgb(d / "image.png")
                    valid.append((d, rgb, alpha))
                except Exception as e:
                    logger.warning("Failed to load %s: %s", d.name, e)
                    failed += 1

            if not valid:
                continue

            dirs, rgbs, alphas = zip(*valid)

            # Run inference
            try:
                if is_normals:
                    results = predict_normals_batch(pipe, list(rgbs), list(alphas), batch_size=batch_size)
                else:
                    results = predict_depth_batch(pipe, list(rgbs), list(alphas), batch_size=batch_size)
            except Exception as e:
                logger.warning("Batch failed at %d: %s — falling back to single", batch_start, e)
                results = []
                predict_fn = predict_normals if is_normals else predict_depth
                for rgb, alpha in zip(rgbs, alphas):
                    try:
                        results.append(predict_fn(pipe, rgb, alpha))
                    except Exception as e2:
                        logger.warning("Single fallback failed: %s", e2)
                        results.append(None)

            # Save results
            for j, d in enumerate(dirs):
                if results[j] is None:
                    failed += 1
                    continue
                try:
                    if is_normals:
                        Image.fromarray(results[j]).save(d / out_file)
                    else:
                        Image.fromarray(results[j], "L").save(d / out_file)
                    mode_done += 1
                except Exception as e:
                    logger.warning("Failed saving %s: %s", d.name, e)
                    failed += 1

            processed = min(batch_start + len(batch_dirs), mode_total)
            if processed % args.batch_log < batch_size or processed == mode_total:
                elapsed = time.monotonic() - mode_start
                speed = mode_done / elapsed if elapsed > 0 else 0
                logger.info(
                    "  %s: %d/%d done (%.1f img/s)",
                    mode, processed, mode_total, speed,
                )

        # Free VRAM before loading next model
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elapsed = time.monotonic() - mode_start
        print(f"  {mode}: {mode_done} done in {elapsed:.0f}s ({mode_done / elapsed:.1f} img/s)")

    # Update metadata for all completed examples (single pass)
    print("\nUpdating metadata...")
    meta_updated = 0
    for d in examples:
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            changed = False
            if do_normals and (d / "normals.png").exists():
                if not meta.get("has_normals"):
                    meta["has_normals"] = True
                    meta["normals_source"] = "marigold_lcm_v0.1"
                    changed = True
            if do_depth and (d / "depth.png").exists():
                if not meta.get("has_depth"):
                    meta["has_depth"] = True
                    meta["depth_source"] = "marigold_depth_lcm_v1.0"
                    changed = True
            if changed:
                meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
                meta_updated += 1
        except Exception:
            pass
    print(f"  Updated {meta_updated} metadata files.")

    elapsed = time.monotonic() - start
    print(f"\nEnrichment complete ({' + '.join(modes)}):")
    print(f"  Failed:    {failed}")
    print(f"  Total:     {total}")
    print(f"  Elapsed:   {elapsed:.1f}s")


if __name__ == "__main__":
    main()
