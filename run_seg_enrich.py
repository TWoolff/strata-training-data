"""Enrich image datasets with 22-class segmentation masks using trained Model 1.

Walks a directory of per-example subdirectories (each containing image.png),
runs the trained segmentation model, and saves segmentation.png (pixel value =
region ID, 0-21) alongside the original image. Updates metadata.json.

Can also process a flat directory of loose images (no subdirectories).

Usage::

    # Process existing pipeline output (subdirs with image.png)
    python run_seg_enrich.py \
        --input-dir ./output/anime_seg \
        --checkpoint checkpoints/segmentation/best.pt

    # Process flat directory of loose images
    python run_seg_enrich.py \
        --input-dir /Volumes/TAMWoolff/data/preprocessed/currated_diverse \
        --checkpoint checkpoints/segmentation/best.pt \
        --output-dir ./output/anime_seg_enriched \
        --flat

    # Skip already-enriched examples
    python run_seg_enrich.py \
        --input-dir ./output/anime_seg \
        --checkpoint checkpoints/segmentation/best.pt \
        --only-missing
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
from training.models.segmentation_model import SegmentationModel
from training.utils.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)

REGION_NAMES = [
    "background", "head", "neck", "chest", "spine", "hips",
    "shoulder_l", "upper_arm_l", "forearm_l", "hand_l",
    "shoulder_r", "upper_arm_r", "forearm_r", "hand_r",
    "upper_leg_l", "lower_leg_l", "foot_l",
    "upper_leg_r", "lower_leg_r", "foot_r",
    "accessory", "hair_back",
]


def load_seg_model(checkpoint_path: Path, device: torch.device) -> SegmentationModel:
    """Load trained segmentation model from checkpoint."""
    model = SegmentationModel(num_classes=22, pretrained_backbone=False)
    info = load_checkpoint(checkpoint_path, model)
    logger.info(
        "Loaded segmentation checkpoint: epoch %d, metrics: %s",
        info["epoch"],
        {k: f"{v:.4f}" for k, v in info["metrics"].items() if isinstance(v, float)},
    )
    model.to(device)
    model.eval()
    return model


def predict_segmentation(
    model: SegmentationModel,
    image_path: Path,
    device: torch.device,
    resolution: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run segmentation inference on a single image.

    Returns:
        (seg_mask, draw_order, confidence) — all as numpy arrays [H, W].
        seg_mask: uint8 with pixel values 0-21 (region IDs).
        draw_order: uint8 [0-255] depth map.
        confidence: float32 [0-1].
    """
    img = Image.open(image_path).convert("RGBA")
    img_resized = img.resize((resolution, resolution), Image.BILINEAR)

    # Convert to RGB float tensor [1, 3, H, W]
    img_arr = np.array(img_resized, dtype=np.float32)[:, :, :3] / 255.0
    tensor = torch.from_numpy(img_arr.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)

    # Segmentation: argmax over 22 classes
    seg_logits = out["segmentation"][0]  # [22, H, W]
    seg_mask = seg_logits.argmax(dim=0).cpu().numpy().astype(np.uint8)

    # Draw order: already sigmoid'd, scale to 0-255
    draw_order = (out["draw_order"][0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    # Confidence
    confidence = out["confidence"][0, 0].cpu().numpy()

    return seg_mask, draw_order, confidence


def process_example_dir(
    model: SegmentationModel,
    example_dir: Path,
    device: torch.device,
    resolution: int,
) -> bool:
    """Process a single example directory (contains image.png)."""
    image_path = example_dir / "image.png"
    if not image_path.exists():
        return False

    seg_mask, draw_order, confidence = predict_segmentation(
        model, image_path, device, resolution
    )

    # Apply fg/bg mask if available — keep background as 0
    existing_seg = example_dir / "segmentation.png"
    if existing_seg.exists():
        old_mask = np.array(Image.open(existing_seg).convert("L"))
        # If old mask is binary (0/255 fg/bg), use it to zero out background
        unique_vals = np.unique(old_mask)
        if set(unique_vals).issubset({0, 255}):
            bg_pixels = old_mask == 0
            seg_mask[bg_pixels] = 0

    # Save
    Image.fromarray(seg_mask).save(example_dir / "segmentation.png")
    Image.fromarray(draw_order).save(example_dir / "draw_order.png")

    # Update metadata
    meta_path = example_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {"id": example_dir.name, "source": example_dir.parent.name}

    meta["has_segmentation_mask"] = True
    meta["has_fg_mask"] = True
    meta["has_draw_order"] = True
    meta["segmentation_source"] = "model_v1"
    if "missing_annotations" in meta:
        meta["missing_annotations"] = [
            a for a in meta["missing_annotations"]
            if a != "strata_segmentation"
        ]

    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return True


def process_flat_image(
    model: SegmentationModel,
    image_path: Path,
    output_dir: Path,
    device: torch.device,
    resolution: int,
    rembg_session=None,
) -> bool:
    """Process a loose image file → create example subdirectory with all outputs."""
    example_id = f"anime_seg_{image_path.stem}"
    example_dir = output_dir / example_id
    example_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(image_path).convert("RGBA")
    original_w, original_h = img.size

    # Remove background if rembg session provided
    if rembg_session is not None:
        from rembg import remove
        img = remove(img, session=rembg_session)

    # Resize and save as image.png (RGBA, 512x512)
    img_resized = img.resize((resolution, resolution), Image.BILINEAR)
    img_resized.save(example_dir / "image.png")

    # Run segmentation on the bg-removed image
    seg_mask, draw_order, confidence = predict_segmentation(
        model, example_dir / "image.png", device, resolution
    )

    # Zero out segmentation where alpha is transparent (background)
    alpha = np.array(img_resized)[:, :, 3]
    seg_mask[alpha < 10] = 0
    draw_order[alpha < 10] = 0

    Image.fromarray(seg_mask).save(example_dir / "segmentation.png")
    Image.fromarray(draw_order).save(example_dir / "draw_order.png")

    # Metadata
    meta = {
        "id": example_id,
        "source": "anime_seg",
        "source_filename": image_path.name,
        "resolution": resolution,
        "original_width": original_w,
        "original_height": original_h,
        "has_segmentation_mask": True,
        "has_fg_mask": True,
        "has_draw_order": True,
        "has_joints": False,
        "segmentation_source": "model_v1",
    }
    (example_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich datasets with 22-class segmentation using trained Model 1."
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Directory of example subdirs (or flat images with --flat).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (required with --flat, otherwise in-place).",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("checkpoints/segmentation/best.pt"),
        help="Path to segmentation model checkpoint.",
    )
    parser.add_argument(
        "--resolution", type=int, default=RENDER_RESOLUTION,
        help=f"Output resolution (default: {RENDER_RESOLUTION}).",
    )
    parser.add_argument(
        "--flat", action="store_true",
        help="Input is a flat directory of images (not example subdirs).",
    )
    parser.add_argument(
        "--only-missing", action="store_true",
        help="Skip examples that already have model-generated segmentation.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (default: auto-detect mps/cuda/cpu).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Auto-detect device
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    model = load_seg_model(args.checkpoint, device)

    start = time.monotonic()
    enriched = 0
    skipped = 0
    failed = 0

    if args.flat:
        # Flat directory of loose images
        if args.output_dir is None:
            parser.error("--output-dir is required with --flat")
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize rembg for background removal
        rembg_session = None
        try:
            from rembg import new_session
            rembg_session = new_session("u2net")
            logger.info("rembg loaded — background removal enabled")
        except ImportError:
            logger.warning("rembg not installed — skipping background removal")

        images = sorted(
            p for p in args.input_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
            and not p.name.startswith("._")
        )
        total = len(images)
        logger.info("Found %d images in %s", total, args.input_dir)

        for i, img_path in enumerate(images):
            example_id = f"anime_seg_{img_path.stem}"
            example_dir = args.output_dir / example_id

            if args.only_missing and (example_dir / "segmentation.png").exists():
                meta_path = example_dir / "metadata.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if meta.get("segmentation_source") == "model_v1":
                        skipped += 1
                        continue

            ok = process_flat_image(model, img_path, args.output_dir, device, args.resolution, rembg_session)
            if ok:
                enriched += 1
            else:
                failed += 1

            if (i + 1) % 50 == 0 or (i + 1) == total:
                elapsed = time.monotonic() - start
                speed = enriched / elapsed if elapsed > 0 else 0
                logger.info(
                    "Progress: %d/%d — %d enriched, %d skipped, %d failed (%.1f img/s)",
                    i + 1, total, enriched, skipped, failed, speed,
                )
    else:
        # Subdirectory mode: look for image.png in each subdir
        examples = sorted(
            d for d in args.input_dir.iterdir()
            if d.is_dir() and (d / "image.png").exists()
        )
        total = len(examples)
        logger.info("Found %d example dirs in %s", total, args.input_dir)

        for i, example_dir in enumerate(examples):
            if args.only_missing:
                meta_path = example_dir / "metadata.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if meta.get("segmentation_source") == "model_v1":
                        skipped += 1
                        continue

            ok = process_example_dir(model, example_dir, device, args.resolution)
            if ok:
                enriched += 1
            else:
                failed += 1

            if (i + 1) % 50 == 0 or (i + 1) == total:
                elapsed = time.monotonic() - start
                speed = enriched / elapsed if elapsed > 0 else 0
                logger.info(
                    "Progress: %d/%d — %d enriched, %d skipped, %d failed (%.1f img/s)",
                    i + 1, total, enriched, skipped, failed, speed,
                )

    elapsed = time.monotonic() - start
    print(f"\nSegmentation enrichment complete:")
    print(f"  Enriched:  {enriched}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")
    print(f"  Total:     {enriched + skipped + failed}")
    print(f"  Elapsed:   {elapsed:.1f}s")
    if enriched > 0:
        print(f"  Speed:     {enriched / elapsed:.1f} img/s")


if __name__ == "__main__":
    main()
