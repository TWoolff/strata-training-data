"""Convert manually curated diverse character images to Strata training format.

Pipeline per image:
1. Background removal via rembg (U²-Net)
2. Resize to 512×512 with aspect-ratio-preserving padding
3. Extract binary foreground mask from alpha channel
4. Save image.png + segmentation.png + metadata.json

Post-processing (separate steps via run_enrich.py / run_depth_enrich.py):
- RTMPose joint enrichment
- Depth Anything v2 draw order estimation

Accepts any common image format (PNG, JPG, JPEG, WebP).
Pure Python — no Blender dependency.  Requires ``rembg`` for background removal.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CURATED_DIVERSE_SOURCE = "curated_diverse"

STRATA_RESOLUTION = 512

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

_ALPHA_THRESHOLD = 128

_MISSING_ANNOTATIONS = [
    "strata_segmentation",
    "joints",
    "draw_order",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AdapterResult:
    """Result of converting curated diverse images to Strata format."""

    images_processed: int = 0
    images_skipped: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Background removal
# ---------------------------------------------------------------------------

_rembg_session = None


def _get_rembg_session() -> Any:
    """Lazily initialize rembg session (downloads model on first use)."""
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session

        _rembg_session = new_session("u2net")
        logger.info("rembg session initialized (u2net)")
    return _rembg_session


def remove_background(img: Image.Image) -> Image.Image:
    """Remove background from an image using rembg.

    Args:
        img: Input image (any mode).

    Returns:
        RGBA image with background removed (transparent).
    """
    from rembg import remove

    session = _get_rembg_session()
    result = remove(img, session=session)
    if result.mode != "RGBA":
        result = result.convert("RGBA")
    return result


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _resize_to_strata(
    img: Image.Image,
    resolution: int = STRATA_RESOLUTION,
) -> Image.Image:
    """Resize to *resolution*×*resolution*, centered on a transparent canvas."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    w, h = img.size
    if w == resolution and h == resolution:
        return img

    scale = resolution / max(w, h)
    new_w = round(w * scale)
    new_h = round(h * scale)
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))
    offset_x = (resolution - new_w) // 2
    offset_y = (resolution - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))

    return canvas


def _extract_mask(img: Image.Image) -> Image.Image:
    """Extract binary fg/bg mask from alpha channel.

    Args:
        img: RGBA image (already resized).

    Returns:
        Grayscale ``L`` image — 255 for foreground, 0 for background.
    """
    alpha = np.array(img.split()[-1])
    mask = np.where(alpha >= _ALPHA_THRESHOLD, 255, 0).astype(np.uint8)
    return Image.fromarray(mask, mode="L")


def _build_metadata(
    image_id: str,
    source_path: Path,
    resolution: int,
    *,
    original_size: tuple[int, int],
) -> dict[str, Any]:
    """Build Strata metadata dict for a single image."""
    ow, oh = original_size
    return {
        "id": image_id,
        "source": CURATED_DIVERSE_SOURCE,
        "source_filename": source_path.name,
        "resolution": resolution,
        "original_width": ow,
        "original_height": oh,
        "padding_applied": ow != oh,
        "has_segmentation_mask": False,
        "has_fg_mask": True,
        "has_joints": False,
        "has_draw_order": False,
        "missing_annotations": list(_MISSING_ANNOTATIONS),
    }


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_images(input_dir: Path) -> list[Path]:
    """Discover all image files under *input_dir* recursively.

    Args:
        input_dir: Root directory containing curated images.

    Returns:
        Sorted list of image file paths.
    """
    if not input_dir.is_dir():
        logger.warning("Not a directory: %s", input_dir)
        return []

    paths = sorted(
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS and not p.name.startswith(".")
    )

    logger.info("Discovered %d images in %s", len(paths), input_dir)
    return paths


# ---------------------------------------------------------------------------
# Per-image conversion
# ---------------------------------------------------------------------------


def convert_image(
    image_path: Path,
    output_dir: Path,
    *,
    image_id: str,
    resolution: int = STRATA_RESOLUTION,
    only_new: bool = False,
    remove_bg: bool = True,
) -> bool:
    """Convert a single curated image to Strata training format.

    Args:
        image_path: Path to the source image.
        output_dir: Root output directory.
        image_id: Strata-format identifier.
        resolution: Target square resolution.
        only_new: Skip if output already exists.
        remove_bg: Run rembg background removal.

    Returns:
        True if saved, False if skipped or errored.
    """
    example_dir = output_dir / image_id

    if only_new and example_dir.exists():
        return False

    try:
        img = Image.open(image_path)
        img.load()
    except OSError as exc:
        logger.warning("Failed to load image %s: %s", image_path, exc)
        return False

    original_size = img.size

    if remove_bg:
        img = remove_background(img)

    resized = _resize_to_strata(img, resolution)
    mask = _extract_mask(resized)

    # Check if mask has enough foreground (skip near-empty results).
    fg_ratio = np.count_nonzero(np.array(mask)) / (resolution * resolution)
    if fg_ratio < 0.02:
        logger.warning(
            "Skipping %s: foreground too small (%.1f%%)", image_path.name, fg_ratio * 100
        )
        return False

    example_dir.mkdir(parents=True, exist_ok=True)

    resized.save(example_dir / "image.png", format="PNG", compress_level=6)
    mask.save(example_dir / "segmentation.png", format="PNG", compress_level=6)

    metadata = _build_metadata(image_id, image_path, resolution, original_size=original_size)
    (example_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return True


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    resolution: int = STRATA_RESOLUTION,
    only_new: bool = False,
    max_images: int = 0,
    random_sample: bool = False,
    seed: int = 42,
    remove_bg: bool = True,
) -> AdapterResult:
    """Convert curated diverse images to Strata format.

    Args:
        input_dir: Root directory with curated images.
        output_dir: Output directory for Strata-formatted examples.
        resolution: Target square resolution.
        only_new: Skip existing output directories.
        max_images: Maximum images to process (0 = unlimited).
        random_sample: Randomly sample (requires *max_images* > 0).
        seed: Random seed for sampling.
        remove_bg: Run rembg background removal.

    Returns:
        :class:`AdapterResult` summarizing the conversion.
    """
    result = AdapterResult()

    image_paths = discover_images(input_dir)
    if not image_paths:
        return result

    if random_sample and max_images > 0:
        rng = random.Random(seed)
        sample_size = min(max_images, len(image_paths))
        image_paths = rng.sample(image_paths, sample_size)
    elif max_images > 0:
        image_paths = image_paths[:max_images]

    total = len(image_paths)
    logger.info("Processing %d images from %s", total, input_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, image_path in enumerate(image_paths):
        image_id = f"{CURATED_DIVERSE_SOURCE}_{image_path.stem}"

        saved = convert_image(
            image_path,
            output_dir,
            image_id=image_id,
            resolution=resolution,
            only_new=only_new,
            remove_bg=remove_bg,
        )

        if saved:
            result.images_processed += 1
        else:
            result.images_skipped += 1

        if (i + 1) % 10 == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            logger.info("Processed %d/%d images (%.1f%%)", i + 1, total, pct)

    logger.info(
        "Curated diverse conversion complete: %d processed, %d skipped, %d errors",
        result.images_processed,
        result.images_skipped,
        len(result.errors),
    )

    return result
