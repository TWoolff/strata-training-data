"""Convert anime-segmentation foreground images to Strata training format.

Handles both the original ``skytnt/anime-segmentation`` dataset (v1) and
the curated ``anime_seg_v2`` variant.  Both provide RGBA PNG foreground
character images where the **alpha channel** serves as a binary
foreground/background mask.

This adapter:

1. Discovers all ``.png`` files under the input directory, skipping any
   paths that contain ``bg`` directory components (background images are
   not useful for character training).
2. Extracts the alpha channel as a binary segmentation mask
   (threshold 128 → 0 or 255).
3. Resizes the image and mask to 512×512, centered on a transparent canvas.
4. Saves ``image.png``, ``segmentation.png`` (grayscale fg/bg mask), and
   ``metadata.json`` per example.

Pure Python — no Blender dependency.
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

ANIMESEG_SOURCE = "animeseg"

STRATA_RESOLUTION = 512

_IMAGE_EXTENSIONS = {".png"}

# Threshold for binarizing the alpha channel.
_ALPHA_THRESHOLD = 128

_MISSING_ANNOTATIONS = [
    "strata_segmentation",
    "joints",
    "draw_order",
]

# Directory name fragments that indicate background images.
_BG_MARKERS = {"bg", "bg 2", "bg 3", "bg 4", "bg 5"}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AdapterResult:
    """Result of converting anime-segmentation images to Strata format."""

    images_processed: int = 0
    images_skipped: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _is_bg_path(path: Path) -> bool:
    """Return True if any parent directory looks like a background folder."""
    for part in path.parts:
        if part.lower() in _BG_MARKERS or part.lower().startswith("bg"):
            return True
    return False


def discover_images(input_dir: Path) -> list[Path]:
    """Discover foreground PNG images under *input_dir*, skipping backgrounds.

    Args:
        input_dir: Root directory to search (recursively).

    Returns:
        Sorted list of foreground image paths.
    """
    if not input_dir.is_dir():
        logger.warning("Not a directory: %s", input_dir)
        return []

    paths = sorted(
        p
        for p in input_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in _IMAGE_EXTENSIONS
        and not _is_bg_path(p)
    )

    logger.info("Discovered %d foreground images in %s", len(paths), input_dir)
    return paths


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def _resize_to_strata(
    img: Image.Image,
    resolution: int = STRATA_RESOLUTION,
) -> Image.Image:
    """Resize an image to *resolution*×*resolution*, centered on a transparent canvas."""
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


def _extract_mask(img: Image.Image, resolution: int = STRATA_RESOLUTION) -> Image.Image:
    """Extract a binary fg/bg mask from the alpha channel.

    Args:
        img: Resized RGBA image (already 512×512).
        resolution: Expected resolution (for sanity check).

    Returns:
        Grayscale ``L`` image — 255 for foreground, 0 for background.
    """
    alpha = np.array(img.split()[-1])  # alpha channel
    mask = np.where(alpha >= _ALPHA_THRESHOLD, 255, 0).astype(np.uint8)
    return Image.fromarray(mask, mode="L")


def _build_metadata(
    image_id: str,
    source_path: Path,
    resolution: int,
    *,
    original_size: tuple[int, int],
    variant: str,
) -> dict[str, Any]:
    """Build Strata metadata dict for a single image."""
    ow, oh = original_size
    return {
        "id": image_id,
        "source": ANIMESEG_SOURCE,
        "source_variant": variant,
        "source_filename": source_path.name,
        "resolution": resolution,
        "original_width": ow,
        "original_height": oh,
        "padding_applied": ow != oh,
        "has_segmentation_mask": False,
        "has_fg_mask": True,
        "has_joints": False,
        "has_draw_order": False,
        "missing_annotations": _MISSING_ANNOTATIONS,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _save_example(
    output_dir: Path,
    image_id: str,
    image: Image.Image,
    mask: Image.Image,
    metadata: dict[str, Any],
    *,
    only_new: bool = False,
) -> bool:
    """Save image, fg/bg mask, and metadata to a per-example directory."""
    example_dir = output_dir / image_id

    if only_new and example_dir.exists():
        return False

    example_dir.mkdir(parents=True, exist_ok=True)

    image.save(example_dir / "image.png", format="PNG", compress_level=6)
    mask.save(example_dir / "segmentation.png", format="PNG", compress_level=6)

    (example_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return True


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def convert_image(
    image_path: Path,
    output_dir: Path,
    *,
    image_id: str,
    variant: str = "v1",
    resolution: int = STRATA_RESOLUTION,
    only_new: bool = False,
) -> bool:
    """Convert a single foreground image to Strata format with fg/bg mask.

    Args:
        image_path: Path to the RGBA foreground PNG.
        output_dir: Root output directory.
        image_id: Strata-format identifier for this example.
        variant: Dataset variant (``"v1"`` or ``"v2"``).
        resolution: Target square resolution.
        only_new: Skip if output already exists.

    Returns:
        True if saved, False if skipped or errored.
    """
    try:
        img = Image.open(image_path)
        img.load()
    except OSError as exc:
        logger.warning("Failed to load image %s: %s", image_path, exc)
        return False

    if img.mode != "RGBA":
        logger.debug("Converting %s from %s to RGBA", image_path, img.mode)
        img = img.convert("RGBA")

    original_size = img.size
    resized = _resize_to_strata(img, resolution)
    mask = _extract_mask(resized, resolution)

    metadata = _build_metadata(
        image_id,
        image_path,
        resolution,
        original_size=original_size,
        variant=variant,
    )

    return _save_example(
        output_dir,
        image_id,
        resized,
        mask,
        metadata,
        only_new=only_new,
    )


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    variant: str = "v1",
    resolution: int = STRATA_RESOLUTION,
    only_new: bool = False,
    max_images: int = 0,
    random_sample: bool = False,
    seed: int = 42,
) -> AdapterResult:
    """Convert anime-segmentation foreground images to Strata format.

    Args:
        input_dir: Root directory with foreground PNGs.
        output_dir: Output directory for Strata-formatted examples.
        variant: Dataset variant (``"v1"`` or ``"v2"``).
        resolution: Target square resolution.
        only_new: Skip existing output directories.
        max_images: Maximum images to process (0 = unlimited).
        random_sample: Randomly sample (requires *max_images* > 0).
        seed: Random seed for sampling.

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
        image_id = f"{ANIMESEG_SOURCE}_{variant}_{i:06d}"

        saved = convert_image(
            image_path,
            output_dir,
            image_id=image_id,
            variant=variant,
            resolution=resolution,
            only_new=only_new,
        )

        if saved:
            result.images_processed += 1
        else:
            result.images_skipped += 1

        if (i + 1) % 100 == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            logger.info("Processed %d/%d images (%.1f%%)", i + 1, total, pct)

    logger.info(
        "anime-segmentation conversion complete: %d processed, %d skipped, %d errors",
        result.images_processed,
        result.images_skipped,
        len(result.errors),
    )

    return result
