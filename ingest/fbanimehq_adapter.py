"""Convert FBAnimeHQ images to Strata training format.

Reads the FBAnimeHQ dataset (full-body anime character images from
HuggingFace ``skytnt/fbanimehq``) and converts them into Strata's
standard per-example directory format.

FBAnimeHQ is organized as sharded zip archives, each containing
numbered bucket directories with 1,000 PNG images::

    data/fbanimehq-00/
    ├── 0000/          # 1,000 images (000000.png … 000999.png)
    ├── 0001/
    └── …0009/         # 10 buckets per shard = 10,000 images

Images are 512×1024 (width×height) RGB PNGs — portrait full-body
anime characters on white backgrounds.  No annotations are provided.

This adapter resizes the longest edge to the target resolution,
centers the image on a transparent canvas, and generates metadata
flagging all annotations as missing.

This module is pure Python (no Blender dependency) so it can be
imported outside Blender for testing and validation.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FBANIMEHQ_SOURCE = "fbanimehq"

STRATA_RESOLUTION = 512

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Annotations that FBAnimeHQ does NOT provide.
_MISSING_ANNOTATIONS = [
    "strata_segmentation",
    "joints",
    "draw_order",
    "fg_mask",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AdapterResult:
    """Result of converting FBAnimeHQ images to Strata format."""

    images_processed: int = 0
    images_skipped: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_images(input_dir: Path) -> list[Path]:
    """Discover all image files under *input_dir* recursively.

    FBAnimeHQ uses a ``shard/bucket/image.png`` hierarchy.  This
    function walks the tree and returns all PNG/JPG files, sorted by
    path for deterministic ordering.

    Args:
        input_dir: Root dataset directory (e.g. ``data/fbanimehq-00``
            or the parent ``data/`` directory containing multiple shards).

    Returns:
        Sorted list of image file paths.
    """
    if not input_dir.is_dir():
        logger.warning("Not a directory: %s", input_dir)
        return []

    paths = sorted(
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )

    logger.info("Discovered %d images in %s", len(paths), input_dir)
    return paths


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def _resize_to_strata(
    img: Image.Image,
    resolution: int = STRATA_RESOLUTION,
) -> Image.Image:
    """Resize an image to *resolution*×*resolution*, preserving aspect ratio.

    The longest edge is scaled to *resolution*, then the image is
    centered on a transparent RGBA canvas.  This avoids cropping
    heads/feet on tall portrait images.

    Args:
        img: Input image (any mode).
        resolution: Target square resolution.

    Returns:
        *resolution*×*resolution* RGBA image.
    """
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


def _build_metadata(
    image_id: str,
    source_path: Path,
    resolution: int,
    *,
    original_size: tuple[int, int],
) -> dict[str, Any]:
    """Build Strata metadata dict for a single image.

    Args:
        image_id: Strata-format image identifier.
        source_path: Original image file path.
        resolution: Output resolution.
        original_size: ``(width, height)`` of the source image.

    Returns:
        Metadata dict ready for JSON serialization.
    """
    ow, oh = original_size
    return {
        "id": image_id,
        "source": FBANIMEHQ_SOURCE,
        "source_filename": source_path.name,
        "resolution": resolution,
        "original_width": ow,
        "original_height": oh,
        "padding_applied": ow != oh,
        "has_segmentation_mask": False,
        "has_fg_mask": False,
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
    metadata: dict[str, Any],
    *,
    only_new: bool = False,
) -> bool:
    """Save a single training example in Strata directory format.

    Output layout::

        output_dir/{image_id}/
        ├── image.png
        └── metadata.json

    Args:
        output_dir: Root output directory.
        image_id: Example identifier (becomes subdirectory name).
        image: Resized RGBA image.
        metadata: Metadata dict.
        only_new: Skip if output directory already exists.

    Returns:
        True if saved, False if skipped or errored.
    """
    example_dir = output_dir / image_id

    if only_new and example_dir.exists():
        logger.debug("Skipping existing example %s", example_dir)
        return False

    example_dir.mkdir(parents=True, exist_ok=True)

    image_path = example_dir / "image.png"
    image.save(image_path, format="PNG", compress_level=6)

    meta_path = example_dir / "metadata.json"
    meta_path.write_text(
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
    resolution: int = STRATA_RESOLUTION,
    only_new: bool = False,
) -> bool:
    """Convert a single FBAnimeHQ image to Strata training format.

    Args:
        image_path: Path to the source image file.
        output_dir: Root output directory.
        resolution: Target square resolution.
        only_new: Skip if output already exists.

    Returns:
        True if saved, False if skipped or errored.
    """
    image_id = f"{FBANIMEHQ_SOURCE}_{image_path.stem}"

    try:
        img = Image.open(image_path)
        img.load()
    except OSError as exc:
        logger.warning("Failed to load image %s: %s", image_path, exc)
        return False

    original_size = img.size
    resized = _resize_to_strata(img, resolution)

    metadata = _build_metadata(
        image_id,
        image_path,
        resolution,
        original_size=original_size,
    )

    return _save_example(
        output_dir,
        image_id,
        resized,
        metadata,
        only_new=only_new,
    )


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    resolution: int = STRATA_RESOLUTION,
    only_new: bool = False,
    max_images: int = 0,
    random_sample: bool = False,
    seed: int = 42,
) -> AdapterResult:
    """Convert FBAnimeHQ images to Strata format.

    Discovers all images under *input_dir* (recursively walking the
    ``shard/bucket/`` hierarchy), optionally samples a random subset,
    and converts each image into a Strata per-example directory.

    Args:
        input_dir: Root dataset directory containing shard/bucket
            subdirectories with images.
        output_dir: Root output directory for Strata-formatted examples.
        resolution: Target image resolution (square).
        only_new: Skip existing output directories.
        max_images: Maximum images to process (0 = unlimited).
        random_sample: Randomly sample from discovered images
            (requires *max_images* > 0 to set sample size).
        seed: Random seed for reproducible sampling.

    Returns:
        :class:`AdapterResult` summarizing the conversion.
    """
    result = AdapterResult()

    image_paths = discover_images(input_dir)
    if not image_paths:
        return result

    # Apply sampling / limiting
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
        saved = convert_image(
            image_path,
            output_dir,
            resolution=resolution,
            only_new=only_new,
        )

        if saved:
            result.images_processed += 1
        else:
            result.images_skipped += 1

        # Progress logging every 100 images
        if (i + 1) % 100 == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            logger.info("Processed %d/%d images (%.1f%%)", i + 1, total, pct)

    logger.info(
        "FBAnimeHQ conversion complete: %d processed, %d skipped, %d errors",
        result.images_processed,
        result.images_skipped,
        len(result.errors),
    )

    return result
