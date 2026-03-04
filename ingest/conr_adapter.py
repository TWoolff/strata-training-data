"""Convert CoNR dataset annotations + images to Strata training format.

Reads the CoNR (Collaborative Neural Rendering) dataset and converts
it into Strata's standard per-example directory format.

CoNR provides ``.npz`` annotation files containing per-pixel body
surface labels (integer values 1–9) for anime character images sourced
from Danbooru.  Original images are NOT included in the annotation
archive — they must be pre-downloaded and placed in a companion image
directory.

Expected input layout::

    input_dir/
    ├── annotation/
    │   ├── 0a0f715b298cc59037ab4f317b97eb7a.jpg.npz
    │   ├── 1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e.jpg.npz
    │   └── …
    └── images/                  # User must download from Danbooru CDN
        ├── 0a0f715b298cc59037ab4f317b97eb7a.jpg
        ├── 1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e.jpg
        └── …

Alternatively, ``--input_dir`` can point directly at the annotation
directory if images are in a sibling ``images/`` directory, or the user
can pass separate annotation and image directories.

The 9-class body surface labels are NOT mappable to Strata's 22-region
anatomy (they are unlabeled surface correspondence IDs).  This adapter
extracts a **binary foreground mask** (label > 0 = foreground) instead.

This module is pure Python (no Blender dependency).

License: CC BY 4.0
Source: https://github.com/P2Oileen/CoNR_Dataset
"""

from __future__ import annotations

import json
import logging
import pickle
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

CONR_SOURCE = "conr"

STRATA_RESOLUTION = 512

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

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
    """Result of converting CoNR examples to Strata format."""

    images_processed: int = 0
    images_skipped: int = 0
    images_missing: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _find_annotation_dir(input_dir: Path) -> Path | None:
    """Locate the annotation directory under *input_dir*.

    Supports two layouts:
    - ``input_dir/annotation/`` (standard CoNR layout)
    - ``input_dir/`` itself contains ``.npz`` files

    Returns:
        Path to directory containing ``.npz`` files, or ``None``.
    """
    ann_dir = input_dir / "annotation"
    if ann_dir.is_dir():
        return ann_dir

    # Check if input_dir itself contains .npz files
    npz_files = list(input_dir.glob("*.npz"))
    if npz_files:
        return input_dir

    return None


def _find_image_dir(input_dir: Path) -> Path | None:
    """Locate the image directory relative to *input_dir*.

    Checks:
    - ``input_dir/images/``
    - ``input_dir/../images/`` (if input_dir is the annotation dir)

    Returns:
        Path to directory containing images, or ``None``.
    """
    img_dir = input_dir / "images"
    if img_dir.is_dir():
        return img_dir

    sibling = input_dir.parent / "images"
    if sibling.is_dir():
        return sibling

    return None


def discover_annotations(annotation_dir: Path) -> list[Path]:
    """Discover all ``.npz`` annotation files.

    Args:
        annotation_dir: Directory containing ``.npz`` files.

    Returns:
        Sorted list of ``.npz`` file paths.
    """
    if not annotation_dir.is_dir():
        logger.warning("Not a directory: %s", annotation_dir)
        return []

    paths = sorted(p for p in annotation_dir.iterdir() if p.suffix == ".npz")

    logger.info("Discovered %d annotations in %s", len(paths), annotation_dir)
    return paths


def annotation_hash(npz_path: Path) -> str:
    """Extract the content hash from an annotation filename.

    CoNR filenames follow the pattern ``{md5hash}.jpg.npz``.

    Args:
        npz_path: Path to a ``.npz`` annotation file.

    Returns:
        The MD5 hash string (e.g. ``"0a0f715b298cc59037ab4f317b97eb7a"``).
    """
    # filename: "0a0f715b298cc59037ab4f317b97eb7a.jpg.npz"
    # .stem strips .npz → "hash.jpg", Path(.stem).stem strips .jpg → "hash"
    return Path(npz_path.stem).stem


def find_image_for_annotation(
    npz_path: Path,
    image_dir: Path,
) -> Path | None:
    """Find the source image corresponding to an annotation file.

    Args:
        npz_path: Path to a ``.npz`` annotation file.
        image_dir: Directory containing downloaded images.

    Returns:
        Path to the image file, or ``None`` if not found.
    """
    h = annotation_hash(npz_path)

    for ext in _IMAGE_EXTENSIONS:
        candidate = image_dir / f"{h}{ext}"
        if candidate.is_file():
            return candidate

    return None


# ---------------------------------------------------------------------------
# Annotation reading
# ---------------------------------------------------------------------------


def load_annotation(npz_path: Path) -> np.ndarray | None:
    """Load the label array from a CoNR ``.npz`` annotation.

    Args:
        npz_path: Path to the ``.npz`` file.

    Returns:
        2D uint8 array with values 0–9, or ``None`` on error.
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        label = dict(data)["label"]
        return label.astype(np.uint8)
    except (OSError, KeyError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        logger.warning("Failed to load annotation %s: %s", npz_path, exc)
        return None


def label_to_fg_mask(label: np.ndarray) -> np.ndarray:
    """Convert CoNR label array to a binary foreground mask.

    Any pixel with label > 0 is foreground (255), otherwise
    background (0).

    Args:
        label: 2D array with integer values 0–9.

    Returns:
        2D uint8 array with values 0 or 255.
    """
    return (label > 0).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def _resize_to_strata(
    img: Image.Image,
    resolution: int = STRATA_RESOLUTION,
) -> Image.Image:
    """Resize an image to *resolution*x*resolution*, preserving aspect ratio.

    Args:
        img: Input image (any mode).
        resolution: Target square resolution.

    Returns:
        *resolution*x*resolution* RGBA image.
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


def _resize_mask(
    mask: np.ndarray,
    resolution: int = STRATA_RESOLUTION,
) -> Image.Image:
    """Resize a binary mask to match Strata output, using nearest-neighbor.

    Args:
        mask: 2D uint8 array (0 or 255).
        resolution: Target square resolution.

    Returns:
        *resolution*x*resolution* grayscale image.
    """
    img = Image.fromarray(mask, mode="L")
    w, h = img.size

    scale = resolution / max(w, h)
    new_w = round(w * scale)
    new_h = round(h * scale)
    resized = img.resize((new_w, new_h), Image.NEAREST)

    canvas = Image.new("L", (resolution, resolution), 0)
    offset_x = (resolution - new_w) // 2
    offset_y = (resolution - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))

    return canvas


def _build_metadata(
    image_id: str,
    npz_filename: str,
    resolution: int,
    *,
    original_size: tuple[int, int],
    content_hash: str,
) -> dict[str, Any]:
    """Build Strata metadata dict for a single example.

    Args:
        image_id: Strata-format image identifier.
        npz_filename: Original annotation filename.
        resolution: Output resolution.
        original_size: ``(width, height)`` of the source image.
        content_hash: MD5 hash from the annotation filename.

    Returns:
        Metadata dict ready for JSON serialization.
    """
    ow, oh = original_size
    return {
        "id": image_id,
        "source": CONR_SOURCE,
        "source_filename": npz_filename,
        "content_hash": content_hash,
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
    fg_mask: Image.Image,
    metadata: dict[str, Any],
    *,
    only_new: bool = False,
) -> bool:
    """Save a single training example in Strata directory format.

    Output layout::

        output_dir/{image_id}/
        ├── image.png
        ├── segmentation.png    (binary foreground mask)
        └── metadata.json

    Args:
        output_dir: Root output directory.
        image_id: Example identifier (becomes subdirectory name).
        image: Resized RGBA image.
        fg_mask: Resized binary foreground mask.
        metadata: Metadata dict.
        only_new: Skip if output directory already exists.

    Returns:
        True if saved, False if skipped.
    """
    example_dir = output_dir / image_id

    if only_new and example_dir.exists():
        logger.debug("Skipping existing example %s", example_dir)
        return False

    example_dir.mkdir(parents=True, exist_ok=True)

    image_path = example_dir / "image.png"
    image.save(image_path, format="PNG", compress_level=6)

    mask_path = example_dir / "segmentation.png"
    fg_mask.save(mask_path, format="PNG", compress_level=6)

    meta_path = example_dir / "metadata.json"
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return True


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def convert_example(
    npz_path: Path,
    image_dir: Path,
    output_dir: Path,
    *,
    resolution: int = STRATA_RESOLUTION,
    only_new: bool = False,
) -> bool | None:
    """Convert a single CoNR annotation + image to Strata format.

    Args:
        npz_path: Path to the ``.npz`` annotation file.
        image_dir: Directory containing the source images.
        output_dir: Root output directory.
        resolution: Target square resolution.
        only_new: Skip if output already exists.

    Returns:
        True if saved, False if skipped, None if image not found.
    """
    content_hash = annotation_hash(npz_path)
    image_id = f"{CONR_SOURCE}_{content_hash}"

    # Find the source image
    image_path = find_image_for_annotation(npz_path, image_dir)
    if image_path is None:
        return None

    # Load annotation
    label = load_annotation(npz_path)
    if label is None:
        return False

    # Load image
    try:
        img = Image.open(image_path)
        img.load()
    except OSError as exc:
        logger.warning("Failed to load image %s: %s", image_path, exc)
        return False

    original_size = img.size

    # Convert
    resized_image = _resize_to_strata(img, resolution)
    fg_mask_arr = label_to_fg_mask(label)
    resized_mask = _resize_mask(fg_mask_arr, resolution)

    metadata = _build_metadata(
        image_id,
        npz_path.name,
        resolution,
        original_size=original_size,
        content_hash=content_hash,
    )

    return _save_example(
        output_dir,
        image_id,
        resized_image,
        resized_mask,
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
    """Convert CoNR annotations + images to Strata format.

    Discovers all ``.npz`` annotations, locates companion images,
    and converts each pair into a Strata per-example directory.

    Args:
        input_dir: Root dataset directory (containing ``annotation/``
            and ``images/`` subdirectories).
        output_dir: Root output directory for Strata-formatted examples.
        resolution: Target image resolution (square).
        only_new: Skip existing output directories.
        max_images: Maximum examples to process (0 = unlimited).
        random_sample: Randomly sample from discovered annotations.
        seed: Random seed for reproducible sampling.

    Returns:
        :class:`AdapterResult` summarizing the conversion.
    """
    result = AdapterResult()

    ann_dir = _find_annotation_dir(input_dir)
    if ann_dir is None:
        logger.error(
            "No annotation directory found under %s. "
            "Expected annotation/ subdirectory or .npz files.",
            input_dir,
        )
        return result

    image_dir = _find_image_dir(input_dir)
    if image_dir is None:
        logger.error(
            "No image directory found. Expected images/ subdirectory "
            "under %s or as sibling of annotation/.",
            input_dir,
        )
        return result

    npz_paths = discover_annotations(ann_dir)
    if not npz_paths:
        return result

    # Apply sampling / limiting
    if random_sample and max_images > 0:
        rng = random.Random(seed)
        sample_size = min(max_images, len(npz_paths))
        npz_paths = rng.sample(npz_paths, sample_size)
    elif max_images > 0:
        npz_paths = npz_paths[:max_images]

    total = len(npz_paths)
    logger.info(
        "Processing %d annotations from %s (images from %s)",
        total,
        ann_dir,
        image_dir,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, npz_path in enumerate(npz_paths):
        saved = convert_example(
            npz_path,
            image_dir,
            output_dir,
            resolution=resolution,
            only_new=only_new,
        )

        if saved is True:
            result.images_processed += 1
        elif saved is None:
            result.images_missing += 1
        else:
            result.images_skipped += 1

        if (i + 1) % 100 == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            logger.info(
                "Processed %d/%d annotations (%.1f%%) — %d converted, %d missing, %d skipped",
                i + 1,
                total,
                pct,
                result.images_processed,
                result.images_missing,
                result.images_skipped,
            )

    logger.info(
        "CoNR conversion complete: %d processed, %d missing images, %d skipped, %d errors",
        result.images_processed,
        result.images_missing,
        result.images_skipped,
        len(result.errors),
    )

    return result
