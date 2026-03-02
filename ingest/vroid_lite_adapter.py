"""Convert VRoid-Lite renders to Strata training format.

Reads the ``vroid_lite`` dataset (4,651 RGBA character renders from 16
VRoid characters) and converts them into Strata's standard per-example
directory format.

The dataset lives at ``data/preprocessed/vroid_lite/`` and is structured::

    data/preprocessed/vroid_lite/
    ├── metadata.jsonl          # 4,651 lines — rich per-image metadata
    └── vroid_dataset/          # 4,651 PNG images (UUID filenames)
        ├── 4ade1940-333a-4a89-bff8-281d3ebf0912.png
        └── …

Images are 1536×1024 RGBA PNGs (landscape).  Each line in
``metadata.jsonl`` describes one image with fields like ``vrm_name``,
``clip_name``, ``camera_profile``, ``facial_expression``, lighting,
camera parameters, and colour shifts.

Discovery is JSONL-driven: we read ``metadata.jsonl`` first and resolve
each entry to its image file.  Images without metadata are ignored;
missing images are logged and skipped.

No annotations are provided (no segmentation masks, joints, or draw
order), so this adapter produces image + metadata only.

This module is pure Python (no Blender dependency).
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

VROID_LITE_SOURCE = "vroid_lite"

STRATA_RESOLUTION = 512

_MISSING_ANNOTATIONS = [
    "strata_segmentation",
    "joints",
    "draw_order",
    "fg_mask",
]

# Subdirectory that contains the actual PNG images.
_IMAGE_SUBDIR = "vroid_dataset"

# Metadata JSONL filename.
_METADATA_FILENAME = "metadata.jsonl"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class VroidLiteEntry:
    """A single metadata row paired with its resolved image path."""

    metadata: dict[str, Any]
    image_path: Path


@dataclass
class AdapterResult:
    """Result of converting VRoid-Lite images to Strata format."""

    images_processed: int = 0
    images_skipped: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_entries(input_dir: Path) -> list[VroidLiteEntry]:
    """Parse ``metadata.jsonl`` and resolve each entry to its image file.

    Entries whose image file is missing are logged and skipped.
    Malformed JSONL lines are logged and skipped.

    Args:
        input_dir: Root dataset directory containing ``metadata.jsonl``
            and the ``vroid_dataset/`` image subdirectory.

    Returns:
        List of :class:`VroidLiteEntry` sorted by filename for
        deterministic ordering.
    """
    jsonl_path = input_dir / _METADATA_FILENAME
    if not jsonl_path.is_file():
        logger.warning("Metadata file not found: %s", jsonl_path)
        return []

    image_dir = input_dir / _IMAGE_SUBDIR

    entries: list[VroidLiteEntry] = []
    malformed = 0
    missing = 0

    with jsonl_path.open(encoding="utf-8") as fh:
        for line_num, raw_line in enumerate(fh, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Malformed JSONL at line %d: %s",
                    line_num,
                    exc,
                )
                malformed += 1
                continue

            file_name = row.get("file_name")
            if not file_name:
                logger.warning("Missing file_name at line %d", line_num)
                malformed += 1
                continue

            image_path = image_dir / file_name
            if not image_path.is_file():
                logger.warning("Image not found: %s (line %d)", image_path, line_num)
                missing += 1
                continue

            entries.append(VroidLiteEntry(metadata=row, image_path=image_path))

    # Sort by filename for deterministic ordering.
    entries.sort(key=lambda e: e.image_path.name)

    if malformed:
        logger.warning("Skipped %d malformed JSONL lines", malformed)
    if missing:
        logger.warning("Skipped %d entries with missing images", missing)

    logger.info("Discovered %d entries in %s", len(entries), jsonl_path)
    return entries


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _resize_to_strata(
    img: Image.Image,
    resolution: int = STRATA_RESOLUTION,
) -> Image.Image:
    """Resize an image to *resolution*×*resolution*, preserving aspect ratio.

    The longest edge is scaled to *resolution*, then the image is
    centered on a transparent RGBA canvas.

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


def _make_example_id(entry: VroidLiteEntry) -> str:
    """Build a unique Strata example ID from a metadata entry.

    Format: ``vroid_lite_{vrm_name}_{uuid_first_segment}``

    The UUID first segment (8 hex chars before the first hyphen) is
    sufficient for uniqueness across all 4,651 images.
    """
    vrm_name = entry.metadata.get("vrm_name", "unknown")
    stem = entry.image_path.stem  # full UUID without extension
    uuid_first = stem.split("-")[0]
    return f"{VROID_LITE_SOURCE}_{vrm_name}_{uuid_first}"


def _build_metadata(
    example_id: str,
    entry: VroidLiteEntry,
    resolution: int,
    *,
    original_size: tuple[int, int],
) -> dict[str, Any]:
    """Build Strata metadata dict for a single VRoid-Lite image.

    Standard Strata fields live at the top level.  All VRoid-specific
    metadata is preserved under the ``"vroid_lite_metadata"`` key.
    The ``"character"`` field is set to ``vrm_name`` to enable
    per-character dataset splits.

    Args:
        example_id: Strata-format example identifier.
        entry: Parsed metadata + image path.
        resolution: Output resolution.
        original_size: ``(width, height)`` of the source image.

    Returns:
        Metadata dict ready for JSON serialization.
    """
    ow, oh = original_size
    row = entry.metadata

    return {
        "id": example_id,
        "source": VROID_LITE_SOURCE,
        "source_filename": entry.image_path.name,
        "resolution": resolution,
        "original_width": ow,
        "original_height": oh,
        "padding_applied": ow != oh,
        "character": row.get("vrm_name", "unknown"),
        "has_segmentation_mask": False,
        "has_fg_mask": False,
        "has_joints": False,
        "has_draw_order": False,
        "missing_annotations": _MISSING_ANNOTATIONS,
        "vroid_lite_metadata": {k: v for k, v in row.items() if k != "file_name"},
    }


def _save_example(
    output_dir: Path,
    example_id: str,
    image: Image.Image,
    metadata: dict[str, Any],
    *,
    only_new: bool = False,
) -> bool:
    """Save a single training example in Strata directory format.

    Output layout::

        output_dir/{example_id}/
        ├── image.png
        └── metadata.json

    Args:
        output_dir: Root output directory.
        example_id: Example identifier (becomes subdirectory name).
        image: Resized RGBA image.
        metadata: Metadata dict.
        only_new: Skip if output directory already exists.

    Returns:
        True if saved, False if skipped.
    """
    example_dir = output_dir / example_id

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


def convert_entry(
    entry: VroidLiteEntry,
    output_dir: Path,
    *,
    resolution: int = STRATA_RESOLUTION,
    only_new: bool = False,
) -> bool:
    """Convert a single VRoid-Lite entry to Strata training format.

    Args:
        entry: Parsed metadata + image path.
        output_dir: Root output directory.
        resolution: Target square resolution.
        only_new: Skip if output already exists.

    Returns:
        True if saved, False if skipped or errored.
    """
    example_id = _make_example_id(entry)

    try:
        img = Image.open(entry.image_path)
        img.load()
    except OSError as exc:
        logger.warning("Failed to load image %s: %s", entry.image_path, exc)
        return False

    original_size = img.size
    resized = _resize_to_strata(img, resolution)

    metadata = _build_metadata(
        example_id,
        entry,
        resolution,
        original_size=original_size,
    )

    return _save_example(
        output_dir,
        example_id,
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
    """Convert VRoid-Lite images to Strata format.

    Reads ``metadata.jsonl`` to discover entries, optionally samples a
    subset, and converts each into a Strata per-example directory.

    Args:
        input_dir: Root dataset directory containing ``metadata.jsonl``
            and the ``vroid_dataset/`` image subdirectory.
        output_dir: Root output directory for Strata-formatted examples.
        resolution: Target image resolution (square).
        only_new: Skip existing output directories.
        max_images: Maximum images to process (0 = unlimited).
        random_sample: Randomly sample from discovered entries
            (requires *max_images* > 0 to set sample size).
        seed: Random seed for reproducible sampling.

    Returns:
        :class:`AdapterResult` summarizing the conversion.
    """
    result = AdapterResult()

    entries = discover_entries(input_dir)
    if not entries:
        return result

    # Apply sampling / limiting.
    if random_sample and max_images > 0:
        rng = random.Random(seed)
        sample_size = min(max_images, len(entries))
        entries = rng.sample(entries, sample_size)
    elif max_images > 0:
        entries = entries[:max_images]

    total = len(entries)
    logger.info("Processing %d entries from %s", total, input_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, entry in enumerate(entries):
        try:
            saved = convert_entry(
                entry,
                output_dir,
                resolution=resolution,
                only_new=only_new,
            )
        except Exception as exc:
            logger.warning("Error processing %s: %s", entry.image_path.name, exc)
            result.errors.append(f"{entry.image_path.name}: {exc}")
            continue

        if saved:
            result.images_processed += 1
        else:
            result.images_skipped += 1

        # Progress logging every 100 images.
        if (i + 1) % 100 == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            logger.info("Processed %d/%d entries (%.1f%%)", i + 1, total, pct)

    logger.info(
        "VRoid-Lite conversion complete: %d processed, %d skipped, %d errors",
        result.images_processed,
        result.images_skipped,
        len(result.errors),
    )

    return result
