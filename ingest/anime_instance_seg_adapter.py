"""Convert AnimeInstanceSegmentationDataset to Strata training format.

Dataset: dreMaz/AnimeInstanceSegmentationDataset (CartoonSegmentation, arXiv 2312.01943)
Source:  github.com/CartoonSegmentation/CartoonSegmentation

The dataset provides:
- 91,082 train + 7,496 val RGB JPEG images at 720×720
- COCO-format annotations with RLE-encoded instance segmentation masks
- One category: ``object`` (anime character instances)
- ``tag_string_character`` field with character name per instance

Since this is **character-instance** segmentation (not body-part), the adapter
produces a binary **foreground mask** — all character instances merged into a
single fg/bg label — matching the same format as the ``animeseg`` adapter.

This adapter:
1. Loads ``det_train.json`` / ``det_val.json`` COCO annotation files.
2. Groups annotations by image.
3. For each image, decodes all RLE masks and merges them into a binary
   foreground mask (255 = any character pixel, 0 = background).
4. Resizes image and mask to 512×512.
5. Saves ``image.png``, ``segmentation.png``, and ``metadata.json``.

Pure Python — no Blender dependency.
"""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANIME_INSTANCE_SOURCE = "anime_instance_seg"
STRATA_RESOLUTION = 512

_MISSING_ANNOTATIONS = [
    "strata_segmentation",
    "joints",
    "draw_order",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ConversionResult:
    """Result of converting one image."""

    example_id: str
    source_path: Path
    output_dir: Path
    character_names: list[str] = field(default_factory=list)
    instance_count: int = 0
    skipped: bool = False
    skip_reason: str = ""

    @property
    def success(self) -> bool:
        return not self.skipped


@dataclass
class ConversionStats:
    """Aggregate stats for a conversion run."""

    total: int = 0
    converted: int = 0
    skipped: int = 0
    errors: int = 0

    def summary(self) -> str:
        return (
            f"{self.converted}/{self.total} converted, "
            f"{self.skipped} skipped, {self.errors} errors"
        )


# ---------------------------------------------------------------------------
# COCO RLE decoding
# ---------------------------------------------------------------------------


def _decode_rle(rle: dict) -> np.ndarray:
    """Decode a COCO compressed RLE mask to a binary numpy array.

    Args:
        rle: Dict with ``size`` (H, W) and ``counts`` (compressed RLE string).

    Returns:
        Boolean mask array of shape (H, W).
    """
    h, w = rle["size"]
    counts_str: str = rle["counts"]

    # COCO uses LEB128-like compressed RLE. Decode the string to run lengths.
    # Each character encodes 6 bits (offset 48). Values ≥ 32 continue to next char.
    counts: list[int] = []
    m = 0
    p = 0
    more = True
    i = 0
    while i < len(counts_str):
        x = ord(counts_str[i]) - 48
        i += 1
        more = bool(x & 32)
        x &= 31
        m |= x << p
        p += 5
        if not more:
            if m & 1:
                m = ~m
            m >>= 1
            if counts:
                m += counts[-1]
            counts.append(m)
            m = 0
            p = 0

    # Build the mask: alternating runs of 0 and 1, column-major order
    mask = np.zeros(h * w, dtype=np.uint8)
    idx = 0
    val = 0
    for run in counts:
        if val == 1:
            mask[idx : idx + run] = 1
        idx += run
        val = 1 - val

    # COCO masks are stored in column-major (Fortran) order
    return mask.reshape((h, w), order="F")


def _masks_to_fg(annotations: list[dict], height: int, width: int) -> np.ndarray:
    """Merge all instance RLE masks into a single binary foreground mask.

    Args:
        annotations: List of COCO annotation dicts for one image.
        height: Image height.
        width: Image width.

    Returns:
        uint8 array (H, W) — 255 for any character pixel, 0 for background.
    """
    merged = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        seg = ann.get("segmentation")
        if not seg or not isinstance(seg, dict):
            continue
        try:
            mask = _decode_rle(seg)
            merged = np.maximum(merged, (mask * 255).astype(np.uint8))
        except Exception:
            logger.debug("Failed to decode RLE for annotation %s", ann.get("id"))
    return merged


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------


def _resize_to_512(img: Image.Image, resample: int = Image.NEAREST) -> Image.Image:
    """Resize image to 512×512."""
    if img.size == (STRATA_RESOLUTION, STRATA_RESOLUTION):
        return img
    return img.resize((STRATA_RESOLUTION, STRATA_RESOLUTION), resample=resample)


def _save_example(
    image: Image.Image,
    fg_mask: np.ndarray,
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    """Save image, segmentation mask, and metadata to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Image — resize with bilinear for quality
    img_512 = _resize_to_512(image, resample=Image.BILINEAR)
    img_512.save(output_dir / "image.png", format="PNG")

    # Segmentation mask — resize with nearest neighbor to preserve binary values
    mask_img = Image.fromarray(fg_mask, mode="L")
    mask_512 = _resize_to_512(mask_img, resample=Image.NEAREST)
    mask_512.save(output_dir / "segmentation.png", format="PNG")

    # Metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------


def convert_split(
    dataset_dir: Path,
    output_dir: Path,
    split: str = "train",
    *,
    max_examples: int | None = None,
) -> ConversionStats:
    """Convert one split (train or val) of the dataset.

    Args:
        dataset_dir: Path to ``anime_instance_dataset/`` (contains ``train/``,
            ``val/``, and ``annotations/``).
        output_dir: Root output directory. Examples go in
            ``output_dir/{split}/{example_id}/``.
        split: ``"train"`` or ``"val"``.
        max_examples: If set, stop after this many images.

    Returns:
        ConversionStats with counts.
    """
    ann_file = dataset_dir / "annotations" / f"det_{split}.json"
    if not ann_file.is_file():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")

    image_dir = dataset_dir / split
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    logger.info("Loading annotations from %s", ann_file)
    with open(ann_file) as f:
        coco = json.load(f)

    # Index annotations by image_id
    ann_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    stats = ConversionStats(total=len(coco["images"]))
    split_out = output_dir / split

    for i, img_info in enumerate(coco["images"]):
        if max_examples is not None and i >= max_examples:
            break

        image_id = img_info["id"]
        file_name = img_info["file_name"]
        img_path = image_dir / file_name

        if not img_path.is_file():
            logger.warning("Image not found: %s", img_path)
            stats.skipped += 1
            continue

        annotations = ann_by_image.get(image_id, [])
        if not annotations:
            stats.skipped += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            h, w = image.size[1], image.size[0]

            fg_mask = _masks_to_fg(annotations, h, w)

            # Skip if mask is empty (no character pixels decoded)
            if fg_mask.max() == 0:
                stats.skipped += 1
                continue

            # Collect character names (deduplicated, ignore empty)
            char_names = sorted({
                ann["tag_string_character"]
                for ann in annotations
                if ann.get("tag_string_character")
            })

            example_id = f"{split}_{image_id:012d}"
            metadata = {
                "example_id": example_id,
                "source": ANIME_INSTANCE_SOURCE,
                "split": split,
                "original_file": file_name,
                "image_size": [w, h],
                "instance_count": len(annotations),
                "character_names": char_names,
                "missing_annotations": _MISSING_ANNOTATIONS,
                "license": "research",
            }

            _save_example(image, fg_mask, split_out / example_id, metadata)
            stats.converted += 1

            if stats.converted % 1000 == 0:
                logger.info(
                    "[%s] %d/%d converted (%d skipped)",
                    split,
                    stats.converted,
                    stats.total,
                    stats.skipped,
                )

        except Exception:
            logger.exception("Error processing image %s", file_name)
            stats.errors += 1

    logger.info("[%s] Done: %s", split, stats.summary())
    return stats


def convert_dataset(
    dataset_dir: Path,
    output_dir: Path,
    *,
    splits: list[str] | None = None,
    max_examples: int | None = None,
) -> dict[str, ConversionStats]:
    """Convert all splits of the AnimeInstanceSegmentation dataset.

    Args:
        dataset_dir: Path to ``anime_instance_dataset/`` directory.
        output_dir: Root output directory for Strata-format examples.
        splits: Which splits to convert. Defaults to ``["train", "val"]``.
        max_examples: Optional per-split cap for testing.

    Returns:
        Dict mapping split name to ConversionStats.
    """
    if splits is None:
        splits = ["train", "val"]

    results: dict[str, ConversionStats] = {}
    for split in splits:
        logger.info("=== Converting split: %s ===", split)
        results[split] = convert_split(
            dataset_dir,
            output_dir,
            split=split,
            max_examples=max_examples,
        )

    total_converted = sum(r.converted for r in results.values())
    total_skipped = sum(r.skipped for r in results.values())
    logger.info(
        "All splits done: %d converted, %d skipped",
        total_converted,
        total_skipped,
    )
    return results
