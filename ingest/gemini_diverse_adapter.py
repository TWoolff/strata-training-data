"""Convert Gemini-generated character images to Strata training format.

Pipeline per image:
1. Background removal via rembg (U²-Net) — skip if already RGBA with transparency
2. Resize to 512×512 with aspect-ratio-preserving padding
3. Extract binary foreground mask from alpha channel
4. Save image.png + segmentation.png + metadata.json

Input: raw PNG images from ``scripts/gemini_batch_generate.py`` with a
``manifest.json`` containing prompt + tags per image.

Post-processing (separate steps):
- RTMPose joint enrichment (``run_enrich.py``)
- Marigold depth + normals (``run_normals_enrich.py``)

Pure Python — no Blender dependency.  Requires ``rembg`` for background removal.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEMINI_DIVERSE_SOURCE = "gemini_diverse"

STRATA_RESOLUTION = 512

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

_ALPHA_THRESHOLD = 128

_MISSING_ANNOTATIONS = [
    "strata_segmentation",
    "joints",
    "depth",
    "normals",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AdapterResult:
    """Result of converting Gemini images to Strata format."""

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


def _needs_bg_removal(img: Image.Image) -> bool:
    """Check if an image needs background removal.

    Returns False if the image already has meaningful transparency.
    """
    if img.mode != "RGBA":
        return True
    alpha = np.array(img.split()[-1])
    transparent_ratio = np.count_nonzero(alpha < _ALPHA_THRESHOLD) / alpha.size
    # If >5% of pixels are transparent, assume bg is already removed
    return transparent_ratio < 0.05


def remove_background(img: Image.Image) -> Image.Image:
    """Remove background from an image using rembg."""
    from rembg import remove

    session = _get_rembg_session()
    result = remove(img, session=session)
    if result.mode != "RGBA":
        result = result.convert("RGBA")
    return result


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _crop_to_foreground(
    img: Image.Image,
    padding_ratio: float = 0.10,
) -> Image.Image:
    """Crop to foreground bounding box with padding, keeping square aspect."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    alpha = np.array(img.split()[-1])
    rows = np.any(alpha >= _ALPHA_THRESHOLD, axis=1)
    cols = np.any(alpha >= _ALPHA_THRESHOLD, axis=0)

    if not rows.any():
        return img

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Add padding
    w, h = img.size
    fg_w = x_max - x_min + 1
    fg_h = y_max - y_min + 1
    pad = int(max(fg_w, fg_h) * padding_ratio)

    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(w - 1, x_max + pad)
    y_max = min(h - 1, y_max + pad)

    # Make square by expanding the shorter side
    crop_w = x_max - x_min + 1
    crop_h = y_max - y_min + 1
    if crop_w > crop_h:
        diff = crop_w - crop_h
        y_min = max(0, y_min - diff // 2)
        y_max = min(h - 1, y_min + crop_w - 1)
        if y_max - y_min + 1 < crop_w:
            y_min = max(0, y_max - crop_w + 1)
    elif crop_h > crop_w:
        diff = crop_h - crop_w
        x_min = max(0, x_min - diff // 2)
        x_max = min(w - 1, x_min + crop_h - 1)
        if x_max - x_min + 1 < crop_h:
            x_min = max(0, x_max - crop_h + 1)

    return img.crop((x_min, y_min, x_max + 1, y_max + 1))


def _resize_to_strata(
    img: Image.Image,
    resolution: int = STRATA_RESOLUTION,
) -> Image.Image:
    """Crop to foreground, then resize to *resolution*×*resolution*."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    # Crop to character bounding box first
    img = _crop_to_foreground(img)

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
    """Extract binary fg/bg mask from alpha channel."""
    alpha = np.array(img.split()[-1])
    mask = np.where(alpha >= _ALPHA_THRESHOLD, 255, 0).astype(np.uint8)
    return Image.fromarray(mask, mode="L")


def _build_metadata(
    image_id: str,
    source_filename: str,
    resolution: int,
    *,
    original_size: tuple[int, int],
    prompt: str = "",
    tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build Strata metadata dict for a single image."""
    ow, oh = original_size
    meta: dict[str, Any] = {
        "id": image_id,
        "source": GEMINI_DIVERSE_SOURCE,
        "source_filename": source_filename,
        "resolution": resolution,
        "original_width": ow,
        "original_height": oh,
        "padding_applied": ow != oh,
        "has_segmentation_mask": False,
        "has_fg_mask": True,
        "has_joints": False,
        "has_depth": False,
        "has_normals": False,
        "missing_annotations": list(_MISSING_ANNOTATIONS),
        "license": "AI-generated (no copyright holder)",
        "generator": "Google Gemini",
    }
    if prompt:
        meta["prompt"] = prompt
    if tags:
        meta["tags"] = tags
    return meta


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_images(input_dir: Path) -> list[Path]:
    """Discover all image files under *input_dir* (non-recursive)."""
    if not input_dir.is_dir():
        logger.warning("Not a directory: %s", input_dir)
        return []

    paths = sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in _IMAGE_EXTENSIONS
        and not p.name.startswith(".")
    )

    logger.info("Discovered %d images in %s", len(paths), input_dir)
    return paths


def _load_manifest(input_dir: Path) -> dict[str, dict]:
    """Load manifest.json keyed by filename."""
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    items = json.loads(manifest_path.read_text())
    return {item["filename"]: item for item in items}


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
    manifest_entry: dict | None = None,
) -> bool:
    """Convert a single Gemini image to Strata training format."""
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

    # Only remove background if needed
    if _needs_bg_removal(img):
        img = remove_background(img)

    resized = _resize_to_strata(img, resolution)
    mask = _extract_mask(resized)

    # Check if mask has enough foreground
    fg_ratio = np.count_nonzero(np.array(mask)) / (resolution * resolution)
    if fg_ratio < 0.02:
        logger.warning(
            "Skipping %s: foreground too small (%.1f%%)",
            image_path.name,
            fg_ratio * 100,
        )
        return False

    example_dir.mkdir(parents=True, exist_ok=True)

    resized.save(example_dir / "image.png", format="PNG", compress_level=6)
    mask.save(example_dir / "segmentation.png", format="PNG", compress_level=6)

    prompt = manifest_entry.get("prompt", "") if manifest_entry else ""
    tags = {
        k: v
        for k, v in (manifest_entry or {}).items()
        if k not in ("filename", "index", "prompt")
    }

    metadata = _build_metadata(
        image_id,
        image_path.name,
        resolution,
        original_size=original_size,
        prompt=prompt,
        tags=tags or None,
    )
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
) -> AdapterResult:
    """Convert Gemini-generated images to Strata format.

    Args:
        input_dir: Directory with raw Gemini PNG images + manifest.json.
        output_dir: Output directory for Strata-formatted examples.
        resolution: Target square resolution.
        only_new: Skip existing output directories.
        max_images: Maximum images to process (0 = unlimited).
        random_sample: Randomly sample (requires *max_images* > 0).
        seed: Random seed for sampling.

    Returns:
        :class:`AdapterResult` summarizing the conversion.
    """
    import random as random_mod

    result = AdapterResult()

    image_paths = discover_images(input_dir)
    if not image_paths:
        return result

    manifest = _load_manifest(input_dir)

    if random_sample and max_images > 0:
        rng = random_mod.Random(seed)
        sample_size = min(max_images, len(image_paths))
        image_paths = rng.sample(image_paths, sample_size)
    elif max_images > 0:
        image_paths = image_paths[:max_images]

    total = len(image_paths)
    logger.info("Processing %d images from %s", total, input_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, image_path in enumerate(image_paths):
        image_id = f"{GEMINI_DIVERSE_SOURCE}_{image_path.stem}"
        manifest_entry = manifest.get(image_path.name)

        try:
            saved = convert_image(
                image_path,
                output_dir,
                image_id=image_id,
                resolution=resolution,
                only_new=only_new,
                manifest_entry=manifest_entry,
            )

            if saved:
                result.images_processed += 1
            else:
                result.images_skipped += 1

        except Exception as exc:
            result.errors.append(f"{image_path.name}: {exc}")
            logger.error("Error processing %s: %s", image_path.name, exc)

        if (i + 1) % 10 == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            logger.info("Processed %d/%d images (%.1f%%)", i + 1, total, pct)

    logger.info(
        "Gemini diverse conversion complete: %d processed, %d skipped, %d errors",
        result.images_processed,
        result.images_skipped,
        len(result.errors),
    )

    return result
