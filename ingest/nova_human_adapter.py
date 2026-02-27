"""Convert NOVA-Human renders to Strata training format.

Reads the NOVA-Human dataset directory structure (10.2K anime characters
rendered from VRoid Hub models) and converts available data into Strata's
standard output format.

NOVA-Human provides per-character:
- ``ortho/`` — front/back orthographic renders
- ``ortho_mask/`` — foreground masks for ortho views
- ``rgb/`` — 16 random-view perspective renders
- ``rgb_mask/`` — foreground masks for random views
- ``ortho_xyza/`` — position + alpha for ortho views
- ``xyza/`` — position + alpha for random views
- ``{character_id}_meta.json`` — character metadata

**Important:** NOVA-Human does NOT provide Strata-specific 19-region
segmentation, joint positions, or draw order.  This adapter converts what
is available and flags missing annotations in the output metadata.

This module is pure Python (no Blender dependency) so it can be imported
outside Blender for testing and validation.

Reference: https://github.com/NOVA-3D-Anime-Character-Synthesis/NOVA-3D
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

NOVA_HUMAN_SOURCE = "nova_human"

# NOVA-Human renders orthographic views at varying resolutions.
# We resize to the Strata standard.
STRATA_RESOLUTION = 512

# Subdirectories inside each NOVA-Human character folder.
_ORTHO_DIR = "ortho"
_ORTHO_MASK_DIR = "ortho_mask"
_RGB_DIR = "rgb"
_RGB_MASK_DIR = "rgb_mask"

# Annotations that NOVA-Human does NOT provide.
_MISSING_ANNOTATIONS = [
    "strata_segmentation",
    "joints",
    "draw_order",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class NovaHumanCharacter:
    """Parsed data for a single NOVA-Human character."""

    character_id: str
    source_dir: Path
    meta: dict[str, Any] = field(default_factory=dict)
    ortho_images: dict[str, Image.Image] = field(default_factory=dict)
    ortho_masks: dict[str, Image.Image] = field(default_factory=dict)
    rgb_images: list[Image.Image] = field(default_factory=list)
    rgb_masks: list[Image.Image] = field(default_factory=list)


@dataclass
class AdapterResult:
    """Result of converting one NOVA-Human character to Strata format."""

    char_id: str
    views_saved: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _load_meta(char_dir: Path) -> dict[str, Any]:
    """Load the character metadata JSON file.

    Args:
        char_dir: Path to the character directory.

    Returns:
        Parsed metadata dict, or empty dict if the file is missing.
    """
    # Metadata filename matches the directory name.
    meta_path = char_dir / f"{char_dir.name}_meta.json"
    if not meta_path.is_file():
        # Try a glob fallback — some characters use different naming.
        candidates = list(char_dir.glob("*_meta.json"))
        if candidates:
            meta_path = candidates[0]
        else:
            logger.debug("No metadata JSON found in %s", char_dir)
            return {}

    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read metadata %s: %s", meta_path, exc)
        return {}


def _load_images_from_dir(
    image_dir: Path,
    *,
    max_images: int = 0,
) -> list[tuple[str, Image.Image]]:
    """Load PNG/JPG images from a directory, sorted by name.

    Args:
        image_dir: Directory containing image files.
        max_images: Maximum number of images to load (0 = unlimited).

    Returns:
        List of (filename_stem, PIL Image) tuples.
    """
    if not image_dir.is_dir():
        return []

    extensions = {".png", ".jpg", ".jpeg"}
    paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in extensions and p.is_file()
    )

    if max_images > 0:
        paths = paths[:max_images]

    results: list[tuple[str, Image.Image]] = []
    for p in paths:
        try:
            img = Image.open(p)
            img.load()  # Force read into memory
            results.append((p.stem, img))
        except OSError as exc:
            logger.warning("Failed to load image %s: %s", p, exc)

    return results


def parse_character(char_dir: Path) -> NovaHumanCharacter | None:
    """Parse a single NOVA-Human character directory.

    Args:
        char_dir: Path to the character directory (e.g.
            ``data/preprocessed/nova_human/human_rutileE/``).

    Returns:
        Parsed character data, or None if the directory is invalid.
    """
    if not char_dir.is_dir():
        logger.warning("Not a directory: %s", char_dir)
        return None

    character_id = char_dir.name
    meta = _load_meta(char_dir)

    character = NovaHumanCharacter(
        character_id=character_id,
        source_dir=char_dir,
        meta=meta,
    )

    # Load orthographic views
    character.ortho_images = dict(_load_images_from_dir(char_dir / _ORTHO_DIR))
    character.ortho_masks = dict(_load_images_from_dir(char_dir / _ORTHO_MASK_DIR))

    # Load RGB (perspective) views
    character.rgb_images = [img for _, img in _load_images_from_dir(char_dir / _RGB_DIR)]
    character.rgb_masks = [img for _, img in _load_images_from_dir(char_dir / _RGB_MASK_DIR)]

    total = len(character.ortho_images) + len(character.rgb_images)
    if total == 0:
        logger.warning("No images found for character %s", character_id)
        return None

    logger.debug(
        "Parsed character %s: %d ortho, %d rgb views",
        character_id,
        len(character.ortho_images),
        len(character.rgb_images),
    )

    return character


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def _resize_to_strata(img: Image.Image, resolution: int = STRATA_RESOLUTION) -> Image.Image:
    """Resize an image to the Strata standard resolution.

    Uses LANCZOS resampling for quality.  Converts to RGBA if needed.

    Args:
        img: Input image.
        resolution: Target resolution (square).

    Returns:
        Resized RGBA image.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    if img.size != (resolution, resolution):
        img = img.resize((resolution, resolution), Image.LANCZOS)

    return img


def _convert_mask_to_binary(
    mask: Image.Image,
    resolution: int = STRATA_RESOLUTION,
) -> np.ndarray:
    """Convert a NOVA-Human foreground mask to a binary mask array.

    NOVA-Human masks are foreground/background (not 19-region segmentation).
    We store them as-is: pixel value 1 = foreground, 0 = background.

    Args:
        mask: Input mask image.
        resolution: Target resolution (square).

    Returns:
        2D uint8 numpy array (0 or 1).
    """
    if mask.mode != "L":
        mask = mask.convert("L")

    if mask.size != (resolution, resolution):
        mask = mask.resize(
            (resolution, resolution),
            Image.NEAREST,
        )

    arr = np.array(mask, dtype=np.uint8)
    # Threshold: anything above 127 is foreground
    return (arr > 127).astype(np.uint8)


def _build_metadata(
    char_id: str,
    character: NovaHumanCharacter,
    view_name: str,
    view_index: int,
    *,
    is_ortho: bool = True,
) -> dict[str, Any]:
    """Build Strata metadata dict for a single view.

    Args:
        char_id: Strata character ID.
        character: Parsed NOVA-Human character.
        view_name: Name of the view (e.g. "front", "back", "rgb_00").
        view_index: Zero-based view index.
        is_ortho: Whether this is an orthographic view.

    Returns:
        Metadata dict ready for JSON serialization.
    """
    return {
        "id": char_id,
        "source": NOVA_HUMAN_SOURCE,
        "source_character_id": character.character_id,
        "view_name": view_name,
        "view_index": view_index,
        "camera_type": "orthographic" if is_ortho else "perspective",
        "resolution": STRATA_RESOLUTION,
        "has_segmentation_mask": False,
        "has_fg_mask": True,
        "has_joints": False,
        "has_draw_order": False,
        "missing_annotations": _MISSING_ANNOTATIONS,
        "nova_human_meta": character.meta,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _save_example(
    output_dir: Path,
    char_id: str,
    view_name: str,
    image: Image.Image,
    mask: np.ndarray | None,
    metadata: dict[str, Any],
    *,
    only_new: bool = False,
) -> bool:
    """Save a single training example in Strata directory format.

    Output layout::

        output_dir/{char_id}_{view_name}/
        ├── image.png
        ├── segmentation.png   (fg/bg mask if available)
        └── metadata.json

    Args:
        output_dir: Root output directory.
        char_id: Strata character ID.
        view_name: View name for subdirectory.
        image: Resized RGBA image.
        mask: Binary mask array, or None if no mask available.
        metadata: Metadata dict.
        only_new: Skip if output directory already exists.

    Returns:
        True if saved, False if skipped or errored.
    """
    example_dir = output_dir / f"{char_id}_{view_name}"

    if only_new and example_dir.exists():
        logger.debug("Skipping existing example %s", example_dir)
        return False

    example_dir.mkdir(parents=True, exist_ok=True)

    # Save image
    image_path = example_dir / "image.png"
    image.save(image_path, format="PNG", compress_level=9)

    # Save mask (fg/bg binary — not Strata 19-region segmentation)
    if mask is not None:
        mask_img = Image.fromarray(mask * 255, mode="L")
        mask_path = example_dir / "segmentation.png"
        mask_img.save(mask_path, format="PNG", compress_level=9)

    # Save metadata
    meta_path = example_dir / "metadata.json"
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return True


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def convert_character(
    char_dir: Path,
    output_dir: Path,
    *,
    resolution: int = STRATA_RESOLUTION,
    include_rgb: bool = False,
    only_new: bool = False,
) -> AdapterResult | None:
    """Convert a single NOVA-Human character to Strata training format.

    By default only orthographic views (front/back) are converted, since
    these are the primary multi-view training data.  Set ``include_rgb=True``
    to also convert the 16 random-perspective views.

    Args:
        char_dir: Path to the NOVA-Human character directory.
        output_dir: Root output directory for Strata-formatted examples.
        resolution: Target image resolution (square).
        include_rgb: Whether to include random-perspective RGB views.
        only_new: Skip existing output directories.

    Returns:
        AdapterResult summarizing the conversion, or None if parsing failed.
    """
    character = parse_character(char_dir)
    if character is None:
        return None

    char_id = f"{NOVA_HUMAN_SOURCE}_{character.character_id}"
    result = AdapterResult(char_id=char_id)
    view_index = 0

    # Convert orthographic views
    for stem, img in sorted(character.ortho_images.items()):
        resized = _resize_to_strata(img, resolution)

        # Find matching mask
        mask_arr: np.ndarray | None = None
        if stem in character.ortho_masks:
            mask_arr = _convert_mask_to_binary(
                character.ortho_masks[stem], resolution
            )

        view_name = f"ortho_{stem}"
        metadata = _build_metadata(
            char_id, character, view_name, view_index, is_ortho=True
        )

        saved = _save_example(
            output_dir, char_id, view_name, resized, mask_arr, metadata,
            only_new=only_new,
        )
        if saved:
            result.views_saved += 1
        view_index += 1

    # Convert RGB (perspective) views
    if include_rgb:
        for i, img in enumerate(character.rgb_images):
            resized = _resize_to_strata(img, resolution)

            mask_arr = None
            if i < len(character.rgb_masks):
                mask_arr = _convert_mask_to_binary(
                    character.rgb_masks[i], resolution
                )

            view_name = f"rgb_{i:02d}"
            metadata = _build_metadata(
                char_id, character, view_name, view_index, is_ortho=False
            )

            saved = _save_example(
                output_dir, char_id, view_name, resized, mask_arr, metadata,
                only_new=only_new,
            )
            if saved:
                result.views_saved += 1
            view_index += 1

    logger.info(
        "Converted character %s: %d views saved",
        char_id,
        result.views_saved,
    )

    return result


def convert_directory(
    nova_human_dir: Path,
    output_dir: Path,
    *,
    resolution: int = STRATA_RESOLUTION,
    include_rgb: bool = False,
    only_new: bool = False,
    max_characters: int = 0,
) -> list[AdapterResult]:
    """Convert all NOVA-Human characters in a directory to Strata format.

    Discovers character subdirectories, processes each one, and saves
    outputs to the output directory.

    Args:
        nova_human_dir: Root NOVA-Human dataset directory containing
            per-character subdirectories.
        output_dir: Root output directory for Strata-formatted examples.
        resolution: Target image resolution (square).
        include_rgb: Whether to include random-perspective RGB views.
        only_new: Skip existing output directories.
        max_characters: Maximum number of characters to process
            (0 = unlimited).

    Returns:
        List of AdapterResult objects for successfully converted characters.
    """
    if not nova_human_dir.is_dir():
        logger.error("NOVA-Human directory not found: %s", nova_human_dir)
        return []

    # Discover character directories — each subdirectory is a character
    # if it contains an ortho/ or rgb/ subfolder.
    char_dirs = sorted(
        d for d in nova_human_dir.iterdir()
        if d.is_dir()
        and not d.name.startswith(".")
        and ((d / _ORTHO_DIR).is_dir() or (d / _RGB_DIR).is_dir())
    )

    if not char_dirs:
        logger.warning("No character directories found in %s", nova_human_dir)
        return []

    if max_characters > 0:
        char_dirs = char_dirs[:max_characters]

    logger.info(
        "Found %d characters in %s",
        len(char_dirs),
        nova_human_dir,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[AdapterResult] = []

    for char_dir in char_dirs:
        adapter_result = convert_character(
            char_dir,
            output_dir,
            resolution=resolution,
            include_rgb=include_rgb,
            only_new=only_new,
        )
        if adapter_result is not None:
            results.append(adapter_result)

    total_views = sum(r.views_saved for r in results)
    logger.info(
        "NOVA-Human conversion complete: %d/%d characters, %d views total",
        len(results),
        len(char_dirs),
        total_views,
    )

    return results
