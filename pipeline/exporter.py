"""File export: save images, masks, JSON metadata to the dataset directory.

Handles the dataset directory structure, file naming conventions, and
incremental-run support (skip existing files).  Pure Python — no Blender
dependency — so it can be imported outside Blender for testing.

Directory layout (PRD §8.1)::

    dataset/
    ├── class_map.json
    ├── images/   {char_id}_pose_{nn}_{style}.png   (RGBA)
    ├── masks/    {char_id}_pose_{nn}.png            (8-bit grayscale)
    ├── joints/   {char_id}_pose_{nn}.json
    ├── weights/  {char_id}_pose_{nn}.json
    └── sources/  {char_id}.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import REGION_NAMES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Subdirectory names
# ---------------------------------------------------------------------------

_SUBDIRS: list[str] = ["images", "masks", "joints", "weights", "sources"]


# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------


def ensure_output_dirs(output_dir: Path) -> None:
    """Create the dataset directory tree if it doesn't already exist.

    Args:
        output_dir: Root dataset directory (e.g. ``./dataset/``).
    """
    for subdir in _SUBDIRS:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    logger.info("Output directories verified under %s", output_dir)


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------


def image_filename(char_id: str, pose_index: int, style: str) -> str:
    """Build the canonical image filename.

    Args:
        char_id: Character identifier (e.g. ``"mixamo_001"``).
        pose_index: Zero-based pose number.
        style: Art style name (e.g. ``"flat"``).

    Returns:
        Filename such as ``"mixamo_001_pose_00_flat.png"``.
    """
    return f"{char_id}_pose_{pose_index:02d}_{style}.png"


def mask_filename(char_id: str, pose_index: int) -> str:
    """Build the canonical mask filename.

    Args:
        char_id: Character identifier.
        pose_index: Zero-based pose number.

    Returns:
        Filename such as ``"mixamo_001_pose_00.png"``.
    """
    return f"{char_id}_pose_{pose_index:02d}.png"


def joints_filename(char_id: str, pose_index: int) -> str:
    """Build the canonical joints filename.

    Args:
        char_id: Character identifier.
        pose_index: Zero-based pose number.

    Returns:
        Filename such as ``"mixamo_001_pose_00.json"``.
    """
    return f"{char_id}_pose_{pose_index:02d}.json"


def weights_filename(char_id: str, pose_index: int) -> str:
    """Build the canonical weights filename.

    Args:
        char_id: Character identifier.
        pose_index: Zero-based pose number.

    Returns:
        Filename such as ``"mixamo_001_pose_00.json"``.
    """
    return f"{char_id}_pose_{pose_index:02d}.json"


def source_filename(char_id: str) -> str:
    """Build the canonical per-character source metadata filename.

    Args:
        char_id: Character identifier.

    Returns:
        Filename such as ``"mixamo_001.json"``.
    """
    return f"{char_id}.json"


# ---------------------------------------------------------------------------
# Image saving
# ---------------------------------------------------------------------------


def save_image(
    img: Image.Image,
    output_dir: Path,
    char_id: str,
    pose_index: int,
    style: str,
    *,
    only_new: bool = False,
) -> Path | None:
    """Save a color render as an RGBA PNG.

    Args:
        img: The rendered image (any PIL mode — converted to RGBA).
        output_dir: Root dataset directory.
        char_id: Character identifier.
        pose_index: Zero-based pose number.
        style: Art style name.
        only_new: If True, skip saving when the file already exists.

    Returns:
        The output path, or None if skipped.
    """
    path = output_dir / "images" / image_filename(char_id, pose_index, style)

    if only_new and path.exists():
        logger.debug("Skipping existing image %s", path)
        return None

    path.parent.mkdir(parents=True, exist_ok=True)

    if img.mode != "RGBA":
        img = img.convert("RGBA")

    img.save(path, format="PNG", compress_level=9)
    logger.info("Saved image %s", path)
    return path


# ---------------------------------------------------------------------------
# Mask saving
# ---------------------------------------------------------------------------


def save_mask(
    mask: Image.Image | np.ndarray,
    output_dir: Path,
    char_id: str,
    pose_index: int,
    *,
    only_new: bool = False,
) -> Path | None:
    """Save a segmentation mask as an 8-bit grayscale PNG.

    Args:
        mask: Mask data — either a PIL Image or a numpy uint8 array where
            each pixel value equals its region ID (0–19).
        output_dir: Root dataset directory.
        char_id: Character identifier.
        pose_index: Zero-based pose number.
        only_new: If True, skip saving when the file already exists.

    Returns:
        The output path, or None if skipped.
    """
    path = output_dir / "masks" / mask_filename(char_id, pose_index)

    if only_new and path.exists():
        logger.debug("Skipping existing mask %s", path)
        return None

    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask.astype(np.uint8), mode="L")
    elif mask.mode != "L":
        mask = mask.convert("L")

    mask.save(path, format="PNG", compress_level=9)
    logger.info("Saved mask %s", path)
    return path


# ---------------------------------------------------------------------------
# Joint data saving
# ---------------------------------------------------------------------------


def save_joints(
    joint_data: dict[str, Any],
    output_dir: Path,
    char_id: str,
    pose_index: int,
    *,
    only_new: bool = False,
) -> Path | None:
    """Save joint position data as a JSON file.

    Args:
        joint_data: Joint data dict (as returned by
            ``joint_extractor.extract_joints``).
        output_dir: Root dataset directory.
        char_id: Character identifier.
        pose_index: Zero-based pose number.
        only_new: If True, skip saving when the file already exists.

    Returns:
        The output path, or None if skipped.
    """
    path = output_dir / "joints" / joints_filename(char_id, pose_index)

    if only_new and path.exists():
        logger.debug("Skipping existing joints %s", path)
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(joint_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Saved joints %s", path)
    return path


# ---------------------------------------------------------------------------
# Weight data saving
# ---------------------------------------------------------------------------


def save_weights(
    weight_data: dict[str, Any],
    output_dir: Path,
    char_id: str,
    pose_index: int,
    *,
    only_new: bool = False,
) -> Path | None:
    """Save per-vertex bone weight data as a JSON file.

    Args:
        weight_data: Weight data dict.
        output_dir: Root dataset directory.
        char_id: Character identifier.
        pose_index: Zero-based pose number.
        only_new: If True, skip saving when the file already exists.

    Returns:
        The output path, or None if skipped.
    """
    path = output_dir / "weights" / weights_filename(char_id, pose_index)

    if only_new and path.exists():
        logger.debug("Skipping existing weights %s", path)
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(weight_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Saved weights %s", path)
    return path


# ---------------------------------------------------------------------------
# Per-character source metadata
# ---------------------------------------------------------------------------


def save_source_metadata(
    output_dir: Path,
    char_id: str,
    *,
    source: str = "",
    name: str = "",
    license_: str = "",
    attribution: str = "",
    bone_mapping: str = "auto",
    bone_mapping_overrides: dict[str, int] | None = None,
    unmapped_bones: list[str] | None = None,
    character_type: str = "humanoid",
    notes: str = "",
    only_new: bool = False,
) -> Path | None:
    """Save per-character source metadata as a JSON file.

    Args:
        output_dir: Root dataset directory.
        char_id: Character identifier (e.g. ``"mixamo_001"``).
        source: Asset source name (e.g. ``"mixamo"``).
        name: Human-readable character name.
        license_: License string (parameter named with trailing underscore to
            avoid shadowing the built-in).
        attribution: Attribution string.
        bone_mapping: Mapping method used (``"auto"`` or ``"manual"``).
        bone_mapping_overrides: Manual bone mapping overrides applied.
        unmapped_bones: Bones that could not be mapped.
        character_type: Character type (``"humanoid"`` for v1).
        notes: Free-form notes.
        only_new: If True, skip saving when the file already exists.

    Returns:
        The output path, or None if skipped.
    """
    path = output_dir / "sources" / source_filename(char_id)

    if only_new and path.exists():
        logger.debug("Skipping existing source metadata %s", path)
        return None

    path.parent.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "id": char_id,
        "source": source,
        "name": name,
        "license": license_,
        "attribution": attribution,
        "bone_mapping": bone_mapping,
        "bone_mapping_overrides": bone_mapping_overrides or {},
        "unmapped_bones": unmapped_bones or [],
        "character_type": character_type,
        "notes": notes,
    }

    path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Saved source metadata %s", path)
    return path


# ---------------------------------------------------------------------------
# Class map
# ---------------------------------------------------------------------------


def save_class_map(
    output_dir: Path,
    *,
    only_new: bool = False,
) -> Path | None:
    """Save region ID → name mapping as ``class_map.json``.

    Uses the canonical ``REGION_NAMES`` from ``config.py``.  Keys are
    stringified integers (``"0"``, ``"1"``, …) per the PRD schema.

    Args:
        output_dir: Root dataset directory.
        only_new: If True, skip saving when the file already exists.

    Returns:
        The output path, or None if skipped.
    """
    path = output_dir / "class_map.json"

    if only_new and path.exists():
        logger.debug("Skipping existing class_map %s", path)
        return None

    class_map: dict[str, str] = {
        str(region_id): name for region_id, name in sorted(REGION_NAMES.items())
    }

    path.write_text(
        json.dumps(class_map, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Saved class_map.json with %d regions to %s", len(class_map), path)
    return path
