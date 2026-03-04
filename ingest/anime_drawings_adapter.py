"""Convert AnimeDrawingsDataset to Strata training format.

Dataset: dragonmeteor/AnimeDrawingsDataset
Source:  https://github.com/dragonmeteor/AnimeDrawingsDataset
License: Not specified — verify before commercial use

The AnimeDrawingsDataset contains 2,000 hand-drawn anime/manga images
with 22-point skeleton annotations (1,400 train / 100 val / 500 test).
Annotations are stored as JSON files with per-image joint coordinates
in pixel space::

    data/
    ├── data.json          # all 2,000 entries
    ├── train.json         # 1,400 training entries
    ├── val.json           # 100 validation entries
    ├── test.json          # 500 test entries
    └── images/            # image files (downloaded via rake build)

Each JSON entry has:
- ``file_name``: relative path like ``"data/images/1850571.png"``
- ``width``, ``height``: original image dimensions
- ``points``: dict of joint name → [x, y] pixel coordinates

This adapter maps 15 of the 22 dataset joints to Strata's 19-bone
standard skeleton.  The 4 unmapped Strata joints (spine, hips,
shoulder_l, shoulder_r) are marked ``visible: false``.  Dataset joints
without Strata equivalents (nose_tip, nose_root, thumb_left,
thumb_right, tiptoe_left, tiptoe_right) are discarded.

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

SOURCE_NAME = "anime_drawings"

STRATA_RESOLUTION = 512

# Annotations that AnimeDrawingsDataset does NOT provide.
_MISSING_ANNOTATIONS = [
    "strata_segmentation",
    "draw_order",
    "fg_mask",
]

# AnimeDrawingsDataset joint name → Strata region ID.
# Only joints with a meaningful Strata equivalent are included.
_JOINT_TO_STRATA_ID: dict[str, int] = {
    "head": 1,
    "neck": 2,
    "body_upper": 3,  # chest
    "arm_left": 7,  # upper_arm_l (shoulder position)
    "arm_right": 11,  # upper_arm_r (shoulder position)
    "elbow_left": 8,  # forearm_l
    "elbow_right": 12,  # forearm_r
    "wrist_left": 9,  # hand_l
    "wrist_right": 13,  # hand_r
    "leg_left": 14,  # upper_leg_l (hip position)
    "leg_right": 17,  # upper_leg_r (hip position)
    "knee_left": 15,  # lower_leg_l
    "knee_right": 18,  # lower_leg_r
    "ankle_left": 16,  # foot_l
    "ankle_right": 19,  # foot_r
}

# Strata region ID → canonical region name (regions 1–19).
_STRATA_ID_TO_NAME: dict[int, str] = {
    1: "head",
    2: "neck",
    3: "chest",
    4: "spine",
    5: "hips",
    6: "shoulder_l",
    7: "upper_arm_l",
    8: "forearm_l",
    9: "hand_l",
    10: "shoulder_r",
    11: "upper_arm_r",
    12: "forearm_r",
    13: "hand_r",
    14: "upper_leg_l",
    15: "lower_leg_l",
    16: "foot_l",
    17: "upper_leg_r",
    18: "lower_leg_r",
    19: "foot_r",
}

NUM_STRATA_JOINTS = 19


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AdapterResult:
    """Result of converting AnimeDrawingsDataset images to Strata format."""

    images_processed: int = 0
    images_skipped: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def load_annotations(json_path: Path) -> list[dict[str, Any]]:
    """Load annotation entries from a dataset JSON file.

    Args:
        json_path: Path to ``data.json``, ``train.json``, etc.

    Returns:
        List of annotation dicts, each with ``file_name``, ``width``,
        ``height``, and ``points`` keys.
    """
    if not json_path.is_file():
        logger.warning("Annotation file not found: %s", json_path)
        return []

    with json_path.open(encoding="utf-8") as fh:
        entries = json.load(fh)

    if not isinstance(entries, list):
        logger.warning("Expected list in %s, got %s", json_path, type(entries).__name__)
        return []

    logger.info("Loaded %d annotations from %s", len(entries), json_path)
    return entries


def _resolve_image_path(dataset_root: Path, entry: dict[str, Any]) -> Path | None:
    """Resolve the image file path from an annotation entry.

    The ``file_name`` field contains a relative path like
    ``"data/images/1850571.png"``.  We try resolving relative to
    *dataset_root* first, then relative to *dataset_root*'s parent.

    Args:
        dataset_root: Root directory of the dataset (the ``data/`` dir
            or the repo root containing ``data/``).
        entry: Single annotation dict with ``file_name`` key.

    Returns:
        Resolved :class:`Path` if found, ``None`` otherwise.
    """
    file_name = entry.get("file_name", "")
    if not file_name:
        return None

    # Try relative to dataset_root directly.
    candidate = dataset_root / file_name
    if candidate.is_file():
        return candidate

    # file_name often starts with "data/" — try stripping that prefix
    # if dataset_root already points to the data/ directory.
    stripped = file_name.removeprefix("data/")
    candidate = dataset_root / stripped
    if candidate.is_file():
        return candidate

    # Try parent directory.
    candidate = dataset_root.parent / file_name
    if candidate.is_file():
        return candidate

    return None


# ---------------------------------------------------------------------------
# Resize + joint coordinate transform
# ---------------------------------------------------------------------------


def _resize_to_strata(
    img: Image.Image,
    resolution: int = STRATA_RESOLUTION,
) -> tuple[Image.Image, float, int, int]:
    """Resize an image to *resolution*×*resolution*, preserving aspect ratio.

    The longest edge is scaled to *resolution*, then the image is
    centered on a transparent RGBA canvas.

    Args:
        img: Input image (any mode).
        resolution: Target square resolution.

    Returns:
        Tuple of (resized RGBA image, scale_factor, x_offset, y_offset).
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    w, h = img.size
    if w == resolution and h == resolution:
        return img, 1.0, 0, 0

    scale = resolution / max(w, h)
    new_w = round(w * scale)
    new_h = round(h * scale)
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))
    offset_x = (resolution - new_w) // 2
    offset_y = (resolution - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))

    return canvas, scale, offset_x, offset_y


# ---------------------------------------------------------------------------
# Joint conversion
# ---------------------------------------------------------------------------


def _build_strata_joints(
    points: dict[str, list[float]],
    scale: float,
    offset_x: int,
    offset_y: int,
    resolution: int,
) -> list[dict[str, Any]]:
    """Convert dataset joint points to Strata joints.json format.

    Applies the resize transform (scale + offset) to each joint
    coordinate, then emits all 19 Strata joints with visibility flags.

    Args:
        points: Raw joint name → [x, y] pixel coordinates from the dataset.
        scale: Scale factor applied to the image.
        offset_x: X offset from centering on the canvas.
        offset_y: Y offset from centering on the canvas.
        resolution: Output canvas size (for bounds checking).

    Returns:
        List of 19 joint dicts ordered by region ID (1–19).
    """
    # Map available joints to Strata IDs.
    mapped: dict[int, tuple[float, float]] = {}
    for joint_name, region_id in _JOINT_TO_STRATA_ID.items():
        coords = points.get(joint_name)
        if coords is None or len(coords) < 2:
            continue

        x_raw, y_raw = coords[:2]
        x = x_raw * scale + offset_x
        y = y_raw * scale + offset_y
        mapped[region_id] = (round(x, 2), round(y, 2))

    # Build the full 19-joint list.
    joints: list[dict[str, Any]] = []
    for region_id in range(1, NUM_STRATA_JOINTS + 1):
        region_name = _STRATA_ID_TO_NAME[region_id]
        if region_id in mapped:
            x, y = mapped[region_id]
            in_bounds = 0 <= x < resolution and 0 <= y < resolution
            joints.append(
                {
                    "id": region_id,
                    "name": region_name,
                    "x": x,
                    "y": y,
                    "visible": in_bounds,
                }
            )
        else:
            joints.append(
                {
                    "id": region_id,
                    "name": region_name,
                    "x": 0,
                    "y": 0,
                    "visible": False,
                }
            )

    return joints


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _build_metadata(
    example_id: str,
    source_filename: str,
    resolution: int,
    *,
    original_size: tuple[int, int],
    split: str | None = None,
    joints_mapped: int = 0,
    joints_total: int = NUM_STRATA_JOINTS,
) -> dict[str, Any]:
    """Build Strata metadata dict for a single example.

    Args:
        example_id: Strata-format example identifier.
        source_filename: Original image filename.
        resolution: Output resolution.
        original_size: ``(width, height)`` of the source image.
        split: Dataset split (``"train"``, ``"val"``, ``"test"``).
        joints_mapped: Number of joints successfully mapped.
        joints_total: Total Strata joints (19).

    Returns:
        Metadata dict ready for JSON serialisation.
    """
    ow, oh = original_size
    return {
        "id": example_id,
        "source": SOURCE_NAME,
        "source_filename": source_filename,
        "resolution": resolution,
        "original_width": ow,
        "original_height": oh,
        "padding_applied": ow != oh,
        "split": split,
        "has_segmentation_mask": False,
        "has_fg_mask": False,
        "has_joints": True,
        "has_draw_order": False,
        "missing_annotations": list(_MISSING_ANNOTATIONS),
        "joints_mapped": joints_mapped,
        "joints_total": joints_total,
        "joints_synthetic": False,
        "license": "unspecified",
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _save_example(
    output_dir: Path,
    example_id: str,
    image: Image.Image,
    joints: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    only_new: bool = False,
) -> bool:
    """Save a single training example in Strata directory format.

    Output layout::

        output_dir/{example_id}/
        ├── image.png
        ├── joints.json
        └── metadata.json

    Args:
        output_dir: Root output directory.
        example_id: Example identifier (becomes subdirectory name).
        image: Resized RGBA image.
        joints: Strata-format joint list.
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

    image.save(example_dir / "image.png", format="PNG", compress_level=6)

    (example_dir / "joints.json").write_text(
        json.dumps(joints, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    (example_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return True


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def convert_image(
    entry: dict[str, Any],
    dataset_root: Path,
    output_dir: Path,
    *,
    resolution: int = STRATA_RESOLUTION,
    only_new: bool = False,
    split: str | None = None,
) -> bool:
    """Convert a single annotated image to Strata training format.

    Args:
        entry: Annotation dict with ``file_name``, ``width``, ``height``,
            and ``points`` keys.
        dataset_root: Root dataset directory.
        output_dir: Root output directory.
        resolution: Target square resolution.
        only_new: Skip if output already exists.
        split: Dataset split label for metadata.

    Returns:
        True if saved, False if skipped or errored.
    """
    image_path = _resolve_image_path(dataset_root, entry)
    if image_path is None:
        file_name = entry.get("file_name", "<unknown>")
        logger.warning("Image not found: %s", file_name)
        return False

    # Build example ID from the image filename stem.
    example_id = f"{SOURCE_NAME}_{image_path.stem}"

    try:
        img = Image.open(image_path)
        img.load()
    except OSError as exc:
        logger.warning("Failed to load image %s: %s", image_path, exc)
        return False

    original_size = img.size
    resized, scale, offset_x, offset_y = _resize_to_strata(img, resolution)

    points = entry.get("points", {})
    joints = _build_strata_joints(points, scale, offset_x, offset_y, resolution)
    joints_mapped = sum(1 for j in joints if j["visible"] or j["x"] != 0 or j["y"] != 0)

    metadata = _build_metadata(
        example_id,
        image_path.name,
        resolution,
        original_size=original_size,
        split=split,
        joints_mapped=joints_mapped,
    )

    return _save_example(
        output_dir,
        example_id,
        resized,
        joints,
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
    """Convert AnimeDrawingsDataset to Strata format.

    Loads ``data.json`` from *input_dir* (the ``data/`` directory of
    the cloned repo), resolves image paths, and converts each example.

    Args:
        input_dir: Dataset root directory containing JSON annotation
            files and an ``images/`` subdirectory.
        output_dir: Root output directory for Strata-formatted examples.
        resolution: Target image resolution (square).
        only_new: Skip existing output directories.
        max_images: Maximum images to process (0 = unlimited).
        random_sample: Randomly sample from available entries.
        seed: Random seed for reproducible sampling.

    Returns:
        :class:`AdapterResult` summarising the conversion.
    """
    result = AdapterResult()

    # Try data.json first, then fall back to individual split files.
    data_json = input_dir / "data.json"
    if data_json.is_file():
        entries = load_annotations(data_json)
        split_label = None
    else:
        # Load individual splits and tag them.
        entries = []
        for split_name in ("train", "val", "test"):
            split_path = input_dir / f"{split_name}.json"
            split_entries = load_annotations(split_path)
            for e in split_entries:
                e["_split"] = split_name
            entries.extend(split_entries)
        split_label = None  # set per-entry below

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
    logger.info("Processing %d images from %s", total, input_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, entry in enumerate(entries):
        split = entry.pop("_split", split_label)

        saved = convert_image(
            entry,
            input_dir,
            output_dir,
            resolution=resolution,
            only_new=only_new,
            split=split,
        )

        if saved:
            result.images_processed += 1
        else:
            result.images_skipped += 1

        if (i + 1) % 100 == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            logger.info("Processed %d/%d images (%.1f%%)", i + 1, total, pct)

    logger.info(
        "AnimeDrawingsDataset conversion complete: %d processed, %d skipped, %d errors",
        result.images_processed,
        result.images_skipped,
        len(result.errors),
    )

    return result
