"""Convert AnimeRun LineArea data to Strata line art training format.

AnimeRun (NeurIPS 2022) provides per-frame line area data for anime frames
derived from 3D movies.  Each frame has three components:

- A grayscale anime frame (JPG)
- A line art extraction (PNG, grayscale)
- A binary line area mask (NPY, float64, 1.0=non-line region, 0.0=line pixel)

These triples are ideal for training line art detection/extraction models.

Input directory structure (AnimeRun_v2)::

    AnimeRun_v2/
    ├── train/
    │   └── LineArea/{scene}/
    │       ├── 0001.jpg          ← grayscale anime frame
    │       ├── 0001_line.png     ← extracted line art
    │       └── 0001.npy          ← binary line area mask
    └── test/
        └── ...

Output per frame::

    output_dir/{source}_{scene}_{frame}/
    ├── image.png              ← anime frame (resized to 512×512)
    ├── line_art.png           ← line art extraction (resized)
    ├── line_mask.png          ← binary line mask (white=line, black=non-line)
    └── metadata.json

This module is pure Python (no Blender dependency).

Reference: https://github.com/lisiyao21/AnimeRun
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

ANIMERUN_LINEAREA_SOURCE = "animerun_linearea"

STRATA_RESOLUTION = 512

# Top-level data type directory within each split.
_LINEAREA_DIR = "LineArea"

# Dataset split directories.
_SPLIT_DIRS = ("train", "test")

# Supported extensions.
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Annotations that this adapter does NOT provide.
_MISSING_ANNOTATIONS = [
    "strata_segmentation",
    "joints",
    "draw_order",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LineAreaFrame:
    """A matched line area frame triple."""

    frame_id: str
    scene_id: str
    split: str
    image_path: Path
    line_path: Path
    mask_path: Path


@dataclass
class AdapterResult:
    """Result of converting LineArea frames for a scene."""

    scene_id: str
    frames_saved: int = 0
    frames_skipped: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_line_mask(mask_path: Path) -> np.ndarray | None:
    """Load a line area mask from NPY or PNG file.

    The NPY mask uses 1.0 for non-line regions and 0.0 for line pixels.
    We invert this so that 255 = line pixel, 0 = non-line (more intuitive).

    Args:
        mask_path: Path to .npy or .png mask file.

    Returns:
        2D uint8 array (0 or 255), where 255 = line pixel, or None on failure.
    """
    try:
        if mask_path.suffix == ".npy":
            arr = np.load(mask_path)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            # Invert: original has 1.0=non-line, 0.0=line → we want 255=line.
            mask = ((1.0 - arr) * 255).clip(0, 255).astype(np.uint8)
            return mask
        if mask_path.suffix in (".png", ".jpg", ".jpeg"):
            img = Image.open(mask_path).convert("L")
            arr = np.array(img, dtype=np.float32)
            # Invert if values suggest 255=non-line convention.
            if arr.mean() > 127:
                arr = 255.0 - arr
            return arr.clip(0, 255).astype(np.uint8)
        logger.warning("Unsupported mask format: %s", mask_path.suffix)
        return None
    except (OSError, ValueError) as exc:
        logger.warning("Failed to load line mask %s: %s", mask_path, exc)
        return None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _find_root(animerun_dir: Path) -> Path:
    """Find the actual data root, handling nested extraction."""
    for candidate in ("AnimeRun_v2", "animerun_v2", "AnimeRun"):
        nested = animerun_dir / candidate
        if nested.is_dir() and any((nested / s).is_dir() for s in _SPLIT_DIRS):
            return nested
    return animerun_dir


def discover_scenes(animerun_dir: Path) -> list[tuple[str, str, Path]]:
    """Discover scene directories with LineArea data.

    Args:
        animerun_dir: Root AnimeRun dataset directory.

    Returns:
        List of (split, scene_id, split_dir) tuples, sorted by split and name.
    """
    root = _find_root(animerun_dir)
    scenes: list[tuple[str, str, Path]] = []

    for split_name in _SPLIT_DIRS:
        split_dir = root / split_name
        if not split_dir.is_dir():
            continue

        linearea_base = split_dir / _LINEAREA_DIR
        if not linearea_base.is_dir():
            continue

        for scene_dir in sorted(linearea_base.iterdir()):
            if not scene_dir.is_dir() or scene_dir.name.startswith("."):
                continue
            scenes.append((split_name, scene_dir.name, split_dir))

    return scenes


def discover_frames(
    scene_dir: Path,
    split: str,
    scene_id: str,
) -> list[LineAreaFrame]:
    """Discover matched line area frame triples in a scene.

    Each frame has: {id}.jpg (image), {id}_line.png (line art), {id}.npy (mask).

    Args:
        scene_dir: Path to the split directory (train/ or test/).
        split: Dataset split name.
        scene_id: Scene identifier.

    Returns:
        List of LineAreaFrame objects, sorted by frame ID.
    """
    linearea_dir = scene_dir / _LINEAREA_DIR / scene_id

    # Index files by type.
    images: dict[str, Path] = {}
    lines: dict[str, Path] = {}
    masks: dict[str, Path] = {}

    for p in sorted(linearea_dir.iterdir()):
        if not p.is_file():
            continue
        stem = p.stem

        if stem.endswith("_line"):
            # Line art file: {id}_line.png
            base_id = stem[:-5]  # Remove "_line" suffix.
            lines[base_id] = p
        elif p.suffix == ".npy":
            masks[stem] = p
        elif p.suffix.lower() in _IMAGE_EXTENSIONS:
            images[stem] = p

    # Only include frames that have all three components.
    common = sorted(set(images) & set(lines) & set(masks))

    if len(images) != len(common):
        logger.debug(
            "Scene %s: %d images, %d lines, %d masks (%d matched)",
            scene_id,
            len(images),
            len(lines),
            len(masks),
            len(common),
        )

    return [
        LineAreaFrame(
            frame_id=fid,
            scene_id=scene_id,
            split=split,
            image_path=images[fid],
            line_path=lines[fid],
            mask_path=masks[fid],
        )
        for fid in common
    ]


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def _resize_image(img: Image.Image, resolution: int = STRATA_RESOLUTION) -> Image.Image:
    """Resize an image to the target resolution, converting to grayscale."""
    if img.mode != "L":
        img = img.convert("L")
    if img.size != (resolution, resolution):
        img = img.resize((resolution, resolution), Image.LANCZOS)
    return img


def _resize_mask(
    mask: np.ndarray,
    resolution: int = STRATA_RESOLUTION,
) -> np.ndarray:
    """Resize a binary mask using nearest-neighbor to preserve sharp edges."""
    h, w = mask.shape[:2]
    if h == resolution and w == resolution:
        return mask
    img = Image.fromarray(mask, mode="L")
    img = img.resize((resolution, resolution), Image.NEAREST)
    return np.array(img)


def _build_metadata(
    frame: LineAreaFrame,
    resolution: int,
) -> dict[str, Any]:
    """Build metadata dict for a line area training example."""
    return {
        "source": ANIMERUN_LINEAREA_SOURCE,
        "scene_id": frame.scene_id,
        "frame_id": frame.frame_id,
        "split": frame.split,
        "resolution": resolution,
        "data_type": "line_area",
        "has_segmentation_mask": False,
        "has_line_mask": True,
        "has_line_art": True,
        "has_joints": False,
        "has_draw_order": False,
        "missing_annotations": _MISSING_ANNOTATIONS,
    }


def _save_frame(
    output_dir: Path,
    frame: LineAreaFrame,
    image: Image.Image,
    line_art: Image.Image,
    line_mask: np.ndarray,
    metadata: dict[str, Any],
    *,
    only_new: bool = False,
) -> bool:
    """Save a line area training example to disk.

    Output layout::

        output_dir/{source}_{scene}_{frame}/
        ├── image.png
        ├── line_art.png
        ├── line_mask.png
        └── metadata.json
    """
    example_id = f"{ANIMERUN_LINEAREA_SOURCE}_{frame.scene_id}_{frame.frame_id}"
    example_dir = output_dir / example_id

    if only_new and example_dir.exists():
        return False

    example_dir.mkdir(parents=True, exist_ok=True)

    image.save(example_dir / "image.png", format="PNG", compress_level=9)
    line_art.save(example_dir / "line_art.png", format="PNG", compress_level=9)

    mask_img = Image.fromarray(line_mask, mode="L")
    mask_img.save(example_dir / "line_mask.png", format="PNG", compress_level=9)

    meta_path = example_dir / "metadata.json"
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return True


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def convert_frame(
    frame: LineAreaFrame,
    output_dir: Path,
    *,
    resolution: int = STRATA_RESOLUTION,
    only_new: bool = False,
) -> bool:
    """Convert a single line area frame to Strata training format.

    Returns:
        True if saved, False if skipped or failed.
    """
    try:
        img_raw = Image.open(frame.image_path)
        img_raw.load()
        line_raw = Image.open(frame.line_path)
        line_raw.load()
    except OSError as exc:
        logger.warning("Failed to load frame %s/%s: %s", frame.scene_id, frame.frame_id, exc)
        return False

    mask = load_line_mask(frame.mask_path)
    if mask is None:
        return False

    image = _resize_image(img_raw, resolution)
    line_art = _resize_image(line_raw, resolution)
    line_mask = _resize_mask(mask, resolution)

    metadata = _build_metadata(frame, resolution)

    return _save_frame(
        output_dir,
        frame,
        image,
        line_art,
        line_mask,
        metadata,
        only_new=only_new,
    )


def convert_scene(
    scene_dir: Path,
    output_dir: Path,
    split: str,
    scene_id: str,
    *,
    resolution: int = STRATA_RESOLUTION,
    only_new: bool = False,
    max_frames: int = 0,
) -> AdapterResult:
    """Convert all line area frames in a scene.

    Returns:
        AdapterResult summarizing the conversion.
    """
    result = AdapterResult(scene_id=scene_id)
    frames = discover_frames(scene_dir, split, scene_id)

    if max_frames > 0:
        frames = frames[:max_frames]

    for frame in frames:
        saved = convert_frame(
            frame,
            output_dir,
            resolution=resolution,
            only_new=only_new,
        )
        if saved:
            result.frames_saved += 1
        else:
            result.frames_skipped += 1

    logger.info(
        "AnimeRun LineArea scene %s (%s): %d saved, %d skipped",
        scene_id,
        split,
        result.frames_saved,
        result.frames_skipped,
    )

    return result


def convert_directory(
    animerun_dir: Path,
    output_dir: Path,
    *,
    resolution: int = STRATA_RESOLUTION,
    only_new: bool = False,
    max_frames_per_scene: int = 0,
    max_scenes: int = 0,
) -> list[AdapterResult]:
    """Convert all AnimeRun LineArea scenes to Strata line art training format.

    Returns:
        List of AdapterResult objects for each scene.
    """
    if not animerun_dir.is_dir():
        logger.error("AnimeRun directory not found: %s", animerun_dir)
        return []

    scenes = discover_scenes(animerun_dir)

    if not scenes:
        logger.warning("No valid LineArea scenes found in %s", animerun_dir)
        return []

    if max_scenes > 0:
        scenes = scenes[:max_scenes]

    logger.info("Found %d LineArea scenes in %s", len(scenes), animerun_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[AdapterResult] = []

    for split, scene_id, scene_dir in scenes:
        scene_result = convert_scene(
            scene_dir,
            output_dir,
            split,
            scene_id,
            resolution=resolution,
            only_new=only_new,
            max_frames=max_frames_per_scene,
        )
        results.append(scene_result)

    total_saved = sum(r.frames_saved for r in results)
    logger.info(
        "AnimeRun LineArea conversion complete: %d scenes, %d frames total",
        len(results),
        total_saved,
    )

    return results
