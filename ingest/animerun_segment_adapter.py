"""Convert AnimeRun instance segmentation maps to Strata training format.

AnimeRun (NeurIPS 2022) provides per-object instance segmentation maps for
anime frames derived from 3D movies.  Each pixel in the segment map is
assigned to an object instance (not a body part — this is instance-level,
not semantic part-level segmentation).

Input directory structure (AnimeRun_v2)::

    AnimeRun_v2/
    ├── train/
    │   ├── Segment/{scene}/*.png or *.npy  ← per-object instance maps
    │   ├── Frame_Anime/{scene}/original/*.png  ← color frames
    │   └── ...
    └── test/
        └── ...

Output per frame::

    output_dir/{source}_{scene}_{frame}/
    ├── image.png              ← anime frame (resized)
    ├── instance_mask.png      ← instance segmentation (pixel value = instance ID)
    ├── instance_overlay.png   ← color overlay visualization (for QA)
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

ANIMERUN_SOURCE = "animerun"

STRATA_RESOLUTION = 512

# Top-level data type directories within each split.
_SEGMENT_DIR = "Segment"
_ANIME_DIR = "Frame_Anime"
_ANIME_VARIANT = "original"

# Dataset split directories.
_SPLIT_DIRS = ("train", "test")

# Image file extensions to consider.
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# NumPy file extension for segment maps.
_NPY_EXTENSION = ".npy"

# Annotations that this adapter does NOT provide.
_MISSING_ANNOTATIONS = [
    "strata_segmentation",
    "joints",
    "draw_order",
]

# Distinct colors for overlay visualization (up to 20 instances, wraps).
_OVERLAY_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (255, 128, 0),
    (255, 0, 128),
    (0, 255, 128),
    (128, 255, 0),
    (128, 0, 255),
    (0, 128, 255),
    (192, 64, 64),
    (64, 192, 64),
]

_OVERLAY_ALPHA = 0.5


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SegmentFrame:
    """A matched segment map / anime frame pair."""

    frame_id: str
    scene_id: str
    split: str
    segment_path: Path
    anime_path: Path


@dataclass
class AdapterResult:
    """Result of converting AnimeRun segment maps to Strata format."""

    scene_id: str
    frames_saved: int = 0
    frames_skipped: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _find_root(animerun_dir: Path) -> Path:
    """Find the actual data root, handling nested extraction.

    The zip may extract to ``AnimeRun_v2/`` inside the target dir, so we
    check for that and return the inner directory if present.
    """
    for candidate in ("AnimeRun_v2", "animerun_v2", "AnimeRun"):
        nested = animerun_dir / candidate
        if nested.is_dir() and any((nested / s).is_dir() for s in _SPLIT_DIRS):
            return nested
    return animerun_dir


def discover_scenes(animerun_dir: Path) -> list[tuple[str, str, Path]]:
    """Discover scene directories with both Segment and Frame_Anime data.

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

        segment_base = split_dir / _SEGMENT_DIR
        anime_base = split_dir / _ANIME_DIR

        if not segment_base.is_dir() or not anime_base.is_dir():
            continue

        for scene_dir in sorted(segment_base.iterdir()):
            if not scene_dir.is_dir() or scene_dir.name.startswith("."):
                continue
            scene_id = scene_dir.name
            # Anime frames live under Frame_Anime/{scene}/original/
            anime_scene = anime_base / scene_id / _ANIME_VARIANT
            if not anime_scene.is_dir():
                anime_scene = anime_base / scene_id
            if anime_scene.is_dir():
                scenes.append((split_name, scene_id, split_dir))

    return scenes


def discover_frames(split_dir: Path, split: str, scene_id: str) -> list[SegmentFrame]:
    """Discover matched segment map / anime frame pairs in a scene.

    Segment maps may be PNG or NPY files.  Only frames present in both
    the Segment and Frame_Anime directories are included.

    Args:
        split_dir: Path to the split directory (train/ or test/).
        split: Dataset split name (train/test).
        scene_id: Scene identifier.

    Returns:
        List of matched SegmentFrame objects, sorted by frame name.
    """
    segment_dir = split_dir / _SEGMENT_DIR / scene_id
    anime_dir = split_dir / _ANIME_DIR / scene_id / _ANIME_VARIANT
    if not anime_dir.is_dir():
        anime_dir = split_dir / _ANIME_DIR / scene_id

    all_extensions = _IMAGE_EXTENSIONS | {_NPY_EXTENSION}

    segment_stems: dict[str, Path] = {
        p.stem: p
        for p in sorted(segment_dir.iterdir())
        if p.suffix.lower() in all_extensions and p.is_file()
    }
    anime_stems: dict[str, Path] = {
        p.stem: p
        for p in sorted(anime_dir.iterdir())
        if p.suffix.lower() in _IMAGE_EXTENSIONS and p.is_file()
    }

    common_stems = sorted(set(segment_stems) & set(anime_stems))

    if len(segment_stems) != len(anime_stems):
        logger.debug(
            "Scene %s: %d segment maps vs %d anime frames (%d matched)",
            scene_id,
            len(segment_stems),
            len(anime_stems),
            len(common_stems),
        )

    return [
        SegmentFrame(
            frame_id=stem,
            scene_id=scene_id,
            split=split,
            segment_path=segment_stems[stem],
            anime_path=anime_stems[stem],
        )
        for stem in common_stems
    ]


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def load_segment_map(path: Path) -> np.ndarray:
    """Load a segment map from PNG or NPY file.

    Returns a 2D array of integer instance IDs.

    Args:
        path: Path to the segment map file.

    Returns:
        2D uint8 array of instance IDs.

    Raises:
        ValueError: If the file format is unsupported.
    """
    if path.suffix.lower() == _NPY_EXTENSION:
        arr = np.load(path)
    else:
        img = Image.open(path)
        img.load()
        # Convert to single-channel grayscale if needed.
        if img.mode != "L":
            img = img.convert("L")
        arr = np.array(img)

    # Ensure 2D.
    if arr.ndim == 3:
        arr = arr[:, :, 0]

    return arr.astype(np.uint8)


def _resize_image(img: Image.Image, resolution: int = STRATA_RESOLUTION) -> Image.Image:
    """Resize an image to the target resolution.

    Converts to RGB.

    Args:
        img: Input image.
        resolution: Target square resolution.

    Returns:
        Resized RGB image.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != (resolution, resolution):
        img = img.resize((resolution, resolution), Image.LANCZOS)
    return img


def _resize_mask(mask: np.ndarray, resolution: int = STRATA_RESOLUTION) -> np.ndarray:
    """Resize a segment mask using nearest-neighbor interpolation.

    Nearest-neighbor preserves exact instance ID values (no interpolation
    artifacts).

    Args:
        mask: 2D uint8 array of instance IDs.
        resolution: Target square resolution.

    Returns:
        Resized 2D uint8 array.
    """
    if mask.shape[0] == resolution and mask.shape[1] == resolution:
        return mask
    mask_img = Image.fromarray(mask, mode="L")
    mask_img = mask_img.resize((resolution, resolution), Image.NEAREST)
    return np.array(mask_img)


def generate_overlay(
    anime_img: Image.Image,
    mask: np.ndarray,
    alpha: float = _OVERLAY_ALPHA,
) -> Image.Image:
    """Generate a color overlay visualization of instance segmentation.

    Each non-zero instance ID gets a distinct color blended over the anime
    frame.

    Args:
        anime_img: RGB anime frame.
        mask: 2D uint8 array of instance IDs.
        alpha: Overlay blend factor.

    Returns:
        RGB overlay image.
    """
    base = np.array(anime_img, dtype=np.float32)
    overlay = base.copy()

    unique_ids = np.unique(mask)
    for inst_id in unique_ids:
        if inst_id == 0:
            continue  # Skip background.
        color_idx = (int(inst_id) - 1) % len(_OVERLAY_COLORS)
        color = np.array(_OVERLAY_COLORS[color_idx], dtype=np.float32)
        region = mask == inst_id
        overlay[region] = base[region] * (1.0 - alpha) + color * alpha

    return Image.fromarray(overlay.astype(np.uint8), mode="RGB")


def _build_metadata(
    frame: SegmentFrame,
    resolution: int,
    instance_ids: list[int],
) -> dict[str, Any]:
    """Build metadata dict for an instance segmentation example.

    Args:
        frame: The source segment/anime pair.
        resolution: Output resolution.
        instance_ids: List of unique instance IDs found in the mask.

    Returns:
        Metadata dict ready for JSON serialization.
    """
    return {
        "source": ANIMERUN_SOURCE,
        "scene_id": frame.scene_id,
        "frame_id": frame.frame_id,
        "split": frame.split,
        "resolution": resolution,
        "data_type": "instance_segmentation",
        "instance_ids": instance_ids,
        "num_instances": len([i for i in instance_ids if i != 0]),
        "has_segmentation_mask": False,
        "has_instance_mask": True,
        "has_contour_mask": False,
        "has_joints": False,
        "has_draw_order": False,
        "missing_annotations": _MISSING_ANNOTATIONS,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _save_example(
    output_dir: Path,
    frame: SegmentFrame,
    anime_img: Image.Image,
    mask: np.ndarray,
    overlay_img: Image.Image,
    metadata: dict[str, Any],
    *,
    only_new: bool = False,
) -> bool:
    """Save an instance segmentation training example to disk.

    Args:
        output_dir: Root output directory.
        frame: Source segment/anime pair.
        anime_img: Resized anime frame.
        mask: Resized instance mask array.
        overlay_img: Color overlay visualization.
        metadata: Metadata dict.
        only_new: Skip if output directory already exists.

    Returns:
        True if saved, False if skipped.
    """
    example_id = f"{ANIMERUN_SOURCE}_{frame.scene_id}_{frame.frame_id}"
    example_dir = output_dir / example_id

    if only_new and example_dir.exists():
        return False

    example_dir.mkdir(parents=True, exist_ok=True)

    anime_img.save(example_dir / "image.png", format="PNG", compress_level=9)

    mask_img = Image.fromarray(mask, mode="L")
    mask_img.save(example_dir / "instance_mask.png", format="PNG", compress_level=9)

    overlay_img.save(example_dir / "instance_overlay.png", format="PNG", compress_level=9)

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
    frame: SegmentFrame,
    output_dir: Path,
    *,
    resolution: int = STRATA_RESOLUTION,
    only_new: bool = False,
) -> bool:
    """Convert a single segment/anime pair to Strata instance segmentation format.

    Args:
        frame: Segment map / anime frame pair.
        output_dir: Root output directory.
        resolution: Target square resolution.
        only_new: Skip if output already exists.

    Returns:
        True if saved, False if skipped or failed.
    """
    try:
        mask_raw = load_segment_map(frame.segment_path)
    except (OSError, ValueError) as exc:
        logger.warning("Failed to load segment map %s/%s: %s", frame.scene_id, frame.frame_id, exc)
        return False

    try:
        anime_raw = Image.open(frame.anime_path)
        anime_raw.load()
    except OSError as exc:
        logger.warning("Failed to load anime frame %s/%s: %s", frame.scene_id, frame.frame_id, exc)
        return False

    anime_img = _resize_image(anime_raw, resolution)
    mask = _resize_mask(mask_raw, resolution)

    overlay_img = generate_overlay(anime_img, mask)

    instance_ids = sorted(int(x) for x in np.unique(mask))
    metadata = _build_metadata(frame, resolution, instance_ids)

    return _save_example(
        output_dir, frame, anime_img, mask, overlay_img, metadata, only_new=only_new
    )


def convert_scene(
    split_dir: Path,
    output_dir: Path,
    split: str,
    scene_id: str,
    *,
    resolution: int = STRATA_RESOLUTION,
    only_new: bool = False,
    max_frames: int = 0,
) -> AdapterResult:
    """Convert all matched frames in a scene to instance segmentation format.

    Args:
        split_dir: Path to the split directory (e.g. train/).
        output_dir: Root output directory.
        split: Dataset split name.
        scene_id: Scene identifier.
        resolution: Target square resolution.
        only_new: Skip existing output directories.
        max_frames: Maximum frames to convert per scene (0 = unlimited).

    Returns:
        AdapterResult summarizing the conversion.
    """
    result = AdapterResult(scene_id=scene_id)
    frames = discover_frames(split_dir, split, scene_id)

    if max_frames > 0:
        frames = frames[:max_frames]

    for frame in frames:
        saved = convert_frame(frame, output_dir, resolution=resolution, only_new=only_new)
        if saved:
            result.frames_saved += 1
        else:
            result.frames_skipped += 1

    logger.info(
        "AnimeRun segment scene %s (%s): %d saved, %d skipped",
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
    """Convert all AnimeRun scenes to Strata instance segmentation format.

    Args:
        animerun_dir: Root AnimeRun dataset directory.
        output_dir: Root output directory.
        resolution: Target square resolution.
        only_new: Skip existing output directories.
        max_frames_per_scene: Max frames per scene (0 = unlimited).
        max_scenes: Max scenes to process (0 = unlimited).

    Returns:
        List of AdapterResult objects for each scene.
    """
    if not animerun_dir.is_dir():
        logger.error("AnimeRun directory not found: %s", animerun_dir)
        return []

    scenes = discover_scenes(animerun_dir)

    if not scenes:
        logger.warning("No valid scenes with segment data found in %s", animerun_dir)
        return []

    if max_scenes > 0:
        scenes = scenes[:max_scenes]

    logger.info("Found %d scenes with segment data in %s", len(scenes), animerun_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[AdapterResult] = []

    for split, scene_id, split_dir in scenes:
        scene_result = convert_scene(
            split_dir,
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
        "AnimeRun segment conversion complete: %d scenes, %d frames total",
        len(results),
        total_saved,
    )

    return results
