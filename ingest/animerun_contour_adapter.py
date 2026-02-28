"""Convert AnimeRun contour/color pairs to Strata contour training format.

AnimeRun (NeurIPS 2022) provides ~8K paired frames from anime-style 3D
rendering — contour maps alongside matching color frames.  These pairs are
ideal for training contour detection and removal models.

Input directory structure::

    animerun/
    ├── train/
    │   ├── scene_001/
    │   │   ├── contour/       ← contour-only renders
    │   │   ├── anime/         ← color frames (matching contour frames)
    │   │   └── ...            ← flow_fwd/, flow_bwd/, etc. (ignored)
    │   └── scene_002/
    └── test/
        └── ...

Output per frame triple::

    output_dir/{source}_{scene}_{frame}/
    ├── with_contours.png       ← color frame with contours visible
    ├── without_contours.png    ← color frame (anime render)
    ├── contour_mask.png        ← binary contour mask (white=contour)
    └── metadata.json

This module is pure Python (no Blender dependency) so it can be imported
outside Blender for testing and validation.

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

# Contour pixel difference threshold (0-255).  Pixels where the absolute
# difference between contour and anime frames exceeds this are contour pixels.
CONTOUR_DIFF_THRESHOLD = 30

# Subdirectories inside each AnimeRun scene folder.
_CONTOUR_DIR = "contour"
_ANIME_DIR = "anime"

# Dataset split directories.
_SPLIT_DIRS = ("train", "test")

# Annotations that AnimeRun does NOT provide for Strata's standard format.
_MISSING_ANNOTATIONS = [
    "strata_segmentation",
    "joints",
    "draw_order",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ContourPair:
    """A matched contour/anime frame pair."""

    frame_id: str
    scene_id: str
    split: str
    contour_path: Path
    anime_path: Path


@dataclass
class AdapterResult:
    """Result of converting AnimeRun frames to Strata contour format."""

    scene_id: str
    frames_saved: int = 0
    frames_skipped: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_scenes(animerun_dir: Path) -> list[tuple[str, str, Path]]:
    """Discover scene directories within the AnimeRun dataset.

    Args:
        animerun_dir: Root AnimeRun dataset directory.

    Returns:
        List of (split, scene_id, scene_dir) tuples, sorted by split and name.
    """
    scenes: list[tuple[str, str, Path]] = []

    for split_name in _SPLIT_DIRS:
        split_dir = animerun_dir / split_name
        if not split_dir.is_dir():
            continue

        for scene_dir in sorted(split_dir.iterdir()):
            if not scene_dir.is_dir() or scene_dir.name.startswith("."):
                continue
            # A valid scene must have both contour/ and anime/ subdirectories.
            if (scene_dir / _CONTOUR_DIR).is_dir() and (scene_dir / _ANIME_DIR).is_dir():
                scenes.append((split_name, scene_dir.name, scene_dir))

    return scenes


def discover_pairs(scene_dir: Path, split: str, scene_id: str) -> list[ContourPair]:
    """Discover matched contour/anime frame pairs in a scene directory.

    Only frames present in both contour/ and anime/ are included.

    Args:
        scene_dir: Path to the scene directory.
        split: Dataset split name (train/test).
        scene_id: Scene identifier.

    Returns:
        List of matched ContourPair objects, sorted by frame name.
    """
    contour_dir = scene_dir / _CONTOUR_DIR
    anime_dir = scene_dir / _ANIME_DIR

    extensions = {".png", ".jpg", ".jpeg"}

    contour_stems = {
        p.stem: p
        for p in sorted(contour_dir.iterdir())
        if p.suffix.lower() in extensions and p.is_file()
    }
    anime_stems = {
        p.stem: p
        for p in sorted(anime_dir.iterdir())
        if p.suffix.lower() in extensions and p.is_file()
    }

    # Only include frames present in both directories.
    common_stems = sorted(set(contour_stems) & set(anime_stems))

    if len(contour_stems) != len(anime_stems):
        logger.debug(
            "Scene %s: %d contour vs %d anime frames (%d matched)",
            scene_id,
            len(contour_stems),
            len(anime_stems),
            len(common_stems),
        )

    return [
        ContourPair(
            frame_id=stem,
            scene_id=scene_id,
            split=split,
            contour_path=contour_stems[stem],
            anime_path=anime_stems[stem],
        )
        for stem in common_stems
    ]


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def _resize_image(img: Image.Image, resolution: int = STRATA_RESOLUTION) -> Image.Image:
    """Resize an image to the target resolution.

    Converts to RGB (contour training doesn't need alpha).

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


def generate_contour_mask(
    contour_img: Image.Image,
    anime_img: Image.Image,
    threshold: int = CONTOUR_DIFF_THRESHOLD,
) -> np.ndarray:
    """Generate a binary contour mask from the difference between frames.

    Contour pixels are where the absolute difference between contour and
    anime frames exceeds the threshold across any channel.

    Args:
        contour_img: Contour frame (RGB).
        anime_img: Anime/color frame (RGB).
        threshold: Pixel difference threshold.

    Returns:
        2D uint8 array (0 or 255), where 255 = contour pixel.
    """
    contour_arr = np.array(contour_img, dtype=np.int16)
    anime_arr = np.array(anime_img, dtype=np.int16)

    diff = np.abs(contour_arr - anime_arr)
    # Contour pixel if any channel exceeds threshold.
    mask = np.max(diff, axis=2) > threshold
    return (mask * 255).astype(np.uint8)


def _build_metadata(
    pair: ContourPair,
    resolution: int,
) -> dict[str, Any]:
    """Build metadata dict for a contour training example.

    Args:
        pair: The source contour/anime pair.
        resolution: Output resolution.

    Returns:
        Metadata dict ready for JSON serialization.
    """
    return {
        "source": ANIMERUN_SOURCE,
        "scene_id": pair.scene_id,
        "frame_id": pair.frame_id,
        "split": pair.split,
        "resolution": resolution,
        "data_type": "contour_pair",
        "has_segmentation_mask": False,
        "has_contour_mask": True,
        "has_joints": False,
        "has_draw_order": False,
        "missing_annotations": _MISSING_ANNOTATIONS,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _save_contour_triple(
    output_dir: Path,
    pair: ContourPair,
    contour_img: Image.Image,
    anime_img: Image.Image,
    contour_mask: np.ndarray,
    metadata: dict[str, Any],
    *,
    only_new: bool = False,
) -> bool:
    """Save a contour training triple to disk.

    Output layout::

        output_dir/{source}_{scene}_{frame}/
        ├── with_contours.png
        ├── without_contours.png
        ├── contour_mask.png
        └── metadata.json

    Args:
        output_dir: Root output directory.
        pair: Source contour/anime pair.
        contour_img: Resized contour frame.
        anime_img: Resized anime frame.
        contour_mask: Binary contour mask array.
        metadata: Metadata dict.
        only_new: Skip if output directory already exists.

    Returns:
        True if saved, False if skipped.
    """
    example_id = f"{ANIMERUN_SOURCE}_{pair.scene_id}_{pair.frame_id}"
    example_dir = output_dir / example_id

    if only_new and example_dir.exists():
        return False

    example_dir.mkdir(parents=True, exist_ok=True)

    contour_img.save(example_dir / "with_contours.png", format="PNG", compress_level=9)
    anime_img.save(example_dir / "without_contours.png", format="PNG", compress_level=9)

    mask_img = Image.fromarray(contour_mask, mode="L")
    mask_img.save(example_dir / "contour_mask.png", format="PNG", compress_level=9)

    meta_path = example_dir / "metadata.json"
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return True


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def convert_pair(
    pair: ContourPair,
    output_dir: Path,
    *,
    resolution: int = STRATA_RESOLUTION,
    threshold: int = CONTOUR_DIFF_THRESHOLD,
    only_new: bool = False,
) -> bool:
    """Convert a single contour/anime pair to Strata contour training format.

    Args:
        pair: Contour/anime frame pair.
        output_dir: Root output directory.
        resolution: Target square resolution.
        threshold: Contour detection threshold.
        only_new: Skip if output already exists.

    Returns:
        True if saved, False if skipped or failed.
    """
    try:
        contour_raw = Image.open(pair.contour_path)
        contour_raw.load()
        anime_raw = Image.open(pair.anime_path)
        anime_raw.load()
    except OSError as exc:
        logger.warning("Failed to load frame pair %s/%s: %s", pair.scene_id, pair.frame_id, exc)
        return False

    contour_img = _resize_image(contour_raw, resolution)
    anime_img = _resize_image(anime_raw, resolution)

    contour_mask = generate_contour_mask(contour_img, anime_img, threshold)
    metadata = _build_metadata(pair, resolution)

    return _save_contour_triple(
        output_dir,
        pair,
        contour_img,
        anime_img,
        contour_mask,
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
    threshold: int = CONTOUR_DIFF_THRESHOLD,
    only_new: bool = False,
    max_frames: int = 0,
) -> AdapterResult:
    """Convert all matched pairs in a scene to contour training format.

    Args:
        scene_dir: Path to the scene directory.
        output_dir: Root output directory.
        split: Dataset split name.
        scene_id: Scene identifier.
        resolution: Target square resolution.
        threshold: Contour detection threshold.
        only_new: Skip existing output directories.
        max_frames: Maximum frames to convert per scene (0 = unlimited).

    Returns:
        AdapterResult summarizing the conversion.
    """
    result = AdapterResult(scene_id=scene_id)
    pairs = discover_pairs(scene_dir, split, scene_id)

    if max_frames > 0:
        pairs = pairs[:max_frames]

    for pair in pairs:
        saved = convert_pair(
            pair,
            output_dir,
            resolution=resolution,
            threshold=threshold,
            only_new=only_new,
        )
        if saved:
            result.frames_saved += 1
        else:
            result.frames_skipped += 1

    logger.info(
        "AnimeRun scene %s (%s): %d saved, %d skipped",
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
    threshold: int = CONTOUR_DIFF_THRESHOLD,
    only_new: bool = False,
    max_frames_per_scene: int = 0,
    max_scenes: int = 0,
) -> list[AdapterResult]:
    """Convert all AnimeRun scenes to Strata contour training format.

    Args:
        animerun_dir: Root AnimeRun dataset directory.
        output_dir: Root output directory.
        resolution: Target square resolution.
        threshold: Contour detection threshold.
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
        logger.warning("No valid scenes found in %s", animerun_dir)
        return []

    if max_scenes > 0:
        scenes = scenes[:max_scenes]

    logger.info("Found %d scenes in %s", len(scenes), animerun_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[AdapterResult] = []

    for split, scene_id, scene_dir in scenes:
        scene_result = convert_scene(
            scene_dir,
            output_dir,
            split,
            scene_id,
            resolution=resolution,
            threshold=threshold,
            only_new=only_new,
            max_frames=max_frames_per_scene,
        )
        results.append(scene_result)

    total_saved = sum(r.frames_saved for r in results)
    logger.info(
        "AnimeRun conversion complete: %d scenes, %d frames total",
        len(results),
        total_saved,
    )

    return results
