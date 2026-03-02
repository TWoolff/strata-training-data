"""Convert AnimeRun temporal correspondence data to Strata training format.

AnimeRun (NeurIPS 2022) provides segment matching and occlusion masks that
track how objects correspond and become visible/hidden across consecutive
animation frames.

Input directory structure (AnimeRun_v2)::

    AnimeRun_v2/
    ├── train/
    │   ├── SegMatching/{scene}/       ← segment correspondence across frames
    │   ├── UnmatchedForward/{scene}/  ← forward occlusion masks (disocclusion)
    │   ├── UnmatchedBackward/{scene}/ ← backward occlusion masks (occlusion)
    │   └── Frame_Anime/{scene}/original/*.png  ← color frames
    └── test/
        └── ...

Output per frame pair::

    output_dir/{source}_{scene}_{frame}/
    ├── frame_t.png               ← source frame (512×512)
    ├── frame_t1.png              ← target frame (512×512)
    ├── seg_matching.npy          ← segment correspondence map
    ├── occlusion_forward.png     ← forward occlusion mask (white = newly occluded)
    ├── occlusion_backward.png    ← backward occlusion mask (white = newly revealed)
    ├── occlusion_overlay.png     ← visualization overlay (for QA)
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

ANIMERUN_CORRESPONDENCE_SOURCE = "animerun_correspondence"

STRATA_RESOLUTION = 512

# Top-level data type directories within each split.
_SEG_MATCHING_DIR = "SegMatching"
_UNMATCHED_FWD_DIR = "UnmatchedForward"
_UNMATCHED_BWD_DIR = "UnmatchedBackward"
_ANIME_DIR = "Frame_Anime"
_ANIME_VARIANT = "original"
_SEG_MATCHING_SUBDIR = "forward"  # SegMatching files live in forward/ subdir.

# Dataset split directories.
_SPLIT_DIRS = ("train", "test")

# Supported file extensions.
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
_DATA_EXTENSIONS = _IMAGE_EXTENSIONS | {".npy", ".json"}

# Annotations that this adapter does NOT provide.
_MISSING_ANNOTATIONS = [
    "strata_segmentation",
    "joints",
    "draw_order",
]

# Colors for occlusion overlay visualization.
_OCCLUDED_COLOR = (255, 0, 0)  # Red = occluded in forward direction.
_REVEALED_COLOR = (0, 0, 255)  # Blue = newly revealed in backward direction.
_OVERLAY_ALPHA = 0.5


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CorrespondencePair:
    """A matched frame pair with temporal correspondence data."""

    frame_id: str
    scene_id: str
    split: str
    frame_t_path: Path
    frame_t1_path: Path
    seg_matching_path: Path | None = None
    unmatched_fwd_path: Path | None = None
    unmatched_bwd_path: Path | None = None


@dataclass
class AdapterResult:
    """Result of converting AnimeRun correspondence data for a scene."""

    scene_id: str
    frames_saved: int = 0
    frames_skipped: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------


def load_correspondence_map(path: Path) -> np.ndarray | None:
    """Load a segment correspondence map from .json, .npy, or image file.

    AnimeRun SegMatching JSON format: ``{"0": [5], "1": [3], ...}``
    where keys are source segment indices and values are single-element
    lists of matched target segment indices (-1 = unmatched).

    Args:
        path: Path to the correspondence data file.

    Returns:
        1D float32 array (from JSON) or 2D/3D array (from npy/image),
        or None if loading fails.
    """
    try:
        if path.suffix.lower() == ".json":
            with open(path, encoding="utf-8") as f:
                match_dict = json.load(f)
            if not match_dict:
                return None
            n = max(int(k) for k in match_dict) + 1
            arr = np.full(n, -1, dtype=np.float32)
            for k, v in match_dict.items():
                arr[int(k)] = int(v[0]) if isinstance(v, list) else int(v)
            return arr
        if path.suffix.lower() == ".npy":
            return np.load(path)
        # Image file — load as grayscale array.
        img = Image.open(path)
        img.load()
        if img.mode != "L":
            img = img.convert("L")
        return np.array(img, dtype=np.uint8)
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("Failed to load correspondence map %s: %s", path, exc)
        return None


def load_occlusion_mask(path: Path) -> np.ndarray | None:
    """Load an occlusion mask from .npy or image file.

    Returns a 2D uint8 binary mask (0 or 255).

    Args:
        path: Path to the occlusion mask file.

    Returns:
        2D uint8 array, or None if loading fails.
    """
    try:
        if path.suffix.lower() == ".npy":
            arr = np.load(path)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            # Normalize to binary 0/255.
            arr = (arr > 0).astype(np.uint8) * 255
            return arr
        img = Image.open(path)
        img.load()
        if img.mode != "L":
            img = img.convert("L")
        arr = np.array(img, dtype=np.uint8)
        # Normalize to binary 0/255.
        return (arr > 0).astype(np.uint8) * 255
    except (OSError, ValueError) as exc:
        logger.warning("Failed to load occlusion mask %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def generate_occlusion_overlay(
    anime_img: Image.Image,
    fwd_mask: np.ndarray | None,
    bwd_mask: np.ndarray | None,
    alpha: float = _OVERLAY_ALPHA,
) -> Image.Image:
    """Generate a color overlay showing forward and backward occlusion regions.

    Forward occlusion (red) = pixels visible in frame_t but hidden in frame_t+1.
    Backward occlusion (blue) = pixels visible in frame_t+1 but hidden in frame_t.

    Args:
        anime_img: RGB anime frame (frame_t).
        fwd_mask: 2D uint8 forward occlusion mask, or None.
        bwd_mask: 2D uint8 backward occlusion mask, or None.
        alpha: Overlay blend factor.

    Returns:
        RGB overlay image.
    """
    base = np.array(anime_img, dtype=np.float32)
    overlay = base.copy()

    if fwd_mask is not None:
        fwd_region = fwd_mask > 0
        color = np.array(_OCCLUDED_COLOR, dtype=np.float32)
        overlay[fwd_region] = base[fwd_region] * (1.0 - alpha) + color * alpha

    if bwd_mask is not None:
        bwd_region = bwd_mask > 0
        color = np.array(_REVEALED_COLOR, dtype=np.float32)
        overlay[bwd_region] = base[bwd_region] * (1.0 - alpha) + color * alpha

    return Image.fromarray(overlay.astype(np.uint8), mode="RGB")


# ---------------------------------------------------------------------------
# Resizing
# ---------------------------------------------------------------------------


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
    """Resize an occlusion mask using nearest-neighbor interpolation.

    Nearest-neighbor preserves binary mask values without interpolation
    artifacts.

    Args:
        mask: 2D uint8 array.
        resolution: Target square resolution.

    Returns:
        Resized 2D uint8 array.
    """
    if mask.shape[0] == resolution and mask.shape[1] == resolution:
        return mask
    mask_img = Image.fromarray(mask, mode="L")
    mask_img = mask_img.resize((resolution, resolution), Image.NEAREST)
    return np.array(mask_img)


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
    """Discover scene directories with SegMatching and Frame_Anime data.

    A valid scene must have a SegMatching directory plus matching anime
    frames in Frame_Anime.  UnmatchedForward/Backward are optional.

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

        seg_matching_base = split_dir / _SEG_MATCHING_DIR
        anime_base = split_dir / _ANIME_DIR

        if not seg_matching_base.is_dir() or not anime_base.is_dir():
            continue

        for scene_dir in sorted(seg_matching_base.iterdir()):
            if not scene_dir.is_dir() or scene_dir.name.startswith("."):
                continue
            scene_id = scene_dir.name

            # Check that matching anime frames exist.
            anime_scene = anime_base / scene_id / _ANIME_VARIANT
            if not anime_scene.is_dir():
                anime_scene = anime_base / scene_id
            if anime_scene.is_dir():
                scenes.append((split_name, scene_id, split_dir))

    return scenes


def discover_pairs(
    scene_dir: Path,
    split: str,
    scene_id: str,
) -> list[CorrespondencePair]:
    """Discover matched frame pairs with correspondence data in a scene.

    Each SegMatching file corresponds to the mapping from frame t to frame t+1.
    We match files by stem to consecutive anime frames sorted by name.

    Args:
        scene_dir: Path to the split directory (train/ or test/).
        split: Dataset split name.
        scene_id: Scene identifier.

    Returns:
        List of CorrespondencePair objects, sorted by frame name.
    """
    seg_dir = scene_dir / _SEG_MATCHING_DIR / scene_id / _SEG_MATCHING_SUBDIR
    anime_dir = scene_dir / _ANIME_DIR / scene_id / _ANIME_VARIANT
    if not anime_dir.is_dir():
        anime_dir = scene_dir / _ANIME_DIR / scene_id

    fwd_dir = scene_dir / _UNMATCHED_FWD_DIR / scene_id
    bwd_dir = scene_dir / _UNMATCHED_BWD_DIR / scene_id

    # Index anime frames by stem, sorted.
    anime_files = sorted(
        p for p in anime_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS and p.is_file()
    )
    anime_by_stem = {p.stem: p for p in anime_files}
    anime_stems_sorted = [p.stem for p in anime_files]

    # Build consecutive frame lookup: stem → next_stem.
    next_frame: dict[str, str] = {}
    for i in range(len(anime_stems_sorted) - 1):
        next_frame[anime_stems_sorted[i]] = anime_stems_sorted[i + 1]

    # Index SegMatching files by stem.
    seg_by_stem: dict[str, Path] = {}
    if seg_dir.is_dir():
        for p in seg_dir.iterdir():
            if p.suffix.lower() in _DATA_EXTENSIONS and p.is_file():
                seg_by_stem[p.stem] = p
    if not seg_by_stem:
        logger.warning(
            "No SegMatching files found in %s (dir exists: %s)",
            seg_dir,
            seg_dir.is_dir(),
        )

    # Index UnmatchedForward files by stem.
    fwd_by_stem: dict[str, Path] = {}
    if fwd_dir.is_dir():
        for p in fwd_dir.iterdir():
            if p.suffix.lower() in _DATA_EXTENSIONS and p.is_file():
                fwd_by_stem[p.stem] = p

    # Index UnmatchedBackward files by stem.
    bwd_by_stem: dict[str, Path] = {}
    if bwd_dir.is_dir():
        for p in bwd_dir.iterdir():
            if p.suffix.lower() in _DATA_EXTENSIONS and p.is_file():
                bwd_by_stem[p.stem] = p

    # Match correspondence data to frame pairs.
    # Use SegMatching stems as the primary set (occlusion masks are optional).
    all_stems = sorted(seg_by_stem.keys())
    pairs: list[CorrespondencePair] = []

    for stem in all_stems:
        if stem not in anime_by_stem:
            continue
        if stem not in next_frame:
            continue  # Last frame has no t+1.

        next_stem = next_frame[stem]
        if next_stem not in anime_by_stem:
            continue

        pairs.append(
            CorrespondencePair(
                frame_id=stem,
                scene_id=scene_id,
                split=split,
                frame_t_path=anime_by_stem[stem],
                frame_t1_path=anime_by_stem[next_stem],
                seg_matching_path=seg_by_stem.get(stem),
                unmatched_fwd_path=fwd_by_stem.get(stem),
                unmatched_bwd_path=bwd_by_stem.get(stem),
            )
        )

    return pairs


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def _build_metadata(
    pair: CorrespondencePair,
    resolution: int,
    *,
    has_seg_matching: bool,
    has_occlusion_fwd: bool,
    has_occlusion_bwd: bool,
) -> dict[str, Any]:
    """Build metadata dict for a temporal correspondence training example.

    Args:
        pair: The source correspondence pair.
        resolution: Output resolution.
        has_seg_matching: Whether segment matching was extracted.
        has_occlusion_fwd: Whether forward occlusion mask was extracted.
        has_occlusion_bwd: Whether backward occlusion mask was extracted.

    Returns:
        Metadata dict ready for JSON serialization.
    """
    return {
        "source": ANIMERUN_CORRESPONDENCE_SOURCE,
        "scene_id": pair.scene_id,
        "frame_id": pair.frame_id,
        "split": pair.split,
        "resolution": resolution,
        "data_type": "temporal_correspondence",
        "has_segmentation_mask": False,
        "has_joints": False,
        "has_draw_order": False,
        "has_seg_matching": has_seg_matching,
        "has_occlusion_forward": has_occlusion_fwd,
        "has_occlusion_backward": has_occlusion_bwd,
        "missing_annotations": _MISSING_ANNOTATIONS,
    }


def _save_correspondence_pair(
    output_dir: Path,
    pair: CorrespondencePair,
    frame_t: Image.Image,
    frame_t1: Image.Image,
    seg_matching: np.ndarray | None,
    occ_fwd: np.ndarray | None,
    occ_bwd: np.ndarray | None,
    metadata: dict[str, Any],
    *,
    only_new: bool = False,
) -> bool:
    """Save a temporal correspondence training example to disk.

    Output layout::

        output_dir/{source}_{scene}_{frame}/
        ├── frame_t.png
        ├── frame_t1.png
        ├── seg_matching.npy
        ├── occlusion_forward.png
        ├── occlusion_backward.png
        ├── occlusion_overlay.png
        └── metadata.json

    Args:
        output_dir: Root output directory.
        pair: Source correspondence pair.
        frame_t: Resized source frame.
        frame_t1: Resized target frame.
        seg_matching: Segment correspondence array, or None.
        occ_fwd: Forward occlusion mask (uint8), or None.
        occ_bwd: Backward occlusion mask (uint8), or None.
        metadata: Metadata dict.
        only_new: Skip if output directory already exists.

    Returns:
        True if saved, False if skipped.
    """
    example_id = f"{ANIMERUN_CORRESPONDENCE_SOURCE}_{pair.scene_id}_{pair.frame_id}"
    example_dir = output_dir / example_id

    if only_new and example_dir.exists():
        return False

    example_dir.mkdir(parents=True, exist_ok=True)

    frame_t.save(example_dir / "frame_t.png", format="PNG", compress_level=9)
    frame_t1.save(example_dir / "frame_t1.png", format="PNG", compress_level=9)

    if seg_matching is not None:
        np.save(example_dir / "seg_matching.npy", seg_matching)

    if occ_fwd is not None:
        occ_fwd_img = Image.fromarray(occ_fwd, mode="L")
        occ_fwd_img.save(example_dir / "occlusion_forward.png", format="PNG", compress_level=9)

    if occ_bwd is not None:
        occ_bwd_img = Image.fromarray(occ_bwd, mode="L")
        occ_bwd_img.save(example_dir / "occlusion_backward.png", format="PNG", compress_level=9)

    # Generate occlusion overlay for QA.
    if occ_fwd is not None or occ_bwd is not None:
        overlay = generate_occlusion_overlay(frame_t, occ_fwd, occ_bwd)
        overlay.save(example_dir / "occlusion_overlay.png", format="PNG", compress_level=9)

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
    pair: CorrespondencePair,
    output_dir: Path,
    *,
    resolution: int = STRATA_RESOLUTION,
    only_new: bool = False,
) -> bool:
    """Convert a single correspondence pair to Strata training format.

    Args:
        pair: Correspondence pair data.
        output_dir: Root output directory.
        resolution: Target square resolution.
        only_new: Skip if output already exists.

    Returns:
        True if saved, False if skipped or failed.
    """
    try:
        img_t_raw = Image.open(pair.frame_t_path)
        img_t_raw.load()
        img_t1_raw = Image.open(pair.frame_t1_path)
        img_t1_raw.load()
    except OSError as exc:
        logger.warning("Failed to load frame pair %s/%s: %s", pair.scene_id, pair.frame_id, exc)
        return False

    frame_t = _resize_image(img_t_raw, resolution)
    frame_t1 = _resize_image(img_t1_raw, resolution)

    # Load segment matching data.
    seg_matching: np.ndarray | None = None
    if pair.seg_matching_path is not None:
        seg_matching = load_correspondence_map(pair.seg_matching_path)

    # Load and resize occlusion masks.
    occ_fwd: np.ndarray | None = None
    occ_bwd: np.ndarray | None = None

    if pair.unmatched_fwd_path is not None:
        occ_fwd = load_occlusion_mask(pair.unmatched_fwd_path)
        if occ_fwd is not None:
            occ_fwd = _resize_mask(occ_fwd, resolution)

    if pair.unmatched_bwd_path is not None:
        occ_bwd = load_occlusion_mask(pair.unmatched_bwd_path)
        if occ_bwd is not None:
            occ_bwd = _resize_mask(occ_bwd, resolution)

    metadata = _build_metadata(
        pair,
        resolution,
        has_seg_matching=seg_matching is not None,
        has_occlusion_fwd=occ_fwd is not None,
        has_occlusion_bwd=occ_bwd is not None,
    )

    return _save_correspondence_pair(
        output_dir,
        pair,
        frame_t,
        frame_t1,
        seg_matching,
        occ_fwd,
        occ_bwd,
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
    """Convert all correspondence pairs in a scene.

    Args:
        scene_dir: Path to the split directory (e.g. train/).
        output_dir: Root output directory.
        split: Dataset split name.
        scene_id: Scene identifier.
        resolution: Target square resolution.
        only_new: Skip existing output directories.
        max_frames: Maximum pairs to convert per scene (0 = unlimited).

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
            only_new=only_new,
        )
        if saved:
            result.frames_saved += 1
        else:
            result.frames_skipped += 1

    logger.info(
        "AnimeRun correspondence scene %s (%s): %d saved, %d skipped",
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
    """Convert all AnimeRun correspondence scenes to Strata training format.

    Args:
        animerun_dir: Root AnimeRun dataset directory.
        output_dir: Root output directory.
        resolution: Target square resolution.
        only_new: Skip existing output directories.
        max_frames_per_scene: Max pairs per scene (0 = unlimited).
        max_scenes: Max scenes to process (0 = unlimited).

    Returns:
        List of AdapterResult objects for each scene.
    """
    if not animerun_dir.is_dir():
        logger.error("AnimeRun directory not found: %s", animerun_dir)
        return []

    scenes = discover_scenes(animerun_dir)

    if not scenes:
        logger.warning("No valid correspondence scenes found in %s", animerun_dir)
        return []

    if max_scenes > 0:
        scenes = scenes[:max_scenes]

    logger.info("Found %d correspondence scenes in %s", len(scenes), animerun_dir)

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
        "AnimeRun correspondence conversion complete: %d scenes, %d pairs total",
        len(results),
        total_saved,
    )

    return results
