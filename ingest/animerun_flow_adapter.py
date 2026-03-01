"""Convert AnimeRun optical flow data to Strata motion training format.

AnimeRun (NeurIPS 2022) provides forward and backward optical flow derived
from open-source 3D animated movies.  Each flow file encodes per-pixel 2D
displacement vectors (dx, dy) between consecutive frames.

Input directory structure (AnimeRun_v2)::

    AnimeRun_v2/
    ├── train/
    │   ├── Flow/{scene}/
    │   │   ├── forward/   ← forward optical flow (.npy or .flo)
    │   │   └── backward/  ← backward optical flow (.npy or .flo)
    │   └── Frame_Anime/{scene}/original/*.png  ← color frames
    └── test/
        └── ...

Output per frame pair::

    output_dir/{source}_{scene}_{frame}/
    ├── frame_t.png          ← source frame (512×512)
    ├── frame_t1.png         ← target frame (512×512)
    ├── flow_forward.npy     ← forward optical flow (H×W×2 float32)
    ├── flow_backward.npy    ← backward optical flow (H×W×2 float32)
    ├── flow_viz.png         ← HSV flow visualization for QA
    └── metadata.json

This module is pure Python (no Blender dependency) so it can be imported
outside Blender for testing and validation.

Reference: https://github.com/lisiyao21/AnimeRun
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

ANIMERUN_FLOW_SOURCE = "animerun_flow"

STRATA_RESOLUTION = 512

# Top-level data type directories within each split.
_FLOW_DIR = "Flow"
_ANIME_DIR = "Frame_Anime"
_ANIME_VARIANT = "original"

# Subdirectories within each scene's Flow directory.
_FORWARD_DIR = "forward"
_BACKWARD_DIR = "backward"

# Dataset split directories.
_SPLIT_DIRS = ("train", "test")

# Supported flow file extensions.
_FLOW_EXTENSIONS = {".npy", ".flo"}

# Supported image extensions.
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Middlebury .flo magic number.
_FLO_MAGIC = 202021.25

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
class FlowPair:
    """A matched frame pair with optical flow data."""

    frame_id: str
    scene_id: str
    split: str
    frame_t_path: Path
    frame_t1_path: Path
    flow_fwd_path: Path | None = None
    flow_bwd_path: Path | None = None


@dataclass
class AdapterResult:
    """Result of converting AnimeRun flow data for a scene."""

    scene_id: str
    frames_saved: int = 0
    frames_skipped: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Optical flow I/O
# ---------------------------------------------------------------------------


def read_flo_file(flo_path: Path) -> np.ndarray | None:
    """Read a Middlebury .flo optical flow file.

    Format: 4-byte magic (float32) + 4-byte width (int32) + 4-byte height
    (int32) + width*height*2 float32 values (u, v interleaved).

    Args:
        flo_path: Path to the .flo file.

    Returns:
        (H, W, 2) float32 array, or None if reading fails.
    """
    try:
        with open(flo_path, "rb") as f:
            magic = struct.unpack("f", f.read(4))[0]
            if magic != _FLO_MAGIC:
                logger.warning("Invalid .flo magic in %s: %f", flo_path, magic)
                return None
            w = struct.unpack("i", f.read(4))[0]
            h = struct.unpack("i", f.read(4))[0]
            data = np.frombuffer(f.read(), dtype=np.float32)
            return data.reshape((h, w, 2))
    except (OSError, struct.error, ValueError) as exc:
        logger.warning("Failed to read .flo file %s: %s", flo_path, exc)
        return None


def load_flow(flow_path: Path) -> np.ndarray | None:
    """Load optical flow from .flo or .npy file.

    Args:
        flow_path: Path to the flow file.

    Returns:
        (H, W, 2) float32 array, or None if loading fails.
    """
    if flow_path.suffix == ".flo":
        return read_flo_file(flow_path)
    if flow_path.suffix == ".npy":
        try:
            arr = np.load(flow_path)
            if arr.ndim == 3 and arr.shape[2] == 2:
                return arr.astype(np.float32)
            logger.warning("Unexpected flow shape in %s: %s", flow_path, arr.shape)
            return None
        except (OSError, ValueError) as exc:
            logger.warning("Failed to load flow %s: %s", flow_path, exc)
            return None
    logger.warning("Unsupported flow format: %s", flow_path.suffix)
    return None


# ---------------------------------------------------------------------------
# Flow visualization
# ---------------------------------------------------------------------------


def flow_to_hsv(flow: np.ndarray) -> np.ndarray:
    """Convert optical flow to an HSV color wheel visualization.

    Uses the standard Middlebury color wheel encoding:
    - Hue encodes flow direction (angle)
    - Saturation is set to maximum
    - Value encodes flow magnitude (normalized to [0, 255])

    Args:
        flow: (H, W, 2) float32 optical flow array.

    Returns:
        (H, W, 3) uint8 RGB image array.
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    magnitude = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u)

    # Normalize angle to [0, 180] for OpenCV HSV hue range.
    hue = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)

    # Normalize magnitude to [0, 255].
    max_mag = magnitude.max()
    if max_mag > 0:
        value = (magnitude / max_mag * 255).astype(np.uint8)
    else:
        value = np.zeros_like(hue)

    saturation = np.full_like(hue, 255)

    # Stack HSV and convert to RGB via PIL.
    hsv_arr = np.stack([hue, saturation, value], axis=2)
    hsv_img = Image.fromarray(hsv_arr, mode="HSV")
    return np.array(hsv_img.convert("RGB"))


# ---------------------------------------------------------------------------
# Flow scaling
# ---------------------------------------------------------------------------


def scale_flow(
    flow: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """Resize optical flow and scale displacement vectors proportionally.

    When frames are resized, flow vectors must be scaled by the same ratio
    so that the displacements remain correct in the new coordinate space.

    Args:
        flow: (H, W, 2) float32 optical flow.
        target_h: Target height.
        target_w: Target width.

    Returns:
        (target_h, target_w, 2) float32 scaled flow.
    """
    src_h, src_w = flow.shape[:2]
    if src_h == target_h and src_w == target_w:
        return flow

    scale_x = target_w / src_w
    scale_y = target_h / src_h

    # Resize each channel using PIL for consistency with image resizing.
    u_img = Image.fromarray(flow[:, :, 0], mode="F")
    v_img = Image.fromarray(flow[:, :, 1], mode="F")

    u_resized = np.array(u_img.resize((target_w, target_h), Image.BILINEAR))
    v_resized = np.array(v_img.resize((target_w, target_h), Image.BILINEAR))

    # Scale displacement values proportionally.
    u_resized *= scale_x
    v_resized *= scale_y

    return np.stack([u_resized, v_resized], axis=2).astype(np.float32)


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
    """Discover scene directories with both Flow and Frame_Anime data.

    A valid scene must have a Flow directory with forward and/or backward
    subdirectories, plus matching anime frames in Frame_Anime.

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

        flow_base = split_dir / _FLOW_DIR
        anime_base = split_dir / _ANIME_DIR

        if not flow_base.is_dir() or not anime_base.is_dir():
            continue

        for scene_dir in sorted(flow_base.iterdir()):
            if not scene_dir.is_dir() or scene_dir.name.startswith("."):
                continue
            scene_id = scene_dir.name

            # Check that forward or backward flow exists.
            has_flow = (scene_dir / _FORWARD_DIR).is_dir() or (scene_dir / _BACKWARD_DIR).is_dir()
            if not has_flow:
                continue

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
) -> list[FlowPair]:
    """Discover matched frame pairs with optical flow in a scene.

    Each flow file corresponds to the displacement from frame t to frame t+1.
    We match flow files by stem to consecutive anime frames sorted by name.

    Args:
        scene_dir: Path to the split directory (train/ or test/).
        split: Dataset split name.
        scene_id: Scene identifier.

    Returns:
        List of FlowPair objects, sorted by frame name.
    """
    flow_dir = scene_dir / _FLOW_DIR / scene_id
    anime_dir = scene_dir / _ANIME_DIR / scene_id / _ANIME_VARIANT
    if not anime_dir.is_dir():
        anime_dir = scene_dir / _ANIME_DIR / scene_id

    fwd_dir = flow_dir / _FORWARD_DIR
    bwd_dir = flow_dir / _BACKWARD_DIR

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

    # Index flow files by stem.
    fwd_by_stem: dict[str, Path] = {}
    if fwd_dir.is_dir():
        for p in fwd_dir.iterdir():
            if p.suffix.lower() in _FLOW_EXTENSIONS and p.is_file():
                fwd_by_stem[p.stem] = p

    bwd_by_stem: dict[str, Path] = {}
    if bwd_dir.is_dir():
        for p in bwd_dir.iterdir():
            if p.suffix.lower() in _FLOW_EXTENSIONS and p.is_file():
                bwd_by_stem[p.stem] = p

    # Match flow files to frame pairs.
    all_flow_stems = sorted(set(fwd_by_stem) | set(bwd_by_stem))
    pairs: list[FlowPair] = []

    for stem in all_flow_stems:
        if stem not in anime_by_stem:
            continue
        if stem not in next_frame:
            continue  # Last frame has no t+1.

        next_stem = next_frame[stem]
        if next_stem not in anime_by_stem:
            continue

        pairs.append(
            FlowPair(
                frame_id=stem,
                scene_id=scene_id,
                split=split,
                frame_t_path=anime_by_stem[stem],
                frame_t1_path=anime_by_stem[next_stem],
                flow_fwd_path=fwd_by_stem.get(stem),
                flow_bwd_path=bwd_by_stem.get(stem),
            )
        )

    return pairs


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def _resize_image(img: Image.Image, resolution: int = STRATA_RESOLUTION) -> Image.Image:
    """Resize an image to the target resolution.

    Converts to RGB (flow training doesn't need alpha).

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


def _build_metadata(
    pair: FlowPair,
    resolution: int,
    *,
    has_forward: bool,
    has_backward: bool,
) -> dict[str, Any]:
    """Build metadata dict for a flow training example.

    Args:
        pair: The source flow pair.
        resolution: Output resolution.
        has_forward: Whether forward flow was extracted.
        has_backward: Whether backward flow was extracted.

    Returns:
        Metadata dict ready for JSON serialization.
    """
    return {
        "source": ANIMERUN_FLOW_SOURCE,
        "scene_id": pair.scene_id,
        "frame_id": pair.frame_id,
        "split": pair.split,
        "resolution": resolution,
        "data_type": "optical_flow_pair",
        "has_segmentation_mask": False,
        "has_joints": False,
        "has_draw_order": False,
        "has_optical_flow": has_forward or has_backward,
        "has_forward_flow": has_forward,
        "has_backward_flow": has_backward,
        "missing_annotations": _MISSING_ANNOTATIONS,
    }


def _save_flow_pair(
    output_dir: Path,
    pair: FlowPair,
    frame_t: Image.Image,
    frame_t1: Image.Image,
    flow_fwd: np.ndarray | None,
    flow_bwd: np.ndarray | None,
    metadata: dict[str, Any],
    *,
    only_new: bool = False,
) -> bool:
    """Save a flow training example to disk.

    Output layout::

        output_dir/{source}_{scene}_{frame}/
        ├── frame_t.png
        ├── frame_t1.png
        ├── flow_forward.npy
        ├── flow_backward.npy
        ├── flow_viz.png
        └── metadata.json

    Args:
        output_dir: Root output directory.
        pair: Source flow pair.
        frame_t: Resized source frame.
        frame_t1: Resized target frame.
        flow_fwd: Forward optical flow array, or None.
        flow_bwd: Backward optical flow array, or None.
        metadata: Metadata dict.
        only_new: Skip if output directory already exists.

    Returns:
        True if saved, False if skipped.
    """
    example_id = f"{ANIMERUN_FLOW_SOURCE}_{pair.scene_id}_{pair.frame_id}"
    example_dir = output_dir / example_id

    if only_new and example_dir.exists():
        return False

    example_dir.mkdir(parents=True, exist_ok=True)

    frame_t.save(example_dir / "frame_t.png", format="PNG", compress_level=9)
    frame_t1.save(example_dir / "frame_t1.png", format="PNG", compress_level=9)

    if flow_fwd is not None:
        np.save(example_dir / "flow_forward.npy", flow_fwd)

        # Generate flow visualization from forward flow.
        viz_rgb = flow_to_hsv(flow_fwd)
        viz_img = Image.fromarray(viz_rgb, mode="RGB")
        viz_img.save(example_dir / "flow_viz.png", format="PNG", compress_level=9)

    if flow_bwd is not None:
        np.save(example_dir / "flow_backward.npy", flow_bwd)

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
    pair: FlowPair,
    output_dir: Path,
    *,
    resolution: int = STRATA_RESOLUTION,
    only_new: bool = False,
) -> bool:
    """Convert a single flow pair to Strata motion training format.

    Args:
        pair: Flow pair data.
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

    # Load and scale optical flow.
    flow_fwd: np.ndarray | None = None
    flow_bwd: np.ndarray | None = None

    if pair.flow_fwd_path is not None:
        flow_fwd = load_flow(pair.flow_fwd_path)
        if flow_fwd is not None:
            flow_fwd = scale_flow(flow_fwd, resolution, resolution)

    if pair.flow_bwd_path is not None:
        flow_bwd = load_flow(pair.flow_bwd_path)
        if flow_bwd is not None:
            flow_bwd = scale_flow(flow_bwd, resolution, resolution)

    metadata = _build_metadata(
        pair,
        resolution,
        has_forward=flow_fwd is not None,
        has_backward=flow_bwd is not None,
    )

    return _save_flow_pair(
        output_dir,
        pair,
        frame_t,
        frame_t1,
        flow_fwd,
        flow_bwd,
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
    """Convert all flow pairs in a scene to motion training format.

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
        "AnimeRun flow scene %s (%s): %d saved, %d skipped",
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
    """Convert all AnimeRun flow scenes to Strata motion training format.

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
        logger.warning("No valid flow scenes found in %s", animerun_dir)
        return []

    if max_scenes > 0:
        scenes = scenes[:max_scenes]

    logger.info("Found %d flow scenes in %s", len(scenes), animerun_dir)

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
        "AnimeRun flow conversion complete: %d scenes, %d pairs total",
        len(results),
        total_saved,
    )

    return results
