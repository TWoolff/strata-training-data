"""Extract skeleton joint positions and optical flow from LinkTo-Anime.

LinkTo-Anime (arXiv 2506.02733, 2025) provides ~29K cel anime frames
rendered from 3D models animated with Mixamo skeletons.  Each video
sequence includes forward/backward optical flow, occlusion masks, and
skeleton data.

Input directory structure::

    linkto_anime/
    ├── train/
    │   ├── seq_001/
    │   │   ├── frames/          ← rendered anime frames (PNG)
    │   │   ├── flow_fwd/        ← forward optical flow (.flo or .npy)
    │   │   ├── flow_bwd/        ← backward optical flow (.flo or .npy)
    │   │   ├── occlusion/       ← occlusion masks (PNG)
    │   │   └── skeleton/        ← joint position data (JSON or .npy)
    │   └── seq_002/
    ├── val/
    └── test/

Output per frame::

    output_dir/{source}_{seq}_{frame}/
    ├── image.png               ← anime frame (512×512)
    ├── joints.json             ← 2D joint positions in Strata format
    ├── flow_fwd.npy            ← forward optical flow (H, W, 2) float32
    ├── flow_bwd.npy            ← backward optical flow (H, W, 2) float32
    └── metadata.json

This module is pure Python (no Blender dependency) so it can be imported
outside Blender for testing and validation.

Reference: arXiv 2506.02733
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

from pipeline.config import (
    MIXAMO_BONE_MAP,
    REGION_NAMES,
    RegionId,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LINKTO_SOURCE = "linkto_anime"

STRATA_RESOLUTION = 512

# Subdirectory names within each sequence directory.
_FRAMES_DIR = "frames"
_FLOW_FWD_DIR = "flow_fwd"
_FLOW_BWD_DIR = "flow_bwd"
_SKELETON_DIR = "skeleton"

# Dataset split directories.
_SPLIT_DIRS = ("train", "val", "test")

# Annotations that LinkTo-Anime does NOT provide.
_MISSING_ANNOTATIONS = [
    "strata_segmentation",
    "draw_order",
]

# Middlebury .flo magic number.
_FLO_MAGIC = 202021.25


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FrameData:
    """Data for a single frame in a LinkTo-Anime sequence."""

    frame_id: str
    sequence_id: str
    split: str
    frame_path: Path
    skeleton_path: Path | None = None
    flow_fwd_path: Path | None = None
    flow_bwd_path: Path | None = None


@dataclass
class JointPosition:
    """A single joint in 2D screen coordinates."""

    region_id: RegionId
    region_name: str
    x: float
    y: float
    visible: bool = True


@dataclass
class AdapterResult:
    """Result of converting a LinkTo-Anime sequence or batch."""

    sequence_id: str
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
# Skeleton parsing
# ---------------------------------------------------------------------------


def _build_mixamo_region_lookup() -> dict[str, RegionId]:
    """Build a lookup from stripped Mixamo bone names to region IDs.

    LinkTo-Anime skeleton data may use bone names with or without the
    ``mixamorig:`` prefix.

    Returns:
        Dict of bone_name → region_id covering both prefixed and stripped names.
    """
    lookup: dict[str, RegionId] = dict(MIXAMO_BONE_MAP)
    # Also add stripped versions (without "mixamorig:" prefix)
    for bone_name, region_id in MIXAMO_BONE_MAP.items():
        if bone_name.startswith("mixamorig:"):
            stripped = bone_name[len("mixamorig:") :]
            lookup[stripped] = region_id
    return lookup


_MIXAMO_LOOKUP = _build_mixamo_region_lookup()


def parse_skeleton_json(skeleton_path: Path) -> list[JointPosition] | None:
    """Parse LinkTo-Anime skeleton JSON into Strata joint positions.

    Expected JSON format (one of):
    - Dict with bone_name → {"x": float, "y": float} or [x, y]
    - Dict with "joints" key containing the above
    - List of {"name": str, "x": float, "y": float}

    Args:
        skeleton_path: Path to skeleton JSON file.

    Returns:
        List of JointPosition objects, or None if parsing fails.
    """
    try:
        data = json.loads(skeleton_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read skeleton %s: %s", skeleton_path, exc)
        return None

    joints: list[JointPosition] = []

    # Unwrap container if present.
    if isinstance(data, dict) and "joints" in data:
        data = data["joints"]

    if isinstance(data, dict):
        for bone_name, pos in data.items():
            region_id = _MIXAMO_LOOKUP.get(bone_name)
            if region_id is None:
                continue
            if isinstance(pos, dict):
                x, y = float(pos.get("x", 0)), float(pos.get("y", 0))
            elif isinstance(pos, (list, tuple)) and len(pos) >= 2:
                x, y = float(pos[0]), float(pos[1])
            else:
                continue
            joints.append(
                JointPosition(
                    region_id=region_id,
                    region_name=REGION_NAMES[region_id],
                    x=x,
                    y=y,
                )
            )
    elif isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                continue
            bone_name = entry.get("name", "")
            region_id = _MIXAMO_LOOKUP.get(bone_name)
            if region_id is None:
                continue
            x = float(entry.get("x", 0))
            y = float(entry.get("y", 0))
            joints.append(
                JointPosition(
                    region_id=region_id,
                    region_name=REGION_NAMES[region_id],
                    x=x,
                    y=y,
                )
            )

    if not joints:
        logger.debug("No Strata-mapped joints found in %s", skeleton_path)
        return None

    return joints


def parse_skeleton_npy(skeleton_path: Path) -> list[JointPosition] | None:
    """Parse LinkTo-Anime skeleton .npy file.

    Expected format: (N, 2) or (N, 3) array where N corresponds to
    Mixamo bone order, or a dict-like .npz with 'positions' and 'names'.

    Args:
        skeleton_path: Path to skeleton .npy/.npz file.

    Returns:
        List of JointPosition objects, or None if parsing fails.
    """
    try:
        data = np.load(skeleton_path, allow_pickle=True)
    except (OSError, ValueError) as exc:
        logger.warning("Failed to load skeleton %s: %s", skeleton_path, exc)
        return None

    if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 2:
        # Assume ordered by some convention — return raw positions
        # with sequential IDs (caller must handle mapping).
        joints: list[JointPosition] = []
        for i in range(min(data.shape[0], 19)):
            region_id = i + 1  # Strata regions 1-19
            if region_id not in REGION_NAMES:
                continue
            joints.append(
                JointPosition(
                    region_id=region_id,
                    region_name=REGION_NAMES[region_id],
                    x=float(data[i, 0]),
                    y=float(data[i, 1]),
                )
            )
        return joints if joints else None

    logger.debug("Unsupported skeleton format in %s", skeleton_path)
    return None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_sequences(linkto_dir: Path) -> list[tuple[str, str, Path]]:
    """Discover video sequences in the LinkTo-Anime dataset.

    Args:
        linkto_dir: Root dataset directory.

    Returns:
        List of (split, sequence_id, sequence_dir) tuples.
    """
    sequences: list[tuple[str, str, Path]] = []

    for split_name in _SPLIT_DIRS:
        split_dir = linkto_dir / split_name
        if not split_dir.is_dir():
            continue

        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir() or seq_dir.name.startswith("."):
                continue
            if (seq_dir / _FRAMES_DIR).is_dir():
                sequences.append((split_name, seq_dir.name, seq_dir))

    return sequences


def discover_frames(
    seq_dir: Path,
    split: str,
    sequence_id: str,
) -> list[FrameData]:
    """Discover frames within a sequence directory.

    Args:
        seq_dir: Path to the sequence directory.
        split: Dataset split name.
        sequence_id: Sequence identifier.

    Returns:
        List of FrameData objects for each discovered frame.
    """
    frames_dir = seq_dir / _FRAMES_DIR
    extensions = {".png", ".jpg", ".jpeg"}

    frame_paths = sorted(
        p for p in frames_dir.iterdir() if p.suffix.lower() in extensions and p.is_file()
    )

    # Build lookup for associated data.
    skeleton_dir = seq_dir / _SKELETON_DIR
    flow_fwd_dir = seq_dir / _FLOW_FWD_DIR
    flow_bwd_dir = seq_dir / _FLOW_BWD_DIR

    frames: list[FrameData] = []
    for frame_path in frame_paths:
        stem = frame_path.stem
        frame = FrameData(
            frame_id=stem,
            sequence_id=sequence_id,
            split=split,
            frame_path=frame_path,
        )

        # Find skeleton file (JSON or npy).
        for ext in (".json", ".npy", ".npz"):
            skel_path = skeleton_dir / f"{stem}{ext}"
            if skel_path.is_file():
                frame.skeleton_path = skel_path
                break

        # Find flow files (.flo or .npy).
        for ext in (".flo", ".npy"):
            fwd_path = flow_fwd_dir / f"{stem}{ext}"
            if fwd_path.is_file():
                frame.flow_fwd_path = fwd_path
                break
        for ext in (".flo", ".npy"):
            bwd_path = flow_bwd_dir / f"{stem}{ext}"
            if bwd_path.is_file():
                frame.flow_bwd_path = bwd_path
                break

        frames.append(frame)

    return frames


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def _resize_image(img: Image.Image, resolution: int = STRATA_RESOLUTION) -> Image.Image:
    """Resize an image to the target resolution."""
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    if img.size != (resolution, resolution):
        img = img.resize((resolution, resolution), Image.LANCZOS)
    return img


def _scale_joints(
    joints: list[JointPosition],
    original_size: tuple[int, int],
    target_resolution: int,
) -> list[JointPosition]:
    """Scale joint positions to the target resolution.

    Args:
        joints: Joint positions in original image coordinates.
        original_size: (width, height) of the original image.
        target_resolution: Target square resolution.

    Returns:
        New list of JointPosition with scaled coordinates.
    """
    orig_w, orig_h = original_size
    if orig_w == 0 or orig_h == 0:
        return joints

    scale_x = target_resolution / orig_w
    scale_y = target_resolution / orig_h

    scaled: list[JointPosition] = []
    for j in joints:
        new_x = j.x * scale_x
        new_y = j.y * scale_y
        visible = 0 <= new_x < target_resolution and 0 <= new_y < target_resolution
        scaled.append(
            JointPosition(
                region_id=j.region_id,
                region_name=j.region_name,
                x=round(new_x, 2),
                y=round(new_y, 2),
                visible=visible,
            )
        )

    return scaled


def _joints_to_strata_json(joints: list[JointPosition]) -> dict[str, Any]:
    """Convert joint positions to Strata joints.json format.

    Args:
        joints: List of joint positions.

    Returns:
        Dict matching Strata joints.json schema.
    """
    return {
        "joints": [
            {
                "region_id": j.region_id,
                "region_name": j.region_name,
                "x": j.x,
                "y": j.y,
                "visible": j.visible,
            }
            for j in joints
        ],
    }


def _build_metadata(
    frame: FrameData,
    resolution: int,
    *,
    has_joints: bool,
    has_flow: bool,
) -> dict[str, Any]:
    """Build metadata dict for a converted frame.

    Args:
        frame: Source frame data.
        resolution: Output resolution.
        has_joints: Whether joint data was extracted.
        has_flow: Whether optical flow was extracted.

    Returns:
        Metadata dict ready for JSON serialization.
    """
    return {
        "source": LINKTO_SOURCE,
        "sequence_id": frame.sequence_id,
        "frame_id": frame.frame_id,
        "split": frame.split,
        "resolution": resolution,
        "has_segmentation_mask": False,
        "has_joints": has_joints,
        "has_draw_order": False,
        "has_optical_flow": has_flow,
        "missing_annotations": _MISSING_ANNOTATIONS,
    }


def _save_frame(
    output_dir: Path,
    frame: FrameData,
    image: Image.Image,
    joints: list[JointPosition] | None,
    flow_fwd: np.ndarray | None,
    flow_bwd: np.ndarray | None,
    metadata: dict[str, Any],
    *,
    only_new: bool = False,
) -> bool:
    """Save a converted frame to disk.

    Args:
        output_dir: Root output directory.
        frame: Source frame data.
        image: Resized frame image.
        joints: Scaled joint positions, or None.
        flow_fwd: Forward optical flow array, or None.
        flow_bwd: Backward optical flow array, or None.
        metadata: Metadata dict.
        only_new: Skip if output directory already exists.

    Returns:
        True if saved, False if skipped.
    """
    example_id = f"{LINKTO_SOURCE}_{frame.sequence_id}_{frame.frame_id}"
    example_dir = output_dir / example_id

    if only_new and example_dir.exists():
        return False

    example_dir.mkdir(parents=True, exist_ok=True)

    image.save(example_dir / "image.png", format="PNG", compress_level=9)

    if joints is not None:
        joints_path = example_dir / "joints.json"
        joints_path.write_text(
            json.dumps(_joints_to_strata_json(joints), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    if flow_fwd is not None:
        np.save(example_dir / "flow_fwd.npy", flow_fwd)

    if flow_bwd is not None:
        np.save(example_dir / "flow_bwd.npy", flow_bwd)

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
    frame: FrameData,
    output_dir: Path,
    *,
    resolution: int = STRATA_RESOLUTION,
    include_flow: bool = True,
    only_new: bool = False,
) -> bool:
    """Convert a single LinkTo-Anime frame to Strata format.

    Args:
        frame: Frame data to convert.
        output_dir: Root output directory.
        resolution: Target square resolution.
        include_flow: Whether to include optical flow data.
        only_new: Skip if output already exists.

    Returns:
        True if saved, False if skipped or failed.
    """
    try:
        img_raw = Image.open(frame.frame_path)
        img_raw.load()
    except OSError as exc:
        logger.warning("Failed to load frame %s: %s", frame.frame_path, exc)
        return False

    original_size = img_raw.size
    image = _resize_image(img_raw, resolution)

    # Parse skeleton data.
    joints: list[JointPosition] | None = None
    if frame.skeleton_path is not None:
        if frame.skeleton_path.suffix == ".json":
            joints = parse_skeleton_json(frame.skeleton_path)
        else:
            joints = parse_skeleton_npy(frame.skeleton_path)

        if joints is not None:
            joints = _scale_joints(joints, original_size, resolution)

    # Load optical flow.
    flow_fwd: np.ndarray | None = None
    flow_bwd: np.ndarray | None = None
    if include_flow:
        if frame.flow_fwd_path is not None:
            flow_fwd = load_flow(frame.flow_fwd_path)
        if frame.flow_bwd_path is not None:
            flow_bwd = load_flow(frame.flow_bwd_path)

    metadata = _build_metadata(
        frame,
        resolution,
        has_joints=joints is not None,
        has_flow=flow_fwd is not None or flow_bwd is not None,
    )

    return _save_frame(
        output_dir,
        frame,
        image,
        joints,
        flow_fwd,
        flow_bwd,
        metadata,
        only_new=only_new,
    )


def convert_sequence(
    seq_dir: Path,
    output_dir: Path,
    split: str,
    sequence_id: str,
    *,
    resolution: int = STRATA_RESOLUTION,
    include_flow: bool = True,
    only_new: bool = False,
    max_frames: int = 0,
) -> AdapterResult:
    """Convert all frames in a sequence to Strata format.

    Args:
        seq_dir: Path to the sequence directory.
        output_dir: Root output directory.
        split: Dataset split name.
        sequence_id: Sequence identifier.
        resolution: Target square resolution.
        include_flow: Whether to include optical flow data.
        only_new: Skip existing output directories.
        max_frames: Maximum frames to convert (0 = unlimited).

    Returns:
        AdapterResult summarizing the conversion.
    """
    result = AdapterResult(sequence_id=sequence_id)
    frames = discover_frames(seq_dir, split, sequence_id)

    if max_frames > 0:
        frames = frames[:max_frames]

    for frame in frames:
        saved = convert_frame(
            frame,
            output_dir,
            resolution=resolution,
            include_flow=include_flow,
            only_new=only_new,
        )
        if saved:
            result.frames_saved += 1
        else:
            result.frames_skipped += 1

    logger.info(
        "LinkTo-Anime seq %s (%s): %d saved, %d skipped",
        sequence_id,
        split,
        result.frames_saved,
        result.frames_skipped,
    )

    return result


def convert_directory(
    linkto_dir: Path,
    output_dir: Path,
    *,
    resolution: int = STRATA_RESOLUTION,
    include_flow: bool = True,
    only_new: bool = False,
    max_frames_per_sequence: int = 0,
    max_sequences: int = 0,
) -> list[AdapterResult]:
    """Convert all LinkTo-Anime sequences to Strata format.

    Args:
        linkto_dir: Root LinkTo-Anime dataset directory.
        output_dir: Root output directory.
        resolution: Target square resolution.
        include_flow: Whether to include optical flow data.
        only_new: Skip existing output directories.
        max_frames_per_sequence: Max frames per sequence (0 = unlimited).
        max_sequences: Max sequences to process (0 = unlimited).

    Returns:
        List of AdapterResult objects for each sequence.
    """
    if not linkto_dir.is_dir():
        logger.error("LinkTo-Anime directory not found: %s", linkto_dir)
        return []

    sequences = discover_sequences(linkto_dir)

    if not sequences:
        logger.warning("No sequences found in %s", linkto_dir)
        return []

    if max_sequences > 0:
        sequences = sequences[:max_sequences]

    logger.info("Found %d sequences in %s", len(sequences), linkto_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[AdapterResult] = []

    for split, seq_id, seq_dir in sequences:
        seq_result = convert_sequence(
            seq_dir,
            output_dir,
            split,
            seq_id,
            resolution=resolution,
            include_flow=include_flow,
            only_new=only_new,
            max_frames=max_frames_per_sequence,
        )
        results.append(seq_result)

    total_saved = sum(r.frames_saved for r in results)
    logger.info(
        "LinkTo-Anime conversion complete: %d sequences, %d frames total",
        len(results),
        total_saved,
    )

    return results
