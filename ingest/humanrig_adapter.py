"""Convert HumanRig dataset to Strata training format.

Dataset: jellyczd/HumanRig (CVPR 2025)
Source:  https://huggingface.co/datasets/jellyczd/HumanRig
License: CC-BY-NC-4.0

HumanRig provides 11,434 T-posed humanoid meshes with:
- ``front.png``      — 1024×1024 RGBA render from the front camera
- ``bone_2d.json``   — 22 Mixamo joint names → [x_px, y_px] in 1024px space
- ``bone_3d.json``   — 22 Mixamo joint names → [x, y, z] world coordinates
- ``extrinsic.npy``  — 4×4 camera extrinsic matrix (world-to-camera)
- ``intrinsics.npy`` — 3×3 camera intrinsic matrix
- ``rigged.glb``     — Rigged mesh for optional multi-angle rendering
- ``vertices.json``  — Per-vertex data (not used by this adapter)

Dataset layout::

    humanrig_opensource_final/
    ├── 0/
    │   ├── front.png
    │   ├── bone_2d.json
    │   ├── bone_3d.json
    │   ├── extrinsic.npy
    │   ├── intrinsics.npy
    │   ├── rigged.glb
    │   └── vertices.json
    ├── 1/
    │   └── …
    └── 11433/

This adapter:
1. Discovers all sample directories.
2. For each sample, produces one or more Strata training examples (one per angle).
3. Front view: copies ``front.png`` resized to 512×512; writes ``joints.json``
   by scaling ``bone_2d.json`` from 1024→512px; maps 22 Mixamo joint names →
   Strata's 19-bone standard.
4. Additional angles (3/4, side, back): reprojects ``bone_3d.json`` through a
   rotated camera to produce ``joints.json``; images are **not produced** by
   this pure-Python adapter — use the Blender GLB renderer for rendered images
   at those angles.

Joint mapping (Mixamo bare names → Strata region IDs):

    Head        → 1  (head)
    Neck        → 2  (neck)
    Spine2      → 3  (chest)
    Spine1      → 4  (spine)
    Spine       → 4  (spine)
    Hips        → 5  (hips)
    LeftShoulder → 6  (shoulder_l)
    LeftArm     → 7  (upper_arm_l)
    LeftForeArm → 8  (forearm_l)
    LeftHand    → 9  (hand_l)
    RightShoulder → 10 (shoulder_r)
    RightArm    → 11 (upper_arm_r)
    RightForeArm → 12 (forearm_r)
    RightHand   → 13 (hand_r)
    LeftUpLeg   → 14 (upper_leg_l)
    LeftLeg     → 15 (lower_leg_l)
    LeftFoot    → 16 (foot_l)
    LeftToeBase → 16 (foot_l, merged with foot)
    RightUpLeg  → 17 (upper_leg_r)
    RightLeg    → 18 (lower_leg_r)
    RightFoot   → 19 (foot_r)
    RightToeBase → 19 (foot_r, merged with foot)

The adapter outputs joints in Strata's standard format (list of 19 dicts,
ordered by region ID 1–19, keyed by Strata region name).

This module is pure Python (no Blender dependency) and handles the
front-view image + all-angle joint projections.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HUMANRIG_SOURCE = "humanrig"
STRATA_RESOLUTION = 512
ORIGINAL_RESOLUTION = 1024  # HumanRig renders are 1024×1024

# Camera angles supported for joint reprojection.
# Each entry: (label, azimuth_degrees)
# Azimuth rotates around the Y-axis (vertical): 0°=front, 90°=side, 180°=back.
ANGLE_CONFIGS = [
    ("front", 0),
    ("three_quarter", 45),
    ("side", 90),
    ("back", 180),
]

# HumanRig joint name → Strata region ID (1–19).
# When a joint maps to the same region as another (e.g. LeftToeBase → foot_l),
# it is merged into that region's joint entry (we keep the primary joint,
# i.e. LeftFoot, not LeftToeBase, since that is closer to the ankle).
_HUMANRIG_TO_STRATA_ID: dict[str, int] = {
    "Head": 1,
    "Neck": 2,
    "Spine2": 3,
    "Spine1": 4,
    "Spine": 4,
    "Hips": 5,
    "LeftShoulder": 6,
    "LeftArm": 7,
    "LeftForeArm": 8,
    "LeftHand": 9,
    "RightShoulder": 10,
    "RightArm": 11,
    "RightForeArm": 12,
    "RightHand": 13,
    "LeftUpLeg": 14,
    "LeftLeg": 15,
    "LeftFoot": 16,
    "LeftToeBase": 16,
    "RightUpLeg": 17,
    "RightLeg": 18,
    "RightFoot": 19,
    "RightToeBase": 19,
}

# Strata region ID → canonical region name.
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

# Primary Mixamo joint for each Strata region (used when multiple joints
# map to the same region — we prefer the more anatomically meaningful one).
_REGION_PRIMARY_JOINT: dict[int, str] = {
    3: "Spine2",   # chest: prefer Spine2 over Spine1
    4: "Spine1",   # spine: prefer Spine1 over Spine
    16: "LeftFoot",
    19: "RightFoot",
}

# Annotations not available from HumanRig for non-front views.
_MISSING_ANNOTATIONS_JOINTS_ONLY = [
    "strata_segmentation",
    "draw_order",
    "fg_mask",
    "rendered_image",
]

# Annotations missing from front-view examples (we have image + joints).
_MISSING_ANNOTATIONS_FRONT = [
    "strata_segmentation",
    "draw_order",
    "fg_mask",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class HumanRigEntry:
    """A single HumanRig sample directory."""

    sample_id: int
    sample_dir: Path
    image_path: Path
    bone_2d_path: Path
    bone_3d_path: Path
    extrinsic_path: Path
    intrinsics_path: Path


@dataclass
class AdapterResult:
    """Result of converting HumanRig samples to Strata format."""

    images_processed: int = 0
    images_skipped: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_entries(input_dir: Path) -> list[HumanRigEntry]:
    """Discover HumanRig sample directories.

    Each valid sample directory must contain ``front.png``, ``bone_2d.json``,
    ``bone_3d.json``, ``extrinsic.npy``, and ``intrinsics.npy``.  Directories
    missing any of these files are logged and skipped.

    Args:
        input_dir: Root ``humanrig_opensource_final/`` directory, or any
            parent directory that contains numbered sample subdirectories.

    Returns:
        List of :class:`HumanRigEntry` sorted by sample ID (numeric order).
    """
    if not input_dir.is_dir():
        logger.warning("Input directory not found: %s", input_dir)
        return []

    entries: list[HumanRigEntry] = []
    skipped = 0

    for sample_dir in sorted(input_dir.iterdir(), key=lambda p: _numeric_sort_key(p)):
        if not sample_dir.is_dir():
            continue

        # Accept numeric directory names only.
        if not sample_dir.name.isdigit():
            continue

        required = {
            "front.png": sample_dir / "front.png",
            "bone_2d.json": sample_dir / "bone_2d.json",
            "bone_3d.json": sample_dir / "bone_3d.json",
            "extrinsic.npy": sample_dir / "extrinsic.npy",
            "intrinsics.npy": sample_dir / "intrinsics.npy",
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        if missing:
            logger.warning(
                "Sample %s missing files: %s",
                sample_dir.name,
                ", ".join(missing),
            )
            skipped += 1
            continue

        entries.append(
            HumanRigEntry(
                sample_id=int(sample_dir.name),
                sample_dir=sample_dir,
                image_path=required["front.png"],
                bone_2d_path=required["bone_2d.json"],
                bone_3d_path=required["bone_3d.json"],
                extrinsic_path=required["extrinsic.npy"],
                intrinsics_path=required["intrinsics.npy"],
            )
        )

    if skipped:
        logger.warning("Skipped %d incomplete sample directories", skipped)

    logger.info("Discovered %d HumanRig samples in %s", len(entries), input_dir)
    return entries


def _numeric_sort_key(path: Path) -> int:
    """Return an integer sort key for a directory named with digits."""
    try:
        return int(path.name)
    except ValueError:
        return -1


# ---------------------------------------------------------------------------
# Joint projection
# ---------------------------------------------------------------------------


def _load_bone_3d(bone_3d_path: Path) -> dict[str, list[float]]:
    """Load ``bone_3d.json`` and return joint name → [x, y, z]."""
    with bone_3d_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_bone_2d(bone_2d_path: Path) -> dict[str, list[float]]:
    """Load ``bone_2d.json`` and return joint name → [x_px, y_px]."""
    with bone_2d_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_camera(
    extrinsic_path: Path,
    intrinsics_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load camera matrices.

    Returns:
        Tuple of (extrinsic 4×4 float64, intrinsic 3×3 float64).
    """
    extrinsic = np.load(str(extrinsic_path)).astype(np.float64)
    intrinsic = np.load(str(intrinsics_path)).astype(np.float64)
    return extrinsic, intrinsic


def _rotation_y(angle_deg: float) -> np.ndarray:
    """4×4 homogeneous rotation matrix around the Y-axis (world space).

    Positive angle rotates the world clockwise when viewed from above,
    which is equivalent to rotating the camera counter-clockwise.

    Args:
        angle_deg: Rotation angle in degrees.

    Returns:
        4×4 float64 rotation matrix.
    """
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    return np.array(
        [
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )


def _project_joints(
    bone_3d: dict[str, list[float]],
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    azimuth_deg: float,
    output_resolution: int,
    original_resolution: int = ORIGINAL_RESOLUTION,
) -> dict[str, list[float]]:
    """Project 3D joint positions to 2D screen coordinates.

    Rotates all world-space joint positions by ``azimuth_deg`` degrees around
    the Y-axis, then projects through the (unchanged) camera intrinsics.
    This simulates orbiting the camera around the character.

    The intrinsics are assumed to describe the original image resolution
    (1024×1024).  Output coordinates are scaled to ``output_resolution``.

    Args:
        bone_3d: Mapping of joint name → [x, y, z] world coordinates.
        extrinsic: 4×4 world-to-camera transform (front view).
        intrinsic: 3×3 camera intrinsic matrix.
        azimuth_deg: Horizontal rotation angle in degrees (0=front, 180=back).
        output_resolution: Target pixel resolution (e.g. 512).
        original_resolution: Resolution the intrinsics were calibrated for.

    Returns:
        Mapping of joint name → [x_px, y_px] in output resolution (float).
    """
    scale = output_resolution / original_resolution
    R_world = _rotation_y(azimuth_deg)

    projected: dict[str, list[float]] = {}
    for joint_name, xyz in bone_3d.items():
        # Homogeneous world point, rotated.
        p_world = np.array([*xyz, 1.0], dtype=np.float64)
        p_rotated = R_world @ p_world  # still in world space

        # Apply extrinsic: world → camera.
        p_cam = extrinsic @ p_rotated  # shape (4,)
        x_c, y_c, z_c = p_cam[:3]

        if z_c <= 0:
            # Behind camera — project to edge; still useful as sentinel.
            projected[joint_name] = [0.0, 0.0]
            continue

        # Apply intrinsic: camera → image.
        x_img = (intrinsic[0, 0] * x_c / z_c + intrinsic[0, 2]) * scale
        y_img = (intrinsic[1, 1] * y_c / z_c + intrinsic[1, 2]) * scale
        projected[joint_name] = [round(x_img, 2), round(y_img, 2)]

    return projected


def _build_strata_joints(
    projected: dict[str, list[float]],
    output_resolution: int,
) -> list[dict[str, Any]]:
    """Convert projected joint pixel coords to Strata joints.json format.

    Strata expects a list of 19 joint dicts ordered by region ID (1–19).
    Each dict has keys: ``id``, ``name``, ``x``, ``y``, ``visible``.

    When multiple Mixamo joints map to the same Strata region, the primary
    joint (defined in ``_REGION_PRIMARY_JOINT``) is used.

    Args:
        projected: Joint name → [x_px, y_px] in output resolution.
        output_resolution: Canvas size (used to determine visibility).

    Returns:
        List of 19 joint dicts, one per Strata region, sorted by ID.
    """
    # Collect candidates for each region.
    region_candidates: dict[int, list[tuple[str, list[float]]]] = {}
    for joint_name, xy in projected.items():
        region_id = _HUMANRIG_TO_STRATA_ID.get(joint_name)
        if region_id is None:
            continue
        region_candidates.setdefault(region_id, []).append((joint_name, xy))

    joints: list[dict[str, Any]] = []
    for region_id in range(1, 20):
        region_name = _STRATA_ID_TO_NAME[region_id]
        candidates = region_candidates.get(region_id, [])

        if not candidates:
            # Joint not present in this sample; mark invisible at origin.
            joints.append(
                {
                    "id": region_id,
                    "name": region_name,
                    "x": 0,
                    "y": 0,
                    "visible": False,
                }
            )
            continue

        # Select primary joint if multiple candidates exist.
        primary_name = _REGION_PRIMARY_JOINT.get(region_id)
        xy = None
        if primary_name:
            for name, coords in candidates:
                if name == primary_name:
                    xy = coords
                    break
        if xy is None:
            xy = candidates[0][1]

        x, y = xy
        x_out = float(x)
        y_out = float(y)
        in_bounds = bool(0 <= x_out < output_resolution and 0 <= y_out < output_resolution)
        joints.append(
            {
                "id": region_id,
                "name": region_name,
                "x": round(x_out, 2),
                "y": round(y_out, 2),
                "visible": in_bounds,
            }
        )

    return joints


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _resize_to_strata(
    img: Image.Image,
    resolution: int = STRATA_RESOLUTION,
) -> tuple[Image.Image, int, int]:
    """Resize image to *resolution*×*resolution*, preserving aspect ratio.

    Pads with transparent pixels on all sides to centre the character.

    Args:
        img: Input image (any mode).
        resolution: Target square resolution.

    Returns:
        Tuple of (resized RGBA image, x_offset, y_offset) where offsets
        describe where the image content starts inside the canvas.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    w, h = img.size
    if w == resolution and h == resolution:
        return img, 0, 0

    scale = resolution / max(w, h)
    new_w = round(w * scale)
    new_h = round(h * scale)
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))
    offset_x = (resolution - new_w) // 2
    offset_y = (resolution - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))

    return canvas, offset_x, offset_y


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------


def _build_metadata(
    example_id: str,
    entry: HumanRigEntry,
    angle_label: str,
    azimuth_deg: int,
    resolution: int,
    *,
    has_image: bool,
) -> dict[str, Any]:
    """Build Strata metadata dict for a HumanRig example.

    Args:
        example_id: Strata-format example identifier.
        entry: Parsed sample entry.
        angle_label: Camera angle label (``"front"``, ``"three_quarter"``, etc.)
        azimuth_deg: Azimuth rotation applied (degrees).
        resolution: Output resolution.
        has_image: Whether an ``image.png`` was written for this example.

    Returns:
        Metadata dict ready for JSON serialisation.
    """
    missing = list(
        _MISSING_ANNOTATIONS_FRONT if has_image else _MISSING_ANNOTATIONS_JOINTS_ONLY
    )
    return {
        "id": example_id,
        "source": HUMANRIG_SOURCE,
        "source_sample_id": entry.sample_id,
        "resolution": resolution,
        "original_width": ORIGINAL_RESOLUTION,
        "original_height": ORIGINAL_RESOLUTION,
        "padding_applied": False,  # HumanRig images are already square
        "character": str(entry.sample_id),
        "camera_angle": angle_label,
        "camera_azimuth_deg": azimuth_deg,
        "has_segmentation_mask": False,
        "has_fg_mask": False,
        "has_joints": True,
        "has_draw_order": False,
        "has_rendered_image": has_image,
        "missing_annotations": missing,
    }


# ---------------------------------------------------------------------------
# Per-entry conversion
# ---------------------------------------------------------------------------


def _make_example_id(entry: HumanRigEntry, angle_label: str) -> str:
    """Build a unique Strata example ID.

    Format: ``humanrig_{sample_id:05d}_{angle_label}``
    """
    return f"{HUMANRIG_SOURCE}_{entry.sample_id:05d}_{angle_label}"


def convert_entry(
    entry: HumanRigEntry,
    output_dir: Path,
    *,
    resolution: int = STRATA_RESOLUTION,
    only_new: bool = False,
    angles: list[str] | None = None,
) -> int:
    """Convert a single HumanRig sample to one or more Strata training examples.

    Produces one example per requested camera angle.  Front-view examples
    include ``image.png``; other angles include ``joints.json`` only (no
    rendered image — render GLB via Blender for full multi-angle examples).

    Args:
        entry: HumanRig sample to convert.
        output_dir: Root output directory.
        resolution: Target square resolution.
        only_new: Skip if output directory already exists.
        angles: List of angle labels to produce (e.g. ``["front", "side"]``).
            Defaults to ``["front"]``.

    Returns:
        Number of examples saved.
    """
    if angles is None:
        angles = ["front"]

    angle_map = {label: deg for label, deg in ANGLE_CONFIGS}
    # Validate requested angles.
    invalid = [a for a in angles if a not in angle_map]
    if invalid:
        logger.warning("Unknown angle(s) ignored: %s", invalid)
        angles = [a for a in angles if a in angle_map]

    # Load shared data once.
    try:
        bone_3d = _load_bone_3d(entry.bone_3d_path)
        extrinsic, intrinsic = _load_camera(entry.extrinsic_path, entry.intrinsics_path)
    except Exception as exc:
        logger.warning("Failed to load camera/bone data for sample %d: %s", entry.sample_id, exc)
        return 0

    # Load front image if needed.
    front_img: Image.Image | None = None
    if "front" in angles:
        try:
            front_img = Image.open(entry.image_path)
            front_img.load()
        except OSError as exc:
            logger.warning("Failed to load image for sample %d: %s", entry.sample_id, exc)
            angles = [a for a in angles if a != "front"]

    saved = 0
    for angle_label in angles:
        azimuth_deg = angle_map[angle_label]
        example_id = _make_example_id(entry, angle_label)
        example_dir = output_dir / example_id

        if only_new and example_dir.exists():
            logger.debug("Skipping existing example %s", example_dir)
            continue

        example_dir.mkdir(parents=True, exist_ok=True)
        has_image = angle_label == "front" and front_img is not None

        # Write image (front only).
        if has_image:
            resized, _, _ = _resize_to_strata(front_img, resolution)
            resized.save(example_dir / "image.png", format="PNG", compress_level=6)

        # Project joints for this angle.
        projected = _project_joints(
            bone_3d,
            extrinsic,
            intrinsic,
            azimuth_deg=azimuth_deg,
            output_resolution=resolution,
            original_resolution=ORIGINAL_RESOLUTION,
        )
        strata_joints = _build_strata_joints(projected, resolution)
        joints_path = example_dir / "joints.json"
        joints_path.write_text(
            json.dumps(strata_joints, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # Write metadata.
        metadata = _build_metadata(
            example_id,
            entry,
            angle_label,
            azimuth_deg,
            resolution,
            has_image=has_image,
        )
        meta_path = example_dir / "metadata.json"
        meta_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        saved += 1

    return saved


# ---------------------------------------------------------------------------
# Directory conversion
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
    angles: list[str] | None = None,
) -> AdapterResult:
    """Convert HumanRig samples to Strata training format.

    Args:
        input_dir: Root ``humanrig_opensource_final/`` directory.
        output_dir: Root output directory for Strata-formatted examples.
        resolution: Target image resolution (square).
        only_new: Skip existing output directories.
        max_images: Maximum samples to process (0 = unlimited).
        random_sample: Randomly sample from discovered entries.
        seed: Random seed for reproducible sampling.
        angles: Camera angles to produce. Defaults to all four:
            ``["front", "three_quarter", "side", "back"]``.

    Returns:
        :class:`AdapterResult` summarising the conversion.
    """
    if angles is None:
        angles = ["front", "three_quarter", "side", "back"]

    result = AdapterResult()

    entries = discover_entries(input_dir)
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
    logger.info(
        "Processing %d HumanRig samples × %d angles from %s",
        total,
        len(angles),
        input_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, entry in enumerate(entries):
        try:
            n_saved = convert_entry(
                entry,
                output_dir,
                resolution=resolution,
                only_new=only_new,
                angles=angles,
            )
        except Exception as exc:
            logger.warning("Error processing sample %d: %s", entry.sample_id, exc)
            result.errors.append(f"sample_{entry.sample_id}: {exc}")
            continue

        if n_saved > 0:
            result.images_processed += n_saved
        else:
            result.images_skipped += len(angles)

        if (i + 1) % 100 == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            logger.info(
                "Processed %d/%d samples (%.1f%%) — %d examples saved so far",
                i + 1,
                total,
                pct,
                result.images_processed,
            )

    logger.info(
        "HumanRig conversion complete: %d examples saved, %d skipped, %d errors",
        result.images_processed,
        result.images_skipped,
        len(result.errors),
    )
    return result
