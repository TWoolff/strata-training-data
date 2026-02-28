"""Extract apparent body part dimensions from 2D segmentation masks.

Reads segmentation masks (8-bit grayscale PNG, pixel value = region ID) and
computes per-region bounding boxes in pixel space.  When paired with ground
truth 3D measurements from ``measurement_ground_truth``, this produces
training pairs for the 2D-to-3D measurement estimation model.

Pure Python with NumPy — no Blender dependency.

PRD reference: Section 13.4 (Multi-View Consistency Pairs).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .config import CAMERA_ANGLES, REGION_NAMES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-image apparent measurement extraction
# ---------------------------------------------------------------------------


def extract_apparent_measurements(
    mask: np.ndarray,
    *,
    camera_angle: str = "front",
) -> dict[str, Any]:
    """Extract per-region apparent dimensions from a segmentation mask.

    For each region present in the mask, computes the axis-aligned bounding
    box in pixel coordinates and records width, height, and pixel count.

    Args:
        mask: 2D uint8 array where each pixel value is a region ID (0-19).
            Region 0 is background and is skipped.
        camera_angle: Name of the camera angle (e.g. ``"front"``,
            ``"side"``).  Must be a key in ``CAMERA_ANGLES``.

    Returns:
        Dict with per-region apparent measurements::

            {
                "camera_angle": "front",
                "azimuth": 0,
                "regions": {
                    "head": {
                        "apparent_width": 48,
                        "apparent_height": 52,
                        "bbox": [x_min, y_min, x_max, y_max],
                        "pixel_count": 1850,
                        "visible": true,
                    },
                    ...
                },
            }

    Raises:
        ValueError: If *camera_angle* is not a recognized angle name.
    """
    if camera_angle not in CAMERA_ANGLES:
        raise ValueError(
            f"Unknown camera angle {camera_angle!r}. Expected one of {list(CAMERA_ANGLES)}"
        )

    azimuth = CAMERA_ANGLES[camera_angle]["azimuth"]
    regions: dict[str, dict[str, Any]] = {}

    for region_id, region_name in REGION_NAMES.items():
        if region_id == 0:
            continue

        pixels = mask == region_id
        pixel_count = int(np.count_nonzero(pixels))

        if pixel_count == 0:
            regions[region_name] = {
                "apparent_width": 0,
                "apparent_height": 0,
                "bbox": None,
                "pixel_count": 0,
                "visible": False,
            }
            continue

        rows, cols = np.where(pixels)
        y_min, y_max = int(rows.min()), int(rows.max())
        x_min, x_max = int(cols.min()), int(cols.max())

        regions[region_name] = {
            "apparent_width": x_max - x_min + 1,
            "apparent_height": y_max - y_min + 1,
            "bbox": [x_min, y_min, x_max, y_max],
            "pixel_count": pixel_count,
            "visible": True,
        }

    return {
        "camera_angle": camera_angle,
        "azimuth": azimuth,
        "regions": regions,
    }


# ---------------------------------------------------------------------------
# Training pair construction
# ---------------------------------------------------------------------------


def build_training_pairs(
    apparent: dict[str, Any],
    ground_truth: dict[str, Any],
    *,
    character_id: str = "",
) -> list[dict[str, Any]]:
    """Combine apparent 2D measurements with ground truth 3D dimensions.

    For each region that is visible in the 2D mask *and* has ground truth
    3D measurements, produces a training pair dict.

    Args:
        apparent: Output of :func:`extract_apparent_measurements`.
        ground_truth: Per-character measurement dict as produced by
            ``measurement_ground_truth.extract_mesh_measurements``.
            Must contain a ``"regions"`` key with per-region dicts
            having ``width``, ``depth``, ``height`` fields.
        character_id: Identifier for the source character.

    Returns:
        List of training pair dicts, one per visible region with ground
        truth data available.
    """
    gt_regions = ground_truth.get("regions", {})
    pairs: list[dict[str, Any]] = []

    for region_name, apparent_data in apparent.get("regions", {}).items():
        if not apparent_data.get("visible", False):
            continue

        gt = gt_regions.get(region_name)
        if gt is None:
            logger.debug(
                "No ground truth for region %r (character %s)",
                region_name,
                character_id,
            )
            continue

        pairs.append(
            {
                "character_id": character_id,
                "region": region_name,
                "camera_angle": apparent["camera_angle"],
                "azimuth": apparent["azimuth"],
                "apparent_width": apparent_data["apparent_width"],
                "apparent_height": apparent_data["apparent_height"],
                "pixel_count": apparent_data["pixel_count"],
                "true_width": gt.get("width", 0.0),
                "true_depth": gt.get("depth", 0.0),
                "true_height": gt.get("height", 0.0),
            }
        )

    return pairs


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def save_apparent_measurements(
    data: dict[str, Any],
    output_dir: Path,
    char_id: str,
    pose_name: str,
    camera_angle: str,
) -> Path:
    """Save per-image apparent measurement data as a JSON file.

    Args:
        data: Output of :func:`extract_apparent_measurements`.
        output_dir: Root dataset directory.
        char_id: Character identifier.
        pose_name: Pose identifier (e.g. ``"pose_01"``).
        camera_angle: Camera angle name (e.g. ``"front"``).

    Returns:
        The output path.
    """
    measurements_dir = output_dir / "measurements_2d"
    measurements_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{char_id}_{pose_name}_{camera_angle}.json"
    path = measurements_dir / filename

    out = {
        "character_id": char_id,
        "pose": pose_name,
        **data,
    }
    path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Saved 2D measurements %s", path)
    return path


def save_training_pairs(
    pairs: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    """Save aggregated training pairs to a JSON file.

    Args:
        pairs: List of training pair dicts (from :func:`build_training_pairs`).
        output_path: Destination file path.

    Returns:
        The output path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": "1.0",
        "pair_count": len(pairs),
        "pairs": pairs,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Saved %d training pairs to %s", len(pairs), output_path)
    return output_path


def load_ground_truth(measurements_dir: Path, char_id: str) -> dict[str, Any] | None:
    """Load ground truth 3D measurements for a character.

    Args:
        measurements_dir: Directory containing per-character measurement
            JSON files (e.g. ``output/segmentation/measurements/``).
        char_id: Character identifier.

    Returns:
        Parsed measurement dict, or ``None`` if the file is missing or
        malformed.
    """
    path = measurements_dir / f"{char_id}.json"
    if not path.is_file():
        logger.warning("No ground truth file for %s at %s", char_id, path)
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read ground truth %s: %s", path, exc)
        return None

    if not isinstance(data, dict) or "regions" not in data:
        logger.warning("Invalid ground truth format in %s", path)
        return None

    return data
