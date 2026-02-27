"""Convert Label Studio annotations to the Strata dataset format.

Reads a Label Studio JSON export file and produces:
- 8-bit grayscale segmentation masks (``masks/``)
- Joint position JSON files (``joints/``)
- Per-character source metadata (``sources/``)
- Color images copied into ``images/`` with correct naming

Masks are rasterized from polygon annotations using ``cv2.fillPoly()``.
Joint positions are extracted from keypoint annotations.

Usage::

    python -m annotation.export_annotations \
        --ls_export ./annotation_export.json \
        --image_dir ./data/sprites_resized/ \
        --output_dir ./output/segmentation/ \
        --start_id 1

The ``--start_id`` flag sets the starting number for ``manual_NNN`` IDs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import (
    JOINT_BBOX_PADDING,
    NUM_JOINT_REGIONS,
    REGION_NAME_TO_ID,
    REGION_NAMES,
)
from pipeline.exporter import (
    ensure_output_dirs,
    save_class_map,
    save_joints,
    save_mask,
    save_source_metadata,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------


def _parse_polygon_results(
    results: list[dict],
    img_width: int,
    img_height: int,
) -> list[tuple[int, np.ndarray]]:
    """Extract polygon annotations from Label Studio results.

    Args:
        results: List of annotation result dicts from Label Studio JSON.
        img_width: Original image width in pixels.
        img_height: Original image height in pixels.

    Returns:
        List of (region_id, polygon_points) tuples, where polygon_points
        is an Nx2 int32 array of pixel coordinates.
    """
    polygons: list[tuple[int, np.ndarray]] = []

    for result in results:
        if result.get("type") != "polygonlabels":
            continue

        value = result.get("value", {})
        labels = value.get("polygonlabels", [])
        points = value.get("points", [])

        if not labels or not points:
            continue

        label_name = labels[0]
        region_id = REGION_NAME_TO_ID.get(label_name)
        if region_id is None or region_id == 0:
            logger.warning("Unknown or background label '%s' — skipping polygon", label_name)
            continue

        # Label Studio stores polygon coords as percentages (0–100)
        pixel_points = np.array(
            [[p[0] / 100.0 * img_width, p[1] / 100.0 * img_height] for p in points],
            dtype=np.int32,
        )

        polygons.append((region_id, pixel_points))

    return polygons


def _parse_keypoint_results(
    results: list[dict],
    img_width: int,
    img_height: int,
) -> dict[str, tuple[int, int]]:
    """Extract keypoint annotations from Label Studio results.

    Args:
        results: List of annotation result dicts from Label Studio JSON.
        img_width: Original image width in pixels.
        img_height: Original image height in pixels.

    Returns:
        Dict mapping region name → (x, y) pixel coordinates.
    """
    keypoints: dict[str, tuple[int, int]] = {}

    for result in results:
        if result.get("type") != "keypointlabels":
            continue

        value = result.get("value", {})
        labels = value.get("keypointlabels", [])
        x_pct = value.get("x", 0.0)
        y_pct = value.get("y", 0.0)

        if not labels:
            continue

        label_name = labels[0]
        if label_name not in REGION_NAME_TO_ID:
            logger.warning("Unknown keypoint label '%s' — skipping", label_name)
            continue

        px_x = round(x_pct / 100.0 * img_width)
        px_y = round(y_pct / 100.0 * img_height)

        # Clamp to image bounds
        px_x = max(0, min(px_x, img_width - 1))
        px_y = max(0, min(px_y, img_height - 1))

        keypoints[label_name] = (px_x, px_y)

    return keypoints


# ---------------------------------------------------------------------------
# Mask rasterization
# ---------------------------------------------------------------------------


def rasterize_mask(
    polygons: list[tuple[int, np.ndarray]],
    width: int,
    height: int,
) -> np.ndarray:
    """Rasterize polygon annotations into an 8-bit segmentation mask.

    Polygons are drawn in order — later polygons overwrite earlier ones.
    Background (region 0) is the default for all unlabeled pixels.

    Args:
        polygons: List of (region_id, points) from ``_parse_polygon_results``.
        width: Mask width in pixels.
        height: Mask height in pixels.

    Returns:
        uint8 array of shape (height, width), pixel value = region ID.
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    for region_id, points in polygons:
        cv2.fillPoly(mask, [points], color=int(region_id))

    return mask


# ---------------------------------------------------------------------------
# Joint data construction
# ---------------------------------------------------------------------------


def build_joint_data(
    keypoints: dict[str, tuple[int, int]],
    image_size: tuple[int, int],
) -> dict:
    """Build joint data dict in the standard pipeline format.

    Args:
        keypoints: Region name → (x, y) from keypoint annotations.
        image_size: (width, height) of the image.

    Returns:
        Joint data dict matching the schema from ``joint_extractor.py``.
    """
    joints: dict[str, dict] = {}
    visible_positions: list[tuple[int, int]] = []

    for region_id in range(1, NUM_JOINT_REGIONS + 1):
        region_name = REGION_NAMES[region_id]

        if region_name in keypoints:
            pos = keypoints[region_name]
            joints[region_name] = {
                "position": list(pos),
                "confidence": 1.0,
                "visible": True,
            }
            visible_positions.append(pos)
        else:
            joints[region_name] = {
                "position": [-1, -1],
                "confidence": 0.0,
                "visible": False,
            }

    # Compute bounding box from visible keypoints
    if visible_positions:
        xs = [p[0] for p in visible_positions]
        ys = [p[1] for p in visible_positions]
        pad_x = max(int((max(xs) - min(xs)) * JOINT_BBOX_PADDING), 5)
        pad_y = max(int((max(ys) - min(ys)) * JOINT_BBOX_PADDING), 5)
        bbox = [
            max(0, min(xs) - pad_x),
            max(0, min(ys) - pad_y),
            min(image_size[0], max(xs) + pad_x),
            min(image_size[1], max(ys) + pad_y),
        ]
    else:
        bbox = [0, 0, image_size[0], image_size[1]]

    return {
        "joints": joints,
        "bbox": bbox,
        "image_size": list(image_size),
    }


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------


def export_annotations(
    ls_export_path: Path,
    image_dir: Path,
    output_dir: Path,
    *,
    start_id: int = 1,
    style: str = "flat",
) -> int:
    """Export Label Studio annotations to the Strata dataset format.

    Args:
        ls_export_path: Path to the Label Studio JSON export file.
        image_dir: Directory containing the resized 512×512 source images.
        output_dir: Root dataset output directory.
        start_id: Starting number for ``manual_NNN`` character IDs.
        style: Style name for the exported images (default: ``"flat"``).

    Returns:
        Number of successfully exported annotations.
    """
    export_data = json.loads(ls_export_path.read_text(encoding="utf-8"))

    if not isinstance(export_data, list):
        logger.error("Expected a JSON array of tasks, got %s", type(export_data).__name__)
        return 0

    ensure_output_dirs(output_dir)
    save_class_map(output_dir)

    exported = 0

    for task_index, task in enumerate(export_data):
        char_id = f"manual_{start_id + task_index:03d}"

        # Find the source image
        image_url = task.get("data", {}).get("image", "")
        image_name = Path(image_url).name if image_url else ""

        # Try to find the image in image_dir
        image_path = image_dir / image_name if image_name else None
        if image_path and not image_path.exists():
            # Try matching by stem (without extension)
            candidates = list(image_dir.glob(f"{Path(image_name).stem}.*"))
            image_path = candidates[0] if candidates else None

        if not image_path or not image_path.exists():
            logger.warning(
                "Image not found for task %d ('%s') — skipping",
                task.get("id", task_index),
                image_name,
            )
            continue

        # Load image to get dimensions
        img = Image.open(image_path)
        img_width, img_height = img.size

        # Get the most recent annotation (last in list)
        annotations = task.get("annotations", [])
        if not annotations:
            logger.warning("No annotations for task %d — skipping", task.get("id", task_index))
            continue

        annotation = annotations[-1]
        results = annotation.get("result", [])

        # Parse polygons and keypoints
        polygons = _parse_polygon_results(results, img_width, img_height)
        keypoints = _parse_keypoint_results(results, img_width, img_height)

        if not polygons:
            logger.warning("No polygon annotations for %s — skipping", char_id)
            continue

        # Rasterize mask
        mask_array = rasterize_mask(polygons, img_width, img_height)

        # Build joint data
        joint_data = build_joint_data(keypoints, (img_width, img_height))
        joint_data["character_id"] = char_id
        joint_data["pose_name"] = "static"
        joint_data["source_animation"] = ""
        joint_data["source_frame"] = 0

        # Save outputs using pipeline.exporter
        save_mask(mask_array, output_dir, char_id, pose_index=0)
        save_joints(joint_data, output_dir, char_id, pose_index=0)

        # Copy image to images/ with correct naming
        dest_path = output_dir / "images" / f"{char_id}_pose_00_{style}.png"
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        img.save(dest_path, format="PNG", compress_level=9)

        # Save source metadata
        annotator = annotation.get("completed_by", {})
        annotator_info = ""
        if isinstance(annotator, dict):
            annotator_info = annotator.get("email", str(annotator.get("id", "")))
        elif annotator:
            annotator_info = str(annotator)

        save_source_metadata(
            output_dir,
            char_id,
            source="manual_annotation",
            name=image_path.stem,
            license_="",
            attribution="",
            bone_mapping="manual",
            character_type="humanoid",
            notes=f"Annotated in Label Studio. Annotator: {annotator_info}".strip(),
        )

        exported += 1
        logger.info("Exported %s from task %d", char_id, task.get("id", task_index))

    logger.info("Exported %d / %d annotations to %s", exported, len(export_data), output_dir)
    return exported


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Label Studio annotations to Strata dataset format.",
    )
    parser.add_argument(
        "--ls_export",
        type=Path,
        required=True,
        help="Path to Label Studio JSON export file.",
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        required=True,
        help="Directory containing resized 512x512 source images.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Root dataset output directory.",
    )
    parser.add_argument(
        "--start_id",
        type=int,
        default=1,
        help="Starting number for manual_NNN character IDs (default: 1).",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="flat",
        help="Style name for exported images (default: flat).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    count = export_annotations(
        args.ls_export,
        args.image_dir,
        args.output_dir,
        start_id=args.start_id,
        style=args.style,
    )

    if count == 0:
        logger.error("No annotations exported.")
        sys.exit(1)

    logger.info(
        "Done. Run validation with: python run_validation.py --dataset_dir %s", args.output_dir
    )


if __name__ == "__main__":
    main()
