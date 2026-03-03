"""Convert Meta Animated Drawings dataset to Strata training format.

Handles the Amateur Drawings dataset (178K hand-drawn figures) which uses
COCO annotation format with 17 keypoints, polygon segmentation, and
bounding boxes.

This adapter:

1. Loads the COCO annotations JSON and indexes by image ID.
2. For each annotation: loads the image, crops to bounding box, resizes
   to 512×512 centered on a transparent canvas.
3. Rasterizes COCO polygon segmentation into a binary fg/bg mask.
4. Maps 17 COCO keypoints → Strata's 19-joint skeleton (6 synthetic joints
   interpolated from neighbors).
5. Saves ``image.png``, ``segmentation.png``, ``joints.json``, and
   ``metadata.json`` per example.

Pure Python — no Blender dependency.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from pipeline.config import REGION_NAMES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_NAME = "animated_drawings"

STRATA_RESOLUTION = 512

# Strata region ID → name for the 19 joint regions (IDs 1–19).
_STRATA_ID_TO_NAME: dict[int, str] = {
    rid: name for rid, name in REGION_NAMES.items() if 1 <= rid <= 19
}

# Direct mapping: COCO keypoint index → Strata region ID.
# Only includes COCO keypoints that map 1:1 to a Strata joint.
_COCO_TO_STRATA_DIRECT: dict[int, int] = {
    0: 1,  # nose → head
    5: 6,  # left_shoulder → shoulder_l
    6: 10,  # right_shoulder → shoulder_r
    7: 8,  # left_elbow → forearm_l
    8: 12,  # right_elbow → forearm_r
    9: 9,  # left_wrist → hand_l
    10: 13,  # right_wrist → hand_r
    11: 14,  # left_hip → upper_leg_l
    12: 17,  # right_hip → upper_leg_r
    13: 15,  # left_knee → lower_leg_l
    14: 18,  # right_knee → lower_leg_r
    15: 16,  # left_ankle → foot_l
    16: 19,  # right_ankle → foot_r
}

# Synthetic joints: Strata region ID → (description of how to compute).
# These are interpolated from COCO keypoints.
_SYNTHETIC_REGION_IDS: set[int] = {2, 3, 4, 5, 7, 11}

_MISSING_ANNOTATIONS = ["strata_segmentation", "draw_order"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AdapterResult:
    """Result of converting Animated Drawings images to Strata format."""

    images_processed: int = 0
    images_skipped: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Annotation loading
# ---------------------------------------------------------------------------


def load_annotations(
    annotations_path: Path,
) -> tuple[dict[int, dict], dict[int, dict]]:
    """Load COCO annotations and build lookup indexes.

    Args:
        annotations_path: Path to the COCO JSON file.

    Returns:
        Tuple of (image_id → image_info, image_id → annotation).
    """
    logger.info("Loading annotations from %s ...", annotations_path)
    with open(annotations_path, encoding="utf-8") as f:
        data = json.load(f)

    images_by_id = {img["id"]: img for img in data["images"]}
    anns_by_image_id = {ann["image_id"]: ann for ann in data["annotations"]}

    logger.info(
        "Loaded %d images and %d annotations",
        len(images_by_id),
        len(anns_by_image_id),
    )
    return images_by_id, anns_by_image_id


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------


def _crop_and_resize(
    img: Image.Image,
    bbox: list[float],
    resolution: int = STRATA_RESOLUTION,
    padding_factor: float = 0.1,
) -> tuple[Image.Image, float, float, float]:
    """Crop image to bbox with padding, resize to resolution×resolution.

    Args:
        img: Source image.
        bbox: COCO bbox [x, y, width, height].
        resolution: Target square resolution.
        padding_factor: Fraction of bbox size to add as padding.

    Returns:
        Tuple of (resized RGBA image, scale, offset_x, offset_y) where
        scale/offsets describe the transform from original bbox coords
        to output pixel coords.
    """
    bx, by, bw, bh = bbox

    # Add padding around bbox.
    pad_x = bw * padding_factor
    pad_y = bh * padding_factor
    x0 = max(0, bx - pad_x)
    y0 = max(0, by - pad_y)
    x1 = min(img.width, bx + bw + pad_x)
    y1 = min(img.height, by + bh + pad_y)

    cropped = img.crop((int(x0), int(y0), int(x1), int(y1)))

    # Resize preserving aspect ratio, centered on transparent canvas.
    cw, ch = cropped.size
    scale = resolution / max(cw, ch)
    new_w = round(cw * scale)
    new_h = round(ch * scale)
    resized = cropped.resize((new_w, new_h), Image.LANCZOS)

    if resized.mode != "RGBA":
        resized = resized.convert("RGBA")

    canvas = Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))
    off_x = (resolution - new_w) // 2
    off_y = (resolution - new_h) // 2
    canvas.paste(resized, (off_x, off_y))

    return canvas, scale, x0, y0, off_x, off_y


# ---------------------------------------------------------------------------
# Segmentation mask
# ---------------------------------------------------------------------------


def rasterize_polygon_mask(
    polygons: list[list[float]],
    img_width: int,
    img_height: int,
) -> np.ndarray:
    """Rasterize COCO polygon(s) into a binary mask.

    Args:
        polygons: List of polygon coordinate lists (flat [x,y,x,y,...]).
        img_width: Image width.
        img_height: Image height.

    Returns:
        Binary mask (uint8) of shape (img_height, img_width), 255=fg.
    """
    mask_img = Image.new("L", (img_width, img_height), 0)
    draw = ImageDraw.Draw(mask_img)
    for poly_coords in polygons:
        # Convert flat [x,y,x,y,...] to list of (x,y) tuples.
        xy_pairs = list(zip(poly_coords[0::2], poly_coords[1::2], strict=True))
        if len(xy_pairs) >= 3:
            draw.polygon(xy_pairs, fill=255)
    return np.array(mask_img, dtype=np.uint8)


def _crop_and_resize_mask(
    mask: np.ndarray,
    bbox: list[float],
    resolution: int,
    padding_factor: float,
) -> Image.Image:
    """Crop and resize a binary mask using the same transform as the image.

    Uses nearest-neighbor interpolation to preserve binary values.
    """
    bx, by, bw, bh = bbox
    pad_x = bw * padding_factor
    pad_y = bh * padding_factor
    x0 = max(0, int(bx - pad_x))
    y0 = max(0, int(by - pad_y))
    x1 = min(mask.shape[1], int(bx + bw + pad_x))
    y1 = min(mask.shape[0], int(by + bh + pad_y))

    cropped = mask[y0:y1, x0:x1]
    mask_pil = Image.fromarray(cropped, mode="L")

    cw, ch = mask_pil.size
    scale = resolution / max(cw, ch)
    new_w = round(cw * scale)
    new_h = round(ch * scale)
    resized = mask_pil.resize((new_w, new_h), Image.NEAREST)

    canvas = Image.new("L", (resolution, resolution), 0)
    off_x = (resolution - new_w) // 2
    off_y = (resolution - new_h) // 2
    canvas.paste(resized, (off_x, off_y))

    return canvas


# ---------------------------------------------------------------------------
# Joint mapping
# ---------------------------------------------------------------------------


def _midpoint(
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> tuple[float, float]:
    """Compute midpoint of two 2D points."""
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def map_coco_to_strata_joints(
    keypoints: list[float],
    bbox: list[float],
    scale: float,
    crop_x0: float,
    crop_y0: float,
    canvas_off_x: float,
    canvas_off_y: float,
    resolution: int = STRATA_RESOLUTION,
) -> list[dict[str, Any]]:
    """Map 17 COCO keypoints to 19 Strata joints.

    Args:
        keypoints: Flat COCO keypoints [x, y, vis, x, y, vis, ...] (51 values).
        bbox: COCO bbox [x, y, w, h] (used for coordinate transform).
        scale: Scale factor from crop-and-resize.
        crop_x0: Left edge of the crop in original image coordinates.
        crop_y0: Top edge of the crop in original image coordinates.
        canvas_off_x: X offset of resized image on the 512×512 canvas.
        canvas_off_y: Y offset of resized image on the 512×512 canvas.
        resolution: Target resolution.

    Returns:
        List of 19 joint dicts sorted by Strata region ID (1–19).
    """
    # Parse COCO keypoints into (x, y, visibility) tuples.
    coco_joints: list[tuple[float, float, int]] = []
    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i], keypoints[i + 1], int(keypoints[i + 2])
        coco_joints.append((x, y, v))

    def _transform(x: float, y: float) -> tuple[float, float]:
        """Transform original image coords → 512×512 canvas coords."""
        tx = (x - crop_x0) * scale + canvas_off_x
        ty = (y - crop_y0) * scale + canvas_off_y
        return (round(tx, 2), round(ty, 2))

    # Build transformed COCO joint positions.
    transformed: dict[int, tuple[float, float]] = {}
    for idx, (x, y, vis) in enumerate(coco_joints):
        if vis > 0:
            transformed[idx] = _transform(x, y)

    # Helper to get transformed position or None.
    def _get(coco_idx: int) -> tuple[float, float] | None:
        return transformed.get(coco_idx)

    # Compute all 19 Strata joints.
    strata_joints: dict[int, tuple[float, float, bool]] = {}  # id → (x, y, synthetic)

    # Direct mappings.
    for coco_idx, strata_id in _COCO_TO_STRATA_DIRECT.items():
        pos = _get(coco_idx)
        if pos is not None:
            strata_joints[strata_id] = (pos[0], pos[1], False)

    # Synthetic: neck (2) = midpoint of left_shoulder(5) + right_shoulder(6).
    ls = _get(5)
    rs = _get(6)
    if ls and rs:
        neck = _midpoint(ls, rs)
        strata_joints[2] = (neck[0], neck[1], True)

    # Synthetic: hips (5) = midpoint of left_hip(11) + right_hip(12).
    lh = _get(11)
    rh = _get(12)
    if lh and rh:
        hips = _midpoint(lh, rh)
        strata_joints[5] = (hips[0], hips[1], True)

    # Synthetic: spine (4) = midpoint of neck and hips.
    neck_pos = strata_joints.get(2)
    hips_pos = strata_joints.get(5)
    if neck_pos and hips_pos:
        spine = _midpoint((neck_pos[0], neck_pos[1]), (hips_pos[0], hips_pos[1]))
        strata_joints[4] = (spine[0], spine[1], True)

    # Synthetic: chest (3) = midpoint of neck and spine.
    spine_pos = strata_joints.get(4)
    if neck_pos and spine_pos:
        chest = _midpoint((neck_pos[0], neck_pos[1]), (spine_pos[0], spine_pos[1]))
        strata_joints[3] = (chest[0], chest[1], True)

    # Synthetic: upper_arm_l (7) = midpoint of left_shoulder(5) + left_elbow(7).
    le = _get(7)
    if ls and le:
        upper_arm_l = _midpoint(ls, le)
        strata_joints[7] = (upper_arm_l[0], upper_arm_l[1], True)

    # Synthetic: upper_arm_r (11) = midpoint of right_shoulder(6) + right_elbow(8).
    re = _get(8)
    if rs and re:
        upper_arm_r = _midpoint(rs, re)
        strata_joints[11] = (upper_arm_r[0], upper_arm_r[1], True)

    # Build output list (19 joints, regions 1–19).
    result: list[dict[str, Any]] = []
    for region_id in range(1, 20):
        region_name = _STRATA_ID_TO_NAME[region_id]
        joint_data = strata_joints.get(region_id)

        if joint_data is None:
            result.append(
                {
                    "id": region_id,
                    "name": region_name,
                    "x": 0,
                    "y": 0,
                    "visible": False,
                    "synthetic": region_id in _SYNTHETIC_REGION_IDS,
                }
            )
        else:
            x, y, synthetic = joint_data
            in_bounds = bool(0 <= x < resolution and 0 <= y < resolution)
            result.append(
                {
                    "id": region_id,
                    "name": region_name,
                    "x": x,
                    "y": y,
                    "visible": in_bounds,
                    "synthetic": synthetic,
                }
            )

    return result


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _build_metadata(
    example_id: str,
    image_info: dict,
    annotation: dict,
    resolution: int,
) -> dict[str, Any]:
    """Build Strata metadata dict for a single example."""
    return {
        "id": example_id,
        "source": SOURCE_NAME,
        "source_filename": image_info["file_name"],
        "source_image_id": image_info["id"],
        "resolution": resolution,
        "original_width": image_info["width"],
        "original_height": image_info["height"],
        "bbox": annotation["bbox"],
        "num_coco_keypoints": annotation["num_keypoints"],
        "has_segmentation_mask": False,
        "has_fg_mask": True,
        "has_joints": True,
        "has_draw_order": False,
        "missing_annotations": _MISSING_ANNOTATIONS,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _save_example(
    output_dir: Path,
    example_id: str,
    image: Image.Image,
    mask: Image.Image,
    joints: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    only_new: bool = False,
) -> bool:
    """Save image, mask, joints, and metadata to a per-example directory."""
    example_dir = output_dir / example_id

    if only_new and example_dir.exists():
        return False

    example_dir.mkdir(parents=True, exist_ok=True)

    image.save(example_dir / "image.png", format="PNG", compress_level=6)
    mask.save(example_dir / "segmentation.png", format="PNG", compress_level=6)

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
# Single-image conversion
# ---------------------------------------------------------------------------


def convert_example(
    image_path: Path,
    image_info: dict,
    annotation: dict,
    output_dir: Path,
    *,
    example_id: str,
    resolution: int = STRATA_RESOLUTION,
    padding_factor: float = 0.1,
    only_new: bool = False,
) -> bool:
    """Convert a single Animated Drawings example to Strata format.

    Args:
        image_path: Resolved path to the source PNG.
        image_info: COCO image dict.
        annotation: COCO annotation dict.
        output_dir: Root output directory.
        example_id: Strata-format identifier.
        resolution: Target square resolution.
        padding_factor: Bbox padding fraction.
        only_new: Skip if output already exists.

    Returns:
        True if saved, False if skipped or errored.
    """
    try:
        img = Image.open(image_path)
        img.load()
    except OSError as exc:
        logger.warning("Failed to load image %s: %s", image_path, exc)
        return False

    if img.mode != "RGBA":
        img = img.convert("RGBA")

    bbox = annotation["bbox"]
    keypoints = annotation["keypoints"]

    # Crop and resize image.
    canvas, scale, crop_x0, crop_y0, off_x, off_y = _crop_and_resize(
        img, bbox, resolution, padding_factor
    )

    # Rasterize polygon segmentation and apply same transform.
    full_mask = rasterize_polygon_mask(annotation["segmentation"], img.width, img.height)
    mask_canvas = _crop_and_resize_mask(full_mask, bbox, resolution, padding_factor)

    # Map joints.
    joints = map_coco_to_strata_joints(
        keypoints, bbox, scale, crop_x0, crop_y0, off_x, off_y, resolution
    )

    # Build metadata.
    metadata = _build_metadata(example_id, image_info, annotation, resolution)

    return _save_example(
        output_dir,
        example_id,
        canvas,
        mask_canvas,
        joints,
        metadata,
        only_new=only_new,
    )


# ---------------------------------------------------------------------------
# Directory conversion
# ---------------------------------------------------------------------------


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    annotations_filename: str = "amateur_drawings_annotations.json",
    resolution: int = STRATA_RESOLUTION,
    only_new: bool = False,
    max_images: int = 0,
    random_sample: bool = False,
    seed: int = 42,
) -> AdapterResult:
    """Convert Animated Drawings dataset to Strata format.

    Args:
        input_dir: Root dataset directory containing the annotations JSON
            and ``amateur_drawings/`` image subdirectory.
        output_dir: Output directory for Strata-formatted examples.
        annotations_filename: Name of the COCO annotations JSON file.
        resolution: Target square resolution.
        only_new: Skip existing output directories.
        max_images: Maximum images to process (0 = unlimited).
        random_sample: Randomly sample (requires *max_images* > 0).
        seed: Random seed for sampling.

    Returns:
        :class:`AdapterResult` summarizing the conversion.
    """
    result = AdapterResult()

    annotations_path = input_dir / annotations_filename
    if not annotations_path.is_file():
        logger.error("Annotations file not found: %s", annotations_path)
        result.errors.append(f"Missing annotations: {annotations_path}")
        return result

    images_by_id, anns_by_image_id = load_annotations(annotations_path)

    # Build ordered list of (image_id, image_info, annotation) triples.
    entries = []
    for img_id, img_info in sorted(images_by_id.items()):
        ann = anns_by_image_id.get(img_id)
        if ann is None:
            continue
        entries.append((img_id, img_info, ann))

    if not entries:
        logger.warning("No matching annotations found")
        return result

    # Apply sampling.
    if random_sample and max_images > 0:
        rng = random.Random(seed)
        sample_size = min(max_images, len(entries))
        entries = rng.sample(entries, sample_size)
    elif max_images > 0:
        entries = entries[:max_images]

    total = len(entries)
    logger.info("Processing %d images from %s", total, input_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (img_id, img_info, ann) in enumerate(entries):
        # Resolve image path relative to input_dir.
        image_path = input_dir / img_info["file_name"]
        if not image_path.is_file():
            logger.warning("Image file not found: %s", image_path)
            result.errors.append(f"Missing image: {image_path}")
            result.images_skipped += 1
            continue

        example_id = f"{SOURCE_NAME}_{img_id:06d}"

        saved = convert_example(
            image_path,
            img_info,
            ann,
            output_dir,
            example_id=example_id,
            resolution=resolution,
            only_new=only_new,
        )

        if saved:
            result.images_processed += 1
        else:
            result.images_skipped += 1

        if (i + 1) % 100 == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            logger.info("Processed %d/%d images (%.1f%%)", i + 1, total, pct)

    logger.info(
        "Animated Drawings conversion complete: %d processed, %d skipped, %d errors",
        result.images_processed,
        result.images_skipped,
        len(result.errors),
    )

    return result
