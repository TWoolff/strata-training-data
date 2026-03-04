"""Convert InstaOrder pairwise depth annotations to Strata draw order maps.

Reads the InstaOrder dataset (2.9M pairwise depth/occlusion ordering
annotations on 101K COCO images) and converts pairwise depth orderings
into per-pixel draw order maps compatible with Strata's ``draw_order.png``
format.

Expected input layout::

    input_dir/
    ├── annotations/
    │   ├── InstaOrder_train2017.json   (or InstaOrder_val2017.json)
    │   ├── instances_train2017.json    (COCO instance annotations)
    │   └── instances_val2017.json
    └── images/
        ├── train2017/
        │   └── 000000000001.jpg ...
        └── val2017/
            └── 000000000001.jpg ...

The pairwise depth orderings between instances are resolved into a global
ranking via topological sort, then each instance's COCO mask pixels are
painted with a normalized depth value (0=back, 255=front).

This module is pure Python (no Blender dependency).

License: CC BY-SA
Source: https://github.com/POSTECH-CVLab/InstaOrder
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as maskutils

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INSTAORDER_SOURCE = "instaorder"

STRATA_RESOLUTION = 512

_MISSING_ANNOTATIONS = [
    "strata_segmentation",
    "joints",
    "fg_mask",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AdapterResult:
    """Result of converting InstaOrder examples to Strata format."""

    images_processed: int = 0
    images_skipped: int = 0
    images_cyclic: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Annotation loading
# ---------------------------------------------------------------------------


def load_instaorder_annotations(ann_path: Path) -> dict[int, list[dict]]:
    """Load InstaOrder annotations grouped by image_id.

    Args:
        ann_path: Path to InstaOrder JSON (e.g. ``InstaOrder_train2017.json``).

    Returns:
        Dict mapping image_id → list of annotation dicts. Each annotation
        contains ``instance_ids``, ``depth``, and ``occlusion`` fields.
    """
    data = json.loads(ann_path.read_text(encoding="utf-8"))

    annotations = data.get("annotations", [])
    by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in annotations:
        by_image[ann["image_id"]].append(ann)

    logger.info(
        "Loaded %d InstaOrder annotations for %d images from %s",
        len(annotations),
        len(by_image),
        ann_path.name,
    )
    return dict(by_image)


def load_coco_instances(
    coco_path: Path,
) -> tuple[dict[int, dict], dict[int, list[dict]]]:
    """Load COCO instance annotations.

    Args:
        coco_path: Path to COCO instances JSON (e.g. ``instances_train2017.json``).

    Returns:
        Tuple of (images_by_id, annotations_by_image_id).
        ``images_by_id`` maps image_id → image info dict (with ``file_name``,
        ``width``, ``height``).
        ``annotations_by_image_id`` maps image_id → list of COCO annotation dicts.
    """
    data = json.loads(coco_path.read_text(encoding="utf-8"))

    images_by_id: dict[int, dict] = {}
    for img in data.get("images", []):
        images_by_id[img["id"]] = img

    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    anns_by_id: dict[int, dict] = {}
    for ann in data.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)
        anns_by_id[ann["id"]] = ann

    logger.info(
        "Loaded COCO instances: %d images, %d annotations from %s",
        len(images_by_id),
        len(anns_by_id),
        coco_path.name,
    )
    return images_by_id, anns_by_image


# ---------------------------------------------------------------------------
# Depth ordering resolution
# ---------------------------------------------------------------------------


def parse_depth_pairs(
    instaorder_anns: list[dict],
) -> list[tuple[int, int]]:
    """Extract pairwise depth orderings from InstaOrder annotations.

    Parses the ``depth`` field of each annotation to extract pairs where
    one instance is in front of another.

    Args:
        instaorder_anns: List of InstaOrder annotation dicts for one image.

    Returns:
        List of (front_idx, back_idx) tuples where front_idx is the
        instance closer to the camera. Indices reference positions in
        the ``instance_ids`` array of the annotation.
    """
    pairs: list[tuple[int, int]] = []

    for ann in instaorder_anns:
        instance_ids = ann.get("instance_ids", [])
        depth_list = ann.get("depth", [])

        for depth_entry in depth_list:
            order_str = depth_entry.get("order", "")
            if "<" in order_str:
                parts = order_str.split("<")
                if len(parts) == 2:
                    try:
                        idx1 = int(parts[0].strip())
                        idx2 = int(parts[1].strip())
                    except ValueError:
                        continue
                    # idx1 < idx2 means idx1 is in front (closer to camera)
                    if idx1 < len(instance_ids) and idx2 < len(instance_ids):
                        front_id = instance_ids[idx1]
                        back_id = instance_ids[idx2]
                        pairs.append((front_id, back_id))

    return pairs


def topological_sort_instances(
    instance_ids: list[int],
    depth_pairs: list[tuple[int, int]],
) -> list[int] | None:
    """Sort instances from back to front using pairwise depth orderings.

    Uses Kahn's algorithm for topological sort. Returns None if the
    ordering contains cycles.

    Args:
        instance_ids: All instance IDs present in the image.
        depth_pairs: List of (front_id, back_id) where front is closer
            to camera.

    Returns:
        List of instance IDs sorted back-to-front (index 0 = farthest),
        or None if a cycle is detected.
    """
    if not instance_ids:
        return []

    id_set = set(instance_ids)

    # Build adjacency: back → front (edge from back to front)
    # We want back-to-front order, so edges go from back to front
    graph: dict[int, list[int]] = defaultdict(list)
    in_degree: dict[int, int] = {iid: 0 for iid in id_set}

    for front_id, back_id in depth_pairs:
        if front_id not in id_set or back_id not in id_set:
            continue
        if front_id == back_id:
            continue
        # Edge: back → front (back comes first in sorted order)
        graph[back_id].append(front_id)
        in_degree[front_id] = in_degree.get(front_id, 0) + 1

    # Kahn's algorithm
    queue = deque(iid for iid in id_set if in_degree.get(iid, 0) == 0)
    sorted_ids: list[int] = []

    while queue:
        node = queue.popleft()
        sorted_ids.append(node)
        for neighbor in graph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(sorted_ids) != len(id_set):
        return None  # Cycle detected

    return sorted_ids


def build_draw_order_map(
    sorted_instance_ids: list[int],
    coco_anns_by_id: dict[int, dict],
    height: int,
    width: int,
) -> np.ndarray:
    """Build a per-pixel draw order map from sorted instance masks.

    Args:
        sorted_instance_ids: Instance IDs sorted back-to-front
            (index 0 = farthest from camera).
        coco_anns_by_id: COCO annotation dict keyed by annotation ID.
        height: Image height.
        width: Image width.

    Returns:
        2D uint8 array of shape (height, width) where each pixel's value
        is the normalized depth (0=back, 255=front). Background stays 0.
    """
    draw_order = np.zeros((height, width), dtype=np.uint8)
    n = len(sorted_instance_ids)

    if n == 0:
        return draw_order

    for rank, inst_id in enumerate(sorted_instance_ids):
        ann = coco_anns_by_id.get(inst_id)
        if ann is None:
            continue

        # Decode instance mask
        mask = _decode_instance_mask(ann, height, width)
        if mask is None:
            continue

        # Assign depth value: rank 0 = back (low value), rank n-1 = front (high value)
        if n > 1:
            depth_val = round(rank / (n - 1) * 255)
        else:
            depth_val = 255

        # Ensure at least 1 for foreground instances (0 = background)
        depth_val = max(1, depth_val)

        draw_order[mask > 0] = depth_val

    return draw_order


def _decode_instance_mask(
    ann: dict,
    height: int,
    width: int,
) -> np.ndarray | None:
    """Decode a COCO annotation's segmentation into a binary mask.

    Handles both RLE and polygon formats.

    Args:
        ann: COCO annotation dict with ``segmentation`` field.
        height: Image height.
        width: Image width.

    Returns:
        Binary uint8 mask of shape (height, width), or None on error.
    """
    seg = ann.get("segmentation")
    if seg is None:
        return None

    try:
        if isinstance(seg, dict):
            # RLE format
            counts = seg.get("counts", "")
            if isinstance(counts, str):
                counts = counts.encode("utf-8")
            rle = {"size": seg["size"], "counts": counts}
            return maskutils.decode(rle)
        elif isinstance(seg, list):
            # Polygon format — convert to RLE first
            rles = maskutils.frPyObjects(seg, height, width)
            rle = maskutils.merge(rles)
            return maskutils.decode(rle)
    except Exception as exc:
        logger.debug("Failed to decode mask for annotation %s: %s", ann.get("id"), exc)

    return None


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def _resize_to_strata(
    img: Image.Image,
    resolution: int = STRATA_RESOLUTION,
) -> Image.Image:
    """Resize an image to *resolution*x*resolution*, preserving aspect ratio.

    Args:
        img: Input image (any mode).
        resolution: Target square resolution.

    Returns:
        *resolution*x*resolution* RGBA image.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    w, h = img.size
    if w == resolution and h == resolution:
        return img

    scale = resolution / max(w, h)
    new_w = round(w * scale)
    new_h = round(h * scale)
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))
    offset_x = (resolution - new_w) // 2
    offset_y = (resolution - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))

    return canvas


def _resize_draw_order(
    draw_order: np.ndarray,
    resolution: int = STRATA_RESOLUTION,
) -> Image.Image:
    """Resize a draw order map, using nearest-neighbor.

    Args:
        draw_order: 2D uint8 array.
        resolution: Target square resolution.

    Returns:
        *resolution*x*resolution* grayscale image.
    """
    img = Image.fromarray(draw_order, mode="L")
    w, h = img.size

    scale = resolution / max(w, h)
    new_w = round(w * scale)
    new_h = round(h * scale)
    resized = img.resize((new_w, new_h), Image.NEAREST)

    canvas = Image.new("L", (resolution, resolution), 0)
    offset_x = (resolution - new_w) // 2
    offset_y = (resolution - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))

    return canvas


def _build_metadata(
    image_id: str,
    coco_image_id: int,
    split: str,
    resolution: int,
    *,
    original_size: tuple[int, int],
    n_instances: int,
    n_depth_pairs: int,
) -> dict[str, Any]:
    """Build Strata metadata dict for a single example.

    Args:
        image_id: Strata-format image identifier.
        coco_image_id: Original COCO image ID.
        split: Dataset split (``train`` or ``val``).
        resolution: Output resolution.
        original_size: ``(width, height)`` of the source image.
        n_instances: Number of instances in the draw order map.
        n_depth_pairs: Number of pairwise depth orderings used.

    Returns:
        Metadata dict ready for JSON serialization.
    """
    ow, oh = original_size
    return {
        "id": image_id,
        "source": INSTAORDER_SOURCE,
        "coco_image_id": coco_image_id,
        "split": split,
        "resolution": resolution,
        "original_width": ow,
        "original_height": oh,
        "padding_applied": ow != oh,
        "has_segmentation_mask": False,
        "has_fg_mask": False,
        "has_joints": False,
        "has_draw_order": True,
        "missing_annotations": _MISSING_ANNOTATIONS,
        "n_instances": n_instances,
        "n_depth_pairs": n_depth_pairs,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _save_example(
    output_dir: Path,
    image_id: str,
    image: Image.Image,
    draw_order: Image.Image,
    metadata: dict[str, Any],
    *,
    only_new: bool = False,
) -> bool:
    """Save a single training example in Strata directory format.

    Output layout::

        output_dir/{image_id}/
        ├── image.png
        ├── draw_order.png
        └── metadata.json

    Args:
        output_dir: Root output directory.
        image_id: Example identifier (becomes subdirectory name).
        image: Resized RGBA image.
        draw_order: Resized draw order map.
        metadata: Metadata dict.
        only_new: Skip if output directory already exists.

    Returns:
        True if saved, False if skipped.
    """
    example_dir = output_dir / image_id

    if only_new and example_dir.exists():
        return False

    example_dir.mkdir(parents=True, exist_ok=True)

    image.save(example_dir / "image.png", format="PNG", compress_level=6)
    draw_order.save(example_dir / "draw_order.png", format="PNG", compress_level=6)

    (example_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return True


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def convert_image(
    coco_image_id: int,
    instaorder_anns: list[dict],
    coco_anns: list[dict],
    image_info: dict,
    image_dir: Path,
    output_dir: Path,
    split: str,
    *,
    resolution: int = STRATA_RESOLUTION,
    only_new: bool = False,
) -> bool | None:
    """Convert a single COCO image with InstaOrder annotations to Strata format.

    Args:
        coco_image_id: COCO image ID.
        instaorder_anns: InstaOrder annotations for this image.
        coco_anns: COCO instance annotations for this image.
        image_info: COCO image info dict.
        image_dir: Directory containing COCO images.
        output_dir: Root output directory.
        split: Dataset split name.
        resolution: Target square resolution.
        only_new: Skip if output already exists.

    Returns:
        True if saved, False if skipped/cyclic, None if image not found.
    """
    image_id = f"{INSTAORDER_SOURCE}_{split}_{coco_image_id:012d}"

    # Parse depth pairs
    depth_pairs = parse_depth_pairs(instaorder_anns)
    if not depth_pairs:
        return False

    # Get all instance IDs referenced in orderings
    coco_anns_by_id = {ann["id"]: ann for ann in coco_anns}
    all_instance_ids = list(
        {
            iid
            for ann in instaorder_anns
            for iid in ann.get("instance_ids", [])
            if iid in coco_anns_by_id
        }
    )

    if len(all_instance_ids) < 2:
        return False

    # Topological sort
    sorted_ids = topological_sort_instances(all_instance_ids, depth_pairs)
    if sorted_ids is None:
        return False  # Cyclic

    # Load image
    file_name = image_info.get("file_name", "")
    image_path = image_dir / file_name
    if not image_path.is_file():
        return None

    try:
        img = Image.open(image_path)
        img.load()
    except OSError as exc:
        logger.warning("Failed to load image %s: %s", image_path, exc)
        return None

    height = image_info.get("height", img.height)
    width = image_info.get("width", img.width)
    original_size = (width, height)

    # Build draw order map
    draw_order_map = build_draw_order_map(sorted_ids, coco_anns_by_id, height, width)

    # Check that the map has meaningful content
    if draw_order_map.max() == 0:
        return False

    # Resize
    resized_image = _resize_to_strata(img, resolution)
    resized_draw_order = _resize_draw_order(draw_order_map, resolution)

    metadata = _build_metadata(
        image_id,
        coco_image_id,
        split,
        resolution,
        original_size=original_size,
        n_instances=len(sorted_ids),
        n_depth_pairs=len(depth_pairs),
    )

    return _save_example(
        output_dir,
        image_id,
        resized_image,
        resized_draw_order,
        metadata,
        only_new=only_new,
    )


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    split: str = "val",
    resolution: int = STRATA_RESOLUTION,
    only_new: bool = False,
    max_images: int = 0,
    random_sample: bool = False,
    seed: int = 42,
) -> AdapterResult:
    """Convert InstaOrder annotations + COCO images to Strata draw order format.

    Args:
        input_dir: Root dataset directory containing ``annotations/`` and
            ``images/`` subdirectories.
        output_dir: Root output directory for Strata-formatted examples.
        split: COCO split to process (``train`` or ``val``).
        resolution: Target image resolution (square).
        only_new: Skip existing output directories.
        max_images: Maximum images to process (0 = unlimited).
        random_sample: Randomly sample from available images.
        seed: Random seed for reproducible sampling.

    Returns:
        :class:`AdapterResult` summarizing the conversion.
    """
    result = AdapterResult()

    # Locate annotation files
    ann_dir = input_dir / "annotations"
    if not ann_dir.is_dir():
        logger.error("No annotations/ directory found under %s", input_dir)
        return result

    instaorder_path = ann_dir / f"InstaOrder_{split}2017.json"
    if not instaorder_path.is_file():
        logger.error("InstaOrder annotation file not found: %s", instaorder_path)
        return result

    coco_path = ann_dir / f"instances_{split}2017.json"
    if not coco_path.is_file():
        logger.error("COCO instances file not found: %s", coco_path)
        return result

    # Locate image directory
    image_dir = input_dir / "images" / f"{split}2017"
    if not image_dir.is_dir():
        # Try flat image dir
        image_dir = input_dir / "images"
        if not image_dir.is_dir():
            logger.error("No image directory found under %s", input_dir)
            return result

    # Load annotations
    instaorder_by_image = load_instaorder_annotations(instaorder_path)
    coco_images, coco_anns_by_image = load_coco_instances(coco_path)

    # Get image IDs that have InstaOrder annotations
    image_ids = sorted(instaorder_by_image.keys())

    # Apply sampling / limiting
    if random_sample and max_images > 0:
        rng = random.Random(seed)
        sample_size = min(max_images, len(image_ids))
        image_ids = rng.sample(image_ids, sample_size)
    elif max_images > 0:
        image_ids = image_ids[:max_images]

    total = len(image_ids)
    logger.info(
        "Processing %d images from %s split (images from %s)",
        total,
        split,
        image_dir,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, coco_image_id in enumerate(image_ids):
        image_info = coco_images.get(coco_image_id)
        if image_info is None:
            result.images_skipped += 1
            continue

        instaorder_anns = instaorder_by_image.get(coco_image_id, [])
        coco_anns = coco_anns_by_image.get(coco_image_id, [])

        saved = convert_image(
            coco_image_id,
            instaorder_anns,
            coco_anns,
            image_info,
            image_dir,
            output_dir,
            split,
            resolution=resolution,
            only_new=only_new,
        )

        if saved is True:
            result.images_processed += 1
        elif saved is None:
            result.images_skipped += 1
        elif saved is False:
            # Could be cyclic or insufficient data — count as cyclic for now
            result.images_cyclic += 1

        if (i + 1) % 500 == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            logger.info(
                "Processed %d/%d images (%.1f%%) — %d converted, %d skipped, %d cyclic/insufficient",
                i + 1,
                total,
                pct,
                result.images_processed,
                result.images_skipped,
                result.images_cyclic,
            )

    logger.info(
        "InstaOrder conversion complete: %d processed, %d skipped, %d cyclic, %d errors",
        result.images_processed,
        result.images_skipped,
        result.images_cyclic,
        len(result.errors),
    )

    return result
