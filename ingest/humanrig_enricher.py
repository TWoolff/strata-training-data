"""Enrich HumanRig examples with segmentation masks, draw order, and FG masks.

Uses per-vertex skinning weights from ``vertices.json`` to generate 22-class
body region segmentation masks.  Draw order is computed from camera-space
Z-depth of the nearest vertex.  FG masks are extracted from alpha thresholding.

Algorithm:
1. Load ``vertices.json`` — per-vertex 3D coords + 22-bone weight vectors.
2. For each vertex, ``argmax(weights)`` → dominant bone → Strata region ID.
3. Project all vertices to 2D using camera intrinsics/extrinsics, scale to 512.
4. Build KDTree from 2D positions for fast nearest-vertex lookup.
5. Get foreground pixels from ``front.png`` alpha > 128.
6. For each fg pixel, query nearest vertex → assign region ID + Z depth.
7. Save ``segmentation.png``, ``draw_order.png``, ``fg_mask.png``.

Pure Python + NumPy/SciPy/PIL (no Blender dependency).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRATA_RESOLUTION = 512
ORIGINAL_RESOLUTION = 1024

# HumanRig bone name → Strata region ID (same as humanrig_adapter.py).
_BONE_TO_REGION: dict[str, int] = {
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


# ---------------------------------------------------------------------------
# Core enrichment
# ---------------------------------------------------------------------------


def enrich_sample(
    raw_dir: Path,
    output_dir: Path,
    *,
    resolution: int = STRATA_RESOLUTION,
) -> bool:
    """Generate segmentation mask, draw order, and FG mask for one sample.

    Reads raw HumanRig data (vertices.json, intrinsics, extrinsics, front.png)
    and writes annotation files into ``output_dir``.

    Args:
        raw_dir: Raw sample directory containing ``vertices.json``, camera
            matrices, and ``front.png``.
        output_dir: Output example directory (already contains ``image.png``,
            ``joints.json``, ``metadata.json`` from the adapter).
        resolution: Target resolution (default 512).

    Returns:
        True if enrichment succeeded.
    """
    # Load vertices
    vertices_path = raw_dir / "vertices.json"
    if not vertices_path.exists():
        logger.warning("No vertices.json in %s", raw_dir)
        return False

    with vertices_path.open(encoding="utf-8") as fh:
        verts_raw: dict[str, dict[str, Any]] = json.load(fh)

    # Load camera matrices
    intrinsic = np.load(str(raw_dir / "intrinsics.npy")).astype(np.float64)
    extrinsic = np.load(str(raw_dir / "extrinsic.npy")).astype(np.float64)

    # Load front.png for alpha channel
    front_path = raw_dir / "front.png"
    if not front_path.exists():
        logger.warning("No front.png in %s", raw_dir)
        return False

    front_img = Image.open(front_path).convert("RGBA")

    # Get bone names from bone_2d.json to establish weight vector ordering
    bone_2d_path = raw_dir / "bone_2d.json"
    if not bone_2d_path.exists():
        logger.warning("No bone_2d.json in %s", raw_dir)
        return False

    with bone_2d_path.open(encoding="utf-8") as fh:
        bone_2d = json.load(fh)
    bone_names = list(bone_2d.keys())

    # Build bone_index → region_id lookup
    num_bones = len(bone_names)
    bone_index_to_region = np.zeros(num_bones, dtype=np.uint8)
    for i, name in enumerate(bone_names):
        bone_index_to_region[i] = _BONE_TO_REGION.get(name, 0)

    # Parse vertices into arrays
    n_verts = len(verts_raw)
    if n_verts == 0:
        logger.warning("Empty vertices.json in %s", raw_dir)
        return False

    coords_3d = np.zeros((n_verts, 3), dtype=np.float64)
    weights = np.zeros((n_verts, num_bones), dtype=np.float32)
    for i, (_, v) in enumerate(verts_raw.items()):
        coords_3d[i] = v["coord"]
        w = v["weight"]
        weights[i, : len(w)] = w

    # Per-vertex dominant region
    dominant_bone = np.argmax(weights, axis=1)
    vertex_regions = bone_index_to_region[dominant_bone]

    # Project vertices to 2D
    ones = np.ones((n_verts, 1), dtype=np.float64)
    homogeneous = np.hstack([coords_3d, ones])
    cam_coords = (extrinsic @ homogeneous.T).T  # [N, 4]
    x_c = cam_coords[:, 0]
    y_c = cam_coords[:, 1]
    z_c = cam_coords[:, 2]

    # Scale from 1024 intrinsics to target resolution
    scale = resolution / ORIGINAL_RESOLUTION
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    x_px = (fx * x_c / z_c + cx) * scale
    y_px = (fy * y_c / z_c + cy) * scale

    # Build KDTree from projected 2D positions
    points_2d = np.stack([x_px, y_px], axis=1)

    # Filter out NaN/Inf vertices (degenerate projections)
    valid_mask = np.isfinite(points_2d).all(axis=1)
    if valid_mask.sum() < 100:
        logger.warning("Too few valid projected vertices (%d) in %s", valid_mask.sum(), raw_dir)
        return False

    valid_points = points_2d[valid_mask]
    valid_regions = vertex_regions[valid_mask]
    valid_z = z_c[valid_mask]

    tree = KDTree(valid_points)

    # Get foreground pixels from alpha
    alpha = np.array(front_img.resize((resolution, resolution), Image.NEAREST))[:, :, 3]
    fg_yx = np.argwhere(alpha > 128)

    if len(fg_yx) == 0:
        logger.warning("No foreground pixels in %s", raw_dir)
        return False

    fg_xy = fg_yx[:, ::-1].astype(np.float64)

    # Query nearest vertex for each foreground pixel
    _, indices = tree.query(fg_xy)

    # --- Segmentation mask ---
    seg_mask = np.zeros((resolution, resolution), dtype=np.uint8)
    seg_mask[fg_yx[:, 0], fg_yx[:, 1]] = valid_regions[indices]

    # --- Draw order from Z-depth ---
    z_values = valid_z[indices]
    z_min, z_max = z_values.min(), z_values.max()
    if z_max - z_min > 1e-6:
        z_norm = (z_values - z_min) / (z_max - z_min)
    else:
        z_norm = np.full_like(z_values, 0.5)

    # HumanRig z values are negative (camera convention).
    # After normalization: 0=nearest vertex, 1=farthest vertex.
    # Strata convention: 0=back, 255=front → invert.
    draw_order_values = ((1.0 - z_norm) * 255).astype(np.uint8)
    draw_order = np.zeros((resolution, resolution), dtype=np.uint8)
    draw_order[fg_yx[:, 0], fg_yx[:, 1]] = draw_order_values

    # --- FG mask ---
    fg_mask = np.where(alpha > 128, 255, 0).astype(np.uint8)

    # --- Smooth draw order with median filter to reduce KDTree noise ---
    try:
        from scipy.ndimage import median_filter

        # Only smooth within foreground region
        draw_order_smoothed = median_filter(draw_order, size=5)
        # Keep background as 0
        draw_order = np.where(fg_mask > 0, draw_order_smoothed, 0).astype(np.uint8)
    except ImportError:
        pass  # scipy.ndimage not available, use raw

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(seg_mask, mode="L").save(
        output_dir / "segmentation.png", format="PNG", compress_level=6
    )
    Image.fromarray(draw_order, mode="L").save(
        output_dir / "draw_order.png", format="PNG", compress_level=6
    )
    Image.fromarray(fg_mask, mode="L").save(
        output_dir / "fg_mask.png", format="PNG", compress_level=6
    )

    # Update metadata if it exists
    meta_path = output_dir / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["has_segmentation_mask"] = True
            meta["has_draw_order"] = True
            meta["has_fg_mask"] = True
            # Remove enriched items from missing_annotations
            missing = meta.get("missing_annotations", [])
            for item in ["strata_segmentation", "draw_order", "fg_mask"]:
                if item in missing:
                    missing.remove(item)
            meta["missing_annotations"] = missing
            meta_path.write_text(
                json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to update metadata in %s: %s", output_dir, exc)

    return True
