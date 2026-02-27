"""Compute per-region Z-depth and produce draw order maps.

For each body region, gathers all vertices belonging to bones mapped to that
region, projects them to camera space, and averages their Z-depth.  Depths are
normalized to [0, 255] (0 = farthest, 255 = nearest) and painted onto the
segmentation mask so each pixel carries its region's depth value.

Output: one 8-bit single-channel grayscale PNG per pose (same naming as masks).
"""

from __future__ import annotations

import logging

import bpy  # type: ignore[import-untyped]
import numpy as np
from bpy_extras.object_utils import world_to_camera_view  # type: ignore[import-untyped]

from .config import NUM_JOINT_REGIONS, REGION_NAMES, RegionId

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-region depth computation
# ---------------------------------------------------------------------------


def _compute_region_depths(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    meshes: list[bpy.types.Object],
    bone_to_region: dict[str, RegionId],
) -> dict[RegionId, float]:
    """Compute mean Z-depth in camera space for each body region.

    For each mesh vertex, finds its dominant bone (highest-weight vertex
    group), looks up the region, and accumulates the vertex's camera-space
    Z coordinate.  Returns the mean depth per region.

    Args:
        scene: The Blender scene.
        camera: The active camera object.
        meshes: Character mesh objects.
        bone_to_region: Bone name → region ID mapping from bone_mapper.

    Returns:
        Dict of region_id → mean Z-depth (camera space, higher = closer).
    """
    # Accumulate depths per region
    region_z_sum: dict[RegionId, float] = {}
    region_z_count: dict[RegionId, int] = {}

    for mesh_obj in meshes:
        mesh_data = mesh_obj.data
        group_names: dict[int, str] = {
            g.index: g.name for g in mesh_obj.vertex_groups
        }

        for vert in mesh_data.vertices:
            if not vert.groups:
                continue

            # Find dominant bone (highest weight)
            best_weight = -1.0
            best_group_name = ""
            for g in vert.groups:
                if g.weight > best_weight:
                    best_weight = g.weight
                    best_group_name = group_names.get(g.group, "")

            region_id = bone_to_region.get(best_group_name, 0)
            if region_id == 0:
                continue

            # Project vertex to camera space
            world_pos = mesh_obj.matrix_world @ vert.co
            cam_coord = world_to_camera_view(scene, camera, world_pos)
            z_depth = cam_coord.z

            region_z_sum[region_id] = region_z_sum.get(region_id, 0.0) + z_depth
            region_z_count[region_id] = region_z_count.get(region_id, 0) + 1

    return {
        rid: region_z_sum[rid] / region_z_count[rid]
        for rid in region_z_sum
    }


def _normalize_depths(
    region_depths: dict[RegionId, float],
) -> dict[RegionId, int]:
    """Normalize raw Z-depths to [0, 255] range.

    The camera-space Z increases with distance from the camera, so a
    higher Z means farther away.  We invert so that:
    - 0 = farthest from camera (back)
    - 255 = nearest to camera (front)

    Args:
        region_depths: Region ID → mean Z-depth (camera space).

    Returns:
        Region ID → normalized depth value (0–255).
    """
    if not region_depths:
        return {}

    depths = list(region_depths.values())
    z_min = min(depths)
    z_max = max(depths)
    z_range = z_max - z_min

    normalized: dict[RegionId, int] = {}
    for region_id, z in region_depths.items():
        if z_range > 0:
            # Invert: higher Z = farther from camera → lower value
            normalized[region_id] = round((1.0 - (z - z_min) / z_range) * 255)
        else:
            normalized[region_id] = 127

    return normalized


# ---------------------------------------------------------------------------
# Draw order map generation
# ---------------------------------------------------------------------------


def extract_draw_order(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    armature: bpy.types.Object,
    meshes: list[bpy.types.Object],
    bone_to_region: dict[str, RegionId],
    segmentation_mask: np.ndarray,
) -> dict:
    """Compute draw order map from vertex depths and segmentation mask.

    For each body region (1–19), computes the mean Z-depth of all
    vertices in that region, normalizes to [0, 255], and paints each
    pixel of the segmentation mask with its region's normalized depth.

    Args:
        scene: The Blender scene.
        camera: The active camera object.
        armature: The character's armature object (unused, kept for API
            consistency with other extractors).
        meshes: Character mesh objects.
        bone_to_region: Bone name → region ID mapping from bone_mapper.
        segmentation_mask: 2D uint8 array where each pixel value is a
            region ID (0–19).

    Returns:
        Dict with draw order data::

            {
                "draw_order_map": np.ndarray (H, W) uint8,
                "region_depths": {"region_name": normalized_depth, ...},
                "image_size": [width, height]
            }
    """
    region_depths = _compute_region_depths(scene, camera, meshes, bone_to_region)
    normalized = _normalize_depths(region_depths)

    # Build the draw order map: replace each pixel's region ID with depth
    draw_order_map = np.zeros_like(segmentation_mask, dtype=np.uint8)

    for region_id, depth_val in normalized.items():
        draw_order_map[segmentation_mask == region_id] = depth_val

    # Background (region 0) stays at 0

    region_depths_named: dict[str, int] = {
        REGION_NAMES[rid]: normalized.get(rid, 0)
        for rid in range(1, NUM_JOINT_REGIONS + 1)
    }

    h, w = segmentation_mask.shape
    logger.info(
        "Draw order extracted: %d regions with depth data, map size %dx%d",
        len(normalized),
        w,
        h,
    )

    return {
        "draw_order_map": draw_order_map,
        "region_depths": region_depths_named,
        "image_size": [w, h],
    }
