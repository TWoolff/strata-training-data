"""Extract body part dimensions from 3D mesh vertex data.

Computes axis-aligned bounding boxes for each body region by grouping
mesh vertices according to their bone weights, using the same bone-to-label
mapping as the segmentation pipeline.  Measurements are taken in T-pose
(before pose application) so dimensions are consistent across characters.

Output: per-region width (X), depth (Y), height (Z), center, and vertex count.
"""

from __future__ import annotations

import logging
from typing import Any

import bpy  # type: ignore[import-untyped]
from mathutils import Vector  # type: ignore[import-untyped]

from .config import REGION_NAMES, WEIGHT_THRESHOLD, RegionId

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _assign_vertex_to_region(
    vert: bpy.types.MeshVertex,
    group_names: dict[int, str],
    bone_to_region: dict[str, RegionId],
) -> RegionId | None:
    """Find the region for a vertex by its highest-weight bone assignment.

    Args:
        vert: A Blender mesh vertex.
        group_names: Vertex group index → group name lookup.
        bone_to_region: Bone name → region ID mapping.

    Returns:
        The region ID of the highest-weight mapped bone, or None if the
        vertex has no valid bone assignments.
    """
    best_weight = 0.0
    best_region: RegionId | None = None

    for g in vert.groups:
        group_name = group_names.get(g.group)
        if group_name is None:
            continue

        region_id = bone_to_region.get(group_name)
        if region_id is None or region_id == 0:
            continue

        if g.weight > best_weight and g.weight >= WEIGHT_THRESHOLD:
            best_weight = g.weight
            best_region = region_id

    return best_region


def _collect_region_vertices(
    meshes: list[bpy.types.Object],
    bone_to_region: dict[str, RegionId],
) -> dict[RegionId, list[Vector]]:
    """Group world-space vertex positions by their assigned region.

    Args:
        meshes: Character mesh objects (already in T-pose with applied transforms).
        bone_to_region: Bone name → region ID mapping from bone_mapper.

    Returns:
        Dict mapping region ID → list of world-space vertex positions.
    """
    region_verts: dict[RegionId, list[Vector]] = {}

    for mesh_obj in meshes:
        mesh_data = mesh_obj.data
        group_names: dict[int, str] = {
            g.index: g.name for g in mesh_obj.vertex_groups
        }

        for vert in mesh_data.vertices:
            region_id = _assign_vertex_to_region(vert, group_names, bone_to_region)
            if region_id is None:
                continue

            world_pos = mesh_obj.matrix_world @ vert.co
            region_verts.setdefault(region_id, []).append(world_pos)

    return region_verts


def _compute_bounding_box(
    vertices: list[Vector],
) -> dict[str, Any]:
    """Compute an axis-aligned bounding box for a set of vertices.

    Args:
        vertices: List of world-space vertex positions.

    Returns:
        Dict with width (X extent), depth (Y extent), height (Z extent),
        center point, and vertex count.
    """
    xs = [v.x for v in vertices]
    ys = [v.y for v in vertices]
    zs = [v.z for v in vertices]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    return {
        "width": round(max_x - min_x, 6),
        "depth": round(max_y - min_y, 6),
        "height": round(max_z - min_z, 6),
        "center": [
            round((min_x + max_x) / 2.0, 6),
            round((min_y + max_y) / 2.0, 6),
            round((min_z + max_z) / 2.0, 6),
        ],
        "vertex_count": len(vertices),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def extract_mesh_measurements(
    meshes: list[bpy.types.Object],
    bone_to_region: dict[str, RegionId],
) -> dict[str, Any]:
    """Extract per-region body part dimensions from mesh vertex data.

    Groups vertices by their highest-weight bone assignment (same logic as
    segmentation), then computes axis-aligned bounding boxes per region.
    Must be called while the character is in T-pose/A-pose for consistent
    measurements.

    Args:
        meshes: Character mesh objects with vertex groups from skinning.
        bone_to_region: Bone name → region ID mapping from bone_mapper.

    Returns:
        Dict with measurement data::

            {
                "regions": {
                    "head": {"width": 0.3, "depth": 0.25, "height": 0.3,
                             "center": [0.0, 0.0, 1.8], "vertex_count": 500},
                    ...
                },
                "total_vertices": 12345,
                "measured_regions": 17,
            }
    """
    region_verts = _collect_region_vertices(meshes, bone_to_region)

    regions: dict[str, dict[str, Any]] = {}
    total_vertices = 0

    for region_id, vertices in sorted(region_verts.items()):
        region_name = REGION_NAMES.get(region_id)
        if region_name is None:
            continue

        regions[region_name] = _compute_bounding_box(vertices)
        total_vertices += len(vertices)

    logger.info(
        "Extracted measurements for %d regions (%d vertices)",
        len(regions),
        total_vertices,
    )

    return {
        "regions": regions,
        "total_vertices": total_vertices,
        "measured_regions": len(regions),
    }
