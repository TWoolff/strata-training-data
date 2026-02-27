"""Extract per-vertex bone weights from rigged characters for weight prediction.

For each vertex in the character's mesh(es), reads the vertex group assignments,
maps bone names to Strata region names, filters small weights, and projects the
vertex position to 2D screen coordinates.

Output: one JSON file per character (T-pose only) with a per-vertex array of
position + weight entries.
"""

from __future__ import annotations

import logging

import bpy  # type: ignore[import-untyped]
from bpy_extras.object_utils import world_to_camera_view  # type: ignore[import-untyped]

from .config import REGION_NAMES, WEIGHT_THRESHOLD, RegionId

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vertex weight extraction
# ---------------------------------------------------------------------------


def _extract_vertex_weights(
    mesh_obj: bpy.types.Object,
    bone_to_region: dict[str, RegionId],
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
) -> list[dict]:
    """Extract weights and 2D positions for all vertices in a single mesh.

    Args:
        mesh_obj: Blender mesh object with vertex groups from skinning.
        bone_to_region: Bone name → region ID mapping from bone_mapper.
        scene: The Blender scene (for camera projection).
        camera: The active camera object.

    Returns:
        List of per-vertex dicts with ``position`` and ``weights`` keys.
    """
    mesh_data = mesh_obj.data
    res_x = scene.render.resolution_x
    res_y = scene.render.resolution_y

    # Build group index → group name lookup
    group_names: dict[int, str] = {g.index: g.name for g in mesh_obj.vertex_groups}

    vertices: list[dict] = []

    for vert in mesh_data.vertices:
        # --- Project vertex to 2D ---
        world_pos = mesh_obj.matrix_world @ vert.co
        cam_coord = world_to_camera_view(scene, camera, world_pos)

        # Convert to pixel coordinates (Y is bottom-up in Blender, top-down in pixels)
        px_x = round(cam_coord.x * res_x)
        px_y = round((1.0 - cam_coord.y) * res_y)

        # Clamp to image bounds
        px_x = max(0, min(px_x, res_x - 1))
        px_y = max(0, min(px_y, res_y - 1))

        # --- Extract bone weights ---
        region_weights: dict[str, float] = {}

        for g in vert.groups:
            group_name = group_names.get(g.group)
            if group_name is None:
                continue

            region_id = bone_to_region.get(group_name)
            if region_id is None or region_id == 0:
                # Unmapped bone or background — skip
                continue

            region_name = REGION_NAMES.get(region_id)
            if region_name is None:
                continue

            weight = g.weight
            if weight < WEIGHT_THRESHOLD:
                continue

            # Aggregate: multiple bones can map to the same region
            region_weights[region_name] = region_weights.get(region_name, 0.0) + weight

        # Round weights to 4 decimal places for cleaner JSON
        region_weights = {k: round(v, 4) for k, v in region_weights.items()}

        vertices.append(
            {
                "position": [px_x, px_y],
                "weights": region_weights,
            }
        )

    return vertices


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def extract_weights(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    meshes: list[bpy.types.Object],
    bone_to_region: dict[str, RegionId],
) -> dict:
    """Extract per-vertex bone weights for all character meshes.

    Args:
        scene: The Blender scene.
        camera: The active camera object (for 2D projection).
        meshes: Character mesh objects.
        bone_to_region: Bone name → region ID mapping from bone_mapper.

    Returns:
        Dict with weight data matching the schema::

            {
                "vertex_count": 12345,
                "vertices": [
                    {"position": [x, y], "weights": {"region_name": weight}},
                    ...
                ],
                "image_size": [512, 512]
            }
    """
    all_vertices: list[dict] = []

    for mesh_obj in meshes:
        verts = _extract_vertex_weights(mesh_obj, bone_to_region, scene, camera)
        all_vertices.extend(verts)

    # Count vertices with and without weights
    weighted = sum(1 for v in all_vertices if v["weights"])
    empty = len(all_vertices) - weighted

    logger.info(
        "Extracted weights for %d vertices (%d with weights, %d empty)",
        len(all_vertices),
        weighted,
        empty,
    )

    return {
        "vertex_count": len(all_vertices),
        "vertices": all_vertices,
        "image_size": [scene.render.resolution_x, scene.render.resolution_y],
    }
