"""Load FBX characters into Blender and normalize scale/position.

Handles scene cleanup, FBX import, armature/mesh discovery, and
transform normalization so downstream modules see a consistent
coordinate system regardless of the source asset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import bpy  # type: ignore[import-untyped]
from mathutils import Vector  # type: ignore[import-untyped]

from config import TARGET_CHARACTER_HEIGHT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass
class ImportResult:
    """Result of importing a single FBX character."""

    character_id: str
    armature: bpy.types.Object
    meshes: list[bpy.types.Object] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------


def clear_scene() -> None:
    """Remove all objects, meshes, armatures, and materials from the scene."""
    # Deselect, then select all and delete
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Purge orphaned data blocks to prevent memory leaks between imports
    for block_collection in (
        bpy.data.meshes,
        bpy.data.armatures,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.actions,
    ):
        for block in list(block_collection):
            if block.users == 0:
                block_collection.remove(block)


# ---------------------------------------------------------------------------
# Bounding-box utilities
# ---------------------------------------------------------------------------


def _combined_bounding_box(meshes: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    """Compute the world-space axis-aligned bounding box of multiple meshes.

    Args:
        meshes: List of mesh objects (must have at least one).

    Returns:
        (bbox_min, bbox_max) as mathutils.Vector.
    """
    all_corners: list[Vector] = []
    for mesh_obj in meshes:
        all_corners.extend(
            mesh_obj.matrix_world @ Vector(corner)
            for corner in mesh_obj.bound_box
        )

    xs = [v.x for v in all_corners]
    ys = [v.y for v in all_corners]
    zs = [v.z for v in all_corners]

    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _apply_transforms(
    objects: list[bpy.types.Object],
    *,
    location: bool = False,
    scale: bool = False,
) -> None:
    """Select *objects*, apply the specified transforms, then deselect."""
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.transform_apply(location=location, rotation=False, scale=scale)


def _normalize_transforms(
    armature: bpy.types.Object,
    meshes: list[bpy.types.Object],
) -> None:
    """Scale and reposition the character so it fits the standard coordinate space.

    After this function:
    - The combined mesh bounding-box height equals TARGET_CHARACTER_HEIGHT.
    - The character is centered on the XY origin.
    - The feet (bounding-box min Z) sit on Z = 0.
    - All transforms are applied (location/rotation/scale baked into mesh data).
    """
    bbox_min, bbox_max = _combined_bounding_box(meshes)
    height = bbox_max.z - bbox_min.z

    if height < 1e-6:
        logger.warning("Character has near-zero height (%.6f) — skipping scale normalization", height)
        return

    # --- Scale ---
    scale_factor = TARGET_CHARACTER_HEIGHT / height
    all_objects = [armature, *meshes]

    for obj in all_objects:
        obj.scale *= scale_factor

    _apply_transforms(all_objects, scale=True)

    # Recompute bounding box after scale
    bbox_min, bbox_max = _combined_bounding_box(meshes)
    center_x = (bbox_min.x + bbox_max.x) / 2
    center_y = (bbox_min.y + bbox_max.y) / 2
    offset = Vector((-center_x, -center_y, -bbox_min.z))

    # --- Translate ---
    for obj in all_objects:
        obj.location += offset

    _apply_transforms(all_objects, location=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def import_character(fbx_path: Path) -> ImportResult | None:
    """Import an FBX character, normalize it, and return structured references.

    Args:
        fbx_path: Path to the .fbx file.

    Returns:
        An ImportResult with armature/mesh references, or None if the file
        is invalid (no armature or no meshes).
    """
    fbx_path = Path(fbx_path)
    character_id = fbx_path.stem

    if not fbx_path.is_file():
        logger.error("FBX file not found: %s", fbx_path)
        return None

    # Clean slate
    clear_scene()

    # Import FBX
    try:
        bpy.ops.import_scene.fbx(filepath=str(fbx_path))
    except Exception:
        logger.exception("Failed to import FBX: %s", fbx_path)
        return None

    # Discover armature(s) and mesh(es)
    armatures: list[bpy.types.Object] = []
    meshes: list[bpy.types.Object] = []

    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            armatures.append(obj)
        elif obj.type == "MESH":
            meshes.append(obj)

    if not armatures:
        logger.error("No armature found in %s — skipping", fbx_path.name)
        return None

    if not meshes:
        logger.error("No mesh found in %s — skipping", fbx_path.name)
        return None

    if len(armatures) > 1:
        logger.warning(
            "Multiple armatures (%d) in %s — using the first one",
            len(armatures),
            fbx_path.name,
        )

    armature = armatures[0]

    # Normalize scale and position
    _normalize_transforms(armature, meshes)

    logger.info(
        "Imported %s: armature=%s, meshes=%d",
        character_id,
        armature.name,
        len(meshes),
    )

    return ImportResult(
        character_id=character_id,
        armature=armature,
        meshes=meshes,
    )
