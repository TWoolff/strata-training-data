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

from .config import TARGET_CHARACTER_HEIGHT

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
    azimuth_offset: float = 0.0
    """Extra azimuth rotation (degrees) to compensate for facing direction.

    VRM/glTF characters face +Y (azimuth_offset=180), Mixamo faces -Y (0).
    Added to camera azimuth at render time so "front" always shows the face.
    """


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
        all_corners.extend(mesh_obj.matrix_world @ Vector(corner) for corner in mesh_obj.bound_box)

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
        logger.warning(
            "Character has near-zero height (%.6f) — skipping scale normalization", height
        )
        return

    # --- Scale ---
    scale_factor = TARGET_CHARACTER_HEIGHT / height
    all_objects = [armature, *meshes]

    for obj in all_objects:
        obj.scale *= scale_factor

    # Only apply (bake) transforms on meshes that are NOT skinned to an armature.
    # Applying scale to an armature with skinned children breaks the deformation
    # (bone positions get baked at the new scale, but vertex group weights still
    # reference the original bone-space positions).  Instead, leave the armature
    # scale as an object-level property — Blender evaluates it correctly at render.
    skinned_meshes = {
        m for m in meshes
        if any(mod.type == "ARMATURE" for mod in m.modifiers)
    }
    unskinned = [o for o in all_objects if o not in skinned_meshes and o != armature]
    if unskinned:
        _apply_transforms(unskinned, scale=True)

    # Recompute bounding box after scale
    bbox_min, bbox_max = _combined_bounding_box(meshes)
    center_x = (bbox_min.x + bbox_max.x) / 2
    center_y = (bbox_min.y + bbox_max.y) / 2
    offset = Vector((-center_x, -center_y, -bbox_min.z))

    # --- Translate ---
    for obj in all_objects:
        obj.location += offset

    if unskinned:
        _apply_transforms(unskinned, location=True)


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

    # Filter out stray meshes not parented to any armature (e.g. default Cube).
    # These inflate the bounding box and break camera framing.
    if armatures:
        armature_names = {a.name for a in armatures}
        parented = [m for m in meshes if m.parent and m.parent.name in armature_names]
        if parented:
            stray = len(meshes) - len(parented)
            if stray > 0:
                logger.info(
                    "Filtered %d stray mesh(es) not parented to armature", stray
                )
                for m in meshes:
                    if m not in parented:
                        bpy.data.objects.remove(m, do_unlink=True)
            meshes = parented

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


def transfer_materials_from_glb(
    meshes: list[bpy.types.Object],
    original_glb: Path,
) -> bool:
    """Transfer materials from an original unrigged GLB onto rigged meshes.

    The Meshy rigging API strips textures. This imports the original GLB
    (which has materials + 2048x2048 textures), copies material slots to
    the rigged meshes (same topology/UVs), then removes the original mesh.

    Args:
        meshes: Rigged mesh objects (no materials).
        original_glb: Path to the original unrigged GLB with textures.

    Returns:
        True if materials were transferred, False otherwise.
    """
    if not original_glb.is_file():
        logger.warning("Original GLB not found: %s", original_glb)
        return False

    # Track existing objects
    existing = set(bpy.data.objects)

    try:
        bpy.ops.import_scene.gltf(filepath=str(original_glb))
    except Exception:
        logger.exception("Failed to import original GLB: %s", original_glb)
        return False

    # Find newly imported meshes
    new_objects = [o for o in bpy.data.objects if o not in existing]
    new_meshes = [o for o in new_objects if o.type == "MESH"]

    if not new_meshes:
        logger.warning("No mesh in original GLB: %s", original_glb.name)
        for o in new_objects:
            bpy.data.objects.remove(o, do_unlink=True)
        return False

    # Collect materials from the original mesh
    original_materials = []
    for orig_mesh in new_meshes:
        for slot in orig_mesh.material_slots:
            if slot.material and slot.material not in original_materials:
                original_materials.append(slot.material)

    if not original_materials:
        logger.warning("No materials in original GLB: %s", original_glb.name)
        for o in new_objects:
            bpy.data.objects.remove(o, do_unlink=True)
        return False

    # Transfer materials to rigged meshes
    for rigged_mesh in meshes:
        # Clear existing empty slots
        rigged_mesh.data.materials.clear()
        # Add materials from original
        for mat in original_materials:
            rigged_mesh.data.materials.append(mat)

    # Remove the imported original objects
    for o in new_objects:
        bpy.data.objects.remove(o, do_unlink=True)

    logger.info(
        "Transferred %d material(s) from %s",
        len(original_materials),
        original_glb.name,
    )
    return True
