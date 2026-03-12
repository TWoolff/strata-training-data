"""Render VRoid CC0 GLB characters with ground-truth segmentation masks.

Imports VRoid Hub CC0 GLB files into Blender, maps VRM J_Bip_* bones to
Strata's 22 body regions, renders color images + segmentation masks + joints
at multiple camera angles. Optionally applies Mixamo FBX poses.

Produces per-example output::

    vroid_cc0_{character}_{pose}_{angle}/
        image.png           <- 512x512 RGBA render
        segmentation.png    <- 8-bit grayscale region IDs (0-21)
        joints.json         <- 2D joint positions with occlusion
        metadata.json       <- Source info, pose, camera angle

This module requires Blender's Python environment (bpy).
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RENDER_RESOLUTION = 512
ORTHO_SCALE = 1.3
CAMERA_DIST = 2.0

ANGLE_AZIMUTHS: dict[str, int] = {
    "front": 0,
    "three_quarter": 45,
    "side": 90,
    "three_quarter_back": 135,
    "back": 180,
}


# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------


def _setup_scene() -> None:
    import bpy

    scene = bpy.context.scene
    # Use EEVEE for fast flat-shaded rendering
    engine = "BLENDER_EEVEE_NEXT" if hasattr(bpy.types, "EEVEE_NEXT") else "BLENDER_EEVEE"
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"

    scene.render.resolution_x = RENDER_RESOLUTION
    scene.render.resolution_y = RENDER_RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"


def _setup_color_render() -> None:
    """Configure scene for color pass rendering."""
    import bpy

    scene = bpy.context.scene
    scene.render.filter_size = 1.5
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.render.image_settings.compression = 15
    scene.render.use_compositing = False
    scene.render.use_sequencer = False

    # Disable AO/bloom if available
    if hasattr(scene.eevee, "use_gtao"):
        scene.eevee.use_gtao = False
    if hasattr(scene.eevee, "use_bloom"):
        scene.eevee.use_bloom = False


def _setup_segmentation_render() -> None:
    """Configure scene for clean segmentation mask pass."""
    import bpy

    scene = bpy.context.scene
    scene.render.filter_size = 0.0  # No AA
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.render.image_settings.compression = 0
    scene.render.use_compositing = False
    scene.render.use_sequencer = False

    if hasattr(scene.eevee, "use_gtao"):
        scene.eevee.use_gtao = False
    if hasattr(scene.eevee, "use_bloom"):
        scene.eevee.use_bloom = False


def _setup_lights() -> None:
    """Set up minimal lighting for color pass."""
    import bpy

    scene = bpy.context.scene

    # Single sun + ambient
    light = bpy.data.lights.new("Sun", "SUN")
    light.energy = 3.0
    light.angle = math.radians(5)
    obj = bpy.data.objects.new("Sun", light)
    scene.collection.objects.link(obj)
    obj.rotation_euler = (math.radians(45), 0, math.radians(30))

    # Ambient via world
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Strength"].default_value = 0.7


def _compute_character_bounds(meshes: list[Any]) -> tuple[Any, Any]:
    """Compute world-space bounding box center and size from all meshes."""
    from mathutils import Vector

    all_min = Vector((float("inf"), float("inf"), float("inf")))
    all_max = Vector((float("-inf"), float("-inf"), float("-inf")))

    for mesh_obj in meshes:
        for corner in mesh_obj.bound_box:
            world_co = mesh_obj.matrix_world @ Vector(corner)
            all_min.x = min(all_min.x, world_co.x)
            all_min.y = min(all_min.y, world_co.y)
            all_min.z = min(all_min.z, world_co.z)
            all_max.x = max(all_max.x, world_co.x)
            all_max.y = max(all_max.y, world_co.y)
            all_max.z = max(all_max.z, world_co.z)

    center = (all_min + all_max) / 2
    size = all_max - all_min
    return center, size


def _make_camera(meshes: list[Any]) -> Any:
    """Create ortho camera auto-framed to fit the character meshes."""
    import bpy

    center, size = _compute_character_bounds(meshes)

    # Ortho scale = largest dimension + 10% padding
    ortho_scale = max(size.x, size.z) * 1.1

    scene = bpy.context.scene
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = ortho_scale
    cam_obj = bpy.data.objects.new("Cam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    # Store center Z for orbit (camera orbits at character center height)
    cam_obj["char_center_z"] = center.z

    logger.info(
        "Auto-framed camera: center=(%.2f, %.2f, %.2f), size=(%.2f, %.2f, %.2f), ortho=%.2f",
        center.x, center.y, center.z, size.x, size.y, size.z, ortho_scale,
    )
    return cam_obj


def _set_camera_orbit(cam_obj: Any, azimuth_deg: float) -> None:
    from mathutils import Matrix, Vector

    center_z = cam_obj.get("char_center_z", 0.0)
    theta = math.radians(azimuth_deg)
    # VRoid/glTF characters face -Y in Blender, so front camera is at +Y
    cam_x = -math.sin(theta) * CAMERA_DIST
    cam_y = math.cos(theta) * CAMERA_DIST
    cam_obj.location = Vector((cam_x, cam_y, center_z))

    forward = (-Vector((cam_x, cam_y, 0.0))).normalized()
    world_up = Vector((0.0, 0.0, 1.0))
    right = forward.cross(world_up).normalized()
    up = right.cross(forward).normalized()

    rot_mat = Matrix([
        [right.x, right.y, right.z, 0.0],
        [up.x, up.y, up.z, 0.0],
        [-forward.x, -forward.y, -forward.z, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]).transposed()
    cam_obj.rotation_euler = rot_mat.to_3x3().to_euler()


# ---------------------------------------------------------------------------
# GLB import
# ---------------------------------------------------------------------------


def _import_glb(glb_path: Path) -> tuple[Any, list[Any]]:
    """Import a VRoid GLB file. Returns (armature, meshes)."""
    import bpy

    try:
        bpy.ops.import_scene.gltf(filepath=str(glb_path))
    except Exception as exc:
        logger.warning("GLB import failed for %s: %s", glb_path, exc)
        return None, []

    # Remove stray objects (Icosphere, etc.)
    for obj in list(bpy.data.objects):
        if obj.name == "Icosphere":
            bpy.data.objects.remove(obj, do_unlink=True)

    armature = None
    meshes = []
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            armature = obj
        elif obj.type == "MESH":
            meshes.append(obj)

    if armature:
        bone_names = [b.name for b in armature.data.bones]
        j_bip_count = sum(1 for n in bone_names if n.startswith("J_Bip_"))
        logger.info(
            "Imported %s: armature=%s, %d meshes, %d bones (%d J_Bip_*)",
            glb_path.name, armature.name, len(meshes), len(bone_names), j_bip_count,
        )

    return armature, meshes


def _store_original_materials(meshes: list[Any]) -> dict[str, list]:
    """Store original material assignments so we can restore after seg pass."""
    import bpy

    stored = {}
    for mesh_obj in meshes:
        mat_list = []
        for slot in mesh_obj.material_slots:
            mat_list.append(slot.material)
        # Also store per-face material indices
        face_indices = [p.material_index for p in mesh_obj.data.polygons]
        stored[mesh_obj.name] = {"materials": mat_list, "face_indices": face_indices}
    return stored


def _restore_original_materials(meshes: list[Any], stored: dict[str, list]) -> None:
    """Restore original materials after segmentation pass."""
    for mesh_obj in meshes:
        if mesh_obj.name not in stored:
            continue
        info = stored[mesh_obj.name]
        mesh_obj.data.materials.clear()
        for mat in info["materials"]:
            mesh_obj.data.materials.append(mat)
        for polygon, mat_idx in zip(mesh_obj.data.polygons, info["face_indices"]):
            polygon.material_index = mat_idx


# ---------------------------------------------------------------------------
# Per-character rendering
# ---------------------------------------------------------------------------


def render_vroid_character(
    glb_path: Path,
    output_dir: Path,
    angles: list[str],
    *,
    pose_dir: Path | None = None,
    only_new: bool = False,
) -> tuple[int, int]:
    """Render one VRoid CC0 character with seg masks + joints.

    Returns:
        (rendered_count, skipped_count)
    """
    import bpy

    from pipeline.bone_mapper import map_bones
    from pipeline.joint_extractor import extract_joints
    from pipeline.renderer import (
        assign_region_materials,
        convert_rgb_to_grayscale_mask,
        create_region_materials,
    )

    # Reset scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    _setup_scene()
    _setup_lights()

    armature, meshes = _import_glb(glb_path)
    if armature is None:
        logger.warning("No armature in %s — skipping", glb_path)
        return 0, 0

    if not meshes:
        logger.warning("No meshes in %s — skipping", glb_path)
        return 0, 0

    character_id = glb_path.stem  # e.g. "vroid_character_name"

    # Map bones using the full pipeline (handles J_Bip_* via COMMON_BONE_ALIASES)
    mapping = map_bones(armature, meshes, character_id)
    bone_to_region = mapping.bone_to_region
    vertex_to_region = mapping.vertex_to_region

    logger.info(
        "Bone mapping for %s: %d mapped, %d unmapped",
        character_id, len(bone_to_region), len(mapping.unmapped_bones),
    )
    if mapping.unmapped_bones:
        logger.info("Unmapped: %s", mapping.unmapped_bones[:20])

    # Store original materials for restoring after seg pass
    original_materials = _store_original_materials(meshes)

    # Create region materials for seg pass
    region_materials = create_region_materials()

    cam_obj = _make_camera(meshes)
    scene = bpy.context.scene

    # Build poses list
    poses = _build_pose_list(armature, pose_dir)

    rendered = 0
    skipped = 0

    for pose_info in poses:
        pose_name = pose_info["name"]

        # Apply pose if not T-pose
        if pose_info.get("apply"):
            from pipeline.pose_applicator import apply_pose, reset_pose
            ok = apply_pose(armature, pose_info["pose_obj"], pose_info["pose_dir"])
            if not ok:
                logger.debug("Failed to apply pose %s", pose_name)
                skipped += len(angles)
                reset_pose(armature)
                continue

        bpy.context.view_layer.update()

        for label in angles:
            example_id = f"vroid_cc0_{character_id}_{pose_name}_{label}"
            example_dir = output_dir / example_id

            if only_new and (example_dir / "image.png").exists():
                skipped += 1
                continue

            example_dir.mkdir(parents=True, exist_ok=True)

            azimuth = ANGLE_AZIMUTHS[label]
            _set_camera_orbit(cam_obj, azimuth)
            bpy.context.view_layer.update()

            # --- Color pass ---
            _restore_original_materials(meshes, original_materials)
            _setup_color_render()

            img_path = example_dir / "image.png"
            scene.render.filepath = str(img_path)
            try:
                bpy.ops.render.render(write_still=True)
            except Exception as exc:
                logger.warning("Color render failed for %s: %s", example_id, exc)
                continue

            # --- Segmentation pass ---
            for mesh_idx, mesh_obj in enumerate(meshes):
                assign_region_materials(
                    mesh_obj, mesh_idx, vertex_to_region, region_materials
                )
            _setup_segmentation_render()

            seg_rgb_path = example_dir / "segmentation_rgb.png"
            scene.render.filepath = str(seg_rgb_path)
            try:
                bpy.ops.render.render(write_still=True)
            except Exception as exc:
                logger.warning("Seg render failed for %s: %s", example_id, exc)
                continue

            # Convert RGB seg → grayscale region IDs
            seg_path = example_dir / "segmentation.png"
            convert_rgb_to_grayscale_mask(seg_rgb_path, seg_path)

            # Clean up intermediate RGB seg
            try:
                seg_rgb_path.unlink()
            except OSError:
                pass

            # --- Joints ---
            # Re-apply original materials and update for joint extraction
            _restore_original_materials(meshes, original_materials)
            bpy.context.view_layer.update()

            joints_data = extract_joints(
                scene, cam_obj, armature, meshes, bone_to_region
            )

            joints_path = example_dir / "joints.json"
            joints_path.write_text(
                json.dumps(joints_data, indent=2) + "\n", encoding="utf-8"
            )

            # --- Metadata ---
            meta = {
                "source": "vroid_cc0",
                "character_id": character_id,
                "source_file": glb_path.name,
                "pose_name": pose_name,
                "camera_angle": label,
                "camera_azimuth_deg": azimuth,
                "render_resolution": RENDER_RESOLUTION,
                "has_rendered_image": True,
                "has_segmentation_mask": True,
                "has_joints": True,
                "joint_source": "blender_raycast",
                "bone_mapping_stats": {
                    "mapped": len(bone_to_region),
                    "unmapped": len(mapping.unmapped_bones),
                    "exact": mapping.mapping_stats.exact,
                    "alias": mapping.mapping_stats.alias,
                    "prefix": mapping.mapping_stats.prefix,
                    "substring": mapping.mapping_stats.substring,
                    "fuzzy": mapping.mapping_stats.fuzzy,
                },
            }
            meta_path = example_dir / "metadata.json"
            meta_path.write_text(
                json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            rendered += 1

        # Reset pose for next iteration
        if pose_info.get("apply"):
            from pipeline.pose_applicator import reset_pose
            reset_pose(armature)

    return rendered, skipped


def _build_pose_list(
    armature: Any,
    pose_dir: Path | None,
) -> list[dict]:
    """Build list of poses to render.

    Always includes T-pose. If pose_dir is provided, adds Mixamo FBX poses.
    """
    poses = [{"name": "tpose", "apply": False}]

    if pose_dir is not None and pose_dir.is_dir():
        from pipeline.pose_applicator import list_poses

        all_poses = list_poses(pose_dir, keyframes_per_clip=3)
        for pose in all_poses:
            poses.append({
                "name": pose.name,
                "apply": True,
                "pose_obj": pose,
                "pose_dir": pose_dir,
            })
        logger.info("Added %d poses from %s", len(all_poses), pose_dir)

    return poses


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def render_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    pose_dir: Path | None = None,
    angles: list[str] | None = None,
    only_new: bool = False,
    max_characters: int = 0,
) -> tuple[int, int, int]:
    """Render all VRoid CC0 GLB files in a directory.

    Args:
        input_dir: Directory containing .glb files.
        output_dir: Root output directory.
        pose_dir: Optional directory with Mixamo FBX/BVH poses.
        angles: Camera angle labels. Defaults to front + three_quarter + side.
        only_new: Skip examples that already exist.
        max_characters: Max characters to process (0 = all).

    Returns:
        (rendered_total, skipped_total, error_count)
    """
    if angles is None:
        angles = ["front", "three_quarter", "side"]

    # Discover GLB files (filter macOS resource fork ._ files)
    glb_files = sorted(
        f for f in input_dir.glob("*.glb") if not f.name.startswith("._")
    )
    if not glb_files:
        glb_files = sorted(
            f for f in input_dir.rglob("*.glb") if not f.name.startswith("._")
        )

    if max_characters > 0:
        glb_files = glb_files[:max_characters]

    total = len(glb_files)
    logger.info("Found %d GLB files in %s", total, input_dir)

    rendered_total = 0
    skipped_total = 0
    errors = 0
    start_time = time.monotonic()

    for i, glb_path in enumerate(glb_files):
        try:
            rendered, skipped = render_vroid_character(
                glb_path,
                output_dir,
                angles,
                pose_dir=pose_dir,
                only_new=only_new,
            )
        except Exception as exc:
            logger.warning("Error on %s: %s", glb_path.name, exc)
            errors += 1
            continue

        rendered_total += rendered
        skipped_total += skipped

        if (i + 1) % 5 == 0 or (i + 1) == total:
            elapsed = time.monotonic() - start_time
            rate = rendered_total / elapsed if elapsed > 0 else 0
            pct = (i + 1) / total * 100
            logger.info(
                "Progress: %d/%d chars (%.1f%%) — %d rendered, %d skipped, "
                "%d errors, %.1f img/s",
                i + 1, total, pct, rendered_total, skipped_total, errors, rate,
            )

    return rendered_total, skipped_total, errors
