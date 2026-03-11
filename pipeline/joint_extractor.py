"""Project bone head positions to 2D screen coordinates for joint prediction.

For each body region, selects a primary bone from the armature, transforms
its head position to world space, projects it through the orthographic camera,
and outputs pixel coordinates plus visibility flags.

Output: one JSON file per pose with 19 joints (one per body region, excluding
background).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import bpy  # type: ignore[import-untyped]
from bpy_extras.object_utils import world_to_camera_view  # type: ignore[import-untyped]
from mathutils import Vector  # type: ignore[import-untyped]

from .config import (
    JOINT_BBOX_PADDING,
    NUM_JOINT_REGIONS,
    PRIMARY_BONE_KEYWORDS,
    REGION_NAMES,
    RegionId,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primary bone selection
# ---------------------------------------------------------------------------


def _select_primary_bone(
    region_id: RegionId,
    bones_in_region: list[str],
) -> str | None:
    """Choose the best bone for a region's joint position.

    Uses PRIMARY_BONE_KEYWORDS to prefer semantically central bones (e.g.,
    "LeftHand" over "LeftHandIndex2" for the hand_l region).

    Args:
        region_id: The Strata region ID (1–19).
        bones_in_region: All bone names mapped to this region.

    Returns:
        The chosen bone name, or None if bones_in_region is empty.
    """
    if not bones_in_region:
        return None

    keywords = PRIMARY_BONE_KEYWORDS.get(region_id, [])

    for keyword in keywords:
        kw_lower = keyword.lower()
        for bone_name in bones_in_region:
            if kw_lower in bone_name.lower():
                return bone_name

    # Fallback: return the first bone (deterministic ordering)
    return bones_in_region[0]


# ---------------------------------------------------------------------------
# 3D → 2D projection
# ---------------------------------------------------------------------------


def _project_bone_to_2d(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    armature: bpy.types.Object,
    bone_name: str,
) -> tuple[tuple[int, int], float]:
    """Project a posed bone's head position to 2D pixel coordinates.

    Args:
        scene: The Blender scene.
        camera: The active camera object.
        armature: The armature containing the bone.
        bone_name: Name of the bone to project.

    Returns:
        ((px_x, px_y), depth) where px_x/px_y are integer pixel coords
        and depth is the bone's distance from the camera (for occlusion).
    """
    pose_bone = armature.pose.bones[bone_name]
    world_pos = armature.matrix_world @ pose_bone.head

    # world_to_camera_view returns (x, y, z) in normalized camera space
    # x, y are in [0, 1] range; z is depth from camera
    cam_coord = world_to_camera_view(scene, camera, world_pos)

    res_x = scene.render.resolution_x
    res_y = scene.render.resolution_y

    # Convert to pixel coordinates (Blender's Y is bottom-up, pixels are top-down)
    px_x = round(cam_coord.x * res_x)
    px_y = round((1.0 - cam_coord.y) * res_y)

    return (px_x, px_y), cam_coord.z


# ---------------------------------------------------------------------------
# Occlusion detection
# ---------------------------------------------------------------------------


def _check_occlusion(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    armature: bpy.types.Object,
    bone_name: str,
    meshes: list[bpy.types.Object],
    depth: float,
) -> bool:
    """Check if a joint is occluded by mesh geometry.

    Casts a ray from the camera through the joint's 3D position. If the
    ray hits mesh geometry at a depth closer than the bone, the joint is
    considered occluded.

    Args:
        scene: The Blender scene.
        camera: The active camera object.
        armature: The armature containing the bone.
        bone_name: Name of the bone to check.
        meshes: Character mesh objects that could occlude the joint.
        depth: The bone's depth from camera (from projection).

    Returns:
        True if the joint is visible, False if occluded.
    """
    pose_bone = armature.pose.bones[bone_name]
    bone_world_pos = armature.matrix_world @ pose_bone.head

    # For orthographic camera, the ray direction is the camera's forward axis
    # (camera looks down its local -Z in Blender)
    cam_forward = camera.matrix_world.to_3x3() @ Vector((0, 0, -1))
    cam_forward.normalize()

    # Ray origin: start from the bone position offset far back along the
    # camera's viewing direction (behind the camera)
    ray_origin = bone_world_pos - cam_forward * 100.0
    ray_direction = cam_forward

    # Ensure depsgraph is current
    depsgraph = bpy.context.evaluated_depsgraph_get()

    # Cast ray through the scene
    hit, location, _normal, _index, hit_obj, _matrix = scene.ray_cast(
        depsgraph,
        ray_origin,
        ray_direction,
    )

    if not hit:
        # No geometry hit at all — joint is visible
        return True

    # Check if the hit object is one of our character meshes
    if hit_obj not in meshes and hit_obj.original not in meshes:
        # Hit something else (shouldn't happen in a clean scene)
        return True

    # Compare distances: if the hit point is significantly closer to the
    # camera than the bone, the joint is occluded
    hit_dist = (location - ray_origin).length
    bone_dist = (bone_world_pos - ray_origin).length

    # Bones sit inside the mesh, so the first ray hit is always the front
    # surface *before* the bone center.  A joint is "visible" when the gap
    # (bone_dist - hit_dist) is small — i.e. the bone is just beneath the
    # surface.  A joint is "occluded" when another body part sits in front,
    # producing a large gap.
    #
    # Use a generous tolerance (0.15 world units ≈ 15-20% of a HumanRig
    # character's height, or ~0.1% of a Mixamo character).  This prevents
    # false occlusion from self-intersection while still detecting real
    # occlusion from other body parts.
    tolerance = 0.15
    return hit_dist >= (bone_dist - tolerance)


# ---------------------------------------------------------------------------
# Bounding box computation
# ---------------------------------------------------------------------------


def _compute_bbox(
    joint_positions: dict[str, tuple[int, int]],
    visible_flags: dict[str, bool],
    image_width: int,
    image_height: int,
) -> list[int]:
    """Compute 2D bounding box from visible joint positions.

    Args:
        joint_positions: Region name → (px_x, px_y) for all joints.
        visible_flags: Region name → visibility flag.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        [x_min, y_min, x_max, y_max] in pixel coordinates, clamped to
        image bounds. Returns [0, 0, image_width, image_height] if no
        visible joints.
    """
    visible_points = [
        pos
        for name, pos in joint_positions.items()
        if visible_flags.get(name, False) and pos != (-1, -1)
    ]

    if not visible_points:
        return [0, 0, image_width, image_height]

    xs = [p[0] for p in visible_points]
    ys = [p[1] for p in visible_points]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Add padding
    width = x_max - x_min
    height = y_max - y_min
    pad_x = max(int(width * JOINT_BBOX_PADDING), 5)
    pad_y = max(int(height * JOINT_BBOX_PADDING), 5)

    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(image_width, x_max + pad_x)
    y_max = min(image_height, y_max + pad_y)

    return [x_min, y_min, x_max, y_max]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def extract_joints(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    armature: bpy.types.Object,
    meshes: list[bpy.types.Object],
    bone_to_region: dict[str, RegionId],
) -> dict:
    """Extract 2D joint positions for all body regions.

    For each of the 19 body regions (1–19), finds the primary bone,
    projects it to 2D pixel coordinates, and checks occlusion.

    Args:
        scene: The Blender scene.
        camera: The active camera object.
        armature: The character's armature object (must be posed).
        meshes: Character mesh objects (for occlusion raycasting).
        bone_to_region: Bone name → region ID mapping from bone_mapper.

    Returns:
        Dict with joint data matching the PRD schema::

            {
                "joints": {
                    "head": {"position": [256, 48], "confidence": 1.0, "visible": true},
                    ...
                },
                "bbox": [128, 20, 384, 480],
                "image_size": [512, 512]
            }
    """
    res_x = scene.render.resolution_x
    res_y = scene.render.resolution_y

    # Invert mapping: region_id → list of bone names
    region_to_bones: dict[RegionId, list[str]] = {}
    for bone_name, region_id in bone_to_region.items():
        if region_id == 0:  # skip background
            continue
        region_to_bones.setdefault(region_id, []).append(bone_name)

    joints: dict[str, dict] = {}
    positions: dict[str, tuple[int, int]] = {}
    visibility: dict[str, bool] = {}

    for region_id in range(1, NUM_JOINT_REGIONS + 1):
        region_name = REGION_NAMES[region_id]
        bones_in_region = region_to_bones.get(region_id, [])

        primary_bone = _select_primary_bone(region_id, bones_in_region)

        if primary_bone is None:
            # No bone mapped for this region
            logger.warning(
                "No bone mapped for region %d (%s) — marking as not visible",
                region_id,
                region_name,
            )
            joints[region_name] = {
                "position": [-1, -1],
                "confidence": 0.0,
                "visible": False,
            }
            positions[region_name] = (-1, -1)
            visibility[region_name] = False
            continue

        # Project to 2D
        (px_x, px_y), depth = _project_bone_to_2d(
            scene,
            camera,
            armature,
            primary_bone,
        )

        # Check if within image bounds
        in_bounds = 0 <= px_x < res_x and 0 <= px_y < res_y

        # Check occlusion
        visible = in_bounds and _check_occlusion(
            scene,
            camera,
            armature,
            primary_bone,
            meshes,
            depth,
        )

        # Clamp to image bounds for the output position
        clamped_x = max(0, min(px_x, res_x - 1))
        clamped_y = max(0, min(px_y, res_y - 1))

        joints[region_name] = {
            "position": [clamped_x, clamped_y],
            "confidence": 1.0 if visible else 0.5,
            "visible": visible,
        }
        positions[region_name] = (clamped_x, clamped_y)
        visibility[region_name] = visible

    bbox = _compute_bbox(positions, visibility, res_x, res_y)

    visible_count = sum(1 for v in visibility.values() if v)
    logger.info(
        "Extracted %d joints (%d visible, %d occluded/missing)",
        len(joints),
        visible_count,
        len(joints) - visible_count,
    )

    return {
        "joints": joints,
        "bbox": bbox,
        "image_size": [res_x, res_y],
    }


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def save_joints(
    joint_data: dict,
    output_path: Path,
    character_id: str,
    pose_name: str,
    source_animation: str = "",
    source_frame: int = 0,
) -> Path:
    """Save joint data to a JSON file.

    Args:
        joint_data: Dict returned by extract_joints().
        output_path: File path for the output JSON.
        character_id: Character identifier for metadata.
        pose_name: Pose name for metadata.
        source_animation: Source animation filename (optional).
        source_frame: Source animation frame number (optional).

    Returns:
        The output path.
    """
    output = {
        "character_id": character_id,
        "pose_name": pose_name,
        "source_animation": source_animation,
        "source_frame": source_frame,
        "image_size": joint_data["image_size"],
        "joints": joint_data["joints"],
        "bbox": joint_data["bbox"],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    logger.info("Joint data saved to %s", output_path)
    return output_path
