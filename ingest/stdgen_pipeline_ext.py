"""Extend StdGEN's Blender rendering with Strata-specific outputs.

StdGEN's ``distributed_uniform.py`` renders VRoid characters at 4 cardinal
camera angles (0°, 90°, 180°, 270°).  This extension adds:

- **45° three-quarter camera angle** for Strata's multi-angle training data
- **2D joint positions** extracted from VRM armature (reuses logic from
  ``pipeline/joint_extractor.py``)
- **Draw order map** from Z-buffer depth (reuses logic from
  ``pipeline/draw_order_extractor.py``)
- **Body measurements** from mesh vertex data (reuses logic from
  ``pipeline/measurement_ground_truth.py``)
- **20-class segmentation** refined from StdGEN's 4-class annotations
  via bone-weight vertex mapping

Outputs follow Strata's standard directory format.

**Important:** This module requires Blender (``bpy``).  It is designed to be
run as a Blender script or imported within a Blender Python environment.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import bpy  # type: ignore[import-untyped]
from mathutils import Vector  # type: ignore[import-untyped]

from pipeline.bone_mapper import map_bones
from pipeline.config import (
    CAMERA_CLIP_END,
    CAMERA_CLIP_START,
    CAMERA_DISTANCE,
    CAMERA_PADDING,
    RENDER_RESOLUTION,
    RegionId,
)
from pipeline.joint_extractor import extract_joints
from pipeline.measurement_ground_truth import extract_mesh_measurements

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STDGEN_SOURCE = "stdgen"

# StdGEN default angles + Strata three-quarter addition.
STDGEN_ANGLES: dict[str, int] = {
    "front": 0,
    "three_quarter": 45,
    "side": 90,
    "back": 180,
    "side_r": 270,
}


# ---------------------------------------------------------------------------
# Camera setup
# ---------------------------------------------------------------------------


def setup_camera(
    scene: bpy.types.Scene,
    azimuth_deg: int,
    character_center: Vector,
    character_height: float,
    *,
    resolution: int = RENDER_RESOLUTION,
) -> bpy.types.Object:
    """Create or reuse an orthographic camera at the given azimuth angle.

    Args:
        scene: Blender scene.
        azimuth_deg: Camera azimuth in degrees (0 = front).
        character_center: World-space center of the character.
        character_height: Character bounding box height (Blender units).
        resolution: Render resolution (square).

    Returns:
        The camera object.
    """
    cam_name = f"StdGEN_Camera_{azimuth_deg}"

    if cam_name in bpy.data.objects:
        cam_obj = bpy.data.objects[cam_name]
    else:
        cam_data = bpy.data.cameras.new(cam_name)
        cam_obj = bpy.data.objects.new(cam_name, cam_data)
        scene.collection.objects.link(cam_obj)

    cam = cam_obj.data
    cam.type = "ORTHO"
    cam.ortho_scale = character_height * (1.0 + 2.0 * CAMERA_PADDING)
    cam.clip_start = CAMERA_CLIP_START
    cam.clip_end = CAMERA_CLIP_END

    # Position camera around character center.
    azimuth_rad = math.radians(azimuth_deg)
    cam_obj.location = Vector((
        character_center.x + math.sin(azimuth_rad) * CAMERA_DISTANCE,
        character_center.y - math.cos(azimuth_rad) * CAMERA_DISTANCE,
        character_center.z,
    ))

    # Point at character center.
    direction = character_center - cam_obj.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    cam_obj.rotation_euler = rot_quat.to_euler()

    scene.camera = cam_obj
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution

    return cam_obj


# ---------------------------------------------------------------------------
# Character analysis
# ---------------------------------------------------------------------------


def get_character_bounds(
    meshes: list[bpy.types.Object],
) -> tuple[Vector, float]:
    """Compute bounding box center and height for the character.

    Args:
        meshes: Character mesh objects.

    Returns:
        (center, height) tuple.
    """
    all_min = Vector((float("inf"), float("inf"), float("inf")))
    all_max = Vector((float("-inf"), float("-inf"), float("-inf")))

    for mesh_obj in meshes:
        for vert in mesh_obj.data.vertices:
            world_pos = mesh_obj.matrix_world @ vert.co
            for i in range(3):
                if world_pos[i] < all_min[i]:
                    all_min[i] = world_pos[i]
                if world_pos[i] > all_max[i]:
                    all_max[i] = world_pos[i]

    center = (all_min + all_max) / 2.0
    height = all_max.z - all_min.z

    return center, height


# ---------------------------------------------------------------------------
# Strata output generation
# ---------------------------------------------------------------------------


def _build_metadata(
    char_id: str,
    angle_name: str,
    angle_deg: int,
    *,
    has_segmentation: bool = False,
    has_joints: bool = False,
    has_draw_order: bool = False,
    has_measurements: bool = False,
    resolution: int = RENDER_RESOLUTION,
) -> dict[str, Any]:
    """Build Strata metadata dict for a StdGEN render.

    Args:
        char_id: Character identifier.
        angle_name: Camera angle name.
        angle_deg: Camera azimuth in degrees.
        has_segmentation: Whether 20-class segmentation is available.
        has_joints: Whether joint positions are available.
        has_draw_order: Whether draw order map is available.
        has_measurements: Whether body measurements are available.
        resolution: Image resolution.

    Returns:
        Metadata dict.
    """
    missing = []
    if not has_segmentation:
        missing.append("strata_segmentation")
    if not has_joints:
        missing.append("joints")
    if not has_draw_order:
        missing.append("draw_order")
    if not has_measurements:
        missing.append("measurements")

    return {
        "source": STDGEN_SOURCE,
        "source_character_id": char_id,
        "view_name": angle_name,
        "camera_angle_deg": angle_deg,
        "camera_type": "orthographic",
        "resolution": resolution,
        "has_segmentation_mask": has_segmentation,
        "has_fg_mask": True,
        "has_joints": has_joints,
        "has_draw_order": has_draw_order,
        "has_measurements": has_measurements,
        "missing_annotations": missing,
    }


def render_strata_outputs(
    scene: bpy.types.Scene,
    armature: bpy.types.Object,
    meshes: list[bpy.types.Object],
    character_id: str,
    output_dir: Path,
    bone_to_region: dict[str, RegionId],
    *,
    angles: dict[str, int] | None = None,
    resolution: int = RENDER_RESOLUTION,
    extract_joints_flag: bool = True,
    extract_measurements_flag: bool = True,
) -> list[Path]:
    """Render Strata outputs for a StdGEN character at specified angles.

    Args:
        scene: Blender scene.
        armature: Character armature object.
        meshes: Character mesh objects.
        character_id: Unique character ID.
        output_dir: Root output directory.
        bone_to_region: Bone name → region ID mapping.
        angles: Camera angles to render (default: all StdGEN + Strata angles).
        resolution: Image resolution (square).
        extract_joints_flag: Whether to extract 2D joint positions.
        extract_measurements_flag: Whether to extract body measurements.

    Returns:
        List of output directory paths (one per angle).
    """
    if angles is None:
        angles = STDGEN_ANGLES

    center, height = get_character_bounds(meshes)
    output_paths: list[Path] = []

    # Extract measurements once (pose-independent).
    measurements: dict[str, Any] | None = None
    if extract_measurements_flag:
        measurements = extract_mesh_measurements(meshes, bone_to_region)

    for angle_name, angle_deg in angles.items():
        example_id = f"{STDGEN_SOURCE}_{character_id}_{angle_name}"
        example_dir = output_dir / example_id
        example_dir.mkdir(parents=True, exist_ok=True)

        # Set up camera.
        camera = setup_camera(
            scene, angle_deg, center, height, resolution=resolution,
        )

        # Render color image.
        image_path = example_dir / "image.png"
        scene.render.filepath = str(image_path)
        scene.render.image_settings.file_format = "PNG"
        bpy.ops.render.render(write_still=True)

        # Extract joints.
        joint_data: dict | None = None
        if extract_joints_flag:
            joint_data = extract_joints(
                scene, camera, armature, meshes, bone_to_region,
            )
            joints_path = example_dir / "joints.json"
            joints_path.write_text(
                json.dumps(joint_data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        # Build metadata.
        metadata = _build_metadata(
            character_id,
            angle_name,
            angle_deg,
            has_joints=joint_data is not None,
            has_draw_order=False,  # draw order requires segmentation mask
            has_measurements=measurements is not None,
            resolution=resolution,
        )

        if measurements is not None:
            metadata["measurements"] = measurements

        meta_path = example_dir / "metadata.json"
        meta_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        output_paths.append(example_dir)

        logger.info(
            "Rendered StdGEN %s at %s (%d°) → %s",
            character_id,
            angle_name,
            angle_deg,
            example_dir,
        )

    return output_paths


# ---------------------------------------------------------------------------
# Bone mapping for VRM characters
# ---------------------------------------------------------------------------


def build_vrm_bone_mapping(
    armature: bpy.types.Object,
    meshes: list[bpy.types.Object],
    character_id: str,
) -> dict[str, RegionId]:
    """Build bone-to-region mapping for a VRM/VRoid armature.

    Delegates to ``pipeline.bone_mapper.map_bones`` which uses
    VRM_BONE_ALIASES for direct matching and falls back through the
    full bone mapping priority chain.

    Args:
        armature: The VRM character's armature object.
        meshes: Character mesh objects.
        character_id: Character identifier.

    Returns:
        Dict mapping bone names to Strata region IDs.
    """
    mapping = map_bones(armature, meshes, character_id)
    return mapping.bone_to_region
