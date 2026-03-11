"""Render HumanRig samples with Mixamo FBX poses at multiple camera angles.

Imports each sample's ``rigged.glb``, applies poses from FBX/BVH files,
renders orthographic views, and extracts 2D joint positions via raycasting.

Produces one output example per (sample × pose × angle) combination:

    humanrig_{sample_id:05d}_{pose_name}_{angle}/
        image.png       ← 512×512 RGBA render
        joints.json     ← 2D joint positions with occlusion
        metadata.json   ← Source info, pose, camera angle

Run via::

    blender --background --python run_humanrig_posed.py -- \\
        --input_dir /path/to/humanrig_opensource_final \\
        --pose_dir /path/to/poses \\
        --output_dir /path/to/output \\
        --max_samples 100
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RENDER_RESOLUTION = 512
CYCLES_SAMPLES = 64
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
# Blender scene setup (identical to humanrig_blender_renderer.py)
# ---------------------------------------------------------------------------


def _setup_scene() -> None:
    import bpy

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = CYCLES_SAMPLES
    scene.cycles.device = "CPU"
    scene.render.resolution_x = RENDER_RESOLUTION
    scene.render.resolution_y = RENDER_RESOLUTION
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"


def _setup_lights() -> None:
    import bpy

    scene = bpy.context.scene
    for energy, rx, rz in [(5.0, 45, 30), (5.0, 45, 210)]:
        light = bpy.data.lights.new("Sun", "SUN")
        light.energy = energy
        light.angle = math.radians(5)
        obj = bpy.data.objects.new("Sun", light)
        scene.collection.objects.link(obj)
        obj.rotation_euler = (math.radians(rx), 0, math.radians(rz))


def _setup_world() -> None:
    import bpy

    scene = bpy.context.scene
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Strength"].default_value = 0.0


def _make_camera() -> Any:
    import bpy

    scene = bpy.context.scene
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = ORTHO_SCALE
    cam_obj = bpy.data.objects.new("Cam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    return cam_obj


def _set_camera_orbit(cam_obj: Any, azimuth_deg: float) -> None:
    from mathutils import Matrix, Vector

    theta = math.radians(azimuth_deg)
    cam_x = math.sin(theta) * CAMERA_DIST
    cam_y = -math.cos(theta) * CAMERA_DIST
    cam_obj.location = Vector((cam_x, cam_y, 0.0))

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


def _import_glb(glb_path: Path) -> tuple[Any, list[Any]]:
    """Import GLB, remove Icosphere, fix materials. Return (armature, meshes)."""
    import bpy

    try:
        bpy.ops.import_scene.gltf(filepath=str(glb_path))
    except Exception as exc:
        logger.warning("GLB import failed for %s: %s", glb_path, exc)
        return None, []

    for obj in list(bpy.data.objects):
        if obj.name == "Icosphere":
            bpy.data.objects.remove(obj, do_unlink=True)

    for mat in bpy.data.materials:
        mat.blend_method = "OPAQUE"

    armature = None
    meshes = []
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            armature = obj
        elif obj.type == "MESH":
            meshes.append(obj)

    return armature, meshes


# ---------------------------------------------------------------------------
# Per-sample rendering
# ---------------------------------------------------------------------------


def render_posed_sample(
    glb_path: Path,
    sample_id: int,
    poses: list,
    pose_dir: Path,
    output_dir: Path,
    angles: list[str],
    *,
    only_new: bool = False,
) -> tuple[int, int]:
    """Render one HumanRig sample with multiple poses and angles.

    Returns:
        (rendered_count, skipped_count)
    """
    import bpy

    from pipeline.bone_mapper import map_bones
    from pipeline.joint_extractor import extract_joints
    from pipeline.pose_applicator import apply_pose, reset_pose

    # Reset scene and import GLB
    bpy.ops.wm.read_factory_settings(use_empty=True)
    _setup_scene()
    _setup_world()
    _setup_lights()

    armature, meshes = _import_glb(glb_path)
    if armature is None:
        logger.warning("No armature in %s — skipping", glb_path)
        return 0, 0

    # Map bones once per sample
    character_id = f"humanrig_{sample_id}"
    mapping = map_bones(armature, meshes, character_id)
    bone_to_region = mapping.bone_to_region

    cam_obj = _make_camera()
    scene = bpy.context.scene

    rendered = 0
    skipped = 0

    for pose in poses:
        # Check if ALL angles for this pose already exist
        if only_new:
            all_exist = True
            for label in angles:
                example_id = f"humanrig_{sample_id:05d}_{pose.name}_{label}"
                img_path = output_dir / example_id / "image.png"
                if not img_path.exists():
                    all_exist = False
                    break
            if all_exist:
                skipped += len(angles)
                continue

        # Apply pose
        ok = apply_pose(armature, pose, pose_dir)
        if not ok:
            logger.debug("Failed to apply pose %s to sample %d", pose.name, sample_id)
            skipped += len(angles)
            reset_pose(armature)
            continue

        bpy.context.view_layer.update()

        for label in angles:
            example_id = f"humanrig_{sample_id:05d}_{pose.name}_{label}"
            example_dir = output_dir / example_id

            if only_new and (example_dir / "image.png").exists():
                skipped += 1
                continue

            example_dir.mkdir(parents=True, exist_ok=True)

            azimuth = ANGLE_AZIMUTHS[label]
            _set_camera_orbit(cam_obj, azimuth)
            bpy.context.view_layer.update()

            # Extract joints
            joints_data = extract_joints(
                scene, cam_obj, armature, meshes, bone_to_region
            )

            # Render
            img_path = example_dir / "image.png"
            scene.render.filepath = str(img_path)
            try:
                bpy.ops.render.render(write_still=True)
            except Exception as exc:
                logger.warning("Render failed for %s: %s", example_id, exc)
                continue

            # Save joints
            joints_path = example_dir / "joints.json"
            joints_path.write_text(
                json.dumps(joints_data, indent=2) + "\n", encoding="utf-8"
            )

            # Save metadata
            meta = {
                "source": "humanrig",
                "sample_id": sample_id,
                "character_id": character_id,
                "pose_name": pose.name,
                "source_animation": pose.source,
                "source_frame": pose.frame,
                "camera_angle": label,
                "camera_azimuth_deg": azimuth,
                "render_resolution": RENDER_RESOLUTION,
                "has_rendered_image": True,
                "has_joints": True,
                "joint_source": "blender_raycast",
            }
            meta_path = example_dir / "metadata.json"
            meta_path.write_text(
                json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            rendered += 1

        reset_pose(armature)

    return rendered, skipped


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def render_directory(
    input_dir: Path,
    output_dir: Path,
    pose_dir: Path,
    *,
    angles: list[str] | None = None,
    only_new: bool = False,
    max_samples: int = 0,
    poses_per_clip: int = 3,
) -> tuple[int, int, int]:
    """Render posed HumanRig samples in batch.

    Args:
        input_dir: ``humanrig_opensource_final/`` directory.
        output_dir: Root output directory.
        pose_dir: Directory containing Mixamo FBX/BVH animation files.
        angles: Camera angle labels. Defaults to all 5 angles.
        only_new: Skip examples that already have image.png.
        max_samples: Max samples to process (0 = all).
        poses_per_clip: Keyframes to sample per animation clip.

    Returns:
        (rendered_total, skipped_total, error_count)
    """
    from pipeline.pose_applicator import list_poses

    if angles is None:
        angles = list(ANGLE_AZIMUTHS.keys())

    # Discover poses
    logger.info("Scanning poses from %s ...", pose_dir)
    all_poses = list_poses(pose_dir, keyframes_per_clip=poses_per_clip)
    logger.info("Found %d poses (%d clips)", len(all_poses), len(set(p.source for p in all_poses)))

    # Discover samples
    sample_dirs = sorted(
        [d for d in input_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    if max_samples > 0:
        sample_dirs = sample_dirs[:max_samples]

    total = len(sample_dirs)
    logger.info(
        "Processing %d samples × %d poses × %d angles = %d potential examples",
        total, len(all_poses), len(angles), total * len(all_poses) * len(angles),
    )

    rendered_total = 0
    skipped_total = 0
    errors = 0
    start_time = time.monotonic()

    for i, sample_dir in enumerate(sample_dirs):
        glb_path = sample_dir / "rigged.glb"
        if not glb_path.is_file():
            logger.warning("No rigged.glb in %s — skipping", sample_dir)
            errors += 1
            continue

        sample_id = int(sample_dir.name)

        try:
            rendered, skipped = render_posed_sample(
                glb_path,
                sample_id,
                all_poses,
                pose_dir,
                output_dir,
                angles,
                only_new=only_new,
            )
        except Exception as exc:
            logger.warning("Error on sample %d: %s", sample_id, exc)
            errors += 1
            continue

        rendered_total += rendered
        skipped_total += skipped

        if (i + 1) % 10 == 0 or (i + 1) == total:
            elapsed = time.monotonic() - start_time
            rate = rendered_total / elapsed if elapsed > 0 else 0
            pct = (i + 1) / total * 100
            logger.info(
                "Progress: %d/%d samples (%.1f%%) — %d rendered, %d skipped, "
                "%d errors, %.1f img/s",
                i + 1, total, pct, rendered_total, skipped_total, errors, rate,
            )

    return rendered_total, skipped_total, errors
