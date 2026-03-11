"""Test posed HumanRig rendering with joint extraction in Blender.

Imports a few HumanRig GLBs, applies BVH poses, renders from multiple
angles, and overlays extracted joints to verify there are no distortions.

Run via Blender:
    blender --background --python scripts/test_posed_humanrig.py -- \
        --samples 3 --poses 2 \
        --output_dir output/posed_humanrig_test
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

# Blender modules
import bpy
from mathutils import Matrix, Vector

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.joint_extractor import extract_joints
from pipeline.pose_applicator import (
    PoseInfo,
    apply_pose,
    list_poses,
    reset_pose,
)
from pipeline.bone_mapper import map_bones

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HUMANRIG_ROOT = Path(
    "/Volumes/TAMWoolff/data/preprocessed/humanrig/data/54T/chuzedong/autorig/preprocess/humanrig_opensource_final"
)
POSE_DIR = Path("/Volumes/TAMWoolff/data/poses_bvh")
RENDER_RESOLUTION = 512
CYCLES_SAMPLES = 32  # Lower for fast test
ORTHO_SCALE = 1.3
CAMERA_DIST = 2.0

ANGLE_AZIMUTHS = {
    "front": 0,
    "three_quarter": 45,
    "side": 90,
    "back": 180,
}

# Joint colors for overlay
JOINT_COLORS = [
    (1, 0, 0, 1),       # head - red
    (1, 0.5, 0, 1),     # neck - orange
    (1, 1, 0, 1),       # chest - yellow
    (0.5, 1, 0, 1),     # spine - lime
    (0, 1, 0, 1),       # hips - green
    (0, 1, 0.5, 1),     # shoulder_l
    (0, 1, 1, 1),       # upper_arm_l - cyan
    (0, 0.5, 1, 1),     # forearm_l
    (0, 0, 1, 1),       # hand_l - blue
    (0.5, 0, 1, 1),     # shoulder_r
    (1, 0, 1, 1),       # upper_arm_r - magenta
    (1, 0, 0.5, 1),     # forearm_r
    (0.5, 0, 0.5, 1),   # hand_r
    (1, 0.5, 0.5, 1),   # upper_leg_l
    (0.5, 1, 0.5, 1),   # lower_leg_l
    (0.5, 0.5, 1, 1),   # foot_l
    (1, 0.5, 1, 1),     # upper_leg_r
    (0.5, 1, 1, 1),     # lower_leg_r
    (1, 1, 0.5, 1),     # foot_r
]


# ---------------------------------------------------------------------------
# Blender helpers (adapted from humanrig_blender_renderer.py)
# ---------------------------------------------------------------------------


def setup_scene():
    """Configure render settings."""
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


def setup_lights():
    """Add two directional lights."""
    scene = bpy.context.scene
    for energy, rx, rz in [(5.0, 45, 30), (5.0, 45, 210)]:
        light = bpy.data.lights.new("Sun", "SUN")
        light.energy = energy
        light.angle = math.radians(5)
        obj = bpy.data.objects.new("Sun", light)
        scene.collection.objects.link(obj)
        obj.rotation_euler = (math.radians(rx), 0, math.radians(rz))


def setup_world():
    """Set world background to black."""
    scene = bpy.context.scene
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Strength"].default_value = 0.0


def make_camera():
    """Create orthographic camera."""
    scene = bpy.context.scene
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = ORTHO_SCALE
    cam_obj = bpy.data.objects.new("Cam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    return cam_obj


def set_camera_orbit(cam_obj, azimuth_deg):
    """Position camera to orbit around world Z."""
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


def import_glb(glb_path):
    """Import GLB, remove Icosphere, fix materials. Return (armature, meshes)."""
    bpy.ops.import_scene.gltf(filepath=str(glb_path))

    # Remove background sphere
    for obj in list(bpy.data.objects):
        if obj.name == "Icosphere":
            bpy.data.objects.remove(obj, do_unlink=True)

    # Fix material blend mode
    for mat in bpy.data.materials:
        mat.blend_method = "OPAQUE"

    # Find armature and meshes
    armature = None
    meshes = []
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            armature = obj
        elif obj.type == "MESH":
            meshes.append(obj)

    return armature, meshes


def draw_joints_on_render(render_path, joints_data, output_path):
    """Overlay joint dots on a rendered image using PIL (post-render)."""
    from PIL import Image, ImageDraw

    img = Image.open(render_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    radius = 6

    joints = joints_data.get("joints", {})
    # Draw skeleton connections
    connections = [
        ("head", "neck"), ("neck", "chest"), ("chest", "spine"), ("spine", "hips"),
        ("neck", "shoulder_l"), ("shoulder_l", "upper_arm_l"),
        ("upper_arm_l", "forearm_l"), ("forearm_l", "hand_l"),
        ("neck", "shoulder_r"), ("shoulder_r", "upper_arm_r"),
        ("upper_arm_r", "forearm_r"), ("forearm_r", "hand_r"),
        ("hips", "upper_leg_l"), ("upper_leg_l", "lower_leg_l"), ("lower_leg_l", "foot_l"),
        ("hips", "upper_leg_r"), ("upper_leg_r", "lower_leg_r"), ("lower_leg_r", "foot_r"),
    ]

    for a, b in connections:
        ja = joints.get(a, {})
        jb = joints.get(b, {})
        if ja.get("visible") and jb.get("visible"):
            pa = ja["position"]
            pb = jb["position"]
            draw.line([(pa[0], pa[1]), (pb[0], pb[1])], fill=(255, 255, 255, 180), width=2)

    region_names = [
        "head", "neck", "chest", "spine", "hips",
        "shoulder_l", "upper_arm_l", "forearm_l", "hand_l",
        "shoulder_r", "upper_arm_r", "forearm_r", "hand_r",
        "upper_leg_l", "lower_leg_l", "foot_l",
        "upper_leg_r", "lower_leg_r", "foot_r",
    ]

    for idx, name in enumerate(region_names):
        j = joints.get(name, {})
        if not j.get("visible"):
            continue
        x, y = j["position"]
        color_f = JOINT_COLORS[idx]
        color = tuple(int(c * 255) for c in color_f[:3]) + (255,)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                     fill=color, outline=(255, 255, 255, 255))

    img.save(output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # Parse args after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--poses", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="output/posed_humanrig_test")
    parser.add_argument("--input_dir", type=str, default=str(HUMANRIG_ROOT))
    parser.add_argument("--pose_dir", type=str, default=str(POSE_DIR))
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    pose_dir = Path(args.pose_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover samples
    sample_dirs = sorted(
        [d for d in input_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    logger.info("Found %d samples", len(sample_dirs))

    # Pick evenly spaced samples
    n = min(args.samples, len(sample_dirs))
    step = max(1, len(sample_dirs) // n)
    selected = [sample_dirs[i * step] for i in range(n)]

    # Discover poses
    logger.info("Scanning poses from %s ...", pose_dir)
    all_poses = list_poses(pose_dir, keyframes_per_clip=3, include_builtins=True)
    logger.info("Found %d total poses", len(all_poses))

    # Pick a few diverse poses (T-pose + mid-frame BVH for visible deformation)
    chosen_poses = []
    # Always include T-pose
    for p in all_poses:
        if p.name == "t_pose":
            chosen_poses.append(p)
            break
    # Add BVH poses — pick FIRST keyframe from each clip.
    # 100STYLE BVH files have action in the first ~20% of frames;
    # evenly-spaced sampling puts mid/end frames in near-static sections.
    bvh_poses = [p for p in all_poses if p.source not in ("built-in", "rest")]
    # Group by source file, pick the first keyframe from each
    from collections import OrderedDict
    by_source: dict[str, list] = OrderedDict()
    for p in bvh_poses:
        by_source.setdefault(p.source, []).append(p)
    action_poses = []
    for source, frames in by_source.items():
        # Pick the first frame (where the action is in 100STYLE BVH)
        action_poses.append(frames[0])
    step_p = max(1, len(action_poses) // args.poses)
    for i in range(min(args.poses, len(action_poses))):
        chosen_poses.append(action_poses[i * step_p])

    logger.info("Using %d poses: %s", len(chosen_poses), [p.name for p in chosen_poses])

    total_rendered = 0
    angles_to_test = ["front", "three_quarter", "side", "back"]

    for sample_dir in selected:
        sample_id = sample_dir.name
        glb_path = sample_dir / "rigged.glb"
        if not glb_path.is_file():
            logger.warning("No rigged.glb in %s", sample_dir)
            continue

        logger.info("=== Sample %s ===", sample_id)

        for pose in chosen_poses:
            logger.info("  Pose: %s", pose.name)

            # Reset scene
            bpy.ops.wm.read_factory_settings(use_empty=True)
            setup_scene()
            setup_world()
            setup_lights()

            # Import GLB
            armature, meshes = import_glb(glb_path)
            if armature is None:
                logger.warning("  No armature found — skipping")
                continue

            # Map bones
            mapping = map_bones(armature, meshes, f"humanrig_{sample_id}")
            bone_to_region = mapping.bone_to_region
            logger.info("  Mapped %d bones to regions", len(bone_to_region))

            # Apply pose
            ok = apply_pose(armature, pose, pose_dir)
            if not ok:
                logger.warning("  Failed to apply pose %s", pose.name)
                continue

            # Force dependency graph update
            bpy.context.view_layer.update()

            # Create camera
            cam_obj = make_camera()

            for angle_label in angles_to_test:
                azimuth = ANGLE_AZIMUTHS[angle_label]
                set_camera_orbit(cam_obj, azimuth)
                bpy.context.view_layer.update()

                # Extract joints
                scene = bpy.context.scene
                joints_data = extract_joints(
                    scene, cam_obj, armature, meshes, bone_to_region
                )

                # Render
                example_name = f"sample_{sample_id}_{pose.name}_{angle_label}"
                render_path = output_dir / f"{example_name}_render.png"
                scene.render.filepath = str(render_path)
                bpy.ops.render.render(write_still=True)

                # Save joints
                joints_path = output_dir / f"{example_name}_joints.json"
                joints_path.write_text(
                    json.dumps(joints_data, indent=2) + "\n"
                )

                # Overlay joints on render
                overlay_path = output_dir / f"{example_name}_overlay.png"
                draw_joints_on_render(str(render_path), joints_data, str(overlay_path))

                total_rendered += 1
                logger.info("    %s: %d joints visible",
                           angle_label,
                           sum(1 for j in joints_data.get("joints", {}).values()
                               if j.get("visible")))

            reset_pose(armature)

    logger.info("Done! Rendered %d images to %s", total_rendered, output_dir)


if __name__ == "__main__":
    main()
