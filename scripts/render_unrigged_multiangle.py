"""Render unrigged FBX characters from multiple angles using Blender.

Imports each FBX, normalizes scale/position, and renders textured RGBA images
from front, three_quarter, and back views. No armature required.

Output structure::

    output_dir/
        CharName_texture_front/
            image.png
        CharName_texture_three_quarter/
            image.png
        CharName_texture_back/
            image.png

Usage::

    blender --background --python scripts/render_unrigged_multiangle.py -- \
        --input-dir /Volumes/TAMWoolff/data/fbx/ \
        --output-dir /Volumes/TAMWoolff/data/output/meshy_unrigged_multiangle \
        --only-new

    # Render a single character
    blender --background --python scripts/render_unrigged_multiangle.py -- \
        --input-dir /Volumes/TAMWoolff/data/fbx/Meshy_AI_foo \
        --output-dir ./output/test_render
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from math import cos, radians, sin
from pathlib import Path

import bpy
from mathutils import Vector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESOLUTION = 512
TARGET_HEIGHT = 2.0  # Normalize characters to this height
CAMERA_DISTANCE = 10.0
CAMERA_PADDING = 0.10

ANGLES = {
    "front": {"azimuth": 0, "elevation": 0},
    "three_quarter": {"azimuth": 45, "elevation": 0},
    "back": {"azimuth": 180, "elevation": 0},
}


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------


def clear_scene() -> None:
    """Remove all objects, meshes, materials, etc."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=True)
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)
    for block in bpy.data.textures:
        bpy.data.textures.remove(block)
    for block in bpy.data.images:
        bpy.data.images.remove(block)
    for block in bpy.data.cameras:
        bpy.data.cameras.remove(block)
    for block in bpy.data.lights:
        bpy.data.lights.remove(block)


def combined_bounding_box(meshes: list) -> tuple[Vector, Vector]:
    """Get world-space axis-aligned bounding box of all meshes."""
    xs, ys, zs = [], [], []
    for obj in meshes:
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            xs.append(world.x)
            ys.append(world.y)
            zs.append(world.z)
    if not xs:
        return Vector((0, 0, 0)), Vector((1, 1, 1))
    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


def import_fbx(fbx_path: Path) -> list:
    """Import FBX and return list of mesh objects."""
    clear_scene()
    try:
        bpy.ops.import_scene.fbx(filepath=str(fbx_path))
    except Exception:
        logger.exception("Failed to import: %s", fbx_path)
        return []

    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    return meshes


def normalize_meshes(meshes: list) -> None:
    """Scale and center meshes to standard size."""
    bbox_min, bbox_max = combined_bounding_box(meshes)
    height = bbox_max.z - bbox_min.z

    if height < 1e-6:
        logger.warning("Near-zero height, skipping normalization")
        return

    scale_factor = TARGET_HEIGHT / height
    center_x = (bbox_min.x + bbox_max.x) / 2
    center_y = (bbox_min.y + bbox_max.y) / 2

    for obj in meshes:
        obj.scale *= scale_factor

    # Apply scale transforms
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    if meshes:
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # Recompute bbox after scale
    bbox_min, bbox_max = combined_bounding_box(meshes)
    center_x = (bbox_min.x + bbox_max.x) / 2
    center_y = (bbox_min.y + bbox_max.y) / 2
    offset = Vector((-center_x, -center_y, -bbox_min.z))

    for obj in meshes:
        obj.location += offset

    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    if meshes:
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)


def setup_camera(meshes: list, azimuth: float, elevation: float) -> None:
    """Create orthographic camera aimed at the character."""
    scene = bpy.context.scene

    # Remove old cameras
    for obj in list(bpy.data.objects):
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)

    bbox_min, bbox_max = combined_bounding_box(meshes)
    bbox_center = (bbox_min + bbox_max) / 2

    width = bbox_max.x - bbox_min.x
    height = bbox_max.z - bbox_min.z
    depth = bbox_max.y - bbox_min.y

    az_rad = radians(azimuth)
    el_rad = radians(elevation)
    apparent_width = abs(width * cos(az_rad)) + abs(depth * sin(az_rad))
    apparent_height = abs(height * cos(el_rad)) + abs(depth * sin(el_rad))
    ortho_scale = max(apparent_width, apparent_height) * (1.0 + 2.0 * CAMERA_PADDING)

    cam_data = bpy.data.cameras.new(name="render_cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = ortho_scale
    cam_data.clip_start = 0.1
    cam_data.clip_end = 100.0

    cam_obj = bpy.data.objects.new(name="render_cam", object_data=cam_data)
    scene.collection.objects.link(cam_obj)

    cam_obj.location = (
        bbox_center.x + CAMERA_DISTANCE * sin(az_rad) * cos(el_rad),
        bbox_center.y - CAMERA_DISTANCE * cos(az_rad) * cos(el_rad),
        bbox_center.z + CAMERA_DISTANCE * sin(el_rad),
    )
    cam_obj.rotation_euler = (radians(90) - el_rad, 0, az_rad)
    scene.camera = cam_obj


def setup_render() -> None:
    """Configure EEVEE render settings."""
    scene = bpy.context.scene

    # Try EEVEE Next (Blender 4.2-4.4), fall back to EEVEE (5.0+)
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"

    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.filter_size = 1.5

    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 15

    scene.render.use_compositing = False
    scene.render.use_sequencer = False

    # Disable ambient occlusion if available
    if hasattr(scene.eevee, "use_gtao"):
        scene.eevee.use_gtao = False

    # Add sun light
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)

    sun_data = bpy.data.lights.new(name="Sun", type="SUN")
    sun_data.energy = 3.0
    sun_obj = bpy.data.objects.new(name="Sun", object_data=sun_data)
    scene.collection.objects.link(sun_obj)
    sun_obj.rotation_euler = (radians(50), radians(10), radians(-30))

    # Ambient light via world
    if not scene.world:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.15, 0.15, 0.15, 1.0)
        bg.inputs["Strength"].default_value = 1.0


def render_to(output_path: Path) -> None:
    """Render current scene to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_fbx_dirs(input_dir: Path) -> list[Path]:
    """Find Meshy FBX files, searching up to 2 levels deep.

    Handles both layouts:
      - Meshy_AI_foo/file.fbx  (old, direct)
      - Meshy_AI_foo/Meshy_AI_foo/file.fbx  (new, nested from zip)
    """
    results = []
    if input_dir.is_file() and input_dir.suffix.lower() == ".fbx":
        return [input_dir]

    for child in sorted(input_dir.iterdir()):
        if not child.is_dir() or not child.name.startswith("Meshy_AI_"):
            continue
        # Search direct and one level deeper
        fbx_files = list(child.glob("*.fbx")) + list(child.glob("*/*.fbx"))
        # Filter out macOS resource forks
        fbx_files = [f for f in fbx_files if not f.name.startswith("._")]
        if fbx_files:
            results.append(fbx_files[0])
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Parse args after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Render unrigged FBX from multiple angles")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only-new", action="store_true", help="Skip already-rendered characters")
    parser.add_argument("--resolution", type=int, default=RESOLUTION)
    args = parser.parse_args(argv)

    fbx_files = discover_fbx_dirs(args.input_dir)
    logger.info("Found %d FBX files to render", len(fbx_files))

    rendered = 0
    skipped = 0
    failed = 0
    t_start = time.time()

    for i, fbx_path in enumerate(fbx_files):
        char_name = fbx_path.parent.name or fbx_path.stem

        # Check if already rendered (all 3 angles)
        if args.only_new:
            all_exist = all(
                (args.output_dir / f"{char_name}_texture_{angle}" / "image.png").exists()
                for angle in ANGLES
            )
            if all_exist:
                skipped += 1
                continue

        logger.info("[%d/%d] Rendering %s", i + 1, len(fbx_files), char_name)

        meshes = import_fbx(fbx_path)
        if not meshes:
            logger.warning("  No meshes found, skipping")
            failed += 1
            continue

        normalize_meshes(meshes)
        setup_render()

        for angle_name, angle_cfg in ANGLES.items():
            output_path = args.output_dir / f"{char_name}_texture_{angle_name}" / "image.png"

            if args.only_new and output_path.exists():
                continue

            setup_camera(meshes, angle_cfg["azimuth"], angle_cfg["elevation"])
            render_to(output_path)

        rendered += 1
        elapsed = time.time() - t_start
        rate = rendered / elapsed if elapsed > 0 else 0
        remaining = len(fbx_files) - i - 1 - skipped
        eta = remaining / rate if rate > 0 else 0
        logger.info("  Done. %d rendered, %d skipped, %d failed — %.1f char/min, ETA %.0fm",
                     rendered, skipped, failed, rate * 60, eta / 60)

    elapsed = time.time() - t_start
    logger.info("\nComplete! %d rendered, %d skipped, %d failed (%.1f min)",
                rendered, skipped, failed, elapsed / 60)


if __name__ == "__main__":
    main()
