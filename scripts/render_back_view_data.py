"""Render FBX/GLB characters at front + three_quarter + back for back view training.

Minimal Blender script — no armature required, no segmentation masks.
Just imports the mesh, applies textures, renders at 3 angles with orthographic camera.

Usage::

    blender --background --python scripts/render_back_view_data.py -- \
        --input-dir /Volumes/TAMWoolff/data/fbx/ \
        --output-dir /Volumes/TAMWoolff/data/output/back_view_pairs/ \
        --max-characters 0 \
        --resolution 512
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import bpy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANGLES = {
    "front": {"azimuth": 0, "elevation": 0},
    "three_quarter": {"azimuth": 45, "elevation": 0},
    "back": {"azimuth": 180, "elevation": 0},
}

CAMERA_DISTANCE = 25.0
CAMERA_PADDING = 0.15
RESOLUTION = 512


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def clear_scene() -> None:
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    # Also purge orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)


def import_model(filepath: Path) -> list[bpy.types.Object]:
    """Import FBX or GLB and return mesh objects."""
    ext = filepath.suffix.lower()
    if ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(filepath))
    elif ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(filepath))
    else:
        raise ValueError(f"Unsupported format: {ext}")

    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    return meshes


def get_bbox(meshes: list[bpy.types.Object]) -> tuple:
    """Get combined bounding box of all meshes."""
    import mathutils

    all_corners = []
    for obj in meshes:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ mathutils.Vector(corner)
            all_corners.append(world_corner)

    if not all_corners:
        return mathutils.Vector((0, 0, 0)), 1.0

    xs = [c.x for c in all_corners]
    ys = [c.y for c in all_corners]
    zs = [c.z for c in all_corners]

    center = mathutils.Vector((
        (min(xs) + max(xs)) / 2,
        (min(ys) + max(ys)) / 2,
        (min(zs) + max(zs)) / 2,
    ))

    max_dim = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    return center, max_dim


def setup_camera(
    scene: bpy.types.Scene,
    meshes: list[bpy.types.Object],
    azimuth: float,
    elevation: float,
    resolution: int,
) -> bpy.types.Object:
    """Create orthographic camera aimed at the character."""
    import mathutils

    center, max_dim = get_bbox(meshes)
    ortho_scale = max_dim * (1.0 + CAMERA_PADDING)

    az_rad = math.radians(azimuth)
    el_rad = math.radians(elevation)

    cam_x = center.x + CAMERA_DISTANCE * math.sin(az_rad) * math.cos(el_rad)
    cam_y = center.y - CAMERA_DISTANCE * math.cos(az_rad) * math.cos(el_rad)
    cam_z = center.z + CAMERA_DISTANCE * math.sin(el_rad)

    cam_data = bpy.data.cameras.new("BackViewCam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = ortho_scale
    cam_data.clip_start = 0.1
    cam_data.clip_end = 100.0

    cam_obj = bpy.data.objects.new("BackViewCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    cam_obj.location = (cam_x, cam_y, cam_z)

    # Point at center
    direction = center - mathutils.Vector((cam_x, cam_y, cam_z))
    rot_quat = direction.to_track_quat("-Z", "Y")
    cam_obj.rotation_euler = rot_quat.to_euler()

    # Render settings
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    return cam_obj


def setup_lighting(scene: bpy.types.Scene) -> None:
    """Add even lighting from all sides so back views aren't dark."""
    # Front light
    front_data = bpy.data.lights.new("FrontLight", "SUN")
    front_data.energy = 2.5
    front_obj = bpy.data.objects.new("FrontLight", front_data)
    scene.collection.objects.link(front_obj)
    front_obj.rotation_euler = (math.radians(45), 0, 0)

    # Back light (lights the back of the character)
    back_data = bpy.data.lights.new("BackLight", "SUN")
    back_data.energy = 2.5
    back_obj = bpy.data.objects.new("BackLight", back_data)
    scene.collection.objects.link(back_obj)
    back_obj.rotation_euler = (math.radians(135), 0, math.radians(180))

    # Left fill
    left_data = bpy.data.lights.new("LeftLight", "SUN")
    left_data.energy = 1.5
    left_obj = bpy.data.objects.new("LeftLight", left_data)
    scene.collection.objects.link(left_obj)
    left_obj.rotation_euler = (math.radians(60), 0, math.radians(90))

    # Right fill
    right_data = bpy.data.lights.new("RightLight", "SUN")
    right_data.energy = 1.5
    right_obj = bpy.data.objects.new("RightLight", right_data)
    scene.collection.objects.link(right_obj)
    right_obj.rotation_euler = (math.radians(60), 0, math.radians(-90))

    # Top ambient
    top_data = bpy.data.lights.new("TopLight", "SUN")
    top_data.energy = 1.0
    top_obj = bpy.data.objects.new("TopLight", top_data)
    scene.collection.objects.link(top_obj)
    top_obj.rotation_euler = (0, 0, 0)


def render_angle(
    scene: bpy.types.Scene,
    meshes: list[bpy.types.Object],
    angle_name: str,
    output_path: Path,
    resolution: int,
) -> bool:
    """Render one angle and save to output_path."""
    cfg = ANGLES[angle_name]

    # Remove old camera
    for obj in scene.objects:
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)

    setup_camera(scene, meshes, cfg["azimuth"], cfg["elevation"], resolution)

    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)
    return output_path.exists()


def discover_models(input_dir: Path) -> list[tuple[str, Path]]:
    """Find all FBX/GLB files at any depth, return [(char_id, filepath)].

    Uses rglob to find all models regardless of nesting depth.
    Deduplicates by character ID (parent directory name or stem).
    """
    seen: set[str] = set()
    models = []

    # Find all FBX and GLB files recursively
    for ext in ("*.fbx", "*.glb"):
        for f in sorted(input_dir.rglob(ext)):
            if f.name.startswith("._") or "withSkin" in f.name:
                continue

            # Use parent dir name as char_id if nested, else stem
            if f.parent != input_dir:
                char_id = f.parent.name
            else:
                char_id = f.stem

            if char_id in seen:
                continue
            seen.add(char_id)
            models.append((char_id, f))

    return models


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Parse args after "--"
    try:
        sep = sys.argv.index("--")
        script_args = sys.argv[sep + 1:]
    except ValueError:
        script_args = []

    parser = argparse.ArgumentParser(description="Render back view training pairs")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=RESOLUTION)
    parser.add_argument("--max-characters", type=int, default=0, help="0 = all")
    parser.add_argument("--only-new", action="store_true", help="Skip existing pairs")
    args = parser.parse_args(script_args)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    models = discover_models(args.input_dir)
    if not models:
        print(f"ERROR: No FBX/GLB files found in {args.input_dir}")
        sys.exit(1)

    if args.max_characters > 0:
        models = models[:args.max_characters]

    print(f"\n{'='*60}")
    print(f"  Back View Pair Renderer")
    print(f"  Characters: {len(models)}")
    print(f"  Angles:     {', '.join(ANGLES.keys())}")
    print(f"  Resolution: {args.resolution}x{args.resolution}")
    print(f"  Output:     {output_dir}")
    print(f"{'='*60}\n")

    # Use EEVEE for speed
    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"

    succeeded = 0
    failed = 0
    skipped = 0

    for i, (char_id, filepath) in enumerate(models):
        pair_dir = output_dir / f"pair_{i:05d}"

        # Skip if already done
        if args.only_new and pair_dir.exists():
            expected = [pair_dir / f"{a}.png" for a in ANGLES]
            if all(f.exists() for f in expected):
                skipped += 1
                continue

        t0 = time.time()
        print(f"[{i+1}/{len(models)}] {char_id}...", end=" ", flush=True)

        try:
            clear_scene()
            meshes = import_model(filepath)

            if not meshes:
                print("SKIP (no meshes)")
                failed += 1
                continue

            setup_lighting(scene)
            pair_dir.mkdir(parents=True, exist_ok=True)

            ok = True
            for angle_name in ANGLES:
                out_path = pair_dir / f"{angle_name}.png"
                if not render_angle(scene, meshes, angle_name, out_path, args.resolution):
                    ok = False
                    break

            elapsed = time.time() - t0
            if ok:
                print(f"OK ({elapsed:.1f}s)")
                succeeded += 1
            else:
                print(f"FAIL ({elapsed:.1f}s)")
                failed += 1

        except Exception as e:
            elapsed = time.time() - t0
            print(f"ERROR: {e} ({elapsed:.1f}s)")
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Done! {succeeded} OK, {failed} failed, {skipped} skipped")
    print(f"  Pairs in: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
