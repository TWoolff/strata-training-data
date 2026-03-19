"""Render FBX/GLB characters in default pose with multiple art styles and angles.

No armature required — imports mesh as-is, applies toon styles, renders.
Works with both rigged and unrigged characters. Produces training images
(not GT masks) for style diversity in segmentation training.

Usage::

    blender --background --python scripts/render_toon_styles.py -- \
        --input-dir /Volumes/TAMWoolff/data/fbx/ \
        --output-dir /Volumes/TAMWoolff/data/output/toon/ \
        --styles soft_cel,ink_wash,watercolor,cel,textured \
        --angles front,three_quarter,side \
        --resolution 512 \
        --only-new
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
import mathutils
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANGLES = {
    "front": {"azimuth": 0, "elevation": 0},
    "three_quarter": {"azimuth": 45, "elevation": 0},
    "side": {"azimuth": 90, "elevation": 0},
    "three_quarter_back": {"azimuth": 135, "elevation": 0},
    "back": {"azimuth": 180, "elevation": 0},
}

DEFAULT_ANGLES = ["front", "three_quarter", "side"]
DEFAULT_STYLES = ["soft_cel", "ink_wash", "watercolor"]

CAMERA_DISTANCE = 25.0
CAMERA_PADDING = 0.15
RESOLUTION = 512

# Cel shading ramp stops (position, brightness)
CEL_RAMP_STOPS = [
    (0.0, 0.3),
    (0.4, 0.7),
    (0.7, 1.0),
]
CEL_OUTLINE_THICKNESS = 2.0

SOFT_CEL_RAMP_STOPS = [
    (0.0, 0.15),
    (0.25, 0.55),
    (0.5, 0.80),
    (0.72, 0.95),
    (0.88, 1.0),
]

# Post-render style params (imported from pipeline but inlined here for standalone use)
INK_WASH_BILATERAL_D = 9
INK_WASH_SIGMA_COLOR = 75
INK_WASH_SIGMA_SPACE = 75
INK_WASH_SATURATION = 0.4
INK_WASH_TINT = (240, 230, 210)
INK_WASH_EDGE_THRESHOLD1 = 30
INK_WASH_EDGE_THRESHOLD2 = 100
INK_WASH_EDGE_THICKNESS = 2

WATERCOLOR_BILATERAL_D = 9
WATERCOLOR_BILATERAL_PASSES = 3
WATERCOLOR_SIGMA_COLOR = 100
WATERCOLOR_SIGMA_SPACE = 100
WATERCOLOR_SAT_BOOST = 1.3
WATERCOLOR_GRAIN_SIGMA = 0.02
WATERCOLOR_EDGE_THRESHOLD1 = 40
WATERCOLOR_EDGE_THRESHOLD2 = 120
WATERCOLOR_EDGE_THICKNESS = 1
WATERCOLOR_EDGE_COLOR = (60, 40, 30)


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------


def clear_scene() -> None:
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
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
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def get_bbox(meshes: list[bpy.types.Object]) -> tuple:
    """Get combined bounding box center and max dimension."""
    all_corners = []
    for obj in meshes:
        for corner in obj.bound_box:
            all_corners.append(obj.matrix_world @ mathutils.Vector(corner))

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
    center, max_dim = get_bbox(meshes)
    ortho_scale = max_dim * (1.0 + CAMERA_PADDING)

    az_rad = math.radians(azimuth)
    el_rad = math.radians(elevation)

    cam_x = center.x + CAMERA_DISTANCE * math.sin(az_rad) * math.cos(el_rad)
    cam_y = center.y - CAMERA_DISTANCE * math.cos(az_rad) * math.cos(el_rad)
    cam_z = center.z + CAMERA_DISTANCE * math.sin(el_rad)

    cam_data = bpy.data.cameras.new("ToonCam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = ortho_scale
    cam_data.clip_start = 0.1
    cam_data.clip_end = 100.0

    cam_obj = bpy.data.objects.new("ToonCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    cam_obj.location = (cam_x, cam_y, cam_z)

    direction = center - mathutils.Vector((cam_x, cam_y, cam_z))
    rot_quat = direction.to_track_quat("-Z", "Y")
    cam_obj.rotation_euler = rot_quat.to_euler()

    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    return cam_obj


def setup_lighting(scene: bpy.types.Scene) -> None:
    """Add even multi-directional lighting."""
    lights = [
        ("FrontLight", 2.5, (45, 0, 0)),
        ("BackLight", 2.5, (135, 0, 180)),
        ("LeftLight", 1.5, (60, 0, 90)),
        ("RightLight", 1.5, (60, 0, -90)),
        ("TopLight", 1.0, (0, 0, 0)),
    ]
    for name, energy, rot_deg in lights:
        data = bpy.data.lights.new(name, "SUN")
        data.energy = energy
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        obj.rotation_euler = tuple(math.radians(d) for d in rot_deg)


def remove_cameras() -> None:
    """Remove all cameras from scene."""
    for obj in list(bpy.context.scene.objects):
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)


# ---------------------------------------------------------------------------
# Material helpers
# ---------------------------------------------------------------------------


def _extract_base_color(
    material: bpy.types.Material | None,
) -> tuple[float, float, float, float]:
    """Extract base color from Principled BSDF, fallback to (0.6, 0.6, 0.6, 1)."""
    if material is None or not material.use_nodes:
        return (0.6, 0.6, 0.6, 1.0)
    for node in material.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            ci = node.inputs.get("Base Color")
            if ci is not None:
                v = ci.default_value
                return (v[0], v[1], v[2], v[3])
    return (0.6, 0.6, 0.6, 1.0)


def _get_image_texture_node(
    material: bpy.types.Material | None,
) -> bpy.types.ShaderNodeTexImage | None:
    """Find Image Texture node connected to Principled BSDF Base Color."""
    if material is None or not material.use_nodes:
        return None
    for node in material.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            ci = node.inputs.get("Base Color")
            if ci is not None and ci.is_linked:
                linked = ci.links[0].from_node
                if linked.type == "TEX_IMAGE" and linked.image:
                    return linked
    return None


def _wire_color_source(nodes, links, target_input, tex_node, base_color, loc=(-300, 0)):
    """Wire texture or solid color into shader input."""
    if tex_node is not None and tex_node.image:
        img_tex = nodes.new(type="ShaderNodeTexImage")
        img_tex.image = tex_node.image
        img_tex.location = loc
        links.new(img_tex.outputs["Color"], target_input)
    else:
        target_input.default_value = base_color


def _backup_materials(meshes: list[bpy.types.Object]) -> dict[str, list]:
    """Save original materials so we can restore after style application."""
    backup = {}
    for mesh_obj in meshes:
        mats = []
        for slot in mesh_obj.material_slots:
            mats.append(slot.material)
        backup[mesh_obj.name] = mats
    return backup


def _restore_materials(meshes: list[bpy.types.Object], backup: dict[str, list]) -> None:
    """Restore original materials from backup."""
    for mesh_obj in meshes:
        if mesh_obj.name in backup:
            for i, mat in enumerate(backup[mesh_obj.name]):
                if i < len(mesh_obj.material_slots):
                    mesh_obj.material_slots[i].material = mat


# ---------------------------------------------------------------------------
# Render-time styles (applied via Blender shader nodes before rendering)
# ---------------------------------------------------------------------------


def apply_cel_style(scene: bpy.types.Scene, meshes: list[bpy.types.Object]) -> None:
    """Cel/toon shading with Freestyle outlines."""
    scene.render.use_freestyle = True
    vl = scene.view_layers[0]
    vl.use_freestyle = True
    if vl.freestyle_settings.linesets:
        ls = vl.freestyle_settings.linesets[0]
        ls.linestyle.thickness = CEL_OUTLINE_THICKNESS
        ls.linestyle.color = (0, 0, 0)

    for mesh_obj in meshes:
        for slot in mesh_obj.material_slots:
            original_mat = slot.material
            base_color = _extract_base_color(original_mat)
            tex_node = _get_image_texture_node(original_mat)

            mat = bpy.data.materials.new(name=f"toon_cel_{slot.name}")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            nodes.clear()

            output = nodes.new(type="ShaderNodeOutputMaterial")
            output.location = (900, 0)

            diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
            diffuse.location = (0, 0)
            diffuse.inputs["Color"].default_value = (1, 1, 1, 1)

            s2r = nodes.new(type="ShaderNodeShaderToRGB")
            s2r.location = (200, 0)
            links.new(diffuse.outputs["BSDF"], s2r.inputs["Shader"])

            ramp = nodes.new(type="ShaderNodeValToRGB")
            ramp.location = (400, 0)
            ramp.color_ramp.interpolation = "CONSTANT"
            cr = ramp.color_ramp
            while len(cr.elements) > len(CEL_RAMP_STOPS):
                cr.elements.remove(cr.elements[-1])
            while len(cr.elements) < len(CEL_RAMP_STOPS):
                cr.elements.new(0.5)
            for i, (pos, bright) in enumerate(CEL_RAMP_STOPS):
                cr.elements[i].position = pos
                cr.elements[i].color = (bright, bright, bright, 1.0)

            links.new(s2r.outputs["Color"], ramp.inputs["Fac"])

            mix = nodes.new(type="ShaderNodeMixRGB")
            mix.blend_type = "MULTIPLY"
            mix.location = (600, 0)
            mix.inputs["Fac"].default_value = 1.0
            _wire_color_source(nodes, links, mix.inputs["Color1"], tex_node, base_color)
            links.new(ramp.outputs["Color"], mix.inputs["Color2"])

            emission = nodes.new(type="ShaderNodeEmission")
            emission.location = (750, 0)
            emission.inputs["Strength"].default_value = 1.0
            links.new(mix.outputs["Color"], emission.inputs["Color"])
            links.new(emission.outputs["Emission"], output.inputs["Surface"])

            slot.material = mat


def apply_soft_cel_style(scene: bpy.types.Scene, meshes: list[bpy.types.Object]) -> None:
    """Soft anime cel shading — gradient ramp, no outlines."""
    scene.render.use_freestyle = False

    for mesh_obj in meshes:
        for slot in mesh_obj.material_slots:
            original_mat = slot.material
            base_color = _extract_base_color(original_mat)
            tex_node = _get_image_texture_node(original_mat)

            mat = bpy.data.materials.new(name=f"toon_soft_cel_{slot.name}")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            nodes.clear()

            output = nodes.new(type="ShaderNodeOutputMaterial")
            output.location = (900, 0)

            diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
            diffuse.location = (0, 0)
            diffuse.inputs["Color"].default_value = (1, 1, 1, 1)

            s2r = nodes.new(type="ShaderNodeShaderToRGB")
            s2r.location = (200, 0)
            links.new(diffuse.outputs["BSDF"], s2r.inputs["Shader"])

            ramp = nodes.new(type="ShaderNodeValToRGB")
            ramp.location = (400, 0)
            ramp.color_ramp.interpolation = "LINEAR"
            cr = ramp.color_ramp
            while len(cr.elements) > len(SOFT_CEL_RAMP_STOPS):
                cr.elements.remove(cr.elements[-1])
            while len(cr.elements) < len(SOFT_CEL_RAMP_STOPS):
                cr.elements.new(0.5)
            for i, (pos, bright) in enumerate(SOFT_CEL_RAMP_STOPS):
                cr.elements[i].position = pos
                cr.elements[i].color = (bright, bright, bright, 1.0)

            links.new(s2r.outputs["Color"], ramp.inputs["Fac"])

            mix = nodes.new(type="ShaderNodeMixRGB")
            mix.blend_type = "MULTIPLY"
            mix.location = (600, 0)
            mix.inputs["Fac"].default_value = 1.0
            _wire_color_source(nodes, links, mix.inputs["Color1"], tex_node, base_color)
            links.new(ramp.outputs["Color"], mix.inputs["Color2"])

            emission = nodes.new(type="ShaderNodeEmission")
            emission.location = (750, 0)
            emission.inputs["Strength"].default_value = 1.0
            links.new(mix.outputs["Color"], emission.inputs["Color"])
            links.new(emission.outputs["Emission"], output.inputs["Surface"])

            slot.material = mat


# ---------------------------------------------------------------------------
# Post-render styles (applied to rendered PIL image)
# ---------------------------------------------------------------------------

import cv2


def apply_ink_wash(image: Image.Image, seed: int = 0) -> Image.Image:
    """Ink wash / sumi-e style: bilateral filter + partial desaturation + ink edges."""
    rgba = image.convert("RGBA")
    r, g, b, a = rgba.split()
    rgb = Image.merge("RGB", (r, g, b))
    img_bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

    img_bgr = cv2.bilateralFilter(img_bgr, d=INK_WASH_BILATERAL_D,
                                   sigmaColor=INK_WASH_SIGMA_COLOR,
                                   sigmaSpace=INK_WASH_SIGMA_SPACE)

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    img_hsv[:, :, 1] *= INK_WASH_SATURATION
    img_hsv = np.clip(img_hsv, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    tint_bgr = np.array([INK_WASH_TINT[2], INK_WASH_TINT[1], INK_WASH_TINT[0]], dtype=np.float32)
    gray_mask = (255 - img_bgr[:, :, 0].astype(np.float32)) / 255.0
    for c in range(3):
        img_bgr[:, :, c] = np.clip(
            img_bgr[:, :, c].astype(np.float32) * (1 - gray_mask * 0.3)
            + tint_bgr[c] * gray_mask * 0.3, 0, 255).astype(np.uint8)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, INK_WASH_EDGE_THRESHOLD1, INK_WASH_EDGE_THRESHOLD2)
    kernel = np.ones((INK_WASH_EDGE_THICKNESS, INK_WASH_EDGE_THICKNESS), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edge_mask = edges > 0
    img_bgr[edge_mask] = np.clip(img_bgr[edge_mask].astype(np.int16) - 120, 0, 255).astype(np.uint8)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = Image.merge("RGBA", (*Image.fromarray(img_rgb).split(), a))
    return result


def apply_watercolor(image: Image.Image, seed: int = 0) -> Image.Image:
    """Watercolor style: multi-pass bilateral + saturation boost + grain + soft edges."""
    rgba = image.convert("RGBA")
    r, g, b, a = rgba.split()
    rgb = Image.merge("RGB", (r, g, b))
    img_bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

    for _ in range(WATERCOLOR_BILATERAL_PASSES):
        img_bgr = cv2.bilateralFilter(img_bgr, d=WATERCOLOR_BILATERAL_D,
                                       sigmaColor=WATERCOLOR_SIGMA_COLOR,
                                       sigmaSpace=WATERCOLOR_SIGMA_SPACE)

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * WATERCOLOR_SAT_BOOST, 0, 255)
    img_hsv = img_hsv.astype(np.uint8)
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    rng = np.random.default_rng(seed)
    grain = rng.normal(0, WATERCOLOR_GRAIN_SIGMA * 255, img_bgr.shape)
    img_bgr = np.clip(img_bgr.astype(np.float32) + grain, 0, 255).astype(np.uint8)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, WATERCOLOR_EDGE_THRESHOLD1, WATERCOLOR_EDGE_THRESHOLD2)
    kernel = np.ones((WATERCOLOR_EDGE_THICKNESS, WATERCOLOR_EDGE_THICKNESS), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edge_mask = edges > 0
    edge_bgr = np.array([WATERCOLOR_EDGE_COLOR[2], WATERCOLOR_EDGE_COLOR[1],
                          WATERCOLOR_EDGE_COLOR[0]], dtype=np.uint8)
    img_bgr[edge_mask] = np.clip(
        img_bgr[edge_mask].astype(np.float32) * 0.3 + edge_bgr.astype(np.float32) * 0.7,
        0, 255).astype(np.uint8)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = Image.merge("RGBA", (*Image.fromarray(img_rgb).split(), a))
    return result


# ---------------------------------------------------------------------------
# Style dispatcher
# ---------------------------------------------------------------------------

# Render-time styles modify Blender materials before render
RENDER_TIME_STYLES = {"cel", "soft_cel"}

# Post-render styles modify the rendered PIL image after render
POST_RENDER_STYLES = {"ink_wash", "watercolor"}

# Textured = keep original materials, just render
PASSTHROUGH_STYLES = {"textured"}

ALL_STYLES = RENDER_TIME_STYLES | POST_RENDER_STYLES | PASSTHROUGH_STYLES


def apply_post_render_style(image: Image.Image, style: str, seed: int = 0) -> Image.Image:
    """Apply post-render style transform to a PIL image."""
    if style == "ink_wash":
        return apply_ink_wash(image, seed=seed)
    if style == "watercolor":
        return apply_watercolor(image, seed=seed)
    return image


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------


def discover_models(input_dir: Path) -> list[tuple[str, Path]]:
    """Find all FBX/GLB files, return [(char_id, filepath)]."""
    seen: set[str] = set()
    models = []

    for ext in ("*.fbx", "*.glb"):
        for f in sorted(input_dir.rglob(ext)):
            if f.name.startswith("._") or "withSkin" in f.name:
                continue
            char_id = f.parent.name if f.parent != input_dir else f.stem
            if char_id in seen:
                continue
            seen.add(char_id)
            models.append((char_id, f))

    return models


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    try:
        sep = sys.argv.index("--")
        script_args = sys.argv[sep + 1:]
    except ValueError:
        script_args = []

    parser = argparse.ArgumentParser(description="Render toon-style training images")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=RESOLUTION)
    parser.add_argument("--max-characters", type=int, default=0, help="0 = all")
    parser.add_argument("--only-new", action="store_true", help="Skip existing renders")
    parser.add_argument(
        "--styles", type=str, default=",".join(DEFAULT_STYLES),
        help=f"Comma-separated styles: {','.join(sorted(ALL_STYLES))}",
    )
    parser.add_argument(
        "--angles", type=str, default=",".join(DEFAULT_ANGLES),
        help=f"Comma-separated angles: {','.join(ANGLES.keys())}",
    )
    args = parser.parse_args(script_args)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    styles = [s.strip() for s in args.styles.split(",")]
    angles = [a.strip() for a in args.angles.split(",")]

    for s in styles:
        if s not in ALL_STYLES:
            print(f"ERROR: Unknown style '{s}'. Available: {sorted(ALL_STYLES)}")
            sys.exit(1)
    for a in angles:
        if a not in ANGLES:
            print(f"ERROR: Unknown angle '{a}'. Available: {list(ANGLES.keys())}")
            sys.exit(1)

    models = discover_models(args.input_dir)
    if not models:
        print(f"ERROR: No FBX/GLB files found in {args.input_dir}")
        sys.exit(1)

    if args.max_characters > 0:
        models = models[:args.max_characters]

    print(f"\n{'='*60}")
    print(f"  Toon Style Renderer")
    print(f"  Characters: {len(models)}")
    print(f"  Styles:     {styles}")
    print(f"  Angles:     {angles}")
    print(f"  Resolution: {args.resolution}x{args.resolution}")
    print(f"  Only new:   {args.only_new}")
    print(f"  Output:     {output_dir}")
    print(f"{'='*60}\n")

    # Use EEVEE
    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"

    succeeded = 0
    failed = 0
    skipped = 0
    example_idx = 0

    for i, (char_id, filepath) in enumerate(models):
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

            # Backup original materials for restoration between styles
            original_backup = _backup_materials(meshes)

            char_ok = True
            for style in styles:
                for angle_name in angles:
                    example_dir = output_dir / f"example_{example_idx:06d}"

                    # Check if already done
                    if args.only_new and (example_dir / "image.png").exists():
                        skipped += 1
                        example_idx += 1
                        continue

                    example_dir.mkdir(parents=True, exist_ok=True)

                    # Restore original materials before applying new style
                    _restore_materials(meshes, original_backup)

                    # Apply render-time style
                    if style in RENDER_TIME_STYLES:
                        if style == "cel":
                            apply_cel_style(scene, meshes)
                        elif style == "soft_cel":
                            apply_soft_cel_style(scene, meshes)
                    elif style in PASSTHROUGH_STYLES:
                        pass  # keep originals
                    elif style in POST_RENDER_STYLES:
                        pass  # render with original materials, post-process after

                    # Set up camera
                    remove_cameras()
                    cfg = ANGLES[angle_name]
                    setup_camera(scene, meshes, cfg["azimuth"], cfg["elevation"], args.resolution)

                    # Render
                    tmp_path = example_dir / "image.png"
                    scene.render.filepath = str(tmp_path)
                    bpy.ops.render.render(write_still=True)

                    # Apply post-render style if needed
                    if style in POST_RENDER_STYLES and tmp_path.exists():
                        img = Image.open(tmp_path)
                        img = apply_post_render_style(img, style, seed=example_idx)
                        img.save(tmp_path)

                    # Write metadata
                    metadata = {
                        "character_id": char_id,
                        "source_file": str(filepath),
                        "style": style,
                        "angle": angle_name,
                        "resolution": args.resolution,
                        "source": "toon_render",
                    }
                    (example_dir / "metadata.json").write_text(
                        json.dumps(metadata, indent=2)
                    )

                    example_idx += 1

            elapsed = time.time() - t0
            styles_rendered = len(styles) * len(angles)
            print(f"OK — {styles_rendered} variants ({elapsed:.1f}s)")
            succeeded += 1

        except Exception as e:
            elapsed = time.time() - t0
            print(f"ERROR: {e} ({elapsed:.1f}s)")
            failed += 1

    total_examples = example_idx
    print(f"\n{'='*60}")
    print(f"  Done! {succeeded} characters OK, {failed} failed, {skipped} skipped")
    print(f"  Total examples: {total_examples}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
