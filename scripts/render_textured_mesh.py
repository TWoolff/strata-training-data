#!/usr/bin/env python3
"""Render a textured mesh from multiple angles to preview the result.

Usage::

    blender --background --python scripts/render_textured_mesh.py -- \\
        --mesh output/lichtung_test/object_0.glb \\
        --texture output/lichtung_test/complete_texture.png \\
        --output output/lichtung_test/preview.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy  # type: ignore[import-untyped]
from mathutils import Vector  # type: ignore[import-untyped]
from math import cos, sin, radians

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block_type in [bpy.data.meshes, bpy.data.materials, bpy.data.textures,
                       bpy.data.images, bpy.data.armatures, bpy.data.cameras,
                       bpy.data.lights]:
        for block in block_type:
            if block.users == 0:
                block_type.remove(block)


def apply_texture_to_mesh(mesh_obj, texture_path):
    """Replace the mesh's base color texture with a new one."""
    for slot in mesh_obj.material_slots:
        mat = slot.material
        if mat and mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == "TEX_IMAGE":
                    node.image = bpy.data.images.load(str(texture_path))
                    return
    # No existing texture node — create material
    mat = bpy.data.materials.new(name="TexMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        tex_node = mat.node_tree.nodes.new("ShaderNodeTexImage")
        tex_node.image = bpy.data.images.load(str(texture_path))
        mat.node_tree.links.new(bsdf.inputs["Base Color"], tex_node.outputs["Color"])
    if mesh_obj.material_slots:
        mesh_obj.material_slots[0].material = mat
    else:
        mesh_obj.data.materials.append(mat)


def setup_camera(scene, meshes, azimuth):
    """Orthographic camera at given azimuth, auto-framed to meshes."""
    from math import cos, radians, sin

    old_cam = bpy.data.objects.get("preview_camera")
    if old_cam:
        bpy.data.objects.remove(old_cam, do_unlink=True)

    all_corners = []
    for obj in meshes:
        for c in obj.bound_box:
            all_corners.append(obj.matrix_world @ Vector(c))

    xs = [v.x for v in all_corners]
    ys = [v.y for v in all_corners]
    zs = [v.z for v in all_corners]
    center = Vector(((min(xs)+max(xs))/2, (min(ys)+max(ys))/2, (min(zs)+max(zs))/2))
    width = max(xs) - min(xs)
    depth = max(ys) - min(ys)
    height = max(zs) - min(zs)

    az_rad = radians(float(azimuth))
    apparent_width = abs(width * cos(az_rad)) + abs(depth * sin(az_rad))
    ortho_scale = max(apparent_width, height) * 1.2

    cam_data = bpy.data.cameras.new("preview_camera")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = ortho_scale
    cam_data.clip_start = 0.01
    cam_data.clip_end = 100.0

    cam_obj = bpy.data.objects.new("preview_camera", cam_data)
    scene.collection.objects.link(cam_obj)

    dist = 10.0
    cam_obj.location = (center.x + dist * sin(az_rad), center.y - dist * cos(az_rad), center.z)
    cam_obj.rotation_euler = (radians(90), 0, az_rad)
    scene.camera = cam_obj


def main():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--texture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--angles", type=str, default="0,45,90,180,270")
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args(argv)

    clear_scene()
    if args.mesh.suffix.lower() in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(args.mesh))
    elif args.mesh.suffix.lower() == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(args.mesh))

    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    for m in meshes:
        apply_texture_to_mesh(m, args.texture)

    scene = bpy.context.scene
    if "BLENDER_EEVEE_NEXT" in [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    else:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.film_transparent = True
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.image_settings.file_format = "PNG"

    # Sun light so textures are visible
    light_data = bpy.data.lights.new("sun", type="SUN")
    light_data.energy = 3.0
    light_obj = bpy.data.objects.new("sun", light_data)
    scene.collection.objects.link(light_obj)
    light_obj.rotation_euler = (radians(45), 0, radians(30))

    angles = [int(a) for a in args.angles.split(",")]

    # Render each angle and composite horizontally
    from PIL import Image as PILImage
    imgs = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for angle in angles:
        setup_camera(scene, meshes, angle)
        tmp_path = args.output.parent / f"_tmp_angle_{angle:03d}.png"
        scene.render.filepath = str(tmp_path)
        bpy.ops.render.render(write_still=True)
        imgs.append(PILImage.open(tmp_path).convert("RGBA"))

    # Composite horizontally with labels
    from PIL import ImageDraw
    cell = args.resolution
    grid = PILImage.new("RGBA", (cell * len(angles), cell + 30), (20, 20, 25, 255))
    draw = ImageDraw.Draw(grid)
    for i, (angle, img) in enumerate(zip(angles, imgs)):
        draw.rectangle([i * cell, 0, (i+1) * cell - 1, 29], fill=(40, 40, 50))
        draw.text((i * cell + 8, 8), f"{angle}°", fill=(220, 220, 220))
        bg = PILImage.new("RGBA", (cell, cell), (20, 20, 25, 255))
        grid.paste(PILImage.alpha_composite(bg, img), (i * cell, 30))

    grid.save(args.output)

    # Clean up temp files
    for angle in angles:
        (args.output.parent / f"_tmp_angle_{angle:03d}.png").unlink(missing_ok=True)

    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
