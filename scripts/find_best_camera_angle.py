#!/usr/bin/env python3
"""Find the camera angle that best matches an illustration silhouette.

Given a mesh and an illustration, searches over azimuth/pitch angles to find
the camera orientation where the mesh silhouette maximally overlaps with the
illustration alpha. This is the angle SAM 3D likely used when generating the
mesh from that illustration.

Usage::

    blender --background --python scripts/find_best_camera_angle.py -- \\
        --mesh output/lichtung_test/object_0.glb \\
        --illustration output/lichtung_test/lichtung-character.png \\
        --output_dir output/lichtung_test/

Outputs:
    - best_angle.json — best azimuth/pitch found + IoU score
    - _silhouette_best.png — mesh silhouette at best angle
    - _silhouette_compare.png — side-by-side mesh silhouette vs illustration
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from math import cos, radians, sin
from pathlib import Path

import bpy  # type: ignore[import-untyped]
import numpy as np
from mathutils import Vector  # type: ignore[import-untyped]
from PIL import Image

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logger = logging.getLogger(__name__)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block_type in [bpy.data.meshes, bpy.data.materials, bpy.data.textures,
                       bpy.data.images, bpy.data.armatures, bpy.data.cameras,
                       bpy.data.lights]:
        for block in block_type:
            if block.users == 0:
                block_type.remove(block)


def setup_camera(scene, meshes, azimuth, pitch=0.0, scale_multiplier=1.2):
    """Orthographic camera at azimuth + pitch."""
    old = bpy.data.objects.get("search_camera")
    if old:
        bpy.data.objects.remove(old, do_unlink=True)

    corners = [m.matrix_world @ Vector(c) for m in meshes for c in m.bound_box]
    center = Vector((
        sum(v.x for v in corners)/len(corners),
        sum(v.y for v in corners)/len(corners),
        sum(v.z for v in corners)/len(corners),
    ))
    width = max(v.x for v in corners) - min(v.x for v in corners)
    depth = max(v.y for v in corners) - min(v.y for v in corners)
    height = max(v.z for v in corners) - min(v.z for v in corners)

    az = radians(float(azimuth))
    pt = radians(float(pitch))
    apparent_width = abs(width * cos(az)) + abs(depth * sin(az))
    ortho_scale = max(apparent_width, height) * scale_multiplier

    cam_data = bpy.data.cameras.new("search_camera")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = ortho_scale
    cam_data.clip_start = 0.01
    cam_data.clip_end = 100.0

    cam = bpy.data.objects.new("search_camera", cam_data)
    scene.collection.objects.link(cam)

    dist = 10.0
    # Horizontal component
    cx = center.x + dist * cos(pt) * sin(az)
    cy = center.y - dist * cos(pt) * cos(az)
    cz = center.z + dist * sin(pt)
    cam.location = (cx, cy, cz)
    cam.rotation_euler = (radians(90) - pt, 0, az)
    scene.camera = cam


def render_silhouette(scene, resolution):
    """Render mesh and return alpha channel as uint8."""
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = True

    orig = scene.render.engine
    scene.render.engine = "BLENDER_WORKBENCH"

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        scene.render.filepath = tmp.name
        bpy.ops.render.render(write_still=True)
        img = Image.open(tmp.name).convert("RGBA")
        alpha = np.array(img)[:, :, 3]

    scene.render.engine = orig
    return alpha


def compute_iou(a: np.ndarray, b: np.ndarray, threshold: int = 15) -> float:
    """IoU between two binary masks (after thresholding alpha)."""
    ma = a > threshold
    mb = b > threshold
    intersection = (ma & mb).sum()
    union = (ma | mb).sum()
    return intersection / max(union, 1)


def load_illustration_alpha(path: Path, resolution: int) -> np.ndarray:
    """Load illustration, resize preserving aspect, return alpha."""
    img = Image.open(path).convert("RGBA")
    w, h = img.size
    scale = min(resolution / w, resolution / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))
    canvas.paste(img, ((resolution - new_w) // 2, (resolution - new_h) // 2), img)
    return np.array(canvas)[:, :, 3]


def align_illustration_bbox(path: Path, mesh_silhouette: np.ndarray, resolution: int) -> np.ndarray:
    """Scale/translate illustration alpha so its bbox matches the mesh silhouette bbox."""
    img = Image.open(path).convert("RGBA")
    arr = np.array(img)
    alpha = arr[:, :, 3]

    ys, xs = np.where(alpha > 15)
    if len(ys) == 0:
        return load_illustration_alpha(path, resolution)

    iy0, iy1 = ys.min(), ys.max()
    ix0, ix1 = xs.min(), xs.max()

    mys, mxs = np.where(mesh_silhouette > 15)
    if len(mys) == 0:
        return load_illustration_alpha(path, resolution)

    my0, my1 = mys.min(), mys.max()
    mx0, mx1 = mxs.min(), mxs.max()

    # Crop illustration to bbox
    cropped = img.crop((ix0, iy0, ix1 + 1, iy1 + 1))
    new_w = mx1 - mx0 + 1
    new_h = my1 - my0 + 1
    scaled = cropped.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))
    canvas.paste(scaled, (mx0, my0), scaled)
    return np.array(canvas)[:, :, 3]


def main():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--illustration", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--azimuth_step", type=int, default=15)
    parser.add_argument("--fine_azimuth_step", type=int, default=1,
                        help="Step size for fine search around best coarse angle")
    parser.add_argument("--pitch_range", type=int, default=15,
                        help="Pitch search range in degrees (+/-)")
    parser.add_argument("--pitch_step", type=int, default=5)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    clear_scene()
    if args.mesh.suffix.lower() in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(args.mesh))
    elif args.mesh.suffix.lower() == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(args.mesh))

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    scene = bpy.context.scene

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: coarse azimuth search at pitch=0
    logger.info("Phase 1: coarse azimuth search (step=%d°)", args.azimuth_step)
    best_iou_coarse = -1
    best_az_coarse = 0
    results_coarse = []

    for az in range(0, 360, args.azimuth_step):
        setup_camera(scene, meshes, az, pitch=0)
        mesh_sil = render_silhouette(scene, args.resolution)
        # Align illustration to mesh bbox before IoU to get fair comparison
        illus_alpha = align_illustration_bbox(args.illustration, mesh_sil, args.resolution)
        iou = compute_iou(mesh_sil, illus_alpha)
        results_coarse.append((az, iou))
        if iou > best_iou_coarse:
            best_iou_coarse = iou
            best_az_coarse = az
        logger.info("  az=%3d°  IoU=%.3f", az, iou)

    logger.info("Best coarse: azimuth=%d° IoU=%.3f", best_az_coarse, best_iou_coarse)

    # Phase 2: fine azimuth + pitch search near best
    logger.info("Phase 2: fine search around az=%d°", best_az_coarse)
    best_iou = best_iou_coarse
    best_az = best_az_coarse
    best_pitch = 0

    az_range = args.azimuth_step
    for az_offset in range(-az_range, az_range + 1, args.fine_azimuth_step):
        for pitch in range(-args.pitch_range, args.pitch_range + 1, args.pitch_step):
            az = (best_az_coarse + az_offset) % 360
            setup_camera(scene, meshes, az, pitch=pitch)
            mesh_sil = render_silhouette(scene, args.resolution)
            illus_alpha = align_illustration_bbox(args.illustration, mesh_sil, args.resolution)
            iou = compute_iou(mesh_sil, illus_alpha)
            if iou > best_iou:
                best_iou = iou
                best_az = az
                best_pitch = pitch

    logger.info("Best overall: azimuth=%d° pitch=%d° IoU=%.3f",
                best_az, best_pitch, best_iou)

    # Save best silhouette render for inspection
    setup_camera(scene, meshes, best_az, pitch=best_pitch)
    best_sil = render_silhouette(scene, args.resolution)
    Image.fromarray(best_sil, mode="L").save(args.output_dir / "_silhouette_best.png")

    # Also save aligned illustration alpha for comparison
    best_illus = align_illustration_bbox(args.illustration, best_sil, args.resolution)
    Image.fromarray(best_illus, mode="L").save(args.output_dir / "_silhouette_illus.png")

    # Visualize overlap: green = both, red = mesh only, blue = illustration only
    overlap = np.zeros((args.resolution, args.resolution, 4), dtype=np.uint8)
    mesh_mask = best_sil > 15
    illus_mask = best_illus > 15
    overlap[mesh_mask & illus_mask] = [0, 255, 0, 200]
    overlap[mesh_mask & ~illus_mask] = [255, 0, 0, 200]
    overlap[~mesh_mask & illus_mask] = [0, 100, 255, 200]
    Image.fromarray(overlap, "RGBA").save(args.output_dir / "_silhouette_compare.png")

    # Save result
    result = {
        "best_azimuth": best_az,
        "best_pitch": best_pitch,
        "best_iou": best_iou,
        "coarse_results": results_coarse,
    }
    (args.output_dir / "best_angle.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    logger.info("Saved best_angle.json")
    logger.info("  _silhouette_best.png: mesh silhouette at best angle")
    logger.info("  _silhouette_illus.png: illustration silhouette (bbox-aligned)")
    logger.info("  _silhouette_compare.png: green=overlap, red=mesh-only, blue=illus-only")


if __name__ == "__main__":
    main()
