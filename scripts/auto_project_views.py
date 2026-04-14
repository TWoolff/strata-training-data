#!/usr/bin/env python3
"""Auto-detect best camera angle for each illustration, then project all onto mesh.

For each illustration, searches over azimuth/pitch for the camera angle where
the mesh silhouette best matches the illustration alpha. Then projects each
illustration at its detected angle onto the UV texture.

Usage::

    blender --background --python scripts/auto_project_views.py -- \\
        --mesh output/lichtung_test/object_0.glb \\
        --illustration output/lichtung_test/lichtung-character_front.png \\
        --illustration output/lichtung_test/lichtung-character.png \\
        --illustration output/lichtung_test/lichtung-character_back.png \\
        --output_dir output/lichtung_test/auto_projected/
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

from mesh.scripts.texture_projection_trainer import (
    build_visibility_map,
    project_view_to_uv,
    apply_texture_margin,
    ensure_uv_map,
    generate_uv_geometry_maps,
)
from pipeline.config import TEXTURE_BAKE_MARGIN

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


def setup_camera_at(scene, meshes, azimuth, pitch=0.0, scale_multiplier=1.2,
                    name="proj_cam"):
    """Orthographic camera at azimuth + pitch. Returns camera object."""
    old = bpy.data.objects.get(name)
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

    cam_data = bpy.data.cameras.new(name)
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = ortho_scale
    cam_data.clip_start = 0.01
    cam_data.clip_end = 100.0

    cam = bpy.data.objects.new(name, cam_data)
    scene.collection.objects.link(cam)

    dist = 10.0
    cam.location = (
        center.x + dist * cos(pt) * sin(az),
        center.y - dist * cos(pt) * cos(az),
        center.z + dist * sin(pt),
    )
    cam.rotation_euler = (radians(90) - pt, 0, az)
    scene.camera = cam
    return cam


def render_silhouette(scene, resolution):
    """Render and return alpha mask."""
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
    ma = a > threshold
    mb = b > threshold
    intersection = (ma & mb).sum()
    union = (ma | mb).sum()
    return intersection / max(union, 1)


def align_illustration_bbox(path: Path, mesh_silhouette: np.ndarray, resolution: int) -> np.ndarray:
    """Scale/translate illustration so its bbox matches mesh bbox. Returns RGBA."""
    img = Image.open(path).convert("RGBA")
    arr = np.array(img)
    alpha = arr[:, :, 3]

    ys, xs = np.where(alpha > 15)
    if len(ys) == 0:
        return np.zeros((resolution, resolution, 4), dtype=np.uint8)

    iy0, iy1 = ys.min(), ys.max()
    ix0, ix1 = xs.min(), xs.max()

    mys, mxs = np.where(mesh_silhouette > 15)
    if len(mys) == 0:
        return np.zeros((resolution, resolution, 4), dtype=np.uint8)

    my0, my1 = mys.min(), mys.max()
    mx0, mx1 = mxs.min(), mxs.max()

    cropped = img.crop((ix0, iy0, ix1 + 1, iy1 + 1))
    new_w = mx1 - mx0 + 1
    new_h = my1 - my0 + 1
    scaled = cropped.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))
    canvas.paste(scaled, (mx0, my0), scaled)
    return np.array(canvas)


def find_best_angle(
    scene, meshes, illustration_path, resolution,
    az_step_coarse=15, az_step_fine=1, pitch_range=15, pitch_step=5,
):
    """Search azimuth + pitch for best silhouette match."""
    # Coarse search (pitch=0)
    best_iou = -1
    best_az = 0
    for az in range(0, 360, az_step_coarse):
        setup_camera_at(scene, meshes, az, pitch=0)
        mesh_sil = render_silhouette(scene, resolution)
        illus_rgba = align_illustration_bbox(illustration_path, mesh_sil, resolution)
        iou = compute_iou(mesh_sil, illus_rgba[:, :, 3])
        if iou > best_iou:
            best_iou = iou
            best_az = az

    # Fine search
    best_pitch = 0
    for az_offset in range(-az_step_coarse, az_step_coarse + 1, az_step_fine):
        az = (best_az + az_offset) % 360
        for pitch in range(-pitch_range, pitch_range + 1, pitch_step):
            setup_camera_at(scene, meshes, az, pitch=pitch)
            mesh_sil = render_silhouette(scene, resolution)
            illus_rgba = align_illustration_bbox(illustration_path, mesh_sil, resolution)
            iou = compute_iou(mesh_sil, illus_rgba[:, :, 3])
            if iou > best_iou:
                best_iou = iou
                best_az = az
                best_pitch = pitch

    return best_az, best_pitch, best_iou


def project_illustration_at_angle(
    scene, meshes, illustration_path, azimuth, pitch, texture, coverage, incidence_weight, resolution,
):
    """Project illustration (bbox-aligned to mesh silhouette) at given camera angle."""
    camera = setup_camera_at(scene, meshes, azimuth, pitch=pitch, name="strata_proj_camera")
    mesh_sil = render_silhouette(scene, resolution)
    # Restore engine (render_silhouette changes it)
    if "BLENDER_EEVEE_NEXT" in [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    else:
        scene.render.engine = "BLENDER_EEVEE"
    illus = align_illustration_bbox(illustration_path, mesh_sil, resolution)

    visibility = build_visibility_map(scene, camera, meshes)
    n_visible = sum(1 for v in visibility.values() if v)
    logger.info("  %d visible polygons", n_visible)

    project_view_to_uv(
        scene, camera, meshes, illus, visibility,
        texture, coverage, incidence_weight=incidence_weight,
    )

    cam_obj = bpy.data.objects.get("strata_proj_camera")
    if cam_obj is not None:
        bpy.data.objects.remove(cam_obj, do_unlink=True)

    return illus


def inpaint_gaps_palette(texture, coverage, iterations=50):
    tex = texture.copy()
    cov = coverage.copy()
    h, w = cov.shape

    for it in range(iterations):
        filled = cov > 0
        if filled.all():
            break

        padded = np.pad(filled, 1, mode="constant", constant_values=False)
        neighbor_filled = (
            padded[:-2, 1:-1] | padded[2:, 1:-1] |
            padded[1:-1, :-2] | padded[1:-1, 2:]
        )
        to_fill = ~filled & neighbor_filled
        if not to_fill.any():
            break

        padded_tex = np.pad(tex, ((1, 1), (1, 1), (0, 0)), mode="edge")
        padded_cov = np.pad(filled.astype(np.float32), 1, mode="constant")

        ys, xs = np.where(to_fill)
        for y, x in zip(ys, xs):
            neighborhood = padded_tex[y:y+3, x:x+3]
            mask = padded_cov[y:y+3, x:x+3] > 0
            if mask.sum() > 0:
                avg = neighborhood[mask].mean(axis=0)
                tex[y, x] = avg.astype(np.uint8)
                cov[y, x] = 255

    return tex


def main():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--illustration", action="append", required=True,
                        help="Path to illustration PNG (can pass multiple)")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--tex_resolution", type=int, default=1024)
    parser.add_argument("--render_resolution", type=int, default=512)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    illustration_paths = [Path(p) for p in args.illustration]
    for p in illustration_paths:
        if not p.exists():
            logger.error("Illustration not found: %s", p)
            sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    clear_scene()
    if args.mesh.suffix.lower() in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(args.mesh))
    elif args.mesh.suffix.lower() == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(args.mesh))

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    for m in meshes:
        ensure_uv_map(m)

    scene = bpy.context.scene

    # Phase 1: find best angle for each illustration
    logger.info("=" * 60)
    logger.info("Phase 1: Finding best camera angle for each illustration")
    logger.info("=" * 60)

    angles = []
    for path in illustration_paths:
        logger.info("Searching for %s...", path.name)
        az, pitch, iou = find_best_angle(scene, meshes, path, args.render_resolution)
        logger.info("  → best: azimuth=%d° pitch=%d° IoU=%.3f", az, pitch, iou)
        angles.append({
            "illustration": str(path),
            "azimuth": az,
            "pitch": pitch,
            "iou": iou,
        })

    (args.output_dir / "detected_angles.json").write_text(
        json.dumps(angles, indent=2) + "\n"
    )

    # Phase 2: project each illustration at its best angle
    logger.info("=" * 60)
    logger.info("Phase 2: Projecting illustrations onto UV texture")
    logger.info("=" * 60)

    tex_res = args.tex_resolution
    texture = np.zeros((tex_res, tex_res, 4), dtype=np.uint8)
    coverage = np.zeros((tex_res, tex_res), dtype=np.uint8)
    incidence_weight = np.zeros((tex_res, tex_res), dtype=np.float32)

    # Sort by IoU descending so best match wins first (highest priority)
    angles_sorted = sorted(angles, key=lambda x: -x["iou"])

    for a in angles_sorted:
        path = Path(a["illustration"])
        logger.info("Projecting %s at az=%d° pitch=%d°...",
                    path.name, a["azimuth"], a["pitch"])
        aligned = project_illustration_at_angle(
            scene, meshes, path, a["azimuth"], a["pitch"],
            texture, coverage, incidence_weight, args.render_resolution,
        )
        Image.fromarray(aligned).save(
            args.output_dir / f"_aligned_{path.stem}.png"
        )

    # Margin + inpaint
    apply_texture_margin(texture, coverage, TEXTURE_BAKE_MARGIN)
    partial = texture.copy()
    mask = (coverage == 0).astype(np.uint8) * 255
    Image.fromarray(partial).save(args.output_dir / "projected_partial.png")
    Image.fromarray(mask, mode="L").save(args.output_dir / "inpainting_mask.png")

    coverage_pct = (coverage > 0).sum() / (tex_res * tex_res) * 100
    logger.info("Projected coverage: %.1f%%", coverage_pct)

    logger.info("Inpainting gaps...")
    final = inpaint_gaps_palette(texture, coverage, iterations=50)
    Image.fromarray(final).save(args.output_dir / "complete_texture.png")

    # Geometry maps
    logger.info("Generating geometry maps...")
    position_map, normal_map = generate_uv_geometry_maps(meshes, tex_resolution=tex_res)
    Image.fromarray(position_map).save(args.output_dir / "position_map.png")
    Image.fromarray(normal_map).save(args.output_dir / "normal_map.png")

    logger.info("Done! Output in %s", args.output_dir)


if __name__ == "__main__":
    main()
