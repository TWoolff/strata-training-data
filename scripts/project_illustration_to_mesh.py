#!/usr/bin/env python3
"""Project 2D character illustrations onto a 3D mesh's UV texture.

Given a GLB mesh and one or more illustration views (front/back/side/etc.),
projects each illustration onto the visible UV regions of the mesh at that
angle. The result is a complete UV texture composed mostly of the artist's
actual drawings, with small gaps that can be inpainted.

Usage::

    blender --background --python scripts/project_illustration_to_mesh.py -- \\
        --mesh output/lichtung_test/object_0.glb \\
        --view front output/lichtung_test/lichtung-character_front.png:0 \\
        --view back output/lichtung_test/lichtung-character_back.png:180 \\
        --output_dir output/lichtung_test/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import bpy  # type: ignore[import-untyped]
import numpy as np
from PIL import Image

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mesh.scripts.texture_projection_trainer import (
    setup_projection_camera,
    build_visibility_map,
    project_view_to_uv,
    apply_texture_margin,
    ensure_uv_map,
    generate_uv_geometry_maps,
)
from pipeline.config import TEXTURE_BAKE_MARGIN

logger = logging.getLogger(__name__)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block_type in [bpy.data.meshes, bpy.data.materials, bpy.data.textures,
                       bpy.data.images, bpy.data.armatures, bpy.data.cameras,
                       bpy.data.lights]:
        for block in block_type:
            if block.users == 0:
                block_type.remove(block)


def load_illustration_as_render(image_path: Path, render_resolution: int) -> np.ndarray:
    """Load a character illustration and prepare it as a 'rendered view'.

    Simple centered paste. Use `load_illustration_aligned_to_silhouette`
    for proper alignment against a mesh silhouette.
    """
    img = Image.open(image_path).convert("RGBA")
    w, h = img.size
    scale = min(render_resolution / w, render_resolution / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (render_resolution, render_resolution), (0, 0, 0, 0))
    paste_x = (render_resolution - new_w) // 2
    paste_y = (render_resolution - new_h) // 2
    canvas.paste(img_resized, (paste_x, paste_y), img_resized)

    return np.array(canvas)


def render_mesh_silhouette(
    scene: bpy.types.Scene,
    meshes: list,
    angle: float,
    render_resolution: int,
) -> np.ndarray:
    """Render the mesh as a silhouette (alpha mask) from the given angle.

    Uses Workbench flat shading with a solid white color, transparent bg.
    Returns (H, W) uint8 alpha mask.
    """
    from mesh.scripts.texture_projection_trainer import setup_projection_camera
    import tempfile

    camera = setup_projection_camera(scene, meshes, angle)
    scene.render.resolution_x = render_resolution
    scene.render.resolution_y = render_resolution
    scene.render.film_transparent = True

    # Save original engine, switch to workbench
    orig_engine = scene.render.engine
    scene.render.engine = "BLENDER_WORKBENCH"

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        scene.render.filepath = tmp.name
        bpy.ops.render.render(write_still=True)
        img = Image.open(tmp.name).convert("RGBA")
        arr = np.array(img)

    # Clean up camera
    cam_obj = bpy.data.objects.get("strata_proj_camera")
    if cam_obj is not None:
        bpy.data.objects.remove(cam_obj, do_unlink=True)

    scene.render.engine = orig_engine

    return arr[:, :, 3]  # alpha channel


def warp_illustration_tps(
    illustration_path: Path,
    src_points: list[tuple[float, float]],
    dst_points: list[tuple[float, float]],
    render_resolution: int,
    mesh_silhouette: np.ndarray | None = None,
) -> np.ndarray:
    """Warp illustration via thin-plate-spline so src_points map to dst_points.

    src_points: landmarks on the illustration (in illustration pixel coords)
    dst_points: target positions in the render canvas (render_resolution sized)
    mesh_silhouette: mesh silhouette (for adding bbox anchor points)
    """
    from scipy.interpolate import Rbf

    illus = np.array(Image.open(illustration_path).convert("RGBA"))
    src_h, src_w = illus.shape[:2]

    # Add bbox anchors to stabilize the warp
    # Use illustration bbox → mesh silhouette bbox as corner/edge anchors
    illus_alpha = illus[:, :, 3]
    illus_ys, illus_xs = np.where(illus_alpha > 15)
    if len(illus_ys) > 0 and mesh_silhouette is not None:
        mesh_ys, mesh_xs = np.where(mesh_silhouette > 15)
        if len(mesh_ys) > 0:
            illus_bbox = (illus_xs.min(), illus_ys.min(), illus_xs.max(), illus_ys.max())
            mesh_bbox = (mesh_xs.min(), mesh_ys.min(), mesh_xs.max(), mesh_ys.max())

            # Add 8 anchor points: 4 corners + 4 edge midpoints of the bbox
            ix0, iy0, ix1, iy1 = illus_bbox
            mx0, my0, mx1, my1 = mesh_bbox
            imx, imy = (ix0 + ix1) / 2, (iy0 + iy1) / 2
            mmx, mmy = (mx0 + mx1) / 2, (my0 + my1) / 2

            anchors_src = [
                (ix0, iy0), (ix1, iy0), (ix0, iy1), (ix1, iy1),  # corners
                (imx, iy0), (imx, iy1), (ix0, imy), (ix1, imy),  # midpoints
            ]
            anchors_dst = [
                (mx0, my0), (mx1, my0), (mx0, my1), (mx1, my1),
                (mmx, my0), (mmx, my1), (mx0, mmy), (mx1, mmy),
            ]
            src_points = list(src_points) + anchors_src
            dst_points = list(dst_points) + anchors_dst
            logger.info("  Added 8 bbox anchors (total: %d points)", len(src_points))

    if len(src_points) != len(dst_points) or len(src_points) < 3:
        raise ValueError(f"Need ≥3 matched points, got src={len(src_points)}, dst={len(dst_points)}")

    src = np.array(src_points, dtype=np.float32)
    dst = np.array(dst_points, dtype=np.float32)

    # TPS: fit a function that maps dst coords back to src coords (inverse warp)
    rbf_x = Rbf(dst[:, 0], dst[:, 1], src[:, 0], function="thin_plate")
    rbf_y = Rbf(dst[:, 0], dst[:, 1], src[:, 1], function="thin_plate")

    # Sample grid in canvas, look up corresponding source pixel
    canvas_ys, canvas_xs = np.mgrid[0:render_resolution, 0:render_resolution]
    cx_flat = canvas_xs.flatten().astype(np.float32)
    cy_flat = canvas_ys.flatten().astype(np.float32)

    src_x = rbf_x(cx_flat, cy_flat).reshape(render_resolution, render_resolution)
    src_y = rbf_y(cx_flat, cy_flat).reshape(render_resolution, render_resolution)

    # Sample illustration with bilinear interpolation
    out = np.zeros((render_resolution, render_resolution, 4), dtype=np.uint8)

    sx0 = np.floor(src_x).astype(np.int32)
    sy0 = np.floor(src_y).astype(np.int32)
    sx1 = sx0 + 1
    sy1 = sy0 + 1
    fx = src_x - sx0
    fy = src_y - sy0

    valid = (sx0 >= 0) & (sx1 < src_w) & (sy0 >= 0) & (sy1 < src_h)

    for ch in range(4):
        v00 = np.where(valid, illus[np.clip(sy0, 0, src_h-1), np.clip(sx0, 0, src_w-1), ch], 0)
        v01 = np.where(valid, illus[np.clip(sy0, 0, src_h-1), np.clip(sx1, 0, src_w-1), ch], 0)
        v10 = np.where(valid, illus[np.clip(sy1, 0, src_h-1), np.clip(sx0, 0, src_w-1), ch], 0)
        v11 = np.where(valid, illus[np.clip(sy1, 0, src_h-1), np.clip(sx1, 0, src_w-1), ch], 0)
        v = (1-fx)*(1-fy)*v00 + fx*(1-fy)*v01 + (1-fx)*fy*v10 + fx*fy*v11
        out[:, :, ch] = np.clip(v, 0, 255).astype(np.uint8)

    # Zero out anything outside the convex hull of dst points (with some margin)
    # to avoid wild extrapolation
    from matplotlib.path import Path as MplPath
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(dst)
        hull_pts = dst[hull.vertices]
        # Expand hull by 10% to allow padding
        center = hull_pts.mean(axis=0)
        hull_pts = center + (hull_pts - center) * 1.15
        path = MplPath(hull_pts)
        points = np.stack([canvas_xs.flatten(), canvas_ys.flatten()], axis=-1)
        inside = path.contains_points(points).reshape(render_resolution, render_resolution)
        out[~inside, 3] = 0
    except Exception as e:
        logger.warning("  Hull masking failed: %s", e)

    return out


def render_mesh_preview(
    scene: bpy.types.Scene,
    meshes: list,
    angle: float,
    render_resolution: int,
    output_path: Path,
) -> None:
    """Render mesh with its current material (for landmark picking)."""
    from mesh.scripts.texture_projection_trainer import setup_projection_camera

    camera = setup_projection_camera(scene, meshes, angle)
    scene.render.resolution_x = render_resolution
    scene.render.resolution_y = render_resolution
    scene.render.film_transparent = True

    orig_engine = scene.render.engine
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "FLAT"
    scene.display.shading.color_type = "TEXTURE"

    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)

    cam_obj = bpy.data.objects.get("strata_proj_camera")
    if cam_obj is not None:
        bpy.data.objects.remove(cam_obj, do_unlink=True)

    scene.render.engine = orig_engine


def align_illustration_to_silhouette(
    illustration_path: Path,
    silhouette: np.ndarray,
    render_resolution: int,
) -> np.ndarray:
    """Scale and translate the illustration so its silhouette matches the mesh silhouette.

    Computes bbox of both alpha masks and warps the illustration so bbox aligns.
    """
    # Load illustration
    illus = Image.open(illustration_path).convert("RGBA")
    illus_arr = np.array(illus)
    illus_alpha = illus_arr[:, :, 3]

    # Find bbox of illustration's foreground
    illus_ys, illus_xs = np.where(illus_alpha > 15)
    if len(illus_ys) == 0:
        logger.warning("  Illustration has no foreground — using simple paste")
        return load_illustration_as_render(illustration_path, render_resolution)

    illus_y_min, illus_y_max = illus_ys.min(), illus_ys.max()
    illus_x_min, illus_x_max = illus_xs.min(), illus_xs.max()
    illus_w = illus_x_max - illus_x_min + 1
    illus_h = illus_y_max - illus_y_min + 1

    # Find bbox of mesh silhouette
    sil_ys, sil_xs = np.where(silhouette > 15)
    if len(sil_ys) == 0:
        logger.warning("  Mesh silhouette is empty — using simple paste")
        return load_illustration_as_render(illustration_path, render_resolution)

    sil_y_min, sil_y_max = sil_ys.min(), sil_ys.max()
    sil_x_min, sil_x_max = sil_xs.min(), sil_xs.max()
    sil_w = sil_x_max - sil_x_min + 1
    sil_h = sil_y_max - sil_y_min + 1

    logger.info("  Illustration silhouette bbox: %dx%d at (%d,%d)",
                illus_w, illus_h, illus_x_min, illus_y_min)
    logger.info("  Mesh silhouette bbox: %dx%d at (%d,%d)",
                sil_w, sil_h, sil_x_min, sil_y_min)

    # Crop illustration to its bbox
    illus_cropped = illus.crop((illus_x_min, illus_y_min, illus_x_max + 1, illus_y_max + 1))

    # Resize cropped illustration to match mesh silhouette bbox size
    illus_scaled = illus_cropped.resize((sil_w, sil_h), Image.LANCZOS)

    # Paste at mesh silhouette position on render-sized canvas
    canvas = Image.new("RGBA", (render_resolution, render_resolution), (0, 0, 0, 0))
    canvas.paste(illus_scaled, (sil_x_min, sil_y_min), illus_scaled)

    return np.array(canvas)


def project_illustration(
    scene: bpy.types.Scene,
    meshes: list,
    illustration: np.ndarray,
    angle: float,
    texture: np.ndarray,
    coverage: np.ndarray,
    incidence_weight: np.ndarray,
    render_resolution: int,
) -> None:
    """Project one illustration onto the UV texture at the given angle."""
    camera = setup_projection_camera(scene, meshes, angle)
    # Override render resolution to match illustration
    scene.render.resolution_x = render_resolution
    scene.render.resolution_y = render_resolution

    visibility = build_visibility_map(scene, camera, meshes)

    n_visible = sum(1 for v in visibility.values() if v)
    logger.info("  Angle %d°: %d visible polygons", int(angle), n_visible)

    project_view_to_uv(
        scene, camera, meshes, illustration, visibility,
        texture, coverage, incidence_weight=incidence_weight,
    )

    # Clean up camera
    cam_obj = bpy.data.objects.get("strata_proj_camera")
    if cam_obj is not None:
        bpy.data.objects.remove(cam_obj, do_unlink=True)


def inpaint_gaps_palette(
    texture: np.ndarray,
    coverage: np.ndarray,
    iterations: int = 20,
) -> np.ndarray:
    """Simple palette-fill inpainting: expand filled regions into gaps.

    Each iteration, any unfilled texel adjacent to filled texels gets the
    average color of its filled neighbors. Repeats until most gaps are filled.
    """
    tex = texture.copy()
    cov = coverage.copy()
    h, w = cov.shape

    for it in range(iterations):
        # Find unfilled texels with at least one filled neighbor
        filled = cov > 0
        if filled.all():
            break

        # Shift filled mask in 4 directions to find boundary texels
        padded = np.pad(filled, 1, mode="constant", constant_values=False)
        neighbor_filled = (
            padded[:-2, 1:-1] | padded[2:, 1:-1] |
            padded[1:-1, :-2] | padded[1:-1, 2:]
        )
        to_fill = ~filled & neighbor_filled

        if not to_fill.any():
            break

        # For each target, average 3x3 neighborhood where filled
        padded_tex = np.pad(tex, ((1, 1), (1, 1), (0, 0)), mode="edge")
        padded_cov = np.pad(filled.astype(np.float32), 1, mode="constant")

        ys, xs = np.where(to_fill)
        for y, x in zip(ys, xs):
            # 3x3 neighborhood in padded coords
            neighborhood = padded_tex[y:y+3, x:x+3]
            mask = padded_cov[y:y+3, x:x+3] > 0
            if mask.sum() > 0:
                avg = neighborhood[mask].mean(axis=0)
                tex[y, x] = avg.astype(np.uint8)
                cov[y, x] = 255

    logger.info("  Inpaint: %d/%d texels filled after %d iterations",
                int((cov > 0).sum()), h * w, it + 1)
    return tex


def main() -> None:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Project illustrations onto 3D mesh UV texture",
    )
    parser.add_argument("--mesh", type=Path, required=True,
                        help="GLB/FBX mesh file")
    parser.add_argument("--view", action="append", required=True,
                        help="View spec: NAME:PATH:ANGLE, e.g. front:img.png:0")
    parser.add_argument("--output_dir", type=Path, default=Path("output"))
    parser.add_argument("--tex_resolution", type=int, default=1024)
    parser.add_argument("--render_resolution", type=int, default=512)
    parser.add_argument("--inpaint_iterations", type=int, default=50)
    parser.add_argument("--align_silhouette", action="store_true", default=True,
                        help="Align illustration silhouette to mesh silhouette (default: on)")
    parser.add_argument("--no_align", dest="align_silhouette", action="store_false",
                        help="Disable silhouette alignment")
    parser.add_argument("--landmarks", type=Path, default=None,
                        help="JSON file with landmark correspondences for TPS warping")
    parser.add_argument("--prepare_landmarks", action="store_true",
                        help="Render mesh previews and write landmarks_template.json, then exit")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Parse view specs
    views = []
    for v in args.view:
        # Format: NAME:PATH:ANGLE
        parts = v.split(":")
        if len(parts) < 3:
            logger.error("Invalid --view spec: %s (expected NAME:PATH:ANGLE)", v)
            sys.exit(1)
        name = parts[0]
        angle = float(parts[-1])
        path = Path(":".join(parts[1:-1]))
        if not path.exists():
            logger.error("View image not found: %s", path)
            sys.exit(1)
        views.append((name, path, angle))

    logger.info("Loading mesh: %s", args.mesh)
    clear_scene()

    if args.mesh.suffix.lower() in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(args.mesh))
    elif args.mesh.suffix.lower() == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(args.mesh))
    else:
        logger.error("Unsupported mesh format: %s", args.mesh.suffix)
        sys.exit(1)

    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        logger.error("No meshes found")
        sys.exit(1)
    logger.info("Loaded %d mesh(es)", len(meshes))

    for m in meshes:
        ensure_uv_map(m)

    scene = bpy.context.scene
    # Blender 4.x uses BLENDER_EEVEE_NEXT, 5.0+ uses BLENDER_EEVEE
    if "BLENDER_EEVEE_NEXT" in [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    else:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.film_transparent = True

    # Initialize UV texture
    tex_res = args.tex_resolution
    texture = np.zeros((tex_res, tex_res, 4), dtype=np.uint8)
    coverage = np.zeros((tex_res, tex_res), dtype=np.uint8)
    incidence_weight = np.zeros((tex_res, tex_res), dtype=np.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Check for landmarks file
    landmarks_path = args.landmarks
    landmarks_data = None
    if landmarks_path and landmarks_path.exists():
        import json as _json
        landmarks_data = _json.loads(landmarks_path.read_text())
        logger.info("Loaded landmarks from %s (%d views configured)",
                    landmarks_path, len(landmarks_data))
    elif args.prepare_landmarks:
        # Render mesh previews for landmark picking, then exit
        logger.info("Rendering mesh previews for landmark picking...")
        for name, path, angle in views:
            preview_path = args.output_dir / f"_mesh_preview_{name}.png"
            render_mesh_preview(scene, meshes, angle, args.render_resolution, preview_path)
            logger.info("  Saved %s (for mesh view)", preview_path)
        # Write empty landmarks template
        import json as _json
        template = {
            name: {
                "illustration": str(path),
                "mesh_preview": f"_mesh_preview_{name}.png",
                "angle": angle,
                "points": [
                    {"name": "eye_l", "illus": [0, 0], "mesh": [0, 0]},
                    {"name": "eye_r", "illus": [0, 0], "mesh": [0, 0]},
                    {"name": "nose", "illus": [0, 0], "mesh": [0, 0]},
                    {"name": "ear_tip_l", "illus": [0, 0], "mesh": [0, 0]},
                    {"name": "ear_tip_r", "illus": [0, 0], "mesh": [0, 0]},
                    {"name": "chin", "illus": [0, 0], "mesh": [0, 0]},
                    {"name": "tail_base", "illus": [0, 0], "mesh": [0, 0]},
                    {"name": "front_paw_l", "illus": [0, 0], "mesh": [0, 0]},
                    {"name": "front_paw_r", "illus": [0, 0], "mesh": [0, 0]},
                ],
            }
            for name, path, angle in views
        }
        tmpl_path = args.output_dir / "landmarks_template.json"
        tmpl_path.write_text(_json.dumps(template, indent=2) + "\n")
        logger.info("Wrote landmarks template to %s", tmpl_path)
        logger.info("Fill in 'illus' and 'mesh' pixel coords, then re-run with --landmarks")
        return

    # Project each view
    for name, path, angle in views:
        logger.info("Projecting %s (%s, angle=%d°)", name, path.name, int(angle))

        if landmarks_data and name in landmarks_data:
            # Use landmark-based TPS warping
            view_data = landmarks_data[name]
            points = view_data["points"]
            src_pts = [p["illus"] for p in points if p["illus"] != [0, 0] and p["mesh"] != [0, 0]]
            dst_pts = [p["mesh"] for p in points if p["illus"] != [0, 0] and p["mesh"] != [0, 0]]
            if len(src_pts) >= 3:
                logger.info("  Warping via %d landmarks...", len(src_pts))
                # Render mesh silhouette for bbox anchors
                silhouette = render_mesh_silhouette(scene, meshes, angle, args.render_resolution)
                illustration = warp_illustration_tps(
                    path, src_pts, dst_pts, args.render_resolution,
                    mesh_silhouette=silhouette,
                )
            else:
                logger.warning("  Not enough landmarks (%d), falling back to silhouette align", len(src_pts))
                silhouette = render_mesh_silhouette(scene, meshes, angle, args.render_resolution)
                illustration = align_illustration_to_silhouette(path, silhouette, args.render_resolution)
        elif args.align_silhouette:
            logger.info("  Rendering mesh silhouette...")
            silhouette = render_mesh_silhouette(scene, meshes, angle, args.render_resolution)
            illustration = align_illustration_to_silhouette(
                path, silhouette, args.render_resolution,
            )
        else:
            illustration = load_illustration_as_render(path, args.render_resolution)

        Image.fromarray(illustration).save(
            args.output_dir / f"_aligned_{name}.png"
        )
        project_illustration(
            scene, meshes, illustration, angle,
            texture, coverage, incidence_weight, args.render_resolution,
        )

    # Apply margin to reduce seam artifacts
    apply_texture_margin(texture, coverage, TEXTURE_BAKE_MARGIN)

    # Save partial texture (before inpainting)
    partial = texture.copy()
    mask = (coverage == 0).astype(np.uint8) * 255

    Image.fromarray(partial).save(args.output_dir / "projected_partial.png")
    Image.fromarray(mask, mode="L").save(args.output_dir / "inpainting_mask.png")

    coverage_pct = (coverage > 0).sum() / (tex_res * tex_res) * 100
    logger.info("Projected coverage: %.1f%% (%d texels)",
                coverage_pct, int((coverage > 0).sum()))

    # Simple inpaint via palette expansion
    logger.info("Inpainting gaps (palette fill, %d iterations)...", args.inpaint_iterations)
    final = inpaint_gaps_palette(texture, coverage, iterations=args.inpaint_iterations)
    Image.fromarray(final).save(args.output_dir / "complete_texture.png")

    # Also generate geometry maps (useful for ControlNet if we retrain)
    logger.info("Generating geometry maps...")
    position_map, normal_map = generate_uv_geometry_maps(meshes, tex_resolution=tex_res)
    Image.fromarray(position_map).save(args.output_dir / "position_map.png")
    Image.fromarray(normal_map).save(args.output_dir / "normal_map.png")

    logger.info("Done! Output in %s", args.output_dir)
    logger.info("  projected_partial.png — texture from illustrations only")
    logger.info("  inpainting_mask.png — where gaps exist (white = gap)")
    logger.info("  complete_texture.png — with palette-fill inpainting")
    logger.info("  position_map.png, normal_map.png — for future ControlNet refinement")


if __name__ == "__main__":
    main()
