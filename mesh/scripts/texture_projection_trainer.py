"""Generate (partial_texture, complete_texture) training pairs for neural inpainting.

For each character × pose, renders from dense camera angles (every 15°, 24 views)
and projects the rendered colors onto UV texture maps.  Produces:

- ``complete_texture.png`` — UV map composited from all 24 views (full coverage)
- ``partial_texture.png`` — UV map from 3 views only (front, three-quarter, back)
- ``inpainting_mask.png`` — Binary mask: 255 where texture is missing in partial

Training pair: (partial_texture, inpainting_mask) → complete_texture

Requires Blender 4.0+ (uses ``bpy`` for rendering and ray casting).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from math import cos, radians, sin
from pathlib import Path
from typing import Any

import bpy  # type: ignore[import-untyped]
import numpy as np
from bpy_extras.object_utils import world_to_camera_view  # type: ignore[import-untyped]
from mathutils import Vector  # type: ignore[import-untyped]
from PIL import Image

# Append project root to sys.path so pipeline imports work when invoked via
# ``blender --background --python mesh/scripts/texture_projection_trainer.py``
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from pipeline.config import (
    CAMERA_CLIP_END,
    CAMERA_CLIP_START,
    CAMERA_DISTANCE,
    CAMERA_PADDING,
    CAMERA_TYPE,
    RENDER_RESOLUTION,
    TEXTURE_BAKE_MARGIN,
    TEXTURE_DENSE_ANGLES,
    TEXTURE_PARTIAL_ANGLES,
    TEXTURE_RESOLUTION,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# UV helpers
# ---------------------------------------------------------------------------


def ensure_uv_map(mesh_obj: bpy.types.Object) -> str:
    """Ensure the mesh has a UV map, creating one via Smart UV Project if needed.

    Args:
        mesh_obj: A Blender mesh object.

    Returns:
        The name of the active UV map.
    """
    mesh_data = mesh_obj.data
    if mesh_data.uv_layers:
        return mesh_data.uv_layers.active.name

    # Select only this object and run Smart UV Project
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
    bpy.ops.object.mode_set(mode="OBJECT")

    logger.info("Created Smart UV Project for %s", mesh_obj.name)
    return mesh_data.uv_layers.active.name


# ---------------------------------------------------------------------------
# Camera setup (mirrors pipeline/renderer.py setup_camera but configurable res)
# ---------------------------------------------------------------------------


def _combined_bounding_box(
    meshes: list[bpy.types.Object],
) -> tuple[Vector, Vector]:
    """Compute the world-space axis-aligned bounding box of all meshes."""
    all_corners: list[Vector] = []
    for obj in meshes:
        for corner in obj.bound_box:
            all_corners.append(obj.matrix_world @ Vector(corner))

    xs = [v.x for v in all_corners]
    ys = [v.y for v in all_corners]
    zs = [v.z for v in all_corners]

    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


def setup_projection_camera(
    scene: bpy.types.Scene,
    meshes: list[bpy.types.Object],
    azimuth: float,
) -> bpy.types.Object:
    """Create an orthographic camera for texture projection at the given azimuth.

    Same orbit logic as ``pipeline.renderer.setup_camera`` but uses
    ``RENDER_RESOLUTION`` for the projection render.

    Args:
        scene: The Blender scene.
        meshes: Character mesh objects to frame.
        azimuth: Horizontal rotation in degrees (0 = front).

    Returns:
        The camera object.
    """
    # Remove previous projection camera if present
    old_cam = bpy.data.objects.get("strata_proj_camera")
    if old_cam is not None:
        bpy.data.objects.remove(old_cam, do_unlink=True)

    bbox_min, bbox_max = _combined_bounding_box(meshes)
    bbox_center = (bbox_min + bbox_max) / 2

    width = bbox_max.x - bbox_min.x
    height = bbox_max.z - bbox_min.z
    depth = bbox_max.y - bbox_min.y

    az_rad = radians(azimuth)
    apparent_width = abs(width * cos(az_rad)) + abs(depth * sin(az_rad))
    ortho_scale = max(apparent_width, height) * (1.0 + 2.0 * CAMERA_PADDING)

    cam_data = bpy.data.cameras.new(name="strata_proj_camera")
    cam_data.type = CAMERA_TYPE
    cam_data.ortho_scale = ortho_scale
    cam_data.clip_start = CAMERA_CLIP_START
    cam_data.clip_end = CAMERA_CLIP_END

    cam_obj = bpy.data.objects.new(name="strata_proj_camera", object_data=cam_data)
    scene.collection.objects.link(cam_obj)

    cam_x = bbox_center.x + CAMERA_DISTANCE * sin(az_rad)
    cam_y = bbox_center.y - CAMERA_DISTANCE * cos(az_rad)
    cam_z = bbox_center.z
    cam_obj.location = (cam_x, cam_y, cam_z)
    cam_obj.rotation_euler = (radians(90), 0, az_rad)

    scene.camera = cam_obj
    scene.render.resolution_x = RENDER_RESOLUTION
    scene.render.resolution_y = RENDER_RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True

    return cam_obj


# ---------------------------------------------------------------------------
# Render a single view
# ---------------------------------------------------------------------------


def render_view(scene: bpy.types.Scene, output_path: Path) -> Path:
    """Render the current camera view and save to *output_path*.

    Uses whatever render engine and materials are currently configured.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)
    return output_path


# ---------------------------------------------------------------------------
# Visibility testing via BVH ray casting
# ---------------------------------------------------------------------------


def build_visibility_map(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    meshes: list[bpy.types.Object],
) -> dict[tuple[int, int], bool]:
    """Determine which (mesh_index, polygon_index) pairs are visible from *camera*.

    Uses depsgraph ray casting for accurate occlusion testing.

    Returns:
        Dict mapping (mesh_idx, poly_idx) → True for visible polygons only.
        Absent entries are implicitly not visible (callers use ``.get(..., False)``).
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cam_pos = Vector(camera.location)
    visibility: dict[tuple[int, int], bool] = {}

    for mesh_idx, mesh_obj in enumerate(meshes):
        eval_obj = mesh_obj.evaluated_get(depsgraph)
        mesh_data = eval_obj.data

        for poly_idx, poly in enumerate(mesh_data.polygons):
            face_center_world = eval_obj.matrix_world @ Vector(poly.center)
            face_normal_world = (eval_obj.matrix_world.to_3x3() @ Vector(poly.normal)).normalized()

            # Back-face test: face must be oriented toward the camera
            view_dir = (cam_pos - face_center_world).normalized()
            if face_normal_world.dot(view_dir) <= 0:
                continue

            # Ray cast from face center toward camera to check for occlusion.
            # Offset origin slightly along normal to avoid self-intersection.
            ray_origin = face_center_world + face_normal_world * 0.001
            ray_dir = (cam_pos - ray_origin).normalized()
            ray_length = (cam_pos - ray_origin).length

            hit, _loc, _normal, _idx, hit_obj, _matrix = scene.ray_cast(
                depsgraph, ray_origin, ray_dir, distance=ray_length
            )

            # Visible if we didn't hit any other object
            if not (hit and hit_obj is not None and hit_obj.name != mesh_obj.name):
                visibility[(mesh_idx, poly_idx)] = True

    return visibility


# ---------------------------------------------------------------------------
# UV-space texture projection (pure-geometry, per-triangle rasterization)
# ---------------------------------------------------------------------------


def _barycentric_coords(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> tuple[float, float, float]:
    """Compute barycentric coordinates of point *p* w.r.t. triangle (a, b, c).

    All inputs are 2D arrays of shape (2,).
    """
    v0 = c - a
    v1 = b - a
    v2 = p - a

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-12:
        return -1.0, -1.0, -1.0  # degenerate triangle

    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    return 1.0 - u - v, v, u


def project_view_to_uv(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    meshes: list[bpy.types.Object],
    rendered_image: np.ndarray,
    visibility: dict[tuple[int, int], bool],
    texture: np.ndarray,
    coverage: np.ndarray,
) -> None:
    """Project a rendered camera view onto the UV texture map in-place.

    For each visible polygon, projects its vertices to screen space, samples
    the rendered image, and writes colors to the UV texture at the polygon's
    UV coordinates.

    Args:
        scene: Blender scene with the camera set.
        camera: The camera used to render ``rendered_image``.
        meshes: Character mesh objects.
        rendered_image: RGBA array (H, W, 4) uint8 from the camera view.
        visibility: Per-face visibility from ``build_visibility_map``.
        texture: UV texture to fill, (tex_h, tex_w, 4) uint8. Modified in-place.
        coverage: Binary coverage array (tex_h, tex_w) uint8 — set to 255
            for texels that have been written. Modified in-place.
    """
    render_h, render_w = rendered_image.shape[:2]
    tex_h, tex_w = texture.shape[:2]
    depsgraph = bpy.context.evaluated_depsgraph_get()

    for mesh_idx, mesh_obj in enumerate(meshes):
        eval_obj = mesh_obj.evaluated_get(depsgraph)
        mesh_data = eval_obj.data

        if not mesh_data.uv_layers:
            continue
        uv_layer = mesh_data.uv_layers.active.data

        for poly_idx, poly in enumerate(mesh_data.polygons):
            if not visibility.get((mesh_idx, poly_idx), False):
                continue

            verts = list(poly.vertices)
            loop_indices = list(poly.loop_indices)

            if len(verts) < 3:
                continue

            # Get world-space vertex positions
            world_positions = [eval_obj.matrix_world @ mesh_data.vertices[vi].co for vi in verts]

            # Project to screen space (normalized 0..1)
            screen_coords = []
            for wp in world_positions:
                sc = world_to_camera_view(scene, camera, wp)
                screen_coords.append(np.array([sc.x, sc.y]))

            # Get UV coordinates
            uv_coords = []
            for li in loop_indices:
                uv = uv_layer[li].uv
                uv_coords.append(np.array([uv[0], uv[1]]))

            # Triangulate polygon (fan triangulation from first vertex)
            for tri_idx in range(1, len(verts) - 1):
                sc_a, sc_b, sc_c = (
                    screen_coords[0],
                    screen_coords[tri_idx],
                    screen_coords[tri_idx + 1],
                )
                uv_a, uv_b, uv_c = (
                    uv_coords[0],
                    uv_coords[tri_idx],
                    uv_coords[tri_idx + 1],
                )

                _rasterize_triangle_to_uv(
                    sc_a,
                    sc_b,
                    sc_c,
                    uv_a,
                    uv_b,
                    uv_c,
                    rendered_image,
                    render_w,
                    render_h,
                    texture,
                    coverage,
                    tex_w,
                    tex_h,
                )


def _rasterize_triangle_to_uv(
    sc_a: np.ndarray,
    sc_b: np.ndarray,
    sc_c: np.ndarray,
    uv_a: np.ndarray,
    uv_b: np.ndarray,
    uv_c: np.ndarray,
    rendered_image: np.ndarray,
    render_w: int,
    render_h: int,
    texture: np.ndarray,
    coverage: np.ndarray,
    tex_w: int,
    tex_h: int,
) -> None:
    """Rasterize a single triangle in UV space, sampling from the rendered image.

    For each texel inside the UV triangle, computes barycentric coordinates,
    maps to screen space via the same barycentrics, samples the rendered image,
    and writes to the texture.
    """
    # UV bounding box in texel coords
    uv_pixels = np.array([uv_a, uv_b, uv_c]) * np.array([tex_w, tex_h])
    min_u = max(0, int(np.floor(uv_pixels[:, 0].min())))
    max_u = min(tex_w - 1, int(np.ceil(uv_pixels[:, 0].max())))
    min_v = max(0, int(np.floor(uv_pixels[:, 1].min())))
    max_v = min(tex_h - 1, int(np.ceil(uv_pixels[:, 1].max())))

    for ty in range(min_v, max_v + 1):
        for tx in range(min_u, max_u + 1):
            # Texel center in UV space (0..1)
            p_uv = np.array([(tx + 0.5) / tex_w, (ty + 0.5) / tex_h])

            w0, w1, w2 = _barycentric_coords(p_uv, uv_a, uv_b, uv_c)

            # Skip if outside triangle
            if w0 < -1e-4 or w1 < -1e-4 or w2 < -1e-4:
                continue

            # Interpolate to screen space
            sc_point = w0 * sc_a + w1 * sc_b + w2 * sc_c

            # Screen coords to pixel coords (Blender: origin bottom-left)
            px = int(sc_point[0] * render_w)
            py = int(sc_point[1] * render_h)

            if 0 <= px < render_w and 0 <= py < render_h:
                # Blender renders with origin at top-left in saved image,
                # but world_to_camera_view has origin at bottom-left.
                img_y = render_h - 1 - py
                color = rendered_image[img_y, px]

                # Only write if the rendered pixel is non-transparent
                if color[3] > 0:
                    # UV texture: origin at bottom-left, image array at top-left
                    tex_y = tex_h - 1 - ty
                    texture[tex_y, tx] = color
                    coverage[tex_y, tx] = 255


# ---------------------------------------------------------------------------
# Margin / bleed pass for UV seams
# ---------------------------------------------------------------------------


def apply_texture_margin(
    texture: np.ndarray,
    coverage: np.ndarray,
    margin: int,
) -> None:
    """Extend filled texels outward by *margin* pixels to reduce UV seam artifacts.

    Uses iterative dilation: each pass extends by 1 pixel in the 4-connected
    neighborhood. Modifies *texture* and *coverage* in-place.
    """
    h, w = coverage.shape
    for _ in range(margin):
        new_texture = texture.copy()
        new_coverage = coverage.copy()

        # Find unfilled texels adjacent to filled ones
        for y in range(h):
            for x in range(w):
                if coverage[y, x] > 0:
                    continue
                # Check 4-connected neighbors
                neighbors: list[tuple[int, int]] = []
                if y > 0 and coverage[y - 1, x] > 0:
                    neighbors.append((y - 1, x))
                if y < h - 1 and coverage[y + 1, x] > 0:
                    neighbors.append((y + 1, x))
                if x > 0 and coverage[y, x - 1] > 0:
                    neighbors.append((y, x - 1))
                if x < w - 1 and coverage[y, x + 1] > 0:
                    neighbors.append((y, x + 1))

                if neighbors:
                    # Average neighbor colors
                    avg = np.zeros(4, dtype=np.float32)
                    for ny, nx in neighbors:
                        avg += texture[ny, nx].astype(np.float32)
                    avg /= len(neighbors)
                    new_texture[y, x] = avg.astype(np.uint8)
                    new_coverage[y, x] = 255

        texture[:] = new_texture
        coverage[:] = new_coverage


# ---------------------------------------------------------------------------
# Multi-view texture compositing
# ---------------------------------------------------------------------------


def composite_texture_from_views(
    scene: bpy.types.Scene,
    meshes: list[bpy.types.Object],
    angles: list[int],
    render_dir: Path,
    tex_resolution: int = TEXTURE_RESOLUTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Render from multiple angles and composite into a UV texture map.

    For each angle:
    1. Set up camera at the given azimuth
    2. Render the color image
    3. Compute face visibility
    4. Project rendered pixels onto UV space

    Args:
        scene: Blender scene with character loaded and materials set.
        meshes: Character mesh objects.
        angles: List of azimuth angles in degrees.
        render_dir: Directory for temporary per-angle renders.
        tex_resolution: Output texture resolution (square).

    Returns:
        Tuple of (texture, coverage):
        - texture: RGBA array (tex_resolution, tex_resolution, 4) uint8
        - coverage: Binary array (tex_resolution, tex_resolution) uint8
    """
    texture = np.zeros((tex_resolution, tex_resolution, 4), dtype=np.uint8)
    coverage = np.zeros((tex_resolution, tex_resolution), dtype=np.uint8)

    # Ensure all meshes have UV maps
    for mesh_obj in meshes:
        ensure_uv_map(mesh_obj)

    render_dir.mkdir(parents=True, exist_ok=True)

    for angle in angles:
        logger.info("Projecting from azimuth=%d°", angle)

        # Set up camera
        camera = setup_projection_camera(scene, meshes, float(angle))

        # Render
        render_path = render_dir / f"view_{angle:03d}.png"
        render_view(scene, render_path)

        # Load rendered image
        rendered = np.array(Image.open(render_path).convert("RGBA"))

        # Compute visibility
        visibility = build_visibility_map(scene, camera, meshes)

        # Project onto UV
        project_view_to_uv(
            scene,
            camera,
            meshes,
            rendered,
            visibility,
            texture,
            coverage,
        )

        # Clean up camera
        cam_obj = bpy.data.objects.get("strata_proj_camera")
        if cam_obj is not None:
            bpy.data.objects.remove(cam_obj, do_unlink=True)

    # Apply margin to reduce seam artifacts
    apply_texture_margin(texture, coverage, TEXTURE_BAKE_MARGIN)

    return texture, coverage


# ---------------------------------------------------------------------------
# Inpainting mask computation
# ---------------------------------------------------------------------------


def compute_inpainting_mask(
    partial_coverage: np.ndarray,
    complete_coverage: np.ndarray,
) -> np.ndarray:
    """Compute binary inpainting mask: 255 where partial is missing but complete has data.

    Args:
        partial_coverage: Coverage from partial views (H, W) uint8.
        complete_coverage: Coverage from complete views (H, W) uint8.

    Returns:
        Binary mask (H, W) uint8 — 255 where inpainting is needed, 0 elsewhere.
    """
    partial_filled = partial_coverage > 0
    complete_filled = complete_coverage > 0

    # Regions present in complete but missing in partial
    needs_inpainting = complete_filled & ~partial_filled

    return (needs_inpainting * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Top-level training pair generation
# ---------------------------------------------------------------------------


def generate_texture_pairs(
    scene: bpy.types.Scene,
    meshes: list[bpy.types.Object],
    output_dir: Path,
    character_id: str,
    pose_id: int,
    *,
    dense_angles: list[int] | None = None,
    partial_angles: list[int] | None = None,
    tex_resolution: int = TEXTURE_RESOLUTION,
) -> dict[str, Any]:
    """Generate complete/partial texture training pair for one character × pose.

    Args:
        scene: Blender scene with character posed and materials ready.
        meshes: Character mesh objects.
        output_dir: Root output directory.
        character_id: Identifier for the character.
        pose_id: Pose index.
        dense_angles: Azimuth angles for complete texture (default: every 15°).
        partial_angles: Azimuth angles for partial texture (default: front/3-4/back).
        tex_resolution: Output UV texture resolution.

    Returns:
        Metadata dict with paths and statistics.
    """
    if dense_angles is None:
        dense_angles = TEXTURE_DENSE_ANGLES
    if partial_angles is None:
        partial_angles = TEXTURE_PARTIAL_ANGLES

    pair_dir = output_dir / f"{character_id}_pose_{pose_id:02d}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    render_dir = pair_dir / "_renders"

    # 1. Complete texture: render from all dense angles
    logger.info(
        "Generating complete texture for %s pose %d (%d views)",
        character_id,
        pose_id,
        len(dense_angles),
    )
    complete_texture, complete_coverage = composite_texture_from_views(
        scene,
        meshes,
        dense_angles,
        render_dir / "complete",
        tex_resolution,
    )

    # 2. Partial texture: render from subset of angles
    logger.info(
        "Generating partial texture for %s pose %d (%d views)",
        character_id,
        pose_id,
        len(partial_angles),
    )
    partial_texture, partial_coverage = composite_texture_from_views(
        scene,
        meshes,
        partial_angles,
        render_dir / "partial",
        tex_resolution,
    )

    # 3. Compute inpainting mask
    inpainting_mask = compute_inpainting_mask(partial_coverage, complete_coverage)

    # 4. Save outputs
    complete_path = pair_dir / "complete_texture.png"
    partial_path = pair_dir / "partial_texture.png"
    mask_path = pair_dir / "inpainting_mask.png"

    Image.fromarray(complete_texture).save(complete_path)
    Image.fromarray(partial_texture).save(partial_path)
    Image.fromarray(inpainting_mask, mode="L").save(mask_path)

    # 5. Compute statistics
    complete_fill = float(np.count_nonzero(complete_coverage)) / complete_coverage.size
    partial_fill = float(np.count_nonzero(partial_coverage)) / partial_coverage.size
    mask_fill = float(np.count_nonzero(inpainting_mask)) / inpainting_mask.size

    metadata = {
        "character_id": character_id,
        "pose_id": pose_id,
        "tex_resolution": tex_resolution,
        "dense_angles": dense_angles,
        "partial_angles": partial_angles,
        "complete_coverage_pct": round(complete_fill * 100, 2),
        "partial_coverage_pct": round(partial_fill * 100, 2),
        "inpainting_pct": round(mask_fill * 100, 2),
        "complete_texture": str(complete_path),
        "partial_texture": str(partial_path),
        "inpainting_mask": str(mask_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    metadata_path = pair_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    logger.info(
        "Texture pair saved: complete=%.1f%% partial=%.1f%% inpaint=%.1f%%",
        complete_fill * 100,
        partial_fill * 100,
        mask_fill * 100,
    )

    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for texture projection training data generation.

    Designed to be invoked via::

        blender --background --python mesh/scripts/texture_projection_trainer.py -- \\
          --input path/to/character.fbx \\
          --output_dir ./output/texture_pairs/ \\
          --tex_resolution 1024
    """
    # Strip Blender args before '--'
    if argv is None:
        argv = sys.argv
        if "--" in argv:
            argv = argv[argv.index("--") + 1 :]
        else:
            argv = []

    parser = argparse.ArgumentParser(
        description="Generate partial/complete UV texture training pairs",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to character FBX file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/texture_pairs"),
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--tex_resolution",
        type=int,
        default=TEXTURE_RESOLUTION,
        help="UV texture resolution (default: %(default)s)",
    )
    parser.add_argument(
        "--character_id",
        type=str,
        default=None,
        help="Character identifier (default: derived from filename)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.input.is_file():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    character_id = args.character_id or args.input.stem

    # Clear scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # Import FBX
    bpy.ops.import_scene.fbx(filepath=str(args.input))

    # Collect mesh objects
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        logger.error("No mesh objects found in %s", args.input)
        sys.exit(1)

    logger.info("Loaded %d meshes from %s", len(meshes), args.input)

    # Generate training pair for default pose (pose 0)
    result = generate_texture_pairs(
        scene,
        meshes,
        args.output_dir,
        character_id,
        pose_id=0,
        tex_resolution=args.tex_resolution,
    )

    print(f"Complete coverage: {result['complete_coverage_pct']}%")
    print(f"Partial coverage: {result['partial_coverage_pct']}%")
    print(f"Inpainting needed: {result['inpainting_pct']}%")
    print(f"Output: {result['complete_texture']}")
    print("Done.")


if __name__ == "__main__":
    main()
