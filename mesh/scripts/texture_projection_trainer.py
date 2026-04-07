"""Generate (partial_texture, complete_texture) training pairs for neural inpainting.

For each character × pose, renders from dense camera angles (every 15°, 24 views)
and projects the rendered colors onto UV texture maps.  Produces:

- ``complete_texture.png`` — UV map composited from all 24 views (full coverage)
- ``partial_texture.png`` — UV map from 3 views only (front, three-quarter, back)
- ``inpainting_mask.png`` — Binary mask: 255 where texture is missing in partial
- ``position_map.png`` — Per-texel 3D world position (RGB = XYZ, normalized [0,255])
- ``normal_map.png`` — Per-texel surface normal (RGB = XYZ, remapped [-1,1]→[0,255])

Training pair: (partial_texture, inpainting_mask, position_map, normal_map) → complete_texture

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


def generate_uv_geometry_maps(
    meshes: list[bpy.types.Object],
    tex_resolution: int = TEXTURE_RESOLUTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate UV-space position and normal maps from mesh geometry.

    For each texel in UV space, computes the corresponding 3D world position
    and surface normal by rasterizing mesh triangles into UV space.

    Args:
        meshes: Character mesh objects (must have UV maps).
        tex_resolution: Output map resolution (square).

    Returns:
        Tuple of (position_map, normal_map):
        - position_map: (H, W, 3) uint8 — world XYZ normalized to [0, 255]
        - normal_map: (H, W, 3) uint8 — surface normal remapped from [-1,1] to [0, 255]
    """
    tex_h = tex_w = tex_resolution
    # Float buffers for accumulation
    position = np.zeros((tex_h, tex_w, 3), dtype=np.float32)
    normal = np.zeros((tex_h, tex_w, 3), dtype=np.float32)
    filled = np.zeros((tex_h, tex_w), dtype=bool)

    depsgraph = bpy.context.evaluated_depsgraph_get()

    # First pass: collect all world positions to compute bounding box
    all_positions: list[Vector] = []
    for mesh_obj in meshes:
        eval_obj = mesh_obj.evaluated_get(depsgraph)
        mesh_data = eval_obj.data
        for v in mesh_data.vertices:
            all_positions.append(eval_obj.matrix_world @ v.co)

    if not all_positions:
        logger.warning("No vertices found for geometry maps")
        return (
            np.zeros((tex_h, tex_w, 3), dtype=np.uint8),
            np.full((tex_h, tex_w, 3), 128, dtype=np.uint8),
        )

    xs = [v.x for v in all_positions]
    ys = [v.y for v in all_positions]
    zs = [v.z for v in all_positions]
    bbox_min = np.array([min(xs), min(ys), min(zs)], dtype=np.float32)
    bbox_max = np.array([max(xs), max(ys), max(zs)], dtype=np.float32)
    bbox_range = bbox_max - bbox_min
    bbox_range[bbox_range < 1e-6] = 1.0  # avoid division by zero

    # Second pass: rasterize each triangle in UV space (vectorized)
    for mesh_obj in meshes:
        eval_obj = mesh_obj.evaluated_get(depsgraph)
        mesh_data = eval_obj.data

        if not mesh_data.uv_layers:
            continue
        uv_layer = mesh_data.uv_layers.active.data
        world_mat = eval_obj.matrix_world
        normal_mat = world_mat.to_3x3()

        for poly in mesh_data.polygons:
            verts = list(poly.vertices)
            loop_indices = list(poly.loop_indices)
            if len(verts) < 3:
                continue

            world_pos = [np.array(world_mat @ mesh_data.vertices[vi].co) for vi in verts]
            face_normal = np.array(normal_mat @ Vector(poly.normal))
            norm_len = np.linalg.norm(face_normal)
            if norm_len > 1e-8:
                face_normal /= norm_len

            uv_coords = [np.array([uv_layer[li].uv[0], uv_layer[li].uv[1]]) for li in loop_indices]

            for tri_idx in range(1, len(verts) - 1):
                uv_a, uv_b, uv_c = uv_coords[0], uv_coords[tri_idx], uv_coords[tri_idx + 1]
                wp_a, wp_b, wp_c = world_pos[0], world_pos[tri_idx], world_pos[tri_idx + 1]

                uv_pixels = np.array([uv_a, uv_b, uv_c]) * np.array([tex_w, tex_h])
                min_u = max(0, int(np.floor(uv_pixels[:, 0].min())))
                max_u = min(tex_w - 1, int(np.ceil(uv_pixels[:, 0].max())))
                min_v = max(0, int(np.floor(uv_pixels[:, 1].min())))
                max_v = min(tex_h - 1, int(np.ceil(uv_pixels[:, 1].max())))

                n_u = max_u - min_u + 1
                n_v = max_v - min_v + 1
                if n_u <= 0 or n_v <= 0:
                    continue

                # Vectorized: build grid of texel centers
                tx_range = (np.arange(min_u, max_u + 1) + 0.5) / tex_w
                ty_range = (np.arange(min_v, max_v + 1) + 0.5) / tex_h
                grid_u, grid_v = np.meshgrid(tx_range, ty_range)
                points = np.stack([grid_u, grid_v], axis=-1)  # (n_v, n_u, 2)

                # Vectorized barycentric coords
                v0 = uv_c - uv_a
                v1 = uv_b - uv_a
                v2 = points - uv_a

                dot00 = np.dot(v0, v0)
                dot01 = np.dot(v0, v1)
                dot11 = np.dot(v1, v1)
                dot02 = v2 @ v0
                dot12 = v2 @ v1

                denom = dot00 * dot11 - dot01 * dot01
                if abs(denom) < 1e-12:
                    continue

                inv_denom = 1.0 / denom
                u = (dot11 * dot02 - dot01 * dot12) * inv_denom
                v = (dot00 * dot12 - dot01 * dot02) * inv_denom
                w0 = 1.0 - u - v

                inside = (w0 >= -1e-4) & (u >= -1e-4) & (v >= -1e-4)
                vy, vx = np.where(inside)

                tex_ys = tex_h - 1 - (vy + min_v)
                tex_xs = vx + min_u
                valid = (tex_ys >= 0) & (tex_ys < tex_h) & (tex_xs >= 0) & (tex_xs < tex_w)
                tex_ys, tex_xs = tex_ys[valid], tex_xs[valid]
                w0_v = w0[vy[valid], vx[valid]]
                u_v = u[vy[valid], vx[valid]]
                v_v = v[vy[valid], vx[valid]]

                # Interpolate world position per texel
                world_p = (w0_v[:, None] * wp_a + v_v[:, None] * wp_b + u_v[:, None] * wp_c)

                position[tex_ys, tex_xs] = world_p
                normal[tex_ys, tex_xs] = face_normal
                filled[tex_ys, tex_xs] = True

    # Normalize position to [0, 1] using bounding box
    position_norm = np.zeros_like(position)
    position_norm[filled] = (position[filled] - bbox_min) / bbox_range

    # Remap normal from [-1, 1] to [0, 1]
    normal_norm = np.zeros_like(normal)
    normal_norm[filled] = normal[filled] * 0.5 + 0.5

    # Convert to uint8
    position_map = (np.clip(position_norm, 0, 1) * 255).astype(np.uint8)
    normal_map = (np.clip(normal_norm, 0, 1) * 255).astype(np.uint8)

    logger.info(
        "Geometry maps: %d/%d texels filled (%.1f%%)",
        int(filled.sum()),
        tex_h * tex_w,
        filled.sum() / (tex_h * tex_w) * 100,
    )

    return position_map, normal_map


def project_view_to_uv(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    meshes: list[bpy.types.Object],
    rendered_image: np.ndarray,
    visibility: dict[tuple[int, int], bool],
    texture: np.ndarray,
    coverage: np.ndarray,
    incidence_weight: np.ndarray | None = None,
) -> None:
    """Project a rendered camera view onto the UV texture map in-place.

    For each visible polygon, projects its vertices to screen space, samples
    the rendered image, and writes colors to the UV texture at the polygon's
    UV coordinates.

    When *incidence_weight* is provided, texels are only overwritten if the
    new view has a higher incidence weight (face more directly facing the
    camera).  This prevents grazing-angle projections from overwriting
    better front-facing data.

    Args:
        scene: Blender scene with the camera set.
        camera: The camera used to render ``rendered_image``.
        meshes: Character mesh objects.
        rendered_image: RGBA array (H, W, 4) uint8 from the camera view.
        visibility: Per-face visibility from ``build_visibility_map``.
        texture: UV texture to fill, (tex_h, tex_w, 4) uint8. Modified in-place.
        coverage: Binary coverage array (tex_h, tex_w) uint8 — set to 255
            for texels that have been written. Modified in-place.
        incidence_weight: Optional float32 array (tex_h, tex_w) tracking the
            best incidence angle per texel.  Higher = more front-facing.
            Modified in-place when provided.
    """
    render_h, render_w = rendered_image.shape[:2]
    tex_h, tex_w = texture.shape[:2]
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cam_pos = Vector(camera.location)

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

            # Compute incidence angle for this face
            face_center = eval_obj.matrix_world @ Vector(poly.center)
            face_normal = (
                eval_obj.matrix_world.to_3x3() @ Vector(poly.normal)
            ).normalized()
            view_dir = (cam_pos - face_center).normalized()
            face_incidence = max(0.0, face_normal.dot(view_dir))

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
                    face_incidence=face_incidence,
                    incidence_weight=incidence_weight,
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
    face_incidence: float = 1.0,
    incidence_weight: np.ndarray | None = None,
) -> None:
    """Rasterize a single triangle in UV space, sampling from the rendered image.

    For each texel inside the UV triangle, computes barycentric coordinates,
    maps to screen space via the same barycentrics, samples the rendered image,
    and writes to the texture.

    When *incidence_weight* is provided, a texel is only written if *face_incidence*
    exceeds the stored weight for that texel.  This ensures front-facing projections
    take priority over grazing-angle ones.
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

                    # Incidence weighting: only overwrite if this view is
                    # more front-facing than what was previously written.
                    if incidence_weight is not None:
                        if face_incidence <= incidence_weight[tex_y, tx]:
                            continue
                        incidence_weight[tex_y, tx] = face_incidence

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

    Uses incidence-angle weighting so that front-facing projections take
    priority over grazing-angle ones, producing sharper textures.

    For each angle:
    1. Set up camera at the given azimuth
    2. Render the color image
    3. Compute face visibility
    4. Project rendered pixels onto UV space (weighted by incidence angle)

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
    incidence_weight = np.zeros((tex_resolution, tex_resolution), dtype=np.float32)

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

        # Project onto UV with incidence-angle weighting
        project_view_to_uv(
            scene,
            camera,
            meshes,
            rendered,
            visibility,
            texture,
            coverage,
            incidence_weight=incidence_weight,
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

    # 4. Generate UV geometry maps (position + normal)
    logger.info("Generating UV geometry maps for %s pose %d", character_id, pose_id)
    position_map, normal_map = generate_uv_geometry_maps(
        meshes, tex_resolution=tex_resolution
    )

    # 5. Save outputs
    complete_path = pair_dir / "complete_texture.png"
    partial_path = pair_dir / "partial_texture.png"
    mask_path = pair_dir / "inpainting_mask.png"
    position_path = pair_dir / "position_map.png"
    normal_path = pair_dir / "normal_map.png"

    Image.fromarray(complete_texture).save(complete_path)
    Image.fromarray(partial_texture).save(partial_path)
    Image.fromarray(inpainting_mask, mode="L").save(mask_path)
    Image.fromarray(position_map).save(position_path)
    Image.fromarray(normal_map).save(normal_path)

    # 6. Compute statistics
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
        "position_map": str(position_path),
        "normal_map": str(normal_path),
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


def _rasterize_triangles_to_mask(
    triangles_uv: np.ndarray,
    tex_resolution: int,
    mask: np.ndarray,
) -> None:
    """Vectorized UV triangle rasterization — marks visible texels in mask.

    Args:
        triangles_uv: (N, 3, 2) float array of UV triangle vertices.
        tex_resolution: Texture resolution (square).
        mask: (tex_h, tex_w) bool array, modified in-place.
    """
    if len(triangles_uv) == 0:
        return

    tex_h = tex_w = tex_resolution

    # Convert UV [0,1] to pixel coords
    tri_px = triangles_uv * np.array([tex_w, tex_h])  # (N, 3, 2)

    for i in range(len(tri_px)):
        a, b, c = tri_px[i]  # each (2,)
        uv_a, uv_b, uv_c = triangles_uv[i]

        min_u = max(0, int(np.floor(min(a[0], b[0], c[0]))))
        max_u = min(tex_w - 1, int(np.ceil(max(a[0], b[0], c[0]))))
        min_v = max(0, int(np.floor(min(a[1], b[1], c[1]))))
        max_v = min(tex_h - 1, int(np.ceil(max(a[1], b[1], c[1]))))

        n_u = max_u - min_u + 1
        n_v = max_v - min_v + 1
        if n_u <= 0 or n_v <= 0:
            continue

        # Build grid of texel centers in UV space
        tx_range = (np.arange(min_u, max_u + 1) + 0.5) / tex_w
        ty_range = (np.arange(min_v, max_v + 1) + 0.5) / tex_h
        grid_u, grid_v = np.meshgrid(tx_range, ty_range)  # (n_v, n_u)
        points = np.stack([grid_u, grid_v], axis=-1)  # (n_v, n_u, 2)

        # Vectorized barycentric coords for all points at once
        v0 = uv_c - uv_a
        v1 = uv_b - uv_a
        v2 = points - uv_a  # (n_v, n_u, 2)

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot11 = np.dot(v1, v1)
        dot02 = v2 @ v0  # (n_v, n_u)
        dot12 = v2 @ v1  # (n_v, n_u)

        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) < 1e-12:
            continue

        inv_denom = 1.0 / denom
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        w0 = 1.0 - u - v

        # Inside triangle test
        inside = (w0 >= -1e-4) & (u >= -1e-4) & (v >= -1e-4)

        # Map to image coords (UV origin bottom-left, array origin top-left)
        vy, vx = np.where(inside)
        tex_ys = tex_h - 1 - (vy + min_v)
        tex_xs = vx + min_u

        valid = (tex_ys >= 0) & (tex_ys < tex_h) & (tex_xs >= 0) & (tex_xs < tex_w)
        mask[tex_ys[valid], tex_xs[valid]] = True


def compute_uv_visibility_mask(
    scene: bpy.types.Scene,
    meshes: list[bpy.types.Object],
    angles: list[int],
    tex_resolution: int = TEXTURE_RESOLUTION,
    fast: bool = True,
) -> np.ndarray:
    """Compute which UV texels are visible from given camera angles.

    For each angle, determines face visibility and rasterizes visible faces
    into UV space using vectorized numpy operations. No rendering is performed.

    Args:
        scene: Blender scene with character loaded.
        meshes: Character mesh objects.
        angles: List of azimuth angles in degrees.
        tex_resolution: UV map resolution.
        fast: If True, use backface-only test (no ray casting). Much faster
              for dense meshes. Slightly overestimates visibility (includes
              occluded but front-facing faces), which is acceptable for
              training data.

    Returns:
        Binary visibility mask (tex_resolution, tex_resolution) uint8 — 255 where
        at least one view angle can see that UV texel, 0 elsewhere.
    """
    tex_h = tex_w = tex_resolution
    visible = np.zeros((tex_h, tex_w), dtype=bool)

    depsgraph = bpy.context.evaluated_depsgraph_get()

    for mesh_obj in meshes:
        ensure_uv_map(mesh_obj)

    for angle in angles:
        az_rad = radians(float(angle))
        # Camera looks from +Y toward origin, rotated by azimuth
        cam_dir = Vector((-sin(az_rad), cos(az_rad), 0)).normalized()

        if not fast:
            camera = setup_projection_camera(scene, meshes, float(angle))
            visibility = build_visibility_map(scene, camera, meshes)

        # Collect all visible triangles' UV coords
        all_triangles: list[np.ndarray] = []

        for mesh_idx, mesh_obj in enumerate(meshes):
            eval_obj = mesh_obj.evaluated_get(depsgraph)
            mesh_data = eval_obj.data

            if not mesh_data.uv_layers:
                continue
            uv_layer = mesh_data.uv_layers.active.data
            normal_mat = eval_obj.matrix_world.to_3x3()

            for poly_idx, poly in enumerate(mesh_data.polygons):
                if fast:
                    # Backface test only: face normal must oppose camera direction
                    face_normal_world = (normal_mat @ Vector(poly.normal)).normalized()
                    if face_normal_world.dot(cam_dir) >= 0:
                        continue  # back-facing
                else:
                    if not visibility.get((mesh_idx, poly_idx), False):
                        continue

                loop_indices = list(poly.loop_indices)
                if len(loop_indices) < 3:
                    continue

                uv_coords = [
                    np.array([uv_layer[li].uv[0], uv_layer[li].uv[1]])
                    for li in loop_indices
                ]

                for tri_idx in range(1, len(loop_indices) - 1):
                    all_triangles.append(np.array([
                        uv_coords[0],
                        uv_coords[tri_idx],
                        uv_coords[tri_idx + 1],
                    ]))

        if all_triangles:
            triangles_uv = np.array(all_triangles)  # (N, 3, 2)
            _rasterize_triangles_to_mask(triangles_uv, tex_resolution, visible)

        if not fast:
            cam_obj = bpy.data.objects.get("strata_proj_camera")
            if cam_obj is not None:
                bpy.data.objects.remove(cam_obj, do_unlink=True)

    return (visible * 255).astype(np.uint8)


def generate_pairs_from_existing_texture(
    scene: bpy.types.Scene,
    meshes: list[bpy.types.Object],
    texture_path: Path,
    output_dir: Path,
    character_id: str,
    *,
    partial_angles: list[int] | None = None,
    tex_resolution: int = TEXTURE_RESOLUTION,
    skip_geometry: bool = False,
) -> dict[str, Any]:
    """Generate training pairs using an existing texture file.

    Instead of rendering the character from multiple views, loads the original
    texture as the complete ground truth and computes a UV visibility mask from
    the given partial view angles to create the partial texture and inpainting mask.

    Args:
        scene: Blender scene with character loaded.
        meshes: Character mesh objects (must share the UV layout of texture_path).
        texture_path: Path to the existing texture PNG.
        output_dir: Root output directory.
        character_id: Identifier for the character.
        partial_angles: Azimuth angles for partial view (default: [0] = front only).
        tex_resolution: Output UV texture resolution.

    Returns:
        Metadata dict with paths and statistics.
    """
    if partial_angles is None:
        partial_angles = [0]

    pair_dir = output_dir / f"{character_id}_pose_00"
    pair_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load existing texture as complete ground truth
    logger.info("Loading existing texture: %s", texture_path)
    complete_img = Image.open(texture_path).convert("RGBA")
    if complete_img.size != (tex_resolution, tex_resolution):
        complete_img = complete_img.resize((tex_resolution, tex_resolution), Image.BILINEAR)
    complete_texture = np.array(complete_img)
    complete_coverage = (complete_texture[:, :, 3] > 10).astype(np.uint8)

    # 2. Compute UV visibility from partial view angles (no rendering)
    logger.info(
        "Computing UV visibility for %s from angles %s",
        character_id,
        partial_angles,
    )
    partial_vis = compute_uv_visibility_mask(
        scene, meshes, partial_angles, tex_resolution
    )

    # 3. Create partial texture: keep only visible texels
    partial_texture = complete_texture.copy()
    not_visible = partial_vis == 0
    partial_texture[not_visible] = 0
    partial_coverage = (partial_vis > 0) & (complete_coverage > 0)

    # 4. Inpainting mask: texels that have data in complete but not visible from partial
    inpainting_mask = (complete_coverage > 0) & not_visible
    inpainting_mask_uint8 = (inpainting_mask * 255).astype(np.uint8)

    # 5. Generate UV geometry maps (skip for speed — dataset falls back to zeros)
    if skip_geometry:
        logger.info("Skipping geometry maps for %s (fast mode)", character_id)
        position_map = np.zeros((tex_resolution, tex_resolution, 3), dtype=np.uint8)
        normal_map = np.full((tex_resolution, tex_resolution, 3), 128, dtype=np.uint8)
    else:
        logger.info("Generating UV geometry maps for %s", character_id)
        position_map, normal_map = generate_uv_geometry_maps(
            meshes, tex_resolution=tex_resolution
        )

    # 6. Save outputs
    complete_path = pair_dir / "complete_texture.png"
    partial_path = pair_dir / "partial_texture.png"
    mask_path = pair_dir / "inpainting_mask.png"
    position_path = pair_dir / "position_map.png"
    normal_path = pair_dir / "normal_map.png"

    Image.fromarray(complete_texture).save(complete_path)
    Image.fromarray(partial_texture).save(partial_path)
    Image.fromarray(inpainting_mask_uint8, mode="L").save(mask_path)
    Image.fromarray(position_map).save(position_path)
    Image.fromarray(normal_map).save(normal_path)

    # 7. Statistics
    total_texels = tex_resolution * tex_resolution
    complete_pct = float(complete_coverage.sum()) / total_texels * 100
    partial_pct = float(partial_coverage.sum()) / total_texels * 100
    inpaint_pct = float(inpainting_mask.sum()) / total_texels * 100

    metadata = {
        "character_id": character_id,
        "source_texture": str(texture_path),
        "tex_resolution": tex_resolution,
        "partial_angles": partial_angles,
        "complete_coverage_pct": round(complete_pct, 2),
        "partial_coverage_pct": round(partial_pct, 2),
        "inpainting_pct": round(inpaint_pct, 2),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    metadata_path = pair_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    logger.info(
        "Pair saved: complete=%.1f%% partial=%.1f%% inpaint=%.1f%%",
        complete_pct,
        partial_pct,
        inpaint_pct,
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
