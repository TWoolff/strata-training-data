"""Camera setup, render passes, and segmentation material assignment.

Provides:
- Orthographic camera with auto-framing
- Per-region Emission materials for segmentation
- Color render pass (flat-shaded with minimal lighting)
- Segmentation render pass (RGB → 8-bit grayscale region ID mask)
"""

from __future__ import annotations

import logging
from collections import Counter
from math import radians
from pathlib import Path

import bpy  # type: ignore[import-untyped]
import numpy as np
from mathutils import Vector  # type: ignore[import-untyped]
from PIL import Image

from .config import (
    AMBIENT_COLOR,
    CAMERA_CLIP_END,
    CAMERA_CLIP_START,
    CAMERA_DISTANCE,
    CAMERA_PADDING,
    CAMERA_TYPE,
    NUM_REGIONS,
    REGION_COLORS,
    RENDER_RESOLUTION,
    SUN_ENERGY,
    SUN_POSITION,
    RegionId,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Material name convention
# ---------------------------------------------------------------------------

MATERIAL_PREFIX = "strata_region_"


# ---------------------------------------------------------------------------
# Material creation
# ---------------------------------------------------------------------------


def create_region_materials() -> list[bpy.types.Material]:
    """Create one Emission-only material per Strata region.

    Each material uses a single Emission shader node wired directly to the
    Material Output.  Region 0 (background) is fully transparent so that
    the render's alpha channel can distinguish character pixels from empty
    space.

    Returns:
        List of 20 materials indexed by region ID.
    """
    materials: list[bpy.types.Material] = []

    for region_id in range(NUM_REGIONS):
        name = f"{MATERIAL_PREFIX}{region_id}"

        # Re-use if already created (e.g. multi-mesh character)
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name=name)

        mat.use_nodes = True
        mat.use_backface_culling = False

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        output = nodes.new(type="ShaderNodeOutputMaterial")
        output.location = (300, 0)

        if region_id == 0:
            # Background region — fully transparent so the render alpha
            # channel distinguishes character pixels from empty space
            mat.blend_method = "CLIP"
            mat.shadow_method = "CLIP"
            mat.alpha_threshold = 0.5
            transparent = nodes.new(type="ShaderNodeBsdfTransparent")
            transparent.location = (0, 0)
            links.new(transparent.outputs["BSDF"], output.inputs["Surface"])
        else:
            # Body regions — flat Emission at the region's color
            emission = nodes.new(type="ShaderNodeEmission")
            emission.location = (0, 0)
            r, g, b = REGION_COLORS[region_id]
            emission.inputs["Color"].default_value = (r / 255, g / 255, b / 255, 1.0)
            emission.inputs["Strength"].default_value = 1.0
            links.new(emission.outputs["Emission"], output.inputs["Surface"])

        materials.append(mat)

    logger.info("Created %d segmentation materials", len(materials))
    return materials


# ---------------------------------------------------------------------------
# Orthographic camera setup
# ---------------------------------------------------------------------------


def _combined_bounding_box(
    meshes: list[bpy.types.Object],
) -> tuple[Vector, Vector]:
    """Compute the world-space axis-aligned bounding box of multiple meshes.

    Args:
        meshes: List of mesh objects (must have at least one).

    Returns:
        (bbox_min, bbox_max) as mathutils.Vector.
    """
    all_corners: list[Vector] = []
    for mesh_obj in meshes:
        all_corners.extend(
            mesh_obj.matrix_world @ Vector(corner)
            for corner in mesh_obj.bound_box
        )

    xs = [v.x for v in all_corners]
    ys = [v.y for v in all_corners]
    zs = [v.z for v in all_corners]

    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


def setup_camera(
    scene: bpy.types.Scene,
    meshes: list[bpy.types.Object],
) -> bpy.types.Object:
    """Create an orthographic camera that auto-frames the character.

    The camera faces front-on (looking along +Y), centered on the character's
    bounding box with padding so the character fills ~80% of the frame.

    Args:
        scene: The Blender scene to add the camera to.
        meshes: Character mesh objects to frame.

    Returns:
        The camera object (already set as the scene's active camera).
    """
    bbox_min, bbox_max = _combined_bounding_box(meshes)
    bbox_center = (bbox_min + bbox_max) / 2

    width = bbox_max.x - bbox_min.x
    height = bbox_max.z - bbox_min.z

    # Orthographic scale: fit the larger dimension with padding
    ortho_scale = max(width, height) * (1.0 + 2.0 * CAMERA_PADDING)

    # Create camera data block
    cam_data = bpy.data.cameras.new(name="strata_camera")
    cam_data.type = CAMERA_TYPE
    cam_data.ortho_scale = ortho_scale
    cam_data.clip_start = CAMERA_CLIP_START
    cam_data.clip_end = CAMERA_CLIP_END

    # Create camera object and link to scene
    cam_obj = bpy.data.objects.new(name="strata_camera", object_data=cam_data)
    scene.collection.objects.link(cam_obj)

    # Position: centered on character, offset along -Y
    cam_obj.location = (bbox_center.x, -CAMERA_DISTANCE, bbox_center.z)

    # Rotation: face along +Y (camera looks down its local -Z axis,
    # so we rotate 90° around X to point it forward)
    cam_obj.rotation_euler = (radians(90), 0, 0)

    # Set as active camera
    scene.camera = cam_obj

    # Render resolution
    scene.render.resolution_x = RENDER_RESOLUTION
    scene.render.resolution_y = RENDER_RESOLUTION
    scene.render.resolution_percentage = 100

    # Transparent background
    scene.render.film_transparent = True

    logger.info(
        "Camera setup: ortho_scale=%.3f, center=(%.2f, %.2f), resolution=%dx%d",
        ortho_scale,
        bbox_center.x,
        bbox_center.z,
        RENDER_RESOLUTION,
        RENDER_RESOLUTION,
    )

    return cam_obj


# ---------------------------------------------------------------------------
# Face → region assignment (majority vote)
# ---------------------------------------------------------------------------


def _face_region_majority_vote(
    polygon: bpy.types.MeshPolygon,
    mesh_data: bpy.types.Mesh,
    mesh_index: int,
    vertex_to_region: dict[int, RegionId],
) -> RegionId:
    """Determine the region for a single face via vertex majority vote.

    Args:
        polygon: The mesh polygon (face) to classify.
        mesh_data: The mesh data block (for vertex positions).
        mesh_index: Index of this mesh in the character's mesh list,
            used to reconstruct composite vertex IDs from bone_mapper.
        vertex_to_region: Composite vertex ID → region ID mapping.

    Returns:
        The region ID for this face.
    """
    base_id = mesh_index * 10_000_000
    vertex_indices = list(polygon.vertices)

    # Collect region votes
    regions = [
        vertex_to_region.get(base_id + vi, 0) for vi in vertex_indices
    ]

    if not regions:
        return 0

    counts = Counter(regions)
    max_count = counts.most_common(1)[0][1]

    # If there's a clear winner, return it
    tied = [region for region, count in counts.items() if count == max_count]
    if len(tied) == 1:
        return tied[0]

    # Tie-break: pick the region of the vertex closest to face center
    face_center = polygon.center
    best_dist = float("inf")
    best_region: RegionId = tied[0]

    for vi, region in zip(vertex_indices, regions):
        if region not in tied:
            continue
        vert_pos = mesh_data.vertices[vi].co
        dist = (Vector(vert_pos) - Vector(face_center)).length_squared
        if dist < best_dist:
            best_dist = dist
            best_region = region

    return best_region


def assign_region_materials(
    mesh_obj: bpy.types.Object,
    mesh_index: int,
    vertex_to_region: dict[int, RegionId],
    materials: list[bpy.types.Material],
) -> None:
    """Assign per-region materials to every face of a mesh object.

    Args:
        mesh_obj: Blender mesh object to modify.
        mesh_index: Index of this mesh in the character's mesh list
            (for composite vertex ID lookup).
        vertex_to_region: Composite vertex ID → region ID mapping from
            bone_mapper.
        materials: List of 20 region materials from create_region_materials().
    """
    mesh_data = mesh_obj.data

    # Clear existing material slots
    mesh_obj.data.materials.clear()

    # Add all region materials as slots
    for mat in materials:
        mesh_obj.data.materials.append(mat)

    # Assign each face to its region's material slot
    region_counts: Counter[int] = Counter()

    for polygon in mesh_data.polygons:
        region = _face_region_majority_vote(
            polygon, mesh_data, mesh_index, vertex_to_region,
        )
        polygon.material_index = region
        region_counts[region] += 1

    logger.info(
        "Assigned %d faces across %d regions on mesh '%s'",
        len(mesh_data.polygons),
        len(region_counts),
        mesh_obj.name,
    )


# ---------------------------------------------------------------------------
# Render settings for segmentation pass
# ---------------------------------------------------------------------------


def setup_segmentation_render(scene: bpy.types.Scene) -> None:
    """Configure EEVEE render settings for a clean segmentation mask pass.

    Disables anti-aliasing, sets transparent background, configures
    resolution, and ensures Emission materials render as flat color
    unaffected by lighting.

    Args:
        scene: The Blender scene to configure.
    """
    # Engine
    scene.render.engine = "BLENDER_EEVEE_NEXT"

    # Resolution
    scene.render.resolution_x = RENDER_RESOLUTION
    scene.render.resolution_y = RENDER_RESOLUTION
    scene.render.resolution_percentage = 100

    # Transparent background (alpha channel)
    scene.render.film_transparent = True

    # Disable anti-aliasing for clean pixel boundaries
    scene.render.filter_size = 0.0

    # Color management: raw (no tone mapping, no gamma correction)
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"

    # Output format: PNG with RGBA
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 0

    # Disable all post-processing
    scene.render.use_compositing = False
    scene.render.use_sequencer = False

    # Disable ambient occlusion and bloom
    scene.eevee.use_gtao = False
    scene.eevee.use_bloom = False

    logger.info(
        "Segmentation render configured: %dx%d, EEVEE, no AA, transparent BG",
        RENDER_RESOLUTION,
        RENDER_RESOLUTION,
    )


# ---------------------------------------------------------------------------
# Render execution
# ---------------------------------------------------------------------------


def render_segmentation(
    scene: bpy.types.Scene,
    output_path: Path,
) -> Path:
    """Execute the segmentation render and save the result.

    Args:
        scene: The Blender scene (must already have segmentation materials
            assigned and render settings configured).
        output_path: File path for the output PNG (without extension — Blender
            appends it automatically, so pass the full path including .png).

    Returns:
        The path to the rendered image.
    """
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)

    logger.info("Segmentation mask rendered to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# RGB → grayscale mask conversion
# ---------------------------------------------------------------------------

def convert_rgb_to_grayscale_mask(rgb_path: Path, output_path: Path) -> Path:
    """Convert an RGB segmentation render to an 8-bit grayscale region ID mask.

    Each pixel's RGB value is mapped to its region ID via the REGION_COLORS
    reverse lookup. Transparent pixels (alpha = 0) are mapped to region 0
    (background). Pixels that don't exactly match any region color are mapped
    to the nearest region color by Euclidean distance.

    Args:
        rgb_path: Path to the RGBA PNG rendered by ``render_segmentation``.
        output_path: Destination path for the 8-bit grayscale PNG.

    Returns:
        The output path.
    """
    img = Image.open(rgb_path).convert("RGBA")
    pixels = np.array(img)  # (H, W, 4) uint8

    r, g, b, a = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2], pixels[:, :, 3]

    # Build region color arrays for vectorized lookup
    region_ids = sorted(REGION_COLORS.keys())
    color_array = np.array([REGION_COLORS[rid] for rid in region_ids], dtype=np.int32)
    id_array = np.array(region_ids, dtype=np.uint8)

    # Reshape pixel RGB for broadcasting: (H, W, 1, 3) vs (1, 1, N, 3)
    pixel_rgb = np.stack([r, g, b], axis=-1).astype(np.int32)
    diff = pixel_rgb[:, :, np.newaxis, :] - color_array[np.newaxis, np.newaxis, :, :]
    dist_sq = (diff ** 2).sum(axis=-1)  # (H, W, N)

    # Nearest region color for each pixel
    nearest_idx = dist_sq.argmin(axis=-1)  # (H, W)
    mask = id_array[nearest_idx]  # (H, W) uint8

    # Transparent pixels → region 0
    mask[a == 0] = 0

    out_img = Image.fromarray(mask, mode="L")
    out_img.save(output_path, format="PNG", compress_level=9)

    logger.info("Grayscale mask saved to %s (%dx%d)", output_path, *mask.shape)
    return output_path


# ---------------------------------------------------------------------------
# Color render setup
# ---------------------------------------------------------------------------


def setup_color_render(scene: bpy.types.Scene) -> None:
    """Configure EEVEE render settings and lighting for the color pass.

    Sets up flat shading with minimal lighting: a single Sun lamp and a
    World ambient background. Anti-aliasing is enabled (default filter size)
    and standard color management is used.

    Args:
        scene: The Blender scene to configure.
    """
    # Engine
    scene.render.engine = "BLENDER_EEVEE_NEXT"

    # Resolution
    scene.render.resolution_x = RENDER_RESOLUTION
    scene.render.resolution_y = RENDER_RESOLUTION
    scene.render.resolution_percentage = 100

    # Transparent background
    scene.render.film_transparent = True

    # Anti-aliasing: restore default filter size (segmentation pass sets 0.0)
    scene.render.filter_size = 1.5

    # Standard color management
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"

    # Output format: PNG with RGBA
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 15

    # Disable post-processing
    scene.render.use_compositing = False
    scene.render.use_sequencer = False

    # Disable ambient occlusion and bloom
    scene.eevee.use_gtao = False
    scene.eevee.use_bloom = False

    # --- Lighting ---

    # Sun lamp
    sun_data = bpy.data.lights.get("strata_sun")
    if sun_data is None:
        sun_data = bpy.data.lights.new(name="strata_sun", type="SUN")
    sun_data.energy = SUN_ENERGY

    sun_obj = bpy.data.objects.get("strata_sun")
    if sun_obj is None:
        sun_obj = bpy.data.objects.new(name="strata_sun", object_data=sun_data)
        scene.collection.objects.link(sun_obj)
    else:
        sun_obj.data = sun_data

    sun_obj.location = SUN_POSITION
    # Point roughly downward-forward (60° tilt from horizontal)
    sun_obj.rotation_euler = (radians(60), 0, 0)

    # World ambient background
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new(name="strata_world")
        scene.world = world

    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    bg_node = nodes.new(type="ShaderNodeBackground")
    bg_node.inputs["Color"].default_value = AMBIENT_COLOR
    bg_node.inputs["Strength"].default_value = 1.0
    bg_node.location = (0, 0)

    output_node = nodes.new(type="ShaderNodeOutputWorld")
    output_node.location = (300, 0)

    links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])

    logger.info(
        "Color render configured: %dx%d, EEVEE, sun energy=%.1f, ambient=(%.1f,%.1f,%.1f)",
        RENDER_RESOLUTION,
        RENDER_RESOLUTION,
        SUN_ENERGY,
        AMBIENT_COLOR[0],
        AMBIENT_COLOR[1],
        AMBIENT_COLOR[2],
    )


def render_color(scene: bpy.types.Scene, output_path: Path) -> Path:
    """Execute the color render pass and save the result.

    The scene must already have color render settings configured via
    ``setup_color_render`` and the character's original materials active
    (not segmentation materials).

    Args:
        scene: The Blender scene to render.
        output_path: File path for the output PNG.

    Returns:
        The path to the rendered image.
    """
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)

    logger.info("Color render saved to %s", output_path)
    return output_path
