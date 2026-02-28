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
from math import cos, radians, sin
from pathlib import Path

import bpy  # type: ignore[import-untyped]
import numpy as np
from mathutils import Vector  # type: ignore[import-untyped]
from PIL import Image

def _eevee_engine_name() -> str:
    """Return the correct EEVEE engine enum for this Blender version.

    Blender 4.2–4.4 used ``BLENDER_EEVEE_NEXT``; Blender 5.0+ renamed
    it back to ``BLENDER_EEVEE``.
    """
    try:
        bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
        return "BLENDER_EEVEE_NEXT"
    except TypeError:
        return "BLENDER_EEVEE"


_EEVEE_ENGINE = _eevee_engine_name()

from .config import (
    AMBIENT_COLOR,
    CAMERA_CLIP_END,
    CAMERA_CLIP_START,
    CAMERA_DISTANCE,
    CAMERA_PADDING,
    CAMERA_TYPE,
    CEL_OUTLINE_THICKNESS,
    CEL_RAMP_STOPS,
    DEFAULT_BASE_COLOR,
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
            # channel distinguishes character pixels from empty space.
            # Blender 4.2+ replaced blend_method/shadow_method with
            # surface_render_method; fall back for older versions.
            if hasattr(mat, "surface_render_method"):
                mat.surface_render_method = "DITHERED"
            elif hasattr(mat, "blend_method"):
                mat.blend_method = "CLIP"
            if hasattr(mat, "shadow_method"):
                mat.shadow_method = "CLIP"
            if hasattr(mat, "alpha_threshold"):
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
        all_corners.extend(mesh_obj.matrix_world @ Vector(corner) for corner in mesh_obj.bound_box)

    xs = [v.x for v in all_corners]
    ys = [v.y for v in all_corners]
    zs = [v.z for v in all_corners]

    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


def setup_camera(
    scene: bpy.types.Scene,
    meshes: list[bpy.types.Object],
    *,
    azimuth: float = 0.0,
) -> bpy.types.Object:
    """Create an orthographic camera that auto-frames the character.

    The camera orbits the character's bounding-box center at ``azimuth``
    degrees around the vertical (Z) axis, always looking toward the center.
    At azimuth 0 the camera faces front-on (looking along +Y).

    Args:
        scene: The Blender scene to add the camera to.
        meshes: Character mesh objects to frame.
        azimuth: Horizontal rotation in degrees (0 = front, 90 = side,
            180 = back).

    Returns:
        The camera object (already set as the scene's active camera).
    """
    bbox_min, bbox_max = _combined_bounding_box(meshes)
    bbox_center = (bbox_min + bbox_max) / 2

    width = bbox_max.x - bbox_min.x
    height = bbox_max.z - bbox_min.z
    depth = bbox_max.y - bbox_min.y

    # For non-frontal views the visible horizontal extent includes depth
    az_rad = radians(azimuth)
    apparent_width = abs(width * cos(az_rad)) + abs(depth * sin(az_rad))

    # Orthographic scale: fit the larger dimension with padding
    ortho_scale = max(apparent_width, height) * (1.0 + 2.0 * CAMERA_PADDING)

    # Create camera data block
    cam_data = bpy.data.cameras.new(name="strata_camera")
    cam_data.type = CAMERA_TYPE
    cam_data.ortho_scale = ortho_scale
    cam_data.clip_start = CAMERA_CLIP_START
    cam_data.clip_end = CAMERA_CLIP_END

    # Create camera object and link to scene
    cam_obj = bpy.data.objects.new(name="strata_camera", object_data=cam_data)
    scene.collection.objects.link(cam_obj)

    # Position: orbit around the character center at CAMERA_DISTANCE.
    # At azimuth=0 the camera is at (center.x, center.y - D, center.z)
    # looking along +Y.  Positive azimuth rotates clockwise when viewed
    # from above (standard screen-space convention).
    cam_x = bbox_center.x + CAMERA_DISTANCE * sin(az_rad)
    cam_y = bbox_center.y - CAMERA_DISTANCE * cos(az_rad)
    cam_z = bbox_center.z
    cam_obj.location = (cam_x, cam_y, cam_z)

    # Rotation: camera always faces the character center.
    # Base rotation is 90° around X (to look along +Y).
    # Azimuth adds a Z rotation.
    cam_obj.rotation_euler = (radians(90), 0, az_rad)

    # Set as active camera
    scene.camera = cam_obj

    # Render resolution
    scene.render.resolution_x = RENDER_RESOLUTION
    scene.render.resolution_y = RENDER_RESOLUTION
    scene.render.resolution_percentage = 100

    # Transparent background
    scene.render.film_transparent = True

    logger.info(
        "Camera setup: azimuth=%.0f°, ortho_scale=%.3f, center=(%.2f, %.2f), resolution=%dx%d",
        azimuth,
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
    regions = [vertex_to_region.get(base_id + vi, 0) for vi in vertex_indices]

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

    for vi, region in zip(vertex_indices, regions, strict=True):
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
            polygon,
            mesh_data,
            mesh_index,
            vertex_to_region,
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
    scene.render.engine = _EEVEE_ENGINE

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

    # Disable ambient occlusion and bloom (removed in Blender 5.0)
    if hasattr(scene.eevee, "use_gtao"):
        scene.eevee.use_gtao = False
    if hasattr(scene.eevee, "use_bloom"):
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
    dist_sq = (diff**2).sum(axis=-1)  # (H, W, N)

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
    scene.render.engine = _EEVEE_ENGINE

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

    # Disable ambient occlusion and bloom (removed in Blender 5.0)
    if hasattr(scene.eevee, "use_gtao"):
        scene.eevee.use_gtao = False
    if hasattr(scene.eevee, "use_bloom"):
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


# ---------------------------------------------------------------------------
# Render-time style helpers
# ---------------------------------------------------------------------------


def _extract_base_color(
    material: bpy.types.Material | None,
) -> tuple[float, float, float, float]:
    """Extract the base color from a material's Principled BSDF node.

    Falls back to DEFAULT_BASE_COLOR if no Principled BSDF is found or
    if the material is None.

    Args:
        material: A Blender material (may be None).

    Returns:
        RGBA tuple with float values in [0, 1].
    """
    if material is None or not material.use_nodes:
        return DEFAULT_BASE_COLOR

    for node in material.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            color_input = node.inputs.get("Base Color")
            if color_input is not None:
                val = color_input.default_value
                return (val[0], val[1], val[2], val[3])

    return DEFAULT_BASE_COLOR


def _get_image_texture_node(
    material: bpy.types.Material | None,
) -> bpy.types.ShaderNodeTexImage | None:
    """Find the first Image Texture node connected to a Principled BSDF Base Color.

    Args:
        material: A Blender material (may be None).

    Returns:
        The Image Texture node, or None if not found.
    """
    if material is None or not material.use_nodes:
        return None

    for node in material.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            color_input = node.inputs.get("Base Color")
            if color_input is not None and color_input.is_linked:
                linked_node = color_input.links[0].from_node
                if linked_node.type == "TEX_IMAGE" and linked_node.image:
                    return linked_node

    return None


# ---------------------------------------------------------------------------
# Render-time style application
# ---------------------------------------------------------------------------


def _wire_color_source(
    nodes: bpy.types.NodeTree,
    links: bpy.types.NodeLinks,
    target_input: bpy.types.NodeSocketColor,
    tex_node: bpy.types.ShaderNodeTexImage | None,
    base_color: tuple[float, float, float, float],
    tex_location: tuple[int, int] = (-300, 0),
) -> None:
    """Wire a texture or solid color into a shader input.

    If the original material had an image texture connected to its
    Principled BSDF, creates a new Image Texture node referencing the
    same image. Otherwise sets the input's default value to base_color.

    Args:
        nodes: The node tree's node collection.
        links: The node tree's link collection.
        target_input: The shader input socket to connect to.
        tex_node: Original Image Texture node (or None).
        base_color: Fallback RGBA color.
        tex_location: Node position for the texture node.
    """
    if tex_node is not None and tex_node.image:
        img_tex = nodes.new(type="ShaderNodeTexImage")
        img_tex.image = tex_node.image
        img_tex.location = tex_location
        links.new(img_tex.outputs["Color"], target_input)
    else:
        target_input.default_value = base_color


def apply_flat_style(meshes: list[bpy.types.Object]) -> None:
    """Apply flat shading style: Diffuse BSDF with no specular, flat face shading.

    Overrides each material slot with a Diffuse BSDF using the original
    material's base color. Sets mesh shading to flat (no smooth normals).

    Args:
        meshes: Character mesh objects whose materials to override.
    """
    for mesh_obj in meshes:
        mesh_data = mesh_obj.data

        # Set flat shading on all faces
        for polygon in mesh_data.polygons:
            polygon.use_smooth = False

        for slot in mesh_obj.material_slots:
            original_mat = slot.material
            base_color = _extract_base_color(original_mat)
            tex_node = _get_image_texture_node(original_mat)

            mat = bpy.data.materials.new(name=f"strata_flat_{slot.name}")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            nodes.clear()

            output = nodes.new(type="ShaderNodeOutputMaterial")
            output.location = (400, 0)

            diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
            diffuse.location = (0, 0)

            _wire_color_source(nodes, links, diffuse.inputs["Color"], tex_node, base_color)

            diffuse.inputs["Roughness"].default_value = 1.0
            links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])

            slot.material = mat

    logger.info("Applied flat style to %d meshes", len(meshes))


def apply_cel_style(
    scene: bpy.types.Scene,
    meshes: list[bpy.types.Object],
) -> None:
    """Apply cel/toon shading: quantized color ramp with Freestyle outlines.

    Builds a node tree per material: Diffuse BSDF -> Shader to RGB ->
    ColorRamp (constant interpolation, 3 stops) -> Material Output.
    Enables Freestyle line rendering for black outlines.

    Args:
        scene: The Blender scene (for Freestyle configuration).
        meshes: Character mesh objects whose materials to override.
    """
    # Enable Freestyle outlines
    scene.render.use_freestyle = True
    view_layer = scene.view_layers[0]
    view_layer.use_freestyle = True

    # Configure Freestyle line set
    if view_layer.freestyle_settings.linesets:
        lineset = view_layer.freestyle_settings.linesets[0]
    else:
        lineset = view_layer.freestyle_settings.linesets.new("outline")

    lineset.linestyle.thickness = CEL_OUTLINE_THICKNESS
    lineset.linestyle.color = (0.0, 0.0, 0.0)  # black outlines

    for mesh_obj in meshes:
        for slot in mesh_obj.material_slots:
            original_mat = slot.material
            base_color = _extract_base_color(original_mat)
            tex_node = _get_image_texture_node(original_mat)

            mat = bpy.data.materials.new(name=f"strata_cel_{slot.name}")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            nodes.clear()

            output = nodes.new(type="ShaderNodeOutputMaterial")
            output.location = (900, 0)

            # Diffuse BSDF captures lighting response
            diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
            diffuse.location = (0, 0)
            diffuse.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)

            # Shader to RGB (EEVEE only) — converts lighting to color data
            shader_to_rgb = nodes.new(type="ShaderNodeShaderToRGB")
            shader_to_rgb.location = (200, 0)
            links.new(diffuse.outputs["BSDF"], shader_to_rgb.inputs["Shader"])

            # ColorRamp quantizes the lighting into hard toon steps
            color_ramp = nodes.new(type="ShaderNodeValToRGB")
            color_ramp.location = (400, 0)
            color_ramp.color_ramp.interpolation = "CONSTANT"

            ramp = color_ramp.color_ramp
            while len(ramp.elements) > len(CEL_RAMP_STOPS):
                ramp.elements.remove(ramp.elements[-1])
            while len(ramp.elements) < len(CEL_RAMP_STOPS):
                ramp.elements.new(0.5)

            for i, (position, brightness) in enumerate(CEL_RAMP_STOPS):
                ramp.elements[i].position = position
                ramp.elements[i].color = (brightness, brightness, brightness, 1.0)

            links.new(shader_to_rgb.outputs["Color"], color_ramp.inputs["Fac"])

            # MixRGB (Multiply) — toon shading * base color
            mix_rgb = nodes.new(type="ShaderNodeMixRGB")
            mix_rgb.blend_type = "MULTIPLY"
            mix_rgb.location = (600, 0)
            mix_rgb.inputs["Fac"].default_value = 1.0

            _wire_color_source(
                nodes, links, mix_rgb.inputs["Color1"], tex_node, base_color, (-300, -200)
            )

            links.new(color_ramp.outputs["Color"], mix_rgb.inputs["Color2"])

            # Emission shader to output the result without additional lighting
            emission = nodes.new(type="ShaderNodeEmission")
            emission.location = (750, 0)
            emission.inputs["Strength"].default_value = 1.0
            links.new(mix_rgb.outputs["Color"], emission.inputs["Color"])
            links.new(emission.outputs["Emission"], output.inputs["Surface"])

            slot.material = mat

    logger.info(
        "Applied cel style to %d meshes (outline=%.1fpx)", len(meshes), CEL_OUTLINE_THICKNESS
    )


def apply_unlit_style(
    scene: bpy.types.Scene,
    meshes: list[bpy.types.Object],
) -> None:
    """Apply unlit/color-only style: Emission shader with no lighting.

    Replaces all materials with Emission BSDF using the character's
    base color. Disables the sun lamp to ensure zero lighting influence.

    Args:
        scene: The Blender scene (to disable lights).
        meshes: Character mesh objects whose materials to override.
    """
    # Disable all lights in the scene
    for obj in scene.objects:
        if obj.type == "LIGHT":
            obj.hide_render = True

    for mesh_obj in meshes:
        for slot in mesh_obj.material_slots:
            original_mat = slot.material
            base_color = _extract_base_color(original_mat)
            tex_node = _get_image_texture_node(original_mat)

            mat = bpy.data.materials.new(name=f"strata_unlit_{slot.name}")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            nodes.clear()

            output = nodes.new(type="ShaderNodeOutputMaterial")
            output.location = (400, 0)

            emission = nodes.new(type="ShaderNodeEmission")
            emission.location = (0, 0)
            emission.inputs["Strength"].default_value = 1.0

            _wire_color_source(nodes, links, emission.inputs["Color"], tex_node, base_color)

            links.new(emission.outputs["Emission"], output.inputs["Surface"])

            slot.material = mat

    logger.info("Applied unlit style to %d meshes", len(meshes))


def restore_style(
    scene: bpy.types.Scene,
    style: str,
) -> None:
    """Clean up scene-level state modified by a render-time style.

    Removes temporary style materials from bpy.data and resets any
    scene-level settings (e.g., Freestyle, light visibility).

    Note: Material slot restoration is handled by the caller via
    ``_restore_materials`` / ``_backup_materials`` in generate_dataset.py.

    Args:
        scene: The Blender scene.
        style: The style name that was applied ("flat", "cel", or "unlit").
    """
    # Clean up temporary materials
    prefix = f"strata_{style}_"
    to_remove = [m for m in bpy.data.materials if m.name.startswith(prefix)]
    for mat in to_remove:
        bpy.data.materials.remove(mat)

    if style == "cel":
        # Disable Freestyle
        scene.render.use_freestyle = False
        view_layer = scene.view_layers[0]
        view_layer.use_freestyle = False

    if style == "unlit":
        # Re-enable all lights
        for obj in scene.objects:
            if obj.type == "LIGHT":
                obj.hide_render = False

    logger.info("Restored scene after '%s' style", style)


def apply_style(
    scene: bpy.types.Scene,
    meshes: list[bpy.types.Object],
    style: str,
) -> None:
    """Apply a render-time art style to the character's materials.

    Dispatches to the appropriate style function. For styles that are
    not render-time (e.g., "pixel", "painterly", "sketch"), this is a no-op.

    Args:
        scene: The Blender scene.
        meshes: Character mesh objects.
        style: Style name ("flat", "cel", "unlit", or post-render styles).
    """
    if style == "flat":
        apply_flat_style(meshes)
    elif style == "cel":
        apply_cel_style(scene, meshes)
    elif style == "unlit":
        apply_unlit_style(scene, meshes)
    # Post-render styles (pixel, painterly, sketch) are no-ops here
