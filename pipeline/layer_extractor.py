"""Per-region RGBA layer extraction for layer decomposition training.

For each body region (1–19), renders the character with only that region
visible and all others transparent.  Produces RGBA layer images that,
when composited back-to-front by draw order, reproduce the original
composite image.

Depends on Blender (``bpy``); must run in ``blender --background`` mode.

Reference: "See Through" (arXiv 2602.03749) — semantic RGBA layer
decomposition for anime illustration editing.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import bpy  # type: ignore[import-untyped]
import numpy as np
from PIL import Image

from .config import NUM_REGIONS

logger = logging.getLogger(__name__)

LAYER_MATERIAL_PREFIX = "strata_layer_"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_active_regions(segmentation_mask: np.ndarray) -> list[int]:
    """Return region IDs (1–19) that have at least one pixel in the mask.

    Args:
        segmentation_mask: 2D uint8 array where pixel value = region ID.

    Returns:
        Sorted list of active region IDs.
    """
    unique_ids = set(int(v) for v in np.unique(segmentation_mask))
    unique_ids.discard(0)
    return sorted(rid for rid in unique_ids if 1 <= rid < NUM_REGIONS)


def _create_transparent_material() -> bpy.types.Material:
    """Create (or reuse) a fully transparent material for hiding non-target regions.

    Uses Principled BSDF with Alpha=0 and Blended render method.  Blended
    gives clean alpha transparency without the stipple/noise artifacts that
    DITHERED mode produces.  Sort-order issues (Blended's main drawback)
    don't matter here because only one region is visible at a time.
    """
    name = f"{LAYER_MATERIAL_PREFIX}transparent"
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)

    mat.use_nodes = True
    mat.use_backface_culling = False

    # Use Blended transparency (clean alpha, no dithering noise).
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "BLENDED"
    elif hasattr(mat, "blend_method"):
        mat.blend_method = "ALPHA_BLEND"
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = "NONE"

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (300, 0)

    # Principled BSDF with Alpha=0 → fully transparent.
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.location = (0, 0)
    principled.inputs["Alpha"].default_value = 0.0
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])

    return mat


def _backup_slot_materials(
    meshes: list[bpy.types.Object],
) -> list[list[bpy.types.Material | None]]:
    """Snapshot material slot contents for all meshes."""
    return [[slot.material for slot in mesh_obj.material_slots] for mesh_obj in meshes]


def _restore_slot_materials(
    meshes: list[bpy.types.Object],
    backup: list[list[bpy.types.Material | None]],
) -> None:
    """Restore material slot contents from a backup."""
    for mesh_obj, saved in zip(meshes, backup, strict=True):
        for i, mat in enumerate(saved):
            if i < len(mesh_obj.material_slots):
                mesh_obj.material_slots[i].material = mat


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------


def extract_layers(
    scene: bpy.types.Scene,
    meshes: list[bpy.types.Object],
    original_materials: list[list[bpy.types.Material | None]],
    segmentation_mask: np.ndarray,
    output_dir: Path,
    file_prefix: str,
) -> dict[int, Path]:
    """Render per-region RGBA layers for a single pose and angle.

    For each region with pixels in the segmentation mask, temporarily
    replaces all non-target material slots with a transparent material,
    renders the target region with an Emission shader using the character's
    texture, then restores the original material state.

    Uses Emission shaders (not Diffuse BSDF) so layers are unaffected by
    scene lighting — consistent flat color regardless of camera angle.
    Disables anti-aliasing during rendering for clean pixel boundaries
    (same as segmentation pass).

    The caller must have already configured the camera and called
    ``setup_color_render()``.  After this function returns, the meshes
    are restored to their pre-call material state and render settings
    are restored.

    Args:
        scene: Blender scene (camera + render settings configured).
        meshes: Character mesh objects with 20 material slots assigned
            from the segmentation pass (face ``material_index`` intact).
        original_materials: Backed-up original character materials
            (from before segmentation assignment).
        segmentation_mask: 2D uint8 array for active-region detection.
        output_dir: Root output directory (uses ``layers/`` subdirectory).
        file_prefix: Filename prefix, e.g. ``"char01_pose_00"``.

    Returns:
        Dict mapping region_id → saved file path for each rendered layer.
    """
    from .renderer import (
        _extract_base_color,
        _get_image_texture_node,
        _wire_color_source,
        render_color,
    )

    active_regions = get_active_regions(segmentation_mask)
    if not active_regions:
        logger.warning("No active regions for %s — skipping layers", file_prefix)
        return {}

    logger.info(
        "Extracting %d region layers for %s",
        len(active_regions),
        file_prefix,
    )

    layers_dir = output_dir / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)
    transparent_mat = _create_transparent_material()

    # Snapshot current slot materials (segmentation Emission materials).
    # After assign_region_materials(), meshes have 20 slots (one per region)
    # with face material_index values pointing to the correct region slot.
    seg_backup = _backup_slot_materials(meshes)

    # Temporarily restore original materials to extract texture/color info.
    # Original materials have the character's real textures (typically a shared
    # UV-mapped atlas) but only 3-5 slots, NOT the 20-slot region layout.
    _restore_slot_materials(meshes, original_materials)

    # Find the shared texture atlas from any original material slot.
    # Mixamo characters use one UV-mapped atlas across all body parts.
    shared_tex_node = None
    shared_base_color = (0.8, 0.8, 0.8, 1.0)
    for mesh_obj in meshes:
        for slot in mesh_obj.material_slots:
            tex = _get_image_texture_node(slot.material)
            if tex is not None:
                shared_tex_node = tex
                shared_base_color = _extract_base_color(slot.material)
                break
        if shared_tex_node is not None:
            break

    # Restore seg materials (back to 20 slots with correct face indices)
    _restore_slot_materials(meshes, seg_backup)

    # Build one shared Emission material using the character's texture.
    # Emission is lighting-independent — consistent flat color regardless of
    # camera angle, matching the "See Through" paper's expectation of unlit layers.
    # Blender mesh UVs are stored on the mesh data, so the same texture + UVs
    # produce the correct appearance regardless of which slot the face is in.
    flat_mat = bpy.data.materials.new(name=f"{LAYER_MATERIAL_PREFIX}flat")
    flat_mat.use_nodes = True
    flat_mat.use_backface_culling = False

    nodes = flat_mat.node_tree.nodes
    links = flat_mat.node_tree.links
    nodes.clear()

    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    output_node.location = (400, 0)
    emission = nodes.new(type="ShaderNodeEmission")
    emission.location = (0, 0)
    emission.inputs["Strength"].default_value = 1.0
    _wire_color_source(nodes, links, emission.inputs["Color"], shared_tex_node, shared_base_color)
    links.new(emission.outputs["Emission"], output_node.inputs["Surface"])

    # Disable anti-aliasing for clean pixel boundaries (like segmentation pass).
    # DITHERED transparency + AA creates stipple bleeding at edges.
    saved_filter_size = scene.render.filter_size
    saved_view_transform = scene.view_settings.view_transform
    scene.render.filter_size = 0.0
    scene.view_settings.view_transform = "Raw"

    # Render each active region in isolation
    saved_layers: dict[int, Path] = {}

    for region_id in active_regions:
        # Set all slots: target region gets flat material, others get transparent
        for mesh_obj in meshes:
            for slot_idx in range(len(mesh_obj.material_slots)):
                if slot_idx == region_id:
                    mesh_obj.material_slots[slot_idx].material = flat_mat
                else:
                    mesh_obj.material_slots[slot_idx].material = transparent_mat

        # Render to temp file then move (Blender render paths are finicky)
        layer_path = layers_dir / f"{file_prefix}_{region_id:02d}.png"
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        render_color(scene, tmp_path)

        # Copy to final destination (render_color saves to tmp_path)
        img = Image.open(tmp_path)
        img.save(layer_path, format="PNG", compress_level=9)
        tmp_path.unlink(missing_ok=True)

        saved_layers[region_id] = layer_path

    # Restore render settings (AA, color management) for subsequent passes.
    scene.render.filter_size = saved_filter_size
    scene.view_settings.view_transform = saved_view_transform

    logger.info("Saved %d region layers for %s", len(saved_layers), file_prefix)

    # Restore segmentation materials so the pipeline continues normally
    _restore_slot_materials(meshes, seg_backup)

    # Clean up layer materials (collect first to avoid mutating during iteration)
    to_remove = [
        mat
        for mat in bpy.data.materials
        if mat.name.startswith(LAYER_MATERIAL_PREFIX) and mat.users == 0
    ]
    for mat in to_remove:
        bpy.data.materials.remove(mat)

    return saved_layers
