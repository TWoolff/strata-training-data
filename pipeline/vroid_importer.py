"""Import VRM/VRoid characters into Blender and normalize for the render pipeline.

Handles VRM file import (via the VRM Add-on for Blender), armature/mesh
discovery, A-pose normalization, MToon material conversion, and scale/position
normalization. Returns the same ImportResult dataclass used by the FBX importer
so downstream modules (bone mapping, rendering, export) work unchanged.

Requires the VRM Add-on for Blender to be installed:
https://vrm-addon-for-blender.info/
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import bpy  # type: ignore[import-untyped]
from mathutils import Euler  # type: ignore[import-untyped]

from .config import A_POSE_SHOULDER_ANGLE
from .importer import ImportResult, _normalize_transforms, clear_scene

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Upper arm bone name keywords for A-pose normalization (VRM humanoid names)
_VRM_LEFT_UPPER_ARM_KEYWORDS = (
    "leftupperarm",
    "upper_arm.l",
    "l_upperarm",
    "upperarm.l",
    "leftarm",
)
_VRM_RIGHT_UPPER_ARM_KEYWORDS = (
    "rightupperarm",
    "upper_arm.r",
    "r_upperarm",
    "upperarm.r",
    "rightarm",
)


def _vrm_addon_available() -> bool:
    """Check whether the VRM Add-on for Blender is installed and enabled."""
    return hasattr(bpy.ops.import_scene, "vrm")


# ---------------------------------------------------------------------------
# A-pose normalization
# ---------------------------------------------------------------------------


def _apply_apose(armature: bpy.types.Object) -> None:
    """Set the character to A-pose by rotating upper arms 45 degrees down.

    VRoid models typically default to T-pose or A-pose. This ensures
    consistent A-pose across all models.
    """
    angle_rad = math.radians(A_POSE_SHOULDER_ANGLE)

    for pbone in armature.pose.bones:
        # Reset all bones to rest pose first
        pbone.rotation_mode = "XYZ"
        pbone.rotation_euler = Euler((0.0, 0.0, 0.0), "XYZ")
        pbone.location = (0.0, 0.0, 0.0)

    for pbone in armature.pose.bones:
        name_lower = pbone.name.lower()

        is_left_arm = any(kw in name_lower for kw in _VRM_LEFT_UPPER_ARM_KEYWORDS)
        is_right_arm = any(kw in name_lower for kw in _VRM_RIGHT_UPPER_ARM_KEYWORDS)

        if is_left_arm:
            pbone.rotation_mode = "XYZ"
            pbone.rotation_euler = Euler((0.0, 0.0, angle_rad), "XYZ")
        elif is_right_arm:
            pbone.rotation_mode = "XYZ"
            pbone.rotation_euler = Euler((0.0, 0.0, -angle_rad), "XYZ")

    bpy.context.view_layer.update()


# ---------------------------------------------------------------------------
# MToon material conversion
# ---------------------------------------------------------------------------


def _convert_mtoon_materials(meshes: list[bpy.types.Object]) -> None:
    """Convert VRM MToon shader materials to Principled BSDF for style compat.

    MToon is a toon shader used by most VRoid models. The pipeline's
    style augmentation (flat, cel, unlit) expects Principled BSDF or at
    minimum a base color + texture. This function extracts the base color
    texture and/or color from MToon nodes and rebuilds the material as
    Principled BSDF.

    Materials that are already Principled BSDF (or have no node tree)
    are left unchanged.
    """
    converted = 0

    for mesh_obj in meshes:
        for slot in mesh_obj.material_slots:
            mat = slot.material
            if mat is None or not mat.use_nodes:
                continue

            nodes = mat.node_tree.nodes
            links = mat.node_tree.links

            # Check if already has Principled BSDF
            principled = None
            for node in nodes:
                if node.type == "BSDF_PRINCIPLED":
                    principled = node
                    break

            if principled is not None:
                continue

            # Find base color texture or color from MToon / Group nodes
            base_color = (0.8, 0.8, 0.8, 1.0)
            base_texture = None

            for node in nodes:
                # MToon group nodes often have a "MainTexture" or "Lit Color" input
                if node.type == "GROUP":
                    for inp in node.inputs:
                        inp_name = inp.name.lower()
                        if "maintexture" in inp_name or "lit" in inp_name:
                            if inp.is_linked:
                                linked_node = inp.links[0].from_node
                                if linked_node.type == "TEX_IMAGE":
                                    base_texture = linked_node
                            elif hasattr(inp, "default_value"):
                                base_color = tuple(inp.default_value)
                        elif (
                            "color" in inp_name
                            and not inp.is_linked
                            and hasattr(inp, "default_value")
                        ):
                            base_color = tuple(inp.default_value)

                # Also check standalone image texture nodes
                elif node.type == "TEX_IMAGE" and base_texture is None:
                    base_texture = node

            # Rebuild as Principled BSDF
            output_node = None
            for node in nodes:
                if node.type == "OUTPUT_MATERIAL":
                    output_node = node
                    break

            if output_node is None:
                continue

            # Clear existing links to output
            for link in list(links):
                if link.to_node == output_node:
                    links.remove(link)

            # Remove old non-texture nodes
            nodes_to_keep = set()
            if base_texture is not None:
                nodes_to_keep.add(base_texture.name)
            nodes_to_keep.add(output_node.name)

            for node in list(nodes):
                if node.name not in nodes_to_keep:
                    nodes.remove(node)

            # Create Principled BSDF
            principled = nodes.new("ShaderNodeBsdfPrincipled")
            principled.location = (0, 300)
            principled.inputs["Roughness"].default_value = 1.0
            principled.inputs["Specular IOR Level"].default_value = 0.0

            if base_texture is not None:
                base_texture.location = (-400, 300)
                links.new(base_texture.outputs["Color"], principled.inputs["Base Color"])
                # Preserve alpha for cutout materials (hair, etc.)
                if base_texture.outputs.get("Alpha"):
                    links.new(
                        base_texture.outputs["Alpha"], principled.inputs["Alpha"]
                    )
                    mat.blend_method = "CLIP"  # type: ignore[attr-defined]
            else:
                principled.inputs["Base Color"].default_value = base_color

            links.new(principled.outputs["BSDF"], output_node.inputs["Surface"])
            converted += 1

    if converted > 0:
        logger.info("Converted %d MToon material(s) to Principled BSDF", converted)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def import_vrm(vrm_path: Path) -> ImportResult | None:
    """Import a VRM character, normalize it, and return structured references.

    Args:
        vrm_path: Path to the .vrm file.

    Returns:
        An ImportResult with armature/mesh references, or None if the file
        is invalid or the VRM add-on is not available.
    """
    vrm_path = Path(vrm_path)
    character_id = f"vroid_{vrm_path.stem}"

    if not vrm_path.is_file():
        logger.error("VRM file not found: %s", vrm_path)
        return None

    if not _vrm_addon_available():
        logger.error(
            "VRM Add-on for Blender is not installed. "
            "Install from: https://vrm-addon-for-blender.info/"
        )
        return None

    # Clean slate
    clear_scene()

    # Import VRM
    try:
        bpy.ops.import_scene.vrm(filepath=str(vrm_path))
    except Exception:
        logger.exception("Failed to import VRM: %s", vrm_path)
        return None

    # Discover armature(s) and mesh(es)
    armatures: list[bpy.types.Object] = []
    meshes: list[bpy.types.Object] = []

    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            armatures.append(obj)
        elif obj.type == "MESH":
            meshes.append(obj)

    if not armatures:
        logger.error("No armature found in %s — skipping", vrm_path.name)
        return None

    if not meshes:
        logger.error("No mesh found in %s — skipping", vrm_path.name)
        return None

    if len(armatures) > 1:
        logger.warning(
            "Multiple armatures (%d) in %s — using the first one",
            len(armatures),
            vrm_path.name,
        )

    armature = armatures[0]

    # Convert MToon materials before any rendering
    _convert_mtoon_materials(meshes)

    # Apply A-pose normalization (VRoid models are typically T-pose)
    _apply_apose(armature)

    # Normalize scale and position
    _normalize_transforms(armature, meshes)

    logger.info(
        "Imported VRM %s: armature=%s, meshes=%d, bones=%d",
        character_id,
        armature.name,
        len(meshes),
        len(armature.data.bones),
    )

    return ImportResult(
        character_id=character_id,
        armature=armature,
        meshes=meshes,
    )
