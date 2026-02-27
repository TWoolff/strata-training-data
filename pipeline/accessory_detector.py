"""Detect and hide non-body accessory objects before rendering.

Characters from Mixamo, Sketchfab, and other sources often include weapons,
shields, capes, wings, and armor as separate mesh objects. For v1, these
are hidden from rendering to produce clean body-only training data.

Detection uses three heuristics (any match triggers detection):
1. **Name-based** — object name contains an accessory keyword.
2. **No skinning** — mesh has no Armature modifier or no vertex groups.
3. **Weak skinning** — most vertices weighted to a single bone.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import bpy  # type: ignore[import-untyped]

from .config import (
    ACCESSORY_MAX_VERTEX_GROUPS,
    ACCESSORY_NAME_PATTERNS,
    ACCESSORY_WEAK_SKIN_THRESHOLD,
)

logger = logging.getLogger(__name__)


@dataclass
class AccessoryInfo:
    """Detection result for a single accessory mesh."""

    name: str
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {"name": self.name, "reasons": self.reasons}


@dataclass
class AccessoryResult:
    """Aggregate result of accessory detection for a character."""

    accessories: list[AccessoryInfo] = field(default_factory=list)
    body_meshes: list[bpy.types.Object] = field(default_factory=list)

    @property
    def has_accessories(self) -> bool:
        return bool(self.accessories)

    @property
    def accessory_names(self) -> list[str]:
        return [a.name for a in self.accessories]

    def to_metadata(self) -> dict[str, Any]:
        """Return metadata fields for inclusion in source JSON."""
        return {
            "has_accessories": self.has_accessories,
            "accessories": [a.to_dict() for a in self.accessories],
        }


# ---------------------------------------------------------------------------
# Detection heuristics
# ---------------------------------------------------------------------------

# Pre-compile name pattern regex (case-insensitive)
_ACCESSORY_NAME_RE = re.compile(
    "|".join(re.escape(p) for p in ACCESSORY_NAME_PATTERNS),
    re.IGNORECASE,
)


def _has_accessory_name(mesh_obj: bpy.types.Object) -> bool:
    """Check if the mesh object name matches any accessory keyword."""
    return bool(_ACCESSORY_NAME_RE.search(mesh_obj.name))


def _has_no_skinning(mesh_obj: bpy.types.Object) -> bool:
    """Check if the mesh has no Armature modifier or no vertex groups."""
    has_armature_modifier = any(
        mod.type == "ARMATURE" for mod in mesh_obj.modifiers
    )
    has_vertex_groups = len(mesh_obj.vertex_groups) > 0
    return not has_armature_modifier or not has_vertex_groups


def _has_weak_skinning(mesh_obj: bpy.types.Object) -> bool:
    """Check if most vertices are weighted to very few bones.

    A mesh where ≥80% of vertices are weighted to a single bone is likely
    an accessory parented to that bone (e.g., a sword in the hand).
    """
    if len(mesh_obj.vertex_groups) == 0:
        return False  # No skinning at all — handled by _has_no_skinning

    if len(mesh_obj.vertex_groups) > ACCESSORY_MAX_VERTEX_GROUPS:
        return False  # Too many vertex groups to be an accessory

    mesh_data = mesh_obj.data
    total_verts = len(mesh_data.vertices)
    if total_verts == 0:
        return False

    # Count vertices weighted to exactly 1 bone
    single_bone_count = 0
    for vert in mesh_data.vertices:
        # Get non-zero weights for this vertex
        nonzero_groups = [g for g in vert.groups if g.weight > 0.01]
        if len(nonzero_groups) <= 1:
            single_bone_count += 1

    ratio = single_bone_count / total_verts
    return ratio >= ACCESSORY_WEAK_SKIN_THRESHOLD


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_accessories(
    meshes: list[bpy.types.Object],
) -> AccessoryResult:
    """Detect accessory meshes among the character's mesh objects.

    Args:
        meshes: All mesh objects belonging to the character.

    Returns:
        AccessoryResult with detected accessories and remaining body meshes.
    """
    result = AccessoryResult()

    for mesh_obj in meshes:
        reasons: list[str] = []

        if _has_no_skinning(mesh_obj):
            reasons.append("no_skinning")

        if _has_accessory_name(mesh_obj):
            reasons.append("name_match")

        if not reasons and _has_weak_skinning(mesh_obj):
            reasons.append("weak_skinning")

        if reasons:
            result.accessories.append(
                AccessoryInfo(name=mesh_obj.name, reasons=reasons)
            )
        else:
            result.body_meshes.append(mesh_obj)

    return result


def hide_accessories(
    meshes: list[bpy.types.Object],
    accessory_result: AccessoryResult,
) -> list[bpy.types.Object]:
    """Hide detected accessories from rendering and return body-only meshes.

    Sets ``hide_render = True`` on each detected accessory so it won't
    appear in any rendered output. Returns the list of body meshes that
    should continue through the pipeline.

    Args:
        meshes: All mesh objects belonging to the character.
        accessory_result: Detection results from ``detect_accessories()``.

    Returns:
        List of body mesh objects (accessories excluded).
    """
    accessory_by_name = {a.name: a for a in accessory_result.accessories}

    for mesh_obj in meshes:
        info = accessory_by_name.get(mesh_obj.name)
        if info is not None:
            mesh_obj.hide_render = True
            mesh_obj.hide_viewport = True
            logger.info("Hiding accessory: %s (reasons: %s)", info.name, info.reasons)

    return accessory_result.body_meshes
