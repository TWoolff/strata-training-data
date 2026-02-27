"""Map armature bones to Strata's 19 body regions and assign vertices.

Mapping priority chain:
1. Per-character override JSON (manual assignments)
2. Exact match against MIXAMO_BONE_MAP
3. Exact match against COMMON_BONE_ALIASES
4. Prefix-stripped match (strip known prefixes, retry exact + alias)
5. Substring keyword match (case-insensitive)

Unmapped bones are tracked separately and logged as warnings.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import bpy  # type: ignore[import-untyped]

from .config import (
    COMMON_BONE_ALIASES,
    COMMON_PREFIXES,
    MIXAMO_BONE_MAP,
    SUBSTRING_KEYWORDS,
    RegionId,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass
class MappingStats:
    """Counts of how each bone was mapped."""

    exact: int = 0
    alias: int = 0
    prefix: int = 0
    substring: int = 0
    override: int = 0


@dataclass
class BoneMapping:
    """Result of mapping an armature's bones to Strata regions."""

    bone_to_region: dict[str, RegionId] = field(default_factory=dict)
    vertex_to_region: dict[int, RegionId] = field(default_factory=dict)
    unmapped_bones: list[str] = field(default_factory=list)
    mapping_stats: MappingStats = field(default_factory=MappingStats)


# ---------------------------------------------------------------------------
# Override loading
# ---------------------------------------------------------------------------


def _load_overrides(character_id: str, source_dir: Path) -> dict[str, RegionId]:
    """Load per-character bone mapping overrides from JSON.

    Args:
        character_id: Character identifier (filename stem).
        source_dir: Directory containing source characters and override files.

    Returns:
        Dict of bone_name → region_id overrides, or empty dict if no file.
    """
    override_path = source_dir / f"{character_id}_overrides.json"
    if not override_path.is_file():
        return {}

    try:
        data = json.loads(override_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            logger.warning("Override file %s is not a JSON object — skipping", override_path)
            return {}
        return {str(k): int(v) for k, v in data.items()}
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse override file %s: %s", override_path, exc)
        return {}


# ---------------------------------------------------------------------------
# Matching strategies
# ---------------------------------------------------------------------------


def _try_exact(bone_name: str) -> RegionId | None:
    """Exact match against MIXAMO_BONE_MAP."""
    return MIXAMO_BONE_MAP.get(bone_name)


def _try_alias(bone_name: str) -> RegionId | None:
    """Exact match against COMMON_BONE_ALIASES."""
    return COMMON_BONE_ALIASES.get(bone_name)


def _try_prefix_strip(bone_name: str) -> RegionId | None:
    """Strip known prefixes and retry exact + alias match."""
    for prefix in COMMON_PREFIXES:
        if bone_name.startswith(prefix):
            stripped = bone_name[len(prefix) :]
            region = MIXAMO_BONE_MAP.get(stripped) or COMMON_BONE_ALIASES.get(stripped)
            if region is not None:
                return region
    return None


def _try_substring(bone_name: str) -> RegionId | None:
    """Case-insensitive substring keyword matching.

    Checks each keyword tuple in SUBSTRING_KEYWORDS. All keywords in the
    tuple must appear as substrings of the lowercased bone name.
    """
    name_lower = bone_name.lower()
    for keywords, region_id in SUBSTRING_KEYWORDS:
        if all(kw in name_lower for kw in keywords):
            return region_id
    return None


# ---------------------------------------------------------------------------
# Bone mapping
# ---------------------------------------------------------------------------


def _map_all_bones(
    armature: bpy.types.Object,
    overrides: dict[str, RegionId],
) -> tuple[dict[str, RegionId], list[str], MappingStats]:
    """Map every bone in the armature to a Strata region.

    Args:
        armature: Blender armature object.
        overrides: Per-character manual bone assignments.

    Returns:
        (bone_to_region, unmapped_bones, mapping_stats).
    """
    bone_to_region: dict[str, RegionId] = {}
    unmapped: list[str] = []
    stats = MappingStats()

    for bone in armature.data.bones:
        name = bone.name

        # 1. Manual override
        if name in overrides:
            bone_to_region[name] = overrides[name]
            stats.override += 1
            continue

        # 2. Exact match (Mixamo)
        region = _try_exact(name)
        if region is not None:
            bone_to_region[name] = region
            stats.exact += 1
            continue

        # 3. Alias match
        region = _try_alias(name)
        if region is not None:
            bone_to_region[name] = region
            stats.alias += 1
            continue

        # 4. Prefix strip + retry
        region = _try_prefix_strip(name)
        if region is not None:
            bone_to_region[name] = region
            stats.prefix += 1
            continue

        # 5. Substring match
        region = _try_substring(name)
        if region is not None:
            bone_to_region[name] = region
            stats.substring += 1
            continue

        # Unmapped
        unmapped.append(name)

    return bone_to_region, unmapped, stats


# ---------------------------------------------------------------------------
# Vertex assignment
# ---------------------------------------------------------------------------


def _assign_vertices(
    meshes: list[bpy.types.Object],
    bone_to_region: dict[str, RegionId],
) -> dict[int, RegionId]:
    """Assign each mesh vertex to a region based on dominant bone weight.

    For each vertex, finds the vertex group with the highest weight, looks up
    that group's name in bone_to_region, and assigns the corresponding region.
    Vertices with no bone weights get region 0 (background).

    The returned dict uses a composite key: ``mesh_index * 10_000_000 + vertex_index``
    to uniquely identify vertices across multiple meshes.

    Args:
        meshes: List of mesh objects parented to the armature.
        bone_to_region: Bone name → region ID mapping from _map_all_bones.

    Returns:
        Dict of composite_vertex_id → region_id.
    """
    vertex_to_region: dict[int, RegionId] = {}
    no_weight_count = 0

    for mesh_idx, mesh_obj in enumerate(meshes):
        mesh_data = mesh_obj.data

        # Build group index → group name lookup
        group_names: dict[int, str] = {g.index: g.name for g in mesh_obj.vertex_groups}

        base_id = mesh_idx * 10_000_000

        for vert in mesh_data.vertices:
            if not vert.groups:
                vertex_to_region[base_id + vert.index] = 0
                no_weight_count += 1
                continue

            # Find dominant bone (highest weight)
            best_weight = -1.0
            best_group_name = ""
            for g in vert.groups:
                if g.weight > best_weight:
                    best_weight = g.weight
                    best_group_name = group_names.get(g.group, "")

            region = bone_to_region.get(best_group_name, 0)
            vertex_to_region[base_id + vert.index] = region

    if no_weight_count > 0:
        logger.warning(
            "%d vertices have no bone weights — assigned to background (region 0)",
            no_weight_count,
        )

    return vertex_to_region


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def map_bones(
    armature: bpy.types.Object,
    meshes: list[bpy.types.Object],
    character_id: str,
    source_dir: Path | None = None,
) -> BoneMapping:
    """Map all bones in an armature to Strata regions and assign vertices.

    Args:
        armature: Blender armature object from the imported character.
        meshes: List of mesh objects belonging to the character.
        character_id: Character identifier (used to find override JSON).
        source_dir: Directory containing source characters. If None,
            override loading is skipped.

    Returns:
        A BoneMapping with bone_to_region, vertex_to_region, unmapped_bones,
        and mapping_stats.
    """
    # Load overrides
    overrides: dict[str, RegionId] = {}
    if source_dir is not None:
        overrides = _load_overrides(character_id, source_dir)
        if overrides:
            logger.info("Loaded %d bone override(s) for %s", len(overrides), character_id)

    # Map bones
    bone_to_region, unmapped, stats = _map_all_bones(armature, overrides)

    # Log results
    total = len(bone_to_region) + len(unmapped)
    logger.info(
        "Bone mapping for %s: %d/%d mapped "
        "(exact=%d, alias=%d, prefix=%d, substring=%d, override=%d), "
        "%d unmapped",
        character_id,
        len(bone_to_region),
        total,
        stats.exact,
        stats.alias,
        stats.prefix,
        stats.substring,
        stats.override,
        len(unmapped),
    )

    if unmapped:
        logger.warning("Unmapped bones in %s: %s", character_id, unmapped)

    # Assign vertices
    vertex_to_region = _assign_vertices(meshes, bone_to_region)

    return BoneMapping(
        bone_to_region=bone_to_region,
        vertex_to_region=vertex_to_region,
        unmapped_bones=unmapped,
        mapping_stats=stats,
    )
