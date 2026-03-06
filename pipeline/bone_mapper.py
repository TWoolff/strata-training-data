"""Map armature bones to Strata's 19 body regions and assign vertices.

Mapping priority chain:
1. Per-character override JSON (manual assignments)
2. Exact match against MIXAMO_BONE_MAP
3. Exact match against COMMON_BONE_ALIASES
4. Exact match against VRM_BONE_ALIASES (VRM humanoid skeleton)
5. Prefix-stripped match (strip known prefixes, retry exact + alias)
6. Substring keyword match (case-insensitive)
7. Fuzzy keyword match (normalized tokens, scored against keyword patterns)

Unmapped bones are tracked separately and logged as warnings.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import bpy  # type: ignore[import-untyped]

from .config import (
    COMMON_BONE_ALIASES,
    COMMON_PREFIXES,
    FUZZY_KEYWORD_PATTERNS,
    FUZZY_MIN_SCORE,
    LATERALITY_ALIASES,
    MIXAMO_BONE_MAP,
    SUBSTRING_KEYWORDS,
    VRM_BONE_ALIASES,
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
    fuzzy: int = 0
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

        overrides: dict[str, RegionId] = {}
        for bone_name, region_id in data.items():
            # Skip null placeholders (template file format)
            if region_id is None:
                continue
            try:
                rid = int(region_id)
            except (TypeError, ValueError):
                logger.warning(
                    "Override %s: bone %r has non-integer region ID %r — skipping",
                    override_path.name, bone_name, region_id,
                )
                continue
            if rid < 1 or rid > 19:
                logger.warning(
                    "Override %s: bone %r has invalid region ID %d (must be 1-19) -- skipping",
                    override_path.name, bone_name, rid,
                )
                continue
            overrides[str(bone_name)] = rid

        return overrides
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse override file %s: %s", override_path, exc)
        return {}


# ---------------------------------------------------------------------------
# Template generation
# ---------------------------------------------------------------------------


def generate_override_template(
    character_id: str,
    unmapped_bones: list[str],
    source_dir: Path,
) -> Path | None:
    """Generate a template override JSON for a character's unmapped bones.

    Writes a JSON file with each unmapped bone name mapped to ``null``,
    ready for manual editing. Skips generation if there are no unmapped bones.

    Args:
        character_id: Character identifier (filename stem).
        unmapped_bones: List of bone names that weren't auto-mapped.
        source_dir: Directory to write the template file into.

    Returns:
        Path to the generated template file, or None if no unmapped bones.
    """
    if not unmapped_bones:
        return None

    template_path = source_dir / f"{character_id}_overrides.json"
    if template_path.exists():
        logger.info(
            "Override file already exists for %s — skipping template generation",
            character_id,
        )
        return None

    template = {bone: None for bone in sorted(unmapped_bones)}
    template_path.write_text(
        json.dumps(template, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        "Generated override template for %s with %d unmapped bones: %s",
        character_id, len(unmapped_bones), template_path,
    )
    return template_path


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

    Single-character laterality markers (``l``, ``r``) are matched as whole
    tokens (bounded by ``_``, ``.``, ``-``, start, or end of string) to
    avoid false matches like ``"l"`` matching inside ``"ball"`` or ``"leaf"``.
    """
    name_lower = bone_name.lower()
    # Pre-split into tokens for single-char laterality checks
    tokens = set(re.split(r"[_.\-\s]+", name_lower))

    for keywords, region_id in SUBSTRING_KEYWORDS:
        matched = True
        for kw in keywords:
            if len(kw) == 1 and kw in ("l", "r"):
                # Single-char laterality: must be a whole token
                if kw not in tokens:
                    matched = False
                    break
            else:
                if kw not in name_lower:
                    matched = False
                    break
        if matched:
            return region_id
    return None


# ---------------------------------------------------------------------------
# Fuzzy keyword matching
# ---------------------------------------------------------------------------

# Regex to insert an underscore at camelCase boundaries: "LeftArm" → "Left_Arm"
_CAMEL_RE = re.compile(r"([a-z])([A-Z])")
# Regex to strip trailing numeric suffixes: "spine.001" → "spine"
_NUMERIC_SUFFIX_RE = re.compile(r"[._]\d+$")


def _normalize_bone_name(name: str) -> list[str]:
    """Normalize a bone name into lowercase tokens for fuzzy matching.

    Steps:
        1. Strip known prefixes (COMMON_PREFIXES).
        2. Strip trailing numeric suffixes (.001, _02, etc.).
        3. Split camelCase boundaries.
        4. Split on ``_``, ``.``, ``-``, and whitespace.
        5. Lowercase all tokens.
        6. Drop empty tokens.

    Args:
        name: Raw bone name from armature.

    Returns:
        List of lowercase string tokens.
    """
    # 1. Strip known prefixes
    for prefix in COMMON_PREFIXES:
        if name.lower().startswith(prefix.lower()):
            name = name[len(prefix) :]
            break

    # 2. Strip trailing numeric suffix
    name = _NUMERIC_SUFFIX_RE.sub("", name)

    # 3. Split camelCase
    name = _CAMEL_RE.sub(r"\1_\2", name)

    # 4. Split on delimiters and lowercase
    tokens = re.split(r"[_.\-\s]+", name.lower())

    # 5. Drop empty
    return [t for t in tokens if t]


def _canonicalize_laterality(tokens: list[str]) -> list[str]:
    """Replace laterality aliases with canonical 'left'/'right'.

    Only replaces whole tokens that are exact laterality markers. This avoids
    turning ``"leg"`` into ``"lefteg"`` — the token ``"l"`` is only replaced
    when it appears as a standalone token (i.e., after splitting on delimiters).

    Args:
        tokens: Normalized tokens from ``_normalize_bone_name``.

    Returns:
        New token list with laterality aliases replaced.
    """
    return [LATERALITY_ALIASES.get(t, t) for t in tokens]


def _try_fuzzy_keyword(bone_name: str) -> tuple[RegionId | None, float]:
    """Score-based fuzzy keyword matching against FUZZY_KEYWORD_PATTERNS.

    Normalizes the bone name into tokens, canonicalizes laterality markers,
    then scores each pattern. A match requires all keywords to appear in the
    token set and the score (matched / total keywords) to meet FUZZY_MIN_SCORE.

    Args:
        bone_name: Raw bone name from the armature.

    Returns:
        Tuple of (region_id, score) for the best match, or (None, 0.0).
    """
    tokens = _normalize_bone_name(bone_name)
    tokens = _canonicalize_laterality(tokens)
    token_set = set(tokens)

    best_region: RegionId | None = None
    best_score: float = 0.0
    best_keyword_count: int = 0

    for keywords, region_id in FUZZY_KEYWORD_PATTERNS:
        matched = sum(1 for kw in keywords if kw in token_set)
        total = len(keywords)
        if total == 0:
            continue

        score = matched / total
        if score < FUZZY_MIN_SCORE:
            continue

        # Prefer higher score, then more-specific pattern (more keywords)
        if score > best_score or (score == best_score and total > best_keyword_count):
            best_region = region_id
            best_score = score
            best_keyword_count = total

    return best_region, best_score


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

    # Warn about overrides that reference nonexistent bones
    armature_bone_names = {bone.name for bone in armature.data.bones}
    for override_name in overrides:
        if override_name not in armature_bone_names:
            logger.warning(
                "Override bone %r not found in armature — will be ignored", override_name
            )

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

        # 4. VRM humanoid bone alias
        region = VRM_BONE_ALIASES.get(name)
        if region is not None:
            bone_to_region[name] = region
            stats.alias += 1
            continue

        # 5. Prefix strip + retry
        region = _try_prefix_strip(name)
        if region is not None:
            bone_to_region[name] = region
            stats.prefix += 1
            continue

        # 6. Substring match
        region = _try_substring(name)
        if region is not None:
            bone_to_region[name] = region
            stats.substring += 1
            continue

        # 7. Fuzzy keyword match
        region, score = _try_fuzzy_keyword(name)
        if region is not None:
            bone_to_region[name] = region
            stats.fuzzy += 1
            logger.debug(
                "Fuzzy match: %r → region %d (score=%.2f)", name, region, score
            )
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
        "(exact=%d, alias=%d, prefix=%d, substring=%d, fuzzy=%d, override=%d), "
        "%d unmapped",
        character_id,
        len(bone_to_region),
        total,
        stats.exact,
        stats.alias,
        stats.prefix,
        stats.substring,
        stats.fuzzy,
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
