"""Map UniRig bone hierarchies to Strata's 20-bone skeleton.

UniRig (SIGGRAPH 2025) provides 14K+ rigged 3D meshes with auto-generated
skeletons from Objaverse and VRoid sources.  Bone naming conventions vary
widely across models, so this mapper reuses the fuzzy matching approach from
``pipeline/bone_mapper.py`` without requiring Blender.

Input data format (per character)::

    {character_id}.npz or {character_id}.json with keys:
    - joint_names:  list of bone/joint name strings
    - joint_positions:  (N, 3) array of 3D joint positions
    - skinning_weights:  (V, N) sparse or dense skinning weight matrix
    - parent_indices:  (N,) array of parent joint indices (-1 for root)

Output per character::

    output_dir/{character_id}_mapping.json
    {
      "character_id": "...",
      "source": "unirig",
      "total_joints": N,
      "mapped_joints": M,
      "unmapped_joints": [...],
      "auto_match_rate": 0.85,
      "joint_mappings": { "bone_name": {"region_id": 3, "region_name": "chest", ...} },
      "validation": { "has_root": true, "has_limbs": true, ... }
    }

This module is pure Python (no Blender dependency) so it can be imported
outside Blender for testing and validation.

Reference: https://github.com/VAST-AI-Research/UniRig
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.config import (
    COMMON_BONE_ALIASES,
    COMMON_PREFIXES,
    FUZZY_KEYWORD_PATTERNS,
    FUZZY_MIN_SCORE,
    LATERALITY_ALIASES,
    MIXAMO_BONE_MAP,
    REGION_NAMES,
    SUBSTRING_KEYWORDS,
    VRM_BONE_ALIASES,
    RegionId,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UNIRIG_SOURCE = "unirig"

# Minimum auto-match rate to consider a character's mapping "good".
GOOD_MATCH_THRESHOLD = 0.80

# Required body regions for a valid humanoid skeleton.
_REQUIRED_REGIONS = frozenset(
    {
        1,  # head
        3,  # chest
        5,  # hips
        7,  # upper_arm_l
        11,  # upper_arm_r
        14,  # upper_leg_l
        17,  # upper_leg_r
    }
)

# Regex patterns reused from bone_mapper.py for normalization.
_CAMEL_RE = re.compile(r"([a-z])([A-Z])")
_NUMERIC_SUFFIX_RE = re.compile(r"[._]\d+$")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class JointMapping:
    """Mapping result for a single joint/bone."""

    joint_name: str
    region_id: RegionId | None
    region_name: str | None
    method: str  # "exact", "alias", "vrm", "prefix", "substring", "fuzzy", or "unmapped"
    score: float = 1.0  # Only meaningful for fuzzy matches


@dataclass
class SkeletonValidation:
    """Validation results for a mapped skeleton."""

    has_root: bool = False
    has_head: bool = False
    has_limbs: bool = False
    has_symmetric_arms: bool = False
    has_symmetric_legs: bool = False
    missing_regions: list[str] = field(default_factory=list)


@dataclass
class CharacterSkeletonMapping:
    """Full mapping result for a UniRig character."""

    character_id: str
    total_joints: int = 0
    joint_mappings: list[JointMapping] = field(default_factory=list)
    validation: SkeletonValidation = field(default_factory=SkeletonValidation)

    @property
    def mapped_joints(self) -> int:
        """Number of successfully mapped joints."""
        return sum(1 for jm in self.joint_mappings if jm.region_id is not None)

    @property
    def unmapped_joints(self) -> list[str]:
        """Names of joints that could not be mapped."""
        return [jm.joint_name for jm in self.joint_mappings if jm.region_id is None]

    @property
    def auto_match_rate(self) -> float:
        """Fraction of joints that were auto-mapped."""
        if self.total_joints == 0:
            return 0.0
        return self.mapped_joints / self.total_joints

    @property
    def region_coverage(self) -> dict[str, int]:
        """Count of joints mapped to each region."""
        dist: dict[str, int] = {}
        for jm in self.joint_mappings:
            if jm.region_name is not None:
                dist[jm.region_name] = dist.get(jm.region_name, 0) + 1
        return dist


@dataclass
class AdapterResult:
    """Result of processing a batch of UniRig characters."""

    characters_processed: int = 0
    characters_good: int = 0  # Match rate >= GOOD_MATCH_THRESHOLD
    characters_poor: int = 0  # Match rate < GOOD_MATCH_THRESHOLD
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Bone matching (reuses logic from pipeline/bone_mapper.py without bpy)
# ---------------------------------------------------------------------------


def _normalize_bone_name(name: str) -> list[str]:
    """Normalize a bone name into lowercase tokens for fuzzy matching.

    Mirrors ``pipeline.bone_mapper._normalize_bone_name``.

    Args:
        name: Raw bone/joint name.

    Returns:
        List of lowercase string tokens.
    """
    for prefix in COMMON_PREFIXES:
        if name.lower().startswith(prefix.lower()):
            name = name[len(prefix) :]
            break

    name = _NUMERIC_SUFFIX_RE.sub("", name)
    name = _CAMEL_RE.sub(r"\1_\2", name)
    tokens = re.split(r"[_.\-\s]+", name.lower())
    return [t for t in tokens if t]


def _canonicalize_laterality(tokens: list[str]) -> list[str]:
    """Replace laterality aliases with canonical left/right."""
    return [LATERALITY_ALIASES.get(t, t) for t in tokens]


def map_joint_name(joint_name: str) -> JointMapping:
    """Map a single joint name to a Strata region using the full matching chain.

    Priority:
    1. Exact match (MIXAMO_BONE_MAP)
    2. Alias match (COMMON_BONE_ALIASES)
    3. VRM alias match (VRM_BONE_ALIASES)
    4. Prefix-stripped retry
    5. Substring keyword match
    6. Fuzzy keyword match

    Args:
        joint_name: Raw joint/bone name from UniRig data.

    Returns:
        JointMapping with region assignment and method used.
    """
    # 1. Exact match (Mixamo)
    region = MIXAMO_BONE_MAP.get(joint_name)
    if region is not None:
        return JointMapping(
            joint_name=joint_name,
            region_id=region,
            region_name=REGION_NAMES[region],
            method="exact",
        )

    # 2. Alias match
    region = COMMON_BONE_ALIASES.get(joint_name)
    if region is not None:
        return JointMapping(
            joint_name=joint_name,
            region_id=region,
            region_name=REGION_NAMES[region],
            method="alias",
        )

    # 3. VRM alias
    region = VRM_BONE_ALIASES.get(joint_name)
    if region is not None:
        return JointMapping(
            joint_name=joint_name,
            region_id=region,
            region_name=REGION_NAMES[region],
            method="vrm",
        )

    # 4. Prefix strip + retry
    for prefix in COMMON_PREFIXES:
        if joint_name.startswith(prefix):
            stripped = joint_name[len(prefix) :]
            region = (
                MIXAMO_BONE_MAP.get(stripped)
                or COMMON_BONE_ALIASES.get(stripped)
                or VRM_BONE_ALIASES.get(stripped)
            )
            if region is not None:
                return JointMapping(
                    joint_name=joint_name,
                    region_id=region,
                    region_name=REGION_NAMES[region],
                    method="prefix",
                )

    # 5. Substring keyword match
    name_lower = joint_name.lower()
    for keywords, region_id in SUBSTRING_KEYWORDS:
        if all(kw in name_lower for kw in keywords):
            return JointMapping(
                joint_name=joint_name,
                region_id=region_id,
                region_name=REGION_NAMES[region_id],
                method="substring",
            )

    # 6. Fuzzy keyword match
    tokens = _normalize_bone_name(joint_name)
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
        if score > best_score or (score == best_score and total > best_keyword_count):
            best_region = region_id
            best_score = score
            best_keyword_count = total

    if best_region is not None:
        return JointMapping(
            joint_name=joint_name,
            region_id=best_region,
            region_name=REGION_NAMES[best_region],
            method="fuzzy",
            score=best_score,
        )

    # Unmapped
    return JointMapping(
        joint_name=joint_name,
        region_id=None,
        region_name=None,
        method="unmapped",
    )


# ---------------------------------------------------------------------------
# Skeleton validation
# ---------------------------------------------------------------------------


def validate_skeleton(mapping: CharacterSkeletonMapping) -> SkeletonValidation:
    """Validate a mapped skeleton for humanoid completeness.

    Checks for presence of root, head, symmetric limbs, and required regions.

    Args:
        mapping: Character skeleton mapping to validate.

    Returns:
        SkeletonValidation with check results.
    """
    mapped_regions = {jm.region_id for jm in mapping.joint_mappings if jm.region_id is not None}

    validation = SkeletonValidation()
    validation.has_root = 5 in mapped_regions  # hips
    validation.has_head = 1 in mapped_regions
    validation.has_symmetric_arms = 7 in mapped_regions and 11 in mapped_regions
    validation.has_symmetric_legs = 14 in mapped_regions and 17 in mapped_regions
    validation.has_limbs = validation.has_symmetric_arms and validation.has_symmetric_legs

    missing = _REQUIRED_REGIONS - mapped_regions
    validation.missing_regions = sorted(REGION_NAMES[r] for r in missing)

    return validation


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_joint_names_npz(npz_path: Path) -> list[str] | None:
    """Load joint names from a UniRig .npz file.

    Supports both ``joint_names`` (legacy schema) and ``names`` (Rig-XL schema).

    Args:
        npz_path: Path to the .npz file.

    Returns:
        List of joint name strings, or None if loading fails.
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        # Rig-XL uses 'names'; older schema used 'joint_names'
        key = "names" if "names" in data else "joint_names" if "joint_names" in data else None
        if key is None:
            logger.warning("No 'names' or 'joint_names' key in %s", npz_path)
            return None
        return [str(n) for n in data[key]]
    except (OSError, ValueError) as exc:
        logger.warning("Failed to load %s: %s", npz_path, exc)
        return None


def load_joint_names_json(json_path: Path) -> list[str] | None:
    """Load joint names from a UniRig JSON file.

    Args:
        json_path: Path to the JSON file.

    Returns:
        List of joint name strings, or None if loading fails.
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            names = data.get("joint_names")
            if names is None:
                logger.warning("No 'joint_names' key in %s", json_path)
                return None
            return [str(n) for n in names]
        logger.warning("Unexpected JSON structure in %s", json_path)
        return None
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load %s: %s", json_path, exc)
        return None


# ---------------------------------------------------------------------------
# Core mapping
# ---------------------------------------------------------------------------


def map_skeleton(
    character_id: str,
    joint_names: list[str],
) -> CharacterSkeletonMapping:
    """Map a UniRig skeleton's joints to Strata regions.

    Args:
        character_id: Unique character identifier.
        joint_names: List of joint/bone names from the skeleton.

    Returns:
        CharacterSkeletonMapping with per-joint results and validation.
    """
    mapping = CharacterSkeletonMapping(
        character_id=character_id,
        total_joints=len(joint_names),
    )

    for name in joint_names:
        jm = map_joint_name(name)
        mapping.joint_mappings.append(jm)

    mapping.validation = validate_skeleton(mapping)

    rate = mapping.auto_match_rate
    logger.info(
        "UniRig mapping for %s: %d/%d joints mapped (%.0f%%), unmapped: %s",
        character_id,
        mapping.mapped_joints,
        mapping.total_joints,
        rate * 100,
        mapping.unmapped_joints or "(none)",
    )

    return mapping


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def export_mapping_json(
    mapping: CharacterSkeletonMapping,
    output_path: Path,
) -> None:
    """Export skeleton mapping to JSON.

    Args:
        mapping: Character skeleton mapping to export.
        output_path: Path for the output JSON file.
    """
    data: dict[str, Any] = {
        "character_id": mapping.character_id,
        "source": UNIRIG_SOURCE,
        "total_joints": mapping.total_joints,
        "mapped_joints": mapping.mapped_joints,
        "unmapped_joints": mapping.unmapped_joints,
        "auto_match_rate": round(mapping.auto_match_rate, 4),
        "region_coverage": mapping.region_coverage,
        "joint_mappings": {
            jm.joint_name: {
                "region_id": jm.region_id,
                "region_name": jm.region_name,
                "method": jm.method,
                "score": round(jm.score, 4) if jm.method == "fuzzy" else None,
            }
            for jm in mapping.joint_mappings
        },
        "validation": {
            "has_root": mapping.validation.has_root,
            "has_head": mapping.validation.has_head,
            "has_limbs": mapping.validation.has_limbs,
            "has_symmetric_arms": mapping.validation.has_symmetric_arms,
            "has_symmetric_legs": mapping.validation.has_symmetric_legs,
            "missing_regions": mapping.validation.missing_regions,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    logger.info("Exported UniRig mapping to %s", output_path)


def load_mapping_json(json_path: Path) -> dict[str, Any]:
    """Load a previously exported mapping JSON.

    Args:
        json_path: Path to the JSON file.

    Returns:
        Parsed mapping dict.
    """
    return json.loads(json_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def convert_directory(
    unirig_dir: Path,
    output_dir: Path,
    *,
    max_characters: int = 0,
    only_new: bool = False,
) -> AdapterResult:
    """Process all UniRig characters in a directory.

    Discovers .npz and .json files, maps each skeleton, and exports
    mapping JSON reports.

    Args:
        unirig_dir: Root UniRig dataset directory.
        output_dir: Output directory for mapping JSON files.
        max_characters: Maximum characters to process (0 = unlimited).
        only_new: Skip characters with existing mapping files.

    Returns:
        AdapterResult summarizing the batch processing.
    """
    if not unirig_dir.is_dir():
        logger.error("UniRig directory not found: %s", unirig_dir)
        return AdapterResult(errors=["Directory not found"])

    # Discover character files: flat .npz/.json files OR subdirectories
    # containing raw_data.npz (Rig-XL layout: rigxl/00000/raw_data.npz)
    candidates: list[Path] = []
    for p in sorted(unirig_dir.iterdir()):
        if p.is_dir():
            raw = p / "raw_data.npz"
            if raw.is_file():
                candidates.append(raw)
        elif p.is_file() and p.suffix in (".npz", ".json") and not p.name.startswith("."):
            candidates.append(p)

    if not candidates:
        logger.warning("No character files found in %s", unirig_dir)
        return AdapterResult()

    if max_characters > 0:
        candidates = candidates[:max_characters]

    logger.info("Found %d character files in %s", len(candidates), unirig_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    result = AdapterResult()

    for char_path in candidates:
        # Use parent dir name for subdirectory layout (e.g. "00000"), else stem
        if char_path.name == "raw_data.npz":
            character_id = char_path.parent.name
        else:
            character_id = char_path.stem
        output_path = output_dir / f"{character_id}_mapping.json"

        if only_new and output_path.exists():
            continue

        # Load joint names based on file type
        if char_path.suffix == ".npz":
            joint_names = load_joint_names_npz(char_path)
        else:
            joint_names = load_joint_names_json(char_path)

        if joint_names is None:
            result.errors.append(f"Failed to load {char_path.name}")
            continue

        mapping = map_skeleton(character_id, joint_names)
        export_mapping_json(mapping, output_path)

        result.characters_processed += 1
        if mapping.auto_match_rate >= GOOD_MATCH_THRESHOLD:
            result.characters_good += 1
        else:
            result.characters_poor += 1

    logger.info(
        "UniRig batch complete: %d processed (%d good, %d poor), %d errors",
        result.characters_processed,
        result.characters_good,
        result.characters_poor,
        len(result.errors),
    )

    return result
