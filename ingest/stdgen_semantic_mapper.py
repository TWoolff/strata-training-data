"""Map StdGEN 4-class semantic annotations to Strata's 20-class taxonomy.

StdGEN (CVPR 2025) provides 10,811 VRoid-derived anime characters annotated
with 4 coarse semantic classes:

- **hair** → head (region 1)
- **face** → head (region 1)
- **body** → per-vertex bone-weight refinement into 16 body sub-regions
- **clothes** → underlying body region (follows bone-based approach)

The ``body`` and ``clothes`` classes require per-vertex refinement using VRM
skeleton bone weights to split into Strata's fine-grained regions (chest,
hips, upper_arm_l, etc.).  This is analogous to what ``pipeline/bone_mapper.py``
does for Mixamo characters, but driven by StdGEN's semantic labels.

This module is pure Python (no Blender dependency) so it can be imported
outside Blender for testing and validation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.config import (
    REGION_NAMES,
    STDGEN_SEMANTIC_CLASSES,
    VRM_BONE_ALIASES,
    RegionId,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STDGEN_SOURCE = "stdgen"

# StdGEN's 4 semantic class labels.
VALID_CLASSES = frozenset(STDGEN_SEMANTIC_CLASSES.keys())

# Default region for vertices with no bone weights (fallback).
FALLBACK_REGION_ID: RegionId = 3  # chest — safe central default


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class VertexMapping:
    """Mapping result for a single vertex."""

    vertex_index: int
    stdgen_class: str
    strata_label: str
    strata_region_id: RegionId


@dataclass
class MeshMapping:
    """Mapping results for an entire mesh."""

    mesh_name: str
    vertex_mappings: list[VertexMapping] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        """Total number of vertices mapped."""
        return len(self.vertex_mappings)

    @property
    def region_distribution(self) -> dict[str, int]:
        """Count of vertices per Strata region."""
        dist: dict[str, int] = {}
        for vm in self.vertex_mappings:
            dist[vm.strata_label] = dist.get(vm.strata_label, 0) + 1
        return dist


@dataclass
class CharacterMapping:
    """Mapping results for an entire StdGEN character."""

    character_id: str
    mesh_mappings: list[MeshMapping] = field(default_factory=list)

    @property
    def total_vertices(self) -> int:
        """Total vertices across all meshes."""
        return sum(m.total_count for m in self.mesh_mappings)

    @property
    def region_distribution(self) -> dict[str, int]:
        """Aggregate region distribution across all meshes."""
        dist: dict[str, int] = {}
        for mm in self.mesh_mappings:
            for label, count in mm.region_distribution.items():
                dist[label] = dist.get(label, 0) + count
        return dist


# ---------------------------------------------------------------------------
# Bone-weight resolution
# ---------------------------------------------------------------------------


# Module-level lookup for performance.
_BONE_TO_REGION: dict[str, RegionId] = dict(VRM_BONE_ALIASES)


def resolve_region_from_weights(
    bone_weights: dict[str, float],
) -> tuple[str, RegionId]:
    """Determine Strata region from a vertex's bone weights.

    Picks the bone with the highest weight that maps to a known Strata
    region.  If no bone maps, returns the fallback region.

    Args:
        bone_weights: Dict of bone_name → weight (0.0–1.0).

    Returns:
        Tuple of (region_name, region_id).
    """
    if not bone_weights:
        return REGION_NAMES[FALLBACK_REGION_ID], FALLBACK_REGION_ID

    # Sort by weight descending, pick first bone with a known region.
    for bone_name, _weight in sorted(
        bone_weights.items(), key=lambda x: x[1], reverse=True
    ):
        region_id = _BONE_TO_REGION.get(bone_name)
        if region_id is not None:
            return REGION_NAMES[region_id], region_id

    return REGION_NAMES[FALLBACK_REGION_ID], FALLBACK_REGION_ID


# ---------------------------------------------------------------------------
# Core mapping functions
# ---------------------------------------------------------------------------


def map_vertex(
    stdgen_class: str,
    bone_weights: dict[str, float],
) -> tuple[str, RegionId]:
    """Map a single vertex from StdGEN class to Strata region.

    For ``hair`` and ``face``, returns head (region 1) directly.
    For ``body`` and ``clothes``, resolves via bone weights.

    Args:
        stdgen_class: StdGEN semantic class label.
        bone_weights: Per-vertex bone weights (bone_name → weight).

    Returns:
        Tuple of (region_name, region_id).

    Raises:
        ValueError: If stdgen_class is not a valid StdGEN class.
    """
    if stdgen_class not in VALID_CLASSES:
        raise ValueError(
            f"Unknown StdGEN class {stdgen_class!r}. "
            f"Valid classes: {sorted(VALID_CLASSES)}"
        )

    direct_id = STDGEN_SEMANTIC_CLASSES[stdgen_class]
    if direct_id is not None:
        return REGION_NAMES[direct_id], direct_id

    # body / clothes → resolve via bone weights
    return resolve_region_from_weights(bone_weights)


def map_mesh(
    mesh_name: str,
    vertex_classes: list[str],
    vertex_bone_weights: list[dict[str, float]],
) -> MeshMapping:
    """Map all vertices of a mesh from StdGEN classes to Strata regions.

    Args:
        mesh_name: Name of the mesh object.
        vertex_classes: StdGEN class label per vertex.
        vertex_bone_weights: Bone weights per vertex.

    Returns:
        MeshMapping with per-vertex results.
    """
    if len(vertex_classes) != len(vertex_bone_weights):
        raise ValueError(
            f"Mismatched lengths: {len(vertex_classes)} classes vs "
            f"{len(vertex_bone_weights)} bone weight entries"
        )

    result = MeshMapping(mesh_name=mesh_name)

    for i, (cls, weights) in enumerate(
        zip(vertex_classes, vertex_bone_weights, strict=True)
    ):
        label, region_id = map_vertex(cls, weights)
        result.vertex_mappings.append(
            VertexMapping(
                vertex_index=i,
                stdgen_class=cls,
                strata_label=label,
                strata_region_id=region_id,
            )
        )

    return result


def map_character(
    character_id: str,
    meshes: list[tuple[str, list[str], list[dict[str, float]]]],
) -> CharacterMapping:
    """Map all meshes of a StdGEN character to Strata regions.

    Args:
        character_id: Unique character identifier.
        meshes: List of (mesh_name, vertex_classes, vertex_bone_weights)
            tuples.

    Returns:
        CharacterMapping with per-mesh, per-vertex results.
    """
    result = CharacterMapping(character_id=character_id)

    for mesh_name, vertex_classes, vertex_bone_weights in meshes:
        mesh_mapping = map_mesh(mesh_name, vertex_classes, vertex_bone_weights)
        result.mesh_mappings.append(mesh_mapping)

    logger.info(
        "StdGEN mapping for %s: %d vertices across %d meshes",
        character_id,
        result.total_vertices,
        len(result.mesh_mappings),
    )

    return result


# ---------------------------------------------------------------------------
# Segmentation mask refinement
# ---------------------------------------------------------------------------


def refine_segmentation_mask(
    coarse_mask: np.ndarray,
    vertex_region_ids: np.ndarray,
    vertex_pixel_coords: np.ndarray,
    image_size: tuple[int, int] = (512, 512),
) -> np.ndarray:
    """Refine a StdGEN 4-class mask into Strata's 20-class mask.

    Takes the coarse StdGEN segmentation and replaces body/clothes pixels
    with fine-grained region IDs derived from bone-weight vertex mappings.

    Args:
        coarse_mask: HxW uint8 array with StdGEN class IDs (0-3).
        vertex_region_ids: Per-vertex Strata region IDs (N,) uint8.
        vertex_pixel_coords: Per-vertex 2D pixel coordinates (N, 2) int.
        image_size: (height, width) of the output mask.

    Returns:
        HxW uint8 array with Strata region IDs (0-19).
    """
    h, w = image_size
    refined = np.zeros((h, w), dtype=np.uint8)

    # Paint vertices onto the mask — later vertices overwrite earlier ones.
    # For production use, a proper rasterization approach (barycentric
    # interpolation over faces) would be more accurate.
    for vid in range(len(vertex_region_ids)):
        x, y = vertex_pixel_coords[vid]
        if 0 <= x < w and 0 <= y < h:
            refined[y, x] = vertex_region_ids[vid]

    return refined


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def export_mapping_json(
    mapping: CharacterMapping,
    output_path: Path,
) -> None:
    """Export character mapping to JSON.

    Args:
        mapping: CharacterMapping to export.
        output_path: Path for the output JSON file.
    """
    data: dict[str, Any] = {
        "character_id": mapping.character_id,
        "source": STDGEN_SOURCE,
        "total_vertices": mapping.total_vertices,
        "region_distribution": mapping.region_distribution,
        "meshes": [],
    }

    for mm in mapping.mesh_mappings:
        mesh_data: dict[str, Any] = {
            "mesh_name": mm.mesh_name,
            "vertex_count": mm.total_count,
            "region_distribution": mm.region_distribution,
        }
        data["meshes"].append(mesh_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    logger.info("Exported StdGEN mapping to %s", output_path)


def load_mapping_json(json_path: Path) -> dict[str, Any]:
    """Load a previously exported mapping JSON.

    Args:
        json_path: Path to the JSON file.

    Returns:
        Parsed mapping dict.
    """
    return json.loads(json_path.read_text(encoding="utf-8"))
