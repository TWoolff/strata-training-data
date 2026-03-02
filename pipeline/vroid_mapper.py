"""Map VRoid material slot names to Strata region IDs.

VRoid models use material slots (Body, Face, Hair, Outfit_Upper, etc.)
that map partially to Strata's 20-label taxonomy (regions 0-19).

Material-level mapping is COARSE — e.g., the "Body" material covers the
entire body mesh.  Fine-grained per-vertex labeling should use bone
weights via VRM_BONE_ALIASES in bone_mapper.py.  This module provides:
    1. Fallback mapping when bone data is incomplete.
    2. Initial classification for diagnostics and quality reporting.
    3. L/R disambiguation for symmetric materials (shoes, gloves)
       based on vertex world-space X position.

Mapping strategy:
    1. Iterate VROID_MATERIAL_PATTERNS in order (case-insensitive regex).
    2. First match wins — patterns are ordered specific → general.
    3. Unmapped materials are returned separately for manual review.

This module is pure Python (no Blender dependency) so it can be imported
outside Blender for testing and validation.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import (
    REGION_NAME_TO_ID,
    VROID_MATERIAL_PATTERNS,
    RegionId,
)

logger = logging.getLogger(__name__)

# Pre-compile all material patterns for performance
_COMPILED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pattern, re.IGNORECASE), region_name)
    for pattern, region_name in VROID_MATERIAL_PATTERNS
]

# Symmetric region pairs: left region name → right region name
_LR_PAIRS: dict[str, str] = {
    "shoulder_l": "shoulder_r",
    "upper_arm_l": "upper_arm_r",
    "forearm_l": "forearm_r",
    "hand_l": "hand_r",
    "upper_leg_l": "upper_leg_r",
    "lower_leg_l": "lower_leg_r",
    "foot_l": "foot_r",
}
_RL_PAIRS: dict[str, str] = {v: k for k, v in _LR_PAIRS.items()}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MaterialMapping:
    """Mapping result for a single VRoid material slot."""

    material_name: str
    strata_label: str  # Region name or "UNMAPPED"
    strata_region_id: RegionId  # Region ID or -1 if unmapped
    confirmed: str  # "auto" or "pending"


@dataclass
class ModelMapping:
    """Mapping results for an entire VRoid model."""

    model_id: str
    mappings: list[MaterialMapping] = field(default_factory=list)

    @property
    def mapped_count(self) -> int:
        """Number of materials successfully mapped."""
        return sum(1 for m in self.mappings if m.strata_region_id >= 0)

    @property
    def unmapped_count(self) -> int:
        """Number of materials that could not be mapped."""
        return self.total_count - self.mapped_count

    @property
    def total_count(self) -> int:
        """Total number of materials."""
        return len(self.mappings)

    @property
    def auto_rate(self) -> float:
        """Fraction of materials that were auto-mapped (0.0-1.0)."""
        if not self.mappings:
            return 0.0
        return self.mapped_count / self.total_count


# ---------------------------------------------------------------------------
# Core mapping functions
# ---------------------------------------------------------------------------


def map_material(material_name: str) -> tuple[str, RegionId]:
    """Map a single VRoid material name to a Strata region.

    Args:
        material_name: Material slot name from the VRoid model.

    Returns:
        Tuple of (region_name, region_id). Returns ("UNMAPPED", -1) if
        no pattern matches.
    """
    for compiled_pattern, region_name in _COMPILED_PATTERNS:
        if compiled_pattern.search(material_name):
            region_id = REGION_NAME_TO_ID[region_name]
            return region_name, region_id
    return "UNMAPPED", -1


def disambiguate_lr(
    material_name: str,
    region_name: str,
    vertex_centroid_x: float,
) -> tuple[str, RegionId]:
    """Resolve L/R for symmetric materials using vertex world position.

    VRoid models use material names like "Shoe" without L/R suffix.
    This function assigns the correct side based on the world-space X
    coordinate of the material's vertex centroid.  Convention: negative
    X = left, positive X = right (Blender / VRM standard).

    Args:
        material_name: Material slot name (for logging).
        region_name: The initially mapped region name (e.g., "foot_l").
        vertex_centroid_x: Average X position of vertices using this
            material in world space.

    Returns:
        Tuple of (disambiguated_region_name, region_id).  If the region
        is not a symmetric pair, returns the original mapping unchanged.
    """
    # Check if this is a left-side label that might need flipping to right
    if region_name in _LR_PAIRS and vertex_centroid_x > 0:
        right_name = _LR_PAIRS[region_name]
        region_id = REGION_NAME_TO_ID[right_name]
        logger.debug(
            "L/R disambiguation for %r: X=%.3f → %s",
            material_name,
            vertex_centroid_x,
            right_name,
        )
        return right_name, region_id

    # Check if this is a right-side label that might need flipping to left
    if region_name in _RL_PAIRS and vertex_centroid_x < 0:
        left_name = _RL_PAIRS[region_name]
        region_id = REGION_NAME_TO_ID[left_name]
        logger.debug(
            "L/R disambiguation for %r: X=%.3f → %s",
            material_name,
            vertex_centroid_x,
            left_name,
        )
        return left_name, region_id

    # No change needed — already correct or not a symmetric region
    if region_name in REGION_NAME_TO_ID:
        return region_name, REGION_NAME_TO_ID[region_name]
    return region_name, -1


def map_model(
    model_id: str,
    material_names: list[str],
) -> ModelMapping:
    """Map all materials of a VRoid model to Strata regions.

    Args:
        model_id: Unique identifier for the model.
        material_names: List of material slot names from the model.

    Returns:
        ModelMapping with results for each material.
    """
    result = ModelMapping(model_id=model_id)

    for name in material_names:
        region_name, region_id = map_material(name)
        confirmed = "auto" if region_id >= 0 else "pending"
        result.mappings.append(
            MaterialMapping(
                material_name=name,
                strata_label=region_name,
                strata_region_id=region_id,
                confirmed=confirmed,
            )
        )

    logger.info(
        "VRoid mapping for %s: %d/%d auto-mapped (%.0f%%), %d unmapped",
        model_id,
        result.mapped_count,
        result.total_count,
        result.auto_rate * 100,
        result.unmapped_count,
    )

    if result.unmapped_count > 0:
        unmapped = [m.material_name for m in result.mappings if m.strata_region_id < 0]
        logger.warning("Unmapped materials in %s: %s", model_id, unmapped)

    return result


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

CSV_HEADER: list[str] = [
    "model_id",
    "material_name",
    "strata_label",
    "strata_region_id",
    "confirmed",
]


def export_csv(
    mappings: list[ModelMapping],
    output_path: Path,
    *,
    append: bool = False,
) -> None:
    """Write model mappings to a CSV file.

    Args:
        mappings: List of ModelMapping results to export.
        output_path: Path to the output CSV file.
        append: If True, append to existing file without header.
    """
    mode = "a" if append else "w"
    write_header = not append or not output_path.exists()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER)
        for model in mappings:
            for m in model.mappings:
                writer.writerow([
                    model.model_id,
                    m.material_name,
                    m.strata_label,
                    m.strata_region_id,
                    m.confirmed,
                ])

    total_rows = sum(m.total_count for m in mappings)
    logger.info("Exported %d material mappings to %s", total_rows, output_path)


def load_csv(csv_path: Path) -> list[ModelMapping]:
    """Load model mappings from a previously exported CSV.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        List of ModelMapping objects reconstructed from the CSV.
    """
    models: dict[str, ModelMapping] = {}

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_id = row["model_id"]
            if model_id not in models:
                models[model_id] = ModelMapping(model_id=model_id)
            models[model_id].mappings.append(
                MaterialMapping(
                    material_name=row["material_name"],
                    strata_label=row["strata_label"],
                    strata_region_id=int(row["strata_region_id"]),
                    confirmed=row["confirmed"],
                )
            )

    return list(models.values())


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def region_summary(mapping: ModelMapping) -> dict[str, list[str]]:
    """Group material names by their assigned Strata region.

    Args:
        mapping: A ModelMapping result.

    Returns:
        Dict of region_name -> list of material names assigned to that region.
    """
    summary: dict[str, list[str]] = {}
    for m in mapping.mappings:
        summary.setdefault(m.strata_label, []).append(m.material_name)
    return summary
