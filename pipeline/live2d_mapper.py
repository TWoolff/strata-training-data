"""Map Live2D ArtMesh fragment names to Strata region IDs.

Live2D models use artist-defined fragment names (often in Japanese romaji)
that need to be mapped to Strata's 20-label taxonomy (regions 0-19).

Mapping strategy:
    1. Iterate LIVE2D_FRAGMENT_PATTERNS in order (case-insensitive regex).
    2. First match wins — patterns are ordered specific → general.
    3. Unmapped fragments are returned separately for manual review.

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
    LIVE2D_FRAGMENT_PATTERNS,
    REGION_NAME_TO_ID,
    RegionId,
)

logger = logging.getLogger(__name__)

# Pre-compile all fragment patterns for performance
_COMPILED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pattern, re.IGNORECASE), region_name)
    for pattern, region_name in LIVE2D_FRAGMENT_PATTERNS
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FragmentMapping:
    """Mapping result for a single Live2D ArtMesh fragment."""

    fragment_name: str
    strata_label: str  # Region name or "UNMAPPED"
    strata_region_id: RegionId  # Region ID or -1 if unmapped
    confirmed: str  # "auto" or "pending"


@dataclass
class ModelMapping:
    """Mapping results for an entire Live2D model."""

    model_id: str
    mappings: list[FragmentMapping] = field(default_factory=list)

    @property
    def mapped_count(self) -> int:
        """Number of fragments successfully mapped."""
        return sum(1 for m in self.mappings if m.strata_region_id >= 0)

    @property
    def unmapped_count(self) -> int:
        """Number of fragments that could not be mapped."""
        return self.total_count - self.mapped_count

    @property
    def total_count(self) -> int:
        """Total number of fragments."""
        return len(self.mappings)

    @property
    def auto_rate(self) -> float:
        """Fraction of fragments that were auto-mapped (0.0–1.0)."""
        if not self.mappings:
            return 0.0
        return self.mapped_count / self.total_count


# ---------------------------------------------------------------------------
# Core mapping function
# ---------------------------------------------------------------------------


def map_fragment(fragment_name: str) -> tuple[str, RegionId]:
    """Map a single Live2D fragment name to a Strata region.

    Args:
        fragment_name: ArtMesh fragment name from the Live2D model.

    Returns:
        Tuple of (region_name, region_id). Returns ("UNMAPPED", -1) if
        no pattern matches.
    """
    for compiled_pattern, region_name in _COMPILED_PATTERNS:
        if compiled_pattern.search(fragment_name):
            region_id = REGION_NAME_TO_ID[region_name]
            return region_name, region_id
    return "UNMAPPED", -1


def map_model(
    model_id: str,
    fragment_names: list[str],
) -> ModelMapping:
    """Map all fragments of a Live2D model to Strata regions.

    Args:
        model_id: Unique identifier for the model.
        fragment_names: List of ArtMesh fragment names from the model.

    Returns:
        ModelMapping with results for each fragment.
    """
    result = ModelMapping(model_id=model_id)

    for name in fragment_names:
        region_name, region_id = map_fragment(name)
        confirmed = "auto" if region_id >= 0 else "pending"
        result.mappings.append(
            FragmentMapping(
                fragment_name=name,
                strata_label=region_name,
                strata_region_id=region_id,
                confirmed=confirmed,
            )
        )

    logger.info(
        "Live2D mapping for %s: %d/%d auto-mapped (%.0f%%), %d unmapped",
        model_id,
        result.mapped_count,
        result.total_count,
        result.auto_rate * 100,
        result.unmapped_count,
    )

    if result.unmapped_count > 0:
        unmapped = [m.fragment_name for m in result.mappings if m.strata_region_id < 0]
        logger.warning("Unmapped fragments in %s: %s", model_id, unmapped)

    return result


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

CSV_HEADER: list[str] = [
    "model_id",
    "fragment_name",
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
                    m.fragment_name,
                    m.strata_label,
                    m.strata_region_id,
                    m.confirmed,
                ])

    total_rows = sum(m.total_count for m in mappings)
    logger.info("Exported %d fragment mappings to %s", total_rows, output_path)


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
                FragmentMapping(
                    fragment_name=row["fragment_name"],
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
    """Group fragment names by their assigned Strata region.

    Args:
        mapping: A ModelMapping result.

    Returns:
        Dict of region_name → list of fragment names assigned to that region.
    """
    summary: dict[str, list[str]] = {}
    for m in mapping.mappings:
        summary.setdefault(m.strata_label, []).append(m.fragment_name)
    return summary
