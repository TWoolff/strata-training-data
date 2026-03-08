"""Dataset manifest generation: metadata, statistics, and quality report.

Scans the output directory after a full batch run to produce
``manifest.json`` — a single file that records what the dataset contains,
how it was generated, and any quality issues found.  Pure Python (no Blender
dependency).

Schema follows the specification in Issue #21 and PRD §8.1.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import NUM_REGIONS, PIPELINE_VERSION, REGION_NAMES

logger = logging.getLogger(__name__)

# Maximum number of mask files to sample for region distribution analysis.
_DISTRIBUTION_SAMPLE_SIZE: int = 100


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_manifest(
    output_dir: Path,
    *,
    results: list[Any],
    styles: list[str],
    resolution: int,
    poses_per_character: int,
) -> Path:
    """Generate ``manifest.json`` at the dataset root.

    Args:
        output_dir: Root dataset directory (e.g. ``./output/segmentation/``).
        results: List of ``CharacterResult`` dataclass instances from the
            batch processing loop.
        styles: Art style names used for this run.
        resolution: Render resolution in pixels.
        poses_per_character: Configured poses per character (0 = all).

    Returns:
        Path to the written ``manifest.json``.
    """
    file_counts = _count_files(output_dir)
    images_by_style = _count_images_by_style(output_dir)
    images_by_source = _count_images_by_source(output_dir)
    region_distribution = _compute_region_distribution(output_dir)
    quality = _collect_quality_info(results, output_dir)

    manifest: dict[str, Any] = {
        "version": "1.0",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pipeline_version": PIPELINE_VERSION,
        "config": {
            "resolution": resolution,
            "styles": styles,
            "poses_per_character": poses_per_character,
        },
        "statistics": {
            "total_characters": file_counts["sources"],
            "total_poses": file_counts["masks"],
            "total_images": file_counts["images"],
            "total_masks": file_counts["masks"],
            "total_joints": file_counts["joints"],
            "total_weights": file_counts["weights"],
            "images_by_style": images_by_style,
            "images_by_source": images_by_source,
            "region_distribution": region_distribution,
        },
        "quality": quality,
    }

    path = output_dir / "manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Saved manifest.json to %s", path)
    return path


# ---------------------------------------------------------------------------
# File counting
# ---------------------------------------------------------------------------


def _count_files(output_dir: Path) -> dict[str, int]:
    """Count files in each dataset subdirectory.

    Returns:
        Dict mapping subdirectory name to file count, e.g.
        ``{"images": 36000, "masks": 6000, ...}``.
    """
    counts: dict[str, int] = {}
    for subdir in ("images", "masks", "joints", "weights", "draw_order", "sources"):
        dir_path = output_dir / subdir
        if dir_path.is_dir():
            counts[subdir] = sum(1 for f in dir_path.iterdir() if f.is_file())
        else:
            counts[subdir] = 0
    return counts


# ---------------------------------------------------------------------------
# Style breakdown
# ---------------------------------------------------------------------------


def _count_images_by_style(output_dir: Path) -> dict[str, int]:
    """Count images per art style by parsing image filenames.

    Expects filenames like ``mixamo_001_pose_00_flat.png``.  The style name
    is the last ``_``-delimited segment before the extension.

    Returns:
        Dict mapping style name to image count.
    """
    style_counts: Counter[str] = Counter()
    images_dir = output_dir / "images"
    if not images_dir.is_dir():
        return {}

    for img_path in images_dir.glob("*.png"):
        # e.g. "mixamo_001_pose_00_flat" → last token is "flat"
        stem = img_path.stem
        parts = stem.rsplit("_", 1)
        if len(parts) == 2:
            style_counts[parts[1]] += 1

    return dict(sorted(style_counts.items()))


# ---------------------------------------------------------------------------
# Source breakdown
# ---------------------------------------------------------------------------


def _count_images_by_source(output_dir: Path) -> dict[str, int]:
    """Count images per asset source.

    Reads per-character source metadata from ``sources/`` to determine the
    source for each character.  Falls back to filename prefix heuristic if
    the metadata file is missing.

    Returns:
        Dict mapping source name to image count.
    """
    source_counts: Counter[str] = Counter()
    images_dir = output_dir / "images"
    sources_dir = output_dir / "sources"
    if not images_dir.is_dir():
        return {}

    # Build char_id → source lookup from source metadata files
    char_source: dict[str, str] = {}
    if sources_dir.is_dir():
        for meta_path in sources_dir.glob("*.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                char_id = meta.get("id", meta_path.stem)
                source = meta.get("source", "unknown")
                char_source[char_id] = source if source else "unknown"
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                logger.warning("Failed to read source metadata %s", meta_path)

    for img_path in images_dir.glob("*.png"):
        # Extract char_id: everything before "_pose_NN_style"
        stem = img_path.stem
        pose_idx = stem.find("_pose_")
        if pose_idx == -1:
            continue
        char_id = stem[:pose_idx]
        source = char_source.get(char_id, _infer_source_from_id(char_id))
        source_counts[source] += 1

    return dict(sorted(source_counts.items()))


def _infer_source_from_id(char_id: str) -> str:
    """Fallback source inference from character ID prefix."""
    lower = char_id.lower()
    for prefix in ("mixamo", "quaternius", "kenney", "sketchfab"):
        if lower.startswith(prefix):
            return prefix
    return "unknown"


# ---------------------------------------------------------------------------
# Region distribution
# ---------------------------------------------------------------------------


def _compute_region_distribution(
    output_dir: Path,
    sample_size: int = _DISTRIBUTION_SAMPLE_SIZE,
) -> dict[str, float]:
    """Compute average region pixel distribution across a sample of masks.

    Samples up to ``sample_size`` mask files, counts the fraction of
    non-background pixels belonging to each region, and returns the
    average across all sampled masks.

    Args:
        output_dir: Root dataset directory.
        sample_size: Max number of masks to sample.

    Returns:
        Dict mapping region name to average pixel fraction (0.0–1.0),
        excluding background.  Returns empty dict if no masks found.
    """
    masks_dir = output_dir / "masks"
    if not masks_dir.is_dir():
        return {}

    mask_paths = list(masks_dir.glob("*.png"))
    if not mask_paths:
        return {}

    # Sample a random subset for performance
    if len(mask_paths) > sample_size:
        mask_paths = random.sample(mask_paths, sample_size)

    # Accumulate pixel counts per region across all sampled masks
    region_pixel_totals = np.zeros(NUM_REGIONS, dtype=np.float64)
    total_foreground_pixels: float = 0.0

    for mask_path in mask_paths:
        try:
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        except (OSError, ValueError):
            logger.warning("Failed to read mask %s", mask_path)
            continue

        # Vectorized pixel counting per region
        counts = np.bincount(mask.ravel(), minlength=NUM_REGIONS)[:NUM_REGIONS]
        region_pixel_totals += counts.astype(np.float64)
        total_foreground_pixels += float(counts[1:].sum())

    if total_foreground_pixels == 0:
        return {}

    # Compute fractions (exclude background region 0)
    distribution: dict[str, float] = {}
    for region_id in range(1, NUM_REGIONS):
        name = REGION_NAMES.get(region_id, f"region_{region_id}")
        fraction = region_pixel_totals[region_id] / total_foreground_pixels
        distribution[name] = round(fraction, 4)

    return distribution


# ---------------------------------------------------------------------------
# Quality info
# ---------------------------------------------------------------------------


def _collect_quality_info(
    results: list[Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Collect quality information from processing results and source metadata.

    Args:
        results: List of ``CharacterResult`` instances.
        output_dir: Root dataset directory.

    Returns:
        Quality report dict with unmapped bone characters, failures, and warnings.
    """
    unmapped_bone_characters: list[str] = []
    failed_characters: list[str] = []
    warnings: list[str] = []

    # Collect from CharacterResult objects
    for r in results:
        if r.error:
            failed_characters.append(r.char_id)
        elif r.poses_failed > 0:
            warnings.append(f"{r.char_id}: {r.poses_failed} pose(s) failed")

    # Collect unmapped bones from source metadata
    sources_dir = output_dir / "sources"
    if sources_dir.is_dir():
        for meta_path in sorted(sources_dir.glob("*.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                unmapped = meta.get("unmapped_bones", [])
                if unmapped:
                    char_id = meta.get("id", meta_path.stem)
                    unmapped_bone_characters.append(char_id)
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                logger.warning("Failed to read source metadata %s", meta_path)

    return {
        "unmapped_bone_characters": sorted(unmapped_bone_characters),
        "failed_characters": sorted(failed_characters),
        "warnings": warnings,
    }
