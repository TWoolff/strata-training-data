"""Compute cross-source dataset statistics.

Scans ``output/`` for all data sources and produces a summary of images,
masks, joints, region distribution, style breakdown, and angle breakdown.

Pure Python (no Blender dependency).

Usage::

    python -m scripts.compute_stats [--output-dir ./output]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pipeline.config import CAMERA_ANGLES, NUM_REGIONS, REGION_NAMES

logger = logging.getLogger(__name__)

# Maximum masks to sample for region distribution.
_DISTRIBUTION_SAMPLE_SIZE: int = 200


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def _discover_output_sources(output_dir: Path) -> list[Path]:
    """Find all output subdirectories that contain dataset artifacts.

    Looks for directories under ``output_dir`` that contain ``images/`` or
    ``masks/`` subdirectories (Blender pipeline format), or directories
    that directly contain PNG files (ingest adapter format).

    Returns:
        List of source root directories.
    """
    sources: list[Path] = []

    if not output_dir.is_dir():
        return sources

    # Check output_dir itself (Blender pipeline writes directly here)
    if (output_dir / "images").is_dir() or (output_dir / "masks").is_dir():
        sources.append(output_dir)

    # Check immediate subdirectories
    for subdir in sorted(output_dir.iterdir()):
        if not subdir.is_dir() or subdir == output_dir:
            continue
        if (subdir / "images").is_dir() or (subdir / "masks").is_dir():
            sources.append(subdir)

    return sources


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


def _count_files_in_source(source_dir: Path) -> dict[str, int]:
    """Count files in standard dataset subdirectories.

    Args:
        source_dir: A single source root directory.

    Returns:
        Dict mapping category name to file count.
    """
    counts: dict[str, int] = {}
    for subdir_name in ("images", "masks", "joints", "weights", "draw_order", "sources"):
        subdir = source_dir / subdir_name
        if subdir.is_dir():
            counts[subdir_name] = sum(1 for f in subdir.iterdir() if f.is_file())
        else:
            counts[subdir_name] = 0
    return counts


def count_all_files(output_dir: Path) -> dict[str, dict[str, int]]:
    """Count files per source across the output directory.

    Args:
        output_dir: Root output directory.

    Returns:
        Dict mapping source name to per-category file counts.
    """
    result: dict[str, dict[str, int]] = {}
    for source_dir in _discover_output_sources(output_dir):
        name = source_dir.name if source_dir != output_dir else "root"
        result[name] = _count_files_in_source(source_dir)
    return result


# ---------------------------------------------------------------------------
# Style and angle distribution
# ---------------------------------------------------------------------------


def count_images_by_style(output_dir: Path) -> dict[str, int]:
    """Count images per art style across all sources.

    Parses filenames like ``mixamo_001_pose_00_flat.png`` — the style is
    the last ``_``-delimited segment before the extension.

    Returns:
        Dict mapping style name to total image count.
    """
    style_counts: Counter[str] = Counter()

    for source_dir in _discover_output_sources(output_dir):
        images_dir = source_dir / "images"
        if not images_dir.is_dir():
            continue
        for img_path in images_dir.glob("*.png"):
            parts = img_path.stem.rsplit("_", 1)
            if len(parts) == 2:
                style_counts[parts[1]] += 1

    return dict(sorted(style_counts.items()))


def count_images_by_angle(output_dir: Path) -> dict[str, int]:
    """Count images per camera angle across all sources.

    Looks for angle names in filenames (e.g. ``_front.png``, ``_side_flat.png``).
    Images without an angle token are counted as ``front`` (the default).

    Returns:
        Dict mapping angle name to total image count.
    """
    angle_names = set(CAMERA_ANGLES.keys())
    angle_counts: Counter[str] = Counter()

    for source_dir in _discover_output_sources(output_dir):
        images_dir = source_dir / "images"
        if not images_dir.is_dir():
            continue
        for img_path in images_dir.glob("*.png"):
            stem = img_path.stem
            found_angle = None
            for angle_name in angle_names:
                if f"_{angle_name}_" in stem or stem.endswith(f"_{angle_name}"):
                    found_angle = angle_name
                    break
            angle_counts[found_angle or "front"] += 1

    return dict(sorted(angle_counts.items()))


def count_images_by_source(output_dir: Path) -> dict[str, int]:
    """Count total images per data source.

    Uses source metadata from ``sources/*.json`` when available, falling
    back to filename prefix heuristic.

    Returns:
        Dict mapping source name to image count.
    """
    source_counts: Counter[str] = Counter()

    for source_root in _discover_output_sources(output_dir):
        images_dir = source_root / "images"
        sources_dir = source_root / "sources"
        if not images_dir.is_dir():
            continue

        # Build char_id → source lookup
        char_source: dict[str, str] = {}
        if sources_dir.is_dir():
            for meta_path in sources_dir.glob("*.json"):
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    char_id = meta.get("id", meta_path.stem)
                    source = meta.get("source", "") or source_root.name
                    char_source[char_id] = source
                except (json.JSONDecodeError, OSError):
                    pass

        for img_path in images_dir.glob("*.png"):
            stem = img_path.stem
            pose_idx = stem.find("_pose_")
            if pose_idx != -1:
                char_id = stem[:pose_idx]
                source_name = char_source.get(char_id, _infer_source(char_id))
            else:
                source_name = source_root.name
            source_counts[source_name] += 1

    return dict(sorted(source_counts.items()))


def _infer_source(char_id: str) -> str:
    """Fallback source inference from character ID prefix."""
    lower = char_id.lower()
    for prefix in ("mixamo", "quaternius", "kenney", "sketchfab"):
        if lower.startswith(prefix):
            return prefix
    return "unknown"


# ---------------------------------------------------------------------------
# Region distribution
# ---------------------------------------------------------------------------


def compute_region_distribution(
    output_dir: Path,
    sample_size: int = _DISTRIBUTION_SAMPLE_SIZE,
) -> dict[str, float]:
    """Compute average region pixel distribution across all mask sources.

    Samples up to ``sample_size`` masks total, computes the fraction of
    non-background pixels per region, and averages across samples.

    Args:
        output_dir: Root output directory.
        sample_size: Max number of masks to sample.

    Returns:
        Dict mapping region name to average pixel fraction (0.0–1.0),
        excluding background.
    """
    all_mask_paths: list[Path] = []

    for source_dir in _discover_output_sources(output_dir):
        masks_dir = source_dir / "masks"
        if masks_dir.is_dir():
            all_mask_paths.extend(masks_dir.glob("*.png"))

    if not all_mask_paths:
        return {}

    if len(all_mask_paths) > sample_size:
        all_mask_paths = random.sample(all_mask_paths, sample_size)

    region_totals = np.zeros(NUM_REGIONS, dtype=np.float64)
    total_fg: float = 0.0

    for mask_path in all_mask_paths:
        try:
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        except (OSError, ValueError):
            logger.warning("Failed to read mask %s", mask_path)
            continue

        counts = np.bincount(mask.ravel(), minlength=NUM_REGIONS)[:NUM_REGIONS]
        region_totals += counts.astype(np.float64)
        total_fg += float(counts[1:].sum())

    if total_fg == 0:
        return {}

    distribution: dict[str, float] = {}
    for region_id in range(1, NUM_REGIONS):
        name = REGION_NAMES.get(region_id, f"region_{region_id}")
        distribution[name] = round(region_totals[region_id] / total_fg, 4)

    return distribution


def compute_coverage_report(distribution: dict[str, float]) -> dict[str, list[str]]:
    """Identify under-represented and missing regions.

    Args:
        distribution: Region name → average fraction mapping.

    Returns:
        Dict with ``under_represented`` (fraction < 1%) and ``missing``
        (fraction == 0) region lists.
    """
    under: list[str] = []
    missing: list[str] = []

    for region_id in range(1, NUM_REGIONS):
        name = REGION_NAMES.get(region_id, f"region_{region_id}")
        fraction = distribution.get(name, 0.0)
        if fraction == 0.0:
            missing.append(name)
        elif fraction < 0.01:
            under.append(name)

    return {"under_represented": under, "missing": missing}


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------


def compute_stats(output_dir: Path) -> dict[str, Any]:
    """Compute full cross-source statistics.

    Args:
        output_dir: Root output directory.

    Returns:
        Statistics dict suitable for JSON serialization.
    """
    file_counts = count_all_files(output_dir)

    # Aggregate totals across sources
    totals = {
        key: sum(c.get(key, 0) for c in file_counts.values())
        for key in ("images", "masks", "joints", "weights")
    }

    by_style = count_images_by_style(output_dir)
    by_angle = count_images_by_angle(output_dir)
    by_source = count_images_by_source(output_dir)
    region_dist = compute_region_distribution(output_dir)
    coverage = compute_coverage_report(region_dist)

    return {
        "totals": totals,
        "per_source": file_counts,
        "images_by_style": by_style,
        "images_by_angle": by_angle,
        "images_by_source": by_source,
        "region_distribution": region_dist,
        "coverage": coverage,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_summary(stats: dict[str, Any]) -> None:
    """Print a human-readable summary table to stdout."""
    totals = stats["totals"]

    print("\n=== Cross-Source Dataset Statistics ===\n")

    # Totals
    print(f"  Total images:  {totals['images']:>8}")
    print(f"  Total masks:   {totals['masks']:>8}")
    print(f"  Total joints:  {totals['joints']:>8}")
    print(f"  Total weights: {totals['weights']:>8}")

    # Per-source breakdown
    per_source = stats["per_source"]
    if per_source:
        print(f"\n{'Source':<25} {'Images':>8} {'Masks':>8} {'Joints':>8}")
        print("-" * 55)
        for source, counts in sorted(per_source.items()):
            print(
                f"{source:<25} {counts.get('images', 0):>8} "
                f"{counts.get('masks', 0):>8} {counts.get('joints', 0):>8}"
            )

    # Style distribution
    by_style = stats.get("images_by_style", {})
    if by_style:
        print(f"\n{'Style':<20} {'Count':>8}")
        print("-" * 30)
        for style, count in sorted(by_style.items()):
            print(f"{style:<20} {count:>8}")

    # Angle distribution
    by_angle = stats.get("images_by_angle", {})
    if by_angle:
        print(f"\n{'Angle':<20} {'Count':>8}")
        print("-" * 30)
        for angle, count in sorted(by_angle.items()):
            print(f"{angle:<20} {count:>8}")

    # Region distribution
    region_dist = stats.get("region_distribution", {})
    if region_dist:
        print(f"\n{'Region':<20} {'Fraction':>10}")
        print("-" * 32)
        for name, frac in sorted(region_dist.items(), key=lambda x: -x[1]):
            print(f"{name:<20} {frac:>10.4f}")

    # Coverage warnings
    coverage = stats.get("coverage", {})
    missing = coverage.get("missing", [])
    under = coverage.get("under_represented", [])
    if missing:
        print(f"\n  Missing regions (0%): {', '.join(missing)}")
    if under:
        print(f"  Under-represented (<1%): {', '.join(under)}")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Compute cross-source dataset statistics.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Root output directory (default: output)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Save statistics as JSON to this path",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    stats = compute_stats(args.output_dir)
    print_summary(stats)

    if args.json:
        args.json.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        print(f"JSON report saved to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
