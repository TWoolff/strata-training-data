"""Generate unified train/val/test splits across all data sources.

Extends the per-source split logic in ``pipeline/splitter.py`` to handle
multiple output directories and produce a single cross-source split manifest
CSV.  Ensures no character appears in multiple splits (prevents data leakage)
and stratifies by source for balanced representation.

Pure Python (no Blender dependency).

Usage::

    python -m scripts.generate_splits [--output-dir ./output]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

from pipeline.config import SPLIT_RATIOS

logger = logging.getLogger(__name__)

SPLIT_SEED: int = 42


# ---------------------------------------------------------------------------
# Character discovery
# ---------------------------------------------------------------------------


def _discover_characters_from_source(source_dir: Path) -> dict[str, str]:
    """Discover characters from a single output source directory.

    Checks ``sources/*.json`` for character metadata.  Falls back to
    inferring character IDs from image filenames.

    Returns:
        Dict mapping character ID to source name.
    """
    char_sources: dict[str, str] = {}

    # Try sources/ metadata first (Blender pipeline format)
    sources_dir = source_dir / "sources"
    if sources_dir.is_dir():
        for meta_path in sorted(sources_dir.glob("*.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                char_id = meta.get("id", meta_path.stem)
                source = meta.get("source", "") or source_dir.name
                char_sources[char_id] = source
            except (json.JSONDecodeError, OSError):
                logger.warning("Failed to read %s", meta_path)

    # Fall back to image filenames if no sources/ metadata found
    if not char_sources:
        images_dir = source_dir / "images"
        if images_dir.is_dir():
            seen: set[str] = set()
            for img_path in sorted(images_dir.glob("*.png")):
                stem = img_path.stem
                pose_idx = stem.find("_pose_")
                if pose_idx != -1:
                    char_id = stem[:pose_idx]
                else:
                    char_id = stem
                if char_id not in seen:
                    seen.add(char_id)
                    char_sources[char_id] = _infer_source(char_id, source_dir.name)

    return char_sources


def _infer_source(char_id: str, fallback: str) -> str:
    """Infer source from character ID prefix."""
    lower = char_id.lower()
    for prefix in ("mixamo", "quaternius", "kenney", "sketchfab"):
        if lower.startswith(prefix):
            return prefix
    return fallback


def discover_all_characters(output_dir: Path) -> dict[str, str]:
    """Discover characters from all output sources.

    Scans ``output_dir`` and its immediate subdirectories for dataset
    artifacts.

    Args:
        output_dir: Root output directory.

    Returns:
        Dict mapping character ID to source name.  Duplicate IDs from
        different sources are disambiguated with a source prefix.
    """
    all_chars: dict[str, str] = {}

    # Collect from each source, tracking conflicts
    source_chars: list[tuple[Path, dict[str, str]]] = []

    # Check output_dir itself
    chars = _discover_characters_from_source(output_dir)
    if chars:
        source_chars.append((output_dir, chars))

    # Check subdirectories
    if output_dir.is_dir():
        for subdir in sorted(output_dir.iterdir()):
            if not subdir.is_dir():
                continue
            if (subdir / "images").is_dir() or (subdir / "sources").is_dir():
                chars = _discover_characters_from_source(subdir)
                if chars:
                    source_chars.append((subdir, chars))

    # Merge, disambiguating duplicates
    for _source_dir, chars in source_chars:
        for char_id, source in chars.items():
            if char_id in all_chars and all_chars[char_id] != source:
                # Disambiguate by prefixing with source
                new_id = f"{source}_{char_id}"
                logger.warning(
                    "Duplicate character ID %s from %s — renaming to %s",
                    char_id,
                    source,
                    new_id,
                )
                all_chars[new_id] = source
            else:
                all_chars[char_id] = source

    return all_chars


# ---------------------------------------------------------------------------
# Splitting logic
# ---------------------------------------------------------------------------


def _group_by_source(char_sources: dict[str, str]) -> dict[str, list[str]]:
    """Group character IDs by their source."""
    by_source: dict[str, list[str]] = defaultdict(list)
    for char_id, source in char_sources.items():
        by_source[source].append(char_id)
    return dict(by_source)


def _assign_proportional(
    ids: list[str],
    splits: dict[str, list[str]],
    ratios: dict[str, float],
) -> None:
    """Assign IDs proportionally to splits based on ratios.

    Modifies ``splits`` in place.
    """
    n = len(ids)
    if n == 0:
        return

    split_names = list(ratios.keys())
    cumulative = 0.0
    start = 0

    for i, name in enumerate(split_names):
        cumulative += ratios[name]
        if i == len(split_names) - 1:
            end = n
        else:
            end = round(cumulative * n)
        splits[name].extend(ids[start:end])
        start = end


def generate_splits(
    output_dir: Path,
    *,
    seed: int = SPLIT_SEED,
    ratios: dict[str, float] | None = None,
) -> dict[str, list[str]]:
    """Generate cross-source stratified train/val/test splits.

    Discovers characters from all output sources, groups them by source,
    and assigns proportional slices to each split.  Each source is
    individually shuffled and split to ensure balanced representation.

    Args:
        output_dir: Root output directory.
        seed: Random seed for deterministic shuffling.
        ratios: Split ratios (default: ``SPLIT_RATIOS`` from config).

    Returns:
        Dict with ``"train"``, ``"val"``, ``"test"`` keys mapping to
        sorted lists of character IDs.
    """
    if ratios is None:
        ratios = SPLIT_RATIOS

    char_sources = discover_all_characters(output_dir)
    if not char_sources:
        logger.warning("No characters found in %s", output_dir)
        return {name: [] for name in ratios}

    by_source = _group_by_source(char_sources)
    splits: dict[str, list[str]] = {name: [] for name in ratios}
    rng = random.Random(seed)

    for source in sorted(by_source):
        ids = sorted(by_source[source])
        rng.shuffle(ids)
        _assign_proportional(ids, splits, ratios)

    for name in splits:
        splits[name].sort()

    return splits


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_splits_csv(
    splits: dict[str, list[str]],
    char_sources: dict[str, str],
    path: Path,
) -> Path:
    """Write split manifest as CSV.

    Each row maps a character ID to its split and source.

    Args:
        splits: Split name → character ID list.
        char_sources: Character ID → source name.
        path: Output CSV file path.

    Returns:
        Path to the written CSV.
    """
    rows: list[dict[str, str]] = []
    for split_name, ids in sorted(splits.items()):
        for char_id in ids:
            rows.append(
                {
                    "character_id": char_id,
                    "split": split_name,
                    "source": char_sources.get(char_id, "unknown"),
                }
            )

    rows.sort(key=lambda r: (r["split"], r["source"], r["character_id"]))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["character_id", "split", "source"])
        writer.writeheader()
        writer.writerows(rows)

    return path


def write_splits_json(splits: dict[str, list[str]], path: Path) -> Path:
    """Write splits as JSON (compatible with ``pipeline/splitter.py`` format).

    Args:
        splits: Split name → character ID list.
        path: Output JSON file path.

    Returns:
        Path to the written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(splits, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def print_split_summary(splits: dict[str, list[str]], char_sources: dict[str, str]) -> None:
    """Print a summary of the split assignment."""
    total = sum(len(ids) for ids in splits.values())

    print("\n=== Cross-Source Split Summary ===\n")
    print(f"  Total characters: {total}")
    for name, ids in splits.items():
        pct = (len(ids) / total * 100) if total else 0
        print(f"  {name}: {len(ids)} ({pct:.1f}%)")

    # Per-source breakdown
    source_split_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for split_name, ids in splits.items():
        for char_id in ids:
            source = char_sources.get(char_id, "unknown")
            source_split_counts[source][split_name] += 1

    if source_split_counts:
        split_names = list(splits.keys())
        header = f"\n{'Source':<25}" + "".join(f" {s:>8}" for s in split_names)
        print(header)
        print("-" * (25 + 9 * len(split_names)))
        for source in sorted(source_split_counts):
            counts = source_split_counts[source]
            row = f"{source:<25}" + "".join(f" {counts.get(s, 0):>8}" for s in split_names)
            print(row)

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate unified cross-source train/val/test splits."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Root output directory (default: output)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Save split manifest as CSV (default: output_dir/splits.csv)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Save splits as JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SPLIT_SEED,
        help=f"Random seed (default: {SPLIT_SEED})",
    )
    parser.add_argument(
        "--ratios",
        type=str,
        default=None,
        help="Split ratios as 'train:val:test' (default: 80:10:10)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Parse ratios
    ratios = SPLIT_RATIOS
    if args.ratios:
        parts = args.ratios.split(":")
        if len(parts) != 3:
            print("Error: --ratios must be 'train:val:test' (e.g. '80:10:10')")
            return 1
        total = sum(float(p) for p in parts)
        ratios = {
            "train": float(parts[0]) / total,
            "val": float(parts[1]) / total,
            "test": float(parts[2]) / total,
        }

    char_sources = discover_all_characters(args.output_dir)
    splits = generate_splits(args.output_dir, seed=args.seed, ratios=ratios)
    print_split_summary(splits, char_sources)

    csv_path = args.csv or args.output_dir / "splits.csv"
    write_splits_csv(splits, char_sources, csv_path)
    print(f"Split manifest saved to {csv_path}")

    if args.json:
        write_splits_json(splits, args.json)
        print(f"Splits JSON saved to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
