"""Dataset splitting: assign characters to train/val/test sets.

Splits by character (not by image) to prevent data leakage — all poses
and styles of one character go into the same split.  Supports
stratification by asset source and incremental updates.  Pure Python
(no Blender dependency).

Schema follows PRD §8.4 (Dataset Splits).
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from .config import SPLIT_RATIOS

logger = logging.getLogger(__name__)

SPLIT_SEED: int = 42


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_splits(
    output_dir: Path,
    *,
    seed: int = SPLIT_SEED,
) -> Path:
    """Generate ``splits.json`` at the dataset root.

    Discovers characters from ``sources/`` metadata files, groups them by
    asset source for stratified assignment, and writes a deterministic
    train/val/test split.  When ``splits.json`` already exists, new
    characters are assigned incrementally to maintain ratio balance.

    Args:
        output_dir: Root dataset directory (e.g. ``./output/segmentation/``).
        seed: Random seed for deterministic shuffling.

    Returns:
        Path to the written ``splits.json``.
    """
    char_sources = _discover_characters(output_dir)
    existing_splits = _load_existing_splits(output_dir)

    if existing_splits is not None:
        splits = _incremental_update(existing_splits, char_sources, seed=seed)
    else:
        splits = _full_split(char_sources, seed=seed)

    path = _write_splits(output_dir, splits)
    _log_summary(splits)
    return path


# ---------------------------------------------------------------------------
# Character discovery
# ---------------------------------------------------------------------------


def _discover_characters(output_dir: Path) -> dict[str, str]:
    """Scan ``sources/`` for character metadata and build ID → source map.

    Falls back to filename prefix heuristic when the ``source`` field is
    missing or empty.

    Returns:
        Dict mapping character ID to source name.
    """
    sources_dir = output_dir / "sources"
    char_sources: dict[str, str] = {}

    if not sources_dir.is_dir():
        logger.warning("No sources/ directory found in %s", output_dir)
        return char_sources

    for meta_path in sorted(sources_dir.glob("*.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            char_id = meta.get("id", meta_path.stem)
            source = meta.get("source", "") or _infer_source(char_id)
            char_sources[char_id] = source
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            logger.warning("Failed to read source metadata %s", meta_path)

    return char_sources


def _infer_source(char_id: str) -> str:
    """Fallback source inference from character ID prefix."""
    lower = char_id.lower()
    for prefix in ("mixamo", "quaternius", "kenney", "sketchfab"):
        if lower.startswith(prefix):
            return prefix
    return "unknown"


# ---------------------------------------------------------------------------
# Splitting logic
# ---------------------------------------------------------------------------


def _full_split(
    char_sources: dict[str, str],
    *,
    seed: int,
) -> dict[str, list[str]]:
    """Create a fresh stratified split from all characters.

    Groups characters by source, shuffles each group deterministically,
    and assigns proportional slices to each split.

    Returns:
        Dict with ``"train"``, ``"val"``, ``"test"`` keys mapping to
        sorted lists of character IDs.
    """
    by_source = _group_by_source(char_sources)
    splits: dict[str, list[str]] = {name: [] for name in SPLIT_RATIOS}

    rng = random.Random(seed)

    for source in sorted(by_source):
        ids = sorted(by_source[source])
        rng.shuffle(ids)
        _assign_proportional(ids, splits)

    # Sort within each split for deterministic output
    for name in splits:
        splits[name].sort()

    return splits


def _incremental_update(
    existing: dict[str, list[str]],
    char_sources: dict[str, str],
    *,
    seed: int,
) -> dict[str, list[str]]:
    """Add newly discovered characters to an existing split.

    Existing assignments are preserved.  New characters are assigned to
    whichever split is most under-represented relative to the target ratios.

    Returns:
        Updated splits dict.
    """
    assigned = {cid for ids in existing.values() for cid in ids}

    new_ids = sorted(cid for cid in char_sources if cid not in assigned)
    if not new_ids:
        logger.info("No new characters to assign — splits unchanged")
        return existing

    # Start from the existing split
    splits: dict[str, list[str]] = {name: list(ids) for name, ids in existing.items()}

    # Shuffle new IDs deterministically
    rng = random.Random(seed)
    rng.shuffle(new_ids)

    # Assign each new character to the most under-represented split
    for char_id in new_ids:
        target = _most_underrepresented_split(splits)
        splits[target].append(char_id)

    for name in splits:
        splits[name].sort()

    logger.info("Incrementally assigned %d new character(s)", len(new_ids))
    return splits


def _assign_proportional(
    ids: list[str],
    splits: dict[str, list[str]],
) -> None:
    """Assign a list of IDs proportionally to splits based on SPLIT_RATIOS.

    Modifies ``splits`` in place.  Uses cumulative ratio boundaries to
    determine cut points, ensuring the ±1 character tolerance is met.
    """
    n = len(ids)
    if n == 0:
        return

    split_names = list(SPLIT_RATIOS.keys())
    cumulative = 0.0
    start = 0

    for i, name in enumerate(split_names):
        cumulative += SPLIT_RATIOS[name]
        if i == len(split_names) - 1:
            # Last split gets the remainder
            end = n
        else:
            end = round(cumulative * n)
        splits[name].extend(ids[start:end])
        start = end


def _most_underrepresented_split(splits: dict[str, list[str]]) -> str:
    """Return the split name that is most under-represented vs target ratios."""
    total = sum(len(ids) for ids in splits.values())
    if total == 0:
        return "train"

    return max(
        SPLIT_RATIOS,
        key=lambda name: SPLIT_RATIOS[name] - len(splits[name]) / total,
    )


def _group_by_source(
    char_sources: dict[str, str],
) -> dict[str, list[str]]:
    """Group character IDs by their source."""
    by_source: dict[str, list[str]] = defaultdict(list)
    for char_id, source in char_sources.items():
        by_source[source].append(char_id)
    return dict(by_source)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_existing_splits(output_dir: Path) -> dict[str, list[str]] | None:
    """Load ``splits.json`` if it exists, or return None."""
    path = output_dir / "splits.json"
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # Validate structure
        if not all(k in data for k in ("train", "val", "test")):
            logger.warning("Existing splits.json has invalid structure — regenerating")
            return None
        return {k: list(data[k]) for k in ("train", "val", "test")}
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to read existing splits.json — regenerating")
        return None


def _write_splits(
    output_dir: Path,
    splits: dict[str, list[str]],
) -> Path:
    """Write ``splits.json`` to the dataset root."""
    path = output_dir / "splits.json"
    path.write_text(
        json.dumps(splits, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Saved splits.json to %s", path)
    return path


def _log_summary(splits: dict[str, list[str]]) -> None:
    """Log a one-line summary of the split sizes."""
    parts = [f"{name}={len(ids)}" for name, ids in splits.items()]
    total = sum(len(ids) for ids in splits.values())
    logger.info("Split %d characters: %s", total, ", ".join(parts))
