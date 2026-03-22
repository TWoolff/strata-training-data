"""Character-level dataset split loader.

Discovers characters across multiple dataset directories, assigns each to
train/val/test by character ID (preventing data leakage), and caches the
result as ``splits.json``.  Reads existing splits when available.

Pure Python (no Blender dependency).
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Character ID extraction patterns
# ---------------------------------------------------------------------------
# Each pattern captures the character portion of an example ID, stripping
# pose index, style, and angle suffixes.

_CHAR_ID_PATTERNS: list[re.Pattern[str]] = [
    # HumanRig posed: "humanrig_00000_catwalk_walk_frame_01_front" → "humanrig_00000"
    # Must be before _pose_ pattern (which would mis-capture on t_pose names)
    re.compile(r"^(humanrig_\d+)_"),
    # Mixamo / pipeline standard: "mixamo_001_pose_05_flat" → "mixamo_001"
    re.compile(r"^(.+?)_pose_\d+"),
    # FBAnimeHQ: "fbanimehq_0000_000005" → "fbanimehq_0000"
    re.compile(r"^(fbanimehq_\d+)_\d+$"),
    # StdGEN: "stdgen_0042_front" → "stdgen_0042"
    re.compile(r"^(stdgen_\d+)"),
    # AnimeRun: "animerun_clip01_000005" → "animerun_clip01"
    re.compile(r"^(animerun_[^_]+)_\d+$"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def character_id_from_example(example_id: str) -> str:
    """Extract the character ID from an example/filename ID.

    Strips pose index, style suffix, and frame number to get the base
    character identifier.

    Args:
        example_id: Example identifier (e.g. ``"mixamo_001_pose_05_flat"``).

    Returns:
        Character ID (e.g. ``"mixamo_001"``).  Returns the full
        ``example_id`` unchanged if no pattern matches (assumes one
        image per character, e.g. NOVA-Human).
    """
    for pattern in _CHAR_ID_PATTERNS:
        m = pattern.match(example_id)
        if m:
            return m.group(1)
    return example_id


def load_or_generate_splits(
    dataset_dirs: list[Path],
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    train_only_dirs: list[Path] | None = None,
) -> dict[str, list[str]]:
    """Load existing splits or generate new character-level splits.

    Scans one or more dataset directories for examples, extracts character
    IDs, and assigns each character to exactly one split.  If any directory
    contains a ``splits.json``, those assignments are loaded and merged
    (existing assignments are preserved).

    Args:
        dataset_dirs: List of dataset root directories to scan.
        seed: Random seed for deterministic shuffling.
        ratios: ``(train, val, test)`` ratios summing to 1.0.
        train_only_dirs: Directories whose characters are always assigned
            to train (never val/test).  Prevents new datasets from
            reshuffling the validation set.

    Returns:
        Dict with ``"train"``, ``"val"``, ``"test"`` keys mapping to
        sorted lists of character IDs.
    """
    # Load any existing splits
    existing = _load_existing_splits(dataset_dirs)

    # Discover character IDs — split-eligible dirs only
    train_only_set = set(train_only_dirs or [])
    split_dirs = [d for d in dataset_dirs if d not in train_only_set]
    all_char_ids = _discover_characters(split_dirs)

    # Discover train-only character IDs separately
    train_only_char_ids: set[str] = set()
    if train_only_set:
        train_only_char_ids = _discover_characters(list(train_only_set))
        # Remove any overlap (train-only chars that also appear in split dirs)
        train_only_char_ids -= all_char_ids

    if not all_char_ids:
        logger.warning("No characters found in dataset directories")
        return {"train": [], "val": [], "test": []}

    # Determine which characters are already assigned
    assigned = set()
    if existing:
        for ids in existing.values():
            assigned.update(ids)

    new_ids = sorted(cid for cid in all_char_ids if cid not in assigned)

    if existing and not new_ids:
        logger.info("All %d characters already assigned — splits unchanged", len(assigned))
        return {k: sorted(v) for k, v in existing.items()}

    # Start from existing or empty splits
    splits: dict[str, list[str]] = (
        {k: list(v) for k, v in existing.items()}
        if existing
        else {"train": [], "val": [], "test": []}
    )

    # Assign new characters
    if new_ids:
        _assign_new_characters(new_ids, splits, ratios=ratios, seed=seed)
        logger.info("Assigned %d new character(s) to splits", len(new_ids))

    # Add train-only characters (never in val/test)
    if train_only_char_ids:
        assigned_all = set()
        for ids in splits.values():
            assigned_all.update(ids)
        new_train_only = sorted(cid for cid in train_only_char_ids if cid not in assigned_all)
        if new_train_only:
            splits["train"].extend(new_train_only)
            logger.info("Added %d train-only character(s)", len(new_train_only))

    # Sort for deterministic output
    for key in splits:
        splits[key].sort()

    return splits


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _discover_characters(dataset_dirs: list[Path]) -> set[str]:
    """Scan dataset directories for character IDs.

    Looks for images in ``images/`` subdirectories (flat layout) and
    per-example subdirectories containing ``image.png`` (per-example
    layout).  Also checks ``sources/`` and ``masks/`` as fallbacks.
    """
    char_ids: set[str] = set()

    for dataset_dir in dataset_dirs:
        if not dataset_dir.is_dir():
            logger.warning("Dataset directory not found, skipping: %s", dataset_dir)
            continue

        # Primary: scan images/ directory (flat layout)
        images_dir = dataset_dir / "images"
        if images_dir.is_dir():
            for img_path in images_dir.glob("*.png"):
                example_id = img_path.stem
                char_ids.add(character_id_from_example(example_id))

        # Per-example layout: {example_id}/image.png or weights.json
        for child in dataset_dir.iterdir():
            if not child.is_dir():
                continue
            # Direct: {id}/image.png or {id}/weights.json
            if (child / "image.png").exists() or (child / "weights.json").exists():
                char_ids.add(character_id_from_example(child.name))
                continue
            # Nested view: {id}/{view}/image.png or {id}/{view}/weights.json
            for view_dir in child.iterdir():
                if view_dir.is_dir() and (
                    (view_dir / "image.png").exists()
                    or (view_dir / "weights.json").exists()
                ):
                    char_ids.add(character_id_from_example(child.name))
                    break

        # Fallback: scan sources/ metadata
        sources_dir = dataset_dir / "sources"
        if sources_dir.is_dir():
            for meta_path in sources_dir.glob("*.json"):
                char_ids.add(meta_path.stem)

        # Fallback: scan masks/ directory
        masks_dir = dataset_dir / "masks"
        if masks_dir.is_dir():
            for mask_path in masks_dir.glob("*.png"):
                example_id = mask_path.stem
                char_ids.add(character_id_from_example(example_id))

    return char_ids


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def _assign_new_characters(
    new_ids: list[str],
    splits: dict[str, list[str]],
    *,
    ratios: tuple[float, float, float],
    seed: int,
) -> None:
    """Assign new character IDs to splits proportionally.

    If splits are empty (fresh generation), uses proportional slicing.
    If splits already have characters (incremental), assigns each new
    character to the most under-represented split.

    Modifies ``splits`` in place.
    """
    rng = random.Random(seed)
    shuffled = list(new_ids)
    rng.shuffle(shuffled)

    total_existing = sum(len(ids) for ids in splits.values())

    if total_existing == 0:
        # Fresh split — proportional assignment
        split_names = ["train", "val", "test"]
        n = len(shuffled)
        cumulative = 0.0
        start = 0
        for i, (name, ratio) in enumerate(zip(split_names, ratios, strict=True)):
            cumulative += ratio
            end = n if i == len(split_names) - 1 else round(cumulative * n)
            splits[name].extend(shuffled[start:end])
            start = end
    else:
        # Incremental — assign to most under-represented
        for char_id in shuffled:
            target = _most_underrepresented(splits, ratios)
            splits[target].append(char_id)


def _most_underrepresented(
    splits: dict[str, list[str]],
    ratios: tuple[float, float, float],
) -> str:
    """Return the split name most under-represented vs target ratios."""
    total = sum(len(ids) for ids in splits.values())
    if total == 0:
        return "train"

    ratio_map = dict(zip(["train", "val", "test"], ratios, strict=True))
    return max(
        ratio_map,
        key=lambda name: ratio_map[name] - len(splits[name]) / total,
    )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _load_existing_splits(dataset_dirs: list[Path]) -> dict[str, list[str]] | None:
    """Load and merge ``splits.json`` from dataset directories.

    Returns:
        Merged splits dict, or None if no splits.json files found.
    """
    merged: dict[str, list[str]] | None = None

    for dataset_dir in dataset_dirs:
        path = dataset_dir / "splits.json"
        if not path.exists():
            continue

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not all(k in data for k in ("train", "val", "test")):
                logger.warning("Invalid splits.json structure in %s — skipping", path)
                continue

            if merged is None:
                merged = {"train": [], "val": [], "test": []}

            # Merge, checking for conflicts
            existing_all = {cid for ids in merged.values() for cid in ids}
            for split_name in ("train", "val", "test"):
                for cid in data[split_name]:
                    if cid in existing_all:
                        logger.warning(
                            "Character %s already assigned — skipping duplicate from %s",
                            cid,
                            path,
                        )
                        continue
                    merged[split_name].append(cid)
                    existing_all.add(cid)

        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to read %s", path)

    return merged
