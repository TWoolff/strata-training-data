"""Merge multiple dataset sources into a unified training dataset.

Combines outputs from the 3D synthetic pipeline, Spine parser, Live2D
renderer, manual annotations, and ingest adapters into a single dataset
with consistent formatting, updated splits, and manifest.  Pure Python
(no Blender dependency).

See Issue #28 and PRD §10.5.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import PIPELINE_VERSION, RENDER_RESOLUTION
from .exporter import _SUBDIRS, save_class_map
from .manifest import (
    _compute_region_distribution,
    _count_files,
    _count_images_by_source,
    _count_images_by_style,
)
from .validator import check_resolution

logger = logging.getLogger(__name__)

# Subdirectories that contain per-pose files (keyed by pose key).
_POSE_SUBDIRS: list[str] = ["images", "masks", "joints", "weights", "draw_order", "contours"]

# Subdirectories that contain per-character files.
_CHAR_SUBDIRS: list[str] = ["sources", "measurements"]

# Regex to extract char_id from pose-keyed filenames.
_POSE_FILE_PATTERN = re.compile(r"^(.+)_pose_(\d{2}.*)$")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class MergeReport:
    """Summary of a dataset merge operation."""

    sources_processed: int = 0
    characters_merged: int = 0
    characters_skipped: int = 0
    characters_renamed: int = 0
    files_copied: int = 0
    files_linked: int = 0
    files_skipped: int = 0
    validation_failures: int = 0
    warnings: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources_processed": self.sources_processed,
            "characters_merged": self.characters_merged,
            "characters_skipped": self.characters_skipped,
            "characters_renamed": self.characters_renamed,
            "files_copied": self.files_copied,
            "files_linked": self.files_linked,
            "files_skipped": self.files_skipped,
            "validation_failures": self.validation_failures,
            "warnings": self.warnings,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


# ---------------------------------------------------------------------------
# Character discovery
# ---------------------------------------------------------------------------


def _discover_characters(source_dir: Path) -> dict[str, dict[str, Any]]:
    """Discover characters in a source directory.

    Reads per-character metadata from ``sources/*.json``.  Falls back to
    inferring character IDs from image filenames when no metadata exists.

    Returns:
        Dict mapping character ID to metadata dict.
    """
    characters: dict[str, dict[str, Any]] = {}

    sources_dir = source_dir / "sources"
    if sources_dir.is_dir():
        for meta_path in sorted(sources_dir.glob("*.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                char_id = meta.get("id", meta_path.stem)
                characters[char_id] = meta
            except (json.JSONDecodeError, OSError):
                logger.warning("Failed to read source metadata %s", meta_path)

    if not characters:
        images_dir = source_dir / "images"
        if images_dir.is_dir():
            seen: set[str] = set()
            for img_path in sorted(images_dir.glob("*.png")):
                stem = img_path.stem
                pose_idx = stem.find("_pose_")
                char_id = stem[:pose_idx] if pose_idx != -1 else stem
                if char_id not in seen:
                    seen.add(char_id)
                    source = _infer_source(char_id, source_dir.name)
                    characters[char_id] = {"id": char_id, "source": source}

    return characters


def _infer_source(char_id: str, fallback: str) -> str:
    """Infer source from character ID prefix."""
    lower = char_id.lower()
    for prefix in ("mixamo", "quaternius", "kenney", "sketchfab", "spine", "live2d", "vroid"):
        if lower.startswith(prefix):
            return prefix
    return fallback


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------


def _collect_character_files(
    source_dir: Path,
    char_id: str,
) -> dict[str, list[Path]]:
    """Collect all files belonging to a character in a source directory.

    Returns:
        Dict mapping subdirectory name to list of file paths.
    """
    files: dict[str, list[Path]] = {}

    for subdir in _POSE_SUBDIRS:
        subdir_path = source_dir / subdir
        if not subdir_path.is_dir():
            continue
        # Match files starting with char_id_pose_ or char_id_ for contours
        matched = []
        for f in sorted(subdir_path.iterdir()):
            if not f.is_file():
                continue
            stem = f.stem
            if stem.startswith(f"{char_id}_pose_") or stem.startswith(f"{char_id}_"):
                # Verify it's actually this character, not a prefix match
                # e.g., "mixamo_01" should not match "mixamo_010"
                rest = stem[len(char_id) :]
                if rest.startswith("_pose_") or rest == "":
                    matched.append(f)
        if matched:
            files[subdir] = matched

    for subdir in _CHAR_SUBDIRS:
        subdir_path = source_dir / subdir
        if not subdir_path.is_dir():
            continue
        char_file = subdir_path / f"{char_id}.json"
        if char_file.is_file():
            files[subdir] = [char_file]

    return files


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_character(
    source_dir: Path,
    char_id: str,
    char_files: dict[str, list[Path]],
    resolution: int,
) -> list[str]:
    """Validate a character's files before merging.

    Returns:
        List of validation failure messages.  Empty list means all checks passed.
    """
    failures: list[str] = []

    for img_path in char_files.get("images", []):
        try:
            passed, detail = check_resolution(img_path, resolution)
            if not passed:
                failures.append(f"{img_path.name}: {detail}")
        except Exception as exc:
            failures.append(f"{img_path.name}: error reading - {exc}")

    for mask_path in char_files.get("masks", []):
        try:
            passed, detail = check_resolution(mask_path, resolution)
            if not passed:
                failures.append(f"{mask_path.name}: {detail}")
        except Exception as exc:
            failures.append(f"{mask_path.name}: error reading - {exc}")

    return failures


# ---------------------------------------------------------------------------
# File transfer
# ---------------------------------------------------------------------------


def _transfer_file(
    src: Path,
    dst: Path,
    *,
    mode: str = "copy",
) -> str:
    """Copy or symlink a file from src to dst.

    Args:
        src: Source file path.
        dst: Destination file path.
        mode: ``"copy"`` or ``"link"`` (symlink).

    Returns:
        The transfer mode used (``"copied"`` or ``"linked"``).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    if mode == "link":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
        return "linked"
    else:
        shutil.copy2(src, dst)
        return "copied"


def _rename_file(filename: str, old_char_id: str, new_char_id: str) -> str:
    """Rename a file by replacing the character ID prefix."""
    if filename.startswith(old_char_id):
        return new_char_id + filename[len(old_char_id) :]
    return filename


def _transfer_character_files(
    char_files: dict[str, list[Path]],
    output_dir: Path,
    char_id: str,
    new_char_id: str | None,
    *,
    mode: str = "copy",
) -> tuple[int, int]:
    """Transfer all files for a character to the output directory.

    Args:
        char_files: Files grouped by subdirectory.
        output_dir: Merged dataset root.
        char_id: Original character ID.
        new_char_id: Renamed character ID (or None if unchanged).
        mode: ``"copy"`` or ``"link"``.

    Returns:
        Tuple of (files_transferred, files_skipped).
    """
    transferred = 0
    skipped = 0

    for subdir, paths in char_files.items():
        for src_path in paths:
            filename = src_path.name
            if new_char_id:
                filename = _rename_file(filename, char_id, new_char_id)

            dst_path = output_dir / subdir / filename
            if dst_path.exists():
                logger.debug("Skipping existing file %s", dst_path)
                skipped += 1
                continue

            _transfer_file(src_path, dst_path, mode=mode)
            transferred += 1

    # Update source metadata if character was renamed
    if new_char_id:
        meta_path = output_dir / "sources" / f"{new_char_id}.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["id"] = new_char_id
                meta["original_id"] = char_id
                meta_path.write_text(
                    json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            except (json.JSONDecodeError, OSError):
                logger.warning("Failed to update metadata for renamed character %s", new_char_id)

    return transferred, skipped


# ---------------------------------------------------------------------------
# Manifest generation (scan-based, no CharacterResult required)
# ---------------------------------------------------------------------------


def _generate_merge_manifest(
    output_dir: Path,
    report: MergeReport,
    source_dirs: list[Path],
) -> Path:
    """Generate ``manifest.json`` for the merged dataset.

    Unlike ``manifest.generate_manifest()``, this does not require
    ``CharacterResult`` objects — it scans the output directory directly.

    Args:
        output_dir: Merged dataset root.
        report: Merge report with summary statistics.
        source_dirs: List of source directories that were merged.

    Returns:
        Path to the written ``manifest.json``.
    """
    from datetime import datetime, timezone

    file_counts = _count_files(output_dir)
    images_by_style = _count_images_by_style(output_dir)
    images_by_source = _count_images_by_source(output_dir)
    region_distribution = _compute_region_distribution(output_dir)

    manifest: dict[str, Any] = {
        "version": "1.0",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pipeline_version": PIPELINE_VERSION,
        "merge": {
            "source_dirs": [str(d) for d in source_dirs],
            "characters_merged": report.characters_merged,
            "characters_skipped": report.characters_skipped,
            "characters_renamed": report.characters_renamed,
        },
        "statistics": {
            "total_characters": file_counts.get("sources", 0),
            "total_poses": file_counts.get("masks", 0),
            "total_images": file_counts.get("images", 0),
            "total_masks": file_counts.get("masks", 0),
            "total_joints": file_counts.get("joints", 0),
            "total_weights": file_counts.get("weights", 0),
            "images_by_style": images_by_style,
            "images_by_source": images_by_source,
            "region_distribution": region_distribution,
        },
    }

    path = output_dir / "manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Saved manifest.json to %s", path)
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def merge_datasets(
    source_dirs: list[Path],
    output_dir: Path,
    *,
    mode: str = "copy",
    validate: bool = True,
    resolution: int = RENDER_RESOLUTION,
    seed: int = 42,
) -> MergeReport:
    """Merge multiple dataset sources into a unified dataset.

    Discovers characters from each source directory, validates files,
    copies/links them into the output directory, and regenerates
    ``class_map.json``, ``splits.json``, and ``manifest.json``.

    Args:
        source_dirs: List of source dataset directories.
        output_dir: Destination directory for the merged dataset.
        mode: File transfer mode — ``"copy"`` (default) or ``"link"``
            (symlink).
        validate: Whether to validate files before merging.
        resolution: Expected image resolution for validation.
        seed: Random seed for split generation.

    Returns:
        MergeReport with summary statistics.
    """
    t_start = time.monotonic()
    report = MergeReport()

    # Ensure output directory structure
    for subdir in _SUBDIRS:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Track all character IDs across sources for collision detection
    merged_char_ids: dict[str, str] = {}  # char_id -> source

    for source_dir in source_dirs:
        if not source_dir.is_dir():
            msg = f"Source directory does not exist: {source_dir}"
            logger.warning(msg)
            report.warnings.append(msg)
            continue

        report.sources_processed += 1
        characters = _discover_characters(source_dir)

        if not characters:
            msg = f"No characters found in {source_dir}"
            logger.warning(msg)
            report.warnings.append(msg)
            continue

        logger.info(
            "Found %d character(s) in %s",
            len(characters),
            source_dir,
        )

        for char_id, meta in characters.items():
            source = meta.get("source", source_dir.name)
            new_char_id: str | None = None

            # Check for ID collisions
            if char_id in merged_char_ids:
                if merged_char_ids[char_id] == source:
                    msg = f"Duplicate character {char_id} from same source {source} — skipping"
                    logger.warning(msg)
                    report.warnings.append(msg)
                    report.characters_skipped += 1
                    continue
                else:
                    new_char_id = f"{source}_{char_id}"
                    msg = (
                        f"ID collision: {char_id} already from "
                        f"{merged_char_ids[char_id]}, renaming to {new_char_id}"
                    )
                    logger.warning(msg)
                    report.warnings.append(msg)
                    report.characters_renamed += 1

            # Collect files
            char_files = _collect_character_files(source_dir, char_id)
            if not char_files:
                msg = f"No files found for character {char_id} in {source_dir}"
                logger.warning(msg)
                report.warnings.append(msg)
                report.characters_skipped += 1
                continue

            # Validate
            if validate:
                failures = _validate_character(source_dir, char_id, char_files, resolution)
                if failures:
                    report.validation_failures += len(failures)
                    for f in failures:
                        msg = f"Validation: {char_id}: {f}"
                        logger.warning(msg)
                        report.warnings.append(msg)
                    report.characters_skipped += 1
                    continue

            # Transfer files
            transferred, skipped = _transfer_character_files(
                char_files, output_dir, char_id, new_char_id, mode=mode
            )

            target_id = new_char_id or char_id
            merged_char_ids[target_id] = source

            if mode == "link":
                report.files_linked += transferred
            else:
                report.files_copied += transferred
            report.files_skipped += skipped
            report.characters_merged += 1

    # Generate class_map.json
    save_class_map(output_dir)

    # Generate splits.json using the splitter
    from .splitter import generate_splits

    generate_splits(output_dir, seed=seed)

    # Generate manifest.json
    _generate_merge_manifest(output_dir, report, source_dirs)

    report.elapsed_seconds = time.monotonic() - t_start

    logger.info(
        "Merge complete: %d characters merged, %d skipped, %d renamed in %.1fs",
        report.characters_merged,
        report.characters_skipped,
        report.characters_renamed,
        report.elapsed_seconds,
    )

    return report


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def print_merge_report(report: MergeReport) -> None:
    """Print a human-readable merge report to stdout."""
    print()
    print("=" * 60)
    print("DATASET MERGE REPORT")
    print("=" * 60)
    print()
    print(f"Sources processed:    {report.sources_processed}")
    print(f"Characters merged:    {report.characters_merged}")
    print(f"Characters skipped:   {report.characters_skipped}")
    print(f"Characters renamed:   {report.characters_renamed}")
    print(f"Files copied:         {report.files_copied}")
    print(f"Files linked:         {report.files_linked}")
    print(f"Files skipped:        {report.files_skipped}")
    print(f"Validation failures:  {report.validation_failures}")
    print(f"Time:                 {report.elapsed_seconds:.2f}s")

    if report.warnings:
        print()
        print(f"WARNINGS ({len(report.warnings)}):")
        print("-" * 50)
        for w in report.warnings[:30]:
            print(f"  - {w}")
        if len(report.warnings) > 30:
            print(f"  ... and {len(report.warnings) - 30} more")

    print()
    print("=" * 60)
