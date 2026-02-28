"""Tests for pipeline/dataset_merger.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.dataset_merger import (
    MergeReport,
    _collect_character_files,
    _discover_characters,
    _rename_file,
    _validate_character,
    merge_datasets,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_image(path: Path, size: tuple[int, int] = (512, 512)) -> None:
    """Create a small RGBA PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((*size, 4), dtype=np.uint8)
    arr[:, :, 3] = 255  # opaque
    Image.fromarray(arr, mode="RGBA").save(path)


def _create_mask(path: Path, size: tuple[int, int] = (512, 512)) -> None:
    """Create an 8-bit grayscale mask PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ones(size, dtype=np.uint8) * 3  # region 3 = chest
    arr[:256, :] = 1  # region 1 = head (top half)
    Image.fromarray(arr, mode="L").save(path)


def _create_joints(path: Path, char_id: str = "test", pose: int = 0) -> None:
    """Create a minimal joints JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joints = {
        "image_size": [512, 512],
        "character_id": char_id,
        "pose_index": pose,
        "joints": {},
    }
    path.write_text(json.dumps(joints, indent=2), encoding="utf-8")


def _create_source_meta(
    sources_dir: Path,
    char_id: str,
    source: str,
) -> None:
    """Create a source metadata JSON file."""
    sources_dir.mkdir(parents=True, exist_ok=True)
    meta = {"id": char_id, "source": source}
    (sources_dir / f"{char_id}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _setup_source_dir(
    base: Path,
    name: str,
    characters: list[tuple[str, str]],
    poses: int = 2,
    resolution: int = 512,
) -> Path:
    """Set up a source directory with characters and pose files.

    Args:
        base: Parent directory.
        name: Source directory name.
        characters: List of (char_id, source) tuples.
        poses: Number of poses per character.
        resolution: Image resolution.

    Returns:
        Path to the source directory.
    """
    source_dir = base / name
    for char_id, source in characters:
        _create_source_meta(source_dir / "sources", char_id, source)
        for p in range(poses):
            _create_image(
                source_dir / "images" / f"{char_id}_pose_{p:02d}_flat.png",
                size=(resolution, resolution),
            )
            _create_mask(
                source_dir / "masks" / f"{char_id}_pose_{p:02d}.png",
                size=(resolution, resolution),
            )
            _create_joints(
                source_dir / "joints" / f"{char_id}_pose_{p:02d}.json",
                char_id=char_id,
                pose=p,
            )
    return source_dir


# ---------------------------------------------------------------------------
# Character discovery
# ---------------------------------------------------------------------------


class TestDiscoverCharacters:
    def test_discovers_from_metadata(self, tmp_path: Path) -> None:
        src = _setup_source_dir(
            tmp_path, "src", [("mixamo_001", "mixamo"), ("mixamo_002", "mixamo")]
        )
        chars = _discover_characters(src)
        assert "mixamo_001" in chars
        assert "mixamo_002" in chars
        assert chars["mixamo_001"]["source"] == "mixamo"

    def test_discovers_from_images_fallback(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _create_image(src / "images" / "char_a_pose_00_flat.png")
        _create_image(src / "images" / "char_a_pose_01_flat.png")
        _create_image(src / "images" / "char_b_pose_00_flat.png")
        chars = _discover_characters(src)
        assert "char_a" in chars
        assert "char_b" in chars

    def test_empty_directory(self, tmp_path: Path) -> None:
        src = tmp_path / "empty"
        src.mkdir()
        chars = _discover_characters(src)
        assert chars == {}


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------


class TestCollectCharacterFiles:
    def test_collects_all_subdirs(self, tmp_path: Path) -> None:
        src = _setup_source_dir(tmp_path, "src", [("mixamo_001", "mixamo")], poses=2)
        files = _collect_character_files(src, "mixamo_001")
        assert "images" in files
        assert "masks" in files
        assert "joints" in files
        assert "sources" in files
        assert len(files["images"]) == 2
        assert len(files["masks"]) == 2

    def test_no_prefix_false_match(self, tmp_path: Path) -> None:
        """mixamo_01 should not match mixamo_010."""
        src = _setup_source_dir(
            tmp_path,
            "src",
            [("mixamo_01", "mixamo"), ("mixamo_010", "mixamo")],
            poses=1,
        )
        files_01 = _collect_character_files(src, "mixamo_01")
        files_010 = _collect_character_files(src, "mixamo_010")
        assert len(files_01["images"]) == 1
        assert len(files_010["images"]) == 1


# ---------------------------------------------------------------------------
# File renaming
# ---------------------------------------------------------------------------


class TestRenameFile:
    def test_renames_prefix(self) -> None:
        assert (
            _rename_file("char_a_pose_00_flat.png", "char_a", "new_char_a")
            == "new_char_a_pose_00_flat.png"
        )

    def test_no_match(self) -> None:
        assert _rename_file("other.png", "char_a", "new_a") == "other.png"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidateCharacter:
    def test_passes_valid_files(self, tmp_path: Path) -> None:
        src = _setup_source_dir(tmp_path, "src", [("char_a", "test")], poses=1)
        files = _collect_character_files(src, "char_a")
        failures = _validate_character(src, "char_a", files, 512)
        assert failures == []

    def test_fails_wrong_resolution(self, tmp_path: Path) -> None:
        src = _setup_source_dir(tmp_path, "src", [("char_a", "test")], poses=1, resolution=256)
        files = _collect_character_files(src, "char_a")
        failures = _validate_character(src, "char_a", files, 512)
        assert len(failures) > 0
        assert "expected 512x512" in failures[0]


# ---------------------------------------------------------------------------
# Full merge
# ---------------------------------------------------------------------------


class TestMergeDatasets:
    def test_basic_merge(self, tmp_path: Path) -> None:
        src_a = _setup_source_dir(tmp_path, "source_a", [("mixamo_001", "mixamo")], poses=2)
        src_b = _setup_source_dir(tmp_path, "source_b", [("spine_001", "spine")], poses=1)
        out = tmp_path / "merged"

        report = merge_datasets([src_a, src_b], out)

        assert report.characters_merged == 2
        assert report.characters_skipped == 0
        assert (out / "class_map.json").exists()
        assert (out / "splits.json").exists()
        assert (out / "manifest.json").exists()
        assert (out / "images" / "mixamo_001_pose_00_flat.png").exists()
        assert (out / "images" / "spine_001_pose_00_flat.png").exists()

    def test_id_collision_different_sources(self, tmp_path: Path) -> None:
        src_a = _setup_source_dir(tmp_path, "source_a", [("char_001", "mixamo")], poses=1)
        src_b = _setup_source_dir(tmp_path, "source_b", [("char_001", "spine")], poses=1)
        out = tmp_path / "merged"

        report = merge_datasets([src_a, src_b], out)

        assert report.characters_merged == 2
        assert report.characters_renamed == 1
        # Original and renamed both exist
        assert (out / "sources" / "char_001.json").exists()
        assert (out / "sources" / "spine_char_001.json").exists()

    def test_id_collision_same_source_skips(self, tmp_path: Path) -> None:
        src_a = _setup_source_dir(tmp_path, "source_a", [("mixamo_001", "mixamo")], poses=1)
        src_b = _setup_source_dir(tmp_path, "source_b", [("mixamo_001", "mixamo")], poses=1)
        out = tmp_path / "merged"

        report = merge_datasets([src_a, src_b], out)

        assert report.characters_merged == 1
        assert report.characters_skipped == 1

    def test_symlink_mode(self, tmp_path: Path) -> None:
        src = _setup_source_dir(tmp_path, "source", [("char_a", "test")], poses=1)
        out = tmp_path / "merged"

        report = merge_datasets([src], out, mode="link")

        assert report.files_linked > 0
        assert report.files_copied == 0
        img = out / "images" / "char_a_pose_00_flat.png"
        assert img.is_symlink()

    def test_skip_invalid_files(self, tmp_path: Path) -> None:
        src = _setup_source_dir(tmp_path, "source", [("char_a", "test")], poses=1, resolution=256)
        out = tmp_path / "merged"

        report = merge_datasets([src], out, resolution=512)

        assert report.characters_skipped == 1
        assert report.validation_failures > 0
        assert report.characters_merged == 0

    def test_no_validate_flag(self, tmp_path: Path) -> None:
        src = _setup_source_dir(tmp_path, "source", [("char_a", "test")], poses=1, resolution=256)
        out = tmp_path / "merged"

        report = merge_datasets([src], out, validate=False, resolution=512)

        assert report.characters_merged == 1
        assert report.validation_failures == 0

    def test_missing_source_dir(self, tmp_path: Path) -> None:
        out = tmp_path / "merged"
        report = merge_datasets([tmp_path / "nonexistent"], out)
        assert report.sources_processed == 0
        assert len(report.warnings) > 0

    def test_splits_json_generated(self, tmp_path: Path) -> None:
        src = _setup_source_dir(
            tmp_path,
            "source",
            [
                ("mixamo_001", "mixamo"),
                ("mixamo_002", "mixamo"),
                ("spine_001", "spine"),
            ],
            poses=1,
        )
        out = tmp_path / "merged"

        merge_datasets([src], out)

        splits = json.loads((out / "splits.json").read_text())
        all_ids = splits["train"] + splits["val"] + splits["test"]
        assert set(all_ids) == {"mixamo_001", "mixamo_002", "spine_001"}

    def test_manifest_json_generated(self, tmp_path: Path) -> None:
        src = _setup_source_dir(tmp_path, "source", [("mixamo_001", "mixamo")], poses=2)
        out = tmp_path / "merged"

        merge_datasets([src], out)

        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["statistics"]["total_characters"] == 1
        assert manifest["statistics"]["total_images"] == 2
        assert "merge" in manifest

    def test_class_map_generated(self, tmp_path: Path) -> None:
        src = _setup_source_dir(tmp_path, "source", [("char_a", "test")], poses=1)
        out = tmp_path / "merged"

        merge_datasets([src], out)

        class_map = json.loads((out / "class_map.json").read_text())
        assert class_map["0"] == "background"
        assert class_map["1"] == "head"
        assert len(class_map) == 20


# ---------------------------------------------------------------------------
# MergeReport
# ---------------------------------------------------------------------------


class TestMergeReport:
    def test_to_dict(self) -> None:
        report = MergeReport(
            sources_processed=2,
            characters_merged=5,
            elapsed_seconds=1.234,
        )
        d = report.to_dict()
        assert d["sources_processed"] == 2
        assert d["characters_merged"] == 5
        assert d["elapsed_seconds"] == 1.23
