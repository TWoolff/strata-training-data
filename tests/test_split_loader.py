"""Tests for training/data/split_loader.py — character-level dataset splitting."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from training.data.split_loader import (
    character_id_from_example,
    load_or_generate_splits,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_image(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    """Create a minimal PNG file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", size, (0, 0, 0, 0)).save(path)


def _create_source_meta(sources_dir: Path, char_id: str, source: str = "mixamo") -> None:
    """Create a character source metadata JSON."""
    sources_dir.mkdir(parents=True, exist_ok=True)
    meta = {"id": char_id, "source": source}
    (sources_dir / f"{char_id}.json").write_text(json.dumps(meta), encoding="utf-8")


def _setup_dataset(tmp_path: Path, char_ids: list[str], poses: int = 3) -> Path:
    """Create a realistic dataset directory with images for multiple characters."""
    dataset_dir = tmp_path / "dataset"
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True)

    for char_id in char_ids:
        for pose_idx in range(poses):
            filename = f"{char_id}_pose_{pose_idx:02d}_flat.png"
            _create_image(images_dir / filename)

    return dataset_dir


# ---------------------------------------------------------------------------
# character_id_from_example
# ---------------------------------------------------------------------------


class TestCharacterIdFromExample:
    def test_mixamo_standard(self):
        assert character_id_from_example("mixamo_001_pose_05_flat") == "mixamo_001"

    def test_mixamo_no_style(self):
        assert character_id_from_example("mixamo_001_pose_05") == "mixamo_001"

    def test_fbanimehq(self):
        assert character_id_from_example("fbanimehq_0000_000005") == "fbanimehq_0000"

    def test_stdgen(self):
        assert character_id_from_example("stdgen_0042_front") == "stdgen_0042"

    def test_animerun(self):
        assert character_id_from_example("animerun_clip01_000005") == "animerun_clip01"

    def test_nova_human_passthrough(self):
        """NOVA-Human has one image per character — ID passes through unchanged."""
        assert character_id_from_example("nova_human_12345") == "nova_human_12345"

    def test_unknown_format_passthrough(self):
        """Unknown naming format should return the full ID."""
        assert character_id_from_example("custom_character") == "custom_character"

    def test_complex_mixamo_name(self):
        """Multi-part character names should extract correctly."""
        assert character_id_from_example("sketchfab_warrior_01_pose_03_cel") == (
            "sketchfab_warrior_01"
        )


# ---------------------------------------------------------------------------
# load_or_generate_splits
# ---------------------------------------------------------------------------


class TestLoadOrGenerateSplits:
    def test_basic_split_ratios(self, tmp_path: Path):
        """Characters should be split roughly 80/10/10."""
        char_ids = [f"char_{i:03d}" for i in range(100)]
        dataset_dir = _setup_dataset(tmp_path, char_ids, poses=1)

        splits = load_or_generate_splits([dataset_dir])

        total = sum(len(ids) for ids in splits.values())
        assert total == 100
        assert len(splits["train"]) == 80
        assert len(splits["val"]) == 10
        assert len(splits["test"]) == 10

    def test_deterministic_with_seed(self, tmp_path: Path):
        """Same seed should produce identical splits."""
        char_ids = [f"char_{i:03d}" for i in range(50)]
        dataset_dir = _setup_dataset(tmp_path, char_ids)

        splits1 = load_or_generate_splits([dataset_dir], seed=42)
        splits2 = load_or_generate_splits([dataset_dir], seed=42)

        assert splits1 == splits2

    def test_different_seed_different_splits(self, tmp_path: Path):
        """Different seeds should produce different splits."""
        char_ids = [f"char_{i:03d}" for i in range(50)]
        dataset_dir = _setup_dataset(tmp_path, char_ids)

        splits1 = load_or_generate_splits([dataset_dir], seed=42)
        splits2 = load_or_generate_splits([dataset_dir], seed=99)

        assert splits1 != splits2

    def test_no_character_in_multiple_splits(self, tmp_path: Path):
        """No character should appear in more than one split."""
        char_ids = [f"char_{i:03d}" for i in range(50)]
        dataset_dir = _setup_dataset(tmp_path, char_ids)

        splits = load_or_generate_splits([dataset_dir])

        all_ids = splits["train"] + splits["val"] + splits["test"]
        assert len(all_ids) == len(set(all_ids))

    def test_reads_existing_splits_json(self, tmp_path: Path):
        """Should respect existing splits.json assignments."""
        char_ids = [f"char_{i:03d}" for i in range(10)]
        dataset_dir = _setup_dataset(tmp_path, char_ids)

        # Write a pre-existing splits.json
        existing = {
            "train": ["char_000", "char_001"],
            "val": ["char_002"],
            "test": ["char_003"],
        }
        (dataset_dir / "splits.json").write_text(json.dumps(existing), encoding="utf-8")

        splits = load_or_generate_splits([dataset_dir])

        # Pre-existing assignments should be preserved
        assert "char_000" in splits["train"]
        assert "char_001" in splits["train"]
        assert "char_002" in splits["val"]
        assert "char_003" in splits["test"]

        # All 10 characters should be assigned
        total = sum(len(ids) for ids in splits.values())
        assert total == 10

    def test_multiple_dataset_dirs(self, tmp_path: Path):
        """Should discover characters from multiple directories."""
        dir1 = _setup_dataset(tmp_path / "d1", [f"mixamo_{i:03d}" for i in range(5)])
        dir2 = _setup_dataset(tmp_path / "d2", [f"sketchfab_{i:03d}" for i in range(5)])

        splits = load_or_generate_splits([dir1, dir2])

        total = sum(len(ids) for ids in splits.values())
        assert total == 10

    def test_empty_directory(self, tmp_path: Path):
        """Should handle empty directories gracefully."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        splits = load_or_generate_splits([empty_dir])

        assert splits == {"train": [], "val": [], "test": []}

    def test_single_character(self, tmp_path: Path):
        """Single character should go to train."""
        dataset_dir = _setup_dataset(tmp_path, ["lone_char"])

        splits = load_or_generate_splits([dataset_dir])

        total = sum(len(ids) for ids in splits.values())
        assert total == 1
        assert "lone_char" in splits["train"]

    def test_splits_are_sorted(self, tmp_path: Path):
        """All split lists should be sorted for deterministic output."""
        char_ids = [f"char_{i:03d}" for i in range(30)]
        dataset_dir = _setup_dataset(tmp_path, char_ids)

        splits = load_or_generate_splits([dataset_dir])

        for name, ids in splits.items():
            assert ids == sorted(ids), f"{name} split is not sorted"

    def test_sources_fallback(self, tmp_path: Path):
        """Should discover characters from sources/ when images/ is absent."""
        dataset_dir = tmp_path / "dataset"
        sources_dir = dataset_dir / "sources"

        for i in range(5):
            _create_source_meta(sources_dir, f"char_{i:03d}")

        splits = load_or_generate_splits([dataset_dir])

        total = sum(len(ids) for ids in splits.values())
        assert total == 5
