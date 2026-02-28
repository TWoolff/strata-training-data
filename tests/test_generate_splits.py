"""Tests for scripts/generate_splits.py."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.generate_splits import (
    _assign_proportional,
    _group_by_source,
    discover_all_characters,
    generate_splits,
    write_splits_csv,
    write_splits_json,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_image(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    """Create a small PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((*size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _create_source_meta(sources_dir: Path, char_id: str, source: str) -> None:
    """Create a source metadata JSON file."""
    sources_dir.mkdir(parents=True, exist_ok=True)
    meta = {"id": char_id, "source": source}
    (sources_dir / f"{char_id}.json").write_text(json.dumps(meta))


def _setup_multi_source(tmp_path: Path) -> Path:
    """Set up an output dir with multiple sources and characters."""
    out = tmp_path / "output"

    # Source A: segmentation pipeline format
    seg = out / "segmentation"
    for i in range(10):
        char_id = f"mixamo_{i:03d}"
        _create_source_meta(seg / "sources", char_id, "mixamo")
        for pose in range(2):
            _create_image(seg / "images" / f"{char_id}_pose_{pose:02d}_flat.png")

    # Source B: ingest adapter format
    ingest = out / "nova_human"
    for i in range(6):
        char_id = f"nova_{i:03d}"
        _create_source_meta(ingest / "sources", char_id, "nova_human")
        _create_image(ingest / "images" / f"{char_id}_pose_00_flat.png")

    return out


# ---------------------------------------------------------------------------
# Group by source
# ---------------------------------------------------------------------------


class TestGroupBySource:
    def test_groups_correctly(self) -> None:
        chars = {"a": "mixamo", "b": "mixamo", "c": "sketchfab"}
        groups = _group_by_source(chars)
        assert set(groups["mixamo"]) == {"a", "b"}
        assert groups["sketchfab"] == ["c"]

    def test_empty(self) -> None:
        assert _group_by_source({}) == {}


# ---------------------------------------------------------------------------
# Proportional assignment
# ---------------------------------------------------------------------------


class TestAssignProportional:
    def test_basic_split(self) -> None:
        ids = [f"char_{i}" for i in range(10)]
        splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
        ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
        _assign_proportional(ids, splits, ratios)
        assert len(splits["train"]) == 8
        assert len(splits["val"]) == 1
        assert len(splits["test"]) == 1

    def test_single_item(self) -> None:
        splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
        ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
        _assign_proportional(["only_one"], splits, ratios)
        total = sum(len(v) for v in splits.values())
        assert total == 1

    def test_empty_ids(self) -> None:
        splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
        ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
        _assign_proportional([], splits, ratios)
        assert all(len(v) == 0 for v in splits.values())


# ---------------------------------------------------------------------------
# Character discovery
# ---------------------------------------------------------------------------


class TestDiscoverAllCharacters:
    def test_discovers_from_sources_metadata(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        _create_source_meta(out / "sources", "char_a", "mixamo")
        _create_source_meta(out / "sources", "char_b", "sketchfab")
        chars = discover_all_characters(out)
        assert chars["char_a"] == "mixamo"
        assert chars["char_b"] == "sketchfab"

    def test_discovers_from_image_filenames(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        _create_image(out / "images" / "mixamo_001_pose_00_flat.png")
        _create_image(out / "images" / "mixamo_001_pose_01_flat.png")
        _create_image(out / "images" / "sketchfab_002_pose_00_flat.png")
        chars = discover_all_characters(out)
        assert "mixamo_001" in chars
        assert "sketchfab_002" in chars

    def test_discovers_from_subdirectories(self, tmp_path: Path) -> None:
        out = _setup_multi_source(tmp_path)
        chars = discover_all_characters(out)
        # Should find both mixamo and nova_human characters
        mixamo_chars = [c for c, s in chars.items() if s == "mixamo"]
        nova_chars = [c for c, s in chars.items() if s == "nova_human"]
        assert len(mixamo_chars) == 10
        assert len(nova_chars) == 6

    def test_empty_directory(self, tmp_path: Path) -> None:
        chars = discover_all_characters(tmp_path)
        assert chars == {}


# ---------------------------------------------------------------------------
# Split generation
# ---------------------------------------------------------------------------


class TestGenerateSplits:
    def test_deterministic(self, tmp_path: Path) -> None:
        out = _setup_multi_source(tmp_path)
        splits1 = generate_splits(out, seed=42)
        splits2 = generate_splits(out, seed=42)
        assert splits1 == splits2

    def test_different_seed_different_result(self, tmp_path: Path) -> None:
        out = _setup_multi_source(tmp_path)
        splits1 = generate_splits(out, seed=42)
        splits2 = generate_splits(out, seed=99)
        # With 16 characters, different seeds should produce different assignments
        assert splits1 != splits2

    def test_no_character_in_multiple_splits(self, tmp_path: Path) -> None:
        out = _setup_multi_source(tmp_path)
        splits = generate_splits(out)
        all_ids: list[str] = []
        for ids in splits.values():
            all_ids.extend(ids)
        assert len(all_ids) == len(set(all_ids))

    def test_all_characters_assigned(self, tmp_path: Path) -> None:
        out = _setup_multi_source(tmp_path)
        chars = discover_all_characters(out)
        splits = generate_splits(out)
        assigned = {cid for ids in splits.values() for cid in ids}
        assert assigned == set(chars.keys())

    def test_approximate_ratios(self, tmp_path: Path) -> None:
        out = _setup_multi_source(tmp_path)
        splits = generate_splits(out)
        total = sum(len(ids) for ids in splits.values())
        # With 16 characters, 80/10/10 → roughly 13/1-2/1-2
        assert len(splits["train"]) >= total * 0.6
        assert len(splits["train"]) <= total * 0.95

    def test_custom_ratios(self, tmp_path: Path) -> None:
        out = _setup_multi_source(tmp_path)
        splits = generate_splits(out, ratios={"train": 0.5, "val": 0.25, "test": 0.25})
        total = sum(len(ids) for ids in splits.values())
        assert total == 16
        assert len(splits["train"]) >= 6

    def test_empty_output(self, tmp_path: Path) -> None:
        splits = generate_splits(tmp_path)
        assert splits == {"train": [], "val": [], "test": []}

    def test_balanced_source_representation(self, tmp_path: Path) -> None:
        """Each source should be represented in the train split."""
        out = _setup_multi_source(tmp_path)
        splits = generate_splits(out)
        chars = discover_all_characters(out)

        train_sources = {chars[cid] for cid in splits["train"] if cid in chars}
        # Both sources should appear in training
        assert "mixamo" in train_sources
        assert "nova_human" in train_sources


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


class TestWriteSplitsCsv:
    def test_writes_valid_csv(self, tmp_path: Path) -> None:
        splits = {"train": ["a", "b"], "val": ["c"], "test": ["d"]}
        char_sources = {"a": "mixamo", "b": "mixamo", "c": "nova", "d": "nova"}
        path = tmp_path / "splits.csv"
        write_splits_csv(splits, char_sources, path)

        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 4
        assert set(rows[0].keys()) == {"character_id", "split", "source"}

        # All characters present
        ids = {r["character_id"] for r in rows}
        assert ids == {"a", "b", "c", "d"}

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "splits.csv"
        write_splits_csv({"train": ["a"]}, {"a": "x"}, path)
        assert path.exists()


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


class TestWriteSplitsJson:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        splits = {"train": ["a", "b"], "val": ["c"], "test": ["d"]}
        path = tmp_path / "splits.json"
        write_splits_json(splits, path)

        data = json.loads(path.read_text())
        assert data == splits
