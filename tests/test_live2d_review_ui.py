"""Tests for the Live2D annotation review UI logic.

Tests exercise the non-GUI data model: ReviewState navigation, fragment
confirmation, CSV round-trip, and fragment discovery — all without requiring
tkinter or a display server.
"""

from __future__ import annotations

import csv
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

# Provide a mock bpy module so pipeline imports work outside Blender.
_bpy_mock = types.ModuleType("bpy")
_bpy_mock.types = MagicMock()
sys.modules.setdefault("bpy", _bpy_mock)

from pipeline.live2d_mapper import FragmentMapping, ModelMapping  # noqa: E402
from pipeline.live2d_review_ui import (  # noqa: E402
    ReviewState,
    _auto_map_model,
    _discover_fragment_names,
    confirm_fragment,
    load_or_create_csv,
    save_review_csv,
    update_fragment_label,
)

# ---------------------------------------------------------------------------
# ReviewState — navigation
# ---------------------------------------------------------------------------


class TestReviewState:
    """Test ReviewState navigation and progress tracking."""

    @staticmethod
    def _make_model(model_id: str, fragments: list[tuple[str, str, int, str]]) -> ModelMapping:
        """Helper to build a ModelMapping from (name, label, rid, confirmed) tuples."""
        m = ModelMapping(model_id=model_id)
        for name, label, rid, confirmed in fragments:
            m.mappings.append(
                FragmentMapping(
                    fragment_name=name,
                    strata_label=label,
                    strata_region_id=rid,
                    confirmed=confirmed,
                )
            )
        return m

    def test_initial_state(self, tmp_path: Path) -> None:
        model = self._make_model(
            "m1",
            [
                ("head", "head", 1, "auto"),
                ("body", "chest", 3, "auto"),
            ],
        )
        state = ReviewState(csv_path=tmp_path / "test.csv", models=[model])

        assert state.total_fragments == 2
        assert state.reviewed_fragments == 0
        assert state.models_completed == 0
        assert state.current_model is not None
        assert state.current_model.model_id == "m1"

    def test_advance_through_fragments(self, tmp_path: Path) -> None:
        model = self._make_model(
            "m1",
            [
                ("head", "head", 1, "auto"),
                ("body", "chest", 3, "auto"),
                ("hand", "hand_l", 8, "pending"),
            ],
        )
        state = ReviewState(csv_path=tmp_path / "test.csv", models=[model])

        frag = state.current_fragment
        assert frag is not None
        assert frag.fragment_name == "head"

        assert state.advance()
        frag = state.current_fragment
        assert frag is not None
        assert frag.fragment_name == "body"

        assert state.advance()
        frag = state.current_fragment
        assert frag is not None
        assert frag.fragment_name == "hand"

        # No more fragments
        assert not state.advance()

    def test_advance_across_models(self, tmp_path: Path) -> None:
        m1 = self._make_model("m1", [("head", "head", 1, "auto")])
        m2 = self._make_model("m2", [("neck", "neck", 2, "auto")])
        state = ReviewState(csv_path=tmp_path / "test.csv", models=[m1, m2])

        assert state.current_model.model_id == "m1"
        assert state.advance()
        assert state.current_model.model_id == "m2"
        assert state.current_fragment.fragment_name == "neck"
        assert not state.advance()

    def test_skip_confirmed_manual(self, tmp_path: Path) -> None:
        """Fragments with confirmed=manual are excluded from pending list."""
        model = self._make_model(
            "m1",
            [
                ("head", "head", 1, "manual"),
                ("body", "chest", 3, "auto"),
            ],
        )
        state = ReviewState(csv_path=tmp_path / "test.csv", models=[model])

        pending = state.pending_fragments(model)
        assert len(pending) == 1
        assert pending[0].fragment_name == "body"

    def test_go_back(self, tmp_path: Path) -> None:
        model = self._make_model(
            "m1",
            [
                ("head", "head", 1, "auto"),
                ("body", "chest", 3, "auto"),
            ],
        )
        state = ReviewState(csv_path=tmp_path / "test.csv", models=[model])

        # At start, can't go back
        assert not state.go_back()

        # Advance then go back
        state.advance()
        assert state.current_fragment.fragment_name == "body"
        assert state.go_back()
        assert state.current_fragment.fragment_name == "head"

    def test_reviewed_count(self, tmp_path: Path) -> None:
        model = self._make_model(
            "m1",
            [
                ("head", "head", 1, "manual"),
                ("body", "chest", 3, "auto"),
                ("hand", "UNMAPPED", -1, "pending"),
            ],
        )
        state = ReviewState(csv_path=tmp_path / "test.csv", models=[model])
        assert state.reviewed_fragments == 1

    def test_models_completed(self, tmp_path: Path) -> None:
        m1 = self._make_model(
            "m1",
            [
                ("head", "head", 1, "manual"),
                ("body", "chest", 3, "manual"),
            ],
        )
        m2 = self._make_model(
            "m2",
            [
                ("neck", "neck", 2, "auto"),
            ],
        )
        state = ReviewState(csv_path=tmp_path / "test.csv", models=[m1, m2])
        assert state.models_completed == 1

    def test_empty_state(self, tmp_path: Path) -> None:
        state = ReviewState(csv_path=tmp_path / "test.csv", models=[])
        assert state.current_model is None
        assert state.current_fragment is None
        assert state.total_fragments == 0
        assert not state.advance()


# ---------------------------------------------------------------------------
# Fragment confirmation / update
# ---------------------------------------------------------------------------


class TestFragmentUpdate:
    """Test updating and confirming fragment labels."""

    def test_confirm_fragment(self) -> None:
        frag = FragmentMapping(
            fragment_name="head",
            strata_label="head",
            strata_region_id=1,
            confirmed="auto",
        )
        confirm_fragment(frag)
        assert frag.confirmed == "manual"
        assert frag.strata_label == "head"
        assert frag.strata_region_id == 1

    def test_update_fragment_label(self) -> None:
        frag = FragmentMapping(
            fragment_name="unknown_part",
            strata_label="UNMAPPED",
            strata_region_id=-1,
            confirmed="pending",
        )
        update_fragment_label(frag, "chest")
        assert frag.strata_label == "chest"
        assert frag.strata_region_id == 3
        assert frag.confirmed == "manual"

    def test_update_to_background(self) -> None:
        frag = FragmentMapping(
            fragment_name="ribbon",
            strata_label="background",
            strata_region_id=0,
            confirmed="auto",
        )
        update_fragment_label(frag, "background")
        assert frag.strata_label == "background"
        assert frag.strata_region_id == 0
        assert frag.confirmed == "manual"

    def test_update_invalid_label(self) -> None:
        frag = FragmentMapping(
            fragment_name="test",
            strata_label="head",
            strata_region_id=1,
            confirmed="auto",
        )
        update_fragment_label(frag, "nonexistent_region")
        assert frag.strata_label == "nonexistent_region"
        assert frag.strata_region_id == -1
        assert frag.confirmed == "manual"


# ---------------------------------------------------------------------------
# CSV round-trip with review state
# ---------------------------------------------------------------------------


class TestReviewCSV:
    """Test saving and loading review state via CSV."""

    def test_save_and_reload(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "review.csv"

        model = ModelMapping(model_id="model_001")
        model.mappings.append(FragmentMapping("head", "head", 1, "manual"))
        model.mappings.append(FragmentMapping("body", "chest", 3, "auto"))
        model.mappings.append(FragmentMapping("unknown", "UNMAPPED", -1, "pending"))

        state = ReviewState(csv_path=csv_path, models=[model])
        save_review_csv(state)

        assert csv_path.exists()

        # Read raw CSV to verify structure
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        assert rows[0]["confirmed"] == "manual"
        assert rows[1]["confirmed"] == "auto"
        assert rows[2]["confirmed"] == "pending"

        # Reload and verify
        from pipeline.live2d_mapper import load_csv

        loaded = load_csv(csv_path)
        assert len(loaded) == 1
        assert loaded[0].model_id == "model_001"
        assert len(loaded[0].mappings) == 3

    def test_load_or_create_from_existing(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "existing.csv"

        # Write a CSV first
        model = ModelMapping(model_id="m1")
        model.mappings.append(FragmentMapping("head", "head", 1, "manual"))

        from pipeline.live2d_mapper import export_csv

        export_csv([model], csv_path)

        # load_or_create should load the existing file
        result = load_or_create_csv(csv_path, [])
        assert len(result) == 1
        assert result[0].model_id == "m1"

    def test_load_or_create_discovers_models(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "new.csv"

        # Create a fake model directory with fragment PNGs
        model_dir = tmp_path / "models" / "char_01"
        model_dir.mkdir(parents=True)

        # Create minimal PNG files (1x1 pixel)
        from PIL import Image

        for name in ["head", "body", "unknown_part"]:
            img = Image.new("RGBA", (10, 10), (255, 0, 0, 255))
            img.save(model_dir / f"{name}.png")

        result = load_or_create_csv(csv_path, [model_dir])
        assert len(result) == 1
        assert result[0].model_id == "char_01"
        assert result[0].total_count == 3
        assert csv_path.exists()


# ---------------------------------------------------------------------------
# Fragment discovery
# ---------------------------------------------------------------------------


class TestFragmentDiscovery:
    """Test fragment name discovery from model directories."""

    def test_discover_from_root(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model_a"
        model_dir.mkdir()
        (model_dir / "head.png").touch()
        (model_dir / "body.png").touch()
        (model_dir / "not_an_image.txt").touch()

        names = _discover_fragment_names(model_dir)
        assert set(names) == {"head", "body"}

    def test_discover_from_parts_subdir(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model_b"
        parts_dir = model_dir / "parts"
        parts_dir.mkdir(parents=True)
        (parts_dir / "arm_upper_L.png").touch()
        (parts_dir / "arm_lower_L.png").touch()

        names = _discover_fragment_names(model_dir)
        assert set(names) == {"arm_upper_L", "arm_lower_L"}

    def test_discover_empty_dir(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "empty"
        model_dir.mkdir()

        names = _discover_fragment_names(model_dir)
        assert names == []

    def test_no_duplicates(self, tmp_path: Path) -> None:
        """PNGs in root should not duplicate those found in parts/."""
        model_dir = tmp_path / "model_c"
        parts_dir = model_dir / "parts"
        parts_dir.mkdir(parents=True)
        (parts_dir / "head.png").touch()
        (model_dir / "head.png").touch()  # duplicate name, different path

        names = _discover_fragment_names(model_dir)
        # Should have 2 entries since they're different files
        # (both "parts/head.png" and "model_c/head.png")
        assert "head" in names


# ---------------------------------------------------------------------------
# Auto-mapper helper
# ---------------------------------------------------------------------------


class TestAutoMapModel:
    """Test the _auto_map_model helper."""

    def test_maps_known_fragments(self) -> None:
        result = _auto_map_model("test", ["head", "neck", "body"])
        assert result.model_id == "test"
        assert result.total_count == 3
        assert all(m.confirmed == "auto" for m in result.mappings)

    def test_marks_unknown_pending(self) -> None:
        result = _auto_map_model("test", ["unknown_xyz"])
        assert result.mappings[0].confirmed == "pending"
        assert result.mappings[0].strata_label == "UNMAPPED"

    def test_empty_fragments(self) -> None:
        result = _auto_map_model("test", [])
        assert result.total_count == 0
