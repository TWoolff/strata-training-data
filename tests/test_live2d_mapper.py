"""Tests for the Live2D fragment-to-Strata label mapper.

These tests exercise the pure-Python mapping logic without requiring Blender.
"""

from __future__ import annotations

import csv
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Provide a mock bpy module so pipeline imports work outside Blender.
_bpy_mock = types.ModuleType("bpy")
_bpy_mock.types = MagicMock()
sys.modules.setdefault("bpy", _bpy_mock)

from pipeline.live2d_mapper import (  # noqa: E402
    export_csv,
    load_csv,
    map_fragment,
    map_model,
    region_summary,
)

# ---------------------------------------------------------------------------
# map_fragment — individual fragment mapping
# ---------------------------------------------------------------------------


class TestMapFragment:
    """Test single-fragment mapping against LIVE2D_FRAGMENT_PATTERNS."""

    # --- Head region (1) ---

    @pytest.mark.parametrize(
        "fragment_name",
        [
            "head",
            "Head",
            "face",
            "kao",
            "hair_front",
            "hair_back",
            "bangs",
            "maegami",
            "ushirogami",
            "eye_L",
            "eye_R",
            "me_l",
            "hitomi",
            "mouth",
            "kuchi",
            "lip_upper",
            "brow_L",
            "mayu",
            "ear",
            "mimi",
            "nose",
            "hana",
            "atama",
            "kami_front",
        ],
    )
    def test_head_fragments(self, fragment_name: str) -> None:
        label, region_id = map_fragment(fragment_name)
        assert label == "head", f"{fragment_name!r} → {label}, expected 'head'"
        assert region_id == 1

    # --- Neck region (2) ---

    @pytest.mark.parametrize("fragment_name", ["neck", "Neck", "kubi"])
    def test_neck_fragments(self, fragment_name: str) -> None:
        label, region_id = map_fragment(fragment_name)
        assert label == "neck"
        assert region_id == 2

    # --- Chest / torso region (3) ---

    @pytest.mark.parametrize(
        "fragment_name",
        ["body", "Body", "torso", "karada", "chest", "mune", "bust"],
    )
    def test_chest_fragments(self, fragment_name: str) -> None:
        label, region_id = map_fragment(fragment_name)
        assert label == "chest"
        assert region_id == 3

    # --- Hips region (5) ---

    @pytest.mark.parametrize("fragment_name", ["hip", "hips", "pelvis", "koshi", "waist"])
    def test_hips_fragments(self, fragment_name: str) -> None:
        label, region_id = map_fragment(fragment_name)
        assert label == "hips"
        assert region_id == 5

    # --- Left arm regions ---

    @pytest.mark.parametrize(
        "fragment_name, expected_label, expected_id",
        [
            ("arm_upper_L", "upper_arm_l", 7),
            ("upper_arm_L", "upper_arm_l", 7),
            ("ude_ue_L", "upper_arm_l", 7),
            ("arm_upper_left", "upper_arm_l", 7),
            ("arm_lower_L", "forearm_l", 8),
            ("arm_fore_L", "forearm_l", 8),
            ("forearm_L", "forearm_l", 8),
            ("forearm_left", "forearm_l", 8),
            ("hand_L", "hand_l", 9),
            ("hand_left", "hand_l", 9),
            ("te_L", "hand_l", 9),
        ],
    )
    def test_left_arm_fragments(
        self, fragment_name: str, expected_label: str, expected_id: int
    ) -> None:
        label, region_id = map_fragment(fragment_name)
        assert label == expected_label, f"{fragment_name!r} → {label}, expected {expected_label!r}"
        assert region_id == expected_id

    # --- Right arm regions ---

    @pytest.mark.parametrize(
        "fragment_name, expected_label, expected_id",
        [
            ("arm_upper_R", "upper_arm_r", 11),
            ("upper_arm_R", "upper_arm_r", 11),
            ("ude_ue_R", "upper_arm_r", 11),
            ("arm_lower_R", "forearm_r", 12),
            ("arm_fore_R", "forearm_r", 12),
            ("forearm_R", "forearm_r", 12),
            ("hand_R", "hand_r", 13),
            ("hand_right", "hand_r", 13),
            ("te_R", "hand_r", 13),
        ],
    )
    def test_right_arm_fragments(
        self, fragment_name: str, expected_label: str, expected_id: int
    ) -> None:
        label, region_id = map_fragment(fragment_name)
        assert label == expected_label, f"{fragment_name!r} → {label}, expected {expected_label!r}"
        assert region_id == expected_id

    # --- Left leg regions ---

    @pytest.mark.parametrize(
        "fragment_name, expected_label, expected_id",
        [
            ("leg_upper_L", "upper_leg_l", 14),
            ("thigh_L", "upper_leg_l", 14),
            ("momo_L", "upper_leg_l", 14),
            ("leg_lower_L", "lower_leg_l", 15),
            ("leg_shin_L", "lower_leg_l", 15),
            ("shin_L", "lower_leg_l", 15),
            ("sune_L", "lower_leg_l", 15),
            ("foot_L", "foot_l", 16),
            ("foot_left", "foot_l", 16),
            ("ashi_L", "foot_l", 16),
        ],
    )
    def test_left_leg_fragments(
        self, fragment_name: str, expected_label: str, expected_id: int
    ) -> None:
        label, region_id = map_fragment(fragment_name)
        assert label == expected_label, f"{fragment_name!r} → {label}, expected {expected_label!r}"
        assert region_id == expected_id

    # --- Right leg regions ---

    @pytest.mark.parametrize(
        "fragment_name, expected_label, expected_id",
        [
            ("leg_upper_R", "upper_leg_r", 17),
            ("thigh_R", "upper_leg_r", 17),
            ("momo_R", "upper_leg_r", 17),
            ("leg_lower_R", "lower_leg_r", 18),
            ("shin_R", "lower_leg_r", 18),
            ("sune_R", "lower_leg_r", 18),
            ("foot_R", "foot_r", 19),
            ("foot_right", "foot_r", 19),
            ("ashi_R", "foot_r", 19),
        ],
    )
    def test_right_leg_fragments(
        self, fragment_name: str, expected_label: str, expected_id: int
    ) -> None:
        label, region_id = map_fragment(fragment_name)
        assert label == expected_label, f"{fragment_name!r} → {label}, expected {expected_label!r}"
        assert region_id == expected_id

    # --- Shoulder regions ---

    @pytest.mark.parametrize(
        "fragment_name, expected_label, expected_id",
        [
            ("shoulder_L", "shoulder_l", 6),
            ("shoulder_left", "shoulder_l", 6),
            ("kata_L", "shoulder_l", 6),
            ("shoulder_R", "shoulder_r", 10),
            ("shoulder_right", "shoulder_r", 10),
            ("kata_R", "shoulder_r", 10),
        ],
    )
    def test_shoulder_fragments(
        self, fragment_name: str, expected_label: str, expected_id: int
    ) -> None:
        label, region_id = map_fragment(fragment_name)
        assert label == expected_label, f"{fragment_name!r} → {label}, expected {expected_label!r}"
        assert region_id == expected_id

    # --- Accessories → background (0) ---

    @pytest.mark.parametrize(
        "fragment_name",
        [
            "cloth_upper",
            "dress",
            "skirt",
            "hat",
            "ribbon",
            "accessory_01",
            "cape",
            "armor_chest",
            "weapon",
            "shield",
        ],
    )
    def test_accessory_fragments(self, fragment_name: str) -> None:
        label, region_id = map_fragment(fragment_name)
        assert label == "background", f"{fragment_name!r} → {label}, expected 'background'"
        assert region_id == 0

    # --- Unmapped ---

    @pytest.mark.parametrize(
        "fragment_name",
        [
            "unknown_mesh_3",
            "ArtMesh42",
            "deformer_ctrl",
            "shadow_overlay",
        ],
    )
    def test_unmapped_fragments(self, fragment_name: str) -> None:
        label, region_id = map_fragment(fragment_name)
        assert label == "UNMAPPED"
        assert region_id == -1

    # --- Case insensitivity ---

    def test_case_insensitive(self) -> None:
        label1, _ = map_fragment("HEAD")
        label2, _ = map_fragment("head")
        label3, _ = map_fragment("Head")
        assert label1 == label2 == label3 == "head"


# ---------------------------------------------------------------------------
# map_model — full model mapping
# ---------------------------------------------------------------------------


class TestMapModel:
    """Test mapping an entire model's fragments."""

    def test_basic_model(self) -> None:
        fragments = ["head", "body", "arm_upper_L", "unknown_part"]
        result = map_model("test_001", fragments)

        assert result.model_id == "test_001"
        assert result.total_count == 4
        assert result.mapped_count == 3
        assert result.unmapped_count == 1
        assert result.auto_rate == pytest.approx(0.75)

    def test_empty_model(self) -> None:
        result = map_model("empty_model", [])
        assert result.total_count == 0
        assert result.mapped_count == 0
        assert result.auto_rate == 0.0

    def test_all_mapped(self) -> None:
        fragments = ["head", "neck", "body", "hand_L", "hand_R"]
        result = map_model("full_model", fragments)
        assert result.unmapped_count == 0
        assert result.auto_rate == 1.0

    def test_confirmed_status(self) -> None:
        fragments = ["head", "unknown_mesh"]
        result = map_model("status_test", fragments)
        statuses = {m.fragment_name: m.confirmed for m in result.mappings}
        assert statuses["head"] == "auto"
        assert statuses["unknown_mesh"] == "pending"


# ---------------------------------------------------------------------------
# CSV export / import round-trip
# ---------------------------------------------------------------------------


class TestCSV:
    """Test CSV export and import."""

    def test_round_trip(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "test_mappings.csv"

        model = map_model("model_001", ["head", "body", "unknown_part"])
        export_csv([model], csv_path)

        # Verify file exists and has correct structure
        assert csv_path.exists()
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        assert rows[0]["model_id"] == "model_001"
        assert rows[0]["fragment_name"] == "head"
        assert rows[0]["strata_label"] == "head"
        assert rows[0]["strata_region_id"] == "1"
        assert rows[0]["confirmed"] == "auto"

        # Unmapped row
        assert rows[2]["strata_label"] == "UNMAPPED"
        assert rows[2]["strata_region_id"] == "-1"
        assert rows[2]["confirmed"] == "pending"

        # Re-import
        loaded = load_csv(csv_path)
        assert len(loaded) == 1
        assert loaded[0].model_id == "model_001"
        assert loaded[0].total_count == 3

    def test_multiple_models(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "multi.csv"

        m1 = map_model("model_a", ["head", "neck"])
        m2 = map_model("model_b", ["body", "hand_L"])
        export_csv([m1, m2], csv_path)

        loaded = load_csv(csv_path)
        assert len(loaded) == 2
        ids = {m.model_id for m in loaded}
        assert ids == {"model_a", "model_b"}

    def test_append_mode(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "append.csv"

        m1 = map_model("model_a", ["head"])
        export_csv([m1], csv_path)

        m2 = map_model("model_b", ["neck"])
        export_csv([m2], csv_path, append=True)

        loaded = load_csv(csv_path)
        assert len(loaded) == 2

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "nested" / "dir" / "mappings.csv"
        m = map_model("test", ["head"])
        export_csv([m], csv_path)
        assert csv_path.exists()


# ---------------------------------------------------------------------------
# region_summary
# ---------------------------------------------------------------------------


class TestRegionSummary:
    """Test grouping fragments by region."""

    def test_groups_by_region(self) -> None:
        model = map_model("test", ["head", "face", "body", "unknown"])
        summary = region_summary(model)

        assert "head" in summary
        assert set(summary["head"]) == {"head", "face"}
        assert "chest" in summary
        assert summary["chest"] == ["body"]
        assert "UNMAPPED" in summary
        assert summary["UNMAPPED"] == ["unknown"]
