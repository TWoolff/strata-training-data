"""Tests for the PSD layer extractor.

These tests exercise the pure-Python mapping logic without requiring Blender
or psd-tools (the PSD processing functions are tested with mocks).
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

from pipeline.psd_extractor import (  # noqa: E402
    export_csv,
    load_csv,
    map_layer,
    map_psd,
    region_summary,
)

# ---------------------------------------------------------------------------
# map_layer — individual layer mapping
# ---------------------------------------------------------------------------


class TestMapLayer:
    """Test single-layer mapping against PSD_LAYER_PATTERNS."""

    # --- Head region (1) ---

    @pytest.mark.parametrize(
        "layer_name",
        [
            "head",
            "Head",
            "face",
            "hair",
            "hair_front",
            "bangs",
            "ponytail",
            "eye",
            "eye_left",
            "iris",
            "eyelash",
            "mouth",
            "lip",
            "teeth",
            "nose",
            "ear",
            "brow",
            "eyebrow",
            "cheek",
            "chin",
            "forehead",
            "fringe",
            "braid",
            "ahoge",
        ],
    )
    def test_head_layers(self, layer_name: str) -> None:
        label, region_id = map_layer(layer_name)
        assert label == "head", f"{layer_name!r} -> {label}, expected 'head'"
        assert region_id == 1

    # --- Neck region (2) ---

    @pytest.mark.parametrize("layer_name", ["neck", "Neck", "NECK"])
    def test_neck_layers(self, layer_name: str) -> None:
        label, region_id = map_layer(layer_name)
        assert label == "neck"
        assert region_id == 2

    # --- Chest region (3) ---

    @pytest.mark.parametrize(
        "layer_name",
        ["torso", "chest", "upper_body", "breast", "bust", "body"],
    )
    def test_chest_layers(self, layer_name: str) -> None:
        label, region_id = map_layer(layer_name)
        assert label == "chest"
        assert region_id == 3

    # --- Hips region (5) ---

    @pytest.mark.parametrize("layer_name", ["hip", "hips", "pelvis", "waist"])
    def test_hips_layers(self, layer_name: str) -> None:
        label, region_id = map_layer(layer_name)
        assert label == "hips"
        assert region_id == 5

    # --- Left arm regions ---

    @pytest.mark.parametrize(
        "layer_name, expected_label, expected_id",
        [
            ("upper_arm_L", "upper_arm_l", 6),
            ("arm_left", "upper_arm_l", 6),
            ("L_arm", "upper_arm_l", 6),
            ("forearm_L", "lower_arm_l", 7),
            ("lower_arm_left", "lower_arm_l", 7),
            ("hand_L", "hand_l", 8),
            ("hand_left", "hand_l", 8),
            ("finger_left", "hand_l", 8),
        ],
    )
    def test_left_arm_layers(self, layer_name: str, expected_label: str, expected_id: int) -> None:
        label, region_id = map_layer(layer_name)
        assert label == expected_label, f"{layer_name!r} -> {label}, expected {expected_label!r}"
        assert region_id == expected_id

    # --- Right arm regions ---

    @pytest.mark.parametrize(
        "layer_name, expected_label, expected_id",
        [
            ("upper_arm_R", "upper_arm_r", 9),
            ("arm_right", "upper_arm_r", 9),
            ("R_arm", "upper_arm_r", 9),
            ("forearm_R", "lower_arm_r", 10),
            ("lower_arm_right", "lower_arm_r", 10),
            ("hand_R", "hand_r", 11),
            ("hand_right", "hand_r", 11),
            ("finger_right", "hand_r", 11),
        ],
    )
    def test_right_arm_layers(self, layer_name: str, expected_label: str, expected_id: int) -> None:
        label, region_id = map_layer(layer_name)
        assert label == expected_label, f"{layer_name!r} -> {label}, expected {expected_label!r}"
        assert region_id == expected_id

    # --- Left leg regions ---

    @pytest.mark.parametrize(
        "layer_name, expected_label, expected_id",
        [
            ("thigh_L", "upper_leg_l", 12),
            ("upper_leg_left", "upper_leg_l", 12),
            ("leg_left", "upper_leg_l", 12),
            ("shin_L", "lower_leg_l", 13),
            ("lower_leg_left", "lower_leg_l", 13),
            ("calf_left", "lower_leg_l", 13),
            ("foot_L", "foot_l", 14),
            ("foot_left", "foot_l", 14),
            ("shoe_left", "foot_l", 14),
        ],
    )
    def test_left_leg_layers(self, layer_name: str, expected_label: str, expected_id: int) -> None:
        label, region_id = map_layer(layer_name)
        assert label == expected_label, f"{layer_name!r} -> {label}, expected {expected_label!r}"
        assert region_id == expected_id

    # --- Right leg regions ---

    @pytest.mark.parametrize(
        "layer_name, expected_label, expected_id",
        [
            ("thigh_R", "upper_leg_r", 15),
            ("upper_leg_right", "upper_leg_r", 15),
            ("leg_right", "upper_leg_r", 15),
            ("shin_R", "lower_leg_r", 16),
            ("lower_leg_right", "lower_leg_r", 16),
            ("calf_right", "lower_leg_r", 16),
            ("foot_R", "foot_r", 17),
            ("foot_right", "foot_r", 17),
            ("boot_right", "foot_r", 17),
        ],
    )
    def test_right_leg_layers(self, layer_name: str, expected_label: str, expected_id: int) -> None:
        label, region_id = map_layer(layer_name)
        assert label == expected_label, f"{layer_name!r} -> {label}, expected {expected_label!r}"
        assert region_id == expected_id

    # --- Shoulder regions ---

    @pytest.mark.parametrize(
        "layer_name, expected_label, expected_id",
        [
            ("shoulder_L", "shoulder_l", 18),
            ("shoulder_left", "shoulder_l", 18),
            ("shoulder_R", "shoulder_r", 19),
            ("shoulder_right", "shoulder_r", 19),
        ],
    )
    def test_shoulder_layers(self, layer_name: str, expected_label: str, expected_id: int) -> None:
        label, region_id = map_layer(layer_name)
        assert label == expected_label, f"{layer_name!r} -> {label}, expected {expected_label!r}"
        assert region_id == expected_id

    # --- Accessories / rendering concerns → background (0) ---

    @pytest.mark.parametrize(
        "layer_name",
        [
            "weapon",
            "shield",
            "cape",
            "hat",
            "ribbon",
            "accessory_01",
            "wing",
            "tail",
            "glasses",
            "lineart",
            "line_art",
            "outline",
            "shadow",
            "shading",
            "highlight",
            "flat_color",
            "base_color",
            "background",
            "bg",
            "effect",
            "fx",
            "glow",
        ],
    )
    def test_background_layers(self, layer_name: str) -> None:
        label, region_id = map_layer(layer_name)
        assert label == "background", f"{layer_name!r} -> {label}, expected 'background'"
        assert region_id == 0

    # --- Unmapped ---

    @pytest.mark.parametrize(
        "layer_name",
        [
            "Layer 1",
            "Group 5",
            "copy of something",
            "merged_final",
        ],
    )
    def test_unmapped_layers(self, layer_name: str) -> None:
        label, region_id = map_layer(layer_name)
        assert label == "UNMAPPED"
        assert region_id == -1

    # --- Case insensitivity ---

    def test_case_insensitive(self) -> None:
        label1, _ = map_layer("HEAD")
        label2, _ = map_layer("head")
        label3, _ = map_layer("Head")
        assert label1 == label2 == label3 == "head"


# ---------------------------------------------------------------------------
# map_psd — full PSD mapping
# ---------------------------------------------------------------------------


class TestMapPSD:
    """Test mapping an entire PSD's layers."""

    def test_basic_mapping(self) -> None:
        layers = ["head", "body", "arm_left", "unknown_part"]
        result = map_psd("test_001", layers)

        assert result.psd_id == "test_001"
        assert result.total_count == 4
        assert result.mapped_count == 3
        assert result.unmapped_count == 1
        assert result.auto_rate == pytest.approx(0.75)

    def test_empty_psd(self) -> None:
        result = map_psd("empty", [])
        assert result.total_count == 0
        assert result.mapped_count == 0
        assert result.auto_rate == 0.0

    def test_all_mapped(self) -> None:
        layers = ["head", "neck", "body", "hand_L", "hand_R"]
        result = map_psd("full", layers)
        assert result.unmapped_count == 0
        assert result.auto_rate == 1.0

    def test_confirmed_status(self) -> None:
        layers = ["head", "unknown_mesh"]
        result = map_psd("status_test", layers)
        statuses = {m.layer_name: m.confirmed for m in result.mappings}
        assert statuses["head"] == "auto"
        assert statuses["unknown_mesh"] == "pending"

    def test_group_paths(self) -> None:
        layers = ["head", "arm_left"]
        paths = ["", "body/arms"]
        result = map_psd("test", layers, group_paths=paths)
        assert result.mappings[0].group_path == ""
        assert result.mappings[1].group_path == "body/arms"

    def test_visibility(self) -> None:
        layers = ["head", "arm_left"]
        vis = [True, False]
        result = map_psd("test", layers, visibilities=vis)
        assert result.mappings[0].is_visible is True
        assert result.mappings[1].is_visible is False

    def test_groups_excluded_from_counts(self) -> None:
        """Group layers should not count toward mapped/unmapped totals."""
        layers = ["head", "body_group", "arm_left"]
        is_groups = [False, True, False]
        result = map_psd("test", layers, is_groups=is_groups)
        assert result.total_count == 2  # Only leaf layers
        assert result.mapped_count == 2

    def test_background_not_counted_as_body_mapped(self) -> None:
        """Layers mapped to background (0) don't count toward mapped_count."""
        layers = ["head", "weapon", "unknown"]
        result = map_psd("test", layers)
        # head=1 (body), weapon=0 (bg), unknown=-1 (unmapped)
        assert result.mapped_count == 1  # Only head
        assert result.unmapped_count == 1  # unknown


# ---------------------------------------------------------------------------
# CSV export / import round-trip
# ---------------------------------------------------------------------------


class TestCSV:
    """Test CSV export and import."""

    def test_round_trip(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "test_mappings.csv"

        psd = map_psd("psd_001", ["head", "body", "unknown_part"])
        export_csv([psd], csv_path)

        assert csv_path.exists()
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        assert rows[0]["psd_id"] == "psd_001"
        assert rows[0]["layer_name"] == "head"
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
        assert loaded[0].psd_id == "psd_001"
        assert loaded[0].total_count == 3

    def test_multiple_psds(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "multi.csv"

        m1 = map_psd("psd_a", ["head", "neck"])
        m2 = map_psd("psd_b", ["body", "hand_L"])
        export_csv([m1, m2], csv_path)

        loaded = load_csv(csv_path)
        assert len(loaded) == 2
        ids = {m.psd_id for m in loaded}
        assert ids == {"psd_a", "psd_b"}

    def test_append_mode(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "append.csv"

        m1 = map_psd("psd_a", ["head"])
        export_csv([m1], csv_path)

        m2 = map_psd("psd_b", ["neck"])
        export_csv([m2], csv_path, append=True)

        loaded = load_csv(csv_path)
        assert len(loaded) == 2

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "nested" / "dir" / "mappings.csv"
        m = map_psd("test", ["head"])
        export_csv([m], csv_path)
        assert csv_path.exists()

    def test_groups_excluded_from_csv(self, tmp_path: Path) -> None:
        """Group layers should not appear in CSV output."""
        csv_path = tmp_path / "groups.csv"

        layers = ["head", "body_group", "arm_left"]
        is_groups = [False, True, False]
        psd = map_psd("test", layers, is_groups=is_groups)
        export_csv([psd], csv_path)

        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2  # Groups excluded
        names = [r["layer_name"] for r in rows]
        assert "body_group" not in names


# ---------------------------------------------------------------------------
# region_summary
# ---------------------------------------------------------------------------


class TestRegionSummary:
    """Test grouping layers by region."""

    def test_groups_by_region(self) -> None:
        psd = map_psd("test", ["head", "face", "body", "unknown"])
        summary = region_summary(psd)

        assert "head" in summary
        assert set(summary["head"]) == {"head", "face"}
        assert "chest" in summary
        assert summary["chest"] == ["body"]
        assert "UNMAPPED" in summary
        assert summary["UNMAPPED"] == ["unknown"]

    def test_excludes_groups(self) -> None:
        layers = ["head", "group1"]
        is_groups = [False, True]
        psd = map_psd("test", layers, is_groups=is_groups)
        summary = region_summary(psd)

        all_names = [name for names in summary.values() for name in names]
        assert "group1" not in all_names
