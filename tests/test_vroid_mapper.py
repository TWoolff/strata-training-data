"""Tests for the VRoid material-to-Strata label mapper.

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

from pipeline.vroid_mapper import (  # noqa: E402
    disambiguate_lr,
    export_csv,
    load_csv,
    map_material,
    map_model,
    region_summary,
)

# ---------------------------------------------------------------------------
# map_material — individual material mapping
# ---------------------------------------------------------------------------


class TestMapMaterial:
    """Test single-material mapping against VROID_MATERIAL_PATTERNS."""

    # --- Head region (1) ---

    @pytest.mark.parametrize(
        "material_name",
        [
            "Face",
            "Face_",
            "face",
            "EyeWhite",
            "EyeIris",
            "EyeHighlight",
            "EyeExtra",
            "Eyebrow",
            "eyebrow",
            "Hair",
            "hair",
            "Hair_Front",
            "Hair_Back",
            "bangs",
            "ponytail",
            "ahoge",
            "Mouth",
            "mouth",
            "Tooth",
            "tongue",
            "Ear",
            "Nose",
            "Head",
            "head",
        ],
    )
    def test_head_materials(self, material_name: str) -> None:
        label, region_id = map_material(material_name)
        assert label == "head", f"{material_name!r} → {label}, expected 'head'"
        assert region_id == 1

    # --- Neck region (2) ---

    @pytest.mark.parametrize("material_name", ["neck", "Neck"])
    def test_neck_materials(self, material_name: str) -> None:
        label, region_id = map_material(material_name)
        assert label == "neck"
        assert region_id == 2

    # --- Chest / torso region (3) ---

    @pytest.mark.parametrize(
        "material_name",
        [
            "Body",
            "body",
            "torso",
            "chest",
            "mune",
            "bust",
            "Outfit_Upper",
            "jacket",
            "shirt",
            "vest",
            "coat",
            "blazer",
            "uniform",
        ],
    )
    def test_chest_materials(self, material_name: str) -> None:
        label, region_id = map_material(material_name)
        assert label == "chest"
        assert region_id == 3

    # --- Hips region (5) ---

    @pytest.mark.parametrize(
        "material_name",
        ["hip", "hips", "pelvis", "waist", "Outfit_Lower", "pants", "skirt", "shorts"],
    )
    def test_hips_materials(self, material_name: str) -> None:
        label, region_id = map_material(material_name)
        assert label == "hips"
        assert region_id == 5

    # --- Left arm regions ---

    @pytest.mark.parametrize(
        "material_name, expected_label, expected_id",
        [
            ("upper_arm_L", "upper_arm_l", 7),
            ("arm_left", "upper_arm_l", 7),
            ("L_arm", "upper_arm_l", 7),
            ("forearm_L", "forearm_l", 8),
            ("forearm_left", "forearm_l", 8),
            ("L_forearm", "forearm_l", 8),
            ("hand_L", "hand_l", 9),
            ("hand_left", "hand_l", 9),
            ("glove_L", "hand_l", 9),
            ("finger_left", "hand_l", 9),
        ],
    )
    def test_left_arm_materials(
        self, material_name: str, expected_label: str, expected_id: int
    ) -> None:
        label, region_id = map_material(material_name)
        assert label == expected_label, f"{material_name!r} → {label}, expected {expected_label!r}"
        assert region_id == expected_id

    # --- Right arm regions ---

    @pytest.mark.parametrize(
        "material_name, expected_label, expected_id",
        [
            ("upper_arm_R", "upper_arm_r", 11),
            ("arm_right", "upper_arm_r", 11),
            ("R_arm", "upper_arm_r", 11),
            ("forearm_R", "forearm_r", 12),
            ("forearm_right", "forearm_r", 12),
            ("R_forearm", "forearm_r", 12),
            ("hand_R", "hand_r", 13),
            ("hand_right", "hand_r", 13),
            ("glove_R", "hand_r", 13),
            ("finger_right", "hand_r", 13),
        ],
    )
    def test_right_arm_materials(
        self, material_name: str, expected_label: str, expected_id: int
    ) -> None:
        label, region_id = map_material(material_name)
        assert label == expected_label, f"{material_name!r} → {label}, expected {expected_label!r}"
        assert region_id == expected_id

    # --- Left leg regions ---

    @pytest.mark.parametrize(
        "material_name, expected_label, expected_id",
        [
            ("thigh_L", "upper_leg_l", 14),
            ("upper_leg_left", "upper_leg_l", 14),
            ("leg_left", "upper_leg_l", 14),
            ("shin_L", "lower_leg_l", 15),
            ("lower_leg_left", "lower_leg_l", 15),
            ("calf_left", "lower_leg_l", 15),
            ("foot_L", "foot_l", 16),
            ("foot_left", "foot_l", 16),
            ("shoe_L", "foot_l", 16),
            ("boot_left", "foot_l", 16),
            ("sock_L", "foot_l", 16),
            ("toe_left", "foot_l", 16),
        ],
    )
    def test_left_leg_materials(
        self, material_name: str, expected_label: str, expected_id: int
    ) -> None:
        label, region_id = map_material(material_name)
        assert label == expected_label, f"{material_name!r} → {label}, expected {expected_label!r}"
        assert region_id == expected_id

    # --- Right leg regions ---

    @pytest.mark.parametrize(
        "material_name, expected_label, expected_id",
        [
            ("thigh_R", "upper_leg_r", 17),
            ("upper_leg_right", "upper_leg_r", 17),
            ("leg_right", "upper_leg_r", 17),
            ("shin_R", "lower_leg_r", 18),
            ("lower_leg_right", "lower_leg_r", 18),
            ("calf_right", "lower_leg_r", 18),
            ("foot_R", "foot_r", 19),
            ("foot_right", "foot_r", 19),
            ("shoe_R", "foot_r", 19),
            ("boot_right", "foot_r", 19),
            ("sock_R", "foot_r", 19),
            ("toe_right", "foot_r", 19),
        ],
    )
    def test_right_leg_materials(
        self, material_name: str, expected_label: str, expected_id: int
    ) -> None:
        label, region_id = map_material(material_name)
        assert label == expected_label, f"{material_name!r} → {label}, expected {expected_label!r}"
        assert region_id == expected_id

    # --- Shoulder regions ---

    @pytest.mark.parametrize(
        "material_name, expected_label, expected_id",
        [
            ("shoulder_L", "shoulder_l", 6),
            ("shoulder_left", "shoulder_l", 6),
            ("shoulder_R", "shoulder_r", 10),
            ("shoulder_right", "shoulder_r", 10),
        ],
    )
    def test_shoulder_materials(
        self, material_name: str, expected_label: str, expected_id: int
    ) -> None:
        label, region_id = map_material(material_name)
        assert label == expected_label, f"{material_name!r} → {label}, expected {expected_label!r}"
        assert region_id == expected_id

    # --- Accessories → background (0) ---

    @pytest.mark.parametrize(
        "material_name",
        [
            "accessory",
            "ribbon",
            "wing",
            "tail",
            "cape",
            "weapon",
            "shield",
            "bag",
            "ornament",
            "jewelry",
            "crown",
            "hat",
            "glasses",
            "belt",
            "scarf",
        ],
    )
    def test_accessory_materials(self, material_name: str) -> None:
        label, region_id = map_material(material_name)
        assert label == "background", f"{material_name!r} → {label}, expected 'background'"
        assert region_id == 0

    # --- Unmapped ---

    @pytest.mark.parametrize(
        "material_name",
        [
            "Material_001",
            "custom_shader_42",
            "effect_glow",
            "shadow_plane",
        ],
    )
    def test_unmapped_materials(self, material_name: str) -> None:
        label, region_id = map_material(material_name)
        assert label == "UNMAPPED"
        assert region_id == -1

    # --- Case insensitivity ---

    def test_case_insensitive(self) -> None:
        label1, _ = map_material("FACE")
        label2, _ = map_material("face")
        label3, _ = map_material("Face")
        assert label1 == label2 == label3 == "head"


# ---------------------------------------------------------------------------
# disambiguate_lr — L/R resolution via vertex centroid
# ---------------------------------------------------------------------------


class TestDisambiguateLR:
    """Test L/R disambiguation using vertex world-space X position."""

    def test_left_stays_left(self) -> None:
        """Negative X → left side (no flip)."""
        label, region_id = disambiguate_lr("Shoe_L", "foot_l", -0.5)
        assert label == "foot_l"
        assert region_id == 16

    def test_left_flips_to_right(self) -> None:
        """Positive X → flip from left to right."""
        label, region_id = disambiguate_lr("Shoe", "foot_l", 0.5)
        assert label == "foot_r"
        assert region_id == 19

    def test_right_stays_right(self) -> None:
        """Positive X → right side (no flip)."""
        label, region_id = disambiguate_lr("Glove_R", "hand_r", 0.5)
        assert label == "hand_r"
        assert region_id == 13

    def test_right_flips_to_left(self) -> None:
        """Negative X → flip from right to left."""
        label, region_id = disambiguate_lr("Glove", "hand_r", -0.5)
        assert label == "hand_l"
        assert region_id == 9

    def test_non_symmetric_region(self) -> None:
        """Non-symmetric regions (head, chest, etc.) are unchanged."""
        label, region_id = disambiguate_lr("Head_mat", "head", 0.3)
        assert label == "head"
        assert region_id == 1

    def test_zero_centroid(self) -> None:
        """X=0 keeps the original mapping (no flip)."""
        label, region_id = disambiguate_lr("Shoe", "foot_l", 0.0)
        assert label == "foot_l"
        assert region_id == 16

    def test_all_lr_pairs(self) -> None:
        """All symmetric pairs can be flipped both directions."""
        pairs = [
            ("shoulder_l", "shoulder_r", 6, 10),
            ("upper_arm_l", "upper_arm_r", 7, 11),
            ("forearm_l", "forearm_r", 8, 12),
            ("hand_l", "hand_r", 9, 13),
            ("upper_leg_l", "upper_leg_r", 14, 17),
            ("lower_leg_l", "lower_leg_r", 15, 18),
            ("foot_l", "foot_r", 16, 19),
        ]
        for left, right, left_id, right_id in pairs:
            # Left → Right (positive X)
            label, rid = disambiguate_lr("test", left, 1.0)
            assert label == right, f"{left} + X>0 → {label}, expected {right}"
            assert rid == right_id

            # Right → Left (negative X)
            label, rid = disambiguate_lr("test", right, -1.0)
            assert label == left, f"{right} + X<0 → {label}, expected {left}"
            assert rid == left_id


# ---------------------------------------------------------------------------
# map_model — full model mapping
# ---------------------------------------------------------------------------


class TestMapModel:
    """Test mapping an entire model's materials."""

    def test_basic_model(self) -> None:
        materials = ["Face", "Body", "Hair", "custom_effect"]
        result = map_model("vroid_001", materials)

        assert result.model_id == "vroid_001"
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
        materials = ["Face", "Neck", "Body", "hand_L", "hand_R"]
        result = map_model("full_model", materials)
        assert result.unmapped_count == 0
        assert result.auto_rate == 1.0

    def test_confirmed_status(self) -> None:
        materials = ["Face", "custom_shader"]
        result = map_model("status_test", materials)
        statuses = {m.material_name: m.confirmed for m in result.mappings}
        assert statuses["Face"] == "auto"
        assert statuses["custom_shader"] == "pending"

    def test_standard_vroid_materials(self) -> None:
        """Standard VRoid Studio material slots should all map."""
        standard_materials = [
            "Face",
            "EyeWhite",
            "EyeIris",
            "EyeHighlight",
            "Eyebrow",
            "Hair",
            "Body",
            "Outfit_Upper",
            "Outfit_Lower",
        ]
        result = map_model("vroid_standard", standard_materials)
        assert result.unmapped_count == 0, (
            f"Unmapped: {[m.material_name for m in result.mappings if m.strata_region_id < 0]}"
        )


# ---------------------------------------------------------------------------
# CSV export / import round-trip
# ---------------------------------------------------------------------------


class TestCSV:
    """Test CSV export and import."""

    def test_round_trip(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "test_mappings.csv"

        model = map_model("vroid_001", ["Face", "Body", "custom_effect"])
        export_csv([model], csv_path)

        # Verify file exists and has correct structure
        assert csv_path.exists()
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        assert rows[0]["model_id"] == "vroid_001"
        assert rows[0]["material_name"] == "Face"
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
        assert loaded[0].model_id == "vroid_001"
        assert loaded[0].total_count == 3

    def test_multiple_models(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "multi.csv"

        m1 = map_model("vroid_a", ["Face", "Neck"])
        m2 = map_model("vroid_b", ["Body", "hand_L"])
        export_csv([m1, m2], csv_path)

        loaded = load_csv(csv_path)
        assert len(loaded) == 2
        ids = {m.model_id for m in loaded}
        assert ids == {"vroid_a", "vroid_b"}

    def test_append_mode(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "append.csv"

        m1 = map_model("vroid_a", ["Face"])
        export_csv([m1], csv_path)

        m2 = map_model("vroid_b", ["Neck"])
        export_csv([m2], csv_path, append=True)

        loaded = load_csv(csv_path)
        assert len(loaded) == 2

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "nested" / "dir" / "mappings.csv"
        m = map_model("test", ["Face"])
        export_csv([m], csv_path)
        assert csv_path.exists()


# ---------------------------------------------------------------------------
# region_summary
# ---------------------------------------------------------------------------


class TestRegionSummary:
    """Test grouping materials by region."""

    def test_groups_by_region(self) -> None:
        model = map_model("test", ["Face", "Hair", "Body", "custom_shader"])
        summary = region_summary(model)

        assert "head" in summary
        assert set(summary["head"]) == {"Face", "Hair"}
        assert "chest" in summary
        assert summary["chest"] == ["Body"]
        assert "UNMAPPED" in summary
        assert summary["UNMAPPED"] == ["custom_shader"]
