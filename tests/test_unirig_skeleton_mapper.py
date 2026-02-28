"""Tests for the UniRig skeleton mapper.

These tests exercise the pure-Python mapping logic without requiring
the actual UniRig dataset or Blender.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ingest.unirig_skeleton_mapper import (
    convert_directory,
    export_mapping_json,
    load_joint_names_json,
    load_joint_names_npz,
    load_mapping_json,
    map_joint_name,
    map_skeleton,
)

# ---------------------------------------------------------------------------
# map_joint_name
# ---------------------------------------------------------------------------


class TestMapJointName:
    """Test single joint name mapping."""

    def test_exact_mixamo_match(self) -> None:
        jm = map_joint_name("mixamorig:Hips")
        assert jm.region_id == 5
        assert jm.region_name == "hips"
        assert jm.method == "exact"

    def test_common_alias_match(self) -> None:
        jm = map_joint_name("Spine")
        assert jm.region_id == 4
        assert jm.region_name == "spine"
        assert jm.method == "alias"

    def test_vrm_alias_match(self) -> None:
        jm = map_joint_name("leftUpperArm")
        assert jm.region_id == 6
        assert jm.region_name == "upper_arm_l"
        assert jm.method == "vrm"

    def test_prefix_strip_match(self) -> None:
        jm = map_joint_name("mixamorig:Head")
        assert jm.region_id == 1
        assert jm.method == "exact"

    def test_substring_match(self) -> None:
        jm = map_joint_name("my_left_forearm_bone")
        assert jm.region_id is not None
        assert jm.method in ("substring", "fuzzy")

    def test_unmapped_bone(self) -> None:
        jm = map_joint_name("CompletelyUnknownXYZ123")
        assert jm.region_id is None
        assert jm.region_name is None
        assert jm.method == "unmapped"

    def test_all_strata_regions_reachable(self) -> None:
        """At least one bone name should reach each major Strata region."""
        vrm_bones = {
            "head": 1,
            "neck": 2,
            "upperChest": 3,
            "spine": 4,
            "hips": 5,
            "leftUpperArm": 6,
            "leftLowerArm": 7,
            "leftHand": 8,
            "rightUpperArm": 9,
            "rightLowerArm": 10,
            "rightHand": 11,
            "leftUpperLeg": 12,
            "leftLowerLeg": 13,
            "leftFoot": 14,
            "rightUpperLeg": 15,
            "rightLowerLeg": 16,
            "rightFoot": 17,
            "leftShoulder": 18,
            "rightShoulder": 19,
        }
        for bone_name, expected_id in vrm_bones.items():
            jm = map_joint_name(bone_name)
            assert jm.region_id == expected_id, (
                f"{bone_name} → {jm.region_id}, expected {expected_id}"
            )


# ---------------------------------------------------------------------------
# validate_skeleton
# ---------------------------------------------------------------------------


class TestValidateSkeleton:
    """Test skeleton validation."""

    def test_complete_humanoid(self) -> None:
        joint_names = [
            "head",
            "neck",
            "upperChest",
            "spine",
            "hips",
            "leftUpperArm",
            "leftLowerArm",
            "leftHand",
            "rightUpperArm",
            "rightLowerArm",
            "rightHand",
            "leftUpperLeg",
            "leftLowerLeg",
            "leftFoot",
            "rightUpperLeg",
            "rightLowerLeg",
            "rightFoot",
            "leftShoulder",
            "rightShoulder",
        ]
        mapping = map_skeleton("complete_char", joint_names)
        v = mapping.validation
        assert v.has_root is True
        assert v.has_head is True
        assert v.has_limbs is True
        assert v.has_symmetric_arms is True
        assert v.has_symmetric_legs is True
        assert v.missing_regions == []

    def test_missing_limbs(self) -> None:
        joint_names = ["head", "neck", "upperChest", "spine", "hips"]
        mapping = map_skeleton("torso_only", joint_names)
        v = mapping.validation
        assert v.has_root is True
        assert v.has_head is True
        assert v.has_limbs is False
        assert len(v.missing_regions) > 0

    def test_empty_skeleton(self) -> None:
        mapping = map_skeleton("empty", [])
        v = mapping.validation
        assert v.has_root is False
        assert v.has_head is False
        assert v.has_limbs is False


# ---------------------------------------------------------------------------
# map_skeleton
# ---------------------------------------------------------------------------


class TestMapSkeleton:
    """Test full skeleton mapping."""

    def test_basic_mapping(self) -> None:
        joint_names = ["head", "hips", "leftUpperArm", "CompletelyUnknownXYZ"]
        mapping = map_skeleton("test_char", joint_names)

        assert mapping.character_id == "test_char"
        assert mapping.total_joints == 4
        assert mapping.mapped_joints == 3
        assert len(mapping.unmapped_joints) == 1
        assert "CompletelyUnknownXYZ" in mapping.unmapped_joints

    def test_auto_match_rate(self) -> None:
        joint_names = ["head", "hips", "unknown_a", "unknown_b"]
        mapping = map_skeleton("rate_test", joint_names)
        assert mapping.auto_match_rate == 0.5

    def test_empty_skeleton(self) -> None:
        mapping = map_skeleton("empty", [])
        assert mapping.total_joints == 0
        assert mapping.mapped_joints == 0
        assert mapping.auto_match_rate == 0.0

    def test_region_coverage(self) -> None:
        joint_names = ["head", "neck", "head"]  # head mapped twice
        mapping = map_skeleton("dup_test", joint_names)
        coverage = mapping.region_coverage
        assert coverage["head"] == 2
        assert coverage["neck"] == 1

    def test_mixamo_skeleton_high_match_rate(self) -> None:
        """A typical Mixamo skeleton should have high match rate."""
        mixamo_names = [
            "mixamorig:Hips",
            "mixamorig:Spine",
            "mixamorig:Spine1",
            "mixamorig:Spine2",
            "mixamorig:Neck",
            "mixamorig:Head",
            "mixamorig:LeftShoulder",
            "mixamorig:LeftArm",
            "mixamorig:LeftForeArm",
            "mixamorig:LeftHand",
            "mixamorig:RightShoulder",
            "mixamorig:RightArm",
            "mixamorig:RightForeArm",
            "mixamorig:RightHand",
            "mixamorig:LeftUpLeg",
            "mixamorig:LeftLeg",
            "mixamorig:LeftFoot",
            "mixamorig:RightUpLeg",
            "mixamorig:RightLeg",
            "mixamorig:RightFoot",
        ]
        mapping = map_skeleton("mixamo_char", mixamo_names)
        assert mapping.auto_match_rate >= 0.9


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


class TestLoadJointNamesNpz:
    """Test NPZ loading."""

    def test_load_valid_npz(self, tmp_path: Path) -> None:
        npz_path = tmp_path / "char.npz"
        joint_names = np.array(["head", "neck", "hips"])
        np.savez(npz_path, joint_names=joint_names)

        result = load_joint_names_npz(npz_path)
        assert result == ["head", "neck", "hips"]

    def test_missing_key(self, tmp_path: Path) -> None:
        npz_path = tmp_path / "bad.npz"
        np.savez(npz_path, other_data=np.array([1, 2, 3]))

        result = load_joint_names_npz(npz_path)
        assert result is None

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        result = load_joint_names_npz(tmp_path / "missing.npz")
        assert result is None


class TestLoadJointNamesJson:
    """Test JSON loading."""

    def test_load_valid_json(self, tmp_path: Path) -> None:
        json_path = tmp_path / "char.json"
        data = {"joint_names": ["head", "neck", "spine"]}
        json_path.write_text(json.dumps(data), encoding="utf-8")

        result = load_joint_names_json(json_path)
        assert result == ["head", "neck", "spine"]

    def test_missing_key(self, tmp_path: Path) -> None:
        json_path = tmp_path / "bad.json"
        json_path.write_text('{"other": [1, 2]}', encoding="utf-8")

        result = load_joint_names_json(json_path)
        assert result is None

    def test_invalid_json(self, tmp_path: Path) -> None:
        json_path = tmp_path / "corrupt.json"
        json_path.write_text("not valid json{{{", encoding="utf-8")

        result = load_joint_names_json(json_path)
        assert result is None

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        result = load_joint_names_json(tmp_path / "missing.json")
        assert result is None


# ---------------------------------------------------------------------------
# JSON export / import
# ---------------------------------------------------------------------------


class TestJSONExport:
    """Test mapping JSON export and import."""

    def test_round_trip(self, tmp_path: Path) -> None:
        json_path = tmp_path / "mapping.json"
        mapping = map_skeleton("test_char", ["head", "hips", "leftUpperArm"])
        export_mapping_json(mapping, json_path)

        assert json_path.exists()
        loaded = load_mapping_json(json_path)
        assert loaded["character_id"] == "test_char"
        assert loaded["source"] == "unirig"
        assert loaded["total_joints"] == 3
        assert loaded["mapped_joints"] == 3
        assert loaded["auto_match_rate"] == 1.0

    def test_joint_mappings_in_json(self, tmp_path: Path) -> None:
        json_path = tmp_path / "mapping.json"
        mapping = map_skeleton("test", ["head", "unknown_xyz"])
        export_mapping_json(mapping, json_path)

        loaded = load_mapping_json(json_path)
        assert "head" in loaded["joint_mappings"]
        assert loaded["joint_mappings"]["head"]["region_id"] == 1
        assert "unknown_xyz" in loaded["joint_mappings"]
        assert loaded["joint_mappings"]["unknown_xyz"]["region_id"] is None

    def test_validation_in_json(self, tmp_path: Path) -> None:
        json_path = tmp_path / "mapping.json"
        mapping = map_skeleton("val_test", ["head", "hips"])
        export_mapping_json(mapping, json_path)

        loaded = load_mapping_json(json_path)
        assert loaded["validation"]["has_head"] is True
        assert loaded["validation"]["has_root"] is True
        assert loaded["validation"]["has_limbs"] is False

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        json_path = tmp_path / "nested" / "dir" / "mapping.json"
        mapping = map_skeleton("test", ["head"])
        export_mapping_json(mapping, json_path)
        assert json_path.exists()


# ---------------------------------------------------------------------------
# convert_directory
# ---------------------------------------------------------------------------


class TestConvertDirectory:
    """Test batch directory conversion."""

    def test_processes_npz_files(self, tmp_path: Path) -> None:
        unirig_dir = tmp_path / "unirig"
        unirig_dir.mkdir()
        output_dir = tmp_path / "output"

        # Create test npz files.
        for i in range(3):
            np.savez(
                unirig_dir / f"char_{i:03d}.npz",
                joint_names=np.array(["head", "hips", "leftUpperArm"]),
            )

        result = convert_directory(unirig_dir, output_dir)
        assert result.characters_processed == 3
        assert result.characters_good == 3  # All should be good (100% match).

    def test_processes_json_files(self, tmp_path: Path) -> None:
        unirig_dir = tmp_path / "unirig"
        unirig_dir.mkdir()
        output_dir = tmp_path / "output"

        data = {"joint_names": ["head", "neck", "hips"]}
        (unirig_dir / "char_001.json").write_text(
            json.dumps(data),
            encoding="utf-8",
        )

        result = convert_directory(unirig_dir, output_dir)
        assert result.characters_processed == 1

    def test_max_characters(self, tmp_path: Path) -> None:
        unirig_dir = tmp_path / "unirig"
        unirig_dir.mkdir()
        output_dir = tmp_path / "output"

        for i in range(5):
            np.savez(
                unirig_dir / f"char_{i:03d}.npz",
                joint_names=np.array(["head"]),
            )

        result = convert_directory(unirig_dir, output_dir, max_characters=2)
        assert result.characters_processed == 2

    def test_only_new_skips_existing(self, tmp_path: Path) -> None:
        unirig_dir = tmp_path / "unirig"
        unirig_dir.mkdir()
        output_dir = tmp_path / "output"

        np.savez(
            unirig_dir / "char_001.npz",
            joint_names=np.array(["head"]),
        )

        result1 = convert_directory(unirig_dir, output_dir)
        assert result1.characters_processed == 1

        result2 = convert_directory(unirig_dir, output_dir, only_new=True)
        assert result2.characters_processed == 0

    def test_handles_corrupt_files(self, tmp_path: Path) -> None:
        unirig_dir = tmp_path / "unirig"
        unirig_dir.mkdir()
        output_dir = tmp_path / "output"

        (unirig_dir / "corrupt.json").write_text("not json{{{", encoding="utf-8")
        np.savez(
            unirig_dir / "good.npz",
            joint_names=np.array(["head"]),
        )

        result = convert_directory(unirig_dir, output_dir)
        assert result.characters_processed == 1
        assert len(result.errors) == 1

    def test_empty_directory(self, tmp_path: Path) -> None:
        unirig_dir = tmp_path / "unirig"
        unirig_dir.mkdir()
        result = convert_directory(unirig_dir, tmp_path / "output")
        assert result.characters_processed == 0

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        result = convert_directory(tmp_path / "nonexistent", tmp_path / "output")
        assert result.characters_processed == 0
        assert len(result.errors) == 1

    def test_poor_match_rate_tracked(self, tmp_path: Path) -> None:
        unirig_dir = tmp_path / "unirig"
        unirig_dir.mkdir()
        output_dir = tmp_path / "output"

        # Mostly unmapped bones → poor match rate.
        names = ["head"] + [f"unknown_{i}" for i in range(20)]
        np.savez(unirig_dir / "poor.npz", joint_names=np.array(names))

        result = convert_directory(unirig_dir, output_dir)
        assert result.characters_poor == 1
