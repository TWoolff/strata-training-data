"""Tests for the StdGEN semantic mapper.

These tests exercise the pure-Python mapping logic that converts StdGEN's
4-class semantic annotations (body, clothes, hair, face) to Strata's
20-class taxonomy using mock bone weight data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ingest.stdgen_semantic_mapper import (
    FALLBACK_REGION_ID,
    export_mapping_json,
    load_mapping_json,
    map_character,
    map_mesh,
    map_vertex,
    refine_segmentation_mask,
    resolve_region_from_weights,
)

# ---------------------------------------------------------------------------
# resolve_region_from_weights
# ---------------------------------------------------------------------------


class TestResolveRegionFromWeights:
    """Test bone-weight → region resolution."""

    def test_single_bone(self) -> None:
        label, region_id = resolve_region_from_weights({"head": 1.0})
        assert label == "head"
        assert region_id == 1

    def test_dominant_bone_wins(self) -> None:
        weights = {"leftUpperArm": 0.7, "leftLowerArm": 0.3}
        label, region_id = resolve_region_from_weights(weights)
        assert label == "upper_arm_l"
        assert region_id == 6

    def test_chest_bone(self) -> None:
        label, region_id = resolve_region_from_weights({"upperChest": 0.9})
        assert label == "chest"
        assert region_id == 3

    def test_hips_bone(self) -> None:
        label, region_id = resolve_region_from_weights({"hips": 1.0})
        assert label == "hips"
        assert region_id == 5

    def test_right_hand(self) -> None:
        weights = {"rightHand": 0.6, "rightLowerArm": 0.4}
        label, region_id = resolve_region_from_weights(weights)
        assert label == "hand_r"
        assert region_id == 11

    def test_left_foot(self) -> None:
        label, region_id = resolve_region_from_weights({"leftFoot": 1.0})
        assert label == "foot_l"
        assert region_id == 14

    def test_right_shoulder(self) -> None:
        label, region_id = resolve_region_from_weights({"rightShoulder": 1.0})
        assert label == "shoulder_r"
        assert region_id == 19

    def test_finger_maps_to_hand(self) -> None:
        weights = {"leftIndexProximal": 0.8, "leftHand": 0.2}
        label, region_id = resolve_region_from_weights(weights)
        assert label == "hand_l"
        assert region_id == 8

    def test_empty_weights_returns_fallback(self) -> None:
        _label, region_id = resolve_region_from_weights({})
        assert region_id == FALLBACK_REGION_ID

    def test_unknown_bones_return_fallback(self) -> None:
        weights = {"UnknownBone_A": 0.5, "UnknownBone_B": 0.5}
        _label, region_id = resolve_region_from_weights(weights)
        assert region_id == FALLBACK_REGION_ID

    def test_mixed_known_unknown_bones(self) -> None:
        weights = {"UnknownBone": 0.9, "neck": 0.1}
        label, region_id = resolve_region_from_weights(weights)
        assert label == "neck"
        assert region_id == 2

    def test_all_vrm_body_regions(self) -> None:
        """Every major VRM bone should resolve to a valid region."""
        vrm_bones = {
            "head": 1,
            "neck": 2,
            "upperChest": 3,
            "chest": 3,
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
        for bone, expected_id in vrm_bones.items():
            _, region_id = resolve_region_from_weights({bone: 1.0})
            assert region_id == expected_id, (
                f"{bone} → region {region_id}, expected {expected_id}"
            )


# ---------------------------------------------------------------------------
# map_vertex
# ---------------------------------------------------------------------------


class TestMapVertex:
    """Test single-vertex mapping from StdGEN class to Strata region."""

    def test_hair_maps_to_head(self) -> None:
        label, region_id = map_vertex("hair", {})
        assert label == "head"
        assert region_id == 1

    def test_face_maps_to_head(self) -> None:
        label, region_id = map_vertex("face", {})
        assert label == "head"
        assert region_id == 1

    def test_hair_ignores_bone_weights(self) -> None:
        label, region_id = map_vertex("hair", {"leftFoot": 1.0})
        assert label == "head"
        assert region_id == 1

    def test_face_ignores_bone_weights(self) -> None:
        label, region_id = map_vertex("face", {"rightHand": 1.0})
        assert label == "head"
        assert region_id == 1

    def test_body_uses_bone_weights(self) -> None:
        label, region_id = map_vertex("body", {"leftUpperArm": 0.9})
        assert label == "upper_arm_l"
        assert region_id == 6

    def test_clothes_uses_bone_weights(self) -> None:
        label, region_id = map_vertex("clothes", {"hips": 0.8})
        assert label == "hips"
        assert region_id == 5

    def test_body_no_weights_returns_fallback(self) -> None:
        _label, region_id = map_vertex("body", {})
        assert region_id == FALLBACK_REGION_ID

    def test_clothes_no_weights_returns_fallback(self) -> None:
        _label, region_id = map_vertex("clothes", {})
        assert region_id == FALLBACK_REGION_ID

    def test_invalid_class_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown StdGEN class"):
            map_vertex("invalid_class", {})

    def test_unknown_class_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown StdGEN class"):
            map_vertex("skin", {"head": 1.0})


# ---------------------------------------------------------------------------
# map_mesh
# ---------------------------------------------------------------------------


class TestMapMesh:
    """Test whole-mesh mapping."""

    def test_basic_mesh(self) -> None:
        vertex_classes = ["hair", "face", "body", "clothes"]
        vertex_weights = [
            {},
            {},
            {"leftUpperArm": 0.9},
            {"hips": 0.8, "spine": 0.2},
        ]
        result = map_mesh("Body_Mesh", vertex_classes, vertex_weights)

        assert result.mesh_name == "Body_Mesh"
        assert result.total_count == 4
        assert result.vertex_mappings[0].strata_label == "head"
        assert result.vertex_mappings[1].strata_label == "head"
        assert result.vertex_mappings[2].strata_label == "upper_arm_l"
        assert result.vertex_mappings[3].strata_label == "hips"

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="Mismatched lengths"):
            map_mesh("test", ["hair", "face"], [{}])

    def test_empty_mesh(self) -> None:
        result = map_mesh("empty", [], [])
        assert result.total_count == 0
        assert result.region_distribution == {}

    def test_region_distribution(self) -> None:
        vertex_classes = ["hair", "hair", "body", "body", "face"]
        vertex_weights = [
            {},
            {},
            {"leftFoot": 1.0},
            {"leftFoot": 1.0},
            {},
        ]
        result = map_mesh("test", vertex_classes, vertex_weights)
        dist = result.region_distribution
        assert dist["head"] == 3  # 2 hair + 1 face
        assert dist["foot_l"] == 2

    def test_preserves_vertex_indices(self) -> None:
        vertex_classes = ["body", "body", "body"]
        vertex_weights = [
            {"head": 1.0},
            {"neck": 1.0},
            {"hips": 1.0},
        ]
        result = map_mesh("test", vertex_classes, vertex_weights)
        for i, vm in enumerate(result.vertex_mappings):
            assert vm.vertex_index == i


# ---------------------------------------------------------------------------
# map_character
# ---------------------------------------------------------------------------


class TestMapCharacter:
    """Test full character mapping."""

    def test_single_mesh(self) -> None:
        meshes = [(
            "Body",
            ["hair", "face", "body"],
            [{}, {}, {"head": 1.0}],
        )]
        result = map_character("char_001", meshes)

        assert result.character_id == "char_001"
        assert result.total_vertices == 3
        assert len(result.mesh_mappings) == 1

    def test_multiple_meshes(self) -> None:
        meshes = [
            ("Body", ["body"], [{"hips": 1.0}]),
            ("Hair", ["hair"], [{}]),
            ("Face", ["face"], [{}]),
        ]
        result = map_character("char_002", meshes)

        assert result.total_vertices == 3
        assert len(result.mesh_mappings) == 3

    def test_region_distribution_aggregates(self) -> None:
        meshes = [
            ("Body", ["body", "body"], [{"hips": 1.0}, {"hips": 1.0}]),
            ("Hair", ["hair", "hair"], [{}, {}]),
        ]
        result = map_character("char_003", meshes)
        dist = result.region_distribution
        assert dist["hips"] == 2
        assert dist["head"] == 2


# ---------------------------------------------------------------------------
# refine_segmentation_mask
# ---------------------------------------------------------------------------


class TestRefineSegmentationMask:
    """Test mask refinement from 4-class to 20-class."""

    def test_basic_refinement(self) -> None:
        coarse = np.zeros((64, 64), dtype=np.uint8)
        vertex_ids = np.array([1, 5, 6], dtype=np.uint8)
        vertex_coords = np.array([[10, 10], [30, 30], [50, 50]], dtype=int)

        refined = refine_segmentation_mask(
            coarse, vertex_ids, vertex_coords, image_size=(64, 64),
        )

        assert refined[10, 10] == 1  # head
        assert refined[30, 30] == 5  # hips
        assert refined[50, 50] == 6  # upper_arm_l

    def test_out_of_bounds_ignored(self) -> None:
        coarse = np.zeros((64, 64), dtype=np.uint8)
        vertex_ids = np.array([1, 5], dtype=np.uint8)
        vertex_coords = np.array([[10, 10], [100, 100]], dtype=int)  # second is OOB

        refined = refine_segmentation_mask(
            coarse, vertex_ids, vertex_coords, image_size=(64, 64),
        )

        assert refined[10, 10] == 1
        # OOB vertex should not cause errors

    def test_empty_vertices(self) -> None:
        coarse = np.zeros((32, 32), dtype=np.uint8)
        vertex_ids = np.array([], dtype=np.uint8)
        vertex_coords = np.empty((0, 2), dtype=int)

        refined = refine_segmentation_mask(
            coarse, vertex_ids, vertex_coords, image_size=(32, 32),
        )

        assert np.all(refined == 0)

    def test_output_shape(self) -> None:
        coarse = np.zeros((128, 128), dtype=np.uint8)
        vertex_ids = np.array([3], dtype=np.uint8)
        vertex_coords = np.array([[64, 64]], dtype=int)

        refined = refine_segmentation_mask(
            coarse, vertex_ids, vertex_coords, image_size=(128, 128),
        )

        assert refined.shape == (128, 128)
        assert refined.dtype == np.uint8


# ---------------------------------------------------------------------------
# JSON export / import
# ---------------------------------------------------------------------------


class TestJSONExport:
    """Test mapping JSON export and import."""

    def test_round_trip(self, tmp_path: Path) -> None:
        json_path = tmp_path / "mapping.json"

        meshes = [(
            "Body",
            ["hair", "body", "clothes"],
            [{}, {"head": 1.0}, {"hips": 0.8}],
        )]
        mapping = map_character("stdgen_test_001", meshes)
        export_mapping_json(mapping, json_path)

        assert json_path.exists()

        loaded = load_mapping_json(json_path)
        assert loaded["character_id"] == "stdgen_test_001"
        assert loaded["source"] == "stdgen"
        assert loaded["total_vertices"] == 3
        assert len(loaded["meshes"]) == 1
        assert loaded["meshes"][0]["mesh_name"] == "Body"

    def test_region_distribution_in_json(self, tmp_path: Path) -> None:
        json_path = tmp_path / "mapping.json"

        meshes = [(
            "Body",
            ["hair", "hair", "body"],
            [{}, {}, {"leftFoot": 1.0}],
        )]
        mapping = map_character("char_dist", meshes)
        export_mapping_json(mapping, json_path)

        loaded = load_mapping_json(json_path)
        assert loaded["region_distribution"]["head"] == 2
        assert loaded["region_distribution"]["foot_l"] == 1

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        json_path = tmp_path / "nested" / "dir" / "mapping.json"
        meshes = [("Body", ["hair"], [{}])]
        mapping = map_character("test", meshes)
        export_mapping_json(mapping, json_path)
        assert json_path.exists()
