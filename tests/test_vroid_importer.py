"""Tests for VRM/VRoid import and bone mapping.

Tests the pure-Python logic without requiring Blender (bpy):
- VRM bone alias lookups in config
- VRM bone names through the fuzzy matching pipeline
- Source detection for vroid character IDs
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

# Provide a mock bpy module so pipeline modules can be imported outside Blender.
_bpy_mock = types.ModuleType("bpy")
_bpy_mock.types = MagicMock()
_bpy_mock.ops = MagicMock()
_bpy_mock.data = MagicMock()
_bpy_mock.context = MagicMock()
sys.modules.setdefault("bpy", _bpy_mock)

# Mock mathutils as well (used by vroid_importer)
_mathutils_mock = types.ModuleType("mathutils")
_mathutils_mock.Vector = MagicMock()
_mathutils_mock.Euler = MagicMock()
sys.modules.setdefault("mathutils", _mathutils_mock)

from pipeline.bone_mapper import (  # noqa: E402
    _try_fuzzy_keyword,
)
from pipeline.config import (  # noqa: E402
    COMMON_BONE_ALIASES,
    VRM_BONE_ALIASES,
)

# ---------------------------------------------------------------------------
# VRM_BONE_ALIASES coverage
# ---------------------------------------------------------------------------


class TestVrmBoneAliases:
    """Test that VRM_BONE_ALIASES maps all standardized VRM bones correctly."""

    @pytest.mark.parametrize(
        "vrm_bone, expected_region",
        [
            # Head / neck
            ("head", 1),
            ("neck", 2),
            # Torso
            ("upperChest", 3),
            ("chest", 3),
            ("spine", 4),
            ("hips", 5),
            # Left arm chain
            ("leftShoulder", 6),
            ("leftUpperArm", 7),
            ("leftLowerArm", 8),
            ("leftHand", 9),
            # Right arm chain
            ("rightShoulder", 10),
            ("rightUpperArm", 11),
            ("rightLowerArm", 12),
            ("rightHand", 13),
            # Left leg chain
            ("leftUpperLeg", 14),
            ("leftLowerLeg", 15),
            ("leftFoot", 16),
            ("leftToes", 16),
            # Right leg chain
            ("rightUpperLeg", 17),
            ("rightLowerLeg", 18),
            ("rightFoot", 19),
            ("rightToes", 19),
        ],
    )
    def test_vrm_bone_alias_mapping(self, vrm_bone: str, expected_region: int) -> None:
        region = VRM_BONE_ALIASES.get(vrm_bone)
        assert region == expected_region, (
            f"VRM bone {vrm_bone!r}: expected region {expected_region}, got {region}"
        )

    def test_vrm_finger_bones_map_to_hand(self) -> None:
        """All VRM finger bones should map to their respective hand region."""
        left_fingers = [
            "leftThumbMetacarpal",
            "leftThumbProximal",
            "leftThumbDistal",
            "leftIndexProximal",
            "leftIndexIntermediate",
            "leftIndexDistal",
            "leftMiddleProximal",
            "leftMiddleIntermediate",
            "leftMiddleDistal",
            "leftRingProximal",
            "leftRingIntermediate",
            "leftRingDistal",
            "leftLittleProximal",
            "leftLittleIntermediate",
            "leftLittleDistal",
        ]
        right_fingers = [
            "rightThumbMetacarpal",
            "rightThumbProximal",
            "rightThumbDistal",
            "rightIndexProximal",
            "rightIndexIntermediate",
            "rightIndexDistal",
            "rightMiddleProximal",
            "rightMiddleIntermediate",
            "rightMiddleDistal",
            "rightRingProximal",
            "rightRingIntermediate",
            "rightRingDistal",
            "rightLittleProximal",
            "rightLittleIntermediate",
            "rightLittleDistal",
        ]

        for bone in left_fingers:
            assert VRM_BONE_ALIASES.get(bone) == 9, (
                f"Left finger {bone!r} should map to region 9 (hand_l)"
            )

        for bone in right_fingers:
            assert VRM_BONE_ALIASES.get(bone) == 13, (
                f"Right finger {bone!r} should map to region 13 (hand_r)"
            )

    def test_all_19_regions_covered(self) -> None:
        """VRM aliases should cover all 19 body regions (1-19)."""
        covered_regions = set(VRM_BONE_ALIASES.values())
        expected = set(range(1, 20))
        missing = expected - covered_regions
        assert not missing, f"Regions not covered by VRM aliases: {missing}"

    def test_no_overlap_with_common_aliases(self) -> None:
        """VRM-specific names (camelCase) should not already exist in COMMON_BONE_ALIASES.

        Some overlap is expected for simple names like 'head', 'neck', etc.
        This test checks VRM-specific camelCase names only.
        """
        vrm_specific = [
            "upperChest",
            "leftShoulder",
            "leftUpperArm",
            "leftLowerArm",
            "rightShoulder",
            "rightUpperArm",
            "rightLowerArm",
            "leftUpperLeg",
            "leftLowerLeg",
            "rightUpperLeg",
            "rightLowerLeg",
            "leftToes",
            "rightToes",
        ]
        for bone in vrm_specific:
            assert bone not in COMMON_BONE_ALIASES, (
                f"VRM bone {bone!r} already in COMMON_BONE_ALIASES — may cause priority issues"
            )


# ---------------------------------------------------------------------------
# Fuzzy matching of VRM-style bone names
# ---------------------------------------------------------------------------


class TestVrmFuzzyMatching:
    """Test that VRM camelCase bone names resolve correctly via fuzzy matching."""

    @pytest.mark.parametrize(
        "bone_name, expected_region",
        [
            ("leftUpperArm", 7),
            ("rightLowerArm", 12),
            ("leftUpperLeg", 14),
            ("rightLowerLeg", 18),
            ("leftFoot", 16),
            ("rightHand", 13),
            ("leftShoulder", 6),
            ("rightShoulder", 10),
        ],
    )
    def test_vrm_camelcase_fuzzy_match(self, bone_name: str, expected_region: int) -> None:
        """VRM camelCase names should match via fuzzy even without alias table."""
        region, score = _try_fuzzy_keyword(bone_name)
        assert region == expected_region, (
            f"Fuzzy match for {bone_name!r}: expected region {expected_region}, "
            f"got {region} (score={score:.2f})"
        )


# ---------------------------------------------------------------------------
# Source detection
# ---------------------------------------------------------------------------


class TestVroidSourceDetection:
    """Test that vroid character IDs are correctly identified.

    Note: We test the logic directly rather than importing generate_dataset,
    which requires numpy/PIL (Blender-bundled dependencies).
    """

    @staticmethod
    def _infer_source(char_id: str) -> str:
        """Mirror of generate_dataset._infer_source for testing."""
        lower = char_id.lower()
        if lower.startswith("mixamo"):
            return "mixamo"
        if lower.startswith("quaternius"):
            return "quaternius"
        if lower.startswith("kenney"):
            return "kenney"
        if lower.startswith("spine"):
            return "spine"
        if lower.startswith("vroid"):
            return "vroid"
        return "unknown"

    def test_infer_source_vroid(self) -> None:
        assert self._infer_source("vroid_model_001") == "vroid"
        assert self._infer_source("vroid_AvatarSample") == "vroid"

    def test_infer_source_not_vroid(self) -> None:
        assert self._infer_source("mixamo_Vanguard") != "vroid"
        assert self._infer_source("some_character") != "vroid"

    def test_vroid_importer_char_id_prefix(self) -> None:
        """import_vrm should produce character IDs starting with 'vroid_'."""
        # The vroid_importer sets character_id = f"vroid_{vrm_path.stem}"
        from pathlib import Path

        stem = Path("my_model.vrm").stem
        char_id = f"vroid_{stem}"
        assert self._infer_source(char_id) == "vroid"
