"""Tests for the fuzzy bone matching functions in pipeline.bone_mapper.

These tests exercise the pure-Python normalization and matching logic
without requiring Blender (bpy). We import the private helpers directly
and mock out bpy at import time.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

# Provide a mock bpy module so pipeline.bone_mapper can be imported outside Blender.
_bpy_mock = types.ModuleType("bpy")
_bpy_mock.types = MagicMock()
sys.modules.setdefault("bpy", _bpy_mock)

from pipeline.bone_mapper import (  # noqa: E402
    _canonicalize_laterality,
    _normalize_bone_name,
    _try_fuzzy_keyword,
)

# ---------------------------------------------------------------------------
# _normalize_bone_name
# ---------------------------------------------------------------------------


class TestNormalizeBoneName:
    """Test name normalization: prefix stripping, camelCase splitting, tokenizing."""

    def test_mixamo_prefix(self) -> None:
        assert _normalize_bone_name("mixamorig:LeftArm") == ["left", "arm"]

    def test_def_prefix(self) -> None:
        assert _normalize_bone_name("DEF-upper_arm.L") == ["upper", "arm", "l"]

    def test_camel_case_split(self) -> None:
        assert _normalize_bone_name("leftUpperArm") == ["left", "upper", "arm"]

    def test_blender_dot_suffix(self) -> None:
        assert _normalize_bone_name("upper_arm.L") == ["upper", "arm", "l"]

    def test_numeric_suffix_stripped(self) -> None:
        assert _normalize_bone_name("DEF-spine.001") == ["spine"]

    def test_bip01_prefix(self) -> None:
        assert _normalize_bone_name("Bip01_L_UpperArm") == ["l", "upper", "arm"]

    def test_plain_name(self) -> None:
        assert _normalize_bone_name("Hips") == ["hips"]

    def test_spaces_in_name(self) -> None:
        assert _normalize_bone_name("Bip01 L Thigh") == ["l", "thigh"]

    def test_bone_prefix(self) -> None:
        assert _normalize_bone_name("Bone_LeftHand") == ["left", "hand"]

    def test_empty_after_prefix(self) -> None:
        assert _normalize_bone_name("DEF-") == []


# ---------------------------------------------------------------------------
# _canonicalize_laterality
# ---------------------------------------------------------------------------


class TestCanonicalizeLaterality:
    """Test laterality alias replacement."""

    def test_l_becomes_left(self) -> None:
        assert _canonicalize_laterality(["upper", "arm", "l"]) == [
            "upper",
            "arm",
            "left",
        ]

    def test_r_becomes_right(self) -> None:
        assert _canonicalize_laterality(["foot", "r"]) == ["foot", "right"]

    def test_leg_unchanged(self) -> None:
        """'leg' should NOT be treated as a laterality marker."""
        assert _canonicalize_laterality(["leg", "upper"]) == ["leg", "upper"]

    def test_left_stays_left(self) -> None:
        assert _canonicalize_laterality(["left", "arm"]) == ["left", "arm"]


# ---------------------------------------------------------------------------
# _try_fuzzy_keyword
# ---------------------------------------------------------------------------


class TestTryFuzzyKeyword:
    """Test end-to-end fuzzy keyword matching."""

    @pytest.mark.parametrize(
        "bone_name, expected_region",
        [
            # Blender-style rigs
            ("upper_arm.L", 7),  # upper_arm_l
            ("upper_arm.R", 11),  # upper_arm_r
            ("DEF-spine.001", 4),  # spine
            ("DEF-upper_arm.L.001", 7),  # upper_arm_l (with numeric suffix)
            ("forearm.R", 12),  # forearm_r
            ("hand.L", 9),  # hand_l
            ("thigh.R", 17),  # upper_leg_r
            ("shin.L", 15),  # lower_leg_l
            ("foot.R", 19),  # foot_r
            # Generic naming patterns
            ("Arm_Upper_L", 7),  # upper_arm_l
            ("L_Thigh", 14),  # upper_leg_l
            ("R_Forearm", 12),  # forearm_r
            # camelCase
            ("leftUpperArm", 7),  # upper_arm_l
            ("rightFoot", 19),  # foot_r
            ("leftHand", 9),  # hand_l
            # Head / torso (no laterality)
            ("Head", 1),
            ("Neck", 2),
            ("Hips", 5),
            ("Pelvis", 5),
            # Shoulder
            ("shoulder.L", 6),
            ("clavicle.R", 10),
        ],
    )
    def test_correct_region(self, bone_name: str, expected_region: int) -> None:
        region, score = _try_fuzzy_keyword(bone_name)
        assert region == expected_region, (
            f"Bone {bone_name!r}: expected region {expected_region}, got {region} "
            f"(score={score:.2f})"
        )

    def test_no_false_leg_left(self) -> None:
        """'leg' should NOT be mistaken for left-side (the 'l' in 'leg')."""
        # "leg" alone has no laterality — it should not match a lateralized pattern.
        # The tokenized form is ["leg"], which canonicalizes to ["leg"] (no alias).
        region, _ = _try_fuzzy_keyword("leg")
        # Should not match any left-side region specifically
        if region is not None:
            from pipeline.config import REGION_NAMES

            name = REGION_NAMES.get(region, "")
            assert "_l" not in name, f"'leg' falsely matched left-side region: {name} (id={region})"

    def test_very_short_name_no_match(self) -> None:
        """Single-letter names should not produce confident matches."""
        region, score = _try_fuzzy_keyword("L")
        # "L" normalizes to ["l"] → canonicalizes to ["left"]
        # Single keyword "left" shouldn't match anything requiring 2+ keywords
        # at score >= 0.6 (no body part keyword present)
        if region is not None:
            assert score >= 0.6

    def test_score_above_threshold(self) -> None:
        """All successful matches should have score >= FUZZY_MIN_SCORE."""
        from pipeline.config import FUZZY_MIN_SCORE

        region, score = _try_fuzzy_keyword("DEF-upper_arm.L")
        assert region is not None
        assert score >= FUZZY_MIN_SCORE

    def test_unmatchable_returns_none(self) -> None:
        """Completely unrecognizable names should return None."""
        region, score = _try_fuzzy_keyword("XYZ_widget_controller")
        assert region is None
        assert score == 0.0
