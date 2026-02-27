"""Tests for animation.scripts.bvh_to_strata module."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from animation.scripts.bvh_parser import parse_bvh
from animation.scripts.bvh_to_strata import (
    STRATA_BONES,
    check_strata_compatibility,
    retarget,
)

# ---------------------------------------------------------------------------
# Fixtures: synthetic BVH content
# ---------------------------------------------------------------------------

CMU_WALK_BVH = textwrap.dedent("""\
    HIERARCHY
    ROOT Hips
    {
        OFFSET 0.0 0.0 0.0
        CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
        JOINT Spine
        {
            OFFSET 0.0 5.21 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT Spine1
            {
                OFFSET 0.0 5.45 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT Spine2
                {
                    OFFSET 0.0 5.60 0.0
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    JOINT Neck
                    {
                        OFFSET 0.0 3.12 0.0
                        CHANNELS 3 Zrotation Xrotation Yrotation
                        JOINT Head
                        {
                            OFFSET 0.0 3.48 0.0
                            CHANNELS 3 Zrotation Xrotation Yrotation
                            End Site
                            {
                                OFFSET 0.0 3.0 0.0
                            }
                        }
                    }
                }
            }
        }
        JOINT LeftShoulder
        {
            OFFSET -2.0 5.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT LeftArm
            {
                OFFSET -5.0 0.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT LeftForeArm
                {
                    OFFSET -10.0 0.0 0.0
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    JOINT LeftHand
                    {
                        OFFSET -8.0 0.0 0.0
                        CHANNELS 3 Zrotation Xrotation Yrotation
                        End Site
                        {
                            OFFSET -3.0 0.0 0.0
                        }
                    }
                }
            }
        }
        JOINT RightShoulder
        {
            OFFSET 2.0 5.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT RightArm
            {
                OFFSET 5.0 0.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT RightForeArm
                {
                    OFFSET 10.0 0.0 0.0
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    JOINT RightHand
                    {
                        OFFSET 8.0 0.0 0.0
                        CHANNELS 3 Zrotation Xrotation Yrotation
                        End Site
                        {
                            OFFSET 3.0 0.0 0.0
                        }
                    }
                }
            }
        }
        JOINT LeftUpLeg
        {
            OFFSET -3.91 0.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT LeftLeg
            {
                OFFSET 0.0 -18.34 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT LeftFoot
                {
                    OFFSET 0.0 -17.37 0.0
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    End Site
                    {
                        OFFSET 0.0 -3.0 5.0
                    }
                }
            }
        }
        JOINT RightUpLeg
        {
            OFFSET 3.91 0.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT RightLeg
            {
                OFFSET 0.0 -18.34 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT RightFoot
                {
                    OFFSET 0.0 -17.37 0.0
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    End Site
                    {
                        OFFSET 0.0 -3.0 5.0
                    }
                }
            }
        }
    }
    MOTION
    Frames: 2
    Frame Time: 0.0333333
    0.0 95.0 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
    0.5 95.5 0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5 9.5 10.5 11.5 12.5 13.5 14.5 15.5 16.5 17.5 18.5 19.5 20.5 21.5 22.5 23.5 24.5 25.5 26.5 27.5 28.5 29.5 30.5 31.5 32.5 33.5 34.5 35.5 36.5 37.5 38.5 39.5 40.5 41.5 42.5 43.5 44.5 45.5 46.5 47.5 48.5 49.5 50.5 51.5 52.5 53.5 54.5 55.5 56.5 57.5 58.5 59.5 60.5 61.5 62.5 63.5
""")

# Simplified skeleton without Spine1/Spine2 — only Spine
NO_SPINE12_BVH = textwrap.dedent("""\
    HIERARCHY
    ROOT Hips
    {
        OFFSET 0.0 0.0 0.0
        CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
        JOINT Spine
        {
            OFFSET 0.0 10.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT Neck
            {
                OFFSET 0.0 8.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT Head
                {
                    OFFSET 0.0 4.0 0.0
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    End Site
                    {
                        OFFSET 0.0 3.0 0.0
                    }
                }
            }
        }
        JOINT LeftUpLeg
        {
            OFFSET -4.0 0.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            End Site
            {
                OFFSET 0.0 -18.0 0.0
            }
        }
    }
    MOTION
    Frames: 1
    Frame Time: 0.0333333
    0.0 90.0 0.0 0.0 0.0 0.0 5.0 10.0 15.0 20.0 25.0 30.0 35.0 40.0 45.0 50.0 55.0 60.0
""")

# SFU-style with YXZ rotation order
SFU_RETARGET_BVH = textwrap.dedent("""\
    HIERARCHY
    ROOT Hips
    {
        OFFSET 0.0 0.0 0.0
        CHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation
        JOINT Spine1
        {
            OFFSET 0.0 10.0 0.0
            CHANNELS 3 Yrotation Xrotation Zrotation
            JOINT Head
            {
                OFFSET 0.0 8.0 0.0
                CHANNELS 3 Yrotation Xrotation Zrotation
                End Site
                {
                    OFFSET 0.0 4.0 0.0
                }
            }
        }
    }
    MOTION
    Frames: 1
    Frame Time: 0.0416667
    1.0 80.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_bvh(tmp_path: Path, content: str, name: str = "test.bvh") -> Path:
    bvh_path = tmp_path / name
    bvh_path.write_text(content, encoding="utf-8")
    return bvh_path


# ---------------------------------------------------------------------------
# Core retargeting tests
# ---------------------------------------------------------------------------


class TestRetargetBasic:
    """Tests for basic retargeting functionality."""

    def test_frame_count(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_WALK_BVH))
        result = retarget(bvh)
        assert result.frame_count == 2

    def test_frame_rate(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_WALK_BVH))
        result = retarget(bvh)
        assert result.frame_rate == pytest.approx(30.0, abs=0.1)

    def test_all_strata_bones_present(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_WALK_BVH))
        result = retarget(bvh)
        for frame in result.frames:
            for bone in STRATA_BONES:
                assert bone in frame.rotations, f"Missing bone: {bone}"

    def test_rotation_values_are_tuples(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_WALK_BVH))
        result = retarget(bvh)
        for bone, rot in result.frames[0].rotations.items():
            assert isinstance(rot, tuple), f"{bone} rotation is not a tuple"
            assert len(rot) == 3, f"{bone} rotation has {len(rot)} components"

    def test_root_position_extracted(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_WALK_BVH))
        result = retarget(bvh)
        pos = result.frames[0].root_position
        assert pos == (0.0, 95.0, 0.0)

    def test_root_position_varies_per_frame(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_WALK_BVH))
        result = retarget(bvh)
        assert result.frames[0].root_position != result.frames[1].root_position


class TestMultiSpineCollapse:
    """Tests for multi-spine hierarchy collapsing."""

    def test_spine1_maps_to_spine(self, tmp_path: Path) -> None:
        """Spine1 rotation should appear on Strata 'spine' bone."""
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_WALK_BVH))
        result = retarget(bvh)
        # Spine1 is the 3rd joint (after Hips, Spine), so channels start
        # at offset 6+3=9 → values [7.0, 8.0, 9.0] for frame 0.
        # _extract_rotation with ZXY order: Z=7.0, X=8.0, Y=9.0 → (8.0, 9.0, 7.0)
        spine_rot = result.frames[0].rotations["spine"]
        assert spine_rot == (8.0, 9.0, 7.0)

    def test_spine2_maps_to_chest(self, tmp_path: Path) -> None:
        """Spine2 rotation should appear on Strata 'chest' bone."""
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_WALK_BVH))
        result = retarget(bvh)
        # Spine2 is after Spine1, channels at offset 12 → values [10.0, 11.0, 12.0]
        # ZXY → (11.0, 12.0, 10.0)
        chest_rot = result.frames[0].rotations["chest"]
        assert chest_rot == (11.0, 12.0, 10.0)

    def test_spine_ignored_in_full_chain(self, tmp_path: Path) -> None:
        """When Spine1+Spine2 exist, Spine should not override spine mapping."""
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_WALK_BVH))
        result = retarget(bvh)
        # Spine has channels [4.0, 5.0, 6.0] but should be ignored.
        # spine should use Spine1 values, not Spine values.
        spine_rot = result.frames[0].rotations["spine"]
        assert spine_rot != (5.0, 6.0, 4.0)  # These would be Spine values

    def test_no_spine12_fallback_to_spine(self, tmp_path: Path) -> None:
        """When only Spine exists (no Spine1/Spine2), it maps to 'spine'."""
        bvh = parse_bvh(_write_bvh(tmp_path, NO_SPINE12_BVH))
        result = retarget(bvh)
        spine_rot = result.frames[0].rotations["spine"]
        # Spine channels: [5.0, 10.0, 15.0] with ZXY → (10.0, 15.0, 5.0)
        assert spine_rot == (10.0, 15.0, 5.0)

    def test_no_spine12_chest_is_zero(self, tmp_path: Path) -> None:
        """Without Spine2, 'chest' should remain zero rotation."""
        bvh = parse_bvh(_write_bvh(tmp_path, NO_SPINE12_BVH))
        result = retarget(bvh)
        assert result.frames[0].rotations["chest"] == (0.0, 0.0, 0.0)


class TestBoneMapping:
    """Tests for bone name mapping and unmapped handling."""

    def test_shoulder_bones_mapped(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_WALK_BVH))
        result = retarget(bvh)
        assert "LeftShoulder" in result.source_bones
        assert "RightShoulder" in result.source_bones

    def test_arm_chain_mapped(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_WALK_BVH))
        result = retarget(bvh)
        expected = {"LeftArm", "LeftForeArm", "LeftHand", "RightArm", "RightForeArm", "RightHand"}
        assert expected.issubset(result.source_bones)

    def test_leg_chain_mapped(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_WALK_BVH))
        result = retarget(bvh)
        expected = {"LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot"}
        assert expected.issubset(result.source_bones)

    def test_end_sites_not_in_unmapped(self, tmp_path: Path) -> None:
        """End Sites should be silently ignored, not listed as unmapped."""
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_WALK_BVH))
        result = retarget(bvh)
        for name in result.unmapped_bones:
            assert not name.endswith("_End")

    def test_no_unmapped_in_full_skeleton(self, tmp_path: Path) -> None:
        """A full CMU skeleton should have no unmapped bones."""
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_WALK_BVH))
        result = retarget(bvh)
        assert result.unmapped_bones == []


class TestRotationOrder:
    """Tests for rotation order detection."""

    def test_zxy_detected(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_WALK_BVH))
        result = retarget(bvh)
        assert result.rotation_order == "ZXY"

    def test_yxz_detected(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, SFU_RETARGET_BVH))
        result = retarget(bvh)
        assert result.rotation_order == "YXZ"


class TestEmptyMotion:
    """Tests for edge cases with empty or no motion."""

    def test_zero_frames(self, tmp_path: Path) -> None:
        skeleton_only = textwrap.dedent("""\
            HIERARCHY
            ROOT Hips
            {
                OFFSET 0.0 0.0 0.0
                CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
                End Site
                {
                    OFFSET 0.0 5.0 0.0
                }
            }
        """)
        bvh = parse_bvh(_write_bvh(tmp_path, skeleton_only))
        result = retarget(bvh)
        assert result.frame_count == 0
        assert result.frames == []


# ---------------------------------------------------------------------------
# BVH with finger bones that have significant motion (incompatible)
# ---------------------------------------------------------------------------

FINGER_MOTION_BVH = textwrap.dedent("""\
    HIERARCHY
    ROOT Hips
    {
        OFFSET 0.0 0.0 0.0
        CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
        JOINT Spine
        {
            OFFSET 0.0 10.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT Neck
            {
                OFFSET 0.0 8.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT Head
                {
                    OFFSET 0.0 4.0 0.0
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    End Site
                    {
                        OFFSET 0.0 3.0 0.0
                    }
                }
            }
        }
        JOINT LeftUpLeg
        {
            OFFSET -4.0 0.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT LeftLeg
            {
                OFFSET 0.0 -18.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT LeftFoot
                {
                    OFFSET 0.0 -17.0 0.0
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    End Site
                    {
                        OFFSET 0.0 -3.0 5.0
                    }
                }
            }
        }
        JOINT LeftShoulder
        {
            OFFSET -2.0 5.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT LeftArm
            {
                OFFSET -5.0 0.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT LeftForeArm
                {
                    OFFSET -10.0 0.0 0.0
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    JOINT LeftHand
                    {
                        OFFSET -8.0 0.0 0.0
                        CHANNELS 3 Zrotation Xrotation Yrotation
                        JOINT LThumb
                        {
                            OFFSET -1.0 0.0 1.0
                            CHANNELS 3 Zrotation Xrotation Yrotation
                            End Site
                            {
                                OFFSET -0.5 0.0 0.5
                            }
                        }
                    }
                }
            }
        }
    }
    MOTION
    Frames: 3
    Frame Time: 0.0333333
    0.0 90.0 0.0 0.0 0.0 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 5.0 10.0 15.0
    0.5 90.5 0.5 0.5 0.5 0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5 9.5 10.5 11.5 12.5 13.5 14.5 15.5 16.5 17.5 18.5 19.5 20.5 21.5 22.5 23.5 24.5 25.5 26.5 27.5 28.5 29.5 30.5 55.0 60.0 65.0
    1.0 91.0 1.0 1.0 1.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0 85.0 90.0 95.0
""")

# BVH with a custom unmapped bone that has NO significant motion (compatible)
INACTIVE_CUSTOM_BONE_BVH = textwrap.dedent("""\
    HIERARCHY
    ROOT Hips
    {
        OFFSET 0.0 0.0 0.0
        CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
        JOINT Spine
        {
            OFFSET 0.0 10.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT Head
            {
                OFFSET 0.0 8.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT FacialJaw
                {
                    OFFSET 0.0 1.0 0.5
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    End Site
                    {
                        OFFSET 0.0 -0.5 0.5
                    }
                }
                End Site
                {
                    OFFSET 0.0 3.0 0.0
                }
            }
        }
    }
    MOTION
    Frames: 3
    Frame Time: 0.0333333
    0.0 90.0 0.0 0.0 0.0 0.0 1.0 2.0 3.0 4.0 5.0 6.0 10.0 20.0 30.0
    0.5 90.5 0.5 0.5 0.5 0.5 1.5 2.5 3.5 4.5 5.5 6.5 10.0 20.0 30.0
    1.0 91.0 1.0 1.0 1.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 10.0 20.0 30.0
""")

# BVH with a custom unmapped bone that HAS significant motion (incompatible)
ACTIVE_CUSTOM_BONE_BVH = textwrap.dedent("""\
    HIERARCHY
    ROOT Hips
    {
        OFFSET 0.0 0.0 0.0
        CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
        JOINT Spine
        {
            OFFSET 0.0 10.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT Head
            {
                OFFSET 0.0 8.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT FacialJaw
                {
                    OFFSET 0.0 1.0 0.5
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    End Site
                    {
                        OFFSET 0.0 -0.5 0.5
                    }
                }
                End Site
                {
                    OFFSET 0.0 3.0 0.0
                }
            }
        }
    }
    MOTION
    Frames: 3
    Frame Time: 0.0333333
    0.0 90.0 0.0 0.0 0.0 0.0 1.0 2.0 3.0 4.0 5.0 6.0 0.0 0.0 0.0
    0.5 90.5 0.5 0.5 0.5 0.5 1.5 2.5 3.5 4.5 5.5 6.5 30.0 45.0 10.0
    1.0 91.0 1.0 1.0 1.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 60.0 90.0 20.0
""")


# ---------------------------------------------------------------------------
# Compatibility check tests
# ---------------------------------------------------------------------------


class TestStrataCompatibility:
    """Tests for check_strata_compatibility()."""

    def test_full_cmu_skeleton_is_compatible(self, tmp_path: Path) -> None:
        """A standard CMU walk skeleton with only mapped bones is compatible."""
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_WALK_BVH))
        result = check_strata_compatibility(bvh)
        assert result.compatible is True
        assert result.active_unmapped == []

    def test_skeleton_only_is_compatible(self, tmp_path: Path) -> None:
        """A BVH with no motion data should be compatible by default."""
        skeleton_only = textwrap.dedent("""\
            HIERARCHY
            ROOT Hips
            {
                OFFSET 0.0 0.0 0.0
                CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
                End Site
                {
                    OFFSET 0.0 5.0 0.0
                }
            }
        """)
        bvh = parse_bvh(_write_bvh(tmp_path, skeleton_only))
        result = check_strata_compatibility(bvh)
        assert result.compatible is True
        assert result.reason == "No motion data — skeleton-only file"

    def test_finger_motion_is_incompatible(self, tmp_path: Path) -> None:
        """A clip with active finger bones should be incompatible."""
        bvh = parse_bvh(_write_bvh(tmp_path, FINGER_MOTION_BVH))
        result = check_strata_compatibility(bvh)
        assert result.compatible is False
        assert "LThumb" in result.active_unmapped

    def test_inactive_unmapped_bone_is_compatible(self, tmp_path: Path) -> None:
        """Unmapped bones with no significant motion should not block compat."""
        bvh = parse_bvh(_write_bvh(tmp_path, INACTIVE_CUSTOM_BONE_BVH))
        result = check_strata_compatibility(bvh)
        assert result.compatible is True
        assert result.active_unmapped == []
        assert result.unmapped_count > 0

    def test_active_custom_bone_is_incompatible(self, tmp_path: Path) -> None:
        """Unmapped bones with significant motion should block compat."""
        bvh = parse_bvh(_write_bvh(tmp_path, ACTIVE_CUSTOM_BONE_BVH))
        result = check_strata_compatibility(bvh)
        assert result.compatible is False
        assert "FacialJaw" in result.active_unmapped

    def test_custom_threshold(self, tmp_path: Path) -> None:
        """A very high threshold should make everything compatible."""
        bvh = parse_bvh(_write_bvh(tmp_path, ACTIVE_CUSTOM_BONE_BVH))
        result = check_strata_compatibility(bvh, threshold=1000.0)
        assert result.compatible is True

    def test_mapped_count(self, tmp_path: Path) -> None:
        """Mapped count should reflect how many bones were successfully mapped."""
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_WALK_BVH))
        result = check_strata_compatibility(bvh)
        assert result.mapped_count > 0

    def test_reason_mentions_compatible(self, tmp_path: Path) -> None:
        """Reason string should indicate compatibility status."""
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_WALK_BVH))
        result = check_strata_compatibility(bvh)
        assert "Compatible" in result.reason

    def test_reason_mentions_incompatible(self, tmp_path: Path) -> None:
        """Reason string should indicate incompatibility status."""
        bvh = parse_bvh(_write_bvh(tmp_path, FINGER_MOTION_BVH))
        result = check_strata_compatibility(bvh)
        assert "Incompatible" in result.reason
