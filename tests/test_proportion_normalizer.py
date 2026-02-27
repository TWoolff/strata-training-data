"""Tests for animation.scripts.proportion_normalizer module."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from animation.scripts.bvh_parser import parse_bvh
from animation.scripts.bvh_to_strata import RetargetedAnimation, RetargetedFrame, retarget
from animation.scripts.proportion_normalizer import (
    BoneLengths,
    extract_bone_lengths,
    normalize_proportions,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CMU_SKELETON_BVH = textwrap.dedent("""\
    HIERARCHY
    ROOT Hips
    {
        OFFSET 0.0 0.0 0.0
        CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
        JOINT Spine
        {
            OFFSET 0.0 5.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT Spine1
            {
                OFFSET 0.0 5.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT Spine2
                {
                    OFFSET 0.0 5.0 0.0
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    JOINT Neck
                    {
                        OFFSET 0.0 3.0 0.0
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
        JOINT RightUpLeg
        {
            OFFSET 4.0 0.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT RightLeg
            {
                OFFSET 0.0 -18.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT RightFoot
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
    }
    MOTION
    Frames: 2
    Frame Time: 0.0333333
    10.0 95.0 5.0 0.0 0.0 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0 32.0 33.0
    20.0 96.0 10.0 0.5 0.5 0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5 9.5 10.5 11.5 12.5 13.5 14.5 15.5 16.5 17.5 18.5 19.5 20.5 21.5 22.5 23.5 24.5 25.5 26.5 27.5 28.5 29.5 30.5 31.5 32.5 33.5
""")


def _write_bvh(tmp_path: Path, content: str, name: str = "test.bvh") -> Path:
    bvh_path = tmp_path / name
    bvh_path.write_text(content, encoding="utf-8")
    return bvh_path


# ---------------------------------------------------------------------------
# Bone length extraction tests
# ---------------------------------------------------------------------------


class TestExtractBoneLengths:
    """Tests for extract_bone_lengths."""

    def test_extracts_mapped_bones(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_SKELETON_BVH))
        lengths = extract_bone_lengths(bvh.skeleton)
        assert "spine" in lengths.lengths
        assert "neck" in lengths.lengths
        assert "head" in lengths.lengths

    def test_bone_length_values(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_SKELETON_BVH))
        lengths = extract_bone_lengths(bvh.skeleton)
        # Spine has offset (0, 5, 0) → length 5.0
        assert lengths.lengths["spine"] == pytest.approx(5.0)
        # Neck has offset (0, 3, 0) → length 3.0
        assert lengths.lengths["neck"] == pytest.approx(3.0)
        # Head has offset (0, 4, 0) → length 4.0
        assert lengths.lengths["head"] == pytest.approx(4.0)

    def test_total_height(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_SKELETON_BVH))
        lengths = extract_bone_lengths(bvh.skeleton)
        # hips(0) + spine(5) + chest(5) + neck(3) + head(4) = 17.0
        assert lengths.total_height == pytest.approx(17.0)

    def test_leg_bone_lengths(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_SKELETON_BVH))
        lengths = extract_bone_lengths(bvh.skeleton)
        # LeftUpLeg offset (-4, 0, 0) → length 4.0
        assert lengths.lengths["upper_leg_l"] == pytest.approx(4.0)
        # LeftLeg offset (0, -18, 0) → length 18.0
        assert lengths.lengths["lower_leg_l"] == pytest.approx(18.0)


# ---------------------------------------------------------------------------
# Proportion normalization tests
# ---------------------------------------------------------------------------


class TestNormalizeProportions:
    """Tests for normalize_proportions."""

    def test_scale_factor_applied_to_root(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_SKELETON_BVH))
        animation = retarget(bvh)
        source = extract_bone_lengths(bvh.skeleton)

        # Create a chibi target at half height
        target = BoneLengths(
            lengths={"hips": 0.0, "spine": 2.5, "chest": 2.5, "neck": 1.5, "head": 2.0},
            total_height=8.5,
        )

        result = normalize_proportions(animation, source, target)
        scale = 8.5 / 17.0  # 0.5

        # Frame 0 root was (10.0, 95.0, 5.0)
        pos = result.frames[0].root_position
        assert pos[0] == pytest.approx(10.0 * scale)
        assert pos[1] == pytest.approx(95.0 * scale)
        assert pos[2] == pytest.approx(5.0 * scale)

    def test_rotations_preserved(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_SKELETON_BVH))
        animation = retarget(bvh)
        source = extract_bone_lengths(bvh.skeleton)
        target = BoneLengths(
            lengths={"hips": 0.0, "spine": 2.5, "chest": 2.5, "neck": 1.5, "head": 2.0},
            total_height=8.5,
        )

        result = normalize_proportions(animation, source, target)

        # All rotations should be identical before and after normalization
        for i, frame in enumerate(result.frames):
            for bone in frame.rotations:
                assert frame.rotations[bone] == animation.frames[i].rotations[bone]

    def test_identity_scale(self, tmp_path: Path) -> None:
        """When source and target have the same height, positions are unchanged."""
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_SKELETON_BVH))
        animation = retarget(bvh)
        source = extract_bone_lengths(bvh.skeleton)

        result = normalize_proportions(animation, source, source)

        for i, frame in enumerate(result.frames):
            orig = animation.frames[i].root_position
            assert frame.root_position[0] == pytest.approx(orig[0])
            assert frame.root_position[1] == pytest.approx(orig[1])
            assert frame.root_position[2] == pytest.approx(orig[2])

    def test_zero_source_height_skips(self, tmp_path: Path) -> None:
        """Zero source height should skip normalization and return unchanged."""
        animation = RetargetedAnimation(
            frames=[RetargetedFrame(rotations={}, root_position=(10.0, 20.0, 30.0))],
            frame_count=1,
        )
        source = BoneLengths(total_height=0.0)
        target = BoneLengths(total_height=10.0)

        result = normalize_proportions(animation, source, target)
        assert result.frames[0].root_position == (10.0, 20.0, 30.0)

    def test_zero_target_height_skips(self, tmp_path: Path) -> None:
        animation = RetargetedAnimation(
            frames=[RetargetedFrame(rotations={}, root_position=(10.0, 20.0, 30.0))],
            frame_count=1,
        )
        source = BoneLengths(total_height=10.0)
        target = BoneLengths(total_height=0.0)

        result = normalize_proportions(animation, source, target)
        assert result.frames[0].root_position == (10.0, 20.0, 30.0)

    def test_frame_count_preserved(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_SKELETON_BVH))
        animation = retarget(bvh)
        source = extract_bone_lengths(bvh.skeleton)
        target = BoneLengths(total_height=10.0)

        result = normalize_proportions(animation, source, target)
        assert result.frame_count == animation.frame_count
        assert result.frame_rate == animation.frame_rate
