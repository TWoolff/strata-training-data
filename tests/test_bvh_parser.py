"""Tests for animation.scripts.bvh_parser module."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from animation.scripts.bvh_parser import (
    BVHParseError,
    get_frame_array,
    get_joint_frame,
    parse_bvh,
)

# ---------------------------------------------------------------------------
# Fixtures: synthetic BVH content
# ---------------------------------------------------------------------------

CMU_STYLE_BVH = textwrap.dedent("""\
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
    Frames: 3
    Frame Time: 0.0333333
    0.0 100.0 0.0 0.0 0.0 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0
    0.1 100.1 0.1 0.1 0.1 0.1 1.1 2.1 3.1 4.1 5.1 6.1 7.1 8.1 9.1 10.1 11.1 12.1 13.1 14.1 15.1 16.1 17.1 18.1 19.1 20.1 21.1 22.1 23.1 24.1 25.1 26.1 27.1 28.1 29.1 30.1
    0.2 100.2 0.2 0.2 0.2 0.2 1.2 2.2 3.2 4.2 5.2 6.2 7.2 8.2 9.2 10.2 11.2 12.2 13.2 14.2 15.2 16.2 17.2 18.2 19.2 20.2 21.2 22.2 23.2 24.2 25.2 26.2 27.2 28.2 29.2 30.2
""")

MINIMAL_BVH = textwrap.dedent("""\
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
    MOTION
    Frames: 1
    Frame Time: 0.0416667
    0.0 90.0 0.0 0.0 0.0 0.0
""")

SKELETON_ONLY_BVH = textwrap.dedent("""\
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

SFU_STYLE_BVH = textwrap.dedent("""\
    HIERARCHY
    ROOT  Hips
    {
      OFFSET  0.00  0.00  0.00
      CHANNELS  6  Xposition  Yposition  Zposition  Yrotation  Xrotation  Zrotation
      JOINT  Chest
      {
        OFFSET  0.00  10.50  0.00
        CHANNELS  3  Yrotation  Xrotation  Zrotation
        JOINT  Head
        {
          OFFSET  0.00  8.20  0.00
          CHANNELS  3  Yrotation  Xrotation  Zrotation
          End Site
          {
            OFFSET  0.00  4.00  0.00
          }
        }
      }
      JOINT  LeftUpLeg
      {
        OFFSET  -4.00  0.00  0.00
        CHANNELS  3  Yrotation  Xrotation  Zrotation
        End Site
        {
          OFFSET  0.00  -18.00  0.00
        }
      }
      JOINT  RightUpLeg
      {
        OFFSET  4.00  0.00  0.00
        CHANNELS  3  Yrotation  Xrotation  Zrotation
        End Site
        {
          OFFSET  0.00  -18.00  0.00
        }
      }
    }
    MOTION
    Frames:  2
    Frame Time:  0.0333333
    0.0  95.0  0.0  0.0  0.0  0.0  1.0  2.0  3.0  4.0  5.0  6.0  7.0  8.0  9.0  10.0  11.0  12.0
    0.5  95.5  0.5  0.5  0.5  0.5  1.5  2.5  3.5  4.5  5.5  6.5  7.5  8.5  9.5  10.5  11.5  12.5
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_bvh(tmp_path: Path, content: str, name: str = "test.bvh") -> Path:
    """Write BVH content to a temp file and return its path."""
    bvh_path = tmp_path / name
    bvh_path.write_text(content, encoding="utf-8")
    return bvh_path


# ---------------------------------------------------------------------------
# Skeleton parsing tests
# ---------------------------------------------------------------------------


class TestHierarchyParsing:
    """Tests for HIERARCHY section parsing."""

    def test_cmu_root_joint(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_STYLE_BVH))
        assert bvh.skeleton.root == "Hips"

    def test_cmu_joint_count(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_STYLE_BVH))
        # 11 named joints + 3 End Sites = 14 total entries
        assert len(bvh.skeleton.joints) == 14

    def test_cmu_joint_names(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_STYLE_BVH))
        expected_named = {
            "Hips", "Spine", "Spine1", "Neck", "Head",
            "LeftUpLeg", "LeftLeg", "LeftFoot",
            "RightUpLeg", "RightLeg", "RightFoot",
        }
        named_joints = {
            name for name in bvh.skeleton.joints
            if not name.endswith("_End")
        }
        assert named_joints == expected_named

    def test_parent_child_relationships(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_STYLE_BVH))
        joints = bvh.skeleton.joints

        assert joints["Hips"].parent is None
        assert joints["Spine"].parent == "Hips"
        assert joints["Spine1"].parent == "Spine"
        assert joints["LeftUpLeg"].parent == "Hips"
        assert joints["LeftFoot"].parent == "LeftLeg"

    def test_children_list(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_STYLE_BVH))
        hips = bvh.skeleton.joints["Hips"]
        assert set(hips.children) == {"Spine", "LeftUpLeg", "RightUpLeg"}

    def test_offsets(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_STYLE_BVH))
        assert bvh.skeleton.joints["Hips"].offset == (0.0, 0.0, 0.0)
        assert bvh.skeleton.joints["Spine"].offset == (0.0, 5.21, 0.0)
        assert bvh.skeleton.joints["LeftUpLeg"].offset == (-3.91, 0.0, 0.0)

    def test_root_channels(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_STYLE_BVH))
        hips = bvh.skeleton.joints["Hips"]
        assert len(hips.channels) == 6
        assert hips.channels[:3] == ["Xposition", "Yposition", "Zposition"]

    def test_non_root_channels(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_STYLE_BVH))
        spine = bvh.skeleton.joints["Spine"]
        assert len(spine.channels) == 3
        assert spine.channels == ["Zrotation", "Xrotation", "Yrotation"]

    def test_end_site_no_channels(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_STYLE_BVH))
        end_sites = [j for j in bvh.skeleton.joints.values() if j.name.endswith("_End")]
        assert len(end_sites) == 3
        for end_site in end_sites:
            assert end_site.channels == []

    def test_joint_order_matches_hierarchy(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_STYLE_BVH))
        order = bvh.skeleton.joint_order
        # Root is first
        assert order[0] == "Hips"
        # Spine chain is in order
        spine_idx = order.index("Spine")
        spine1_idx = order.index("Spine1")
        neck_idx = order.index("Neck")
        assert spine_idx < spine1_idx < neck_idx


# ---------------------------------------------------------------------------
# Motion parsing tests
# ---------------------------------------------------------------------------


class TestMotionParsing:
    """Tests for MOTION section parsing."""

    def test_frame_count(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_STYLE_BVH))
        assert bvh.motion.frame_count == 3

    def test_frame_time(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_STYLE_BVH))
        assert pytest.approx(bvh.motion.frame_time, abs=1e-6) == 0.0333333

    def test_frame_data_indexed_by_joint(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_STYLE_BVH))
        frame0 = bvh.motion.frames[0]
        assert "Hips" in frame0
        assert "Spine" in frame0
        assert "Head" in frame0

    def test_root_frame_values(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_STYLE_BVH))
        hips_f0 = bvh.motion.frames[0]["Hips"]
        assert len(hips_f0) == 6
        assert hips_f0[:3] == [0.0, 100.0, 0.0]  # position

    def test_non_root_frame_values(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_STYLE_BVH))
        spine_f0 = bvh.motion.frames[0]["Spine"]
        assert len(spine_f0) == 3
        assert spine_f0 == [1.0, 2.0, 3.0]

    def test_frame_values_change_per_frame(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_STYLE_BVH))
        hips_f0 = bvh.motion.frames[0]["Hips"]
        hips_f1 = bvh.motion.frames[1]["Hips"]
        assert hips_f0 != hips_f1
        assert hips_f1[0] == pytest.approx(0.1)

    def test_end_sites_not_in_frames(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_STYLE_BVH))
        frame0 = bvh.motion.frames[0]
        end_sites = [name for name in frame0 if name.endswith("_End")]
        assert len(end_sites) == 0

    def test_all_channeled_joints_in_frames(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_STYLE_BVH))
        channeled = {
            name for name, j in bvh.skeleton.joints.items() if j.channels
        }
        for frame in bvh.motion.frames:
            assert set(frame.keys()) == channeled


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and variant formats."""

    def test_minimal_single_frame(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, MINIMAL_BVH))
        assert bvh.skeleton.root == "Hips"
        assert bvh.motion.frame_count == 1
        assert bvh.motion.frames[0]["Hips"] == [0.0, 90.0, 0.0, 0.0, 0.0, 0.0]

    def test_skeleton_only_no_motion(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, SKELETON_ONLY_BVH))
        assert bvh.skeleton.root == "Hips"
        assert bvh.motion.frame_count == 0
        assert bvh.motion.frames == []

    def test_sfu_format_parses(self, tmp_path: Path) -> None:
        """SFU BVH uses extra whitespace and different rotation order."""
        bvh = parse_bvh(_write_bvh(tmp_path, SFU_STYLE_BVH))
        assert bvh.skeleton.root == "Hips"
        assert bvh.motion.frame_count == 2

    def test_sfu_rotation_order_preserved(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, SFU_STYLE_BVH))
        hips = bvh.skeleton.joints["Hips"]
        # SFU uses YXZ order for rotations
        assert hips.channels[3:] == ["Yrotation", "Xrotation", "Zrotation"]

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parse_bvh(tmp_path / "nonexistent.bvh")

    def test_empty_file(self, tmp_path: Path) -> None:
        with pytest.raises(BVHParseError):
            parse_bvh(_write_bvh(tmp_path, ""))

    def test_no_root_joint(self, tmp_path: Path) -> None:
        bad_bvh = "HIERARCHY\nMOTION\nFrames: 0\nFrame Time: 0.033\n"
        with pytest.raises(BVHParseError, match="No ROOT joint"):
            parse_bvh(_write_bvh(tmp_path, bad_bvh))

    def test_string_path_accepted(self, tmp_path: Path) -> None:
        path = _write_bvh(tmp_path, MINIMAL_BVH)
        bvh = parse_bvh(str(path))
        assert bvh.skeleton.root == "Hips"


# ---------------------------------------------------------------------------
# Convenience API tests
# ---------------------------------------------------------------------------


class TestConvenienceAPI:
    """Tests for get_frame_array and get_joint_frame."""

    def test_get_frame_array_shape(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_STYLE_BVH))
        arr = get_frame_array(bvh.motion)
        # 11 joints with channels: Hips(6) + 10 others(3 each) = 36
        assert arr.shape == (3, 36)
        assert arr.dtype == np.float64

    def test_get_frame_array_empty_motion(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, SKELETON_ONLY_BVH))
        arr = get_frame_array(bvh.motion)
        assert arr.shape == (0, 0)

    def test_get_joint_frame(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_STYLE_BVH))
        values = get_joint_frame(bvh.motion, "Hips", 0)
        assert values[:3] == [0.0, 100.0, 0.0]

    def test_get_joint_frame_last(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_STYLE_BVH))
        values = get_joint_frame(bvh.motion, "Hips", 2)
        assert values[0] == pytest.approx(0.2)

    def test_get_joint_frame_invalid_index(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_STYLE_BVH))
        with pytest.raises(IndexError):
            get_joint_frame(bvh.motion, "Hips", 99)

    def test_get_joint_frame_invalid_name(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, CMU_STYLE_BVH))
        with pytest.raises(KeyError):
            get_joint_frame(bvh.motion, "NonExistent", 0)
