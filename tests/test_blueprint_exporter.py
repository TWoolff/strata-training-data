"""Tests for animation.scripts.blueprint_exporter module."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from animation.scripts.blueprint_exporter import (
    SKELETON_ID,
    build_blueprint,
    export_blueprint,
)
from animation.scripts.bvh_parser import parse_bvh
from animation.scripts.bvh_to_strata import STRATA_BONES, retarget

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_BVH = textwrap.dedent("""\
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
            End Site
            {
                OFFSET 0.0 -18.0 0.0
            }
        }
    }
    MOTION
    Frames: 2
    Frame Time: 0.0333333
    1.0 90.0 2.0 0.0 0.0 0.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0
    2.0 91.0 3.0 0.5 0.5 0.5 3.5 4.5 5.5 6.5 7.5 8.5 9.5 10.5 11.5 12.5 13.5 14.5 15.5 16.5 17.5 18.5 19.5 20.5
""")


def _write_bvh(tmp_path: Path, content: str, name: str = "test.bvh") -> Path:
    bvh_path = tmp_path / name
    bvh_path.write_text(content, encoding="utf-8")
    return bvh_path


def _retarget_simple(tmp_path: Path):
    bvh = parse_bvh(_write_bvh(tmp_path, SIMPLE_BVH))
    return retarget(bvh)


# ---------------------------------------------------------------------------
# Blueprint structure tests
# ---------------------------------------------------------------------------


class TestBuildBlueprint:
    """Tests for build_blueprint."""

    def test_skeleton_id(self, tmp_path: Path) -> None:
        animation = _retarget_simple(tmp_path)
        bp = build_blueprint(animation)
        assert bp["skeleton"] == SKELETON_ID

    def test_frame_count(self, tmp_path: Path) -> None:
        animation = _retarget_simple(tmp_path)
        bp = build_blueprint(animation)
        assert bp["frame_count"] == 2

    def test_frame_rate(self, tmp_path: Path) -> None:
        animation = _retarget_simple(tmp_path)
        bp = build_blueprint(animation)
        assert bp["frame_rate"] == pytest.approx(30.0, abs=0.1)

    def test_rotation_order(self, tmp_path: Path) -> None:
        animation = _retarget_simple(tmp_path)
        bp = build_blueprint(animation)
        assert bp["rotation_order"] == "ZXY"

    def test_all_19_bones_per_frame(self, tmp_path: Path) -> None:
        animation = _retarget_simple(tmp_path)
        bp = build_blueprint(animation)
        for frame in bp["frames"]:
            for bone in STRATA_BONES:
                assert bone in frame, f"Missing bone: {bone}"

    def test_each_bone_has_rotation(self, tmp_path: Path) -> None:
        animation = _retarget_simple(tmp_path)
        bp = build_blueprint(animation)
        for frame in bp["frames"]:
            for bone in STRATA_BONES:
                assert "rotation" in frame[bone]
                assert len(frame[bone]["rotation"]) == 3

    def test_hips_has_position(self, tmp_path: Path) -> None:
        animation = _retarget_simple(tmp_path)
        bp = build_blueprint(animation)
        for frame in bp["frames"]:
            assert "position" in frame["hips"]
            assert len(frame["hips"]["position"]) == 3

    def test_non_hips_no_position(self, tmp_path: Path) -> None:
        animation = _retarget_simple(tmp_path)
        bp = build_blueprint(animation)
        for frame in bp["frames"]:
            for bone in STRATA_BONES:
                if bone != "hips":
                    assert "position" not in frame[bone]

    def test_values_are_rounded(self, tmp_path: Path) -> None:
        animation = _retarget_simple(tmp_path)
        bp = build_blueprint(animation)
        pos = bp["frames"][0]["hips"]["position"]
        # Values should be rounded to 4 decimal places
        for v in pos:
            rounded = round(v, 4)
            assert v == rounded


# ---------------------------------------------------------------------------
# File export tests
# ---------------------------------------------------------------------------


class TestExportBlueprint:
    """Tests for export_blueprint."""

    def test_creates_file(self, tmp_path: Path) -> None:
        animation = _retarget_simple(tmp_path)
        out = tmp_path / "output" / "walk.json"
        result = export_blueprint(animation, out)
        assert result.exists()
        assert result == out

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        animation = _retarget_simple(tmp_path)
        out = tmp_path / "deep" / "nested" / "dir" / "anim.json"
        export_blueprint(animation, out)
        assert out.exists()

    def test_valid_json(self, tmp_path: Path) -> None:
        animation = _retarget_simple(tmp_path)
        out = tmp_path / "test.json"
        export_blueprint(animation, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["skeleton"] == SKELETON_ID
        assert len(data["frames"]) == 2

    def test_string_path_accepted(self, tmp_path: Path) -> None:
        animation = _retarget_simple(tmp_path)
        out = str(tmp_path / "test.json")
        result = export_blueprint(animation, out)
        assert result.exists()

    def test_roundtrip_consistency(self, tmp_path: Path) -> None:
        """Blueprint JSON should be identical to build_blueprint output."""
        animation = _retarget_simple(tmp_path)
        out = tmp_path / "test.json"
        export_blueprint(animation, out)

        loaded = json.loads(out.read_text(encoding="utf-8"))
        expected = build_blueprint(animation)

        assert loaded == expected
