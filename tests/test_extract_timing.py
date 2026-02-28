"""Tests for animation.scripts.extract_timing module."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pytest

from animation.scripts.bvh_parser import parse_bvh
from animation.scripts.extract_timing import (
    ActionNormStats,
    ActionTimingNorms,
    ClipTiming,
    ClipTimingStats,
    VelocityStats,
    aggregate_by_action,
    compute_joint_angular_velocity,
    compute_root_velocity,
    compute_timing_stats,
    extract_clip_timing,
    load_labels,
    write_timing_norms,
)

# ---------------------------------------------------------------------------
# Fixtures: synthetic BVH content
# ---------------------------------------------------------------------------

# Simple skeleton with root position channels and 5 frames showing movement.
# Root moves along X: 0 -> 1 -> 3 -> 6 -> 10 -> 15 (accelerating).
# Hips rotation also changes between frames.
WALK_BVH = textwrap.dedent("""\
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
                End Site
                {
                    OFFSET 0.0 4.0 0.0
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
    Frames: 5
    Frame Time: 0.0333333
    0.0 90.0 0.0  0.0 0.0 0.0  0.0 0.0 0.0  0.0 0.0 0.0  0.0 0.0 0.0
    1.0 90.0 0.0  1.0 2.0 3.0  5.0 5.0 5.0  10.0 10.0 10.0  0.0 0.0 0.0
    3.0 90.0 0.0  2.0 4.0 6.0  10.0 10.0 10.0  20.0 20.0 20.0  0.0 0.0 0.0
    6.0 90.0 0.0  3.0 6.0 9.0  15.0 15.0 15.0  30.0 30.0 30.0  0.0 0.0 0.0
    10.0 90.0 0.0  4.0 8.0 12.0  20.0 20.0 20.0  40.0 40.0 40.0  0.0 0.0 0.0
""")

# Minimal 2-frame BVH with root moving along X.
TWO_FRAME_BVH = textwrap.dedent("""\
    HIERARCHY
    ROOT Hips
    {
        OFFSET 0.0 0.0 0.0
        CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
        JOINT Head
        {
            OFFSET 0.0 10.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            End Site
            {
                OFFSET 0.0 4.0 0.0
            }
        }
    }
    MOTION
    Frames: 2
    Frame Time: 0.0333333
    0.0 90.0 0.0  0.0 0.0 0.0  0.0 0.0 0.0
    3.0 90.0 0.0  0.0 0.0 0.0  10.0 20.0 30.0
""")

# Single frame — edge case, cannot compute velocities.
SINGLE_FRAME_BVH = textwrap.dedent("""\
    HIERARCHY
    ROOT Hips
    {
        OFFSET 0.0 0.0 0.0
        CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
        End Site
        {
            OFFSET 0.0 10.0 0.0
        }
    }
    MOTION
    Frames: 1
    Frame Time: 0.0333333
    0.0 90.0 0.0  0.0 0.0 0.0
""")

# BVH with stationary root (all hold frames).
STATIONARY_BVH = textwrap.dedent("""\
    HIERARCHY
    ROOT Hips
    {
        OFFSET 0.0 0.0 0.0
        CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
        JOINT Head
        {
            OFFSET 0.0 10.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            End Site
            {
                OFFSET 0.0 4.0 0.0
            }
        }
    }
    MOTION
    Frames: 4
    Frame Time: 0.0333333
    0.0 90.0 0.0  0.0 0.0 0.0  5.0 10.0 15.0
    0.0 90.0 0.0  0.0 0.0 0.0  5.0 10.0 15.0
    0.0 90.0 0.0  0.0 0.0 0.0  5.0 10.0 15.0
    0.0 90.0 0.0  0.0 0.0 0.0  5.0 10.0 15.0
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_bvh(tmp_path: Path, content: str, name: str = "test.bvh") -> Path:
    bvh_path = tmp_path / name
    bvh_path.write_text(content, encoding="utf-8")
    return bvh_path


def _write_csv(tmp_path: Path, rows: list[dict[str, str]], name: str = "labels.csv") -> Path:
    csv_path = tmp_path / name
    header = "filename,action_type,subcategory,quality,strata_compatible\n"
    lines = [header]
    for row in rows:
        lines.append(
            f"{row['filename']},{row['action_type']},{row['subcategory']},"
            f"{row['quality']},{row['strata_compatible']}\n"
        )
    csv_path.write_text("".join(lines), encoding="utf-8")
    return csv_path


# ---------------------------------------------------------------------------
# CSV loading tests
# ---------------------------------------------------------------------------


class TestLoadLabels:
    """Tests for load_labels function."""

    def test_load_basic_csv(self, tmp_path: Path) -> None:
        csv_path = _write_csv(
            tmp_path,
            [
                {
                    "filename": "01_01.bvh",
                    "action_type": "walk",
                    "subcategory": "forward",
                    "quality": "high",
                    "strata_compatible": "yes",
                },
                {
                    "filename": "02_01.bvh",
                    "action_type": "run",
                    "subcategory": "forward",
                    "quality": "high",
                    "strata_compatible": "yes",
                },
            ],
        )
        labels = load_labels(csv_path)
        assert len(labels) == 2
        assert labels[0].filename == "01_01.bvh"
        assert labels[0].action_type == "walk"
        assert labels[1].action_type == "run"

    def test_missing_csv_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_labels(tmp_path / "nonexistent.csv")

    def test_empty_csv(self, tmp_path: Path) -> None:
        csv_path = _write_csv(tmp_path, [])
        labels = load_labels(csv_path)
        assert labels == []


# ---------------------------------------------------------------------------
# Root velocity tests
# ---------------------------------------------------------------------------


class TestRootVelocity:
    """Tests for compute_root_velocity."""

    def test_velocity_shape(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, WALK_BVH))
        vel = compute_root_velocity(bvh)
        assert vel.shape == (5,)

    def test_first_frame_zero(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, WALK_BVH))
        vel = compute_root_velocity(bvh)
        assert vel[0] == 0.0

    def test_velocity_positive(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, WALK_BVH))
        vel = compute_root_velocity(bvh)
        # Root moves along X: 0->1->3->6->10, all positive displacements
        assert all(v >= 0.0 for v in vel)

    def test_velocity_increases(self, tmp_path: Path) -> None:
        """Accelerating root should produce increasing velocity."""
        bvh = parse_bvh(_write_bvh(tmp_path, WALK_BVH))
        vel = compute_root_velocity(bvh)
        # Displacements: 1, 2, 3, 4 → velocities should increase
        for i in range(2, len(vel)):
            assert vel[i] > vel[i - 1]

    def test_two_frame_velocity(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, TWO_FRAME_BVH))
        vel = compute_root_velocity(bvh)
        assert vel.shape == (2,)
        assert vel[0] == 0.0
        # Root moves 3.0 along X in 0.0333333s
        expected = 3.0 / 0.0333333
        assert vel[1] == pytest.approx(expected, rel=1e-3)

    def test_single_frame_empty(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, SINGLE_FRAME_BVH))
        vel = compute_root_velocity(bvh)
        assert vel.size == 0

    def test_stationary_root_zero_velocity(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, STATIONARY_BVH))
        vel = compute_root_velocity(bvh)
        assert all(v == pytest.approx(0.0) for v in vel)

    def test_no_nan_or_inf(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, WALK_BVH))
        vel = compute_root_velocity(bvh)
        assert not np.any(np.isnan(vel))
        assert not np.any(np.isinf(vel))


# ---------------------------------------------------------------------------
# Joint angular velocity tests
# ---------------------------------------------------------------------------


class TestJointAngularVelocity:
    """Tests for compute_joint_angular_velocity."""

    def test_angular_velocity_shape(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, WALK_BVH))
        ang_vel = compute_joint_angular_velocity(bvh, "Spine")
        assert ang_vel.shape == (5,)

    def test_first_frame_zero(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, WALK_BVH))
        ang_vel = compute_joint_angular_velocity(bvh, "Spine")
        assert ang_vel[0] == 0.0

    def test_stationary_joint_zero(self, tmp_path: Path) -> None:
        """LeftUpLeg has constant (0,0,0) rotation across all frames."""
        bvh = parse_bvh(_write_bvh(tmp_path, WALK_BVH))
        ang_vel = compute_joint_angular_velocity(bvh, "LeftUpLeg")
        assert all(v == pytest.approx(0.0) for v in ang_vel)

    def test_nonexistent_joint_empty(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, WALK_BVH))
        ang_vel = compute_joint_angular_velocity(bvh, "NonexistentJoint")
        assert ang_vel.size == 0

    def test_two_frame_angular_velocity(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, TWO_FRAME_BVH))
        ang_vel = compute_joint_angular_velocity(bvh, "Head")
        assert ang_vel.shape == (2,)
        # Head rotations: frame0=(0,0,0), frame1=(10,20,30) in ZXY order
        # Delta = (10, 20, 30) → norm = sqrt(100+400+900) = sqrt(1400)
        frame_time = 0.0333333
        expected = np.sqrt(1400.0) / frame_time
        assert ang_vel[1] == pytest.approx(expected, rel=1e-3)

    def test_no_nan_or_inf(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, WALK_BVH))
        for joint_name in bvh.skeleton.joints:
            ang_vel = compute_joint_angular_velocity(bvh, joint_name)
            if ang_vel.size > 0:
                assert not np.any(np.isnan(ang_vel)), f"NaN in {joint_name}"
                assert not np.any(np.isinf(ang_vel)), f"Inf in {joint_name}"


# ---------------------------------------------------------------------------
# Timing stats tests
# ---------------------------------------------------------------------------


class TestTimingStats:
    """Tests for compute_timing_stats."""

    def test_accelerating_clip(self) -> None:
        # Monotonically increasing velocity
        vel = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        stats = compute_timing_stats(vel, hold_threshold=0.1)
        # First frame (vel=0.0) is a hold frame
        assert stats.hold_frames == 1
        # Remaining 4 frames all have positive derivative → acceleration
        assert stats.acceleration_frames == 4
        assert stats.deceleration_frames == 0

    def test_decelerating_clip(self) -> None:
        vel = np.array([4.0, 3.0, 2.0, 1.0, 0.0])
        stats = compute_timing_stats(vel, hold_threshold=0.1)
        assert stats.hold_frames == 1  # last frame vel=0.0
        assert stats.deceleration_frames == 3
        # Frame 0 has derivative 0 (padded), not a hold → neither accel nor decel
        assert stats.acceleration_frames == 0

    def test_all_hold(self) -> None:
        vel = np.array([0.0, 0.0, 0.0, 0.0])
        stats = compute_timing_stats(vel, hold_threshold=0.1)
        assert stats.hold_frames == 4
        assert stats.acceleration_frames == 0
        assert stats.deceleration_frames == 0
        assert stats.hold_ratio == pytest.approx(1.0)

    def test_ratios_sum_to_one_or_less(self) -> None:
        vel = np.array([0.0, 1.0, 3.0, 2.0, 0.5, 0.0])
        stats = compute_timing_stats(vel, hold_threshold=0.1)
        total = stats.acceleration_ratio + stats.deceleration_ratio + stats.hold_ratio
        assert total <= 1.0 + 1e-6

    def test_empty_input(self) -> None:
        stats = compute_timing_stats(np.array([]))
        assert stats.hold_frames == 0
        assert stats.acceleration_frames == 0

    def test_single_value(self) -> None:
        stats = compute_timing_stats(np.array([5.0]))
        assert stats.hold_frames == 0
        assert stats.acceleration_frames == 0


# ---------------------------------------------------------------------------
# Clip extraction integration tests
# ---------------------------------------------------------------------------


class TestExtractClipTiming:
    """Tests for extract_clip_timing."""

    def test_basic_extraction(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, WALK_BVH))
        clip = extract_clip_timing(bvh, "test_walk.bvh", "forward")
        assert clip.filename == "test_walk.bvh"
        assert clip.subcategory == "forward"
        assert clip.frame_count == 5
        assert clip.frame_rate == pytest.approx(30.0, abs=0.1)

    def test_duration(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, WALK_BVH))
        clip = extract_clip_timing(bvh, "test.bvh", "forward")
        expected = 5 * 0.0333333
        assert clip.duration_seconds == pytest.approx(expected, abs=0.01)

    def test_velocity_curve_populated(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, WALK_BVH))
        clip = extract_clip_timing(bvh, "test.bvh", "forward")
        assert len(clip.root_velocity_curve) == 5
        assert clip.root_velocity_curve[0] == 0.0

    def test_joint_angular_velocities_populated(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, WALK_BVH))
        clip = extract_clip_timing(bvh, "test.bvh", "forward")
        # Should have entries for Hips, Spine, Head, LeftUpLeg
        assert len(clip.joint_angular_velocities) > 0
        assert "Spine" in clip.joint_angular_velocities

    def test_timing_stats_present(self, tmp_path: Path) -> None:
        bvh = parse_bvh(_write_bvh(tmp_path, WALK_BVH))
        clip = extract_clip_timing(bvh, "test.bvh", "forward")
        assert isinstance(clip.timing, ClipTimingStats)
        total = (
            clip.timing.acceleration_frames
            + clip.timing.deceleration_frames
            + clip.timing.hold_frames
        )
        assert total <= clip.frame_count


# ---------------------------------------------------------------------------
# Aggregation tests
# ---------------------------------------------------------------------------


class TestAggregation:
    """Tests for aggregate_by_action."""

    def _make_clip(
        self, filename: str, subcategory: str, duration: float, vel_mean: float
    ) -> ClipTiming:
        return ClipTiming(
            filename=filename,
            subcategory=subcategory,
            duration_seconds=duration,
            frame_count=100,
            frame_rate=30.0,
            root_velocity=VelocityStats(mean=vel_mean, max=vel_mean * 2, min=0.0),
            timing=ClipTimingStats(
                acceleration_frames=30,
                deceleration_frames=20,
                hold_frames=10,
                acceleration_ratio=0.3,
                deceleration_ratio=0.2,
                hold_ratio=0.1,
            ),
        )

    def test_groups_by_action(self) -> None:
        clips = {
            "walk": [self._make_clip("01.bvh", "forward", 3.0, 1.0)],
            "run": [self._make_clip("02.bvh", "forward", 2.0, 3.0)],
        }
        norms = aggregate_by_action(clips)
        assert "walk" in norms
        assert "run" in norms
        assert norms["walk"].clip_count == 1
        assert norms["run"].clip_count == 1

    def test_norms_computed(self) -> None:
        clips = {
            "walk": [
                self._make_clip("01.bvh", "forward", 3.0, 1.0),
                self._make_clip("02.bvh", "slow", 4.0, 0.5),
            ],
        }
        norms = aggregate_by_action(clips)
        walk_norms = norms["walk"].norms
        assert "duration_seconds" in walk_norms
        assert walk_norms["duration_seconds"].mean == pytest.approx(3.5)
        assert walk_norms["duration_seconds"].min == pytest.approx(3.0)
        assert walk_norms["duration_seconds"].max == pytest.approx(4.0)

    def test_empty_input(self) -> None:
        norms = aggregate_by_action({})
        assert norms == {}


# ---------------------------------------------------------------------------
# Output tests
# ---------------------------------------------------------------------------


class TestWriteTimingNorms:
    """Tests for write_timing_norms."""

    def _make_norms(self) -> dict[str, ActionTimingNorms]:
        return {
            "walk": ActionTimingNorms(
                action_type="walk",
                clip_count=2,
                clips=[
                    ClipTiming(filename="01.bvh", subcategory="forward"),
                    ClipTiming(filename="02.bvh", subcategory="slow"),
                ],
                norms={
                    "duration_seconds": ActionNormStats(mean=3.5, std=0.5, min=3.0, max=4.0),
                },
            ),
            "run": ActionTimingNorms(
                action_type="run",
                clip_count=1,
                clips=[ClipTiming(filename="03.bvh", subcategory="forward")],
                norms={
                    "duration_seconds": ActionNormStats(mean=2.0, std=0.0, min=2.0, max=2.0),
                },
            ),
        }

    def test_creates_per_action_json(self, tmp_path: Path) -> None:
        norms = self._make_norms()
        write_timing_norms(norms, tmp_path / "output")
        assert (tmp_path / "output" / "walk.json").is_file()
        assert (tmp_path / "output" / "run.json").is_file()

    def test_creates_summary_json(self, tmp_path: Path) -> None:
        norms = self._make_norms()
        write_timing_norms(norms, tmp_path / "output")
        summary_path = tmp_path / "output" / "summary.json"
        assert summary_path.is_file()
        summary = json.loads(summary_path.read_text())
        assert "walk" in summary
        assert "run" in summary
        assert summary["walk"]["clip_count"] == 2

    def test_action_json_valid(self, tmp_path: Path) -> None:
        norms = self._make_norms()
        write_timing_norms(norms, tmp_path / "output")
        walk_data = json.loads((tmp_path / "output" / "walk.json").read_text())
        assert walk_data["action_type"] == "walk"
        assert walk_data["clip_count"] == 2
        assert len(walk_data["clips"]) == 2

    def test_returns_written_paths(self, tmp_path: Path) -> None:
        norms = self._make_norms()
        written = write_timing_norms(norms, tmp_path / "output")
        # 2 action types + 1 summary
        assert len(written) == 3

    def test_summary_excludes_clip_details(self, tmp_path: Path) -> None:
        norms = self._make_norms()
        write_timing_norms(norms, tmp_path / "output")
        summary = json.loads((tmp_path / "output" / "summary.json").read_text())
        assert "clips" not in summary["walk"]
        assert "norms" in summary["walk"]
