"""Tests for multi-view segmentation consistency validation.

Exercises region presence, pixel area consistency, and measurement ratio
checks using synthetic measurement data.  No Blender dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline.multiview_validator import (
    ConsistencyCheckSummary,
    ConsistencyFailure,
    ConsistencyReport,
    check_measurement_ratio,
    check_pixel_area_consistency,
    check_region_presence,
    save_consistency_report,
    validate_multiview_consistency,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GROUND_TRUTH = {
    "regions": {
        "head": {
            "width": 0.25,
            "depth": 0.22,
            "height": 0.28,
            "center": [0.0, 0.0, 1.8],
            "vertex_count": 500,
        },
        "chest": {
            "width": 0.35,
            "depth": 0.20,
            "height": 0.30,
            "center": [0.0, 0.0, 1.4],
            "vertex_count": 800,
        },
    },
    "total_vertices": 5000,
    "measured_regions": 2,
}


def _make_measurement(
    char_id: str,
    pose: str,
    angle: str,
    regions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Create a synthetic 2D measurement dict."""
    from pipeline.config import CAMERA_ANGLES

    azimuth = CAMERA_ANGLES[angle]["azimuth"]
    return {
        "character_id": char_id,
        "pose": pose,
        "camera_angle": angle,
        "azimuth": azimuth,
        "regions": regions,
    }


def _visible_region(
    width: int = 50,
    height: int = 60,
    pixel_count: int | None = None,
) -> dict[str, Any]:
    """Create a visible region measurement entry."""
    return {
        "apparent_width": width,
        "apparent_height": height,
        "bbox": [10, 10, 10 + width - 1, 10 + height - 1],
        "pixel_count": pixel_count if pixel_count is not None else width * height,
        "visible": True,
    }


def _invisible_region() -> dict[str, Any]:
    """Create an invisible region measurement entry."""
    return {
        "apparent_width": 0,
        "apparent_height": 0,
        "bbox": None,
        "pixel_count": 0,
        "visible": False,
    }


def _write_measurement_file(
    measurements_dir: Path,
    char_id: str,
    pose: str,
    angle: str,
    regions: dict[str, dict[str, Any]],
) -> None:
    """Write a synthetic 2D measurement JSON file to disk."""
    data = _make_measurement(char_id, pose, angle, regions)
    path = measurements_dir / f"{char_id}_{pose}_{angle}.json"
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# check_region_presence
# ---------------------------------------------------------------------------


class TestCheckRegionPresence:
    """Test midline region visibility across angles."""

    def test_all_midline_visible_passes(self) -> None:
        summary = ConsistencyCheckSummary(name="region_presence")
        angles = {
            "front": _make_measurement(
                "char",
                "pose_01",
                "front",
                {
                    "head": _visible_region(),
                    "neck": _visible_region(),
                    "chest": _visible_region(),
                    "spine": _visible_region(),
                    "hips": _visible_region(),
                },
            ),
            "side": _make_measurement(
                "char",
                "pose_01",
                "side",
                {
                    "head": _visible_region(),
                    "neck": _visible_region(),
                    "chest": _visible_region(),
                    "spine": _visible_region(),
                    "hips": _visible_region(),
                },
            ),
        }
        check_region_presence(angles, "char", "pose_01", summary)
        assert summary.failed == 0
        assert summary.passed > 0

    def test_midline_missing_in_one_angle_fails(self) -> None:
        summary = ConsistencyCheckSummary(name="region_presence")
        angles = {
            "front": _make_measurement(
                "char",
                "pose_01",
                "front",
                {
                    "head": _visible_region(),
                    "neck": _visible_region(),
                    "chest": _visible_region(),
                    "spine": _visible_region(),
                    "hips": _visible_region(),
                },
            ),
            "side": _make_measurement(
                "char",
                "pose_01",
                "side",
                {
                    "head": _visible_region(),
                    "neck": _invisible_region(),
                    "chest": _visible_region(),
                    "spine": _visible_region(),
                    "hips": _visible_region(),
                },
            ),
        }
        check_region_presence(angles, "char", "pose_01", summary)
        assert summary.failed == 1
        assert summary.failures[0].region == "neck"

    def test_single_angle_skipped(self) -> None:
        summary = ConsistencyCheckSummary(name="region_presence")
        angles = {
            "front": _make_measurement(
                "char",
                "pose_01",
                "front",
                {
                    "head": _visible_region(),
                },
            ),
        }
        check_region_presence(angles, "char", "pose_01", summary)
        assert summary.passed == 0
        assert summary.failed == 0

    def test_non_midline_not_checked(self) -> None:
        """Limb regions missing in some angles should NOT cause failures."""
        summary = ConsistencyCheckSummary(name="region_presence")
        angles = {
            "front": _make_measurement(
                "char",
                "pose_01",
                "front",
                {
                    "head": _visible_region(),
                    "upper_arm_l": _visible_region(),
                },
            ),
            "side": _make_measurement(
                "char",
                "pose_01",
                "side",
                {
                    "head": _visible_region(),
                    "upper_arm_l": _invisible_region(),
                },
            ),
        }
        check_region_presence(angles, "char", "pose_01", summary)
        # Only midline checked — upper_arm_l absence not flagged
        assert summary.failed == 0


# ---------------------------------------------------------------------------
# check_pixel_area_consistency
# ---------------------------------------------------------------------------


class TestCheckPixelAreaConsistency:
    """Test pixel count consistency across angles."""

    def test_consistent_pixel_counts_passes(self) -> None:
        summary = ConsistencyCheckSummary(name="pixel_area_consistency")
        angles = {
            "front": _make_measurement(
                "char",
                "pose_01",
                "front",
                {
                    "head": _visible_region(pixel_count=1000),
                },
            ),
            "side": _make_measurement(
                "char",
                "pose_01",
                "side",
                {
                    "head": _visible_region(pixel_count=1050),
                },
            ),
        }
        check_pixel_area_consistency(angles, "char", "pose_01", summary, threshold=0.10)
        assert summary.failed == 0
        assert summary.passed > 0

    def test_inconsistent_pixel_counts_fails(self) -> None:
        summary = ConsistencyCheckSummary(name="pixel_area_consistency")
        angles = {
            "front": _make_measurement(
                "char",
                "pose_01",
                "front",
                {
                    "head": _visible_region(pixel_count=1000),
                },
            ),
            "side": _make_measurement(
                "char",
                "pose_01",
                "side",
                {
                    "head": _visible_region(pixel_count=500),
                },
            ),
        }
        check_pixel_area_consistency(angles, "char", "pose_01", summary, threshold=0.10)
        assert summary.failed == 1
        assert "pixel count deviation" in summary.failures[0].detail

    def test_small_regions_skipped(self) -> None:
        """Regions with pixel count below MIN_PIXEL_COUNT are skipped."""
        summary = ConsistencyCheckSummary(name="pixel_area_consistency")
        angles = {
            "front": _make_measurement(
                "char",
                "pose_01",
                "front",
                {
                    "head": _visible_region(pixel_count=30),
                },
            ),
            "side": _make_measurement(
                "char",
                "pose_01",
                "side",
                {
                    "head": _visible_region(pixel_count=10),
                },
            ),
        }
        check_pixel_area_consistency(angles, "char", "pose_01", summary, threshold=0.10)
        # Both below MIN_PIXEL_COUNT — no checks performed
        assert summary.failed == 0
        assert summary.passed == 0

    def test_invisible_regions_skipped(self) -> None:
        summary = ConsistencyCheckSummary(name="pixel_area_consistency")
        angles = {
            "front": _make_measurement(
                "char",
                "pose_01",
                "front",
                {
                    "head": _visible_region(pixel_count=1000),
                },
            ),
            "side": _make_measurement(
                "char",
                "pose_01",
                "side",
                {
                    "head": _invisible_region(),
                },
            ),
        }
        check_pixel_area_consistency(angles, "char", "pose_01", summary, threshold=0.10)
        # head only visible in one angle — no pair to compare
        assert summary.failed == 0
        assert summary.passed == 0

    def test_multiple_regions_checked(self) -> None:
        summary = ConsistencyCheckSummary(name="pixel_area_consistency")
        angles = {
            "front": _make_measurement(
                "char",
                "pose_01",
                "front",
                {
                    "head": _visible_region(pixel_count=1000),
                    "chest": _visible_region(pixel_count=2000),
                },
            ),
            "side": _make_measurement(
                "char",
                "pose_01",
                "side",
                {
                    "head": _visible_region(pixel_count=1000),
                    "chest": _visible_region(pixel_count=2000),
                },
            ),
        }
        check_pixel_area_consistency(angles, "char", "pose_01", summary, threshold=0.10)
        assert summary.failed == 0
        assert summary.passed == 2

    def test_custom_threshold(self) -> None:
        summary = ConsistencyCheckSummary(name="pixel_area_consistency")
        angles = {
            "front": _make_measurement(
                "char",
                "pose_01",
                "front",
                {
                    "head": _visible_region(pixel_count=1000),
                },
            ),
            "side": _make_measurement(
                "char",
                "pose_01",
                "side",
                {
                    "head": _visible_region(pixel_count=800),
                },
            ),
        }
        # With 30% threshold, 22% deviation should pass
        check_pixel_area_consistency(angles, "char", "pose_01", summary, threshold=0.30)
        assert summary.failed == 0

    def test_three_angles_pairwise(self) -> None:
        """Three angles produce three pairwise comparisons per region."""
        summary = ConsistencyCheckSummary(name="pixel_area_consistency")
        angles = {
            "front": _make_measurement(
                "char",
                "pose_01",
                "front",
                {
                    "head": _visible_region(pixel_count=1000),
                },
            ),
            "side": _make_measurement(
                "char",
                "pose_01",
                "side",
                {
                    "head": _visible_region(pixel_count=1000),
                },
            ),
            "back": _make_measurement(
                "char",
                "pose_01",
                "back",
                {
                    "head": _visible_region(pixel_count=1000),
                },
            ),
        }
        check_pixel_area_consistency(angles, "char", "pose_01", summary, threshold=0.10)
        # 3 pairs: front-side, front-back, side-back
        assert summary.passed == 3
        assert summary.failed == 0


# ---------------------------------------------------------------------------
# check_measurement_ratio
# ---------------------------------------------------------------------------


class TestCheckMeasurementRatio:
    """Test 2D apparent measurements against 3D ground truth."""

    def test_consistent_ratio_passes(self) -> None:
        summary = ConsistencyCheckSummary(name="measurement_ratio")
        # GT: width=0.25, depth=0.22 → ratio = 1.136
        # Apparent: front_w=50, side_w=44 → ratio = 1.136
        angles = {
            "front": _make_measurement(
                "char",
                "pose_01",
                "front",
                {
                    "head": _visible_region(width=50),
                },
            ),
            "side": _make_measurement(
                "char",
                "pose_01",
                "side",
                {
                    "head": _visible_region(width=44),
                },
            ),
        }
        check_measurement_ratio(angles, GROUND_TRUTH, "char", "pose_01", summary)
        assert summary.failed == 0
        assert summary.passed > 0

    def test_inconsistent_ratio_fails(self) -> None:
        summary = ConsistencyCheckSummary(name="measurement_ratio")
        # GT: width=0.25, depth=0.22 → ratio = 1.136
        # Apparent: front_w=50, side_w=100 → ratio = 0.5 (way off)
        angles = {
            "front": _make_measurement(
                "char",
                "pose_01",
                "front",
                {
                    "head": _visible_region(width=50),
                },
            ),
            "side": _make_measurement(
                "char",
                "pose_01",
                "side",
                {
                    "head": _visible_region(width=100),
                },
            ),
        }
        check_measurement_ratio(angles, GROUND_TRUTH, "char", "pose_01", summary)
        assert summary.failed == 1
        assert "apparent ratio" in summary.failures[0].detail

    def test_missing_front_skips(self) -> None:
        summary = ConsistencyCheckSummary(name="measurement_ratio")
        angles = {
            "side": _make_measurement(
                "char",
                "pose_01",
                "side",
                {
                    "head": _visible_region(width=44),
                },
            ),
        }
        check_measurement_ratio(angles, GROUND_TRUTH, "char", "pose_01", summary)
        assert summary.failed == 0
        assert summary.passed == 0

    def test_missing_side_skips(self) -> None:
        summary = ConsistencyCheckSummary(name="measurement_ratio")
        angles = {
            "front": _make_measurement(
                "char",
                "pose_01",
                "front",
                {
                    "head": _visible_region(width=50),
                },
            ),
        }
        check_measurement_ratio(angles, GROUND_TRUTH, "char", "pose_01", summary)
        assert summary.failed == 0
        assert summary.passed == 0

    def test_invisible_region_skipped(self) -> None:
        summary = ConsistencyCheckSummary(name="measurement_ratio")
        angles = {
            "front": _make_measurement(
                "char",
                "pose_01",
                "front",
                {
                    "head": _visible_region(width=50),
                },
            ),
            "side": _make_measurement(
                "char",
                "pose_01",
                "side",
                {
                    "head": _invisible_region(),
                },
            ),
        }
        check_measurement_ratio(angles, GROUND_TRUTH, "char", "pose_01", summary)
        assert summary.passed == 0
        assert summary.failed == 0

    def test_no_ground_truth_for_region_skipped(self) -> None:
        summary = ConsistencyCheckSummary(name="measurement_ratio")
        angles = {
            "front": _make_measurement(
                "char",
                "pose_01",
                "front",
                {
                    "hips": _visible_region(width=50),
                },
            ),
            "side": _make_measurement(
                "char",
                "pose_01",
                "side",
                {
                    "hips": _visible_region(width=100),
                },
            ),
        }
        # hips not in GROUND_TRUTH
        check_measurement_ratio(angles, GROUND_TRUTH, "char", "pose_01", summary)
        assert summary.failed == 0
        assert summary.passed == 0


# ---------------------------------------------------------------------------
# ConsistencyReport
# ---------------------------------------------------------------------------


class TestConsistencyReport:
    """Test report aggregation and serialization."""

    def test_empty_report_passes(self) -> None:
        report = ConsistencyReport()
        assert report.all_passed is True
        assert report.total_failures == 0

    def test_report_with_failures(self) -> None:
        report = ConsistencyReport()
        summary = ConsistencyCheckSummary(name="test_check")
        summary.record_fail(
            ConsistencyFailure(
                character_id="char",
                pose="pose_01",
                region="head",
                check="test_check",
                angle_a="front",
                angle_b="side",
                detail="test detail",
            )
        )
        report.checks["test_check"] = summary
        assert report.all_passed is False
        assert report.total_failures == 1

    def test_worst_regions(self) -> None:
        report = ConsistencyReport()
        summary = ConsistencyCheckSummary(name="test")
        for _ in range(3):
            summary.record_fail(
                ConsistencyFailure(
                    character_id="c",
                    pose="p",
                    region="head",
                    check="test",
                    angle_a="front",
                    angle_b="side",
                    detail="d",
                )
            )
        summary.record_fail(
            ConsistencyFailure(
                character_id="c",
                pose="p",
                region="chest",
                check="test",
                angle_a="front",
                angle_b="side",
                detail="d",
            )
        )
        report.checks["test"] = summary

        worst = report.worst_regions(top_n=2)
        assert worst[0] == ("head", 3)
        assert worst[1] == ("chest", 1)

    def test_worst_angle_pairs(self) -> None:
        report = ConsistencyReport()
        summary = ConsistencyCheckSummary(name="test")
        for _ in range(2):
            summary.record_fail(
                ConsistencyFailure(
                    character_id="c",
                    pose="p",
                    region="head",
                    check="test",
                    angle_a="front",
                    angle_b="side",
                    detail="d",
                )
            )
        summary.record_fail(
            ConsistencyFailure(
                character_id="c",
                pose="p",
                region="head",
                check="test",
                angle_a="front",
                angle_b="back",
                detail="d",
            )
        )
        report.checks["test"] = summary

        worst = report.worst_angle_pairs(top_n=2)
        assert worst[0] == ("front vs side", 2)
        assert worst[1] == ("front vs back", 1)

    def test_to_dict_round_trips(self) -> None:
        report = ConsistencyReport()
        report.characters_checked = 3
        report.pose_groups_checked = 10
        report.elapsed_seconds = 1.23
        summary = ConsistencyCheckSummary(name="test")
        summary.record_pass()
        report.checks["test"] = summary

        d = report.to_dict()
        assert d["passed"] is True
        assert d["characters_checked"] == 3
        assert d["pose_groups_checked"] == 10
        assert d["checks"]["test"]["passed"] == 1


# ---------------------------------------------------------------------------
# save_consistency_report
# ---------------------------------------------------------------------------


class TestSaveConsistencyReport:
    """Test report file output."""

    def test_saves_json(self, tmp_path: Path) -> None:
        report = ConsistencyReport()
        report.characters_checked = 1
        summary = ConsistencyCheckSummary(name="test")
        summary.record_pass()
        report.checks["test"] = summary

        path = tmp_path / "report.json"
        save_consistency_report(report, path)

        assert path.is_file()
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["passed"] is True
        assert saved["characters_checked"] == 1

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        report = ConsistencyReport()
        path = tmp_path / "a" / "b" / "report.json"
        save_consistency_report(report, path)
        assert path.is_file()


# ---------------------------------------------------------------------------
# validate_multiview_consistency (integration)
# ---------------------------------------------------------------------------


class TestValidateMultiviewConsistency:
    """Integration tests using file-based measurement discovery."""

    def test_no_measurement_dir(self, tmp_path: Path) -> None:
        report = validate_multiview_consistency(tmp_path)
        assert report.all_passed is True
        assert report.pose_groups_checked == 0

    def test_single_angle_skipped(self, tmp_path: Path) -> None:
        m_dir = tmp_path / "measurements_2d"
        m_dir.mkdir()
        _write_measurement_file(
            m_dir,
            "char_a",
            "pose_01",
            "front",
            {
                "head": _visible_region(pixel_count=1000),
            },
        )
        report = validate_multiview_consistency(tmp_path)
        # Only one angle — nothing to compare
        assert report.all_passed is True
        assert report.pose_groups_checked == 0

    def test_two_angles_consistent(self, tmp_path: Path) -> None:
        m_dir = tmp_path / "measurements_2d"
        m_dir.mkdir()
        _write_measurement_file(
            m_dir,
            "char_a",
            "pose_01",
            "front",
            {
                "head": _visible_region(pixel_count=1000),
                "neck": _visible_region(pixel_count=200),
                "chest": _visible_region(pixel_count=2000),
                "spine": _visible_region(pixel_count=1500),
                "hips": _visible_region(pixel_count=1200),
            },
        )
        _write_measurement_file(
            m_dir,
            "char_a",
            "pose_01",
            "side",
            {
                "head": _visible_region(pixel_count=1000),
                "neck": _visible_region(pixel_count=200),
                "chest": _visible_region(pixel_count=2000),
                "spine": _visible_region(pixel_count=1500),
                "hips": _visible_region(pixel_count=1200),
            },
        )
        report = validate_multiview_consistency(tmp_path)
        assert report.all_passed is True
        assert report.characters_checked == 1
        assert report.pose_groups_checked == 1

    def test_two_angles_inconsistent(self, tmp_path: Path) -> None:
        m_dir = tmp_path / "measurements_2d"
        m_dir.mkdir()
        _write_measurement_file(
            m_dir,
            "char_a",
            "pose_01",
            "front",
            {
                "head": _visible_region(pixel_count=1000),
                "neck": _visible_region(pixel_count=200),
                "chest": _visible_region(pixel_count=2000),
                "spine": _visible_region(pixel_count=1500),
                "hips": _visible_region(pixel_count=1200),
            },
        )
        _write_measurement_file(
            m_dir,
            "char_a",
            "pose_01",
            "side",
            {
                "head": _visible_region(pixel_count=500),  # 67% deviation
                "neck": _visible_region(pixel_count=200),
                "chest": _visible_region(pixel_count=2000),
                "spine": _visible_region(pixel_count=1500),
                "hips": _visible_region(pixel_count=1200),
            },
        )
        report = validate_multiview_consistency(tmp_path)
        assert report.all_passed is False
        assert report.checks["pixel_area_consistency"].failed >= 1

    def test_character_filter(self, tmp_path: Path) -> None:
        m_dir = tmp_path / "measurements_2d"
        m_dir.mkdir()
        # Two characters
        _write_measurement_file(
            m_dir,
            "char_a",
            "pose_01",
            "front",
            {
                "head": _visible_region(pixel_count=1000),
            },
        )
        _write_measurement_file(
            m_dir,
            "char_a",
            "pose_01",
            "side",
            {
                "head": _visible_region(pixel_count=500),
            },
        )
        _write_measurement_file(
            m_dir,
            "char_b",
            "pose_01",
            "front",
            {
                "head": _visible_region(pixel_count=1000),
            },
        )
        _write_measurement_file(
            m_dir,
            "char_b",
            "pose_01",
            "side",
            {
                "head": _visible_region(pixel_count=1000),
            },
        )
        # Only validate char_b (consistent)
        report = validate_multiview_consistency(tmp_path, characters=["char_b"])
        assert report.all_passed is True
        assert report.characters_checked == 1

    def test_with_ground_truth(self, tmp_path: Path) -> None:
        m_dir = tmp_path / "measurements_2d"
        m_dir.mkdir()
        gt_dir = tmp_path / "measurements"
        gt_dir.mkdir()

        # Write ground truth
        gt_path = gt_dir / "char_a.json"
        gt_path.write_text(json.dumps(GROUND_TRUTH) + "\n", encoding="utf-8")

        # Consistent ratio: GT width/depth = 0.25/0.22 ≈ 1.136
        _write_measurement_file(
            m_dir,
            "char_a",
            "pose_01",
            "front",
            {
                "head": _visible_region(width=50, pixel_count=500),
            },
        )
        _write_measurement_file(
            m_dir,
            "char_a",
            "pose_01",
            "side",
            {
                "head": _visible_region(width=44, pixel_count=500),
            },
        )
        report = validate_multiview_consistency(tmp_path)
        assert report.checks["measurement_ratio"].failed == 0
        assert report.checks["measurement_ratio"].passed > 0

    def test_custom_threshold(self, tmp_path: Path) -> None:
        m_dir = tmp_path / "measurements_2d"
        m_dir.mkdir()
        _write_measurement_file(
            m_dir,
            "char_a",
            "pose_01",
            "front",
            {
                "head": _visible_region(pixel_count=1000),
            },
        )
        _write_measurement_file(
            m_dir,
            "char_a",
            "pose_01",
            "side",
            {
                "head": _visible_region(pixel_count=800),
            },
        )
        # With generous threshold of 30%, 22% deviation should pass
        report = validate_multiview_consistency(tmp_path, threshold=0.30)
        assert report.checks["pixel_area_consistency"].failed == 0
