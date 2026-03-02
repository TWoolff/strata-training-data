"""Tests for the 2D-to-3D measurement extraction pipeline.

Exercises apparent measurement extraction from segmentation masks and
training pair construction without requiring Blender or actual renders.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.measurement_extractor import (
    build_training_pairs,
    extract_apparent_measurements,
    load_ground_truth,
    save_apparent_measurements,
    save_training_pairs,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
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
        "upper_arm_l": {
            "width": 0.10,
            "depth": 0.10,
            "height": 0.25,
            "center": [-0.3, 0.0, 1.5],
            "vertex_count": 200,
        },
    },
    "total_vertices": 5000,
    "measured_regions": 3,
}


def _make_mask(regions: dict[int, tuple[int, int, int, int]], size: int = 64) -> np.ndarray:
    """Create a test segmentation mask with specified region bounding boxes.

    Args:
        regions: Mapping of region_id -> (x_min, y_min, x_max, y_max).
            The region fills the entire bounding box with solid pixels.
        size: Mask dimensions (square).

    Returns:
        uint8 array of shape (size, size).
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    for region_id, (x_min, y_min, x_max, y_max) in regions.items():
        mask[y_min : y_max + 1, x_min : x_max + 1] = region_id
    return mask


# ---------------------------------------------------------------------------
# extract_apparent_measurements
# ---------------------------------------------------------------------------


class TestExtractApparentMeasurements:
    """Test per-image apparent measurement extraction."""

    def test_single_region(self) -> None:
        mask = _make_mask({1: (10, 5, 30, 25)})
        result = extract_apparent_measurements(mask, camera_angle="front")

        assert result["camera_angle"] == "front"
        assert result["azimuth"] == 0

        head = result["regions"]["head"]
        assert head["visible"] is True
        assert head["apparent_width"] == 21  # 30 - 10 + 1
        assert head["apparent_height"] == 21  # 25 - 5 + 1
        assert head["bbox"] == [10, 5, 30, 25]
        assert head["pixel_count"] == 21 * 21

    def test_multiple_regions(self) -> None:
        mask = _make_mask({1: (10, 5, 20, 15), 3: (30, 40, 50, 60)})
        result = extract_apparent_measurements(mask, camera_angle="front")

        assert result["regions"]["head"]["visible"] is True
        assert result["regions"]["chest"]["visible"] is True
        assert result["regions"]["head"]["apparent_width"] == 11
        assert result["regions"]["chest"]["apparent_width"] == 21

    def test_missing_region_marked_invisible(self) -> None:
        mask = _make_mask({1: (10, 5, 20, 15)})
        result = extract_apparent_measurements(mask, camera_angle="front")

        chest = result["regions"]["chest"]
        assert chest["visible"] is False
        assert chest["apparent_width"] == 0
        assert chest["apparent_height"] == 0
        assert chest["bbox"] is None
        assert chest["pixel_count"] == 0

    def test_all_21_regions_present(self) -> None:
        mask = _make_mask({1: (10, 5, 20, 15)})
        result = extract_apparent_measurements(mask, camera_angle="front")

        # Should have entries for all 21 body regions (excluding background)
        assert len(result["regions"]) == 21

    def test_background_excluded(self) -> None:
        mask = _make_mask({0: (0, 0, 63, 63)})
        result = extract_apparent_measurements(mask, camera_angle="front")
        assert "background" not in result["regions"]

    def test_side_view_azimuth(self) -> None:
        mask = _make_mask({1: (10, 5, 20, 15)})
        result = extract_apparent_measurements(mask, camera_angle="side")
        assert result["azimuth"] == 90

    def test_three_quarter_view(self) -> None:
        mask = _make_mask({1: (10, 5, 20, 15)})
        result = extract_apparent_measurements(mask, camera_angle="three_quarter")
        assert result["azimuth"] == 45

    def test_unknown_camera_angle_raises(self) -> None:
        mask = _make_mask({1: (10, 5, 20, 15)})
        with pytest.raises(ValueError, match="Unknown camera angle"):
            extract_apparent_measurements(mask, camera_angle="top_down")

    def test_empty_mask(self) -> None:
        mask = np.zeros((64, 64), dtype=np.uint8)
        result = extract_apparent_measurements(mask, camera_angle="front")

        for _region_name, data in result["regions"].items():
            assert data["visible"] is False
            assert data["pixel_count"] == 0

    def test_single_pixel_region(self) -> None:
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[32, 32] = 1
        result = extract_apparent_measurements(mask, camera_angle="front")

        head = result["regions"]["head"]
        assert head["visible"] is True
        assert head["apparent_width"] == 1
        assert head["apparent_height"] == 1
        assert head["pixel_count"] == 1
        assert head["bbox"] == [32, 32, 32, 32]

    def test_non_contiguous_region(self) -> None:
        """Region with disjoint pixel clusters — bbox spans the full extent."""
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[5, 10] = 1
        mask[50, 60] = 1
        result = extract_apparent_measurements(mask, camera_angle="front")

        head = result["regions"]["head"]
        assert head["visible"] is True
        assert head["apparent_width"] == 51  # 60 - 10 + 1
        assert head["apparent_height"] == 46  # 50 - 5 + 1
        assert head["pixel_count"] == 2


# ---------------------------------------------------------------------------
# build_training_pairs
# ---------------------------------------------------------------------------


class TestBuildTrainingPairs:
    """Test training pair construction from apparent + ground truth data."""

    def test_basic_pairing(self) -> None:
        mask = _make_mask({1: (10, 5, 30, 25), 3: (30, 40, 50, 60)})
        apparent = extract_apparent_measurements(mask, camera_angle="front")
        pairs = build_training_pairs(apparent, GROUND_TRUTH, character_id="mixamo_ybot")

        # head and chest are visible and have ground truth
        assert len(pairs) == 2

        head_pair = next(p for p in pairs if p["region"] == "head")
        assert head_pair["character_id"] == "mixamo_ybot"
        assert head_pair["camera_angle"] == "front"
        assert head_pair["azimuth"] == 0
        assert head_pair["apparent_width"] == 21
        assert head_pair["true_width"] == 0.25
        assert head_pair["true_depth"] == 0.22
        assert head_pair["true_height"] == 0.28

    def test_skips_invisible_regions(self) -> None:
        mask = _make_mask({1: (10, 5, 30, 25)})
        apparent = extract_apparent_measurements(mask, camera_angle="front")
        pairs = build_training_pairs(apparent, GROUND_TRUTH, character_id="test")

        # Only head is visible; chest and upper_arm_l are not
        regions = [p["region"] for p in pairs]
        assert "head" in regions
        assert "chest" not in regions

    def test_skips_regions_without_ground_truth(self) -> None:
        mask = _make_mask({1: (10, 5, 30, 25), 5: (30, 40, 50, 60)})
        apparent = extract_apparent_measurements(mask, camera_angle="front")
        pairs = build_training_pairs(apparent, GROUND_TRUTH, character_id="test")

        # hips (5) is visible but has no ground truth
        regions = [p["region"] for p in pairs]
        assert "head" in regions
        assert "hips" not in regions

    def test_empty_ground_truth(self) -> None:
        mask = _make_mask({1: (10, 5, 30, 25)})
        apparent = extract_apparent_measurements(mask, camera_angle="front")
        pairs = build_training_pairs(apparent, {"regions": {}}, character_id="test")
        assert pairs == []

    def test_preserves_camera_angle(self) -> None:
        mask = _make_mask({1: (10, 5, 30, 25)})
        apparent = extract_apparent_measurements(mask, camera_angle="side")
        pairs = build_training_pairs(apparent, GROUND_TRUTH, character_id="test")

        assert pairs[0]["camera_angle"] == "side"
        assert pairs[0]["azimuth"] == 90


# ---------------------------------------------------------------------------
# save_apparent_measurements
# ---------------------------------------------------------------------------


class TestSaveApparentMeasurements:
    """Test per-image measurement file output."""

    def test_saves_json(self, tmp_path: Path) -> None:
        mask = _make_mask({1: (10, 5, 30, 25)})
        data = extract_apparent_measurements(mask, camera_angle="front")

        path = save_apparent_measurements(data, tmp_path, "mixamo_ybot", "pose_01", "front")

        assert path.is_file()
        assert path.name == "mixamo_ybot_pose_01_front.json"
        assert path.parent.name == "measurements_2d"

        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["character_id"] == "mixamo_ybot"
        assert saved["pose"] == "pose_01"
        assert saved["camera_angle"] == "front"
        assert "head" in saved["regions"]

    def test_creates_directory(self, tmp_path: Path) -> None:
        data = extract_apparent_measurements(_make_mask({1: (10, 5, 20, 15)}), camera_angle="front")
        path = save_apparent_measurements(
            data, tmp_path / "nested" / "deep", "char", "pose_01", "front"
        )
        assert path.is_file()


# ---------------------------------------------------------------------------
# save_training_pairs
# ---------------------------------------------------------------------------


class TestSaveTrainingPairs:
    """Test aggregated training pair output."""

    def test_saves_json(self, tmp_path: Path) -> None:
        pairs = [
            {
                "character_id": "test",
                "region": "head",
                "camera_angle": "front",
                "azimuth": 0,
                "apparent_width": 21,
                "apparent_height": 21,
                "pixel_count": 441,
                "true_width": 0.25,
                "true_depth": 0.22,
                "true_height": 0.28,
            }
        ]
        path = tmp_path / "pairs.json"
        save_training_pairs(pairs, path)

        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["version"] == "1.0"
        assert saved["pair_count"] == 1
        assert len(saved["pairs"]) == 1
        assert saved["pairs"][0]["region"] == "head"

    def test_empty_pairs(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.json"
        save_training_pairs([], path)

        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["pair_count"] == 0
        assert saved["pairs"] == []

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "a" / "b" / "pairs.json"
        save_training_pairs([], path)
        assert path.is_file()


# ---------------------------------------------------------------------------
# load_ground_truth
# ---------------------------------------------------------------------------


class TestLoadGroundTruth:
    """Test ground truth file loading and validation."""

    def test_valid_file(self, tmp_path: Path) -> None:
        path = tmp_path / "mixamo_ybot.json"
        path.write_text(json.dumps(GROUND_TRUTH), encoding="utf-8")

        result = load_ground_truth(tmp_path, "mixamo_ybot")
        assert result is not None
        assert "head" in result["regions"]

    def test_missing_file(self, tmp_path: Path) -> None:
        result = load_ground_truth(tmp_path, "nonexistent")
        assert result is None

    def test_malformed_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not valid", encoding="utf-8")
        result = load_ground_truth(tmp_path, "bad")
        assert result is None

    def test_missing_regions_key(self, tmp_path: Path) -> None:
        path = tmp_path / "no_regions.json"
        path.write_text(json.dumps({"total_vertices": 100}), encoding="utf-8")
        result = load_ground_truth(tmp_path, "no_regions")
        assert result is None

    def test_non_dict_root(self, tmp_path: Path) -> None:
        path = tmp_path / "list.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        result = load_ground_truth(tmp_path, "list")
        assert result is None
