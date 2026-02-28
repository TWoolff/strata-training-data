"""Tests for scripts/compute_stats.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.compute_stats import (
    compute_coverage_report,
    compute_region_distribution,
    compute_stats,
    count_all_files,
    count_images_by_angle,
    count_images_by_source,
    count_images_by_style,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_image(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    """Create a small RGBA PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.randint(0, 255, (*size, 4), dtype=np.uint8)
    Image.fromarray(arr, "RGBA").save(path)


def _create_mask(path: Path, region_ids: list[int], size: int = 64) -> None:
    """Create a grayscale mask with specified region IDs.

    Divides the image into horizontal bands, one per region ID.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((size, size), dtype=np.uint8)
    band_h = size // len(region_ids) if region_ids else size
    for i, rid in enumerate(region_ids):
        arr[i * band_h : (i + 1) * band_h, :] = rid
    Image.fromarray(arr, "L").save(path)


def _create_source_meta(sources_dir: Path, char_id: str, source: str) -> None:
    """Create a source metadata JSON file."""
    sources_dir.mkdir(parents=True, exist_ok=True)
    meta = {"id": char_id, "source": source}
    (sources_dir / f"{char_id}.json").write_text(json.dumps(meta))


def _setup_basic_output(tmp_path: Path, name: str = "segmentation") -> Path:
    """Set up a basic output directory with images, masks, and sources."""
    out = tmp_path / name
    images = out / "images"
    masks = out / "masks"
    joints = out / "joints"
    sources = out / "sources"

    # Two characters, 2 poses, 2 styles each
    for char in ("mixamo_001", "mixamo_002"):
        _create_source_meta(sources, char, "mixamo")
        for pose in ("00", "01"):
            for style in ("flat", "cel"):
                _create_image(images / f"{char}_pose_{pose}_{style}.png")
            _create_mask(masks / f"{char}_pose_{pose}.png", [0, 1, 3, 5])
            (joints / f"{char}_pose_{pose}.json").parent.mkdir(parents=True, exist_ok=True)
            (joints / f"{char}_pose_{pose}.json").write_text("{}")

    return out


# ---------------------------------------------------------------------------
# File counting
# ---------------------------------------------------------------------------


class TestCountAllFiles:
    def test_empty_directory(self, tmp_path: Path) -> None:
        result = count_all_files(tmp_path)
        assert result == {}

    def test_single_source(self, tmp_path: Path) -> None:
        out = _setup_basic_output(tmp_path)
        result = count_all_files(out)
        # out itself has images/ and masks/ so it's discovered
        assert "segmentation" in result or "root" in result
        counts = next(iter(result.values()))
        assert counts["images"] == 8  # 2 chars × 2 poses × 2 styles
        assert counts["masks"] == 4  # 2 chars × 2 poses

    def test_multiple_sources(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        # Create two sub-sources
        for sub in ("source_a", "source_b"):
            images_dir = out / sub / "images"
            _create_image(images_dir / "img_001.png")

        result = count_all_files(out)
        assert "source_a" in result
        assert "source_b" in result


# ---------------------------------------------------------------------------
# Style distribution
# ---------------------------------------------------------------------------


class TestCountByStyle:
    def test_counts_styles_from_filenames(self, tmp_path: Path) -> None:
        out = _setup_basic_output(tmp_path)
        result = count_images_by_style(out)
        assert result.get("flat") == 4
        assert result.get("cel") == 4

    def test_empty_output(self, tmp_path: Path) -> None:
        result = count_images_by_style(tmp_path)
        assert result == {}


# ---------------------------------------------------------------------------
# Angle distribution
# ---------------------------------------------------------------------------


class TestCountByAngle:
    def test_default_front_angle(self, tmp_path: Path) -> None:
        out = _setup_basic_output(tmp_path)
        result = count_images_by_angle(out)
        # No angle token in filenames → all counted as "front"
        assert result.get("front") == 8

    def test_explicit_angle_in_filename(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        images = out / "images"
        _create_image(images / "char_001_pose_00_side_flat.png")
        _create_image(images / "char_001_pose_00_front_flat.png")
        result = count_images_by_angle(out)
        assert result.get("side") == 1
        assert result.get("front") == 1


# ---------------------------------------------------------------------------
# Source distribution
# ---------------------------------------------------------------------------


class TestCountBySource:
    def test_source_from_metadata(self, tmp_path: Path) -> None:
        out = _setup_basic_output(tmp_path)
        result = count_images_by_source(out)
        assert result.get("mixamo") == 8

    def test_source_from_prefix_fallback(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        images = out / "images"
        _create_image(images / "sketchfab_001_pose_00_flat.png")
        result = count_images_by_source(out)
        assert result.get("sketchfab") == 1


# ---------------------------------------------------------------------------
# Region distribution
# ---------------------------------------------------------------------------


class TestRegionDistribution:
    def test_basic_distribution(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        masks = out / "masks"
        # Create mask with regions 0 (background), 1 (head), 3 (chest)
        _create_mask(masks / "test_pose_00.png", [0, 1, 3])
        result = compute_region_distribution(out, sample_size=10)
        # Regions 1 and 3 should have non-zero fractions
        assert result.get("head", 0) > 0
        assert result.get("chest", 0) > 0

    def test_empty_masks_dir(self, tmp_path: Path) -> None:
        result = compute_region_distribution(tmp_path)
        assert result == {}


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------


class TestCoverageReport:
    def test_identifies_missing_regions(self) -> None:
        distribution = {"head": 0.5, "chest": 0.3}
        coverage = compute_coverage_report(distribution)
        # Most regions should be missing since only head and chest provided
        assert len(coverage["missing"]) > 0

    def test_identifies_under_represented(self) -> None:
        # Create distribution with one region at 0.5% (under 1%)
        distribution = {"head": 0.005, "chest": 0.5}
        coverage = compute_coverage_report(distribution)
        assert "head" in coverage["under_represented"]


# ---------------------------------------------------------------------------
# Full stats
# ---------------------------------------------------------------------------


class TestComputeStats:
    def test_full_stats(self, tmp_path: Path) -> None:
        out = _setup_basic_output(tmp_path)
        stats = compute_stats(out)
        assert stats["totals"]["images"] == 8
        assert stats["totals"]["masks"] == 4
        assert "images_by_style" in stats
        assert "region_distribution" in stats
        assert "coverage" in stats

    def test_empty_output(self, tmp_path: Path) -> None:
        stats = compute_stats(tmp_path)
        assert stats["totals"]["images"] == 0
