"""Tests for scripts/verify_downloads.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from scripts.verify_downloads import (
    DATASET_SPECS,
    DatasetCheckResult,
    DatasetSpec,
    VerificationReport,
    save_report,
    verify_all,
    verify_dataset,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_png(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    """Create a small valid PNG image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((*size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


# ---------------------------------------------------------------------------
# DatasetSpec registry
# ---------------------------------------------------------------------------


class TestDatasetSpecs:
    def test_all_known_datasets_have_specs(self) -> None:
        expected = {
            "nova_human",
            "stdgen",
            "animerun",
            "unirig",
            "linkto_anime",
            "fbanimehq",
            "anime_segmentation",
            "anime_instance_seg",
            "charactergen",
        }
        assert set(DATASET_SPECS.keys()) == expected

    def test_specs_are_frozen_dataclasses(self) -> None:
        for spec in DATASET_SPECS.values():
            assert isinstance(spec, DatasetSpec)


# ---------------------------------------------------------------------------
# Single dataset verification
# ---------------------------------------------------------------------------


class TestVerifyDataset:
    def test_missing_directory(self, tmp_path: Path) -> None:
        spec = DatasetSpec(name="nonexistent")
        result = verify_dataset(tmp_path, spec)
        assert not result.exists
        assert not result.passed
        assert result.file_count == 0

    def test_empty_directory(self, tmp_path: Path) -> None:
        (tmp_path / "testds").mkdir()
        spec = DatasetSpec(name="testds")
        result = verify_dataset(tmp_path, spec)
        assert result.exists
        # No errors if no requirements
        assert result.passed

    def test_required_pattern_present(self, tmp_path: Path) -> None:
        ds_dir = tmp_path / "testds" / "ortho"
        ds_dir.mkdir(parents=True)
        _create_png(ds_dir / "view01.png")

        spec = DatasetSpec(
            name="testds",
            required_patterns=["ortho/*.png"],
        )
        result = verify_dataset(tmp_path, spec)
        assert result.passed
        assert not result.missing_required

    def test_required_pattern_missing(self, tmp_path: Path) -> None:
        (tmp_path / "testds").mkdir()
        spec = DatasetSpec(
            name="testds",
            required_patterns=["ortho/*.png"],
        )
        result = verify_dataset(tmp_path, spec)
        assert not result.passed
        assert len(result.missing_required) == 1

    def test_min_files_satisfied(self, tmp_path: Path) -> None:
        ds_dir = tmp_path / "testds"
        ds_dir.mkdir()
        for i in range(5):
            _create_png(ds_dir / f"img_{i}.png")

        spec = DatasetSpec(name="testds", min_files=5)
        result = verify_dataset(tmp_path, spec)
        assert result.passed
        assert result.file_count == 5

    def test_min_files_not_satisfied(self, tmp_path: Path) -> None:
        ds_dir = tmp_path / "testds"
        ds_dir.mkdir()
        _create_png(ds_dir / "img_0.png")

        spec = DatasetSpec(name="testds", min_files=10)
        result = verify_dataset(tmp_path, spec)
        assert not result.passed
        assert any("at least 10" in e for e in result.missing_required)

    def test_corrupt_image_detected(self, tmp_path: Path) -> None:
        ds_dir = tmp_path / "testds"
        ds_dir.mkdir()
        corrupt = ds_dir / "bad.png"
        corrupt.write_bytes(b"not a png file")

        spec = DatasetSpec(name="testds", expected_formats=[".png"])
        result = verify_dataset(tmp_path, spec)
        assert len(result.format_errors) == 1

    def test_readme_excluded_from_count(self, tmp_path: Path) -> None:
        ds_dir = tmp_path / "testds"
        ds_dir.mkdir()
        (ds_dir / "README.md").write_text("hello")

        spec = DatasetSpec(name="testds")
        result = verify_dataset(tmp_path, spec)
        assert result.file_count == 0


# ---------------------------------------------------------------------------
# DatasetCheckResult
# ---------------------------------------------------------------------------


class TestDatasetCheckResult:
    def test_errors_aggregation(self) -> None:
        result = DatasetCheckResult(
            name="test",
            exists=False,
            missing_required=["no files matching 'x'"],
            format_errors=["corrupt image a.png"],
        )
        errors = result.errors
        assert len(errors) == 3
        assert "directory does not exist" in errors[0]


# ---------------------------------------------------------------------------
# VerificationReport
# ---------------------------------------------------------------------------


class TestVerificationReport:
    def test_all_passed_true(self) -> None:
        report = VerificationReport(
            results=[
                DatasetCheckResult(name="a", passed=True),
                DatasetCheckResult(name="b", passed=True),
            ]
        )
        assert report.all_passed
        assert report.passed_count == 2
        assert report.failed_count == 0

    def test_all_passed_false(self) -> None:
        report = VerificationReport(
            results=[
                DatasetCheckResult(name="a", passed=True),
                DatasetCheckResult(name="b", passed=False),
            ]
        )
        assert not report.all_passed
        assert report.passed_count == 1
        assert report.failed_count == 1


# ---------------------------------------------------------------------------
# Full verification
# ---------------------------------------------------------------------------


class TestVerifyAll:
    def test_empty_data_dir(self, tmp_path: Path) -> None:
        report = verify_all(tmp_path)
        assert report.total_datasets == len(DATASET_SPECS)
        # All fail because no directories exist
        assert report.failed_count == report.total_datasets


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------


class TestSaveReport:
    def test_save_and_load(self, tmp_path: Path) -> None:
        report = VerificationReport(
            results=[
                DatasetCheckResult(name="a", exists=True, file_count=5, passed=True),
                DatasetCheckResult(name="b", exists=False, passed=False),
            ]
        )
        path = tmp_path / "report.json"
        save_report(report, path)

        import json

        data = json.loads(path.read_text())
        assert data["total_datasets"] == 2
        assert data["passed"] == 1
        assert data["failed"] == 1
        assert len(data["datasets"]) == 2
