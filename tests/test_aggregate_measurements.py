"""Tests for the measurement aggregation script.

Exercises the pure-Python aggregation logic without requiring Blender
or actual pipeline output.
"""

from __future__ import annotations

import json
from pathlib import Path

from mesh.scripts.aggregate_measurements import (
    aggregate_measurements,
    build_character_entry,
    infer_source,
    parse_measurement_file,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SAMPLE_MEASUREMENT = {
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
    "character_id": "mixamo_ybot",
}


def _write_measurement(path: Path, data: dict) -> Path:
    """Write a measurement JSON file and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# infer_source
# ---------------------------------------------------------------------------


class TestInferSource:
    """Test source inference from character ID prefix."""

    def test_mixamo(self) -> None:
        assert infer_source("mixamo_ybot") == "mixamo"

    def test_vroid(self) -> None:
        assert infer_source("vroid_char001") == "vroid"

    def test_stdgen(self) -> None:
        assert infer_source("stdgen_model042") == "stdgen"

    def test_nova_human(self) -> None:
        assert infer_source("nova_human_test001") == "nova_human"

    def test_unknown_prefix(self) -> None:
        assert infer_source("custom_model") == "unknown"

    def test_no_underscore(self) -> None:
        assert infer_source("someid") == "unknown"


# ---------------------------------------------------------------------------
# parse_measurement_file
# ---------------------------------------------------------------------------


class TestParseMeasurementFile:
    """Test reading and validation of per-character JSONs."""

    def test_valid_file(self, tmp_path: Path) -> None:
        path = _write_measurement(tmp_path / "char.json", SAMPLE_MEASUREMENT)
        result = parse_measurement_file(path)
        assert result is not None
        assert result["regions"]["head"]["width"] == 0.25

    def test_missing_regions_key(self, tmp_path: Path) -> None:
        path = _write_measurement(tmp_path / "bad.json", {"total_vertices": 100})
        result = parse_measurement_file(path)
        assert result is None

    def test_malformed_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{invalid json", encoding="utf-8")
        result = parse_measurement_file(path)
        assert result is None

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        result = parse_measurement_file(tmp_path / "nope.json")
        assert result is None

    def test_non_object_root(self, tmp_path: Path) -> None:
        path = tmp_path / "list.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        result = parse_measurement_file(path)
        assert result is None


# ---------------------------------------------------------------------------
# build_character_entry
# ---------------------------------------------------------------------------


class TestBuildCharacterEntry:
    """Test character entry construction."""

    def test_basic_entry(self) -> None:
        entry = build_character_entry("mixamo_ybot", SAMPLE_MEASUREMENT)
        assert entry["character_id"] == "mixamo_ybot"
        assert entry["source"] == "mixamo"
        assert entry["total_vertices"] == 5000
        assert entry["measured_regions"] == 2
        assert "head" in entry["measurements"]
        assert "chest" in entry["measurements"]

    def test_strips_internal_fields(self) -> None:
        entry = build_character_entry("mixamo_ybot", SAMPLE_MEASUREMENT)
        head = entry["measurements"]["head"]
        assert "center" not in head
        assert "vertex_count" not in head
        assert "width" in head
        assert "depth" in head
        assert "height" in head

    def test_missing_optional_fields(self) -> None:
        data = {"regions": {"head": {"width": 0.1}}}
        entry = build_character_entry("test_char", data)
        assert entry["total_vertices"] == 0
        assert entry["measured_regions"] == 0
        assert entry["measurements"]["head"]["depth"] == 0.0


# ---------------------------------------------------------------------------
# aggregate_measurements
# ---------------------------------------------------------------------------


class TestAggregateMeasurements:
    """Test the full aggregation pipeline."""

    def test_aggregate_multiple_characters(self, tmp_path: Path) -> None:
        measurements_dir = tmp_path / "measurements"
        output_path = tmp_path / "profiles.json"

        for name in ["mixamo_ybot", "vroid_char001", "stdgen_m042"]:
            data = {**SAMPLE_MEASUREMENT, "character_id": name}
            _write_measurement(measurements_dir / f"{name}.json", data)

        profiles = aggregate_measurements(measurements_dir, output_path)

        assert profiles["version"] == "1.0"
        assert profiles["character_count"] == 3
        assert len(profiles["characters"]) == 3
        assert output_path.is_file()

        # Verify sorted by character_id
        ids = [c["character_id"] for c in profiles["characters"]]
        assert ids == sorted(ids)

    def test_aggregate_empty_directory(self, tmp_path: Path) -> None:
        measurements_dir = tmp_path / "measurements"
        measurements_dir.mkdir()
        output_path = tmp_path / "profiles.json"

        profiles = aggregate_measurements(measurements_dir, output_path)
        assert profiles["character_count"] == 0
        assert profiles["characters"] == []

    def test_skips_malformed_files(self, tmp_path: Path) -> None:
        measurements_dir = tmp_path / "measurements"

        # One valid, one malformed
        _write_measurement(
            measurements_dir / "mixamo_good.json",
            {**SAMPLE_MEASUREMENT, "character_id": "mixamo_good"},
        )
        bad = measurements_dir / "bad.json"
        bad.write_text("not json", encoding="utf-8")

        output_path = tmp_path / "profiles.json"
        profiles = aggregate_measurements(measurements_dir, output_path)
        assert profiles["character_count"] == 1

    def test_incremental_merge(self, tmp_path: Path) -> None:
        measurements_dir = tmp_path / "measurements"
        output_path = tmp_path / "profiles.json"

        # First run: 2 characters
        for name in ["mixamo_a", "mixamo_b"]:
            _write_measurement(
                measurements_dir / f"{name}.json",
                {**SAMPLE_MEASUREMENT, "character_id": name},
            )
        aggregate_measurements(measurements_dir, output_path)

        # Second run: add a third character
        new_dir = tmp_path / "measurements2"
        _write_measurement(
            new_dir / "vroid_c.json",
            {**SAMPLE_MEASUREMENT, "character_id": "vroid_c"},
        )
        profiles = aggregate_measurements(new_dir, output_path, incremental=True)

        assert profiles["character_count"] == 3
        ids = [c["character_id"] for c in profiles["characters"]]
        assert "mixamo_a" in ids
        assert "mixamo_b" in ids
        assert "vroid_c" in ids

    def test_incremental_overwrites_existing(self, tmp_path: Path) -> None:
        measurements_dir = tmp_path / "measurements"
        output_path = tmp_path / "profiles.json"

        # First run
        _write_measurement(
            measurements_dir / "mixamo_ybot.json",
            {**SAMPLE_MEASUREMENT, "character_id": "mixamo_ybot", "total_vertices": 100},
        )
        aggregate_measurements(measurements_dir, output_path)

        # Second run with updated data
        _write_measurement(
            measurements_dir / "mixamo_ybot.json",
            {**SAMPLE_MEASUREMENT, "character_id": "mixamo_ybot", "total_vertices": 999},
        )
        profiles = aggregate_measurements(measurements_dir, output_path, incremental=True)

        assert profiles["character_count"] == 1
        assert profiles["characters"][0]["total_vertices"] == 999

    def test_no_incremental_overwrites(self, tmp_path: Path) -> None:
        measurements_dir = tmp_path / "measurements"
        output_path = tmp_path / "profiles.json"

        # First run: 2 characters
        for name in ["mixamo_a", "mixamo_b"]:
            _write_measurement(
                measurements_dir / f"{name}.json",
                {**SAMPLE_MEASUREMENT, "character_id": name},
            )
        aggregate_measurements(measurements_dir, output_path)

        # Second run with incremental=False and only 1 character in a new dir
        new_dir = tmp_path / "measurements2"
        _write_measurement(
            new_dir / "vroid_c.json",
            {**SAMPLE_MEASUREMENT, "character_id": "vroid_c"},
        )
        profiles = aggregate_measurements(new_dir, output_path, incremental=False)

        assert profiles["character_count"] == 1
        assert profiles["characters"][0]["character_id"] == "vroid_c"

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        measurements_dir = tmp_path / "measurements"
        _write_measurement(
            measurements_dir / "test.json",
            {**SAMPLE_MEASUREMENT, "character_id": "test_char"},
        )
        output_path = tmp_path / "nested" / "deep" / "profiles.json"

        profiles = aggregate_measurements(measurements_dir, output_path)
        assert output_path.is_file()
        assert profiles["character_count"] == 1

    def test_character_id_from_filename(self, tmp_path: Path) -> None:
        measurements_dir = tmp_path / "measurements"
        # Data without character_id key — should fall back to filename stem
        data = {"regions": {"head": {"width": 0.1, "depth": 0.1, "height": 0.1}}}
        _write_measurement(measurements_dir / "custom_model.json", data)

        output_path = tmp_path / "profiles.json"
        profiles = aggregate_measurements(measurements_dir, output_path)

        assert profiles["character_count"] == 1
        assert profiles["characters"][0]["character_id"] == "custom_model"

    def test_output_has_generated_at(self, tmp_path: Path) -> None:
        measurements_dir = tmp_path / "measurements"
        measurements_dir.mkdir()
        output_path = tmp_path / "profiles.json"

        profiles = aggregate_measurements(measurements_dir, output_path)
        assert "generated_at" in profiles
