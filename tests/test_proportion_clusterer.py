"""Tests for the proportion clustering script.

Exercises the pure-Python clustering logic without requiring Blender
or actual pipeline output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mesh.scripts.proportion_clusterer import (
    FEATURE_NAMES,
    assign_cluster_label,
    build_feature_matrix,
    cluster_profiles,
    compute_proportion_features,
    impute_and_scale,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_measurements(
    *,
    head_h: float = 0.28,
    chest_w: float = 0.35,
    chest_h: float = 0.30,
    spine_h: float = 0.20,
    hips_w: float = 0.30,
    hips_h: float = 0.15,
    upper_arm_h: float = 0.25,
    lower_arm_h: float = 0.22,
    hand_h: float = 0.10,
    upper_leg_h: float = 0.35,
    lower_leg_h: float = 0.30,
    foot_h: float = 0.08,
    shoulder_w: float = 0.12,
    head_w: float = 0.25,
    arm_w: float = 0.08,
    leg_w: float = 0.12,
) -> dict[str, dict[str, float]]:
    """Build a complete measurements dict with configurable dimensions."""
    return {
        "head": {"width": head_w, "depth": 0.22, "height": head_h},
        "neck": {"width": 0.10, "depth": 0.10, "height": 0.08},
        "chest": {"width": chest_w, "depth": 0.20, "height": chest_h},
        "spine": {"width": 0.30, "depth": 0.18, "height": spine_h},
        "hips": {"width": hips_w, "depth": 0.22, "height": hips_h},
        "upper_arm_l": {"width": arm_w, "depth": 0.08, "height": upper_arm_h},
        "forearm_l": {"width": arm_w, "depth": 0.07, "height": lower_arm_h},
        "hand_l": {"width": 0.06, "depth": 0.03, "height": hand_h},
        "upper_arm_r": {"width": arm_w, "depth": 0.08, "height": upper_arm_h},
        "forearm_r": {"width": arm_w, "depth": 0.07, "height": lower_arm_h},
        "hand_r": {"width": 0.06, "depth": 0.03, "height": hand_h},
        "upper_leg_l": {"width": leg_w, "depth": 0.12, "height": upper_leg_h},
        "lower_leg_l": {"width": leg_w * 0.8, "depth": 0.10, "height": lower_leg_h},
        "foot_l": {"width": 0.10, "depth": 0.25, "height": foot_h},
        "upper_leg_r": {"width": leg_w, "depth": 0.12, "height": upper_leg_h},
        "lower_leg_r": {"width": leg_w * 0.8, "depth": 0.10, "height": lower_leg_h},
        "foot_r": {"width": 0.10, "depth": 0.25, "height": foot_h},
        "shoulder_l": {"width": shoulder_w, "depth": 0.08, "height": 0.06},
        "shoulder_r": {"width": shoulder_w, "depth": 0.08, "height": 0.06},
    }


def _make_character(
    character_id: str,
    measurements: dict[str, dict[str, float]] | None = None,
    measured_regions: int | None = None,
) -> dict[str, Any]:
    """Build a character entry matching the profiles schema."""
    if measurements is None:
        measurements = _make_measurements()
    if measured_regions is None:
        measured_regions = len(measurements)
    return {
        "character_id": character_id,
        "source": "mixamo",
        "measurements": measurements,
        "total_vertices": 5000,
        "measured_regions": measured_regions,
    }


def _make_profiles(characters: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a complete profiles dict."""
    return {
        "version": "1.0",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "character_count": len(characters),
        "characters": characters,
    }


# ---------------------------------------------------------------------------
# compute_proportion_features
# ---------------------------------------------------------------------------


class TestComputeProportionFeatures:
    """Test proportion ratio feature extraction."""

    def test_all_features_computed(self) -> None:
        measurements = _make_measurements()
        features = compute_proportion_features(measurements)
        for name in FEATURE_NAMES:
            assert features[name] is not None, f"Feature {name} should not be None"
            assert isinstance(features[name], float), f"Feature {name} should be float"

    def test_head_to_body_height(self) -> None:
        measurements = _make_measurements(head_h=0.30, chest_h=0.30, spine_h=0.20, hips_h=0.15)
        features = compute_proportion_features(measurements)
        # torso_h = 0.30 + 0.20 + 0.15 = 0.65
        expected = 0.30 / 0.65
        assert features["head_to_body_height"] == pytest.approx(expected, rel=1e-3)

    def test_missing_region_returns_none(self) -> None:
        # Remove head entirely
        measurements = _make_measurements()
        del measurements["head"]
        features = compute_proportion_features(measurements)
        assert features["head_to_body_height"] is None
        assert features["head_width_to_shoulder"] is None

    def test_zero_denominator_returns_none(self) -> None:
        measurements = _make_measurements(chest_h=0.0, spine_h=0.0, hips_h=0.0)
        features = compute_proportion_features(measurements)
        assert features["head_to_body_height"] is None

    def test_symmetry_averaging(self) -> None:
        """Left and right arm should be averaged."""
        measurements = _make_measurements()
        # Make left arm height different from right
        measurements["upper_arm_l"]["height"] = 0.20
        measurements["upper_arm_r"]["height"] = 0.30
        features = compute_proportion_features(measurements)
        # arm_upper_h should be avg of 0.20 and 0.30 = 0.25
        assert features["arm_length_to_torso"] is not None


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------


class TestBuildFeatureMatrix:
    """Test feature matrix construction from profiles."""

    def test_basic_matrix(self) -> None:
        chars = [_make_character(f"char_{i}") for i in range(5)]
        profiles = _make_profiles(chars)
        matrix, ids, names = build_feature_matrix(profiles)
        assert matrix.shape == (5, len(FEATURE_NAMES))
        assert len(ids) == 5
        assert names == FEATURE_NAMES

    def test_skips_low_region_characters(self) -> None:
        chars = [
            _make_character("good_char"),
            _make_character("bad_char", measured_regions=3),
        ]
        profiles = _make_profiles(chars)
        matrix, ids, _names = build_feature_matrix(profiles)
        assert matrix.shape[0] == 1
        assert ids == ["good_char"]

    def test_empty_profiles(self) -> None:
        profiles = _make_profiles([])
        matrix, ids, names = build_feature_matrix(profiles)
        assert matrix.shape == (0, len(FEATURE_NAMES))
        assert ids == []
        assert names == FEATURE_NAMES


# ---------------------------------------------------------------------------
# impute_and_scale
# ---------------------------------------------------------------------------


class TestImputeAndScale:
    """Test NaN imputation and standard scaling."""

    def test_no_nans(self) -> None:
        matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaled, _medians, _stds = impute_and_scale(matrix)
        assert not np.any(np.isnan(scaled))
        assert scaled.shape == matrix.shape

    def test_with_nans(self) -> None:
        matrix = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
        scaled, medians, _stds = impute_and_scale(matrix)
        assert not np.any(np.isnan(scaled))
        # NaN in column 1 should be replaced with median of [4.0, 6.0] = 5.0
        assert medians[1] == pytest.approx(5.0)

    def test_constant_column(self) -> None:
        """Constant columns should not cause division by zero."""
        matrix = np.array([[1.0, 5.0], [1.0, 5.0], [1.0, 5.0]])
        scaled, _medians, _stds = impute_and_scale(matrix)
        assert not np.any(np.isnan(scaled))
        # All values same → scaled to 0
        assert np.allclose(scaled, 0.0)


# ---------------------------------------------------------------------------
# assign_cluster_label
# ---------------------------------------------------------------------------


class TestAssignClusterLabel:
    """Test automatic cluster labeling."""

    def test_chibi(self) -> None:
        centroid = {"head_to_body_height": 0.6, "leg_length_to_torso": 0.8}
        assert assign_cluster_label(centroid) == "chibi"

    def test_realistic(self) -> None:
        centroid = {
            "head_to_body_height": 0.25,
            "leg_length_to_torso": 1.3,
            "shoulder_to_hip_width": 2.5,
            "arm_thickness_ratio": 0.20,
            "leg_thickness_ratio": 0.25,
        }
        assert assign_cluster_label(centroid) == "realistic"

    def test_stylized(self) -> None:
        centroid = {
            "head_to_body_height": 0.40,
            "leg_length_to_torso": 1.0,
            "shoulder_to_hip_width": 2.0,
            "arm_thickness_ratio": 0.20,
            "leg_thickness_ratio": 0.25,
        }
        assert assign_cluster_label(centroid) == "stylized"

    def test_tall_thin(self) -> None:
        centroid = {
            "head_to_body_height": 0.20,
            "leg_length_to_torso": 1.8,
            "shoulder_to_hip_width": 2.0,
            "arm_thickness_ratio": 0.15,
            "leg_thickness_ratio": 0.15,
        }
        assert assign_cluster_label(centroid) == "tall_thin"

    def test_muscular(self) -> None:
        centroid = {
            "head_to_body_height": 0.25,
            "leg_length_to_torso": 1.2,
            "shoulder_to_hip_width": 4.0,
            "arm_thickness_ratio": 0.40,
            "leg_thickness_ratio": 0.40,
        }
        assert assign_cluster_label(centroid) == "muscular"


# ---------------------------------------------------------------------------
# cluster_profiles
# ---------------------------------------------------------------------------


class TestClusterProfiles:
    """Test the full clustering pipeline."""

    def _make_varied_profiles(self, n: int = 20) -> dict[str, Any]:
        """Create profiles with enough variation for meaningful clustering."""
        rng = np.random.default_rng(42)
        chars: list[dict[str, Any]] = []
        for i in range(n):
            # Create two distinct body types
            if i < n // 2:
                # Chibi-like: big head, short limbs
                m = _make_measurements(
                    head_h=0.40 + rng.uniform(-0.05, 0.05),
                    chest_h=0.20 + rng.uniform(-0.02, 0.02),
                    spine_h=0.12 + rng.uniform(-0.02, 0.02),
                    hips_h=0.10 + rng.uniform(-0.02, 0.02),
                    upper_leg_h=0.18 + rng.uniform(-0.02, 0.02),
                    lower_leg_h=0.15 + rng.uniform(-0.02, 0.02),
                )
            else:
                # Realistic: normal head, long limbs
                m = _make_measurements(
                    head_h=0.22 + rng.uniform(-0.02, 0.02),
                    chest_h=0.35 + rng.uniform(-0.02, 0.02),
                    spine_h=0.25 + rng.uniform(-0.02, 0.02),
                    hips_h=0.18 + rng.uniform(-0.02, 0.02),
                    upper_leg_h=0.40 + rng.uniform(-0.02, 0.02),
                    lower_leg_h=0.35 + rng.uniform(-0.02, 0.02),
                )
            chars.append(_make_character(f"char_{i:03d}", m))
        return _make_profiles(chars)

    def test_basic_clustering(self) -> None:
        profiles = self._make_varied_profiles(20)
        result = cluster_profiles(profiles)

        assert result["version"] == "1.0"
        assert result["method"] == "kmeans"
        assert result["optimal_k"] >= 2
        assert result["characters_analyzed"] == 20
        assert len(result["clusters"]) == result["optimal_k"]
        assert result["features_used"] == FEATURE_NAMES

        # All characters assigned to exactly one cluster
        all_chars = set()
        for cluster in result["clusters"]:
            for cid in cluster["characters"]:
                assert cid not in all_chars, f"{cid} assigned to multiple clusters"
                all_chars.add(cid)
        assert len(all_chars) == 20

    def test_forced_k(self) -> None:
        profiles = self._make_varied_profiles(20)
        result = cluster_profiles(profiles, k_override=3)
        assert result["optimal_k"] == 3
        assert len(result["clusters"]) == 3

    def test_empty_profiles(self) -> None:
        profiles = _make_profiles([])
        result = cluster_profiles(profiles)
        assert result["optimal_k"] == 0
        assert result["clusters"] == []
        assert result["characters_analyzed"] == 0

    def test_too_few_characters(self) -> None:
        chars = [_make_character(f"char_{i}") for i in range(3)]
        profiles = _make_profiles(chars)
        result = cluster_profiles(profiles)
        assert result["method"] == "single_cluster"
        assert result["optimal_k"] == 1
        assert len(result["clusters"]) == 1
        assert result["clusters"][0]["character_count"] == 3

    def test_cluster_labels_unique(self) -> None:
        profiles = self._make_varied_profiles(20)
        result = cluster_profiles(profiles, k_override=4)
        labels = [c["label"] for c in result["clusters"]]
        assert len(labels) == len(set(labels)), "Cluster labels must be unique"

    def test_centroids_are_proportion_ratios(self) -> None:
        profiles = self._make_varied_profiles(20)
        result = cluster_profiles(profiles)
        for cluster in result["clusters"]:
            centroid = cluster["centroid"]
            assert set(centroid.keys()) == set(FEATURE_NAMES)
            for val in centroid.values():
                assert isinstance(val, float)
                assert val >= 0, "Proportion ratios should be non-negative"

    def test_clusters_sorted_by_size(self) -> None:
        profiles = self._make_varied_profiles(20)
        result = cluster_profiles(profiles)
        counts = [c["character_count"] for c in result["clusters"]]
        assert counts == sorted(counts, reverse=True)

    def test_characters_sorted_within_clusters(self) -> None:
        profiles = self._make_varied_profiles(20)
        result = cluster_profiles(profiles)
        for cluster in result["clusters"]:
            chars = cluster["characters"]
            assert chars == sorted(chars)

    def test_output_serializable(self) -> None:
        profiles = self._make_varied_profiles(20)
        result = cluster_profiles(profiles)
        # Should be JSON-serializable without errors
        json_str = json.dumps(result, indent=2)
        roundtrip = json.loads(json_str)
        assert roundtrip["optimal_k"] == result["optimal_k"]

    def test_skipped_characters_counted(self) -> None:
        chars = [
            _make_character("good", measured_regions=19),
            _make_character("bad1", measured_regions=3),
            _make_character("bad2", measured_regions=5),
        ]
        profiles = _make_profiles(chars)
        result = cluster_profiles(profiles)
        assert result["characters_skipped"] == 2


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLI:
    """Test the CLI entry point."""

    def test_main_with_valid_input(self, tmp_path: Path) -> None:
        from mesh.scripts.proportion_clusterer import main

        # Create profiles with enough characters
        rng = np.random.default_rng(123)
        chars: list[dict[str, Any]] = []
        for i in range(10):
            m = _make_measurements(
                head_h=0.25 + rng.uniform(-0.05, 0.05),
                chest_h=0.30 + rng.uniform(-0.05, 0.05),
            )
            chars.append(_make_character(f"test_{i:03d}", m))

        profiles = _make_profiles(chars)
        input_path = tmp_path / "profiles.json"
        input_path.write_text(json.dumps(profiles, indent=2), encoding="utf-8")

        output_path = tmp_path / "clusters.json"
        viz_path = tmp_path / "clusters.png"

        main(
            [
                "-i",
                str(input_path),
                "-o",
                str(output_path),
                "--viz",
                str(viz_path),
            ]
        )

        assert output_path.is_file()
        result = json.loads(output_path.read_text(encoding="utf-8"))
        assert result["characters_analyzed"] == 10

    def test_main_no_viz(self, tmp_path: Path) -> None:
        from mesh.scripts.proportion_clusterer import main

        chars = [_make_character(f"test_{i}") for i in range(6)]
        profiles = _make_profiles(chars)
        input_path = tmp_path / "profiles.json"
        input_path.write_text(json.dumps(profiles, indent=2), encoding="utf-8")

        output_path = tmp_path / "clusters.json"
        viz_path = tmp_path / "clusters.png"

        main(
            [
                "-i",
                str(input_path),
                "-o",
                str(output_path),
                "--viz",
                str(viz_path),
                "--no-viz",
            ]
        )

        assert output_path.is_file()
        assert not viz_path.exists()

    def test_main_missing_input(self, tmp_path: Path) -> None:
        from mesh.scripts.proportion_clusterer import main

        with pytest.raises(SystemExit):
            main(["-i", str(tmp_path / "nonexistent.json")])
