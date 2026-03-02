"""Tests for training/data/weight_dataset.py — feature construction and dataset loading."""

from __future__ import annotations

import json

import pytest

from training.data.weight_dataset import (
    NUM_BONES,
    NUM_FEATURES,
    WeightDatasetConfig,
    build_features,
)

# ---------------------------------------------------------------------------
# build_features
# ---------------------------------------------------------------------------


class TestBuildFeatures:
    """Per-vertex 31-dim feature tensor construction."""

    def _make_vertices(self, n: int = 4) -> list[dict]:
        """Create synthetic vertex data."""
        return [
            {
                "position": [i * 100, i * 50],
                "weights": {"hips": 0.6, "spine": 0.4} if i % 2 == 0 else {},
            }
            for i in range(n)
        ]

    def _make_joint_positions(self) -> dict[str, tuple[float, float]]:
        """Create synthetic joint positions."""
        return {
            "hips": (200.0, 300.0),
            "spine": (200.0, 250.0),
            "chest": (200.0, 200.0),
            "neck": (200.0, 150.0),
            "head": (200.0, 100.0),
        }

    def test_output_shapes(self):
        """Feature tensor and targets should have correct shapes."""
        vertices = self._make_vertices(10)
        joints = self._make_joint_positions()

        features, weights_target, conf_target, num_verts = build_features(
            vertices, joints, (512, 512), max_vertices=64
        )

        assert features.shape == (NUM_FEATURES, 64, 1)
        assert weights_target.shape == (NUM_BONES, 64)
        assert conf_target.shape == (64,)
        assert num_verts == 10

    def test_zero_padding(self):
        """Vertices beyond actual count should be zero-padded."""
        vertices = self._make_vertices(2)
        joints = self._make_joint_positions()

        features, weights_target, conf_target, num_verts = build_features(
            vertices, joints, (512, 512), max_vertices=32
        )

        assert num_verts == 2
        # Padded region should be all zeros
        assert features[:, 10:, :].sum() == 0.0
        assert weights_target[:, 10:].sum() == 0.0
        assert conf_target[10:].sum() == 0.0

    def test_position_normalized(self):
        """Positions should be normalized to [0, 1] range."""
        vertices = [
            {"position": [0, 0], "weights": {"hips": 1.0}},
            {"position": [100, 200], "weights": {"spine": 1.0}},
        ]
        joints = self._make_joint_positions()

        features, _, _, _ = build_features(vertices, joints, (512, 512), max_vertices=32)

        # Vertex 0 at min → normalized to 0
        assert features[0, 0, 0] == pytest.approx(0.0)
        assert features[1, 0, 0] == pytest.approx(0.0)

        # Vertex 1 at max → normalized to 1
        assert features[0, 1, 0] == pytest.approx(1.0)
        assert features[1, 1, 0] == pytest.approx(1.0)

    def test_bone_distances_computed(self):
        """Distance features (slots 2-21) should be non-zero for vertices near bones."""
        vertices = [
            {"position": [200, 300], "weights": {"hips": 1.0}},  # At hips position
        ]
        joints = {"hips": (200.0, 300.0), "spine": (200.0, 250.0)}

        features, _, _, _ = build_features(vertices, joints, (512, 512), max_vertices=32)

        # Distance to hips (slot 2, bone index 0) should be 0
        assert features[2, 0, 0] == pytest.approx(0.0)
        # Distance to spine (slot 3, bone index 1) should be > 0
        assert features[3, 0, 0] > 0.0

    def test_heat_diffusion_zeroed(self):
        """Heat diffusion features (slots 22-29) should be zero."""
        vertices = self._make_vertices(2)
        joints = self._make_joint_positions()

        features, _, _, _ = build_features(vertices, joints, (512, 512), max_vertices=32)

        # Slots 22-29 should be zero (runtime-only data)
        assert features[22:30, :, :].sum() == 0.0

    def test_gt_weights_normalized(self):
        """Ground truth weights should sum to 1.0 per vertex."""
        vertices = [
            {"position": [100, 100], "weights": {"hips": 0.8, "spine": 0.4}},
        ]
        joints = self._make_joint_positions()

        _, weights_target, conf_target, _ = build_features(
            vertices, joints, (512, 512), max_vertices=32
        )

        # Weights should be normalized to sum to 1.0
        from training.data.transforms import BONE_TO_INDEX

        hips_idx = BONE_TO_INDEX["hips"]
        spine_idx = BONE_TO_INDEX["spine"]
        total = weights_target[hips_idx, 0] + weights_target[spine_idx, 0]
        assert total == pytest.approx(1.0, abs=1e-6)

        # Confidence should be 1.0 for vertex with weights
        assert conf_target[0] == 1.0

    def test_empty_weights_zero_confidence(self):
        """Vertices without weights should have confidence 0."""
        vertices = [
            {"position": [100, 100], "weights": {}},
        ]
        joints = self._make_joint_positions()

        _, weights_target, conf_target, _ = build_features(
            vertices, joints, (512, 512), max_vertices=32
        )

        assert conf_target[0] == 0.0
        assert weights_target[:, 0].sum() == 0.0

    def test_max_vertices_truncation(self):
        """Should truncate to max_vertices when mesh has more."""
        vertices = self._make_vertices(100)
        joints = self._make_joint_positions()

        features, _, _, num_verts = build_features(vertices, joints, (512, 512), max_vertices=32)

        assert num_verts == 32
        assert features.shape == (NUM_FEATURES, 32, 1)

    def test_empty_vertices(self):
        """Should handle empty vertex list gracefully."""
        features, weights_target, conf_target, num_verts = build_features(
            [], {}, (512, 512), max_vertices=32
        )

        assert num_verts == 0
        assert features.sum() == 0.0
        assert weights_target.sum() == 0.0
        assert conf_target.sum() == 0.0


# ---------------------------------------------------------------------------
# WeightDatasetConfig
# ---------------------------------------------------------------------------


class TestWeightDatasetConfig:
    """Config parsing from dict."""

    def test_from_dict_defaults(self):
        """Default config values."""
        config = WeightDatasetConfig.from_dict({})
        assert config.split_seed == 42
        assert config.split_ratios == (0.8, 0.1, 0.1)
        assert config.max_vertices == 2048

    def test_from_dict_custom(self):
        """Custom config values."""
        config = WeightDatasetConfig.from_dict(
            {
                "data": {
                    "split_seed": 123,
                    "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
                    "max_vertices": 1024,
                }
            }
        )
        assert config.split_seed == 123
        assert config.split_ratios == (0.7, 0.15, 0.15)
        assert config.max_vertices == 1024


# ---------------------------------------------------------------------------
# Dataset discovery (filesystem)
# ---------------------------------------------------------------------------


class TestWeightDatasetDiscovery:
    """Dataset file discovery for flat and per-example layouts."""

    def test_flat_layout_discovery(self, tmp_path):
        """Should discover examples from flat layout."""
        from training.data.weight_dataset import _discover_flat

        weights_dir = tmp_path / "weights"
        joints_dir = tmp_path / "joints"
        weights_dir.mkdir()
        joints_dir.mkdir()

        # Create matching pairs
        for i in range(3):
            (weights_dir / f"char_{i:03d}_pose_00.json").write_text("{}")
            (joints_dir / f"char_{i:03d}_pose_00.json").write_text("{}")

        # One without matching joints (should be skipped)
        (weights_dir / "orphan_pose_00.json").write_text("{}")

        examples = _discover_flat(tmp_path)
        assert len(examples) == 3

    def test_per_example_layout_discovery(self, tmp_path):
        """Should discover examples from per-example layout."""
        from training.data.weight_dataset import _discover_per_example

        for i in range(3):
            ex_dir = tmp_path / f"example_{i:03d}"
            ex_dir.mkdir()
            (ex_dir / "weights.json").write_text("{}")
            (ex_dir / "joints.json").write_text("{}")

        # One without weights (should be skipped)
        missing_dir = tmp_path / "missing"
        missing_dir.mkdir()
        (missing_dir / "joints.json").write_text("{}")

        examples = _discover_per_example(tmp_path)
        assert len(examples) == 3

    def test_layout_detection(self, tmp_path):
        """Should detect layout type from directory structure."""
        from training.data.weight_dataset import _detect_layout

        # Flat layout
        (tmp_path / "weights").mkdir()
        assert _detect_layout(tmp_path) == "flat"

    def test_dataset_loads_examples(self, tmp_path):
        """Full dataset should load and return correct tensors."""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not installed")

        from training.data.weight_dataset import WeightDataset

        # Create per-example layout with real data
        for i in range(5):
            ex_dir = tmp_path / f"char_00{i}_pose_00"
            ex_dir.mkdir()
            weight_data = {
                "vertex_count": 4,
                "vertices": [
                    {"position": [j * 50, j * 30], "weights": {"hips": 0.7, "spine": 0.3}}
                    for j in range(4)
                ],
                "image_size": [512, 512],
            }
            joints_data = {
                "joints": {
                    "hips": {"position": [256, 400], "visible": True},
                    "spine": {"position": [256, 300], "visible": True},
                },
                "image_size": [512, 512],
            }
            (ex_dir / "weights.json").write_text(json.dumps(weight_data))
            (ex_dir / "joints.json").write_text(json.dumps(joints_data))

        dataset = WeightDataset([tmp_path], split="train")
        if len(dataset) == 0:
            pytest.skip("No examples in train split (all went to val/test)")

        sample = dataset[0]
        assert sample["features"].shape == (31, 2048, 1)
        assert sample["weights_target"].shape == (20, 2048)
        assert sample["confidence_target"].shape == (2048,)
        assert isinstance(sample["num_vertices"], int)
        assert sample["num_vertices"] == 4
