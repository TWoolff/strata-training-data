"""Tests for training/data/joint_dataset.py — loading, mapping, augmentation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from training.data.joint_dataset import _flip_joint_example, parse_joints_json
from training.data.transforms import BONE_TO_INDEX

try:
    import torch  # noqa: F401

    from training.data.joint_dataset import JointDataset, JointDatasetConfig

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_joints_json(
    joints: dict | None = None,
    image_size: tuple[int, int] = (512, 512),
) -> dict:
    """Create a minimal joints.json data dict."""
    if joints is None:
        joints = {
            "head": {"position": [256, 48], "confidence": 1.0, "visible": True},
            "neck": {"position": [256, 80], "confidence": 1.0, "visible": True},
            "chest": {"position": [256, 140], "confidence": 1.0, "visible": True},
            "hips": {"position": [256, 220], "confidence": 1.0, "visible": True},
            "shoulder_l": {"position": [200, 100], "confidence": 1.0, "visible": True},
            "upper_arm_l": {"position": [170, 130], "confidence": 1.0, "visible": True},
            "forearm_l": {"position": [150, 170], "confidence": 1.0, "visible": True},
            "hand_l": {"position": [140, 210], "confidence": 1.0, "visible": True},
            "shoulder_r": {"position": [312, 100], "confidence": 1.0, "visible": True},
            "upper_arm_r": {"position": [342, 130], "confidence": 1.0, "visible": True},
            "forearm_r": {"position": [362, 170], "confidence": 1.0, "visible": True},
            "hand_r": {"position": [372, 210], "confidence": 1.0, "visible": True},
            "upper_leg_l": {"position": [230, 280], "confidence": 1.0, "visible": True},
            "lower_leg_l": {"position": [225, 360], "confidence": 1.0, "visible": True},
            "foot_l": {"position": [220, 440], "confidence": 1.0, "visible": True},
            "upper_leg_r": {"position": [282, 280], "confidence": 1.0, "visible": True},
            "lower_leg_r": {"position": [287, 360], "confidence": 1.0, "visible": True},
            "foot_r": {"position": [292, 440], "confidence": 1.0, "visible": True},
            "spine": {"position": [256, 180], "confidence": 1.0, "visible": True},
        }
    return {"joints": joints, "bbox": [128, 20, 384, 480], "image_size": list(image_size)}


def _create_flat_dataset(tmp_path, num_chars=2, num_poses=2):
    """Create a minimal flat-layout dataset with images and joints."""
    from PIL import Image

    images_dir = tmp_path / "images"
    joints_dir = tmp_path / "joints"
    images_dir.mkdir()
    joints_dir.mkdir()

    for char_idx in range(num_chars):
        char_id = f"mixamo_{char_idx:03d}"
        for pose_idx in range(num_poses):
            stem = f"{char_id}_pose_{pose_idx:02d}"
            img_stem = f"{stem}_flat"

            # Create image
            img = Image.new("RGBA", (512, 512), (128, 128, 128, 255))
            img.save(images_dir / f"{img_stem}.png")

            # Create joints.json
            joints_data = _make_joints_json()
            (joints_dir / f"{stem}.json").write_text(json.dumps(joints_data))

    return tmp_path


# ---------------------------------------------------------------------------
# parse_joints_json (pure numpy, no torch needed)
# ---------------------------------------------------------------------------


class TestParseJointsJson:
    """Joint JSON parsing and 20-slot mapping."""

    def test_maps_19_joints(self, tmp_path):
        """Pipeline produces 19 joints, all should map to correct slots."""
        joints_data = _make_joints_json()
        joints_path = tmp_path / "test_joints.json"
        joints_path.write_text(json.dumps(joints_data))

        positions, visible = parse_joints_json(joints_path, 512)

        assert positions.shape == (20, 2)
        assert visible.shape == (20,)

        # 19 joints should be visible
        assert visible.sum() == 19.0

    def test_hair_back_always_absent(self, tmp_path):
        """Slot 19 (hair_back) should always be marked as absent."""
        joints_data = _make_joints_json()
        joints_path = tmp_path / "test_joints.json"
        joints_path.write_text(json.dumps(joints_data))

        _, visible = parse_joints_json(joints_path, 512)

        hair_back_idx = BONE_TO_INDEX["hair_back"]
        assert visible[hair_back_idx] == 0.0

    def test_position_normalization(self, tmp_path):
        """Positions should be normalized to [0, 1] range."""
        joints_data = _make_joints_json()
        joints_path = tmp_path / "test_joints.json"
        joints_path.write_text(json.dumps(joints_data))

        positions, _ = parse_joints_json(joints_path, 512)

        assert positions.min() >= 0.0
        assert positions.max() <= 1.0

        # Check specific position: head at [256, 48] / 512 = [0.5, 0.09375]
        head_idx = BONE_TO_INDEX["head"]
        assert positions[head_idx, 0] == pytest.approx(256 / 512)
        assert positions[head_idx, 1] == pytest.approx(48 / 512)

    def test_pipeline_name_mapping(self, tmp_path):
        """Pipeline names should map correctly via PIPELINE_TO_BONE."""
        joints = {"forearm_l": {"position": [100, 200], "visible": True}}
        joints_data = {"joints": joints, "image_size": [512, 512]}
        joints_path = tmp_path / "test_joints.json"
        joints_path.write_text(json.dumps(joints_data))

        positions, visible = parse_joints_json(joints_path, 512)

        forearm_idx = BONE_TO_INDEX["forearm_l"]
        assert visible[forearm_idx] == 1.0
        assert positions[forearm_idx, 0] == pytest.approx(100 / 512)


# ---------------------------------------------------------------------------
# Horizontal flip (pure numpy, no torch needed)
# ---------------------------------------------------------------------------


class TestFlipJointExample:
    """Horizontal flip with L/R joint swap."""

    def test_mirrors_x_positions(self):
        """Flipping should mirror x coordinates."""
        img = np.zeros((4, 4, 3), dtype=np.float32)
        positions = np.zeros((20, 2), dtype=np.float32)
        visible = np.zeros(20, dtype=np.float32)

        # Set head position at x=0.25
        head_idx = BONE_TO_INDEX["head"]
        positions[head_idx] = [0.25, 0.5]
        visible[head_idx] = 1.0

        _, new_pos, _ = _flip_joint_example(img, positions, visible)

        assert new_pos[head_idx, 0] == pytest.approx(0.75)  # 1.0 - 0.25
        assert new_pos[head_idx, 1] == pytest.approx(0.5)  # y unchanged

    def test_swaps_lr_joints(self):
        """Flipping should swap L/R joint slots."""
        img = np.zeros((4, 4, 3), dtype=np.float32)
        positions = np.zeros((20, 2), dtype=np.float32)
        visible = np.zeros(20, dtype=np.float32)

        # Set shoulder_l at [0.3, 0.4]
        sl_idx = BONE_TO_INDEX["shoulder_l"]
        sr_idx = BONE_TO_INDEX["shoulder_r"]
        positions[sl_idx] = [0.3, 0.4]
        visible[sl_idx] = 1.0
        positions[sr_idx] = [0.7, 0.4]
        visible[sr_idx] = 1.0

        _, new_pos, _new_vis = _flip_joint_example(img, positions, visible)

        # After flip: shoulder_l slot should have mirrored shoulder_r values
        # shoulder_r was at x=0.7, mirrored to 0.3
        assert new_pos[sl_idx, 0] == pytest.approx(0.3)  # was sr at 0.7, mirrored to 0.3
        assert new_pos[sr_idx, 0] == pytest.approx(0.7)  # was sl at 0.3, mirrored to 0.7

    def test_flips_image(self):
        """Image should be horizontally flipped."""
        img = np.zeros((4, 8, 3), dtype=np.float32)
        img[0, 0, 0] = 1.0  # mark top-left pixel

        positions = np.zeros((20, 2), dtype=np.float32)
        visible = np.zeros(20, dtype=np.float32)

        flipped_img, _, _ = _flip_joint_example(img, positions, visible)

        assert flipped_img[0, 7, 0] == pytest.approx(1.0)  # moved to top-right
        assert flipped_img[0, 0, 0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# JointDataset (requires torch)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestJointDataset:
    """Dataset loading and __getitem__."""

    def test_discovers_flat_layout(self, tmp_path):
        """Dataset should discover examples in flat layout."""
        _create_flat_dataset(tmp_path)

        config = JointDatasetConfig(split_seed=42)
        # Use all chars in train by setting ratio to 1.0
        config.split_ratios = (1.0, 0.0, 0.0)
        dataset = JointDataset([tmp_path], split="train", augment=False, config=config)

        # 2 chars * 2 poses = 4 examples, all in train
        assert len(dataset) == 4

    def test_getitem_returns_correct_keys(self, tmp_path):
        """__getitem__ should return all required keys."""
        _create_flat_dataset(tmp_path, num_chars=1, num_poses=1)

        config = JointDatasetConfig(split_seed=42)
        config.split_ratios = (1.0, 0.0, 0.0)
        dataset = JointDataset([tmp_path], split="train", augment=False, config=config)

        item = dataset[0]
        assert set(item.keys()) == {
            "image",
            "gt_positions",
            "gt_visible",
            "geo_positions",
            "gt_offsets",
        }

    def test_getitem_shapes(self, tmp_path):
        """__getitem__ should return correct tensor shapes."""
        _create_flat_dataset(tmp_path, num_chars=1, num_poses=1)

        config = JointDatasetConfig(split_seed=42)
        config.split_ratios = (1.0, 0.0, 0.0)
        dataset = JointDataset([tmp_path], split="train", augment=False, config=config)

        item = dataset[0]
        assert item["image"].shape == (3, 512, 512)
        assert item["gt_positions"].shape == (20, 2)
        assert item["gt_visible"].shape == (20,)
        assert item["geo_positions"].shape == (20, 2)
        assert item["gt_offsets"].shape == (2, 20)

    def test_offset_layout_dx_first(self, tmp_path):
        """gt_offsets should be [2, 20] with row 0 = dx, row 1 = dy."""
        _create_flat_dataset(tmp_path, num_chars=1, num_poses=1)

        config = JointDatasetConfig(split_seed=42, geo_noise_std=0.0)
        config.split_ratios = (1.0, 0.0, 0.0)
        dataset = JointDataset([tmp_path], split="train", augment=False, config=config)

        item = dataset[0]
        # With zero noise, geo = gt, so offsets should be zero
        assert item["gt_offsets"].abs().max() < 1e-6

    def test_geo_noise_calibrated(self, tmp_path):
        """Geometric noise should be approximately std=0.03."""
        _create_flat_dataset(tmp_path, num_chars=1, num_poses=1)

        config = JointDatasetConfig(split_seed=42, geo_noise_std=0.03)
        config.split_ratios = (1.0, 0.0, 0.0)
        dataset = JointDataset([tmp_path], split="train", augment=False, config=config)

        # Sample multiple times to estimate noise std
        diffs = []
        for _ in range(100):
            item = dataset[0]
            gt = item["gt_positions"].numpy()
            geo = item["geo_positions"].numpy()
            vis = item["gt_visible"].numpy()
            diff = (gt - geo)[vis > 0]
            diffs.append(diff)

        all_diffs = np.concatenate(diffs)
        empirical_std = np.std(all_diffs)
        assert 0.01 < empirical_std < 0.06  # Roughly ~0.03

    def test_invisible_joints_no_noise(self, tmp_path):
        """Invisible joints should not get geometric noise."""
        _create_flat_dataset(tmp_path, num_chars=1, num_poses=1)

        config = JointDatasetConfig(split_seed=42, geo_noise_std=0.1)
        config.split_ratios = (1.0, 0.0, 0.0)
        dataset = JointDataset([tmp_path], split="train", augment=False, config=config)

        item = dataset[0]
        hair_back_idx = BONE_TO_INDEX["hair_back"]

        # hair_back is always invisible and at position [0, 0]
        assert item["gt_visible"][hair_back_idx] == 0.0
        assert item["geo_positions"][hair_back_idx, 0] == pytest.approx(0.0)
        assert item["geo_positions"][hair_back_idx, 1] == pytest.approx(0.0)
