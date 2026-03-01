"""Tests for training/data/transforms.py — L/R region swap, joint flip, normalization."""

from __future__ import annotations

import numpy as np
import pytest

from training.data.transforms import (
    BONE_ORDER,
    BONE_TO_INDEX,
    FLIP_JOINT_SWAP,
    FLIP_REGION_SWAP,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PIPELINE_TO_BONE,
    flip_joints,
    flip_mask,
    normalize_imagenet,
)

# ---------------------------------------------------------------------------
# flip_mask
# ---------------------------------------------------------------------------


class TestFlipMask:
    def test_round_trip_is_identity(self):
        """Flipping twice should return the original mask."""
        mask = np.array([[1, 6, 9], [12, 0, 15]], dtype=np.uint8)
        result = flip_mask(flip_mask(mask))
        np.testing.assert_array_equal(result, mask)

    def test_swaps_all_lr_pairs(self):
        """Each L/R pair should be swapped after flip."""
        # Build a mask with one pixel per swappable region
        lr_pairs = [(6, 9), (7, 10), (8, 11), (12, 15), (13, 16), (14, 17), (18, 19)]
        left_ids = [p[0] for p in lr_pairs]
        right_ids = [p[1] for p in lr_pairs]

        # Single row: [left_ids...]
        mask = np.array([left_ids], dtype=np.uint8)
        flipped = flip_mask(mask)

        # After flip: reversed order + swapped IDs
        expected = np.array([list(reversed(right_ids))], dtype=np.uint8)
        np.testing.assert_array_equal(flipped, expected)

    def test_center_regions_unchanged(self):
        """Center regions (head, neck, chest, etc.) should not swap."""
        center_ids = [0, 1, 2, 3, 4, 5]
        mask = np.array([center_ids], dtype=np.uint8)
        flipped = flip_mask(mask)
        # Should just be reversed (horizontal flip) but same IDs
        expected = np.array([list(reversed(center_ids))], dtype=np.uint8)
        np.testing.assert_array_equal(flipped, expected)

    def test_no_in_place_corruption(self):
        """Swapping must not corrupt data by overwriting before reading."""
        # All pixels region 6 (upper_arm_l) — after flip all should be 9
        mask = np.full((4, 4), 6, dtype=np.uint8)
        flipped = flip_mask(mask)
        assert np.all(flipped == 9)

        # Reverse: all 9 → all 6
        mask2 = np.full((4, 4), 9, dtype=np.uint8)
        flipped2 = flip_mask(mask2)
        assert np.all(flipped2 == 6)

    def test_empty_mask(self):
        """Empty mask should work without error."""
        mask = np.zeros((0, 0), dtype=np.uint8)
        result = flip_mask(mask)
        assert result.shape == (0, 0)

    def test_single_pixel(self):
        """Single pixel mask should swap correctly."""
        mask = np.array([[18]], dtype=np.uint8)  # shoulder_l
        result = flip_mask(mask)
        assert result[0, 0] == 19  # shoulder_r


# ---------------------------------------------------------------------------
# flip_joints
# ---------------------------------------------------------------------------


class TestFlipJoints:
    def test_round_trip_is_identity(self):
        """Flipping twice should return the original joints."""
        joints = {
            "head": {"x": 100, "y": 50},
            "upper_arm_l": {"x": 80, "y": 100},
            "upper_arm_r": {"x": 120, "y": 100},
        }
        width = 200
        result = flip_joints(flip_joints(joints, width), width)
        for name, pos in joints.items():
            assert result[name]["x"] == pos["x"]
            assert result[name]["y"] == pos["y"]

    def test_swaps_lr_names(self):
        """Left joints should become right and vice versa."""
        joints = {
            "upper_arm_l": {"x": 80, "y": 100},
            "hand_r": {"x": 150, "y": 200},
        }
        result = flip_joints(joints, 256)
        assert "upper_arm_r" in result
        assert "hand_l" in result
        assert "upper_arm_l" not in result
        assert "hand_r" not in result

    def test_mirrors_x_coordinate(self):
        """X coordinate should be mirrored: new_x = width - 1 - old_x."""
        joints = {"head": {"x": 10, "y": 50}}
        result = flip_joints(joints, 256)
        assert result["head"]["x"] == 245  # 256 - 1 - 10
        assert result["head"]["y"] == 50  # unchanged

    def test_preserves_visibility(self):
        """Visible flag should be preserved through flip."""
        joints = {
            "head": {"x": 100, "y": 50, "visible": True},
            "foot_l": {"x": 50, "y": 400, "visible": False},
        }
        result = flip_joints(joints, 512)
        assert result["head"]["visible"] is True
        assert result["foot_r"]["visible"] is False

    def test_center_joints_unchanged_name(self):
        """Center joints (head, neck, etc.) keep their name."""
        joints = {"head": {"x": 100, "y": 50}, "neck": {"x": 100, "y": 80}}
        result = flip_joints(joints, 200)
        assert "head" in result
        assert "neck" in result

    def test_all_swap_pairs_covered(self):
        """Every joint in FLIP_JOINT_SWAP should have a bidirectional pair."""
        for src, dst in FLIP_JOINT_SWAP.items():
            assert FLIP_JOINT_SWAP[dst] == src, f"Missing reverse for {src} ↔ {dst}"


# ---------------------------------------------------------------------------
# normalize_imagenet
# ---------------------------------------------------------------------------


try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestNormalizeImagenet:
    def test_output_shape_matches_input(self):
        """Output shape should match input shape."""
        img = torch.rand(3, 64, 64)
        result = normalize_imagenet(img)
        assert result.shape == img.shape

    def test_zero_image_gives_negative_mean_over_std(self):
        """All-zero image should give -mean/std for each channel."""
        img = torch.zeros(3, 4, 4)
        result = normalize_imagenet(img)
        for c in range(3):
            expected = -IMAGENET_MEAN[c] / IMAGENET_STD[c]
            assert torch.allclose(result[c], torch.full((4, 4), expected))

    def test_mean_image_gives_zero(self):
        """Image with ImageNet mean values should normalize to ~0."""
        img = torch.zeros(3, 4, 4)
        for c in range(3):
            img[c] = IMAGENET_MEAN[c]
        result = normalize_imagenet(img)
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)


# ---------------------------------------------------------------------------
# Constants validation
# ---------------------------------------------------------------------------


class TestConstants:
    def test_bone_order_length(self):
        """BONE_ORDER should have exactly 20 entries."""
        assert len(BONE_ORDER) == 20

    def test_bone_to_index_matches_bone_order(self):
        """BONE_TO_INDEX should be the inverse of BONE_ORDER."""
        for i, name in enumerate(BONE_ORDER):
            assert BONE_TO_INDEX[name] == i

    def test_flip_region_swap_bidirectional(self):
        """Every swap pair should be bidirectional."""
        for src, dst in FLIP_REGION_SWAP.items():
            assert FLIP_REGION_SWAP[dst] == src

    def test_flip_region_swap_pairs_count(self):
        """Should have 7 pairs (14 entries)."""
        assert len(FLIP_REGION_SWAP) == 14

    def test_pipeline_to_bone_forearm_mapping(self):
        """Key difference: lower_arm → forearm."""
        assert PIPELINE_TO_BONE["lower_arm_l"] == "forearm_l"
        assert PIPELINE_TO_BONE["lower_arm_r"] == "forearm_r"

    def test_pipeline_to_bone_covers_all_pipeline_regions(self):
        """Should cover all 19 non-background pipeline regions."""
        # Pipeline regions 1-19 (names from config)
        from pipeline.config import REGION_NAMES

        for region_id in range(1, 20):
            region_name = REGION_NAMES[region_id]
            assert region_name in PIPELINE_TO_BONE, f"Missing mapping for {region_name}"
