"""Tests for the Live2D renderer module.

These tests exercise the pure-Python rendering logic without requiring Blender.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

# Provide a mock bpy module so pipeline imports work outside Blender.
_bpy_mock = types.ModuleType("bpy")
_bpy_mock.types = MagicMock()
sys.modules.setdefault("bpy", _bpy_mock)

from pipeline.live2d_renderer import (  # noqa: E402
    _apply_color_jitter,
    _apply_flip,
    _apply_rotation,
    _apply_scale,
    _composite_from_fragments,
    _discover_fragment_images,
    _extract_joints_from_mask,
    _find_model_json,
    _parse_model_json,
    generate_augmentations,
    process_live2d_directory,
    process_live2d_model,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_model_dir(tmp_path: Path) -> Path:
    """Create a temporary Live2D model directory with fragment PNGs."""
    model_dir = tmp_path / "test_model"
    model_dir.mkdir()

    # Create fragment images (small colored rectangles)
    # Head fragment (red)
    head_img = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
    head_img.save(model_dir / "head.png")

    # Body/chest fragment (green)
    body_img = Image.new("RGBA", (80, 120), (0, 255, 0, 255))
    body_img.save(model_dir / "body.png")

    # Left arm fragment (blue)
    arm_img = Image.new("RGBA", (30, 80), (0, 0, 255, 255))
    arm_img.save(model_dir / "arm_upper_L.png")

    # Unknown fragment (should map to background)
    unknown_img = Image.new("RGBA", (20, 20), (128, 128, 128, 255))
    unknown_img.save(model_dir / "shadow_overlay.png")

    return model_dir


@pytest.fixture()
def tmp_model_json_dir(tmp_path: Path) -> Path:
    """Create a model directory with a .model3.json file."""
    model_dir = tmp_path / "json_model"
    model_dir.mkdir()

    model_json = {
        "Version": 3,
        "FileReferences": {
            "Moc": "model.moc3",
            "Textures": ["texture_00.png", "texture_01.png"],
        },
    }
    (model_dir / "json_model.model3.json").write_text(json.dumps(model_json), encoding="utf-8")

    # Create a small texture atlas
    atlas = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    atlas.save(model_dir / "texture_00.png")
    atlas.save(model_dir / "texture_01.png")

    return model_dir


@pytest.fixture()
def sample_mask() -> np.ndarray:
    """Create a simple segmentation mask for testing."""
    mask = np.zeros((512, 512), dtype=np.uint8)
    # Head region (1) in top-center
    mask[50:150, 200:312] = 1
    # Chest region (3) in middle
    mask[150:300, 180:332] = 3
    # Upper arm left (6) on the right side
    mask[170:280, 332:380] = 6
    # Upper arm right (9) on the left side
    mask[170:280, 132:180] = 9
    return mask


# ---------------------------------------------------------------------------
# Model JSON parsing
# ---------------------------------------------------------------------------


class TestParseModelJson:
    """Test .model3.json parsing."""

    def test_find_model_json(self, tmp_model_json_dir: Path) -> None:
        result = _find_model_json(tmp_model_json_dir)
        assert result is not None
        assert result.name == "json_model.model3.json"

    def test_find_model_json_missing(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = _find_model_json(empty_dir)
        assert result is None

    def test_parse_model_json(self, tmp_model_json_dir: Path) -> None:
        json_path = tmp_model_json_dir / "json_model.model3.json"
        model = _parse_model_json(json_path)
        assert model is not None
        assert model.name == "json_model"
        assert len(model.texture_paths) == 2
        assert model.texture_paths[0] == "texture_00.png"

    def test_parse_invalid_json(self, tmp_path: Path) -> None:
        bad_json = tmp_path / "bad.model3.json"
        bad_json.write_text("not json!", encoding="utf-8")
        result = _parse_model_json(bad_json)
        assert result is None


# ---------------------------------------------------------------------------
# Fragment discovery
# ---------------------------------------------------------------------------


class TestDiscoverFragments:
    """Test fragment image discovery in model directories."""

    def test_discovers_fragments(self, tmp_model_dir: Path) -> None:
        fragments = _discover_fragment_images(tmp_model_dir)
        names = [name for name, _ in fragments]
        assert "head" in names
        assert "body" in names
        assert "arm_upper_L" in names
        assert "shadow_overlay" in names

    def test_skips_large_atlases(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "atlas_model"
        model_dir.mkdir()
        # Create a large atlas image that should be skipped
        large_img = Image.new("RGBA", (4096, 4096), (0, 0, 0, 0))
        large_img.save(model_dir / "atlas.png")
        # Create a small fragment that should be found
        small_img = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
        small_img.save(model_dir / "head.png")

        fragments = _discover_fragment_images(model_dir)
        names = [name for name, _ in fragments]
        assert "atlas" not in names
        assert "head" in names

    def test_empty_directory(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        fragments = _discover_fragment_images(empty_dir)
        assert fragments == []


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------


class TestCompositeFromFragments:
    """Test character compositing from fragment images."""

    def test_basic_composite(self) -> None:
        head = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
        body = Image.new("RGBA", (80, 120), (0, 255, 0, 255))

        fragments = [
            ("head", head, 0),
            ("body", body, 1),
        ]
        region_map = {"head": 1, "body": 3}

        image, mask, draw_order = _composite_from_fragments(fragments, region_map, resolution=256)

        assert image.size == (256, 256)
        assert image.mode == "RGBA"
        assert mask.shape == (256, 256)
        assert mask.dtype == np.uint8
        assert draw_order.shape == (256, 256)

    def test_mask_has_correct_region_ids(self) -> None:
        head = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
        fragments = [("head", head, 0)]
        region_map = {"head": 1}

        _, mask, _ = _composite_from_fragments(fragments, region_map, resolution=128)

        unique_ids = set(np.unique(mask))
        assert 0 in unique_ids  # background
        assert 1 in unique_ids  # head

    def test_draw_order_values(self) -> None:
        frag1 = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
        frag2 = Image.new("RGBA", (64, 64), (0, 255, 0, 255))

        fragments = [
            ("back_part", frag1, 0),
            ("front_part", frag2, 10),
        ]
        region_map = {"back_part": 1, "front_part": 3}

        _, _, draw_order = _composite_from_fragments(fragments, region_map, resolution=128)

        # Draw order should have values from 0 to 255
        assert draw_order.max() <= 255
        assert draw_order.min() >= 0

    def test_empty_fragments(self) -> None:
        image, mask, draw_order = _composite_from_fragments([], {}, resolution=128)
        assert image.size == (128, 128)
        assert mask.max() == 0
        assert draw_order.max() == 0

    def test_unmapped_fragments_go_to_background(self) -> None:
        frag = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
        fragments = [("unknown_thing", frag, 0)]
        region_map = {"unknown_thing": 0}  # background

        _, mask, _ = _composite_from_fragments(fragments, region_map, resolution=128)

        # Should only have background (0)
        assert set(np.unique(mask)) == {0}


# ---------------------------------------------------------------------------
# Joint extraction from mask
# ---------------------------------------------------------------------------


class TestExtractJointsFromMask:
    """Test joint extraction via region centroid computation."""

    def test_basic_joint_extraction(self, sample_mask: np.ndarray) -> None:
        joint_data = _extract_joints_from_mask(sample_mask)

        assert "joints" in joint_data
        assert "bbox" in joint_data
        assert "image_size" in joint_data
        assert joint_data["image_size"] == [512, 512]

    def test_visible_joints(self, sample_mask: np.ndarray) -> None:
        joint_data = _extract_joints_from_mask(sample_mask)
        joints = joint_data["joints"]

        # Head (region 1) should be visible
        assert joints["head"]["visible"] is True
        assert joints["head"]["confidence"] > 0

        # Chest (region 3) should be visible
        assert joints["chest"]["visible"] is True

    def test_missing_joints(self, sample_mask: np.ndarray) -> None:
        joint_data = _extract_joints_from_mask(sample_mask)
        joints = joint_data["joints"]

        # Foot regions are not in our sample mask
        assert joints["foot_l"]["visible"] is False
        assert joints["foot_l"]["position"] == [-1, -1]

    def test_empty_mask(self) -> None:
        empty_mask = np.zeros((512, 512), dtype=np.uint8)
        joint_data = _extract_joints_from_mask(empty_mask)
        joints = joint_data["joints"]

        # All joints should be invisible
        for _region_name, joint in joints.items():
            assert joint["visible"] is False

    def test_joint_count(self, sample_mask: np.ndarray) -> None:
        joint_data = _extract_joints_from_mask(sample_mask)
        # Should have 19 joints (regions 1-19)
        assert len(joint_data["joints"]) == 19


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


class TestFlipAugmentation:
    """Test horizontal flip augmentation."""

    def test_flip_image_dimensions(self) -> None:
        img = Image.new("RGBA", (128, 128), (255, 0, 0, 255))
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[0:64, 0:64] = 6  # upper_arm_l in top-left
        do_map = np.zeros((128, 128), dtype=np.uint8)

        flipped_img, flipped_mask, flipped_do = _apply_flip(img, mask, do_map)

        assert flipped_img.size == (128, 128)
        assert flipped_mask.shape == (128, 128)
        assert flipped_do.shape == (128, 128)

    def test_flip_swaps_regions(self) -> None:
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[0:64, 0:64] = 6  # upper_arm_l
        img = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
        do_map = np.zeros((128, 128), dtype=np.uint8)

        _, flipped_mask, _ = _apply_flip(img, mask, do_map)

        # upper_arm_l (6) should become upper_arm_r (9)
        # and should be in top-right (flipped from top-left)
        assert 9 in np.unique(flipped_mask)


class TestRotationAugmentation:
    """Test rotation augmentation."""

    def test_zero_rotation(self) -> None:
        img = Image.new("RGBA", (128, 128), (255, 0, 0, 255))
        mask = np.ones((128, 128), dtype=np.uint8)
        do_map = np.ones((128, 128), dtype=np.uint8) * 127

        rot_img, rot_mask, _rot_do = _apply_rotation(img, mask, do_map, 0.0)

        assert rot_img.size == (128, 128)
        np.testing.assert_array_equal(rot_mask, mask)

    def test_rotation_preserves_dimensions(self) -> None:
        img = Image.new("RGBA", (256, 256), (255, 0, 0, 255))
        mask = np.ones((256, 256), dtype=np.uint8)
        do_map = np.zeros((256, 256), dtype=np.uint8)

        rot_img, rot_mask, rot_do = _apply_rotation(img, mask, do_map, 5.0)

        assert rot_img.size == (256, 256)
        assert rot_mask.shape == (256, 256)
        assert rot_do.shape == (256, 256)


class TestScaleAugmentation:
    """Test scale augmentation."""

    def test_identity_scale(self) -> None:
        img = Image.new("RGBA", (128, 128), (255, 0, 0, 255))
        mask = np.ones((128, 128), dtype=np.uint8) * 3
        do_map = np.ones((128, 128), dtype=np.uint8) * 100

        sc_img, sc_mask, _sc_do = _apply_scale(img, mask, do_map, 1.0, 128)

        assert sc_img.size == (128, 128)
        np.testing.assert_array_equal(sc_mask, mask)

    def test_scale_down(self) -> None:
        img = Image.new("RGBA", (128, 128), (255, 0, 0, 255))
        mask = np.ones((128, 128), dtype=np.uint8) * 3
        do_map = np.ones((128, 128), dtype=np.uint8) * 100

        sc_img, sc_mask, _sc_do = _apply_scale(img, mask, do_map, 0.5, 128)

        assert sc_img.size == (128, 128)
        # After scaling down, edges should be background (0)
        assert sc_mask[0, 0] == 0

    def test_scale_up(self) -> None:
        img = Image.new("RGBA", (128, 128), (255, 0, 0, 255))
        mask = np.ones((128, 128), dtype=np.uint8) * 3
        do_map = np.ones((128, 128), dtype=np.uint8) * 100

        sc_img, sc_mask, _sc_do = _apply_scale(img, mask, do_map, 1.5, 128)

        assert sc_img.size == (128, 128)
        # After scaling up, center should still be region 3
        assert sc_mask[64, 64] == 3


class TestColorJitter:
    """Test color jitter augmentation."""

    def test_preserves_alpha(self) -> None:
        img = Image.new("RGBA", (64, 64), (255, 128, 0, 200))
        jittered = _apply_color_jitter(img)

        # Alpha channel should be preserved
        _, _, _, a = jittered.split()
        a_arr = np.array(a)
        assert (a_arr == 200).all()

    def test_preserves_dimensions(self) -> None:
        img = Image.new("RGBA", (64, 64), (128, 128, 128, 255))
        jittered = _apply_color_jitter(img)
        assert jittered.size == (64, 64)
        assert jittered.mode == "RGBA"

    def test_no_jitter_config(self) -> None:
        img = Image.new("RGBA", (32, 32), (100, 100, 100, 255))
        no_jitter = {
            "hue": (0.0, 0.0),
            "saturation": (1.0, 1.0),
            "brightness": (1.0, 1.0),
        }
        result = _apply_color_jitter(img, no_jitter)
        # Should be approximately the same (no jitter applied)
        orig_arr = np.array(img)
        result_arr = np.array(result)
        np.testing.assert_array_almost_equal(orig_arr, result_arr, decimal=0)


class TestGenerateAugmentations:
    """Test the full augmentation pipeline."""

    def test_produces_four_variants(self) -> None:
        img = Image.new("RGBA", (128, 128), (255, 0, 0, 255))
        mask = np.ones((128, 128), dtype=np.uint8) * 3
        do_map = np.ones((128, 128), dtype=np.uint8) * 100

        variants = generate_augmentations(img, mask, do_map, resolution=128)

        assert len(variants) == 4

    def test_identity_is_first(self) -> None:
        img = Image.new("RGBA", (128, 128), (255, 0, 0, 255))
        mask = np.ones((128, 128), dtype=np.uint8) * 3
        do_map = np.ones((128, 128), dtype=np.uint8) * 100

        variants = generate_augmentations(img, mask, do_map, resolution=128)

        assert variants[0][0] == "identity"

    def test_flip_is_second(self) -> None:
        img = Image.new("RGBA", (128, 128), (255, 0, 0, 255))
        mask = np.ones((128, 128), dtype=np.uint8) * 3
        do_map = np.ones((128, 128), dtype=np.uint8) * 100

        variants = generate_augmentations(img, mask, do_map, resolution=128)

        assert variants[1][0] == "flip"

    def test_all_variants_have_correct_shapes(self) -> None:
        resolution = 128
        img = Image.new("RGBA", (resolution, resolution), (255, 0, 0, 255))
        mask = np.ones((resolution, resolution), dtype=np.uint8) * 3
        do_map = np.ones((resolution, resolution), dtype=np.uint8) * 100

        variants = generate_augmentations(img, mask, do_map, resolution=resolution)

        for label, v_img, v_mask, v_do in variants:
            assert v_img.size == (resolution, resolution), f"Image size wrong for {label}"
            assert v_mask.shape == (resolution, resolution), f"Mask shape wrong for {label}"
            assert v_do.shape == (resolution, resolution), f"Draw order shape wrong for {label}"


# ---------------------------------------------------------------------------
# End-to-end: process_live2d_model
# ---------------------------------------------------------------------------


class TestProcessLive2DModel:
    """Test the full model processing pipeline."""

    def test_process_model_with_fragments(self, tmp_model_dir: Path) -> None:
        result = process_live2d_model(tmp_model_dir, resolution=256)

        assert result is not None
        assert result.char_id == "live2d_test_model"
        assert result.image.size == (256, 256)
        assert result.mask.shape == (256, 256)
        assert result.draw_order_map.shape == (256, 256)
        assert result.fragment_count > 0

    def test_mapped_fragments(self, tmp_model_dir: Path) -> None:
        result = process_live2d_model(tmp_model_dir, resolution=256)

        assert result is not None
        # "head" and "body" and "arm_upper_L" should map
        assert result.mapped_count >= 3
        # "shadow_overlay" should be unmapped
        assert "shadow_overlay" in result.unmapped_fragments

    def test_mask_has_region_ids(self, tmp_model_dir: Path) -> None:
        result = process_live2d_model(tmp_model_dir, resolution=256)

        assert result is not None
        unique_ids = set(np.unique(result.mask))
        assert 0 in unique_ids  # background
        # At least one body region should be present
        assert len(unique_ids) > 1

    def test_joint_data_schema(self, tmp_model_dir: Path) -> None:
        result = process_live2d_model(tmp_model_dir, resolution=256)

        assert result is not None
        assert "joints" in result.joint_data
        assert "bbox" in result.joint_data
        assert "image_size" in result.joint_data
        assert len(result.joint_data["joints"]) == 19

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        result = process_live2d_model(tmp_path / "nonexistent")
        assert result is None

    def test_empty_directory(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty_model"
        empty_dir.mkdir()
        result = process_live2d_model(empty_dir, resolution=256)
        assert result is None


# ---------------------------------------------------------------------------
# End-to-end: process_live2d_directory
# ---------------------------------------------------------------------------


class TestProcessLive2DDirectory:
    """Test batch processing of Live2D models."""

    def test_process_directory(self, tmp_model_dir: Path, tmp_path: Path) -> None:
        # tmp_model_dir is a model subdirectory — wrap it in a parent
        parent_dir = tmp_model_dir.parent
        output_dir = tmp_path / "output"

        results = process_live2d_directory(
            parent_dir,
            output_dir,
            resolution=256,
            styles=["flat"],
            enable_augmentation=False,
            only_new=False,
        )

        assert len(results) == 1
        assert results[0].char_id == "live2d_test_model"

        # Check output files exist
        assert (output_dir / "images").is_dir()
        assert (output_dir / "masks").is_dir()
        assert (output_dir / "draw_order").is_dir()
        assert (output_dir / "joints").is_dir()
        assert (output_dir / "sources").is_dir()

    def test_empty_directory(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "no_models"
        empty_dir.mkdir()
        output_dir = tmp_path / "output"

        results = process_live2d_directory(empty_dir, output_dir, resolution=256, only_new=False)
        assert results == []

    def test_augmentation_creates_multiple_poses(self, tmp_model_dir: Path, tmp_path: Path) -> None:
        parent_dir = tmp_model_dir.parent
        output_dir = tmp_path / "output"

        results = process_live2d_directory(
            parent_dir,
            output_dir,
            resolution=256,
            styles=["flat"],
            enable_augmentation=True,
            only_new=False,
        )

        assert len(results) == 1
        # With augmentation, should have 4 pose variants (4 mask files)
        mask_files = sorted((output_dir / "masks").glob("*.png"))
        assert len(mask_files) == 4
