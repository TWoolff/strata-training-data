"""Tests for the contour augmenter module.

Pure Python tests — requires OpenCV (``cv2``) but no Blender dependency.
Skipped automatically if ``cv2`` is not installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

cv2 = pytest.importorskip("cv2", reason="opencv-python not installed")

from pipeline.config import CONTOUR_STYLES  # noqa: E402
from pipeline.contour_augmenter import (  # noqa: E402
    _apply_wobble,
    _composite_contours,
    _composite_per_region,
    _dilate_mask,
    augment_all_styles,
    augment_contour_style,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_base_image(size: int = 64) -> Image.Image:
    """Create a solid-color RGBA test image."""
    arr = np.full((size, size, 4), (200, 200, 200, 255), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def _make_contour_mask(size: int = 64) -> np.ndarray:
    """Create a test contour mask with horizontal lines every 16 rows."""
    mask = np.zeros((size, size), dtype=np.uint8)
    for y in range(0, size, 16):
        mask[y, :] = 255
    return mask


def _make_seg_mask(size: int = 64) -> np.ndarray:
    """Create a test segmentation mask with 4 quadrants (regions 1–4)."""
    mask = np.zeros((size, size), dtype=np.uint8)
    half = size // 2
    mask[:half, :half] = 1
    mask[:half, half:] = 2
    mask[half:, :half] = 3
    mask[half:, half:] = 4
    return mask


# ---------------------------------------------------------------------------
# _dilate_mask
# ---------------------------------------------------------------------------


class TestDilateMask:
    def test_width_1_no_change(self) -> None:
        mask = _make_contour_mask()
        result = _dilate_mask(mask, 1)
        np.testing.assert_array_equal(result, mask)

    def test_width_3_grows_lines(self) -> None:
        mask = _make_contour_mask()
        result = _dilate_mask(mask, 3)
        # Dilated mask should have more nonzero pixels.
        assert np.count_nonzero(result) > np.count_nonzero(mask)
        # Should still be binary.
        unique = set(np.unique(result))
        assert unique.issubset({0, 255})


# ---------------------------------------------------------------------------
# _apply_wobble
# ---------------------------------------------------------------------------


class TestApplyWobble:
    def test_deterministic(self) -> None:
        mask = _make_contour_mask()
        a = _apply_wobble(mask, seed=42)
        b = _apply_wobble(mask, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds(self) -> None:
        mask = _make_contour_mask()
        a = _apply_wobble(mask, seed=1)
        b = _apply_wobble(mask, seed=999)
        # Very unlikely to be identical with different seeds.
        assert not np.array_equal(a, b)

    def test_preserves_shape(self) -> None:
        mask = _make_contour_mask(32)
        result = _apply_wobble(mask, seed=0)
        assert result.shape == (32, 32)


# ---------------------------------------------------------------------------
# _composite_contours
# ---------------------------------------------------------------------------


class TestCompositeContours:
    def test_full_opacity_black(self) -> None:
        base = _make_base_image(32)
        mask = _make_contour_mask(32)
        result = _composite_contours(base, mask, (0, 0, 0), 1.0)
        arr = np.array(result)
        # Contour pixels should be black.
        contour_rows = mask > 0
        assert np.all(arr[:, :, :3][contour_rows] == 0)

    def test_zero_opacity_unchanged(self) -> None:
        base = _make_base_image(32)
        mask = _make_contour_mask(32)
        result = _composite_contours(base, mask, (0, 0, 0), 0.0)
        arr = np.array(result)
        base_arr = np.array(base)
        np.testing.assert_array_equal(arr, base_arr)

    def test_partial_opacity_blends(self) -> None:
        base = _make_base_image(32)
        mask = _make_contour_mask(32)
        result = _composite_contours(base, mask, (0, 0, 0), 0.5)
        arr = np.array(result)
        # Contour pixels should be darker than original (200) but not black.
        contour_rows = mask > 0
        contour_values = arr[:, :, 0][contour_rows]
        assert np.all(contour_values < 200)
        assert np.all(contour_values > 0)

    def test_output_is_rgba(self) -> None:
        base = _make_base_image(16)
        mask = _make_contour_mask(16)
        result = _composite_contours(base, mask, (255, 0, 0), 1.0)
        assert result.mode == "RGBA"


# ---------------------------------------------------------------------------
# _composite_per_region
# ---------------------------------------------------------------------------


class TestCompositePerRegion:
    def test_different_regions_get_different_colors(self) -> None:
        base = _make_base_image(64)
        mask = _make_contour_mask(64)
        seg = _make_seg_mask(64)
        result = _composite_per_region(base, mask, seg, 1.0)
        arr = np.array(result)
        # Sample contour pixels in two different regions.
        # Row 0 is in regions 1 and 2 (top half).
        pixel_r1 = arr[0, 0, :3]  # top-left quadrant, region 1
        pixel_r2 = arr[0, 48, :3]  # top-right quadrant, region 2
        # Different regions should produce different colors.
        assert not np.array_equal(pixel_r1, pixel_r2)

    def test_output_is_rgba(self) -> None:
        base = _make_base_image(32)
        mask = _make_contour_mask(32)
        seg = _make_seg_mask(32)
        result = _composite_per_region(base, mask, seg, 0.8)
        assert result.mode == "RGBA"


# ---------------------------------------------------------------------------
# augment_contour_style
# ---------------------------------------------------------------------------


class TestAugmentContourStyle:
    def test_thin_black(self) -> None:
        base = _make_base_image()
        mask = _make_contour_mask()
        result = augment_contour_style(base, mask, CONTOUR_STYLES[0])
        assert result.size == base.size
        assert result.mode == "RGBA"

    def test_per_region_without_seg_mask_falls_back(self) -> None:
        base = _make_base_image()
        mask = _make_contour_mask()
        # per_region style without seg_mask should still produce output.
        per_region_style = CONTOUR_STYLES[3]
        assert per_region_style["color"] == "per_region"
        result = augment_contour_style(base, mask, per_region_style, seg_mask=None)
        assert result.size == base.size

    def test_per_region_with_seg_mask(self) -> None:
        base = _make_base_image()
        mask = _make_contour_mask()
        seg = _make_seg_mask()
        per_region_style = CONTOUR_STYLES[3]
        result = augment_contour_style(base, mask, per_region_style, seg_mask=seg)
        assert result.size == base.size

    def test_wobbly_style(self) -> None:
        base = _make_base_image()
        mask = _make_contour_mask()
        wobbly_style = CONTOUR_STYLES[4]
        assert wobbly_style["wobble"] is True
        result = augment_contour_style(base, mask, wobbly_style, seed=42)
        assert result.size == base.size


# ---------------------------------------------------------------------------
# augment_all_styles
# ---------------------------------------------------------------------------


class TestAugmentAllStyles:
    def test_produces_all_variants(self, tmp_path: Path) -> None:
        base = _make_base_image()
        mask = _make_contour_mask()
        seg = _make_seg_mask()
        paths = augment_all_styles(base, mask, tmp_path, "test_char_pose_00", seg_mask=seg)
        assert len(paths) == len(CONTOUR_STYLES)
        for p in paths:
            assert p.exists()
            img = Image.open(p)
            assert img.size == base.size

    def test_files_in_contours_subdir(self, tmp_path: Path) -> None:
        base = _make_base_image()
        mask = _make_contour_mask()
        paths = augment_all_styles(base, mask, tmp_path, "char01")
        for p in paths:
            assert p.parent == tmp_path / "contours"

    def test_filenames_contain_style_names(self, tmp_path: Path) -> None:
        base = _make_base_image()
        mask = _make_contour_mask()
        paths = augment_all_styles(base, mask, tmp_path, "char01")
        names = [p.name for p in paths]
        for style in CONTOUR_STYLES:
            style_name = str(style["name"])
            assert any(style_name in n for n in names)
