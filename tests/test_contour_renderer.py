"""Tests for the contour renderer module.

Only tests the pure-Python ``compute_contour_mask`` function.  The Blender
functions (``enable_freestyle``, ``disable_freestyle``, ``render_contour_pair``)
require ``bpy`` and are exercised via integration tests.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

# Provide a mock bpy module so pipeline.contour_renderer can be imported outside Blender.
_bpy_mock = types.ModuleType("bpy")
_bpy_mock.types = MagicMock()
_bpy_mock.ops = MagicMock()
_bpy_mock.data = MagicMock()
sys.modules.setdefault("bpy", _bpy_mock)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from pipeline.contour_renderer import compute_contour_mask  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid_image(size: int, color: tuple[int, int, int], *, mode: str = "RGB") -> Image.Image:
    """Create a solid-color image."""
    arr = np.full((size, size, 3), color, dtype=np.uint8)
    if mode == "RGBA":
        alpha = np.full((size, size, 1), 255, dtype=np.uint8)
        arr = np.concatenate([arr, alpha], axis=2)
    return Image.fromarray(arr, mode=mode)


def _image_with_lines(
    size: int = 64,
    bg: tuple[int, int, int] = (200, 200, 200),
    line: tuple[int, int, int] = (20, 20, 20),
    spacing: int = 16,
) -> Image.Image:
    """Create an image with horizontal dark lines."""
    arr = np.full((size, size, 3), bg, dtype=np.uint8)
    for y in range(0, size, spacing):
        arr[y : y + 2, :, :] = line
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# compute_contour_mask
# ---------------------------------------------------------------------------


class TestComputeContourMask:
    def test_identical_images(self) -> None:
        img = _solid_image(32, (128, 128, 128))
        mask = compute_contour_mask(img, img, threshold=30)
        assert mask.shape == (32, 32)
        assert mask.dtype == np.uint8
        assert np.all(mask == 0)

    def test_detects_lines(self) -> None:
        clean = _solid_image(64, (200, 200, 200))
        with_lines = _image_with_lines(64)
        mask = compute_contour_mask(with_lines, clean, threshold=30)
        assert np.any(mask == 255)

    def test_binary_output(self) -> None:
        clean = _solid_image(32, (200, 200, 200))
        with_lines = _image_with_lines(32)
        mask = compute_contour_mask(with_lines, clean, threshold=30)
        unique = set(np.unique(mask))
        assert unique.issubset({0, 255})

    def test_high_threshold_no_contours(self) -> None:
        a = _solid_image(16, (100, 100, 100))
        b = _solid_image(16, (110, 110, 110))
        mask = compute_contour_mask(a, b, threshold=50)
        assert np.all(mask == 0)

    def test_low_threshold_all_contours(self) -> None:
        a = _solid_image(16, (100, 100, 100))
        b = _solid_image(16, (110, 110, 110))
        mask = compute_contour_mask(a, b, threshold=5)
        assert np.all(mask == 255)

    def test_rgba_input(self) -> None:
        clean = _solid_image(32, (200, 200, 200), mode="RGBA")
        with_lines = _image_with_lines(32)
        mask = compute_contour_mask(with_lines, clean, threshold=30)
        assert mask.shape == (32, 32)
        assert np.any(mask == 255)

    def test_shape_matches_input(self) -> None:
        a = _solid_image(128, (100, 100, 100))
        b = _solid_image(128, (200, 200, 200))
        mask = compute_contour_mask(a, b, threshold=10)
        assert mask.shape == (128, 128)

    def test_asymmetric_diff(self) -> None:
        """Contour mask should be the same regardless of argument order."""
        a = _solid_image(16, (100, 100, 100))
        b = _solid_image(16, (200, 200, 200))
        mask_ab = compute_contour_mask(a, b, threshold=30)
        mask_ba = compute_contour_mask(b, a, threshold=30)
        np.testing.assert_array_equal(mask_ab, mask_ba)
