"""Tests for the texture projection training data generator.

Exercises the pure-Python geometry and image-processing logic without
requiring Blender. Blender-dependent functions (camera setup, rendering,
visibility) are not tested here.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import helpers — mock bpy so we can import the module outside Blender
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_blender(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub out Blender modules so the trainer can be imported outside Blender."""
    import sys
    from unittest.mock import MagicMock

    mock_bpy = MagicMock()
    mock_bpy_extras = MagicMock()
    mock_mathutils = MagicMock()

    monkeypatch.setitem(sys.modules, "bpy", mock_bpy)
    monkeypatch.setitem(sys.modules, "bpy.ops", mock_bpy.ops)
    monkeypatch.setitem(sys.modules, "bpy_extras", mock_bpy_extras)
    monkeypatch.setitem(sys.modules, "bpy_extras.object_utils", mock_bpy_extras.object_utils)
    monkeypatch.setitem(sys.modules, "mathutils", mock_mathutils)


def _import_trainer():
    """Import the trainer module after mocking Blender."""
    import importlib
    import sys

    mod_name = "mesh.scripts.texture_projection_trainer"
    if mod_name in sys.modules:
        return importlib.reload(sys.modules[mod_name])
    return importlib.import_module(mod_name)


# ---------------------------------------------------------------------------
# Barycentric coordinate tests
# ---------------------------------------------------------------------------


class TestBarycentricCoords:
    """Tests for _barycentric_coords."""

    def test_vertex_a(self) -> None:
        mod = _import_trainer()
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        c = np.array([0.0, 1.0])

        w0, w1, w2 = mod._barycentric_coords(a, a, b, c)
        assert abs(w0 - 1.0) < 1e-6
        assert abs(w1) < 1e-6
        assert abs(w2) < 1e-6

    def test_vertex_b(self) -> None:
        mod = _import_trainer()
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        c = np.array([0.0, 1.0])

        w0, w1, w2 = mod._barycentric_coords(b, a, b, c)
        assert abs(w0) < 1e-6
        assert abs(w1 - 1.0) < 1e-6
        assert abs(w2) < 1e-6

    def test_vertex_c(self) -> None:
        mod = _import_trainer()
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        c = np.array([0.0, 1.0])

        w0, w1, w2 = mod._barycentric_coords(c, a, b, c)
        assert abs(w0) < 1e-6
        assert abs(w1) < 1e-6
        assert abs(w2 - 1.0) < 1e-6

    def test_centroid(self) -> None:
        mod = _import_trainer()
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        c = np.array([0.0, 1.0])
        centroid = (a + b + c) / 3.0

        w0, w1, w2 = mod._barycentric_coords(centroid, a, b, c)
        assert abs(w0 - 1 / 3) < 1e-6
        assert abs(w1 - 1 / 3) < 1e-6
        assert abs(w2 - 1 / 3) < 1e-6

    def test_outside_triangle(self) -> None:
        mod = _import_trainer()
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        c = np.array([0.0, 1.0])
        outside = np.array([1.0, 1.0])

        w0, _w1, _w2 = mod._barycentric_coords(outside, a, b, c)
        assert w0 < 0  # outside — at least one weight is negative

    def test_degenerate_triangle(self) -> None:
        mod = _import_trainer()
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        c = np.array([2.0, 0.0])  # collinear

        w0, _w1, _w2 = mod._barycentric_coords(np.array([0.5, 0.0]), a, b, c)
        assert w0 == -1.0  # sentinel for degenerate

    def test_weights_sum_to_one(self) -> None:
        mod = _import_trainer()
        a = np.array([0.2, 0.1])
        b = np.array([0.8, 0.3])
        c = np.array([0.5, 0.9])
        p = np.array([0.5, 0.4])

        w0, w1, w2 = mod._barycentric_coords(p, a, b, c)
        assert abs(w0 + w1 + w2 - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Inpainting mask tests
# ---------------------------------------------------------------------------


class TestInpaintingMask:
    """Tests for compute_inpainting_mask."""

    def test_no_gap(self) -> None:
        """When partial has full coverage, mask is all zeros."""
        mod = _import_trainer()
        partial = np.full((64, 64), 255, dtype=np.uint8)
        complete = np.full((64, 64), 255, dtype=np.uint8)

        mask = mod.compute_inpainting_mask(partial, complete)
        assert mask.shape == (64, 64)
        assert mask.max() == 0

    def test_full_gap(self) -> None:
        """When partial is empty but complete is full, mask is all 255."""
        mod = _import_trainer()
        partial = np.zeros((64, 64), dtype=np.uint8)
        complete = np.full((64, 64), 255, dtype=np.uint8)

        mask = mod.compute_inpainting_mask(partial, complete)
        assert mask.max() == 255
        assert np.all(mask == 255)

    def test_partial_gap(self) -> None:
        """Mask should be 255 only where complete has data and partial doesn't."""
        mod = _import_trainer()
        partial = np.zeros((64, 64), dtype=np.uint8)
        partial[:32, :] = 255  # top half filled

        complete = np.full((64, 64), 255, dtype=np.uint8)

        mask = mod.compute_inpainting_mask(partial, complete)

        # Top half: both filled → mask = 0
        assert np.all(mask[:32, :] == 0)
        # Bottom half: complete filled, partial empty → mask = 255
        assert np.all(mask[32:, :] == 255)

    def test_both_empty(self) -> None:
        """If neither has coverage, mask is all zeros (nothing to inpaint)."""
        mod = _import_trainer()
        partial = np.zeros((32, 32), dtype=np.uint8)
        complete = np.zeros((32, 32), dtype=np.uint8)

        mask = mod.compute_inpainting_mask(partial, complete)
        assert np.all(mask == 0)


# ---------------------------------------------------------------------------
# Texture margin / bleed tests
# ---------------------------------------------------------------------------


class TestTextureMargin:
    """Tests for apply_texture_margin."""

    def test_single_pixel_dilates(self) -> None:
        """A single filled texel should bleed outward by 1 pixel per margin step."""
        mod = _import_trainer()
        texture = np.zeros((16, 16, 4), dtype=np.uint8)
        coverage = np.zeros((16, 16), dtype=np.uint8)

        # Fill center pixel
        texture[8, 8] = [255, 0, 0, 255]
        coverage[8, 8] = 255

        mod.apply_texture_margin(texture, coverage, margin=1)

        # The 4-connected neighbors should now be filled
        assert coverage[7, 8] == 255  # up
        assert coverage[9, 8] == 255  # down
        assert coverage[8, 7] == 255  # left
        assert coverage[8, 9] == 255  # right

    def test_margin_zero_no_change(self) -> None:
        """Margin of 0 should not change anything."""
        mod = _import_trainer()
        texture = np.zeros((8, 8, 4), dtype=np.uint8)
        coverage = np.zeros((8, 8), dtype=np.uint8)

        texture[4, 4] = [0, 255, 0, 255]
        coverage[4, 4] = 255

        orig_coverage = coverage.copy()
        mod.apply_texture_margin(texture, coverage, margin=0)

        assert np.array_equal(coverage, orig_coverage)

    def test_margin_two_steps(self) -> None:
        """Margin of 2 should reach 2 pixels outward."""
        mod = _import_trainer()
        texture = np.zeros((16, 16, 4), dtype=np.uint8)
        coverage = np.zeros((16, 16), dtype=np.uint8)

        texture[8, 8] = [100, 100, 100, 255]
        coverage[8, 8] = 255

        mod.apply_texture_margin(texture, coverage, margin=2)

        # 2 steps away (Manhattan distance ≤ 2)
        assert coverage[6, 8] == 255  # 2 up
        assert coverage[10, 8] == 255  # 2 down
        assert coverage[8, 6] == 255  # 2 left
        assert coverage[8, 10] == 255  # 2 right

    def test_edge_handling(self) -> None:
        """Margin at image edges should not crash."""
        mod = _import_trainer()
        texture = np.zeros((4, 4, 4), dtype=np.uint8)
        coverage = np.zeros((4, 4), dtype=np.uint8)

        # Fill corner pixel
        texture[0, 0] = [200, 100, 50, 255]
        coverage[0, 0] = 255

        mod.apply_texture_margin(texture, coverage, margin=2)

        # Should not crash, and neighbors should be filled where possible
        assert coverage[0, 1] == 255
        assert coverage[1, 0] == 255


# ---------------------------------------------------------------------------
# Config constants tests
# ---------------------------------------------------------------------------


class TestConfigConstants:
    """Verify texture projection constants in config.py."""

    def test_dense_angles_count(self) -> None:
        from pipeline.config import TEXTURE_DENSE_ANGLES

        assert len(TEXTURE_DENSE_ANGLES) == 24

    def test_dense_angles_range(self) -> None:
        from pipeline.config import TEXTURE_DENSE_ANGLES

        assert TEXTURE_DENSE_ANGLES[0] == 0
        assert TEXTURE_DENSE_ANGLES[-1] == 345
        assert all(0 <= a < 360 for a in TEXTURE_DENSE_ANGLES)

    def test_dense_angles_spacing(self) -> None:
        from pipeline.config import TEXTURE_DENSE_ANGLES

        for i in range(1, len(TEXTURE_DENSE_ANGLES)):
            assert TEXTURE_DENSE_ANGLES[i] - TEXTURE_DENSE_ANGLES[i - 1] == 15

    def test_partial_angles(self) -> None:
        from pipeline.config import TEXTURE_PARTIAL_ANGLES

        assert TEXTURE_PARTIAL_ANGLES == [0, 45, 180]

    def test_partial_is_subset_of_dense(self) -> None:
        from pipeline.config import TEXTURE_DENSE_ANGLES, TEXTURE_PARTIAL_ANGLES

        dense_set = set(TEXTURE_DENSE_ANGLES)
        for angle in TEXTURE_PARTIAL_ANGLES:
            assert angle in dense_set

    def test_texture_resolution(self) -> None:
        from pipeline.config import TEXTURE_RESOLUTION

        assert TEXTURE_RESOLUTION == 1024

    def test_bake_margin(self) -> None:
        from pipeline.config import TEXTURE_BAKE_MARGIN

        assert TEXTURE_BAKE_MARGIN > 0


# ---------------------------------------------------------------------------
# Rasterize triangle tests
# ---------------------------------------------------------------------------


class TestRasterizeTriangle:
    """Tests for _rasterize_triangle_to_uv."""

    def test_identity_projection(self) -> None:
        """When screen coords == UV coords, texture should mirror rendered image."""
        mod = _import_trainer()

        # Create a simple 8x8 rendered image (red square)
        rendered = np.zeros((8, 8, 4), dtype=np.uint8)
        rendered[2:6, 2:6] = [255, 0, 0, 255]

        texture = np.zeros((8, 8, 4), dtype=np.uint8)
        coverage = np.zeros((8, 8), dtype=np.uint8)

        # Triangle covering the red area in both screen and UV space
        # Screen coords are normalized 0..1, UV coords are 0..1
        sc_a = np.array([0.25, 0.25])  # (2/8, 2/8)
        sc_b = np.array([0.75, 0.25])  # (6/8, 2/8)
        sc_c = np.array([0.25, 0.75])  # (2/8, 6/8)

        uv_a = np.array([0.25, 0.25])
        uv_b = np.array([0.75, 0.25])
        uv_c = np.array([0.25, 0.75])

        mod._rasterize_triangle_to_uv(
            sc_a,
            sc_b,
            sc_c,
            uv_a,
            uv_b,
            uv_c,
            rendered,
            8,
            8,
            texture,
            coverage,
            8,
            8,
        )

        # Some texels should be filled with the red color
        filled_count = np.count_nonzero(coverage)
        assert filled_count > 0

    def test_empty_triangle(self) -> None:
        """Degenerate triangle should produce no filled texels."""
        mod = _import_trainer()

        rendered = np.full((8, 8, 4), 128, dtype=np.uint8)
        texture = np.zeros((8, 8, 4), dtype=np.uint8)
        coverage = np.zeros((8, 8), dtype=np.uint8)

        # Collinear points → degenerate triangle
        sc_a = np.array([0.0, 0.0])
        sc_b = np.array([1.0, 0.0])
        sc_c = np.array([0.5, 0.0])

        uv_a = np.array([0.0, 0.0])
        uv_b = np.array([1.0, 0.0])
        uv_c = np.array([0.5, 0.0])

        mod._rasterize_triangle_to_uv(
            sc_a,
            sc_b,
            sc_c,
            uv_a,
            uv_b,
            uv_c,
            rendered,
            8,
            8,
            texture,
            coverage,
            8,
            8,
        )

        assert np.count_nonzero(coverage) == 0
