"""Tests for per-region RGBA layer extraction.

Tests the pure-Python helpers in ``layer_extractor`` (active region
detection) and the Live2D ``extract_region_layers`` function.  The
Blender rendering path (``extract_layers``) requires ``bpy`` and is
tested via integration tests.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

# Provide a mock bpy module so pipeline imports work outside Blender.
_bpy_mock = types.ModuleType("bpy")
_bpy_mock.types = MagicMock()
_bpy_mock.data = MagicMock()
_bpy_mock.ops = MagicMock()
_bpy_mock.context = MagicMock()
sys.modules.setdefault("bpy", _bpy_mock)

# Also mock bpy_extras which renderer.py imports
_bpy_extras_mock = types.ModuleType("bpy_extras")
_bpy_extras_mock.object_utils = MagicMock()
sys.modules.setdefault("bpy_extras", _bpy_extras_mock)
sys.modules.setdefault("bpy_extras.object_utils", _bpy_extras_mock.object_utils)

from pipeline.exporter import layer_filename  # noqa: E402
from pipeline.layer_extractor import get_active_regions  # noqa: E402
from pipeline.live2d_renderer import extract_region_layers  # noqa: E402

# ---------------------------------------------------------------------------
# get_active_regions
# ---------------------------------------------------------------------------


class TestGetActiveRegions:
    def test_basic(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[0:10, 0:10] = 1  # head
        mask[10:20, 10:20] = 3  # chest
        mask[30:40, 30:40] = 14  # upper_leg_l
        result = get_active_regions(mask)
        assert result == [1, 3, 14]

    def test_empty_mask(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        assert get_active_regions(mask) == []

    def test_background_only(self):
        mask = np.full((64, 64), 0, dtype=np.uint8)
        assert get_active_regions(mask) == []

    def test_single_region(self):
        mask = np.full((64, 64), 5, dtype=np.uint8)
        assert get_active_regions(mask) == [5]

    def test_excludes_background(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[0:30, :] = 0  # background
        mask[30:64, :] = 8  # forearm_l
        result = get_active_regions(mask)
        assert 0 not in result
        assert result == [8]

    def test_all_regions(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        for r in range(1, 22):
            mask[r, 0] = r
        result = get_active_regions(mask)
        assert result == list(range(1, 22))

    def test_ignores_out_of_range(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[0, 0] = 1
        mask[1, 1] = 25  # out of range, should be ignored
        result = get_active_regions(mask)
        assert result == [1]


# ---------------------------------------------------------------------------
# layer_filename (exporter)
# ---------------------------------------------------------------------------


class TestLayerFilename:
    def test_front_angle(self):
        assert layer_filename("mixamo_001", 5, 3) == "mixamo_001_pose_05_03.png"

    def test_other_angle(self):
        name = layer_filename("mixamo_001", 5, 3, angle="three_quarter")
        assert name == "mixamo_001_pose_05_three_quarter_03.png"

    def test_region_padding(self):
        assert layer_filename("char", 0, 1) == "char_pose_00_01.png"
        assert layer_filename("char", 0, 19) == "char_pose_00_19.png"


# ---------------------------------------------------------------------------
# extract_region_layers (Live2D)
# ---------------------------------------------------------------------------


def _make_fragment(
    name: str,
    width: int,
    height: int,
    color: tuple,
    draw_order: int,
) -> tuple[str, Image.Image, int]:
    """Create a test fragment with a solid color and full alpha."""
    img = Image.new("RGBA", (width, height), color)
    return (name, img, draw_order)


class TestExtractRegionLayers:
    def test_groups_by_region(self):
        fragments = [
            _make_fragment("head_frag", 100, 100, (255, 0, 0, 255), 10),
            _make_fragment("chest_frag", 100, 100, (0, 255, 0, 255), 5),
            _make_fragment("arm_frag", 100, 100, (0, 0, 255, 255), 8),
        ]
        frag_to_region = {
            "head_frag": 1,
            "chest_frag": 3,
            "arm_frag": 7,
        }
        layers = extract_region_layers(fragments, frag_to_region, resolution=128)
        assert set(layers.keys()) == {1, 3, 7}
        for _rid, img in layers.items():
            assert img.size == (128, 128)
            assert img.mode == "RGBA"

    def test_multiple_fragments_same_region(self):
        fragments = [
            _make_fragment("hair", 100, 100, (255, 0, 0, 255), 10),
            _make_fragment("face", 100, 100, (200, 0, 0, 255), 5),
        ]
        frag_to_region = {"hair": 1, "face": 1}
        layers = extract_region_layers(fragments, frag_to_region, resolution=128)
        assert set(layers.keys()) == {1}

    def test_excludes_background_region(self):
        fragments = [
            _make_fragment("bg", 100, 100, (0, 0, 0, 255), 0),
            _make_fragment("head", 100, 100, (255, 0, 0, 255), 5),
        ]
        frag_to_region = {"bg": 0, "head": 1}
        layers = extract_region_layers(fragments, frag_to_region, resolution=128)
        assert 0 not in layers
        assert 1 in layers

    def test_empty_fragments(self):
        layers = extract_region_layers([], {}, resolution=128)
        assert layers == {}

    def test_layers_have_transparent_background(self):
        fragments = [
            _make_fragment("head", 50, 50, (255, 0, 0, 255), 5),
        ]
        frag_to_region = {"head": 1}
        layers = extract_region_layers(fragments, frag_to_region, resolution=128)
        layer = layers[1]
        arr = np.array(layer)
        # Corners should be transparent (fragment is centered, smaller than canvas)
        assert arr[0, 0, 3] == 0
        # Some pixels should be opaque (the fragment content)
        assert np.any(arr[:, :, 3] > 0)
