"""Tests for the UniRig adapter.

These tests exercise pure-Python adapter logic without requiring Blender
or the actual UniRig dataset.  Blender modules (bpy, bmesh, mathutils) are
mocked in sys.modules before importing the adapter so the import succeeds
outside a Blender environment.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock Blender modules before importing the adapter
# ---------------------------------------------------------------------------

sys.modules.setdefault("bpy", MagicMock())
sys.modules.setdefault("bmesh", MagicMock())
sys.modules.setdefault("mathutils", MagicMock())

from ingest.unirig_adapter import (  # noqa: E402  (after sys.modules patch)
    MIN_HUMANOID_COVERAGE,
    ConversionStats,
    _is_humanoid,
    load_npz,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_skin(n_verts: int, n_joints: int, dominant_bones: list[int]) -> np.ndarray:
    """Build an (n_verts, n_joints) weight matrix where each vertex is weighted
    entirely to the bone index given by dominant_bones[i].
    """
    skin = np.zeros((n_verts, n_joints), dtype=np.float32)
    for vi, bone in enumerate(dominant_bones):
        skin[vi, bone] = 1.0
    return skin


def _full_humanoid_bone_to_region() -> dict[int, int]:
    """Return a bone_to_region mapping with bilateral arms and legs.

    Bone indices:
        0 → hips (5)
        1 → upper_arm_l (7)
        2 → upper_arm_r (11)
        3 → upper_leg_l (14)
        4 → upper_leg_r (17)
    """
    return {0: 5, 1: 7, 2: 11, 3: 14, 4: 17}


# ---------------------------------------------------------------------------
# TestLoadNpz
# ---------------------------------------------------------------------------


class TestLoadNpz:
    def test_valid_npz_returns_dict(self, tmp_path: Path) -> None:
        npz_path = tmp_path / "raw_data.npz"
        np.savez(
            npz_path,
            vertices=np.zeros((10, 3), dtype=np.float32),
            faces=np.zeros((4, 3), dtype=np.int32),
            names=np.array(["Hips", "Spine"], dtype=object),
            skin=np.ones((10, 2), dtype=np.float32),
        )
        result = load_npz(npz_path)
        assert result is not None
        assert "vertices" in result
        assert "faces" in result
        assert "names" in result
        assert "skin" in result

    def test_missing_skin_returns_none(self, tmp_path: Path) -> None:
        npz_path = tmp_path / "raw_data.npz"
        np.savez(
            npz_path,
            vertices=np.zeros((10, 3)),
            faces=np.zeros((4, 3), dtype=np.int32),
            names=np.array(["Hips"], dtype=object),
        )
        assert load_npz(npz_path) is None

    def test_missing_names_returns_none(self, tmp_path: Path) -> None:
        npz_path = tmp_path / "raw_data.npz"
        np.savez(
            npz_path,
            vertices=np.zeros((10, 3)),
            faces=np.zeros((4, 3), dtype=np.int32),
            skin=np.ones((10, 1)),
        )
        assert load_npz(npz_path) is None

    def test_missing_vertices_returns_none(self, tmp_path: Path) -> None:
        npz_path = tmp_path / "raw_data.npz"
        np.savez(
            npz_path,
            faces=np.zeros((4, 3), dtype=np.int32),
            names=np.array(["Hips"], dtype=object),
            skin=np.ones((10, 1)),
        )
        assert load_npz(npz_path) is None

    def test_nonexistent_path_returns_none(self, tmp_path: Path) -> None:
        result = load_npz(tmp_path / "does_not_exist.npz")
        assert result is None


# ---------------------------------------------------------------------------
# TestIsHumanoid
# ---------------------------------------------------------------------------


class TestIsHumanoid:
    def _fields(self, skin: np.ndarray) -> dict:
        return {"skin": skin}

    def test_full_humanoid_returns_true(self) -> None:
        bone_to_region = _full_humanoid_bone_to_region()
        n_verts = 100
        # All 100 vertices distributed across 5 bones → 100% coverage
        dominant = [i % 5 for i in range(n_verts)]
        skin = _make_skin(n_verts, 5, dominant)
        assert _is_humanoid(self._fields(skin), bone_to_region) is True

    def test_low_coverage_returns_false(self) -> None:
        bone_to_region = _full_humanoid_bone_to_region()
        n_verts = 100
        # Only 10 vertices mapped to known bones (10% coverage < 60%)
        dominant = [0] * 10 + [99] * 90  # bone 99 is unmapped
        skin = _make_skin(n_verts, 100, dominant)
        assert _is_humanoid(self._fields(skin), bone_to_region) is False

    def test_missing_left_upper_arm_returns_false(self) -> None:
        # Remove region 7 (upper_arm_l) — no left arm
        bone_to_region = {0: 5, 2: 11, 3: 14, 4: 17}
        n_verts = 100
        dominant = [i % 4 for i in range(n_verts)]
        skin = _make_skin(n_verts, 5, dominant)
        assert _is_humanoid(self._fields(skin), bone_to_region) is False

    def test_missing_right_upper_leg_returns_false(self) -> None:
        # Remove region 17 (upper_leg_r) — no right leg
        bone_to_region = {0: 5, 1: 7, 2: 11, 3: 14}
        n_verts = 100
        dominant = [i % 4 for i in range(n_verts)]
        skin = _make_skin(n_verts, 4, dominant)
        assert _is_humanoid(self._fields(skin), bone_to_region) is False

    def test_zero_vertices_returns_false(self) -> None:
        bone_to_region = _full_humanoid_bone_to_region()
        skin = np.zeros((0, 5), dtype=np.float32)
        assert _is_humanoid({"skin": skin}, bone_to_region) is False

    def test_exactly_threshold_coverage_returns_true(self) -> None:
        # n_verts chosen so that exactly MIN_HUMANOID_COVERAGE fraction
        # are assigned to mapped bones.
        n_verts = 100
        n_mapped = int(round(MIN_HUMANOID_COVERAGE * n_verts))
        n_unmapped = n_verts - n_mapped
        bone_to_region = _full_humanoid_bone_to_region()
        # bone 99 is not in bone_to_region
        dominant = [i % 5 for i in range(n_mapped)] + [99] * n_unmapped
        skin = _make_skin(n_verts, 100, dominant)
        assert _is_humanoid({"skin": skin}, bone_to_region) is True


# ---------------------------------------------------------------------------
# TestConversionStats
# ---------------------------------------------------------------------------


class TestConversionStats:
    def test_default_values_are_zero(self) -> None:
        stats = ConversionStats()
        assert stats.total == 0
        assert stats.converted == 0
        assert stats.skipped == 0
        assert stats.errors == 0

    def test_summary_returns_string(self) -> None:
        stats = ConversionStats(total=10, converted=8, skipped=1, errors=1)
        result = stats.summary()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_summary_reflects_counts(self) -> None:
        stats = ConversionStats(total=50, converted=40, skipped=5, errors=5)
        summary = stats.summary()
        assert "40" in summary
        assert "5" in summary

    def test_fields_are_mutable(self) -> None:
        stats = ConversionStats()
        stats.total += 1
        stats.converted += 1
        assert stats.total == 1
        assert stats.converted == 1
