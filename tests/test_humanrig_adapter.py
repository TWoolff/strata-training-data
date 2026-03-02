"""Tests for the HumanRig adapter.

These tests exercise pure-Python adapter logic without requiring Blender
or the actual HumanRig dataset (163 GB).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ingest.humanrig_adapter import (
    ANGLE_CONFIGS,
    HUMANRIG_SOURCE,
    AdapterResult,
    HumanRigEntry,
    _build_metadata,
    _build_strata_joints,
    _make_example_id,
    _project_joints,
    _resize_to_strata,
    _rotation_y,
    convert_directory,
    convert_entry,
    discover_entries,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal set of 22 Mixamo joints matching the real HumanRig dataset.
_ALL_JOINTS_3D = {
    "Hips": [0.0, 1.0, 0.0],
    "Spine": [0.0, 1.15, 0.0],
    "Spine1": [0.0, 1.3, 0.0],
    "Spine2": [0.0, 1.45, 0.0],
    "Neck": [0.0, 1.6, 0.0],
    "Head": [0.0, 1.75, 0.0],
    "LeftShoulder": [-0.1, 1.5, 0.0],
    "LeftArm": [-0.25, 1.5, 0.0],
    "LeftForeArm": [-0.5, 1.5, 0.0],
    "LeftHand": [-0.75, 1.5, 0.0],
    "RightShoulder": [0.1, 1.5, 0.0],
    "RightArm": [0.25, 1.5, 0.0],
    "RightForeArm": [0.5, 1.5, 0.0],
    "RightHand": [0.75, 1.5, 0.0],
    "LeftUpLeg": [-0.1, 0.9, 0.0],
    "LeftLeg": [-0.1, 0.5, 0.0],
    "LeftFoot": [-0.1, 0.1, 0.0],
    "LeftToeBase": [-0.1, 0.0, -0.1],
    "RightUpLeg": [0.1, 0.9, 0.0],
    "RightLeg": [0.1, 0.5, 0.0],
    "RightFoot": [0.1, 0.1, 0.0],
    "RightToeBase": [0.1, 0.0, -0.1],
}

_ALL_JOINTS_2D = {name: [500.0, float(400 + i * 5)] for i, name in enumerate(_ALL_JOINTS_3D)}


def _make_extrinsic() -> np.ndarray:
    """Front-view camera extrinsic: world → camera.

    Camera is placed at world z=3, looking towards the origin along -z.
    In OpenCV convention the extrinsic is [R | t] where t = -R * cam_pos.
    For a camera at (0, 0, 3) with identity rotation, t = (0, 0, -3) but
    the z_c = R*p + t = z_world - 3, which gives negative z for points near
    the origin.  We flip z so the camera looks along +z: negate the z row.
    """
    # Camera looks along +Z in camera space (OpenGL-style flip).
    E = np.eye(4, dtype=np.float64)
    # Flip Y and Z so camera looks towards origin from z=3.
    E[1, 1] = -1.0  # flip y (image y increases downwards)
    E[2, 2] = -1.0  # camera looks toward -z_world
    E[2, 3] = 3.0   # translate: z_c = -z_world + 3, so z_world=0 → z_c=3 > 0
    return E


def _make_intrinsic(res: int = 1024) -> np.ndarray:
    """Simple pinhole intrinsics for a square image."""
    f = res * 0.75
    cx = cy = res / 2.0
    return np.array(
        [[f, 0, cx], [0, f, cy], [0, 0, 1]],
        dtype=np.float64,
    )


def _setup_sample_dir(
    root: Path,
    sample_id: int = 0,
    *,
    include_image: bool = True,
    include_bone_2d: bool = True,
    include_bone_3d: bool = True,
    include_extrinsic: bool = True,
    include_intrinsics: bool = True,
) -> Path:
    """Create a fake HumanRig sample directory.

    Layout matches the real dataset::

        root/{sample_id}/
        ├── front.png
        ├── bone_2d.json
        ├── bone_3d.json
        ├── extrinsic.npy
        └── intrinsics.npy
    """
    sample_dir = root / str(sample_id)
    sample_dir.mkdir(parents=True, exist_ok=True)

    if include_image:
        img = Image.new("RGBA", (1024, 1024), (128, 64, 32, 255))
        img.save(sample_dir / "front.png")

    if include_bone_2d:
        (sample_dir / "bone_2d.json").write_text(
            json.dumps(_ALL_JOINTS_2D), encoding="utf-8"
        )

    if include_bone_3d:
        (sample_dir / "bone_3d.json").write_text(
            json.dumps(_ALL_JOINTS_3D), encoding="utf-8"
        )

    if include_extrinsic:
        np.save(str(sample_dir / "extrinsic.npy"), _make_extrinsic())

    if include_intrinsics:
        np.save(str(sample_dir / "intrinsics.npy"), _make_intrinsic())

    return sample_dir


def _setup_dataset(root: Path, n: int = 3) -> Path:
    """Create a fake HumanRig dataset with *n* samples."""
    dataset_dir = root / "humanrig_opensource_final"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        _setup_sample_dir(dataset_dir, sample_id=i)
    return dataset_dir


# ---------------------------------------------------------------------------
# discover_entries
# ---------------------------------------------------------------------------


class TestDiscoverEntries:
    def test_discovers_all_samples(self, tmp_path: Path) -> None:
        dataset_dir = _setup_dataset(tmp_path, n=5)
        entries = discover_entries(dataset_dir)
        assert len(entries) == 5

    def test_sorted_by_numeric_id(self, tmp_path: Path) -> None:
        dataset_dir = _setup_dataset(tmp_path, n=5)
        entries = discover_entries(dataset_dir)
        ids = [e.sample_id for e in entries]
        assert ids == sorted(ids)

    def test_skips_incomplete_samples(self, tmp_path: Path) -> None:
        dataset_dir = _setup_dataset(tmp_path, n=3)
        # Add an incomplete sample (missing bone_3d.json).
        bad = dataset_dir / "99"
        bad.mkdir()
        Image.new("RGBA", (1024, 1024), (0, 0, 0, 255)).save(bad / "front.png")
        (bad / "bone_2d.json").write_text("{}", encoding="utf-8")

        entries = discover_entries(dataset_dir)
        assert len(entries) == 3  # the 3 good ones only

    def test_ignores_non_numeric_dirs(self, tmp_path: Path) -> None:
        dataset_dir = _setup_dataset(tmp_path, n=2)
        (dataset_dir / "metadata").mkdir()  # non-numeric
        entries = discover_entries(dataset_dir)
        assert len(entries) == 2

    def test_missing_input_dir(self, tmp_path: Path) -> None:
        entries = discover_entries(tmp_path / "nope")
        assert entries == []

    def test_entry_fields(self, tmp_path: Path) -> None:
        dataset_dir = _setup_dataset(tmp_path, n=1)
        entries = discover_entries(dataset_dir)
        e = entries[0]
        assert e.sample_id == 0
        assert e.image_path.name == "front.png"
        assert e.bone_2d_path.name == "bone_2d.json"
        assert e.bone_3d_path.name == "bone_3d.json"


# ---------------------------------------------------------------------------
# _rotation_y
# ---------------------------------------------------------------------------


class TestRotationY:
    def test_identity_at_zero(self) -> None:
        R = _rotation_y(0)
        np.testing.assert_allclose(R, np.eye(4), atol=1e-10)

    def test_180_flips_x_and_z(self) -> None:
        R = _rotation_y(180)
        p = np.array([1.0, 0.0, 0.0, 1.0])
        result = R @ p
        np.testing.assert_allclose(result[:3], [-1.0, 0.0, 0.0], atol=1e-10)

    def test_90_maps_x_to_negative_z(self) -> None:
        R = _rotation_y(90)
        p = np.array([1.0, 0.0, 0.0, 1.0])
        result = R @ p
        # After 90° rotation around Y: x→z axis flip
        np.testing.assert_allclose(result[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(abs(result[2]), 1.0, atol=1e-10)

    def test_preserves_y(self) -> None:
        R = _rotation_y(45)
        p = np.array([0.0, 2.0, 0.0, 1.0])
        result = R @ p
        np.testing.assert_allclose(result[1], 2.0, atol=1e-10)


# ---------------------------------------------------------------------------
# _project_joints
# ---------------------------------------------------------------------------


class TestProjectJoints:
    def test_front_view_returns_all_joints(self) -> None:
        E = _make_extrinsic()
        K = _make_intrinsic()
        projected = _project_joints(_ALL_JOINTS_3D, E, K, 0.0, 512)
        assert len(projected) == len(_ALL_JOINTS_3D)

    def test_symmetric_character_is_symmetric_at_front(self) -> None:
        """Left and right shoulder should have symmetric x coords at 0°."""
        E = _make_extrinsic()
        K = _make_intrinsic()
        projected = _project_joints(_ALL_JOINTS_3D, E, K, 0.0, 512)
        left_x = projected["LeftArm"][0]
        right_x = projected["RightArm"][0]
        cx = 512 / 2
        # left should be to the right of centre (screen left = character right)
        assert abs(left_x - cx) == pytest.approx(abs(right_x - cx), rel=0.05)

    def test_180_flips_left_right(self) -> None:
        """At 180°, character faces away — left/right are swapped in screen space."""
        E = _make_extrinsic()
        K = _make_intrinsic()
        front = _project_joints(_ALL_JOINTS_3D, E, K, 0.0, 512)
        back = _project_joints(_ALL_JOINTS_3D, E, K, 180.0, 512)
        cx = 512 / 2
        # LeftArm at 0° is on one side; at 180° it should be on the other.
        front_side = front["LeftArm"][0] - cx
        back_side = back["LeftArm"][0] - cx
        assert front_side * back_side < 0  # opposite signs

    def test_returns_floats(self) -> None:
        E = _make_extrinsic()
        K = _make_intrinsic()
        projected = _project_joints(_ALL_JOINTS_3D, E, K, 45.0, 512)
        for xy in projected.values():
            assert isinstance(xy[0], float)
            assert isinstance(xy[1], float)


# ---------------------------------------------------------------------------
# _build_strata_joints
# ---------------------------------------------------------------------------


class TestBuildStrataJoints:
    def _project_front(self) -> dict[str, list[float]]:
        return _project_joints(
            _ALL_JOINTS_3D, _make_extrinsic(), _make_intrinsic(), 0.0, 512
        )

    def test_returns_19_joints(self) -> None:
        projected = self._project_front()
        joints = _build_strata_joints(projected, 512)
        assert len(joints) == 19

    def test_ordered_by_id(self) -> None:
        projected = self._project_front()
        joints = _build_strata_joints(projected, 512)
        ids = [j["id"] for j in joints]
        assert ids == list(range(1, 20))

    def test_each_joint_has_required_keys(self) -> None:
        projected = self._project_front()
        joints = _build_strata_joints(projected, 512)
        for j in joints:
            assert "id" in j
            assert "name" in j
            assert "x" in j
            assert "y" in j
            assert "visible" in j

    def test_visible_joints_in_bounds(self) -> None:
        projected = self._project_front()
        joints = _build_strata_joints(projected, 512)
        for j in joints:
            if j["visible"]:
                assert 0 <= j["x"] < 512
                assert 0 <= j["y"] < 512

    def test_head_region_present(self) -> None:
        projected = self._project_front()
        joints = _build_strata_joints(projected, 512)
        head = next(j for j in joints if j["id"] == 1)
        assert head["name"] == "head"

    def test_spine_uses_spine1_not_spine(self) -> None:
        """Region 4 (spine) should use Spine1, not Spine."""
        projected = self._project_front()
        joints = _build_strata_joints(projected, 512)
        spine_joint = next(j for j in joints if j["id"] == 4)
        # Spine1 and Spine map to same region — the value should come from Spine1.
        spine1_xy = projected["Spine1"]
        assert spine_joint["x"] == pytest.approx(spine1_xy[0], rel=1e-3)


# ---------------------------------------------------------------------------
# _resize_to_strata
# ---------------------------------------------------------------------------


class TestResizeToStrata:
    def test_square_output(self) -> None:
        img = Image.new("RGBA", (1024, 1024), (0, 0, 0, 255))
        result, ox, oy = _resize_to_strata(img, 512)
        assert result.size == (512, 512)

    def test_no_padding_for_square(self) -> None:
        img = Image.new("RGBA", (1024, 1024), (0, 0, 0, 255))
        _, ox, oy = _resize_to_strata(img, 512)
        assert ox == 0
        assert oy == 0

    def test_converts_rgb_to_rgba(self) -> None:
        img = Image.new("RGB", (1024, 1024))
        result, _, _ = _resize_to_strata(img, 512)
        assert result.mode == "RGBA"


# ---------------------------------------------------------------------------
# _make_example_id
# ---------------------------------------------------------------------------


class TestMakeExampleId:
    def test_format(self, tmp_path: Path) -> None:
        sample_dir = _setup_sample_dir(tmp_path, sample_id=42)
        entry = HumanRigEntry(
            sample_id=42,
            sample_dir=sample_dir,
            image_path=sample_dir / "front.png",
            bone_2d_path=sample_dir / "bone_2d.json",
            bone_3d_path=sample_dir / "bone_3d.json",
            extrinsic_path=sample_dir / "extrinsic.npy",
            intrinsics_path=sample_dir / "intrinsics.npy",
        )
        assert _make_example_id(entry, "front") == "humanrig_00042_front"

    def test_angle_suffix(self, tmp_path: Path) -> None:
        sample_dir = _setup_sample_dir(tmp_path, sample_id=7)
        entry = HumanRigEntry(
            sample_id=7,
            sample_dir=sample_dir,
            image_path=sample_dir / "front.png",
            bone_2d_path=sample_dir / "bone_2d.json",
            bone_3d_path=sample_dir / "bone_3d.json",
            extrinsic_path=sample_dir / "extrinsic.npy",
            intrinsics_path=sample_dir / "intrinsics.npy",
        )
        assert _make_example_id(entry, "side") == "humanrig_00007_side"

    def test_zero_padded(self, tmp_path: Path) -> None:
        sample_dir = _setup_sample_dir(tmp_path, sample_id=1)
        entry = HumanRigEntry(
            sample_id=1,
            sample_dir=sample_dir,
            image_path=sample_dir / "front.png",
            bone_2d_path=sample_dir / "bone_2d.json",
            bone_3d_path=sample_dir / "bone_3d.json",
            extrinsic_path=sample_dir / "extrinsic.npy",
            intrinsics_path=sample_dir / "intrinsics.npy",
        )
        eid = _make_example_id(entry, "front")
        # Should be zero-padded to 5 digits
        assert eid == "humanrig_00001_front"


# ---------------------------------------------------------------------------
# convert_entry
# ---------------------------------------------------------------------------


class TestConvertEntry:
    def test_creates_front_example(self, tmp_path: Path) -> None:
        dataset_dir = _setup_dataset(tmp_path / "input", n=1)
        entries = discover_entries(dataset_dir)
        out = tmp_path / "output"

        n_saved = convert_entry(entries[0], out, angles=["front"])
        assert n_saved == 1

        example_dir = out / "humanrig_00000_front"
        assert (example_dir / "image.png").is_file()
        assert (example_dir / "joints.json").is_file()
        assert (example_dir / "metadata.json").is_file()

    def test_front_image_is_512(self, tmp_path: Path) -> None:
        dataset_dir = _setup_dataset(tmp_path / "input", n=1)
        entries = discover_entries(dataset_dir)
        out = tmp_path / "output"

        convert_entry(entries[0], out, angles=["front"])
        img = Image.open(out / "humanrig_00000_front" / "image.png")
        assert img.size == (512, 512)

    def test_joints_json_has_19_entries(self, tmp_path: Path) -> None:
        dataset_dir = _setup_dataset(tmp_path / "input", n=1)
        entries = discover_entries(dataset_dir)
        out = tmp_path / "output"

        convert_entry(entries[0], out, angles=["front"])
        joints = json.loads((out / "humanrig_00000_front" / "joints.json").read_text())
        assert len(joints) == 19

    def test_non_front_has_no_image(self, tmp_path: Path) -> None:
        dataset_dir = _setup_dataset(tmp_path / "input", n=1)
        entries = discover_entries(dataset_dir)
        out = tmp_path / "output"

        convert_entry(entries[0], out, angles=["side"])
        example_dir = out / "humanrig_00000_side"
        assert not (example_dir / "image.png").is_file()
        assert (example_dir / "joints.json").is_file()
        assert (example_dir / "metadata.json").is_file()

    def test_all_four_angles(self, tmp_path: Path) -> None:
        dataset_dir = _setup_dataset(tmp_path / "input", n=1)
        entries = discover_entries(dataset_dir)
        out = tmp_path / "output"

        n_saved = convert_entry(
            entries[0],
            out,
            angles=["front", "three_quarter", "side", "back"],
        )
        assert n_saved == 4
        for label in ["front", "three_quarter", "side", "back"]:
            assert (out / f"humanrig_00000_{label}").is_dir()

    def test_metadata_source(self, tmp_path: Path) -> None:
        dataset_dir = _setup_dataset(tmp_path / "input", n=1)
        entries = discover_entries(dataset_dir)
        out = tmp_path / "output"

        convert_entry(entries[0], out, angles=["front"])
        meta = json.loads((out / "humanrig_00000_front" / "metadata.json").read_text())
        assert meta["source"] == HUMANRIG_SOURCE

    def test_metadata_has_joints_true(self, tmp_path: Path) -> None:
        dataset_dir = _setup_dataset(tmp_path / "input", n=1)
        entries = discover_entries(dataset_dir)
        out = tmp_path / "output"

        convert_entry(entries[0], out, angles=["side"])
        meta = json.loads((out / "humanrig_00000_side" / "metadata.json").read_text())
        assert meta["has_joints"] is True
        assert meta["has_rendered_image"] is False

    def test_metadata_front_has_image(self, tmp_path: Path) -> None:
        dataset_dir = _setup_dataset(tmp_path / "input", n=1)
        entries = discover_entries(dataset_dir)
        out = tmp_path / "output"

        convert_entry(entries[0], out, angles=["front"])
        meta = json.loads((out / "humanrig_00000_front" / "metadata.json").read_text())
        assert meta["has_rendered_image"] is True

    def test_only_new_skips_existing(self, tmp_path: Path) -> None:
        dataset_dir = _setup_dataset(tmp_path / "input", n=1)
        entries = discover_entries(dataset_dir)
        out = tmp_path / "output"

        assert convert_entry(entries[0], out, angles=["front"]) == 1
        assert convert_entry(entries[0], out, angles=["front"], only_new=True) == 0

    def test_camera_angle_in_metadata(self, tmp_path: Path) -> None:
        dataset_dir = _setup_dataset(tmp_path / "input", n=1)
        entries = discover_entries(dataset_dir)
        out = tmp_path / "output"

        convert_entry(entries[0], out, angles=["three_quarter"])
        meta = json.loads(
            (out / "humanrig_00000_three_quarter" / "metadata.json").read_text()
        )
        assert meta["camera_angle"] == "three_quarter"
        assert meta["camera_azimuth_deg"] == 45


# ---------------------------------------------------------------------------
# convert_directory
# ---------------------------------------------------------------------------


class TestConvertDirectory:
    def test_converts_all_samples(self, tmp_path: Path) -> None:
        dataset_dir = _setup_dataset(tmp_path / "input", n=3)
        out = tmp_path / "output"

        # Default: 4 angles per sample.
        result = convert_directory(dataset_dir, out)
        assert isinstance(result, AdapterResult)
        assert result.images_processed == 3 * 4

    def test_front_only(self, tmp_path: Path) -> None:
        dataset_dir = _setup_dataset(tmp_path / "input", n=3)
        out = tmp_path / "output"

        result = convert_directory(dataset_dir, out, angles=["front"])
        assert result.images_processed == 3

    def test_max_images_limits_samples(self, tmp_path: Path) -> None:
        dataset_dir = _setup_dataset(tmp_path / "input", n=10)
        out = tmp_path / "output"

        result = convert_directory(dataset_dir, out, max_images=2, angles=["front"])
        assert result.images_processed == 2

    def test_random_sample_reproducible(self, tmp_path: Path) -> None:
        dataset_dir = _setup_dataset(tmp_path / "input", n=10)
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"

        convert_directory(
            dataset_dir, out1, max_images=3, random_sample=True, seed=42, angles=["front"]
        )
        convert_directory(
            dataset_dir, out2, max_images=3, random_sample=True, seed=42, angles=["front"]
        )

        dirs1 = sorted(d.name for d in out1.iterdir() if d.is_dir())
        dirs2 = sorted(d.name for d in out2.iterdir() if d.is_dir())
        assert dirs1 == dirs2

    def test_only_new_skips_existing(self, tmp_path: Path) -> None:
        dataset_dir = _setup_dataset(tmp_path / "input", n=3)
        out = tmp_path / "output"

        r1 = convert_directory(dataset_dir, out, angles=["front"])
        assert r1.images_processed == 3

        r2 = convert_directory(dataset_dir, out, only_new=True, angles=["front"])
        assert r2.images_processed == 0
        assert r2.images_skipped == 3

    def test_empty_input_dir(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        result = convert_directory(empty, tmp_path / "output")
        assert result.images_processed == 0

    def test_nonexistent_input_dir(self, tmp_path: Path) -> None:
        result = convert_directory(tmp_path / "nope", tmp_path / "output")
        assert result.images_processed == 0
