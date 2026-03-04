"""Tests for the AnimeDrawingsDataset adapter.

These tests exercise the pure-Python adapter logic without requiring
the actual dataset or any external downloads.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from ingest.anime_drawings_adapter import (
    AdapterResult,
    _build_metadata,
    _build_strata_joints,
    _resize_to_strata,
    _resolve_image_path,
    convert_directory,
    convert_image,
    load_annotations,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Sample annotation entry matching the real dataset format.
_SAMPLE_POINTS = {
    "head": [186, 50],
    "neck": [186, 100],
    "nose_tip": [190, 65],
    "nose_root": [188, 75],
    "body_upper": [186, 150],
    "arm_left": [220, 120],
    "arm_right": [150, 120],
    "elbow_left": [250, 180],
    "elbow_right": [120, 180],
    "wrist_left": [270, 240],
    "wrist_right": [100, 240],
    "thumb_left": [275, 250],
    "thumb_right": [95, 250],
    "leg_left": [200, 300],
    "leg_right": [170, 300],
    "knee_left": [210, 400],
    "knee_right": [160, 400],
    "ankle_left": [215, 480],
    "ankle_right": [155, 480],
    "tiptoe_left": [220, 500],
    "tiptoe_right": [150, 500],
}


def _create_test_image(
    size: tuple[int, int] = (371, 600),
    mode: str = "RGB",
) -> Image.Image:
    """Create a test image matching typical AnimeDrawingsDataset dimensions."""
    channels = 4 if mode == "RGBA" else 3
    arr = np.random.randint(0, 255, (size[1], size[0], channels), dtype=np.uint8)
    return Image.fromarray(arr, mode=mode)


def _make_annotation(
    file_name: str = "data/images/test_001.png",
    width: int = 371,
    height: int = 600,
    points: dict | None = None,
) -> dict:
    """Build a single annotation entry."""
    return {
        "file_name": file_name,
        "width": width,
        "height": height,
        "points": points if points is not None else dict(_SAMPLE_POINTS),
    }


def _setup_dataset_dir(
    tmp_path: Path,
    *,
    num_images: int = 3,
    image_size: tuple[int, int] = (371, 600),
    split: str | None = None,
) -> Path:
    """Create a fake AnimeDrawingsDataset directory.

    Layout::

        dataset_dir/
        ├── data.json (or train.json etc.)
        └── images/
            ├── test_000.png
            ├── test_001.png
            └── test_002.png
    """
    dataset_dir = tmp_path / "anime_drawings"
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True)

    entries = []
    for i in range(num_images):
        filename = f"test_{i:03d}.png"
        img = _create_test_image(image_size)
        img.save(images_dir / filename)
        entries.append(
            _make_annotation(
                file_name=f"images/{filename}",
                width=image_size[0],
                height=image_size[1],
            )
        )

    json_name = f"{split}.json" if split else "data.json"
    (dataset_dir / json_name).write_text(
        json.dumps(entries, indent=2),
        encoding="utf-8",
    )

    return dataset_dir


# ---------------------------------------------------------------------------
# load_annotations
# ---------------------------------------------------------------------------


class TestLoadAnnotations:
    """Test JSON annotation loading."""

    def test_loads_valid_json(self, tmp_path: Path) -> None:
        dataset = _setup_dataset_dir(tmp_path, num_images=3)
        entries = load_annotations(dataset / "data.json")
        assert len(entries) == 3

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        entries = load_annotations(tmp_path / "nope.json")
        assert entries == []

    def test_invalid_json_type(self, tmp_path: Path) -> None:
        bad_json = tmp_path / "bad.json"
        bad_json.write_text('{"not": "a list"}')
        entries = load_annotations(bad_json)
        assert entries == []


# ---------------------------------------------------------------------------
# _resolve_image_path
# ---------------------------------------------------------------------------


class TestResolveImagePath:
    """Test image path resolution logic."""

    def test_resolves_relative_to_root(self, tmp_path: Path) -> None:
        dataset = _setup_dataset_dir(tmp_path, num_images=1)
        entry = _make_annotation(file_name="images/test_000.png")
        path = _resolve_image_path(dataset, entry)
        assert path is not None
        assert path.is_file()

    def test_resolves_with_data_prefix(self, tmp_path: Path) -> None:
        # Create layout where images are at data/images/ relative to parent.
        data_dir = tmp_path / "repo" / "data"
        images_dir = data_dir / "images"
        images_dir.mkdir(parents=True)
        img = _create_test_image()
        img.save(images_dir / "test.png")

        entry = _make_annotation(file_name="data/images/test.png")
        path = _resolve_image_path(data_dir.parent, entry)
        assert path is not None

    def test_missing_image_returns_none(self, tmp_path: Path) -> None:
        entry = _make_annotation(file_name="images/nonexistent.png")
        path = _resolve_image_path(tmp_path, entry)
        assert path is None

    def test_empty_filename(self, tmp_path: Path) -> None:
        entry = _make_annotation(file_name="")
        path = _resolve_image_path(tmp_path, entry)
        assert path is None


# ---------------------------------------------------------------------------
# _resize_to_strata
# ---------------------------------------------------------------------------


class TestResizeToStrata:
    """Test aspect-ratio-preserving resize with padding."""

    def test_tall_image_padded(self) -> None:
        img = _create_test_image((371, 600))
        result, scale, _ox, _oy = _resize_to_strata(img, 512)
        assert result.size == (512, 512)
        assert result.mode == "RGBA"
        assert scale > 0

    def test_already_correct_size(self) -> None:
        img = _create_test_image((512, 512), mode="RGBA")
        result, scale, ox, oy = _resize_to_strata(img, 512)
        assert result.size == (512, 512)
        assert scale == 1.0
        assert ox == 0
        assert oy == 0

    def test_converts_rgb_to_rgba(self) -> None:
        img = _create_test_image((300, 400), mode="RGB")
        result, _, _, _ = _resize_to_strata(img, 512)
        assert result.mode == "RGBA"

    def test_custom_resolution(self) -> None:
        img = _create_test_image((371, 600))
        result, _, _, _ = _resize_to_strata(img, 256)
        assert result.size == (256, 256)


# ---------------------------------------------------------------------------
# _build_strata_joints
# ---------------------------------------------------------------------------


class TestBuildStrataJoints:
    """Test joint coordinate mapping to Strata format."""

    def test_produces_19_joints(self) -> None:
        joints = _build_strata_joints(_SAMPLE_POINTS, 1.0, 0, 0, 512)
        assert len(joints) == 19

    def test_joints_ordered_by_id(self) -> None:
        joints = _build_strata_joints(_SAMPLE_POINTS, 1.0, 0, 0, 512)
        ids = [j["id"] for j in joints]
        assert ids == list(range(1, 20))

    def test_mapped_joints_have_coordinates(self) -> None:
        joints = _build_strata_joints(_SAMPLE_POINTS, 1.0, 0, 0, 512)
        head = joints[0]  # id=1, head
        assert head["name"] == "head"
        assert head["x"] == 186
        assert head["y"] == 50
        assert head["visible"] is True

    def test_unmapped_joints_invisible(self) -> None:
        joints = _build_strata_joints(_SAMPLE_POINTS, 1.0, 0, 0, 512)
        # spine (id=4), hips (id=5), shoulder_l (id=6), shoulder_r (id=10)
        for region_id in (4, 5, 6, 10):
            joint = joints[region_id - 1]
            assert joint["visible"] is False
            assert joint["x"] == 0
            assert joint["y"] == 0

    def test_scale_and_offset_applied(self) -> None:
        scale = 0.5
        ox, oy = 50, 100
        joints = _build_strata_joints(_SAMPLE_POINTS, scale, ox, oy, 512)
        head = joints[0]
        # head raw coords: [186, 50]
        assert head["x"] == round(186 * 0.5 + 50, 2)
        assert head["y"] == round(50 * 0.5 + 100, 2)

    def test_out_of_bounds_marked_invisible(self) -> None:
        # Place joints far outside canvas with large offset.
        joints = _build_strata_joints({"head": [1000, 1000]}, 1.0, 0, 0, 512)
        head = joints[0]
        assert head["visible"] is False
        assert head["x"] == 1000

    def test_empty_points(self) -> None:
        joints = _build_strata_joints({}, 1.0, 0, 0, 512)
        assert len(joints) == 19
        assert all(j["visible"] is False for j in joints)


# ---------------------------------------------------------------------------
# _build_metadata
# ---------------------------------------------------------------------------


class TestBuildMetadata:
    """Test metadata generation."""

    def test_required_fields(self) -> None:
        meta = _build_metadata(
            "anime_drawings_test",
            "test.png",
            512,
            original_size=(371, 600),
        )
        assert meta["id"] == "anime_drawings_test"
        assert meta["source"] == "anime_drawings"
        assert meta["resolution"] == 512
        assert meta["has_joints"] is True

    def test_missing_annotations(self) -> None:
        meta = _build_metadata(
            "test",
            "test.png",
            512,
            original_size=(371, 600),
        )
        assert meta["has_segmentation_mask"] is False
        assert meta["has_draw_order"] is False
        assert "strata_segmentation" in meta["missing_annotations"]

    def test_split_recorded(self) -> None:
        meta = _build_metadata(
            "test",
            "test.png",
            512,
            original_size=(371, 600),
            split="train",
        )
        assert meta["split"] == "train"

    def test_joints_mapped_count(self) -> None:
        meta = _build_metadata(
            "test",
            "test.png",
            512,
            original_size=(371, 600),
            joints_mapped=15,
        )
        assert meta["joints_mapped"] == 15
        assert meta["joints_total"] == 19


# ---------------------------------------------------------------------------
# convert_image
# ---------------------------------------------------------------------------


class TestConvertImage:
    """Test single-image conversion."""

    def test_creates_output_structure(self, tmp_path: Path) -> None:
        dataset = _setup_dataset_dir(tmp_path / "input", num_images=1)
        entry = load_annotations(dataset / "data.json")[0]
        output_dir = tmp_path / "output"

        saved = convert_image(entry, dataset, output_dir)
        assert saved is True

        example_dir = output_dir / "anime_drawings_test_000"
        assert (example_dir / "image.png").is_file()
        assert (example_dir / "joints.json").is_file()
        assert (example_dir / "metadata.json").is_file()

    def test_output_is_square(self, tmp_path: Path) -> None:
        dataset = _setup_dataset_dir(tmp_path / "input", num_images=1)
        entry = load_annotations(dataset / "data.json")[0]
        output_dir = tmp_path / "output"

        convert_image(entry, dataset, output_dir)

        example_dir = output_dir / "anime_drawings_test_000"
        img = Image.open(example_dir / "image.png")
        assert img.size == (512, 512)

    def test_joints_json_has_19_entries(self, tmp_path: Path) -> None:
        dataset = _setup_dataset_dir(tmp_path / "input", num_images=1)
        entry = load_annotations(dataset / "data.json")[0]
        output_dir = tmp_path / "output"

        convert_image(entry, dataset, output_dir)

        example_dir = output_dir / "anime_drawings_test_000"
        joints = json.loads((example_dir / "joints.json").read_text())
        assert len(joints) == 19
        assert all("id" in j and "name" in j and "x" in j and "y" in j for j in joints)

    def test_only_new_skips_existing(self, tmp_path: Path) -> None:
        dataset = _setup_dataset_dir(tmp_path / "input", num_images=1)
        entry = load_annotations(dataset / "data.json")[0]
        output_dir = tmp_path / "output"

        assert convert_image(entry, dataset, output_dir) is True
        assert convert_image(entry, dataset, output_dir, only_new=True) is False

    def test_missing_image_returns_false(self, tmp_path: Path) -> None:
        entry = _make_annotation(file_name="images/nonexistent.png")
        assert convert_image(entry, tmp_path, tmp_path / "output") is False


# ---------------------------------------------------------------------------
# convert_directory
# ---------------------------------------------------------------------------


class TestConvertDirectory:
    """Test batch directory conversion."""

    def test_converts_all_images(self, tmp_path: Path) -> None:
        dataset = _setup_dataset_dir(tmp_path / "input", num_images=5)
        output_dir = tmp_path / "output"

        result = convert_directory(dataset, output_dir)
        assert isinstance(result, AdapterResult)
        assert result.images_processed == 5

    def test_max_images_limits_output(self, tmp_path: Path) -> None:
        dataset = _setup_dataset_dir(tmp_path / "input", num_images=5)
        output_dir = tmp_path / "output"

        result = convert_directory(dataset, output_dir, max_images=2)
        assert result.images_processed == 2

    def test_random_sample_reproducible(self, tmp_path: Path) -> None:
        dataset = _setup_dataset_dir(tmp_path / "input", num_images=10)
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"

        convert_directory(dataset, out1, max_images=3, random_sample=True, seed=42)
        convert_directory(dataset, out2, max_images=3, random_sample=True, seed=42)

        dirs1 = sorted(d.name for d in out1.iterdir() if d.is_dir())
        dirs2 = sorted(d.name for d in out2.iterdir() if d.is_dir())
        assert dirs1 == dirs2

    def test_random_sample_different_seed(self, tmp_path: Path) -> None:
        dataset = _setup_dataset_dir(tmp_path / "input", num_images=10)
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"

        convert_directory(dataset, out1, max_images=3, random_sample=True, seed=1)
        convert_directory(dataset, out2, max_images=3, random_sample=True, seed=99)

        dirs1 = sorted(d.name for d in out1.iterdir() if d.is_dir())
        dirs2 = sorted(d.name for d in out2.iterdir() if d.is_dir())
        assert dirs1 != dirs2

    def test_only_new_skips(self, tmp_path: Path) -> None:
        dataset = _setup_dataset_dir(tmp_path / "input", num_images=3)
        output_dir = tmp_path / "output"

        r1 = convert_directory(dataset, output_dir)
        assert r1.images_processed == 3

        r2 = convert_directory(dataset, output_dir, only_new=True)
        assert r2.images_processed == 0
        assert r2.images_skipped == 3

    def test_empty_directory(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        result = convert_directory(empty, tmp_path / "output")
        assert result.images_processed == 0

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        result = convert_directory(tmp_path / "nope", tmp_path / "output")
        assert result.images_processed == 0

    def test_falls_back_to_split_files(self, tmp_path: Path) -> None:
        """When data.json doesn't exist, loads train/val/test.json."""
        dataset = _setup_dataset_dir(tmp_path / "input", num_images=3, split="train")
        output_dir = tmp_path / "output"

        result = convert_directory(dataset, output_dir)
        assert result.images_processed == 3
