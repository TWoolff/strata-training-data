"""Tests for the CoNR adapter.

These tests exercise the pure-Python adapter logic without requiring
the actual CoNR dataset or Danbooru images.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from ingest.conr_adapter import (
    AdapterResult,
    _build_metadata,
    _resize_mask,
    _resize_to_strata,
    annotation_hash,
    convert_directory,
    convert_example,
    discover_annotations,
    find_image_for_annotation,
    label_to_fg_mask,
    load_annotation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HASH = "0a0f715b298cc59037ab4f317b97eb7a"


def _create_test_npz(
    path: Path,
    shape: tuple[int, int] = (256, 256),
    *,
    n_classes: int = 9,
) -> None:
    """Create a fake CoNR .npz annotation file."""
    label = np.random.randint(0, n_classes + 1, shape, dtype=np.uint8)
    np.savez_compressed(path, label=label)


def _create_test_image(
    path: Path,
    size: tuple[int, int] = (256, 256),
) -> None:
    """Create a test JPEG image."""
    arr = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    img.save(path)


def _setup_conr_dir(
    tmp_path: Path,
    *,
    num_examples: int = 3,
    image_size: tuple[int, int] = (256, 256),
    include_images: bool = True,
) -> Path:
    """Create a fake CoNR dataset directory.

    Layout::

        dataset_dir/
        ├── annotation/
        │   └── {hash}.jpg.npz
        └── images/
            └── {hash}.jpg
    """
    dataset_dir = tmp_path / "conr"
    ann_dir = dataset_dir / "annotation"
    img_dir = dataset_dir / "images"
    ann_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)

    for i in range(num_examples):
        h = f"{i:032x}"
        _create_test_npz(ann_dir / f"{h}.jpg.npz", shape=image_size)
        if include_images:
            _create_test_image(img_dir / f"{h}.jpg", size=image_size)

    return dataset_dir


# ---------------------------------------------------------------------------
# annotation_hash
# ---------------------------------------------------------------------------


class TestAnnotationHash:
    """Test hash extraction from annotation filenames."""

    def test_standard_filename(self) -> None:
        p = Path(f"/data/annotation/{_HASH}.jpg.npz")
        assert annotation_hash(p) == _HASH

    def test_png_extension(self) -> None:
        p = Path(f"/data/annotation/{_HASH}.png.npz")
        assert annotation_hash(p) == _HASH

    def test_jpeg_extension(self) -> None:
        p = Path(f"/data/annotation/{_HASH}.jpeg.npz")
        assert annotation_hash(p) == _HASH


# ---------------------------------------------------------------------------
# discover_annotations
# ---------------------------------------------------------------------------


class TestDiscoverAnnotations:
    """Test annotation file discovery."""

    def test_discovers_all_npz(self, tmp_path: Path) -> None:
        dataset = _setup_conr_dir(tmp_path, num_examples=5)
        ann_dir = dataset / "annotation"
        paths = discover_annotations(ann_dir)
        assert len(paths) == 5

    def test_sorted_by_path(self, tmp_path: Path) -> None:
        dataset = _setup_conr_dir(tmp_path, num_examples=3)
        ann_dir = dataset / "annotation"
        paths = discover_annotations(ann_dir)
        assert paths == sorted(paths)

    def test_ignores_non_npz(self, tmp_path: Path) -> None:
        dataset = _setup_conr_dir(tmp_path, num_examples=2)
        ann_dir = dataset / "annotation"
        (ann_dir / "readme.txt").write_text("ignore me")
        paths = discover_annotations(ann_dir)
        assert len(paths) == 2

    def test_empty_directory(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        assert discover_annotations(empty) == []

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        assert discover_annotations(tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# find_image_for_annotation
# ---------------------------------------------------------------------------


class TestFindImage:
    """Test image lookup for annotation files."""

    def test_finds_jpg(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img_path = img_dir / f"{_HASH}.jpg"
        _create_test_image(img_path)

        npz_path = tmp_path / f"{_HASH}.jpg.npz"
        result = find_image_for_annotation(npz_path, img_dir)
        assert result == img_path

    def test_finds_png(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img_path = img_dir / f"{_HASH}.png"
        _create_test_image(img_path)

        npz_path = tmp_path / f"{_HASH}.jpg.npz"
        result = find_image_for_annotation(npz_path, img_dir)
        assert result == img_path

    def test_returns_none_when_missing(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        npz_path = tmp_path / f"{_HASH}.jpg.npz"
        assert find_image_for_annotation(npz_path, img_dir) is None


# ---------------------------------------------------------------------------
# load_annotation / label_to_fg_mask
# ---------------------------------------------------------------------------


class TestAnnotationLoading:
    """Test annotation loading and mask conversion."""

    def test_load_annotation(self, tmp_path: Path) -> None:
        npz_path = tmp_path / "test.jpg.npz"
        expected = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
        np.savez_compressed(npz_path, label=expected)

        loaded = load_annotation(npz_path)
        assert loaded is not None
        np.testing.assert_array_equal(loaded, expected)

    def test_load_annotation_invalid(self, tmp_path: Path) -> None:
        bad_path = tmp_path / "bad.npz"
        bad_path.write_bytes(b"not a valid npz")
        assert load_annotation(bad_path) is None

    def test_label_to_fg_mask_binary(self) -> None:
        label = np.array([[0, 0, 1], [3, 0, 9]], dtype=np.uint8)
        mask = label_to_fg_mask(label)
        expected = np.array([[0, 0, 255], [255, 0, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(mask, expected)

    def test_fg_mask_all_background(self) -> None:
        label = np.zeros((4, 4), dtype=np.uint8)
        mask = label_to_fg_mask(label)
        assert mask.max() == 0

    def test_fg_mask_all_foreground(self) -> None:
        label = np.ones((4, 4), dtype=np.uint8) * 5
        mask = label_to_fg_mask(label)
        assert mask.min() == 255

    def test_fg_mask_255_is_background(self) -> None:
        """Value 255 in CoNR labels is unlabeled — should be background."""
        label = np.array([[0, 255, 1], [9, 255, 0]], dtype=np.uint8)
        mask = label_to_fg_mask(label)
        expected = np.array([[0, 0, 255], [255, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(mask, expected)


# ---------------------------------------------------------------------------
# _resize_to_strata / _resize_mask
# ---------------------------------------------------------------------------


class TestResize:
    """Test image and mask resizing."""

    def test_resize_preserves_aspect(self) -> None:
        arr = np.random.randint(0, 255, (512, 256, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        result = _resize_to_strata(img, 512)
        assert result.size == (512, 512)
        assert result.mode == "RGBA"

    def test_resize_mask_nearest_neighbor(self) -> None:
        mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        result = _resize_mask(mask, 512)
        assert result.size == (512, 512)
        assert result.mode == "L"
        arr = np.array(result)
        # Values should only be 0 or 255 (nearest-neighbor, no interpolation)
        unique = set(np.unique(arr))
        assert unique.issubset({0, 255})


# ---------------------------------------------------------------------------
# _build_metadata
# ---------------------------------------------------------------------------


class TestBuildMetadata:
    """Test metadata generation."""

    def test_required_fields(self) -> None:
        meta = _build_metadata(
            "conr_abc123",
            "abc123.jpg.npz",
            512,
            original_size=(256, 256),
            content_hash="abc123",
        )
        assert meta["id"] == "conr_abc123"
        assert meta["source"] == "conr"
        assert meta["has_fg_mask"] is True
        assert meta["has_segmentation_mask"] is False
        assert meta["has_joints"] is False
        assert meta["content_hash"] == "abc123"

    def test_missing_annotations(self) -> None:
        meta = _build_metadata(
            "conr_x",
            "x.jpg.npz",
            512,
            original_size=(256, 256),
            content_hash="x",
        )
        assert "strata_segmentation" in meta["missing_annotations"]
        assert "joints" in meta["missing_annotations"]
        assert "draw_order" in meta["missing_annotations"]


# ---------------------------------------------------------------------------
# convert_example
# ---------------------------------------------------------------------------


class TestConvertExample:
    """Test single-example conversion."""

    def test_creates_output_structure(self, tmp_path: Path) -> None:
        dataset = _setup_conr_dir(tmp_path, num_examples=1)
        ann_dir = dataset / "annotation"
        img_dir = dataset / "images"
        output_dir = tmp_path / "output"

        npz_path = next(ann_dir.glob("*.npz"))
        saved = convert_example(npz_path, img_dir, output_dir)

        assert saved is True
        h = annotation_hash(npz_path)
        example_dir = output_dir / f"conr_{h}"
        assert (example_dir / "image.png").is_file()
        assert (example_dir / "segmentation.png").is_file()
        assert (example_dir / "metadata.json").is_file()

    def test_output_is_square(self, tmp_path: Path) -> None:
        dataset = _setup_conr_dir(tmp_path, num_examples=1)
        ann_dir = dataset / "annotation"
        img_dir = dataset / "images"
        output_dir = tmp_path / "output"

        npz_path = next(ann_dir.glob("*.npz"))
        convert_example(npz_path, img_dir, output_dir)

        h = annotation_hash(npz_path)
        img = Image.open(output_dir / f"conr_{h}" / "image.png")
        assert img.size == (512, 512)

    def test_returns_none_when_image_missing(self, tmp_path: Path) -> None:
        dataset = _setup_conr_dir(tmp_path, num_examples=1, include_images=False)
        ann_dir = dataset / "annotation"
        img_dir = dataset / "images"
        output_dir = tmp_path / "output"

        npz_path = next(ann_dir.glob("*.npz"))
        result = convert_example(npz_path, img_dir, output_dir)
        assert result is None

    def test_only_new_skips_existing(self, tmp_path: Path) -> None:
        dataset = _setup_conr_dir(tmp_path, num_examples=1)
        ann_dir = dataset / "annotation"
        img_dir = dataset / "images"
        output_dir = tmp_path / "output"

        npz_path = next(ann_dir.glob("*.npz"))
        assert convert_example(npz_path, img_dir, output_dir) is True
        assert convert_example(npz_path, img_dir, output_dir, only_new=True) is False

    def test_metadata_source(self, tmp_path: Path) -> None:
        dataset = _setup_conr_dir(tmp_path, num_examples=1)
        ann_dir = dataset / "annotation"
        img_dir = dataset / "images"
        output_dir = tmp_path / "output"

        npz_path = next(ann_dir.glob("*.npz"))
        convert_example(npz_path, img_dir, output_dir)

        h = annotation_hash(npz_path)
        meta = json.loads((output_dir / f"conr_{h}" / "metadata.json").read_text())
        assert meta["source"] == "conr"
        assert meta["has_fg_mask"] is True


# ---------------------------------------------------------------------------
# convert_directory
# ---------------------------------------------------------------------------


class TestConvertDirectory:
    """Test batch directory conversion."""

    def test_converts_all_examples(self, tmp_path: Path) -> None:
        dataset = _setup_conr_dir(tmp_path, num_examples=4)
        output_dir = tmp_path / "output"

        result = convert_directory(dataset, output_dir)
        assert isinstance(result, AdapterResult)
        assert result.images_processed == 4

    def test_max_images_limits_output(self, tmp_path: Path) -> None:
        dataset = _setup_conr_dir(tmp_path, num_examples=5)
        output_dir = tmp_path / "output"

        result = convert_directory(dataset, output_dir, max_images=2)
        assert result.images_processed == 2

    def test_random_sample_reproducible(self, tmp_path: Path) -> None:
        dataset = _setup_conr_dir(tmp_path, num_examples=5)
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"

        convert_directory(dataset, out1, max_images=3, random_sample=True, seed=42)
        convert_directory(dataset, out2, max_images=3, random_sample=True, seed=42)

        dirs1 = sorted(d.name for d in out1.iterdir() if d.is_dir())
        dirs2 = sorted(d.name for d in out2.iterdir() if d.is_dir())
        assert dirs1 == dirs2

    def test_only_new_skips(self, tmp_path: Path) -> None:
        dataset = _setup_conr_dir(tmp_path, num_examples=3)
        output_dir = tmp_path / "output"

        r1 = convert_directory(dataset, output_dir)
        assert r1.images_processed == 3

        r2 = convert_directory(dataset, output_dir, only_new=True)
        assert r2.images_processed == 0
        assert r2.images_skipped == 3

    def test_missing_images_counted(self, tmp_path: Path) -> None:
        dataset = _setup_conr_dir(tmp_path, num_examples=3, include_images=False)
        output_dir = tmp_path / "output"

        result = convert_directory(dataset, output_dir)
        assert result.images_missing == 3
        assert result.images_processed == 0

    def test_empty_annotation_dir(self, tmp_path: Path) -> None:
        dataset = tmp_path / "conr"
        (dataset / "annotation").mkdir(parents=True)
        (dataset / "images").mkdir(parents=True)

        result = convert_directory(dataset, tmp_path / "output")
        assert result.images_processed == 0

    def test_no_annotation_dir(self, tmp_path: Path) -> None:
        result = convert_directory(tmp_path / "nope", tmp_path / "output")
        assert result.images_processed == 0
