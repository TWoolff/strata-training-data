"""Tests for the FBAnimeHQ adapter.

These tests exercise the pure-Python adapter logic without requiring
Blender or the actual FBAnimeHQ dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from ingest.fbanimehq_adapter import (
    AdapterResult,
    _build_metadata,
    _resize_to_strata,
    _save_example,
    convert_directory,
    convert_image,
    discover_images,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_test_image(
    size: tuple[int, int] = (512, 1024),
    mode: str = "RGB",
) -> Image.Image:
    """Create a test image matching FBAnimeHQ's default dimensions."""
    channels = 4 if mode == "RGBA" else 3
    arr = np.random.randint(0, 255, (size[1], size[0], channels), dtype=np.uint8)
    return Image.fromarray(arr, mode=mode)


def _setup_shard_dir(
    tmp_path: Path,
    *,
    num_buckets: int = 2,
    images_per_bucket: int = 3,
    image_size: tuple[int, int] = (512, 1024),
) -> Path:
    """Create a fake FBAnimeHQ shard directory.

    Mimics the real layout::

        shard_dir/
        ├── 0000/
        │   ├── 000000.png
        │   ├── 000001.png
        │   └── 000002.png
        └── 0001/
            ├── 000003.png
            └── …
    """
    shard_dir = tmp_path / "fbanimehq-00"
    idx = 0
    for bucket in range(num_buckets):
        bucket_dir = shard_dir / f"{bucket:04d}"
        bucket_dir.mkdir(parents=True)
        for _ in range(images_per_bucket):
            img = _create_test_image(image_size, mode="RGB")
            img.save(bucket_dir / f"{idx:06d}.png")
            idx += 1
    return shard_dir


# ---------------------------------------------------------------------------
# discover_images
# ---------------------------------------------------------------------------


class TestDiscoverImages:
    """Test image discovery across the shard/bucket hierarchy."""

    def test_discovers_all_images(self, tmp_path: Path) -> None:
        shard = _setup_shard_dir(tmp_path, num_buckets=2, images_per_bucket=3)
        paths = discover_images(shard)
        assert len(paths) == 6

    def test_sorted_by_path(self, tmp_path: Path) -> None:
        shard = _setup_shard_dir(tmp_path, num_buckets=2, images_per_bucket=3)
        paths = discover_images(shard)
        assert paths == sorted(paths)

    def test_discovers_jpg(self, tmp_path: Path) -> None:
        shard = _setup_shard_dir(tmp_path, num_buckets=1, images_per_bucket=1)
        # Add a JPG file
        img = _create_test_image()
        img.save(shard / "0000" / "extra.jpg")
        paths = discover_images(shard)
        assert len(paths) == 2

    def test_ignores_non_image_files(self, tmp_path: Path) -> None:
        shard = _setup_shard_dir(tmp_path, num_buckets=1, images_per_bucket=2)
        (shard / "0000" / "readme.txt").write_text("ignore me")
        (shard / "0000" / "meta.json").write_text("{}")
        paths = discover_images(shard)
        assert len(paths) == 2

    def test_empty_directory(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        assert discover_images(empty) == []

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        assert discover_images(tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# _resize_to_strata
# ---------------------------------------------------------------------------


class TestResizeToStrata:
    """Test aspect-ratio-preserving resize with padding."""

    def test_tall_image_padded_horizontally(self) -> None:
        """512×1024 (portrait) → 512×512 with horizontal padding."""
        img = _create_test_image((512, 1024))
        result = _resize_to_strata(img, 512)
        assert result.size == (512, 512)
        assert result.mode == "RGBA"

    def test_wide_image_padded_vertically(self) -> None:
        """1024×512 (landscape) → 512×512 with vertical padding."""
        img = _create_test_image((1024, 512))
        result = _resize_to_strata(img, 512)
        assert result.size == (512, 512)
        assert result.mode == "RGBA"

    def test_square_image_no_change(self) -> None:
        img = _create_test_image((512, 512))
        result = _resize_to_strata(img, 512)
        assert result.size == (512, 512)

    def test_already_correct_size_rgba(self) -> None:
        img = _create_test_image((512, 512), mode="RGBA")
        result = _resize_to_strata(img, 512)
        assert result.size == (512, 512)

    def test_converts_rgb_to_rgba(self) -> None:
        img = _create_test_image((512, 1024), mode="RGB")
        result = _resize_to_strata(img, 512)
        assert result.mode == "RGBA"

    def test_transparent_padding(self) -> None:
        """Padding pixels should be fully transparent."""
        img = _create_test_image((512, 1024))
        result = _resize_to_strata(img, 512)
        arr = np.array(result)
        # Top-left corner should be padding (alpha = 0) for a tall image
        # The image is centered: 256 wide centered in 512 → padding on sides
        assert arr[0, 0, 3] == 0  # top-left is padding

    def test_custom_resolution(self) -> None:
        img = _create_test_image((512, 1024))
        result = _resize_to_strata(img, 256)
        assert result.size == (256, 256)

    def test_small_image_scaled_up(self) -> None:
        img = _create_test_image((64, 128))
        result = _resize_to_strata(img, 512)
        assert result.size == (512, 512)


# ---------------------------------------------------------------------------
# _build_metadata
# ---------------------------------------------------------------------------


class TestBuildMetadata:
    """Test metadata generation."""

    def test_required_fields(self) -> None:
        meta = _build_metadata(
            "fbanimehq_000000",
            Path("/fake/000000.png"),
            512,
            original_size=(512, 1024),
        )
        assert meta["id"] == "fbanimehq_000000"
        assert meta["source"] == "fbanimehq"
        assert meta["source_filename"] == "000000.png"
        assert meta["resolution"] == 512

    def test_missing_annotations(self) -> None:
        meta = _build_metadata(
            "fbanimehq_000000",
            Path("/fake/000000.png"),
            512,
            original_size=(512, 1024),
        )
        assert meta["has_segmentation_mask"] is False
        assert meta["has_fg_mask"] is False
        assert meta["has_joints"] is False
        assert meta["has_draw_order"] is False
        assert "strata_segmentation" in meta["missing_annotations"]
        assert "fg_mask" in meta["missing_annotations"]

    def test_original_size_recorded(self) -> None:
        meta = _build_metadata(
            "fbanimehq_000000",
            Path("/fake/000000.png"),
            512,
            original_size=(512, 1024),
        )
        assert meta["original_width"] == 512
        assert meta["original_height"] == 1024

    def test_padding_flag_nonsquare(self) -> None:
        meta = _build_metadata(
            "test", Path("/f.png"), 512, original_size=(512, 1024),
        )
        assert meta["padding_applied"] is True

    def test_padding_flag_square(self) -> None:
        meta = _build_metadata(
            "test", Path("/f.png"), 512, original_size=(512, 512),
        )
        assert meta["padding_applied"] is False


# ---------------------------------------------------------------------------
# convert_image
# ---------------------------------------------------------------------------


class TestConvertImage:
    """Test single-image conversion."""

    def test_creates_output_structure(self, tmp_path: Path) -> None:
        shard = _setup_shard_dir(tmp_path / "input", num_buckets=1, images_per_bucket=1)
        image_path = next((shard / "0000").glob("*.png"))
        output_dir = tmp_path / "output"

        saved = convert_image(image_path, output_dir)
        assert saved is True

        example_dir = output_dir / f"fbanimehq_{image_path.parent.name}_{image_path.stem}"
        assert (example_dir / "image.png").is_file()
        assert (example_dir / "metadata.json").is_file()

    def test_output_is_square(self, tmp_path: Path) -> None:
        shard = _setup_shard_dir(tmp_path / "input", num_buckets=1, images_per_bucket=1)
        image_path = next((shard / "0000").glob("*.png"))
        output_dir = tmp_path / "output"

        convert_image(image_path, output_dir)

        example_dir = output_dir / f"fbanimehq_{image_path.parent.name}_{image_path.stem}"
        img = Image.open(example_dir / "image.png")
        assert img.size == (512, 512)

    def test_only_new_skips_existing(self, tmp_path: Path) -> None:
        shard = _setup_shard_dir(tmp_path / "input", num_buckets=1, images_per_bucket=1)
        image_path = next((shard / "0000").glob("*.png"))
        output_dir = tmp_path / "output"

        assert convert_image(image_path, output_dir) is True
        assert convert_image(image_path, output_dir, only_new=True) is False

    def test_invalid_path_returns_false(self, tmp_path: Path) -> None:
        assert convert_image(tmp_path / "nope.png", tmp_path / "out") is False

    def test_metadata_source(self, tmp_path: Path) -> None:
        shard = _setup_shard_dir(tmp_path / "input", num_buckets=1, images_per_bucket=1)
        image_path = next((shard / "0000").glob("*.png"))
        output_dir = tmp_path / "output"

        convert_image(image_path, output_dir)

        example_dir = output_dir / f"fbanimehq_{image_path.parent.name}_{image_path.stem}"
        meta = json.loads((example_dir / "metadata.json").read_text())
        assert meta["source"] == "fbanimehq"

    def test_custom_resolution(self, tmp_path: Path) -> None:
        shard = _setup_shard_dir(tmp_path / "input", num_buckets=1, images_per_bucket=1)
        image_path = next((shard / "0000").glob("*.png"))
        output_dir = tmp_path / "output"

        convert_image(image_path, output_dir, resolution=256)

        example_dir = output_dir / f"fbanimehq_{image_path.parent.name}_{image_path.stem}"
        img = Image.open(example_dir / "image.png")
        assert img.size == (256, 256)


# ---------------------------------------------------------------------------
# convert_directory
# ---------------------------------------------------------------------------


class TestConvertDirectory:
    """Test batch directory conversion."""

    def test_converts_all_images(self, tmp_path: Path) -> None:
        shard = _setup_shard_dir(tmp_path / "input", num_buckets=2, images_per_bucket=3)
        output_dir = tmp_path / "output"

        result = convert_directory(shard, output_dir)
        assert isinstance(result, AdapterResult)
        assert result.images_processed == 6

    def test_max_images_limits_output(self, tmp_path: Path) -> None:
        shard = _setup_shard_dir(tmp_path / "input", num_buckets=2, images_per_bucket=5)
        output_dir = tmp_path / "output"

        result = convert_directory(shard, output_dir, max_images=3)
        assert result.images_processed == 3

    def test_random_sample_reproducible(self, tmp_path: Path) -> None:
        shard = _setup_shard_dir(tmp_path / "input", num_buckets=2, images_per_bucket=5)
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"

        convert_directory(shard, out1, max_images=3, random_sample=True, seed=42)
        convert_directory(shard, out2, max_images=3, random_sample=True, seed=42)

        dirs1 = sorted(d.name for d in out1.iterdir() if d.is_dir())
        dirs2 = sorted(d.name for d in out2.iterdir() if d.is_dir())
        assert dirs1 == dirs2

    def test_random_sample_different_seed(self, tmp_path: Path) -> None:
        shard = _setup_shard_dir(tmp_path / "input", num_buckets=2, images_per_bucket=5)
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"

        convert_directory(shard, out1, max_images=3, random_sample=True, seed=1)
        convert_directory(shard, out2, max_images=3, random_sample=True, seed=99)

        dirs1 = sorted(d.name for d in out1.iterdir() if d.is_dir())
        dirs2 = sorted(d.name for d in out2.iterdir() if d.is_dir())
        assert dirs1 != dirs2

    def test_only_new_skips(self, tmp_path: Path) -> None:
        shard = _setup_shard_dir(tmp_path / "input", num_buckets=1, images_per_bucket=3)
        output_dir = tmp_path / "output"

        r1 = convert_directory(shard, output_dir)
        assert r1.images_processed == 3

        r2 = convert_directory(shard, output_dir, only_new=True)
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
