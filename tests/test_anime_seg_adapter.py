"""Tests for the anime-segmentation ingest adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ingest.anime_seg_adapter import (
    AdapterResult,
    _extract_mask,
    _is_bg_path,
    _resize_to_strata,
    convert_directory,
    convert_image,
    discover_images,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rgba_image(w: int = 200, h: int = 300, alpha: int = 255) -> Image.Image:
    """Create a simple RGBA test image with a uniform alpha channel."""
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:, :, 0] = 180  # R
    arr[:, :, 1] = 100  # G
    arr[:, :, 2] = 50  # B
    arr[:, :, 3] = alpha
    return Image.fromarray(arr, "RGBA")


def _make_character_image(w: int = 200, h: int = 300) -> Image.Image:
    """Create an RGBA image with a character-shaped alpha mask.

    Inner rectangle is opaque (alpha=255), outer region is transparent (alpha=0).
    """
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:, :, :3] = 128
    # Character in center 50% of the image
    y1, y2 = h // 4, 3 * h // 4
    x1, x2 = w // 4, 3 * w // 4
    arr[y1:y2, x1:x2, 3] = 255
    return Image.fromarray(arr, "RGBA")


# ---------------------------------------------------------------------------
# Discovery tests
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_finds_png_files(self, tmp_path: Path) -> None:
        fg_dir = tmp_path / "fg"
        fg_dir.mkdir()
        (fg_dir / "001.png").write_bytes(b"fake")
        (fg_dir / "002.png").write_bytes(b"fake")

        paths = discover_images(tmp_path)
        assert len(paths) == 2

    def test_skips_bg_directories(self, tmp_path: Path) -> None:
        fg_dir = tmp_path / "fg"
        fg_dir.mkdir()
        (fg_dir / "001.png").write_bytes(b"fake")

        bg_dir = tmp_path / "bg"
        bg_dir.mkdir()
        (bg_dir / "001.png").write_bytes(b"fake")

        paths = discover_images(tmp_path)
        assert len(paths) == 1
        assert "fg" in str(paths[0])

    def test_skips_bg_numbered_dirs(self, tmp_path: Path) -> None:
        fg_dir = tmp_path / "fg"
        fg_dir.mkdir()
        (fg_dir / "001.png").write_bytes(b"fake")

        bg_dir = tmp_path / "bg 3"
        bg_dir.mkdir()
        (bg_dir / "001.png").write_bytes(b"fake")

        paths = discover_images(tmp_path)
        assert len(paths) == 1

    def test_ignores_non_png(self, tmp_path: Path) -> None:
        (tmp_path / "image.jpg").write_bytes(b"fake")
        (tmp_path / "image.txt").write_bytes(b"fake")

        paths = discover_images(tmp_path)
        assert len(paths) == 0

    def test_empty_dir(self, tmp_path: Path) -> None:
        paths = discover_images(tmp_path)
        assert paths == []


class TestIsBgPath:
    def test_bg_root(self) -> None:
        assert _is_bg_path(Path("data/bg/001.png"))

    def test_bg_numbered(self) -> None:
        assert _is_bg_path(Path("data/bg 2/001.png"))

    def test_fg_path(self) -> None:
        assert not _is_bg_path(Path("data/fg/001.png"))

    def test_fg_numbered(self) -> None:
        assert not _is_bg_path(Path("data/fg 3/001.png"))


# ---------------------------------------------------------------------------
# Mask extraction tests
# ---------------------------------------------------------------------------


class TestExtractMask:
    def test_fully_opaque_with_padding(self) -> None:
        # Non-square image → padding adds transparent pixels
        img = _make_rgba_image(100, 200, alpha=255)
        resized = _resize_to_strata(img, 512)
        mask = _extract_mask(resized)
        arr = np.array(mask)
        assert arr.max() == 255  # Character pixels
        assert arr.min() == 0  # Padding pixels

    def test_fully_opaque_square(self) -> None:
        # Square image fills entire canvas — all foreground
        img = _make_rgba_image(100, 100, alpha=255)
        resized = _resize_to_strata(img, 512)
        mask = _extract_mask(resized)
        arr = np.array(mask)
        assert (arr == 255).all()

    def test_fully_transparent(self) -> None:
        img = _make_rgba_image(100, 100, alpha=0)
        resized = _resize_to_strata(img, 512)
        mask = _extract_mask(resized)
        arr = np.array(mask)
        assert arr.max() == 0  # All background

    def test_binary_values_only(self) -> None:
        img = _make_character_image()
        resized = _resize_to_strata(img, 512)
        mask = _extract_mask(resized)
        unique = set(np.unique(np.array(mask)))
        assert unique <= {0, 255}

    def test_mask_is_grayscale(self) -> None:
        img = _make_character_image()
        resized = _resize_to_strata(img, 512)
        mask = _extract_mask(resized)
        assert mask.mode == "L"


# ---------------------------------------------------------------------------
# Resize tests
# ---------------------------------------------------------------------------


class TestResize:
    def test_output_size(self) -> None:
        img = _make_rgba_image(200, 400)
        resized = _resize_to_strata(img, 512)
        assert resized.size == (512, 512)

    def test_preserves_rgba_mode(self) -> None:
        img = _make_rgba_image(200, 400)
        resized = _resize_to_strata(img, 512)
        assert resized.mode == "RGBA"

    def test_already_correct_size(self) -> None:
        img = _make_rgba_image(512, 512)
        resized = _resize_to_strata(img, 512)
        assert resized.size == (512, 512)


# ---------------------------------------------------------------------------
# Convert image tests
# ---------------------------------------------------------------------------


class TestConvertImage:
    def test_saves_all_files(self, tmp_path: Path) -> None:
        img = _make_character_image()
        src = tmp_path / "src" / "001.png"
        src.parent.mkdir()
        img.save(src)

        out = tmp_path / "output"
        result = convert_image(
            src, out, image_id="animeseg_v1_000000", resolution=512
        )

        assert result is True
        example_dir = out / "animeseg_v1_000000"
        assert (example_dir / "image.png").exists()
        assert (example_dir / "segmentation.png").exists()
        assert (example_dir / "metadata.json").exists()

    def test_metadata_schema(self, tmp_path: Path) -> None:
        img = _make_character_image()
        src = tmp_path / "src" / "001.png"
        src.parent.mkdir()
        img.save(src)

        out = tmp_path / "output"
        convert_image(src, out, image_id="animeseg_v1_000000", variant="v1")

        meta = json.loads(
            (out / "animeseg_v1_000000" / "metadata.json").read_text()
        )
        assert meta["source"] == "animeseg"
        assert meta["source_variant"] == "v1"
        assert meta["has_fg_mask"] is True
        assert meta["has_segmentation_mask"] is False
        assert "strata_segmentation" in meta["missing_annotations"]

    def test_segmentation_is_binary(self, tmp_path: Path) -> None:
        img = _make_character_image()
        src = tmp_path / "src" / "001.png"
        src.parent.mkdir()
        img.save(src)

        out = tmp_path / "output"
        convert_image(src, out, image_id="animeseg_v1_000000")

        mask = Image.open(out / "animeseg_v1_000000" / "segmentation.png")
        unique = set(np.unique(np.array(mask)))
        assert unique <= {0, 255}

    def test_only_new_skips_existing(self, tmp_path: Path) -> None:
        img = _make_character_image()
        src = tmp_path / "src" / "001.png"
        src.parent.mkdir()
        img.save(src)

        out = tmp_path / "output"
        convert_image(src, out, image_id="animeseg_v1_000000")
        result = convert_image(
            src, out, image_id="animeseg_v1_000000", only_new=True
        )
        assert result is False


# ---------------------------------------------------------------------------
# Convert directory tests
# ---------------------------------------------------------------------------


class TestConvertDirectory:
    def test_processes_all_fg_images(self, tmp_path: Path) -> None:
        fg_dir = tmp_path / "fg"
        fg_dir.mkdir()
        for i in range(5):
            _make_character_image().save(fg_dir / f"{i:06d}.png")

        out = tmp_path / "output"
        result = convert_directory(tmp_path, out, variant="v1")

        assert isinstance(result, AdapterResult)
        assert result.images_processed == 5

    def test_max_images_limit(self, tmp_path: Path) -> None:
        fg_dir = tmp_path / "fg"
        fg_dir.mkdir()
        for i in range(10):
            _make_character_image().save(fg_dir / f"{i:06d}.png")

        out = tmp_path / "output"
        result = convert_directory(tmp_path, out, variant="v1", max_images=3)

        assert result.images_processed == 3

    def test_skips_bg_images(self, tmp_path: Path) -> None:
        fg_dir = tmp_path / "fg"
        fg_dir.mkdir()
        _make_character_image().save(fg_dir / "000000.png")

        bg_dir = tmp_path / "bg"
        bg_dir.mkdir()
        _make_character_image().save(bg_dir / "000000.png")

        out = tmp_path / "output"
        result = convert_directory(tmp_path, out, variant="v1")

        assert result.images_processed == 1
