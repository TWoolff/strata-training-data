"""Tests for the NOVA-Human adapter.

These tests exercise the pure-Python adapter logic without requiring
Blender or the actual NOVA-Human dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from ingest.nova_human_adapter import (
    NovaHumanCharacter,
    _build_metadata,
    _convert_mask_to_binary,
    _resize_to_strata,
    convert_character,
    convert_directory,
    parse_character,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_test_image(size: tuple[int, int] = (256, 256)) -> Image.Image:
    """Create a simple RGBA test image."""
    arr = np.random.randint(0, 255, (*size, 4), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def _create_test_mask(size: tuple[int, int] = (256, 256)) -> Image.Image:
    """Create a simple binary mask image (L mode)."""
    arr = np.zeros(size, dtype=np.uint8)
    # Set center region as foreground
    h, w = size
    arr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 255
    return Image.fromarray(arr, mode="L")


def _setup_character_dir(
    tmp_path: Path,
    name: str = "human_test001",
    *,
    with_ortho: bool = True,
    with_rgb: bool = False,
    with_meta: bool = True,
    with_masks: bool = True,
) -> Path:
    """Create a fake NOVA-Human character directory structure.

    Args:
        tmp_path: Pytest tmp_path fixture.
        name: Character directory name.
        with_ortho: Create ortho/ subdirectory with images.
        with_rgb: Create rgb/ subdirectory with images.
        with_meta: Create metadata JSON file.
        with_masks: Create mask directories and images.

    Returns:
        Path to the created character directory.
    """
    char_dir = tmp_path / name
    char_dir.mkdir(parents=True)

    if with_ortho:
        ortho_dir = char_dir / "ortho"
        ortho_dir.mkdir()
        for view_name in ("front", "back"):
            img = _create_test_image()
            img.save(ortho_dir / f"{view_name}.png")

        if with_masks:
            mask_dir = char_dir / "ortho_mask"
            mask_dir.mkdir()
            for view_name in ("front", "back"):
                mask = _create_test_mask()
                mask.save(mask_dir / f"{view_name}.png")

    if with_rgb:
        rgb_dir = char_dir / "rgb"
        rgb_dir.mkdir()
        for i in range(4):  # 4 test views instead of 16
            img = _create_test_image()
            img.save(rgb_dir / f"view_{i:03d}.png")

        if with_masks:
            rgb_mask_dir = char_dir / "rgb_mask"
            rgb_mask_dir.mkdir()
            for i in range(4):
                mask = _create_test_mask()
                mask.save(rgb_mask_dir / f"view_{i:03d}.png")

    if with_meta:
        meta = {"model_name": name, "source": "vroidhub"}
        meta_path = char_dir / f"{name}_meta.json"
        meta_path.write_text(
            json.dumps(meta, indent=2) + "\n",
            encoding="utf-8",
        )

    return char_dir


# ---------------------------------------------------------------------------
# _resize_to_strata
# ---------------------------------------------------------------------------


class TestResizeToStrata:
    """Test image resizing to Strata resolution."""

    def test_resize_from_smaller(self) -> None:
        img = _create_test_image((128, 128))
        resized = _resize_to_strata(img, 512)
        assert resized.size == (512, 512)
        assert resized.mode == "RGBA"

    def test_resize_from_larger(self) -> None:
        img = _create_test_image((1024, 1024))
        resized = _resize_to_strata(img, 512)
        assert resized.size == (512, 512)

    def test_already_correct_size(self) -> None:
        img = _create_test_image((512, 512))
        resized = _resize_to_strata(img, 512)
        assert resized.size == (512, 512)

    def test_converts_rgb_to_rgba(self) -> None:
        arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        resized = _resize_to_strata(img, 512)
        assert resized.mode == "RGBA"


# ---------------------------------------------------------------------------
# _convert_mask_to_binary
# ---------------------------------------------------------------------------


class TestConvertMaskToBinary:
    """Test mask conversion to binary array."""

    def test_binary_values(self) -> None:
        mask = _create_test_mask((256, 256))
        arr = _convert_mask_to_binary(mask, 512)
        assert arr.shape == (512, 512)
        assert arr.dtype == np.uint8
        assert set(np.unique(arr)).issubset({0, 1})

    def test_foreground_preserved(self) -> None:
        # All-white mask should be all foreground
        white = Image.new("L", (64, 64), 255)
        arr = _convert_mask_to_binary(white, 64)
        assert np.all(arr == 1)

    def test_background_preserved(self) -> None:
        # All-black mask should be all background
        black = Image.new("L", (64, 64), 0)
        arr = _convert_mask_to_binary(black, 64)
        assert np.all(arr == 0)

    def test_converts_rgb_mask(self) -> None:
        # RGB mask should be converted to L mode
        arr = np.full((64, 64, 3), 200, dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        result = _convert_mask_to_binary(img, 64)
        assert result.shape == (64, 64)
        assert np.all(result == 1)  # 200 > 127


# ---------------------------------------------------------------------------
# _build_metadata
# ---------------------------------------------------------------------------


class TestBuildMetadata:
    """Test metadata generation."""

    def test_ortho_metadata(self) -> None:
        character = NovaHumanCharacter(
            character_id="human_test",
            source_dir=Path("/fake"),
            meta={"model_name": "test"},
        )
        meta = _build_metadata(
            "nova_human_human_test", character, "ortho_front", 0, is_ortho=True,
        )
        assert meta["source"] == "nova_human"
        assert meta["camera_type"] == "orthographic"
        assert meta["has_segmentation_mask"] is False
        assert meta["has_fg_mask"] is True
        assert meta["has_joints"] is False
        assert "strata_segmentation" in meta["missing_annotations"]
        assert "joints" in meta["missing_annotations"]

    def test_perspective_metadata(self) -> None:
        character = NovaHumanCharacter(
            character_id="human_test",
            source_dir=Path("/fake"),
        )
        meta = _build_metadata(
            "nova_human_human_test", character, "rgb_00", 0, is_ortho=False,
        )
        assert meta["camera_type"] == "perspective"


# ---------------------------------------------------------------------------
# parse_character
# ---------------------------------------------------------------------------


class TestParseCharacter:
    """Test character directory parsing."""

    def test_parse_ortho_only(self, tmp_path: Path) -> None:
        char_dir = _setup_character_dir(tmp_path, with_ortho=True, with_rgb=False)
        result = parse_character(char_dir)
        assert result is not None
        assert result.character_id == "human_test001"
        assert len(result.ortho_images) == 2
        assert "front" in result.ortho_images
        assert "back" in result.ortho_images
        assert len(result.ortho_masks) == 2

    def test_parse_with_rgb(self, tmp_path: Path) -> None:
        char_dir = _setup_character_dir(tmp_path, with_ortho=True, with_rgb=True)
        result = parse_character(char_dir)
        assert result is not None
        assert len(result.rgb_images) == 4
        assert len(result.rgb_masks) == 4

    def test_parse_metadata(self, tmp_path: Path) -> None:
        char_dir = _setup_character_dir(tmp_path, with_meta=True)
        result = parse_character(char_dir)
        assert result is not None
        assert result.meta["model_name"] == "human_test001"

    def test_parse_no_images(self, tmp_path: Path) -> None:
        char_dir = _setup_character_dir(
            tmp_path, with_ortho=False, with_rgb=False,
        )
        result = parse_character(char_dir)
        assert result is None

    def test_parse_nonexistent_dir(self, tmp_path: Path) -> None:
        result = parse_character(tmp_path / "does_not_exist")
        assert result is None

    def test_parse_without_metadata(self, tmp_path: Path) -> None:
        char_dir = _setup_character_dir(tmp_path, with_meta=False)
        result = parse_character(char_dir)
        assert result is not None
        assert result.meta == {}

    def test_parse_without_masks(self, tmp_path: Path) -> None:
        char_dir = _setup_character_dir(tmp_path, with_masks=False)
        result = parse_character(char_dir)
        assert result is not None
        assert len(result.ortho_masks) == 0


# ---------------------------------------------------------------------------
# convert_character
# ---------------------------------------------------------------------------


class TestConvertCharacter:
    """Test single-character conversion."""

    def test_convert_ortho(self, tmp_path: Path) -> None:
        char_dir = _setup_character_dir(tmp_path / "input")
        output_dir = tmp_path / "output"

        result = convert_character(char_dir, output_dir)
        assert result is not None
        assert result.char_id == "nova_human_human_test001"
        assert result.views_saved == 2

        # Check output structure
        for view in ("ortho_back", "ortho_front"):
            example_dir = output_dir / f"nova_human_human_test001_{view}"
            assert example_dir.is_dir()
            assert (example_dir / "image.png").is_file()
            assert (example_dir / "segmentation.png").is_file()
            assert (example_dir / "metadata.json").is_file()

            # Check image size
            img = Image.open(example_dir / "image.png")
            assert img.size == (512, 512)

            # Check metadata
            meta = json.loads(
                (example_dir / "metadata.json").read_text(encoding="utf-8")
            )
            assert meta["source"] == "nova_human"

    def test_convert_with_rgb(self, tmp_path: Path) -> None:
        char_dir = _setup_character_dir(
            tmp_path / "input", with_ortho=True, with_rgb=True,
        )
        output_dir = tmp_path / "output"

        result = convert_character(char_dir, output_dir, include_rgb=True)
        assert result is not None
        assert result.views_saved == 6  # 2 ortho + 4 rgb

    def test_convert_only_new_skips_existing(self, tmp_path: Path) -> None:
        char_dir = _setup_character_dir(tmp_path / "input")
        output_dir = tmp_path / "output"

        # First run
        result1 = convert_character(char_dir, output_dir)
        assert result1 is not None
        assert result1.views_saved == 2

        # Second run with only_new=True
        result2 = convert_character(char_dir, output_dir, only_new=True)
        assert result2 is not None
        assert result2.views_saved == 0

    def test_convert_invalid_dir(self, tmp_path: Path) -> None:
        result = convert_character(tmp_path / "nonexistent", tmp_path / "out")
        assert result is None

    def test_convert_custom_resolution(self, tmp_path: Path) -> None:
        char_dir = _setup_character_dir(tmp_path / "input")
        output_dir = tmp_path / "output"

        result = convert_character(char_dir, output_dir, resolution=256)
        assert result is not None

        example_dir = output_dir / "nova_human_human_test001_ortho_back"
        img = Image.open(example_dir / "image.png")
        assert img.size == (256, 256)


# ---------------------------------------------------------------------------
# convert_directory
# ---------------------------------------------------------------------------


class TestConvertDirectory:
    """Test batch directory conversion."""

    def test_convert_multiple_characters(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "nova_human"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # Create 3 character directories
        for i in range(3):
            _setup_character_dir(input_dir, name=f"human_char{i:03d}")

        results = convert_directory(input_dir, output_dir)
        assert len(results) == 3
        total_views = sum(r.views_saved for r in results)
        assert total_views == 6  # 2 ortho views each

    def test_convert_max_characters(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "nova_human"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        for i in range(5):
            _setup_character_dir(input_dir, name=f"human_char{i:03d}")

        results = convert_directory(input_dir, output_dir, max_characters=2)
        assert len(results) == 2

    def test_convert_empty_directory(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "nova_human"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        results = convert_directory(input_dir, output_dir)
        assert results == []

    def test_convert_nonexistent_directory(self, tmp_path: Path) -> None:
        results = convert_directory(
            tmp_path / "nonexistent",
            tmp_path / "output",
        )
        assert results == []

    def test_skips_hidden_directories(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "nova_human"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # Regular character
        _setup_character_dir(input_dir, name="human_char001")
        # Hidden directory (e.g. .repo from git clone)
        hidden = input_dir / ".repo"
        hidden.mkdir()
        (hidden / "ortho").mkdir()

        results = convert_directory(input_dir, output_dir)
        assert len(results) == 1
        assert results[0].char_id == "nova_human_human_char001"
