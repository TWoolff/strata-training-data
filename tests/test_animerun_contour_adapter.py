"""Tests for the AnimeRun contour adapter.

These tests exercise the pure-Python adapter logic without requiring
the actual AnimeRun dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from ingest.animerun_contour_adapter import (
    AdapterResult,
    ContourPair,
    _build_metadata,
    _resize_image,
    convert_directory,
    convert_pair,
    convert_scene,
    discover_pairs,
    discover_scenes,
    generate_contour_mask,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_test_image(
    size: tuple[int, int] = (256, 256),
    color: tuple[int, int, int] = (128, 128, 128),
) -> Image.Image:
    """Create a simple RGB test image with a solid color."""
    arr = np.full((*size, 3), color, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _create_image_with_lines(
    size: tuple[int, int] = (256, 256),
    bg_color: tuple[int, int, int] = (200, 200, 200),
    line_color: tuple[int, int, int] = (20, 20, 20),
) -> Image.Image:
    """Create an image with dark horizontal lines (simulating contours)."""
    arr = np.full((*size, 3), bg_color, dtype=np.uint8)
    # Draw horizontal lines every 32 pixels.
    for y in range(0, size[0], 32):
        arr[y : y + 2, :, :] = line_color
    return Image.fromarray(arr, mode="RGB")


def _setup_scene_dir(
    tmp_path: Path,
    split: str = "train",
    scene_name: str = "scene_001",
    *,
    num_frames: int = 3,
    with_contours: bool = True,
    with_anime: bool = True,
    mismatched: bool = False,
) -> Path:
    """Create a fake AnimeRun scene directory.

    Args:
        tmp_path: Pytest tmp_path fixture.
        split: Split name (train/test).
        scene_name: Scene directory name.
        num_frames: Number of frames to create.
        with_contours: Create contour/ directory.
        with_anime: Create anime/ directory.
        mismatched: If True, anime has one extra frame not in contour.

    Returns:
        Path to the root AnimeRun directory.
    """
    root = tmp_path / "animerun"
    scene_dir = root / split / scene_name

    if with_contours:
        contour_dir = scene_dir / "contour"
        contour_dir.mkdir(parents=True)
        for i in range(num_frames):
            img = _create_image_with_lines()
            img.save(contour_dir / f"frame_{i:04d}.png")

    if with_anime:
        anime_dir = scene_dir / "anime"
        anime_dir.mkdir(parents=True)
        extra = 1 if mismatched else 0
        for i in range(num_frames + extra):
            img = _create_test_image()
            img.save(anime_dir / f"frame_{i:04d}.png")

    return root


# ---------------------------------------------------------------------------
# _resize_image
# ---------------------------------------------------------------------------


class TestResizeImage:
    """Test image resizing."""

    def test_resize_from_smaller(self) -> None:
        img = _create_test_image((128, 128))
        resized = _resize_image(img, 512)
        assert resized.size == (512, 512)
        assert resized.mode == "RGB"

    def test_resize_from_larger(self) -> None:
        img = _create_test_image((1024, 1024))
        resized = _resize_image(img, 512)
        assert resized.size == (512, 512)

    def test_already_correct_size(self) -> None:
        img = _create_test_image((512, 512))
        resized = _resize_image(img, 512)
        assert resized.size == (512, 512)

    def test_converts_rgba_to_rgb(self) -> None:
        arr = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGBA")
        resized = _resize_image(img, 64)
        assert resized.mode == "RGB"


# ---------------------------------------------------------------------------
# generate_contour_mask
# ---------------------------------------------------------------------------


class TestGenerateContourMask:
    """Test contour mask generation."""

    def test_identical_images_no_contours(self) -> None:
        img = _create_test_image((64, 64), (100, 100, 100))
        mask = generate_contour_mask(img, img, threshold=30)
        assert mask.shape == (64, 64)
        assert np.all(mask == 0)

    def test_different_images_produce_contours(self) -> None:
        anime = _create_test_image((64, 64), (200, 200, 200))
        contour = _create_image_with_lines((64, 64))
        mask = generate_contour_mask(contour, anime, threshold=30)
        assert mask.shape == (64, 64)
        assert np.any(mask == 255)  # Should detect contour lines.

    def test_mask_is_binary(self) -> None:
        anime = _create_test_image((64, 64), (200, 200, 200))
        contour = _create_image_with_lines((64, 64))
        mask = generate_contour_mask(contour, anime, threshold=30)
        unique = set(np.unique(mask))
        assert unique.issubset({0, 255})

    def test_threshold_sensitivity(self) -> None:
        # Small difference below threshold should produce no contours.
        img_a = _create_test_image((32, 32), (100, 100, 100))
        img_b = _create_test_image((32, 32), (110, 110, 110))
        mask_high = generate_contour_mask(img_a, img_b, threshold=50)
        assert np.all(mask_high == 0)

        # Same difference above a lower threshold should produce contours.
        mask_low = generate_contour_mask(img_a, img_b, threshold=5)
        assert np.all(mask_low == 255)


# ---------------------------------------------------------------------------
# _build_metadata
# ---------------------------------------------------------------------------


class TestBuildMetadata:
    """Test metadata generation."""

    def test_metadata_fields(self) -> None:
        pair = ContourPair(
            frame_id="frame_0001",
            scene_id="scene_001",
            split="train",
            contour_path=Path("/fake/contour.png"),
            anime_path=Path("/fake/anime.png"),
        )
        meta = _build_metadata(pair, 512)
        assert meta["source"] == "animerun"
        assert meta["scene_id"] == "scene_001"
        assert meta["frame_id"] == "frame_0001"
        assert meta["split"] == "train"
        assert meta["data_type"] == "contour_pair"
        assert meta["has_contour_mask"] is True
        assert meta["has_joints"] is False


# ---------------------------------------------------------------------------
# discover_scenes
# ---------------------------------------------------------------------------


class TestDiscoverScenes:
    """Test scene discovery."""

    def test_discovers_train_and_test(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, "train", "scene_A")
        # Add a test scene.
        test_scene = root / "test" / "scene_B"
        (test_scene / "contour").mkdir(parents=True)
        (test_scene / "anime").mkdir(parents=True)

        scenes = discover_scenes(root)
        assert len(scenes) == 2
        splits = [s[0] for s in scenes]
        assert "train" in splits
        assert "test" in splits

    def test_skips_incomplete_scenes(self, tmp_path: Path) -> None:
        root = tmp_path / "animerun"
        # Scene with only contour/ (no anime/).
        (root / "train" / "bad_scene" / "contour").mkdir(parents=True)
        scenes = discover_scenes(root)
        assert len(scenes) == 0

    def test_skips_hidden_dirs(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, "train", "scene_A")
        hidden = root / "train" / ".hidden"
        (hidden / "contour").mkdir(parents=True)
        (hidden / "anime").mkdir(parents=True)

        scenes = discover_scenes(root)
        assert len(scenes) == 1
        assert scenes[0][1] == "scene_A"

    def test_empty_directory(self, tmp_path: Path) -> None:
        root = tmp_path / "animerun"
        root.mkdir()
        scenes = discover_scenes(root)
        assert scenes == []


# ---------------------------------------------------------------------------
# discover_pairs
# ---------------------------------------------------------------------------


class TestDiscoverPairs:
    """Test frame pair discovery."""

    def test_discovers_matched_pairs(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, num_frames=5)
        scene_dir = root / "train" / "scene_001"

        pairs = discover_pairs(scene_dir, "train", "scene_001")
        assert len(pairs) == 5
        assert all(p.scene_id == "scene_001" for p in pairs)
        assert all(p.split == "train" for p in pairs)

    def test_mismatched_frames_only_include_common(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, num_frames=3, mismatched=True)
        scene_dir = root / "train" / "scene_001"

        pairs = discover_pairs(scene_dir, "train", "scene_001")
        assert len(pairs) == 3  # Only the 3 common frames.

    def test_empty_scene(self, tmp_path: Path) -> None:
        scene_dir = tmp_path / "scene"
        (scene_dir / "contour").mkdir(parents=True)
        (scene_dir / "anime").mkdir(parents=True)

        pairs = discover_pairs(scene_dir, "train", "empty_scene")
        assert pairs == []


# ---------------------------------------------------------------------------
# convert_pair
# ---------------------------------------------------------------------------


class TestConvertPair:
    """Test single pair conversion."""

    def test_convert_creates_output(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, num_frames=1)
        scene_dir = root / "train" / "scene_001"
        pairs = discover_pairs(scene_dir, "train", "scene_001")
        output_dir = tmp_path / "output"

        saved = convert_pair(pairs[0], output_dir, resolution=64)
        assert saved is True

        example_dir = output_dir / "animerun_scene_001_frame_0000"
        assert example_dir.is_dir()
        assert (example_dir / "with_contours.png").is_file()
        assert (example_dir / "without_contours.png").is_file()
        assert (example_dir / "contour_mask.png").is_file()
        assert (example_dir / "metadata.json").is_file()

        # Check image size.
        img = Image.open(example_dir / "with_contours.png")
        assert img.size == (64, 64)

        # Check metadata.
        meta = json.loads((example_dir / "metadata.json").read_text(encoding="utf-8"))
        assert meta["source"] == "animerun"
        assert meta["resolution"] == 64

    def test_only_new_skips_existing(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, num_frames=1)
        scene_dir = root / "train" / "scene_001"
        pairs = discover_pairs(scene_dir, "train", "scene_001")
        output_dir = tmp_path / "output"

        assert convert_pair(pairs[0], output_dir, resolution=64)
        assert not convert_pair(pairs[0], output_dir, resolution=64, only_new=True)

    def test_invalid_image_path(self, tmp_path: Path) -> None:
        pair = ContourPair(
            frame_id="bad",
            scene_id="scene",
            split="train",
            contour_path=tmp_path / "nonexistent.png",
            anime_path=tmp_path / "nonexistent2.png",
        )
        assert not convert_pair(pair, tmp_path / "output")


# ---------------------------------------------------------------------------
# convert_scene
# ---------------------------------------------------------------------------


class TestConvertScene:
    """Test scene-level conversion."""

    def test_converts_all_frames(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, num_frames=4)
        scene_dir = root / "train" / "scene_001"
        output_dir = tmp_path / "output"

        result = convert_scene(scene_dir, output_dir, "train", "scene_001", resolution=64)
        assert isinstance(result, AdapterResult)
        assert result.scene_id == "scene_001"
        assert result.frames_saved == 4

    def test_max_frames(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, num_frames=10)
        scene_dir = root / "train" / "scene_001"
        output_dir = tmp_path / "output"

        result = convert_scene(
            scene_dir,
            output_dir,
            "train",
            "scene_001",
            resolution=64,
            max_frames=3,
        )
        assert result.frames_saved == 3


# ---------------------------------------------------------------------------
# convert_directory
# ---------------------------------------------------------------------------


class TestConvertDirectory:
    """Test batch directory conversion."""

    def test_converts_multiple_scenes(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, "train", "scene_A", num_frames=2)
        # Add second scene.
        scene_b = root / "train" / "scene_B"
        (scene_b / "contour").mkdir(parents=True)
        (scene_b / "anime").mkdir(parents=True)
        _create_image_with_lines().save(scene_b / "contour" / "f001.png")
        _create_test_image().save(scene_b / "anime" / "f001.png")

        output_dir = tmp_path / "output"
        results = convert_directory(root, output_dir, resolution=64)
        assert len(results) == 2
        total = sum(r.frames_saved for r in results)
        assert total == 3  # 2 + 1

    def test_max_scenes(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, "train", "scene_A", num_frames=1)
        scene_b = root / "train" / "scene_B"
        (scene_b / "contour").mkdir(parents=True)
        (scene_b / "anime").mkdir(parents=True)
        _create_image_with_lines().save(scene_b / "contour" / "f001.png")
        _create_test_image().save(scene_b / "anime" / "f001.png")

        output_dir = tmp_path / "output"
        results = convert_directory(root, output_dir, resolution=64, max_scenes=1)
        assert len(results) == 1

    def test_empty_directory(self, tmp_path: Path) -> None:
        root = tmp_path / "animerun"
        root.mkdir()
        results = convert_directory(root, tmp_path / "output")
        assert results == []

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        results = convert_directory(tmp_path / "nonexistent", tmp_path / "output")
        assert results == []
