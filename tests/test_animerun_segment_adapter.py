"""Tests for the AnimeRun instance segmentation adapter.

These tests exercise the pure-Python adapter logic without requiring
the actual AnimeRun dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from ingest.animerun_segment_adapter import (
    AdapterResult,
    SegmentFrame,
    _build_metadata,
    _resize_image,
    _resize_mask,
    convert_directory,
    convert_frame,
    convert_scene,
    discover_frames,
    discover_scenes,
    generate_overlay,
    load_segment_map,
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


def _create_segment_map(
    size: tuple[int, int] = (256, 256),
    num_instances: int = 3,
) -> np.ndarray:
    """Create a fake instance segmentation map with horizontal bands."""
    arr = np.zeros(size, dtype=np.uint8)
    band_height = size[0] // (num_instances + 1)  # +1 for background
    for i in range(1, num_instances + 1):
        y_start = i * band_height
        y_end = (i + 1) * band_height
        arr[y_start:y_end, :] = i
    return arr


def _setup_scene_dir(
    tmp_path: Path,
    split: str = "train",
    scene_name: str = "scene_001",
    *,
    num_frames: int = 3,
    with_segments: bool = True,
    with_anime: bool = True,
    use_npy: bool = False,
    mismatched: bool = False,
) -> Path:
    """Create a fake AnimeRun v2 scene directory with Segment data.

    Args:
        tmp_path: Pytest tmp_path fixture.
        split: Split name (train/test).
        scene_name: Scene directory name.
        num_frames: Number of frames to create.
        with_segments: Create Segment directory.
        with_anime: Create Frame_Anime directory.
        use_npy: Use .npy format for segment maps instead of .png.
        mismatched: If True, anime has one extra frame not in segments.

    Returns:
        Path to the root AnimeRun directory.
    """
    root = tmp_path / "animerun"
    split_dir = root / split

    if with_segments:
        segment_dir = split_dir / "Segment" / scene_name
        segment_dir.mkdir(parents=True)
        for i in range(num_frames):
            mask = _create_segment_map()
            if use_npy:
                np.save(segment_dir / f"frame_{i:04d}.npy", mask)
            else:
                img = Image.fromarray(mask, mode="L")
                img.save(segment_dir / f"frame_{i:04d}.png")

    if with_anime:
        anime_dir = split_dir / "Frame_Anime" / scene_name / "original"
        anime_dir.mkdir(parents=True)
        extra = 1 if mismatched else 0
        for i in range(num_frames + extra):
            img = _create_test_image()
            img.save(anime_dir / f"frame_{i:04d}.png")

    return root


# ---------------------------------------------------------------------------
# load_segment_map
# ---------------------------------------------------------------------------


class TestLoadSegmentMap:
    """Test segment map loading."""

    def test_load_png(self, tmp_path: Path) -> None:
        mask = _create_segment_map((64, 64), num_instances=2)
        path = tmp_path / "seg.png"
        Image.fromarray(mask, mode="L").save(path)

        loaded = load_segment_map(path)
        assert loaded.shape == (64, 64)
        assert loaded.dtype == np.uint8
        np.testing.assert_array_equal(loaded, mask)

    def test_load_npy(self, tmp_path: Path) -> None:
        mask = _create_segment_map((64, 64), num_instances=2)
        path = tmp_path / "seg.npy"
        np.save(path, mask)

        loaded = load_segment_map(path)
        assert loaded.shape == (64, 64)
        assert loaded.dtype == np.uint8
        np.testing.assert_array_equal(loaded, mask)

    def test_load_rgb_png_takes_first_channel(self, tmp_path: Path) -> None:
        mask = _create_segment_map((64, 64), num_instances=2)
        rgb = np.stack([mask, mask, mask], axis=-1)
        path = tmp_path / "seg_rgb.png"
        Image.fromarray(rgb, mode="RGB").save(path)

        loaded = load_segment_map(path)
        assert loaded.shape == (64, 64)
        assert loaded.dtype == np.uint8

    def test_load_npy_3d_takes_first_channel(self, tmp_path: Path) -> None:
        mask = _create_segment_map((64, 64), num_instances=2)
        arr_3d = np.stack([mask, mask], axis=-1)
        path = tmp_path / "seg_3d.npy"
        np.save(path, arr_3d)

        loaded = load_segment_map(path)
        assert loaded.ndim == 2


# ---------------------------------------------------------------------------
# _resize_image / _resize_mask
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


class TestResizeMask:
    """Test mask resizing with nearest-neighbor interpolation."""

    def test_resize_preserves_ids(self) -> None:
        mask = _create_segment_map((128, 128), num_instances=3)
        resized = _resize_mask(mask, 64)
        assert resized.shape == (64, 64)
        # All original instance IDs should be preserved.
        original_ids = set(np.unique(mask))
        resized_ids = set(np.unique(resized))
        assert resized_ids == original_ids

    def test_already_correct_size(self) -> None:
        mask = _create_segment_map((512, 512))
        resized = _resize_mask(mask, 512)
        assert resized.shape == (512, 512)
        np.testing.assert_array_equal(resized, mask)


# ---------------------------------------------------------------------------
# generate_overlay
# ---------------------------------------------------------------------------


class TestGenerateOverlay:
    """Test overlay visualization generation."""

    def test_overlay_shape(self) -> None:
        img = _create_test_image((64, 64))
        mask = _create_segment_map((64, 64), num_instances=2)
        overlay = generate_overlay(img, mask)
        assert overlay.size == (64, 64)
        assert overlay.mode == "RGB"

    def test_overlay_differs_from_original(self) -> None:
        img = _create_test_image((64, 64), (128, 128, 128))
        mask = _create_segment_map((64, 64), num_instances=2)
        overlay = generate_overlay(img, mask)
        # Where mask != 0, overlay should differ from original.
        orig_arr = np.array(img)
        over_arr = np.array(overlay)
        assert not np.array_equal(orig_arr, over_arr)

    def test_overlay_background_unchanged(self) -> None:
        img = _create_test_image((64, 64), (128, 128, 128))
        mask = np.zeros((64, 64), dtype=np.uint8)  # All background.
        overlay = generate_overlay(img, mask)
        np.testing.assert_array_equal(np.array(img), np.array(overlay))


# ---------------------------------------------------------------------------
# _build_metadata
# ---------------------------------------------------------------------------


class TestBuildMetadata:
    """Test metadata generation."""

    def test_metadata_fields(self) -> None:
        frame = SegmentFrame(
            frame_id="frame_0001",
            scene_id="scene_001",
            split="train",
            segment_path=Path("/fake/seg.png"),
            anime_path=Path("/fake/anime.png"),
        )
        meta = _build_metadata(frame, 512, [0, 1, 2, 3])
        assert meta["source"] == "animerun"
        assert meta["scene_id"] == "scene_001"
        assert meta["frame_id"] == "frame_0001"
        assert meta["split"] == "train"
        assert meta["data_type"] == "instance_segmentation"
        assert meta["instance_ids"] == [0, 1, 2, 3]
        assert meta["num_instances"] == 3  # Excludes background (0).
        assert meta["has_instance_mask"] is True
        assert meta["has_segmentation_mask"] is False
        assert meta["has_joints"] is False


# ---------------------------------------------------------------------------
# discover_scenes
# ---------------------------------------------------------------------------


class TestDiscoverScenes:
    """Test scene discovery."""

    def test_discovers_train_and_test(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, "train", "scene_A")
        # Add a test scene.
        test_segment = root / "test" / "Segment" / "scene_B"
        test_segment.mkdir(parents=True)
        test_anime = root / "test" / "Frame_Anime" / "scene_B" / "original"
        test_anime.mkdir(parents=True)

        scenes = discover_scenes(root)
        assert len(scenes) == 2
        splits = [s[0] for s in scenes]
        assert "train" in splits
        assert "test" in splits

    def test_skips_scene_without_anime(self, tmp_path: Path) -> None:
        root = tmp_path / "animerun"
        (root / "train" / "Segment" / "bad_scene").mkdir(parents=True)
        scenes = discover_scenes(root)
        assert len(scenes) == 0

    def test_skips_hidden_dirs(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, "train", "scene_A")
        hidden = root / "train" / "Segment" / ".hidden"
        hidden.mkdir(parents=True)

        scenes = discover_scenes(root)
        assert len(scenes) == 1
        assert scenes[0][1] == "scene_A"

    def test_empty_directory(self, tmp_path: Path) -> None:
        root = tmp_path / "animerun"
        root.mkdir()
        scenes = discover_scenes(root)
        assert scenes == []


# ---------------------------------------------------------------------------
# discover_frames
# ---------------------------------------------------------------------------


class TestDiscoverFrames:
    """Test frame discovery."""

    def test_discovers_matched_frames(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, num_frames=5)
        split_dir = root / "train"

        frames = discover_frames(split_dir, "train", "scene_001")
        assert len(frames) == 5
        assert all(f.scene_id == "scene_001" for f in frames)
        assert all(f.split == "train" for f in frames)

    def test_mismatched_frames_only_include_common(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, num_frames=3, mismatched=True)
        split_dir = root / "train"

        frames = discover_frames(split_dir, "train", "scene_001")
        assert len(frames) == 3

    def test_npy_segment_maps(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, num_frames=2, use_npy=True)
        split_dir = root / "train"

        frames = discover_frames(split_dir, "train", "scene_001")
        assert len(frames) == 2
        assert all(f.segment_path.suffix == ".npy" for f in frames)

    def test_empty_scene(self, tmp_path: Path) -> None:
        split_dir = tmp_path / "split"
        (split_dir / "Segment" / "empty_scene").mkdir(parents=True)
        (split_dir / "Frame_Anime" / "empty_scene" / "original").mkdir(parents=True)

        frames = discover_frames(split_dir, "train", "empty_scene")
        assert frames == []


# ---------------------------------------------------------------------------
# convert_frame
# ---------------------------------------------------------------------------


class TestConvertFrame:
    """Test single frame conversion."""

    def test_convert_creates_output(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, num_frames=1)
        split_dir = root / "train"
        frames = discover_frames(split_dir, "train", "scene_001")
        output_dir = tmp_path / "output"

        saved = convert_frame(frames[0], output_dir, resolution=64)
        assert saved is True

        example_dir = output_dir / "animerun_scene_001_frame_0000"
        assert example_dir.is_dir()
        assert (example_dir / "image.png").is_file()
        assert (example_dir / "instance_mask.png").is_file()
        assert (example_dir / "instance_overlay.png").is_file()
        assert (example_dir / "metadata.json").is_file()

        # Check image size.
        img = Image.open(example_dir / "image.png")
        assert img.size == (64, 64)

        # Check mask size.
        mask = Image.open(example_dir / "instance_mask.png")
        assert mask.size == (64, 64)
        assert mask.mode == "L"

        # Check metadata.
        meta = json.loads((example_dir / "metadata.json").read_text(encoding="utf-8"))
        assert meta["source"] == "animerun"
        assert meta["data_type"] == "instance_segmentation"
        assert meta["resolution"] == 64
        assert meta["has_instance_mask"] is True

    def test_convert_npy_segment_map(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, num_frames=1, use_npy=True)
        split_dir = root / "train"
        frames = discover_frames(split_dir, "train", "scene_001")
        output_dir = tmp_path / "output"

        saved = convert_frame(frames[0], output_dir, resolution=64)
        assert saved is True
        assert (output_dir / "animerun_scene_001_frame_0000" / "instance_mask.png").is_file()

    def test_only_new_skips_existing(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, num_frames=1)
        split_dir = root / "train"
        frames = discover_frames(split_dir, "train", "scene_001")
        output_dir = tmp_path / "output"

        assert convert_frame(frames[0], output_dir, resolution=64)
        assert not convert_frame(frames[0], output_dir, resolution=64, only_new=True)

    def test_invalid_paths(self, tmp_path: Path) -> None:
        frame = SegmentFrame(
            frame_id="bad",
            scene_id="scene",
            split="train",
            segment_path=tmp_path / "nonexistent.png",
            anime_path=tmp_path / "nonexistent2.png",
        )
        assert not convert_frame(frame, tmp_path / "output")


# ---------------------------------------------------------------------------
# convert_scene
# ---------------------------------------------------------------------------


class TestConvertScene:
    """Test scene-level conversion."""

    def test_converts_all_frames(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, num_frames=4)
        split_dir = root / "train"
        output_dir = tmp_path / "output"

        result = convert_scene(split_dir, output_dir, "train", "scene_001", resolution=64)
        assert isinstance(result, AdapterResult)
        assert result.scene_id == "scene_001"
        assert result.frames_saved == 4

    def test_max_frames(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, num_frames=10)
        split_dir = root / "train"
        output_dir = tmp_path / "output"

        result = convert_scene(
            split_dir, output_dir, "train", "scene_001", resolution=64, max_frames=3
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
        segment_b = root / "train" / "Segment" / "scene_B"
        segment_b.mkdir(parents=True)
        anime_b = root / "train" / "Frame_Anime" / "scene_B" / "original"
        anime_b.mkdir(parents=True)
        mask = _create_segment_map()
        Image.fromarray(mask, mode="L").save(segment_b / "f001.png")
        _create_test_image().save(anime_b / "f001.png")

        output_dir = tmp_path / "output"
        results = convert_directory(root, output_dir, resolution=64)
        assert len(results) == 2
        total = sum(r.frames_saved for r in results)
        assert total == 3  # 2 + 1

    def test_max_scenes(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, "train", "scene_A", num_frames=1)
        segment_b = root / "train" / "Segment" / "scene_B"
        segment_b.mkdir(parents=True)
        anime_b = root / "train" / "Frame_Anime" / "scene_B" / "original"
        anime_b.mkdir(parents=True)
        mask = _create_segment_map()
        Image.fromarray(mask, mode="L").save(segment_b / "f001.png")
        _create_test_image().save(anime_b / "f001.png")

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
