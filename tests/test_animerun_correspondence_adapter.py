"""Tests for the AnimeRun temporal correspondence adapter.

These tests exercise the pure-Python adapter logic without requiring
the actual AnimeRun dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from ingest.animerun_correspondence_adapter import (
    AdapterResult,
    CorrespondencePair,
    _build_metadata,
    _resize_image,
    _resize_mask,
    convert_directory,
    convert_pair,
    convert_scene,
    discover_pairs,
    discover_scenes,
    generate_occlusion_overlay,
    load_correspondence_map,
    load_occlusion_mask,
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


def _create_occlusion_mask(
    size: tuple[int, int] = (256, 256),
    fill_ratio: float = 0.3,
) -> np.ndarray:
    """Create a fake binary occlusion mask with a filled band."""
    arr = np.zeros(size, dtype=np.uint8)
    band_h = int(size[0] * fill_ratio)
    arr[:band_h, :] = 255
    return arr


def _create_seg_matching(size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Create a fake segment correspondence map with horizontal bands."""
    arr = np.zeros(size, dtype=np.uint8)
    band_height = size[0] // 4
    for i in range(1, 4):
        arr[i * band_height : (i + 1) * band_height, :] = i
    return arr


def _setup_scene_dir(
    tmp_path: Path,
    split: str = "train",
    scene_name: str = "scene_001",
    *,
    num_frames: int = 3,
    with_seg_matching: bool = True,
    with_anime: bool = True,
    with_fwd: bool = False,
    with_bwd: bool = False,
    use_npy: bool = False,
) -> Path:
    """Create a fake AnimeRun v2 scene directory with correspondence data.

    Args:
        tmp_path: Pytest tmp_path fixture.
        split: Split name (train/test).
        scene_name: Scene directory name.
        num_frames: Number of frames to create.
        with_seg_matching: Create SegMatching directory.
        with_anime: Create Frame_Anime directory.
        with_fwd: Create UnmatchedForward directory.
        with_bwd: Create UnmatchedBackward directory.
        use_npy: Use .npy format instead of .png.

    Returns:
        Path to the root AnimeRun directory.
    """
    root = tmp_path / "animerun"
    split_dir = root / split

    if with_seg_matching:
        seg_dir = split_dir / "SegMatching" / scene_name
        seg_dir.mkdir(parents=True)
        for i in range(num_frames):
            data = _create_seg_matching()
            if use_npy:
                np.save(seg_dir / f"frame_{i:04d}.npy", data)
            else:
                img = Image.fromarray(data, mode="L")
                img.save(seg_dir / f"frame_{i:04d}.png")

    if with_fwd:
        fwd_dir = split_dir / "UnmatchedForward" / scene_name
        fwd_dir.mkdir(parents=True)
        for i in range(num_frames):
            mask = _create_occlusion_mask()
            if use_npy:
                np.save(fwd_dir / f"frame_{i:04d}.npy", mask)
            else:
                img = Image.fromarray(mask, mode="L")
                img.save(fwd_dir / f"frame_{i:04d}.png")

    if with_bwd:
        bwd_dir = split_dir / "UnmatchedBackward" / scene_name
        bwd_dir.mkdir(parents=True)
        for i in range(num_frames):
            mask = _create_occlusion_mask()
            if use_npy:
                np.save(bwd_dir / f"frame_{i:04d}.npy", mask)
            else:
                img = Image.fromarray(mask, mode="L")
                img.save(bwd_dir / f"frame_{i:04d}.png")

    if with_anime:
        anime_dir = split_dir / "Frame_Anime" / scene_name / "original"
        anime_dir.mkdir(parents=True)
        for i in range(num_frames):
            img = _create_test_image()
            img.save(anime_dir / f"frame_{i:04d}.png")

    return root


# ---------------------------------------------------------------------------
# load_correspondence_map
# ---------------------------------------------------------------------------


class TestLoadCorrespondenceMap:
    """Test correspondence map loading."""

    def test_load_png(self, tmp_path: Path) -> None:
        data = _create_seg_matching((64, 64))
        path = tmp_path / "seg.png"
        Image.fromarray(data, mode="L").save(path)

        loaded = load_correspondence_map(path)
        assert loaded is not None
        assert loaded.shape == (64, 64)
        np.testing.assert_array_equal(loaded, data)

    def test_load_npy(self, tmp_path: Path) -> None:
        data = _create_seg_matching((64, 64))
        path = tmp_path / "seg.npy"
        np.save(path, data)

        loaded = load_correspondence_map(path)
        assert loaded is not None
        assert loaded.shape == (64, 64)
        np.testing.assert_array_equal(loaded, data)

    def test_returns_none_on_missing_file(self, tmp_path: Path) -> None:
        result = load_correspondence_map(tmp_path / "nonexistent.npy")
        assert result is None


# ---------------------------------------------------------------------------
# load_occlusion_mask
# ---------------------------------------------------------------------------


class TestLoadOcclusionMask:
    """Test occlusion mask loading."""

    def test_load_png(self, tmp_path: Path) -> None:
        mask = _create_occlusion_mask((64, 64))
        path = tmp_path / "occ.png"
        Image.fromarray(mask, mode="L").save(path)

        loaded = load_occlusion_mask(path)
        assert loaded is not None
        assert loaded.shape == (64, 64)
        assert loaded.dtype == np.uint8
        # Binary: only 0 and 255.
        assert set(np.unique(loaded)).issubset({0, 255})

    def test_load_npy(self, tmp_path: Path) -> None:
        mask = _create_occlusion_mask((64, 64))
        path = tmp_path / "occ.npy"
        np.save(path, mask)

        loaded = load_occlusion_mask(path)
        assert loaded is not None
        assert set(np.unique(loaded)).issubset({0, 255})

    def test_normalizes_nonbinary_values(self, tmp_path: Path) -> None:
        # Mask with values 0, 50, 100 should be normalized to 0, 255.
        arr = np.array([[0, 50], [100, 0]], dtype=np.uint8)
        path = tmp_path / "multi.npy"
        np.save(path, arr)

        loaded = load_occlusion_mask(path)
        assert loaded is not None
        expected = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(loaded, expected)

    def test_load_npy_3d_takes_first_channel(self, tmp_path: Path) -> None:
        mask = _create_occlusion_mask((64, 64))
        arr_3d = np.stack([mask, mask], axis=-1)
        path = tmp_path / "occ_3d.npy"
        np.save(path, arr_3d)

        loaded = load_occlusion_mask(path)
        assert loaded is not None
        assert loaded.ndim == 2

    def test_returns_none_on_missing_file(self, tmp_path: Path) -> None:
        result = load_occlusion_mask(tmp_path / "nonexistent.png")
        assert result is None


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

    def test_resize_preserves_binary_values(self) -> None:
        mask = _create_occlusion_mask((128, 128))
        resized = _resize_mask(mask, 64)
        assert resized.shape == (64, 64)
        assert set(np.unique(resized)).issubset({0, 255})

    def test_already_correct_size(self) -> None:
        mask = _create_occlusion_mask((512, 512))
        resized = _resize_mask(mask, 512)
        assert resized.shape == (512, 512)
        np.testing.assert_array_equal(resized, mask)


# ---------------------------------------------------------------------------
# generate_occlusion_overlay
# ---------------------------------------------------------------------------


class TestGenerateOcclusionOverlay:
    """Test overlay visualization generation."""

    def test_overlay_shape(self) -> None:
        img = _create_test_image((64, 64))
        fwd = _create_occlusion_mask((64, 64))
        overlay = generate_occlusion_overlay(img, fwd, None)
        assert overlay.size == (64, 64)
        assert overlay.mode == "RGB"

    def test_overlay_differs_from_original(self) -> None:
        img = _create_test_image((64, 64), (128, 128, 128))
        fwd = _create_occlusion_mask((64, 64), fill_ratio=0.5)
        overlay = generate_occlusion_overlay(img, fwd, None)
        assert not np.array_equal(np.array(img), np.array(overlay))

    def test_overlay_no_masks(self) -> None:
        img = _create_test_image((64, 64), (128, 128, 128))
        overlay = generate_occlusion_overlay(img, None, None)
        np.testing.assert_array_equal(np.array(img), np.array(overlay))

    def test_overlay_both_masks(self) -> None:
        img = _create_test_image((64, 64))
        fwd = _create_occlusion_mask((64, 64), fill_ratio=0.3)
        bwd = np.zeros((64, 64), dtype=np.uint8)
        bwd[48:, :] = 255  # Bottom band.
        overlay = generate_occlusion_overlay(img, fwd, bwd)
        assert overlay.size == (64, 64)


# ---------------------------------------------------------------------------
# _build_metadata
# ---------------------------------------------------------------------------


class TestBuildMetadata:
    """Test metadata generation."""

    def test_metadata_fields(self) -> None:
        pair = CorrespondencePair(
            frame_id="frame_0001",
            scene_id="scene_001",
            split="train",
            frame_t_path=Path("/fake/t.png"),
            frame_t1_path=Path("/fake/t1.png"),
        )
        meta = _build_metadata(
            pair,
            512,
            has_seg_matching=True,
            has_occlusion_fwd=True,
            has_occlusion_bwd=False,
        )
        assert meta["source"] == "animerun_correspondence"
        assert meta["scene_id"] == "scene_001"
        assert meta["frame_id"] == "frame_0001"
        assert meta["split"] == "train"
        assert meta["data_type"] == "temporal_correspondence"
        assert meta["has_seg_matching"] is True
        assert meta["has_occlusion_forward"] is True
        assert meta["has_occlusion_backward"] is False
        assert meta["has_segmentation_mask"] is False
        assert meta["has_joints"] is False

    def test_metadata_all_missing(self) -> None:
        pair = CorrespondencePair(
            frame_id="f",
            scene_id="s",
            split="test",
            frame_t_path=Path("/a"),
            frame_t1_path=Path("/b"),
        )
        meta = _build_metadata(
            pair, 256, has_seg_matching=False, has_occlusion_fwd=False, has_occlusion_bwd=False
        )
        assert meta["has_seg_matching"] is False
        assert meta["has_occlusion_forward"] is False
        assert meta["has_occlusion_backward"] is False


# ---------------------------------------------------------------------------
# discover_scenes
# ---------------------------------------------------------------------------


class TestDiscoverScenes:
    """Test scene discovery."""

    def test_discovers_train_and_test(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, "train", "scene_A")
        # Add a test scene.
        test_seg = root / "test" / "SegMatching" / "scene_B"
        test_seg.mkdir(parents=True)
        test_anime = root / "test" / "Frame_Anime" / "scene_B" / "original"
        test_anime.mkdir(parents=True)

        scenes = discover_scenes(root)
        assert len(scenes) == 2
        splits = [s[0] for s in scenes]
        assert "train" in splits
        assert "test" in splits

    def test_skips_scene_without_anime(self, tmp_path: Path) -> None:
        root = tmp_path / "animerun"
        (root / "train" / "SegMatching" / "bad_scene").mkdir(parents=True)
        scenes = discover_scenes(root)
        assert len(scenes) == 0

    def test_skips_hidden_dirs(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, "train", "scene_A")
        hidden = root / "train" / "SegMatching" / ".hidden"
        hidden.mkdir(parents=True)

        scenes = discover_scenes(root)
        assert len(scenes) == 1
        assert scenes[0][1] == "scene_A"

    def test_empty_directory(self, tmp_path: Path) -> None:
        root = tmp_path / "animerun"
        root.mkdir()
        scenes = discover_scenes(root)
        assert scenes == []

    def test_nested_animerun_v2(self, tmp_path: Path) -> None:
        outer = tmp_path / "data"
        outer.mkdir()
        inner = outer / "AnimeRun_v2" / "train" / "SegMatching" / "sc1"
        inner.mkdir(parents=True)
        anime = outer / "AnimeRun_v2" / "train" / "Frame_Anime" / "sc1" / "original"
        anime.mkdir(parents=True)

        scenes = discover_scenes(outer)
        assert len(scenes) == 1
        assert scenes[0][1] == "sc1"


# ---------------------------------------------------------------------------
# discover_pairs
# ---------------------------------------------------------------------------


class TestDiscoverPairs:
    """Test pair discovery."""

    def test_discovers_consecutive_pairs(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, num_frames=5)
        split_dir = root / "train"

        pairs = discover_pairs(split_dir, "train", "scene_001")
        # 5 frames → 4 consecutive pairs, but only where SegMatching exists.
        # We created 5 SegMatching files, so frames 0-3 each have a next frame.
        assert len(pairs) == 4
        assert all(p.scene_id == "scene_001" for p in pairs)
        assert all(p.split == "train" for p in pairs)

    def test_last_frame_excluded(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, num_frames=2)
        split_dir = root / "train"

        pairs = discover_pairs(split_dir, "train", "scene_001")
        assert len(pairs) == 1
        assert pairs[0].frame_id == "frame_0000"

    def test_includes_occlusion_masks(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, num_frames=3, with_fwd=True, with_bwd=True)
        split_dir = root / "train"

        pairs = discover_pairs(split_dir, "train", "scene_001")
        assert len(pairs) == 2
        assert all(p.unmatched_fwd_path is not None for p in pairs)
        assert all(p.unmatched_bwd_path is not None for p in pairs)

    def test_missing_occlusion_still_discovers(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, num_frames=3)
        split_dir = root / "train"

        pairs = discover_pairs(split_dir, "train", "scene_001")
        assert len(pairs) == 2
        assert all(p.unmatched_fwd_path is None for p in pairs)
        assert all(p.unmatched_bwd_path is None for p in pairs)

    def test_npy_seg_matching(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, num_frames=2, use_npy=True)
        split_dir = root / "train"

        pairs = discover_pairs(split_dir, "train", "scene_001")
        assert len(pairs) == 1
        assert pairs[0].seg_matching_path is not None
        assert pairs[0].seg_matching_path.suffix == ".npy"


# ---------------------------------------------------------------------------
# convert_pair
# ---------------------------------------------------------------------------


class TestConvertPair:
    """Test single pair conversion."""

    def test_convert_creates_output(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, num_frames=2, with_fwd=True, with_bwd=True)
        split_dir = root / "train"
        pairs = discover_pairs(split_dir, "train", "scene_001")
        output_dir = tmp_path / "output"

        saved = convert_pair(pairs[0], output_dir, resolution=64)
        assert saved is True

        example_dir = output_dir / "animerun_correspondence_scene_001_frame_0000"
        assert example_dir.is_dir()
        assert (example_dir / "frame_t.png").is_file()
        assert (example_dir / "frame_t1.png").is_file()
        assert (example_dir / "seg_matching.npy").is_file()
        assert (example_dir / "occlusion_forward.png").is_file()
        assert (example_dir / "occlusion_backward.png").is_file()
        assert (example_dir / "occlusion_overlay.png").is_file()
        assert (example_dir / "metadata.json").is_file()

        # Check image size.
        img = Image.open(example_dir / "frame_t.png")
        assert img.size == (64, 64)

        # Check occlusion mask is grayscale.
        occ = Image.open(example_dir / "occlusion_forward.png")
        assert occ.mode == "L"
        assert occ.size == (64, 64)

        # Check metadata.
        meta = json.loads((example_dir / "metadata.json").read_text(encoding="utf-8"))
        assert meta["source"] == "animerun_correspondence"
        assert meta["data_type"] == "temporal_correspondence"
        assert meta["resolution"] == 64
        assert meta["has_seg_matching"] is True
        assert meta["has_occlusion_forward"] is True
        assert meta["has_occlusion_backward"] is True

    def test_convert_without_occlusion(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, num_frames=2)
        split_dir = root / "train"
        pairs = discover_pairs(split_dir, "train", "scene_001")
        output_dir = tmp_path / "output"

        saved = convert_pair(pairs[0], output_dir, resolution=64)
        assert saved is True

        example_dir = output_dir / "animerun_correspondence_scene_001_frame_0000"
        assert (example_dir / "frame_t.png").is_file()
        assert (example_dir / "seg_matching.npy").is_file()
        assert not (example_dir / "occlusion_forward.png").exists()
        assert not (example_dir / "occlusion_backward.png").exists()
        assert not (example_dir / "occlusion_overlay.png").exists()

        meta = json.loads((example_dir / "metadata.json").read_text(encoding="utf-8"))
        assert meta["has_occlusion_forward"] is False
        assert meta["has_occlusion_backward"] is False

    def test_only_new_skips_existing(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, num_frames=2)
        split_dir = root / "train"
        pairs = discover_pairs(split_dir, "train", "scene_001")
        output_dir = tmp_path / "output"

        assert convert_pair(pairs[0], output_dir, resolution=64)
        assert not convert_pair(pairs[0], output_dir, resolution=64, only_new=True)

    def test_invalid_paths(self, tmp_path: Path) -> None:
        pair = CorrespondencePair(
            frame_id="bad",
            scene_id="scene",
            split="train",
            frame_t_path=tmp_path / "nonexistent.png",
            frame_t1_path=tmp_path / "nonexistent2.png",
        )
        assert not convert_pair(pair, tmp_path / "output")


# ---------------------------------------------------------------------------
# convert_scene
# ---------------------------------------------------------------------------


class TestConvertScene:
    """Test scene-level conversion."""

    def test_converts_all_pairs(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, num_frames=5)
        split_dir = root / "train"
        output_dir = tmp_path / "output"

        result = convert_scene(split_dir, output_dir, "train", "scene_001", resolution=64)
        assert isinstance(result, AdapterResult)
        assert result.scene_id == "scene_001"
        assert result.frames_saved == 4  # 5 frames → 4 pairs.

    def test_max_frames(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, num_frames=10)
        split_dir = root / "train"
        output_dir = tmp_path / "output"

        result = convert_scene(
            split_dir, output_dir, "train", "scene_001", resolution=64, max_frames=2
        )
        assert result.frames_saved == 2


# ---------------------------------------------------------------------------
# convert_directory
# ---------------------------------------------------------------------------


class TestConvertDirectory:
    """Test batch directory conversion."""

    def test_converts_multiple_scenes(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, "train", "scene_A", num_frames=3)
        # Add second scene.
        seg_b = root / "train" / "SegMatching" / "scene_B"
        seg_b.mkdir(parents=True)
        anime_b = root / "train" / "Frame_Anime" / "scene_B" / "original"
        anime_b.mkdir(parents=True)
        data = _create_seg_matching()
        Image.fromarray(data, mode="L").save(seg_b / "f001.png")
        Image.fromarray(data, mode="L").save(seg_b / "f002.png")
        _create_test_image().save(anime_b / "f001.png")
        _create_test_image().save(anime_b / "f002.png")

        output_dir = tmp_path / "output"
        results = convert_directory(root, output_dir, resolution=64)
        assert len(results) == 2
        total = sum(r.frames_saved for r in results)
        assert (
            total == 3
        )  # 2 from scene_A (3 frames → 2 pairs) + 1 from scene_B (2 frames → 1 pair)

    def test_max_scenes(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, "train", "scene_A", num_frames=2)
        seg_b = root / "train" / "SegMatching" / "scene_B"
        seg_b.mkdir(parents=True)
        anime_b = root / "train" / "Frame_Anime" / "scene_B" / "original"
        anime_b.mkdir(parents=True)
        data = _create_seg_matching()
        Image.fromarray(data, mode="L").save(seg_b / "f001.png")
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
