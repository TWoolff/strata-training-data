"""Tests for the AnimeRun LineArea adapter.

These tests exercise the pure-Python adapter logic without requiring
the actual AnimeRun dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from ingest.animerun_linearea_adapter import (
    ANIMERUN_LINEAREA_SOURCE,
    AdapterResult,
    LineAreaFrame,
    _build_metadata,
    _resize_image,
    _resize_mask,
    convert_directory,
    convert_frame,
    convert_scene,
    discover_frames,
    discover_scenes,
    load_line_mask,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_grayscale_image(
    size: tuple[int, int] = (256, 256),
    value: int = 128,
) -> Image.Image:
    """Create a simple grayscale test image."""
    arr = np.full(size, value, dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def _create_line_mask_npy(
    size: tuple[int, int] = (256, 256),
    line_fraction: float = 0.2,
) -> np.ndarray:
    """Create a fake line area mask in AnimeRun NPY convention.

    Returns float64 array: 1.0 = non-line, 0.0 = line pixel.
    """
    arr = np.ones(size, dtype=np.float64)
    # Put line pixels in the top rows.
    line_rows = int(size[0] * line_fraction)
    arr[:line_rows, :] = 0.0
    return arr


def _setup_scene_dir(
    tmp_path: Path,
    split: str = "train",
    scene_name: str = "scene_001",
    *,
    num_frames: int = 3,
    with_images: bool = True,
    with_lines: bool = True,
    with_masks: bool = True,
    extra_image: bool = False,
) -> Path:
    """Create a fake AnimeRun v2 scene directory with LineArea data.

    Each frame gets: {id}.jpg (image), {id}_line.png (line art), {id}.npy (mask).

    Args:
        tmp_path: Pytest tmp_path fixture.
        split: Split name (train/test).
        scene_name: Scene directory name.
        num_frames: Number of complete frames to create.
        with_images: Create image JPGs.
        with_lines: Create line art PNGs.
        with_masks: Create mask NPY files.
        extra_image: Add one extra image without matching line/mask.

    Returns:
        Path to the root AnimeRun directory.
    """
    root = tmp_path / "animerun"
    linearea_dir = root / split / "LineArea" / scene_name
    linearea_dir.mkdir(parents=True)

    for i in range(num_frames):
        fid = f"{i:04d}"
        if with_images:
            img = _create_grayscale_image()
            img.save(linearea_dir / f"{fid}.jpg")
        if with_lines:
            line = _create_grayscale_image(value=200)
            line.save(linearea_dir / f"{fid}_line.png")
        if with_masks:
            mask = _create_line_mask_npy()
            np.save(linearea_dir / f"{fid}.npy", mask)

    if extra_image:
        fid = f"{num_frames:04d}"
        img = _create_grayscale_image()
        img.save(linearea_dir / f"{fid}.jpg")

    return root


# ---------------------------------------------------------------------------
# load_line_mask
# ---------------------------------------------------------------------------


class TestLoadLineMask:
    """Test line mask loading and inversion."""

    def test_load_npy_inverts_convention(self, tmp_path: Path) -> None:
        """NPY: 1.0=non-line, 0.0=line → output: 255=line, 0=non-line."""
        arr = np.ones((64, 64), dtype=np.float64)
        arr[10:20, :] = 0.0  # Line pixels.
        path = tmp_path / "mask.npy"
        np.save(path, arr)

        mask = load_line_mask(path)
        assert mask is not None
        assert mask.dtype == np.uint8
        assert mask.shape == (64, 64)
        # Line region should be 255.
        assert mask[15, 0] == 255
        # Non-line region should be 0.
        assert mask[0, 0] == 0

    def test_load_npy_3d_takes_first_channel(self, tmp_path: Path) -> None:
        arr = np.ones((64, 64, 3), dtype=np.float64)
        arr[10:20, :, :] = 0.0
        path = tmp_path / "mask_3d.npy"
        np.save(path, arr)

        mask = load_line_mask(path)
        assert mask is not None
        assert mask.ndim == 2
        assert mask[15, 0] == 255

    def test_load_png_inverts_bright_mask(self, tmp_path: Path) -> None:
        """PNG with mean > 127 (bright = non-line convention) gets inverted."""
        arr = np.full((64, 64), 255, dtype=np.uint8)
        arr[10:20, :] = 0  # Line pixels (dark).
        path = tmp_path / "mask.png"
        Image.fromarray(arr, mode="L").save(path)

        mask = load_line_mask(path)
        assert mask is not None
        assert mask.dtype == np.uint8
        # After inversion, line pixels should be 255.
        assert mask[15, 0] == 255
        # Non-line should be 0.
        assert mask[0, 0] == 0

    def test_load_png_dark_mask_no_inversion(self, tmp_path: Path) -> None:
        """PNG with mean <= 127 (dark = non-line convention) stays as-is."""
        arr = np.zeros((64, 64), dtype=np.uint8)
        arr[10:20, :] = 255  # Line pixels already bright.
        path = tmp_path / "mask_dark.png"
        Image.fromarray(arr, mode="L").save(path)

        mask = load_line_mask(path)
        assert mask is not None
        assert mask[15, 0] == 255
        assert mask[0, 0] == 0

    def test_load_unsupported_format(self, tmp_path: Path) -> None:
        path = tmp_path / "mask.bmp"
        path.write_bytes(b"not a real bmp")
        mask = load_line_mask(path)
        assert mask is None

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        mask = load_line_mask(tmp_path / "nonexistent.npy")
        assert mask is None


# ---------------------------------------------------------------------------
# _resize_image / _resize_mask
# ---------------------------------------------------------------------------


class TestResizeImage:
    """Test grayscale image resizing."""

    def test_resize_from_smaller(self) -> None:
        img = _create_grayscale_image((128, 128))
        resized = _resize_image(img, 512)
        assert resized.size == (512, 512)
        assert resized.mode == "L"

    def test_resize_from_larger(self) -> None:
        img = _create_grayscale_image((1024, 1024))
        resized = _resize_image(img, 512)
        assert resized.size == (512, 512)

    def test_already_correct_size(self) -> None:
        img = _create_grayscale_image((512, 512))
        resized = _resize_image(img, 512)
        assert resized.size == (512, 512)

    def test_converts_rgb_to_grayscale(self) -> None:
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        resized = _resize_image(img, 64)
        assert resized.mode == "L"


class TestResizeMask:
    """Test mask resizing with nearest-neighbor interpolation."""

    def test_resize_preserves_binary(self) -> None:
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[:30, :] = 255
        resized = _resize_mask(mask, 64)
        assert resized.shape == (64, 64)
        unique = set(np.unique(resized))
        assert unique <= {0, 255}

    def test_already_correct_size(self) -> None:
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[:100, :] = 255
        resized = _resize_mask(mask, 512)
        assert resized.shape == (512, 512)
        np.testing.assert_array_equal(resized, mask)


# ---------------------------------------------------------------------------
# _build_metadata
# ---------------------------------------------------------------------------


class TestBuildMetadata:
    """Test metadata generation."""

    def test_metadata_fields(self) -> None:
        frame = LineAreaFrame(
            frame_id="0001",
            scene_id="scene_001",
            split="train",
            image_path=Path("/fake/0001.jpg"),
            line_path=Path("/fake/0001_line.png"),
            mask_path=Path("/fake/0001.npy"),
        )
        meta = _build_metadata(frame, 512)
        assert meta["source"] == ANIMERUN_LINEAREA_SOURCE
        assert meta["scene_id"] == "scene_001"
        assert meta["frame_id"] == "0001"
        assert meta["split"] == "train"
        assert meta["resolution"] == 512
        assert meta["data_type"] == "line_area"
        assert meta["has_line_mask"] is True
        assert meta["has_line_art"] is True
        assert meta["has_segmentation_mask"] is False
        assert meta["has_joints"] is False
        assert meta["has_draw_order"] is False


# ---------------------------------------------------------------------------
# discover_scenes
# ---------------------------------------------------------------------------


class TestDiscoverScenes:
    """Test scene discovery."""

    def test_discovers_train_and_test(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, "train", "scene_A")
        # Add a test scene.
        test_la = root / "test" / "LineArea" / "scene_B"
        test_la.mkdir(parents=True)

        scenes = discover_scenes(root)
        assert len(scenes) == 2
        splits = [s[0] for s in scenes]
        assert "train" in splits
        assert "test" in splits

    def test_skips_hidden_dirs(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, "train", "scene_A")
        hidden = root / "train" / "LineArea" / ".hidden"
        hidden.mkdir(parents=True)

        scenes = discover_scenes(root)
        assert len(scenes) == 1
        assert scenes[0][1] == "scene_A"

    def test_empty_directory(self, tmp_path: Path) -> None:
        root = tmp_path / "animerun"
        root.mkdir()
        scenes = discover_scenes(root)
        assert scenes == []

    def test_finds_nested_animerun_v2(self, tmp_path: Path) -> None:
        """The zip may extract to AnimeRun_v2/ inside the target dir."""
        root = tmp_path / "animerun"
        nested = root / "AnimeRun_v2" / "train" / "LineArea" / "scene_A"
        nested.mkdir(parents=True)

        scenes = discover_scenes(root)
        assert len(scenes) == 1
        assert scenes[0][1] == "scene_A"


# ---------------------------------------------------------------------------
# discover_frames
# ---------------------------------------------------------------------------


class TestDiscoverFrames:
    """Test frame discovery."""

    def test_discovers_matched_triples(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, num_frames=5)
        split_dir = root / "train"

        frames = discover_frames(split_dir, "train", "scene_001")
        assert len(frames) == 5
        assert all(f.scene_id == "scene_001" for f in frames)
        assert all(f.split == "train" for f in frames)
        # Check all paths exist.
        for f in frames:
            assert f.image_path.suffix == ".jpg"
            assert f.line_path.suffix == ".png"
            assert f.mask_path.suffix == ".npy"

    def test_extra_image_without_match(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, num_frames=3, extra_image=True)
        split_dir = root / "train"

        frames = discover_frames(split_dir, "train", "scene_001")
        assert len(frames) == 3  # Only matched triples.

    def test_missing_masks(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, num_frames=3, with_masks=False)
        split_dir = root / "train"

        frames = discover_frames(split_dir, "train", "scene_001")
        assert len(frames) == 0

    def test_missing_lines(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, num_frames=3, with_lines=False)
        split_dir = root / "train"

        frames = discover_frames(split_dir, "train", "scene_001")
        assert len(frames) == 0

    def test_empty_scene(self, tmp_path: Path) -> None:
        split_dir = tmp_path / "split"
        (split_dir / "LineArea" / "empty_scene").mkdir(parents=True)

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

        example_dir = output_dir / f"{ANIMERUN_LINEAREA_SOURCE}_scene_001_0000"
        assert example_dir.is_dir()
        assert (example_dir / "image.png").is_file()
        assert (example_dir / "line_art.png").is_file()
        assert (example_dir / "line_mask.png").is_file()
        assert (example_dir / "metadata.json").is_file()

        # Check image size.
        img = Image.open(example_dir / "image.png")
        assert img.size == (64, 64)

        # Check line art size.
        line = Image.open(example_dir / "line_art.png")
        assert line.size == (64, 64)

        # Check mask size and mode.
        mask = Image.open(example_dir / "line_mask.png")
        assert mask.size == (64, 64)
        assert mask.mode == "L"

        # Check metadata.
        meta = json.loads((example_dir / "metadata.json").read_text(encoding="utf-8"))
        assert meta["source"] == ANIMERUN_LINEAREA_SOURCE
        assert meta["data_type"] == "line_area"
        assert meta["resolution"] == 64
        assert meta["has_line_mask"] is True
        assert meta["has_line_art"] is True

    def test_mask_inversion(self, tmp_path: Path) -> None:
        """Verify the mask is correctly inverted (255 = line pixel)."""
        root = _setup_scene_dir(tmp_path, num_frames=1)
        split_dir = root / "train"
        frames = discover_frames(split_dir, "train", "scene_001")
        output_dir = tmp_path / "output"

        convert_frame(frames[0], output_dir, resolution=256)

        example_dir = output_dir / f"{ANIMERUN_LINEAREA_SOURCE}_scene_001_0000"
        mask = np.array(Image.open(example_dir / "line_mask.png"))
        # Our test mask has line pixels (0.0) in the top 20% of rows.
        # After inversion: those rows should be 255.
        unique = set(np.unique(mask))
        assert unique <= {0, 255}
        # Top area should have line pixels (255).
        assert mask[0, 0] == 255

    def test_only_new_skips_existing(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, num_frames=1)
        split_dir = root / "train"
        frames = discover_frames(split_dir, "train", "scene_001")
        output_dir = tmp_path / "output"

        assert convert_frame(frames[0], output_dir, resolution=64)
        assert not convert_frame(frames[0], output_dir, resolution=64, only_new=True)

    def test_invalid_paths(self, tmp_path: Path) -> None:
        frame = LineAreaFrame(
            frame_id="bad",
            scene_id="scene",
            split="train",
            image_path=tmp_path / "nonexistent.jpg",
            line_path=tmp_path / "nonexistent_line.png",
            mask_path=tmp_path / "nonexistent.npy",
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
        scene_b = root / "train" / "LineArea" / "scene_B"
        scene_b.mkdir(parents=True)
        img = _create_grayscale_image()
        img.save(scene_b / "0001.jpg")
        line = _create_grayscale_image(value=200)
        line.save(scene_b / "0001_line.png")
        mask = _create_line_mask_npy()
        np.save(scene_b / "0001.npy", mask)

        output_dir = tmp_path / "output"
        results = convert_directory(root, output_dir, resolution=64)
        assert len(results) == 2
        total = sum(r.frames_saved for r in results)
        assert total == 3  # 2 + 1

    def test_max_scenes(self, tmp_path: Path) -> None:
        root = _setup_scene_dir(tmp_path, "train", "scene_A", num_frames=1)
        scene_b = root / "train" / "LineArea" / "scene_B"
        scene_b.mkdir(parents=True)
        img = _create_grayscale_image()
        img.save(scene_b / "0001.jpg")
        line = _create_grayscale_image(value=200)
        line.save(scene_b / "0001_line.png")
        mask = _create_line_mask_npy()
        np.save(scene_b / "0001.npy", mask)

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
