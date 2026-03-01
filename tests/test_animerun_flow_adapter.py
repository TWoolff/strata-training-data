"""Tests for the AnimeRun optical flow adapter.

These tests exercise the pure-Python adapter logic without requiring
the actual AnimeRun dataset.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from ingest.animerun_flow_adapter import (
    AdapterResult,
    FlowPair,
    _build_metadata,
    _resize_image,
    convert_directory,
    convert_pair,
    convert_scene,
    discover_pairs,
    discover_scenes,
    flow_to_hsv,
    load_flow,
    read_flo_file,
    scale_flow,
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


def _create_flow_array(
    h: int = 64,
    w: int = 64,
    dx: float = 1.0,
    dy: float = 0.5,
) -> np.ndarray:
    """Create a uniform optical flow array."""
    flow = np.zeros((h, w, 2), dtype=np.float32)
    flow[:, :, 0] = dx
    flow[:, :, 1] = dy
    return flow


def _write_flo_file(path: Path, flow: np.ndarray) -> None:
    """Write a Middlebury .flo file."""
    h, w = flow.shape[:2]
    with open(path, "wb") as f:
        f.write(struct.pack("f", 202021.25))
        f.write(struct.pack("i", w))
        f.write(struct.pack("i", h))
        f.write(flow.astype(np.float32).tobytes())


def _setup_flow_scene(
    tmp_path: Path,
    split: str = "train",
    scene_name: str = "scene_001",
    *,
    num_frames: int = 4,
    flow_format: str = "npy",
    with_forward: bool = True,
    with_backward: bool = True,
    with_anime: bool = True,
) -> Path:
    """Create a fake AnimeRun flow scene directory.

    Creates num_frames anime frames and (num_frames - 1) flow files
    (flow is between consecutive frames).

    Args:
        tmp_path: Pytest tmp_path fixture.
        split: Split name (train/test).
        scene_name: Scene directory name.
        num_frames: Number of anime frames.
        flow_format: "npy" or "flo" for flow files.
        with_forward: Create forward flow directory.
        with_backward: Create backward flow directory.
        with_anime: Create Frame_Anime directory.

    Returns:
        Path to the root AnimeRun directory.
    """
    root = tmp_path / "animerun"
    split_dir = root / split

    if with_anime:
        anime_dir = split_dir / "Frame_Anime" / scene_name / "original"
        anime_dir.mkdir(parents=True)
        for i in range(num_frames):
            img = _create_test_image()
            img.save(anime_dir / f"frame_{i:04d}.png")

    flow_scene_dir = split_dir / "Flow" / scene_name

    if with_forward:
        fwd_dir = flow_scene_dir / "forward"
        fwd_dir.mkdir(parents=True)
        for i in range(num_frames - 1):
            flow = _create_flow_array(64, 64, dx=float(i + 1), dy=float(i))
            stem = f"frame_{i:04d}"
            if flow_format == "npy":
                np.save(fwd_dir / f"{stem}.npy", flow)
            else:
                _write_flo_file(fwd_dir / f"{stem}.flo", flow)

    if with_backward:
        bwd_dir = flow_scene_dir / "backward"
        bwd_dir.mkdir(parents=True)
        for i in range(num_frames - 1):
            flow = _create_flow_array(64, 64, dx=-float(i + 1), dy=-float(i))
            stem = f"frame_{i:04d}"
            if flow_format == "npy":
                np.save(bwd_dir / f"{stem}.npy", flow)
            else:
                _write_flo_file(bwd_dir / f"{stem}.flo", flow)

    return root


# ---------------------------------------------------------------------------
# read_flo_file / load_flow
# ---------------------------------------------------------------------------


class TestFlowIO:
    """Test optical flow I/O functions."""

    def test_read_flo_file(self, tmp_path: Path) -> None:
        flow = _create_flow_array(32, 48, dx=2.5, dy=-1.0)
        flo_path = tmp_path / "test.flo"
        _write_flo_file(flo_path, flow)

        loaded = read_flo_file(flo_path)
        assert loaded is not None
        assert loaded.shape == (32, 48, 2)
        assert loaded.dtype == np.float32
        np.testing.assert_allclose(loaded, flow)

    def test_read_flo_invalid_magic(self, tmp_path: Path) -> None:
        flo_path = tmp_path / "bad.flo"
        with open(flo_path, "wb") as f:
            f.write(struct.pack("f", 0.0))  # Wrong magic.
            f.write(struct.pack("i", 2))
            f.write(struct.pack("i", 2))
        assert read_flo_file(flo_path) is None

    def test_load_flow_npy(self, tmp_path: Path) -> None:
        flow = _create_flow_array(16, 16)
        npy_path = tmp_path / "flow.npy"
        np.save(npy_path, flow)

        loaded = load_flow(npy_path)
        assert loaded is not None
        np.testing.assert_allclose(loaded, flow)

    def test_load_flow_flo(self, tmp_path: Path) -> None:
        flow = _create_flow_array(16, 16)
        flo_path = tmp_path / "flow.flo"
        _write_flo_file(flo_path, flow)

        loaded = load_flow(flo_path)
        assert loaded is not None
        np.testing.assert_allclose(loaded, flow)

    def test_load_flow_bad_shape(self, tmp_path: Path) -> None:
        bad_arr = np.zeros((16, 16, 3), dtype=np.float32)  # Wrong channels.
        npy_path = tmp_path / "bad.npy"
        np.save(npy_path, bad_arr)
        assert load_flow(npy_path) is None

    def test_load_flow_unsupported_ext(self, tmp_path: Path) -> None:
        path = tmp_path / "flow.bin"
        path.write_bytes(b"data")
        assert load_flow(path) is None


# ---------------------------------------------------------------------------
# flow_to_hsv
# ---------------------------------------------------------------------------


class TestFlowToHsv:
    """Test HSV flow visualization."""

    def test_output_shape_and_dtype(self) -> None:
        flow = _create_flow_array(32, 32, dx=1.0, dy=0.0)
        rgb = flow_to_hsv(flow)
        assert rgb.shape == (32, 32, 3)
        assert rgb.dtype == np.uint8

    def test_zero_flow_produces_dark_image(self) -> None:
        flow = _create_flow_array(16, 16, dx=0.0, dy=0.0)
        rgb = flow_to_hsv(flow)
        # Zero magnitude → value channel is zero → dark output.
        assert rgb.max() < 128

    def test_uniform_flow_produces_uniform_color(self) -> None:
        flow = _create_flow_array(16, 16, dx=5.0, dy=0.0)
        rgb = flow_to_hsv(flow)
        # All pixels should have the same color.
        assert np.all(rgb == rgb[0, 0])


# ---------------------------------------------------------------------------
# scale_flow
# ---------------------------------------------------------------------------


class TestScaleFlow:
    """Test optical flow resizing."""

    def test_no_resize_needed(self) -> None:
        flow = _create_flow_array(64, 64, dx=2.0, dy=1.0)
        scaled = scale_flow(flow, 64, 64)
        np.testing.assert_allclose(scaled, flow)

    def test_upscale_doubles_values(self) -> None:
        flow = _create_flow_array(32, 32, dx=1.0, dy=1.0)
        scaled = scale_flow(flow, 64, 64)
        assert scaled.shape == (64, 64, 2)
        # Displacement should be doubled (2x scale in each direction).
        np.testing.assert_allclose(scaled[:, :, 0], 2.0, atol=0.1)
        np.testing.assert_allclose(scaled[:, :, 1], 2.0, atol=0.1)

    def test_downscale_halves_values(self) -> None:
        flow = _create_flow_array(64, 64, dx=4.0, dy=2.0)
        scaled = scale_flow(flow, 32, 32)
        assert scaled.shape == (32, 32, 2)
        np.testing.assert_allclose(scaled[:, :, 0], 2.0, atol=0.1)
        np.testing.assert_allclose(scaled[:, :, 1], 1.0, atol=0.1)


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
# _build_metadata
# ---------------------------------------------------------------------------


class TestBuildMetadata:
    """Test metadata generation."""

    def test_metadata_fields(self) -> None:
        pair = FlowPair(
            frame_id="frame_0001",
            scene_id="scene_001",
            split="train",
            frame_t_path=Path("/fake/t.png"),
            frame_t1_path=Path("/fake/t1.png"),
        )
        meta = _build_metadata(pair, 512, has_forward=True, has_backward=False)
        assert meta["source"] == "animerun_flow"
        assert meta["scene_id"] == "scene_001"
        assert meta["frame_id"] == "frame_0001"
        assert meta["data_type"] == "optical_flow_pair"
        assert meta["has_optical_flow"] is True
        assert meta["has_forward_flow"] is True
        assert meta["has_backward_flow"] is False
        assert meta["has_joints"] is False

    def test_no_flow_flags(self) -> None:
        pair = FlowPair(
            frame_id="f",
            scene_id="s",
            split="train",
            frame_t_path=Path("/fake/t.png"),
            frame_t1_path=Path("/fake/t1.png"),
        )
        meta = _build_metadata(pair, 512, has_forward=False, has_backward=False)
        assert meta["has_optical_flow"] is False


# ---------------------------------------------------------------------------
# discover_scenes
# ---------------------------------------------------------------------------


class TestDiscoverScenes:
    """Test scene discovery."""

    def test_discovers_flow_scenes(self, tmp_path: Path) -> None:
        root = _setup_flow_scene(tmp_path, "train", "scene_A")
        scenes = discover_scenes(root)
        assert len(scenes) == 1
        assert scenes[0][0] == "train"
        assert scenes[0][1] == "scene_A"

    def test_discovers_multiple_splits(self, tmp_path: Path) -> None:
        root = _setup_flow_scene(tmp_path, "train", "scene_A")
        # Add a test scene.
        _setup_flow_scene(tmp_path, "test", "scene_B")
        scenes = discover_scenes(root)
        assert len(scenes) == 2
        splits = {s[0] for s in scenes}
        assert splits == {"train", "test"}

    def test_skips_scene_without_anime(self, tmp_path: Path) -> None:
        root = _setup_flow_scene(tmp_path, "train", "scene_A", with_anime=False)
        scenes = discover_scenes(root)
        assert len(scenes) == 0

    def test_skips_scene_without_flow_subdirs(self, tmp_path: Path) -> None:
        root = _setup_flow_scene(
            tmp_path, "train", "scene_A", with_forward=False, with_backward=False
        )
        scenes = discover_scenes(root)
        assert len(scenes) == 0

    def test_skips_hidden_dirs(self, tmp_path: Path) -> None:
        root = _setup_flow_scene(tmp_path, "train", "scene_A")
        hidden = root / "train" / "Flow" / ".hidden"
        hidden.mkdir(parents=True)
        scenes = discover_scenes(root)
        assert len(scenes) == 1

    def test_empty_directory(self, tmp_path: Path) -> None:
        root = tmp_path / "animerun"
        root.mkdir()
        scenes = discover_scenes(root)
        assert scenes == []

    def test_forward_only_scene(self, tmp_path: Path) -> None:
        root = _setup_flow_scene(tmp_path, "train", "scene_A", with_backward=False)
        scenes = discover_scenes(root)
        assert len(scenes) == 1


# ---------------------------------------------------------------------------
# discover_pairs
# ---------------------------------------------------------------------------


class TestDiscoverPairs:
    """Test frame pair discovery."""

    def test_discovers_consecutive_pairs(self, tmp_path: Path) -> None:
        root = _setup_flow_scene(tmp_path, num_frames=4)
        split_dir = root / "train"
        pairs = discover_pairs(split_dir, "train", "scene_001")
        # 4 frames → 3 flow files → 3 pairs.
        assert len(pairs) == 3
        assert all(p.scene_id == "scene_001" for p in pairs)

    def test_pairs_have_correct_frame_paths(self, tmp_path: Path) -> None:
        root = _setup_flow_scene(tmp_path, num_frames=3)
        split_dir = root / "train"
        pairs = discover_pairs(split_dir, "train", "scene_001")
        assert len(pairs) == 2

        # First pair: frame_0000 → frame_0001.
        assert pairs[0].frame_id == "frame_0000"
        assert pairs[0].frame_t_path.stem == "frame_0000"
        assert pairs[0].frame_t1_path.stem == "frame_0001"

        # Second pair: frame_0001 → frame_0002.
        assert pairs[1].frame_id == "frame_0001"
        assert pairs[1].frame_t_path.stem == "frame_0001"
        assert pairs[1].frame_t1_path.stem == "frame_0002"

    def test_forward_only(self, tmp_path: Path) -> None:
        root = _setup_flow_scene(tmp_path, num_frames=3, with_backward=False)
        split_dir = root / "train"
        pairs = discover_pairs(split_dir, "train", "scene_001")
        assert len(pairs) == 2
        assert all(p.flow_fwd_path is not None for p in pairs)
        assert all(p.flow_bwd_path is None for p in pairs)

    def test_flo_format(self, tmp_path: Path) -> None:
        root = _setup_flow_scene(tmp_path, num_frames=3, flow_format="flo")
        split_dir = root / "train"
        pairs = discover_pairs(split_dir, "train", "scene_001")
        assert len(pairs) == 2
        assert all(p.flow_fwd_path.suffix == ".flo" for p in pairs)

    def test_no_anime_frames(self, tmp_path: Path) -> None:
        root = _setup_flow_scene(tmp_path, num_frames=3, with_anime=False)
        # Create the anime dir but keep it empty.
        anime_dir = root / "train" / "Frame_Anime" / "scene_001" / "original"
        anime_dir.mkdir(parents=True)
        split_dir = root / "train"
        pairs = discover_pairs(split_dir, "train", "scene_001")
        assert pairs == []


# ---------------------------------------------------------------------------
# convert_pair
# ---------------------------------------------------------------------------


class TestConvertPair:
    """Test single pair conversion."""

    def test_convert_creates_output(self, tmp_path: Path) -> None:
        root = _setup_flow_scene(tmp_path, num_frames=3)
        split_dir = root / "train"
        pairs = discover_pairs(split_dir, "train", "scene_001")
        output_dir = tmp_path / "output"

        saved = convert_pair(pairs[0], output_dir, resolution=64)
        assert saved is True

        example_dir = output_dir / "animerun_flow_scene_001_frame_0000"
        assert example_dir.is_dir()
        assert (example_dir / "frame_t.png").is_file()
        assert (example_dir / "frame_t1.png").is_file()
        assert (example_dir / "flow_forward.npy").is_file()
        assert (example_dir / "flow_backward.npy").is_file()
        assert (example_dir / "flow_viz.png").is_file()
        assert (example_dir / "metadata.json").is_file()

        # Check image size.
        img = Image.open(example_dir / "frame_t.png")
        assert img.size == (64, 64)

        # Check flow shape.
        flow = np.load(example_dir / "flow_forward.npy")
        assert flow.shape == (64, 64, 2)

        # Check metadata.
        meta = json.loads((example_dir / "metadata.json").read_text(encoding="utf-8"))
        assert meta["source"] == "animerun_flow"
        assert meta["resolution"] == 64
        assert meta["has_optical_flow"] is True
        assert meta["has_forward_flow"] is True
        assert meta["has_backward_flow"] is True

    def test_only_new_skips_existing(self, tmp_path: Path) -> None:
        root = _setup_flow_scene(tmp_path, num_frames=3)
        split_dir = root / "train"
        pairs = discover_pairs(split_dir, "train", "scene_001")
        output_dir = tmp_path / "output"

        assert convert_pair(pairs[0], output_dir, resolution=64)
        assert not convert_pair(pairs[0], output_dir, resolution=64, only_new=True)

    def test_invalid_image_path(self, tmp_path: Path) -> None:
        pair = FlowPair(
            frame_id="bad",
            scene_id="scene",
            split="train",
            frame_t_path=tmp_path / "nonexistent.png",
            frame_t1_path=tmp_path / "nonexistent2.png",
        )
        assert not convert_pair(pair, tmp_path / "output")

    def test_flow_viz_not_created_without_forward(self, tmp_path: Path) -> None:
        root = _setup_flow_scene(tmp_path, num_frames=3, with_forward=False)
        split_dir = root / "train"
        pairs = discover_pairs(split_dir, "train", "scene_001")
        output_dir = tmp_path / "output"

        saved = convert_pair(pairs[0], output_dir, resolution=64)
        assert saved is True

        example_dir = output_dir / "animerun_flow_scene_001_frame_0000"
        assert not (example_dir / "flow_forward.npy").exists()
        assert not (example_dir / "flow_viz.png").exists()
        assert (example_dir / "flow_backward.npy").is_file()


# ---------------------------------------------------------------------------
# convert_scene
# ---------------------------------------------------------------------------


class TestConvertScene:
    """Test scene-level conversion."""

    def test_converts_all_pairs(self, tmp_path: Path) -> None:
        root = _setup_flow_scene(tmp_path, num_frames=5)
        split_dir = root / "train"
        output_dir = tmp_path / "output"

        result = convert_scene(split_dir, output_dir, "train", "scene_001", resolution=64)
        assert isinstance(result, AdapterResult)
        assert result.scene_id == "scene_001"
        assert result.frames_saved == 4  # 5 frames → 4 pairs.

    def test_max_frames(self, tmp_path: Path) -> None:
        root = _setup_flow_scene(tmp_path, num_frames=10)
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
        root = _setup_flow_scene(tmp_path, "train", "scene_A", num_frames=3)
        # Add second scene by manually creating the directory structure.
        split_dir = root / "train"
        anime_b = split_dir / "Frame_Anime" / "scene_B" / "original"
        anime_b.mkdir(parents=True)
        for i in range(2):
            _create_test_image().save(anime_b / f"frame_{i:04d}.png")
        fwd_b = split_dir / "Flow" / "scene_B" / "forward"
        fwd_b.mkdir(parents=True)
        flow = _create_flow_array()
        np.save(fwd_b / "frame_0000.npy", flow)

        output_dir = tmp_path / "output"
        results = convert_directory(root, output_dir, resolution=64)
        assert len(results) == 2
        total = sum(r.frames_saved for r in results)
        assert total == 3  # 2 from scene_A + 1 from scene_B.

    def test_max_scenes(self, tmp_path: Path) -> None:
        root = _setup_flow_scene(tmp_path, "train", "scene_A", num_frames=3)
        split_dir = root / "train"
        anime_b = split_dir / "Frame_Anime" / "scene_B" / "original"
        anime_b.mkdir(parents=True)
        _create_test_image().save(anime_b / "frame_0000.png")
        _create_test_image().save(anime_b / "frame_0001.png")
        fwd_b = split_dir / "Flow" / "scene_B" / "forward"
        fwd_b.mkdir(parents=True)
        np.save(fwd_b / "frame_0000.npy", _create_flow_array())

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
