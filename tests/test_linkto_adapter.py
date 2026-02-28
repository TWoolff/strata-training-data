"""Tests for the LinkTo-Anime adapter.

These tests exercise the pure-Python adapter logic without requiring
the actual LinkTo-Anime dataset or Blender.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from ingest.linkto_adapter import (
    AdapterResult,
    FrameData,
    JointPosition,
    _build_metadata,
    _joints_to_strata_json,
    _resize_image,
    _scale_joints,
    convert_directory,
    convert_frame,
    convert_sequence,
    discover_frames,
    discover_sequences,
    load_flow,
    parse_skeleton_json,
    parse_skeleton_npy,
    read_flo_file,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_test_image(size: tuple[int, int] = (256, 256)) -> Image.Image:
    """Create a simple RGB test image."""
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _create_flo_file(path: Path, width: int = 64, height: int = 32) -> None:
    """Create a valid Middlebury .flo file."""
    with open(path, "wb") as f:
        f.write(struct.pack("f", 202021.25))  # magic
        f.write(struct.pack("i", width))
        f.write(struct.pack("i", height))
        data = np.random.randn(height, width, 2).astype(np.float32)
        f.write(data.tobytes())


def _setup_sequence_dir(
    tmp_path: Path,
    split: str = "train",
    seq_name: str = "seq_001",
    *,
    num_frames: int = 3,
    with_skeleton: bool = False,
    with_flow: bool = False,
    skeleton_format: str = "json",
) -> Path:
    """Create a fake LinkTo-Anime sequence directory.

    Returns:
        Path to the root LinkTo-Anime directory.
    """
    root = tmp_path / "linkto_anime"
    seq_dir = root / split / seq_name
    frames_dir = seq_dir / "frames"
    frames_dir.mkdir(parents=True)

    for i in range(num_frames):
        img = _create_test_image()
        img.save(frames_dir / f"frame_{i:04d}.png")

    if with_skeleton:
        skel_dir = seq_dir / "skeleton"
        skel_dir.mkdir(parents=True)
        for i in range(num_frames):
            if skeleton_format == "json":
                skel_data = {
                    "mixamorig:Head": {"x": 128.0, "y": 32.0},
                    "mixamorig:Hips": {"x": 128.0, "y": 128.0},
                    "mixamorig:LeftArm": [64.0, 96.0],
                }
                (skel_dir / f"frame_{i:04d}.json").write_text(
                    json.dumps(skel_data),
                    encoding="utf-8",
                )
            elif skeleton_format == "npy":
                positions = np.array([[128.0, 32.0], [128.0, 128.0]], dtype=np.float32)
                np.save(skel_dir / f"frame_{i:04d}.npy", positions)

    if with_flow:
        fwd_dir = seq_dir / "flow_fwd"
        fwd_dir.mkdir(parents=True)
        bwd_dir = seq_dir / "flow_bwd"
        bwd_dir.mkdir(parents=True)
        for i in range(num_frames):
            _create_flo_file(fwd_dir / f"frame_{i:04d}.flo")
            _create_flo_file(bwd_dir / f"frame_{i:04d}.flo")

    return root


# ---------------------------------------------------------------------------
# read_flo_file / load_flow
# ---------------------------------------------------------------------------


class TestReadFloFile:
    """Test Middlebury .flo file reading."""

    def test_read_valid_flo(self, tmp_path: Path) -> None:
        flo_path = tmp_path / "test.flo"
        _create_flo_file(flo_path, width=32, height=16)
        flow = read_flo_file(flo_path)
        assert flow is not None
        assert flow.shape == (16, 32, 2)
        assert flow.dtype == np.float32

    def test_invalid_magic(self, tmp_path: Path) -> None:
        flo_path = tmp_path / "bad.flo"
        with open(flo_path, "wb") as f:
            f.write(struct.pack("f", 12345.0))  # Wrong magic.
            f.write(struct.pack("i", 4))
            f.write(struct.pack("i", 4))
        flow = read_flo_file(flo_path)
        assert flow is None

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        flow = read_flo_file(tmp_path / "missing.flo")
        assert flow is None


class TestLoadFlow:
    """Test optical flow loading."""

    def test_load_flo(self, tmp_path: Path) -> None:
        flo_path = tmp_path / "test.flo"
        _create_flo_file(flo_path, width=16, height=8)
        flow = load_flow(flo_path)
        assert flow is not None
        assert flow.shape == (8, 16, 2)

    def test_load_npy(self, tmp_path: Path) -> None:
        npy_path = tmp_path / "test.npy"
        data = np.random.randn(8, 16, 2).astype(np.float32)
        np.save(npy_path, data)
        flow = load_flow(npy_path)
        assert flow is not None
        assert flow.shape == (8, 16, 2)

    def test_load_npy_wrong_shape(self, tmp_path: Path) -> None:
        npy_path = tmp_path / "bad.npy"
        data = np.random.randn(8, 16).astype(np.float32)
        np.save(npy_path, data)
        flow = load_flow(npy_path)
        assert flow is None

    def test_unsupported_format(self, tmp_path: Path) -> None:
        path = tmp_path / "test.bin"
        path.write_bytes(b"data")
        flow = load_flow(path)
        assert flow is None


# ---------------------------------------------------------------------------
# parse_skeleton_json
# ---------------------------------------------------------------------------


class TestParseSkeletonJson:
    """Test skeleton JSON parsing."""

    def test_dict_format_with_xy_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "skel.json"
        data = {
            "mixamorig:Head": {"x": 100.0, "y": 50.0},
            "mixamorig:Hips": {"x": 100.0, "y": 128.0},
        }
        path.write_text(json.dumps(data), encoding="utf-8")

        joints = parse_skeleton_json(path)
        assert joints is not None
        assert len(joints) == 2
        head_joints = [j for j in joints if j.region_name == "head"]
        assert len(head_joints) == 1
        assert head_joints[0].x == 100.0
        assert head_joints[0].y == 50.0

    def test_dict_format_with_xy_list(self, tmp_path: Path) -> None:
        path = tmp_path / "skel.json"
        data = {"mixamorig:Head": [100.0, 50.0]}
        path.write_text(json.dumps(data), encoding="utf-8")

        joints = parse_skeleton_json(path)
        assert joints is not None
        assert len(joints) == 1

    def test_wrapped_in_joints_key(self, tmp_path: Path) -> None:
        path = tmp_path / "skel.json"
        data = {"joints": {"mixamorig:Head": [100.0, 50.0]}}
        path.write_text(json.dumps(data), encoding="utf-8")

        joints = parse_skeleton_json(path)
        assert joints is not None
        assert len(joints) == 1

    def test_list_format(self, tmp_path: Path) -> None:
        path = tmp_path / "skel.json"
        data = [
            {"name": "mixamorig:Head", "x": 100.0, "y": 50.0},
            {"name": "mixamorig:Hips", "x": 100.0, "y": 128.0},
        ]
        path.write_text(json.dumps(data), encoding="utf-8")

        joints = parse_skeleton_json(path)
        assert joints is not None
        assert len(joints) == 2

    def test_unmapped_bones_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "skel.json"
        data = {
            "mixamorig:Head": [100.0, 50.0],
            "WeirdUnknownBone": [0.0, 0.0],
        }
        path.write_text(json.dumps(data), encoding="utf-8")

        joints = parse_skeleton_json(path)
        assert joints is not None
        assert len(joints) == 1

    def test_stripped_mixamo_names(self, tmp_path: Path) -> None:
        path = tmp_path / "skel.json"
        data = {"Head": [100.0, 50.0], "Hips": [100.0, 128.0]}
        path.write_text(json.dumps(data), encoding="utf-8")

        joints = parse_skeleton_json(path)
        assert joints is not None
        assert len(joints) == 2

    def test_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not json{{{", encoding="utf-8")
        joints = parse_skeleton_json(path)
        assert joints is None

    def test_no_mapped_joints(self, tmp_path: Path) -> None:
        path = tmp_path / "skel.json"
        data = {"UnknownBone": [0.0, 0.0]}
        path.write_text(json.dumps(data), encoding="utf-8")

        joints = parse_skeleton_json(path)
        assert joints is None


# ---------------------------------------------------------------------------
# parse_skeleton_npy
# ---------------------------------------------------------------------------


class TestParseSkeletonNpy:
    """Test skeleton NPY parsing."""

    def test_load_valid_npy(self, tmp_path: Path) -> None:
        path = tmp_path / "skel.npy"
        data = np.array([[100.0, 50.0], [100.0, 128.0]], dtype=np.float32)
        np.save(path, data)

        joints = parse_skeleton_npy(path)
        assert joints is not None
        assert len(joints) == 2
        assert joints[0].region_id == 1  # First maps to region 1

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        joints = parse_skeleton_npy(tmp_path / "missing.npy")
        assert joints is None


# ---------------------------------------------------------------------------
# _scale_joints
# ---------------------------------------------------------------------------


class TestScaleJoints:
    """Test joint position scaling."""

    def test_scale_up(self) -> None:
        joints = [JointPosition(region_id=1, region_name="head", x=64.0, y=32.0)]
        scaled = _scale_joints(joints, (256, 256), 512)
        assert scaled[0].x == 128.0
        assert scaled[0].y == 64.0

    def test_scale_down(self) -> None:
        joints = [JointPosition(region_id=1, region_name="head", x=256.0, y=256.0)]
        scaled = _scale_joints(joints, (512, 512), 256)
        assert scaled[0].x == 128.0
        assert scaled[0].y == 128.0

    def test_out_of_bounds_not_visible(self) -> None:
        joints = [JointPosition(region_id=1, region_name="head", x=600.0, y=600.0)]
        scaled = _scale_joints(joints, (256, 256), 512)
        # 600 * (512/256) = 1200, which is > 512.
        assert scaled[0].visible is False

    def test_zero_size_original(self) -> None:
        joints = [JointPosition(region_id=1, region_name="head", x=100.0, y=100.0)]
        scaled = _scale_joints(joints, (0, 0), 512)
        # Should return unchanged.
        assert scaled[0].x == 100.0


# ---------------------------------------------------------------------------
# _joints_to_strata_json
# ---------------------------------------------------------------------------


class TestJointsToStrataJson:
    """Test Strata JSON format conversion."""

    def test_format(self) -> None:
        joints = [
            JointPosition(region_id=1, region_name="head", x=100.0, y=50.0),
            JointPosition(region_id=5, region_name="hips", x=100.0, y=128.0, visible=False),
        ]
        data = _joints_to_strata_json(joints)
        assert "joints" in data
        assert len(data["joints"]) == 2
        assert data["joints"][0]["region_id"] == 1
        assert data["joints"][0]["visible"] is True
        assert data["joints"][1]["visible"] is False


# ---------------------------------------------------------------------------
# _build_metadata
# ---------------------------------------------------------------------------


class TestBuildMetadata:
    """Test metadata generation."""

    def test_metadata_fields(self) -> None:
        frame = FrameData(
            frame_id="frame_0001",
            sequence_id="seq_001",
            split="train",
            frame_path=Path("/fake/frame.png"),
        )
        meta = _build_metadata(frame, 512, has_joints=True, has_flow=True)
        assert meta["source"] == "linkto_anime"
        assert meta["sequence_id"] == "seq_001"
        assert meta["has_joints"] is True
        assert meta["has_optical_flow"] is True
        assert meta["has_segmentation_mask"] is False

    def test_no_joints_no_flow(self) -> None:
        frame = FrameData(
            frame_id="f",
            sequence_id="s",
            split="test",
            frame_path=Path("/fake/f.png"),
        )
        meta = _build_metadata(frame, 256, has_joints=False, has_flow=False)
        assert meta["has_joints"] is False
        assert meta["has_optical_flow"] is False


# ---------------------------------------------------------------------------
# _resize_image
# ---------------------------------------------------------------------------


class TestResizeImage:
    """Test image resizing."""

    def test_resize(self) -> None:
        img = _create_test_image((128, 128))
        resized = _resize_image(img, 512)
        assert resized.size == (512, 512)

    def test_already_correct(self) -> None:
        img = _create_test_image((512, 512))
        resized = _resize_image(img, 512)
        assert resized.size == (512, 512)


# ---------------------------------------------------------------------------
# discover_sequences
# ---------------------------------------------------------------------------


class TestDiscoverSequences:
    """Test sequence discovery."""

    def test_discovers_splits(self, tmp_path: Path) -> None:
        root = _setup_sequence_dir(tmp_path, "train", "seq_A")
        # Add val sequence.
        val_seq = root / "val" / "seq_B" / "frames"
        val_seq.mkdir(parents=True)

        seqs = discover_sequences(root)
        assert len(seqs) == 2

    def test_skips_hidden(self, tmp_path: Path) -> None:
        root = _setup_sequence_dir(tmp_path, "train", "seq_A")
        hidden = root / "train" / ".hidden" / "frames"
        hidden.mkdir(parents=True)

        seqs = discover_sequences(root)
        assert len(seqs) == 1

    def test_empty(self, tmp_path: Path) -> None:
        root = tmp_path / "linkto_anime"
        root.mkdir()
        seqs = discover_sequences(root)
        assert seqs == []


# ---------------------------------------------------------------------------
# discover_frames
# ---------------------------------------------------------------------------


class TestDiscoverFrames:
    """Test frame discovery."""

    def test_discovers_frames(self, tmp_path: Path) -> None:
        root = _setup_sequence_dir(tmp_path, num_frames=5)
        seq_dir = root / "train" / "seq_001"

        frames = discover_frames(seq_dir, "train", "seq_001")
        assert len(frames) == 5
        assert all(f.sequence_id == "seq_001" for f in frames)

    def test_finds_skeleton_json(self, tmp_path: Path) -> None:
        root = _setup_sequence_dir(tmp_path, num_frames=2, with_skeleton=True)
        seq_dir = root / "train" / "seq_001"

        frames = discover_frames(seq_dir, "train", "seq_001")
        assert all(f.skeleton_path is not None for f in frames)

    def test_finds_flow_files(self, tmp_path: Path) -> None:
        root = _setup_sequence_dir(tmp_path, num_frames=2, with_flow=True)
        seq_dir = root / "train" / "seq_001"

        frames = discover_frames(seq_dir, "train", "seq_001")
        assert all(f.flow_fwd_path is not None for f in frames)
        assert all(f.flow_bwd_path is not None for f in frames)


# ---------------------------------------------------------------------------
# convert_frame
# ---------------------------------------------------------------------------


class TestConvertFrame:
    """Test single frame conversion."""

    def test_convert_basic_frame(self, tmp_path: Path) -> None:
        root = _setup_sequence_dir(tmp_path, num_frames=1)
        seq_dir = root / "train" / "seq_001"
        frames = discover_frames(seq_dir, "train", "seq_001")
        output_dir = tmp_path / "output"

        saved = convert_frame(frames[0], output_dir, resolution=64, include_flow=False)
        assert saved is True

        example_dir = output_dir / "linkto_anime_seq_001_frame_0000"
        assert (example_dir / "image.png").is_file()
        assert (example_dir / "metadata.json").is_file()

        img = Image.open(example_dir / "image.png")
        assert img.size == (64, 64)

    def test_convert_with_skeleton(self, tmp_path: Path) -> None:
        root = _setup_sequence_dir(tmp_path, num_frames=1, with_skeleton=True)
        seq_dir = root / "train" / "seq_001"
        frames = discover_frames(seq_dir, "train", "seq_001")
        output_dir = tmp_path / "output"

        convert_frame(frames[0], output_dir, resolution=64, include_flow=False)

        example_dir = output_dir / "linkto_anime_seq_001_frame_0000"
        assert (example_dir / "joints.json").is_file()

        joints_data = json.loads((example_dir / "joints.json").read_text(encoding="utf-8"))
        assert "joints" in joints_data
        assert len(joints_data["joints"]) > 0

    def test_convert_with_flow(self, tmp_path: Path) -> None:
        root = _setup_sequence_dir(tmp_path, num_frames=1, with_flow=True)
        seq_dir = root / "train" / "seq_001"
        frames = discover_frames(seq_dir, "train", "seq_001")
        output_dir = tmp_path / "output"

        convert_frame(frames[0], output_dir, resolution=64, include_flow=True)

        example_dir = output_dir / "linkto_anime_seq_001_frame_0000"
        assert (example_dir / "flow_fwd.npy").is_file()
        assert (example_dir / "flow_bwd.npy").is_file()

    def test_only_new_skips(self, tmp_path: Path) -> None:
        root = _setup_sequence_dir(tmp_path, num_frames=1)
        seq_dir = root / "train" / "seq_001"
        frames = discover_frames(seq_dir, "train", "seq_001")
        output_dir = tmp_path / "output"

        assert convert_frame(frames[0], output_dir, resolution=64, include_flow=False)
        assert not convert_frame(
            frames[0],
            output_dir,
            resolution=64,
            include_flow=False,
            only_new=True,
        )

    def test_invalid_frame_path(self, tmp_path: Path) -> None:
        frame = FrameData(
            frame_id="bad",
            sequence_id="seq",
            split="train",
            frame_path=tmp_path / "nonexistent.png",
        )
        assert not convert_frame(frame, tmp_path / "output", include_flow=False)


# ---------------------------------------------------------------------------
# convert_sequence
# ---------------------------------------------------------------------------


class TestConvertSequence:
    """Test sequence-level conversion."""

    def test_converts_all_frames(self, tmp_path: Path) -> None:
        root = _setup_sequence_dir(tmp_path, num_frames=4)
        seq_dir = root / "train" / "seq_001"
        output_dir = tmp_path / "output"

        result = convert_sequence(
            seq_dir,
            output_dir,
            "train",
            "seq_001",
            resolution=64,
            include_flow=False,
        )
        assert isinstance(result, AdapterResult)
        assert result.frames_saved == 4

    def test_max_frames(self, tmp_path: Path) -> None:
        root = _setup_sequence_dir(tmp_path, num_frames=10)
        seq_dir = root / "train" / "seq_001"
        output_dir = tmp_path / "output"

        result = convert_sequence(
            seq_dir,
            output_dir,
            "train",
            "seq_001",
            resolution=64,
            include_flow=False,
            max_frames=3,
        )
        assert result.frames_saved == 3


# ---------------------------------------------------------------------------
# convert_directory
# ---------------------------------------------------------------------------


class TestConvertDirectory:
    """Test batch directory conversion."""

    def test_converts_multiple_sequences(self, tmp_path: Path) -> None:
        root = _setup_sequence_dir(tmp_path, "train", "seq_A", num_frames=2)
        seq_b = root / "train" / "seq_B" / "frames"
        seq_b.mkdir(parents=True)
        _create_test_image().save(seq_b / "frame_0000.png")

        output_dir = tmp_path / "output"
        results = convert_directory(root, output_dir, resolution=64, include_flow=False)
        assert len(results) == 2
        total = sum(r.frames_saved for r in results)
        assert total == 3  # 2 + 1

    def test_max_sequences(self, tmp_path: Path) -> None:
        root = _setup_sequence_dir(tmp_path, "train", "seq_A", num_frames=1)
        seq_b = root / "train" / "seq_B" / "frames"
        seq_b.mkdir(parents=True)
        _create_test_image().save(seq_b / "frame_0000.png")

        output_dir = tmp_path / "output"
        results = convert_directory(
            root,
            output_dir,
            resolution=64,
            include_flow=False,
            max_sequences=1,
        )
        assert len(results) == 1

    def test_empty_directory(self, tmp_path: Path) -> None:
        root = tmp_path / "linkto_anime"
        root.mkdir()
        results = convert_directory(root, tmp_path / "output")
        assert results == []

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        results = convert_directory(tmp_path / "nonexistent", tmp_path / "output")
        assert results == []
