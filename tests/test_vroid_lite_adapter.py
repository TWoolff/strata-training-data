"""Tests for the VRoid-Lite adapter.

These tests exercise the pure-Python adapter logic without requiring
Blender or the actual VRoid-Lite dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from ingest.vroid_lite_adapter import (
    AdapterResult,
    VroidLiteEntry,
    _build_metadata,
    _make_example_id,
    _resize_to_strata,
    convert_directory,
    convert_entry,
    discover_entries,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_ROW = {
    "file_name": "4ade1940-333a-4a89-bff8-281d3ebf0912.png",
    "vrm_name": "vrm_yomox9_take",
    "clip_name": "050_0060",
    "camera_profile": "upper-body-mid-lens",
    "facial_expression": "Joy",
    "lighting": "directional",
    "lighting_color": {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0},
    "outline": "bold",
    "shade_toony": 0.0,
    "skin_profile": "natural-skin",
    "looking_label": "looking-at-viewer",
    "camera_position": {"x": 0.506, "y": 2.234, "z": 0.339},
    "camera_rotation": {"x": 31.456, "y": 229.544, "z": 359.591},
    "camera_fov": 45.0,
}


def _create_test_image(
    size: tuple[int, int] = (1536, 1024),
    mode: str = "RGBA",
) -> Image.Image:
    """Create a test image matching VRoid-Lite's default dimensions."""
    channels = 4 if mode == "RGBA" else 3
    arr = np.random.randint(0, 255, (size[1], size[0], channels), dtype=np.uint8)
    return Image.fromarray(arr, mode=mode)


def _setup_vroid_dir(
    tmp_path: Path,
    *,
    num_entries: int = 3,
    image_size: tuple[int, int] = (1536, 1024),
    extra_malformed: int = 0,
    missing_images: int = 0,
) -> Path:
    """Create a fake VRoid-Lite dataset directory.

    Mimics the real layout::

        input_dir/
        ├── metadata.jsonl
        └── vroid_dataset/
            ├── {uuid}.png
            └── …
    """
    input_dir = tmp_path / "vroid_lite"
    image_dir = input_dir / "vroid_dataset"
    image_dir.mkdir(parents=True)

    lines: list[str] = []
    for i in range(num_entries):
        uuid = f"{i:08x}-0000-0000-0000-000000000000"
        filename = f"{uuid}.png"

        img = _create_test_image(image_size, mode="RGBA")
        img.save(image_dir / filename)

        row = {
            "file_name": filename,
            "vrm_name": f"vrm_char{i % 3}",
            "clip_name": f"clip_{i:03d}",
            "camera_profile": "full-body",
            "facial_expression": "Neutral",
            "lighting": "directional",
        }
        lines.append(json.dumps(row))

    # Add malformed lines.
    for _ in range(extra_malformed):
        lines.append("THIS IS NOT JSON {{{")

    # Add entries with missing images.
    for j in range(missing_images):
        row = {
            "file_name": f"missing_{j:04d}.png",
            "vrm_name": "vrm_ghost",
            "clip_name": "clip_missing",
        }
        lines.append(json.dumps(row))

    jsonl_path = input_dir / "metadata.jsonl"
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return input_dir


# ---------------------------------------------------------------------------
# discover_entries
# ---------------------------------------------------------------------------


class TestDiscoverEntries:
    """Test JSONL-driven entry discovery."""

    def test_discovers_all_entries(self, tmp_path: Path) -> None:
        input_dir = _setup_vroid_dir(tmp_path, num_entries=5)
        entries = discover_entries(input_dir)
        assert len(entries) == 5

    def test_sorted_by_filename(self, tmp_path: Path) -> None:
        input_dir = _setup_vroid_dir(tmp_path, num_entries=5)
        entries = discover_entries(input_dir)
        names = [e.image_path.name for e in entries]
        assert names == sorted(names)

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        input_dir = _setup_vroid_dir(tmp_path, num_entries=3, extra_malformed=2)
        entries = discover_entries(input_dir)
        assert len(entries) == 3

    def test_skips_missing_images(self, tmp_path: Path) -> None:
        input_dir = _setup_vroid_dir(tmp_path, num_entries=3, missing_images=2)
        entries = discover_entries(input_dir)
        assert len(entries) == 3

    def test_missing_jsonl(self, tmp_path: Path) -> None:
        empty = tmp_path / "no_jsonl"
        empty.mkdir()
        assert discover_entries(empty) == []

    def test_empty_jsonl(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "empty_jsonl"
        input_dir.mkdir()
        (input_dir / "metadata.jsonl").write_text("", encoding="utf-8")
        assert discover_entries(input_dir) == []

    def test_entry_has_metadata_and_path(self, tmp_path: Path) -> None:
        input_dir = _setup_vroid_dir(tmp_path, num_entries=1)
        entries = discover_entries(input_dir)
        assert len(entries) == 1
        assert isinstance(entries[0].metadata, dict)
        assert entries[0].image_path.is_file()
        assert "vrm_name" in entries[0].metadata


# ---------------------------------------------------------------------------
# _make_example_id
# ---------------------------------------------------------------------------


class TestMakeExampleId:
    """Test example ID generation."""

    def test_format(self) -> None:
        entry = VroidLiteEntry(
            metadata={"vrm_name": "vrm_yomox9_take"},
            image_path=Path("4ade1940-333a-4a89-bff8-281d3ebf0912.png"),
        )
        assert _make_example_id(entry) == "vroid_lite_vrm_yomox9_take_4ade1940"

    def test_missing_vrm_name(self) -> None:
        entry = VroidLiteEntry(
            metadata={},
            image_path=Path("abcd1234-0000-0000-0000-000000000000.png"),
        )
        assert _make_example_id(entry) == "vroid_lite_unknown_abcd1234"

    def test_uuid_first_segment_only(self) -> None:
        entry = VroidLiteEntry(
            metadata={"vrm_name": "vrm_test"},
            image_path=Path("deadbeef-1111-2222-3333-444444444444.png"),
        )
        example_id = _make_example_id(entry)
        assert "deadbeef" in example_id
        assert "1111" not in example_id


# ---------------------------------------------------------------------------
# _resize_to_strata
# ---------------------------------------------------------------------------


class TestResizeToStrata:
    """Test aspect-ratio-preserving resize with padding."""

    def test_landscape_image_padded_vertically(self) -> None:
        """1536×1024 (landscape) → 512×512 with vertical padding."""
        img = _create_test_image((1536, 1024))
        result = _resize_to_strata(img, 512)
        assert result.size == (512, 512)
        assert result.mode == "RGBA"

    def test_square_image_no_padding(self) -> None:
        img = _create_test_image((512, 512))
        result = _resize_to_strata(img, 512)
        assert result.size == (512, 512)

    def test_already_correct_size(self) -> None:
        img = _create_test_image((512, 512), mode="RGBA")
        result = _resize_to_strata(img, 512)
        assert result.size == (512, 512)

    def test_converts_rgb_to_rgba(self) -> None:
        img = _create_test_image((1536, 1024), mode="RGB")
        result = _resize_to_strata(img, 512)
        assert result.mode == "RGBA"

    def test_transparent_padding(self) -> None:
        """Padding pixels should be fully transparent."""
        img = _create_test_image((1536, 1024))
        result = _resize_to_strata(img, 512)
        arr = np.array(result)
        # Top-left corner should be padding for a landscape image
        assert arr[0, 0, 3] == 0

    def test_custom_resolution(self) -> None:
        img = _create_test_image((1536, 1024))
        result = _resize_to_strata(img, 256)
        assert result.size == (256, 256)


# ---------------------------------------------------------------------------
# _build_metadata
# ---------------------------------------------------------------------------


class TestBuildMetadata:
    """Test metadata generation."""

    def _make_entry(self) -> VroidLiteEntry:
        return VroidLiteEntry(
            metadata=_SAMPLE_ROW,
            image_path=Path("4ade1940-333a-4a89-bff8-281d3ebf0912.png"),
        )

    def test_required_strata_fields(self) -> None:
        entry = self._make_entry()
        meta = _build_metadata(
            "vroid_lite_vrm_yomox9_take_4ade1940",
            entry,
            512,
            original_size=(1536, 1024),
        )
        assert meta["id"] == "vroid_lite_vrm_yomox9_take_4ade1940"
        assert meta["source"] == "vroid_lite"
        assert meta["resolution"] == 512

    def test_missing_annotations(self) -> None:
        entry = self._make_entry()
        meta = _build_metadata(
            "test",
            entry,
            512,
            original_size=(1536, 1024),
        )
        assert meta["has_segmentation_mask"] is False
        assert meta["has_fg_mask"] is False
        assert meta["has_joints"] is False
        assert meta["has_draw_order"] is False
        assert "strata_segmentation" in meta["missing_annotations"]

    def test_character_field(self) -> None:
        entry = self._make_entry()
        meta = _build_metadata(
            "test",
            entry,
            512,
            original_size=(1536, 1024),
        )
        assert meta["character"] == "vrm_yomox9_take"

    def test_vroid_lite_metadata_nested(self) -> None:
        entry = self._make_entry()
        meta = _build_metadata(
            "test",
            entry,
            512,
            original_size=(1536, 1024),
        )
        vroid_meta = meta["vroid_lite_metadata"]
        assert vroid_meta["vrm_name"] == "vrm_yomox9_take"
        assert vroid_meta["camera_profile"] == "upper-body-mid-lens"
        assert vroid_meta["facial_expression"] == "Joy"
        # file_name should be excluded from nested metadata.
        assert "file_name" not in vroid_meta

    def test_original_size_recorded(self) -> None:
        entry = self._make_entry()
        meta = _build_metadata(
            "test",
            entry,
            512,
            original_size=(1536, 1024),
        )
        assert meta["original_width"] == 1536
        assert meta["original_height"] == 1024

    def test_padding_flag_nonsquare(self) -> None:
        entry = self._make_entry()
        meta = _build_metadata(
            "test",
            entry,
            512,
            original_size=(1536, 1024),
        )
        assert meta["padding_applied"] is True

    def test_padding_flag_square(self) -> None:
        entry = self._make_entry()
        meta = _build_metadata(
            "test",
            entry,
            512,
            original_size=(512, 512),
        )
        assert meta["padding_applied"] is False


# ---------------------------------------------------------------------------
# convert_entry
# ---------------------------------------------------------------------------


class TestConvertEntry:
    """Test single-entry conversion."""

    def test_creates_output_structure(self, tmp_path: Path) -> None:
        input_dir = _setup_vroid_dir(tmp_path / "input", num_entries=1)
        entries = discover_entries(input_dir)
        output_dir = tmp_path / "output"

        saved = convert_entry(entries[0], output_dir)
        assert saved is True

        example_id = _make_example_id(entries[0])
        example_dir = output_dir / example_id
        assert (example_dir / "image.png").is_file()
        assert (example_dir / "metadata.json").is_file()

    def test_output_is_square(self, tmp_path: Path) -> None:
        input_dir = _setup_vroid_dir(tmp_path / "input", num_entries=1)
        entries = discover_entries(input_dir)
        output_dir = tmp_path / "output"

        convert_entry(entries[0], output_dir)

        example_id = _make_example_id(entries[0])
        img = Image.open(output_dir / example_id / "image.png")
        assert img.size == (512, 512)

    def test_output_is_rgba(self, tmp_path: Path) -> None:
        input_dir = _setup_vroid_dir(tmp_path / "input", num_entries=1)
        entries = discover_entries(input_dir)
        output_dir = tmp_path / "output"

        convert_entry(entries[0], output_dir)

        example_id = _make_example_id(entries[0])
        img = Image.open(output_dir / example_id / "image.png")
        assert img.mode == "RGBA"

    def test_only_new_skips_existing(self, tmp_path: Path) -> None:
        input_dir = _setup_vroid_dir(tmp_path / "input", num_entries=1)
        entries = discover_entries(input_dir)
        output_dir = tmp_path / "output"

        assert convert_entry(entries[0], output_dir) is True
        assert convert_entry(entries[0], output_dir, only_new=True) is False

    def test_metadata_source(self, tmp_path: Path) -> None:
        input_dir = _setup_vroid_dir(tmp_path / "input", num_entries=1)
        entries = discover_entries(input_dir)
        output_dir = tmp_path / "output"

        convert_entry(entries[0], output_dir)

        example_id = _make_example_id(entries[0])
        meta = json.loads((output_dir / example_id / "metadata.json").read_text())
        assert meta["source"] == "vroid_lite"

    def test_metadata_has_character(self, tmp_path: Path) -> None:
        input_dir = _setup_vroid_dir(tmp_path / "input", num_entries=1)
        entries = discover_entries(input_dir)
        output_dir = tmp_path / "output"

        convert_entry(entries[0], output_dir)

        example_id = _make_example_id(entries[0])
        meta = json.loads((output_dir / example_id / "metadata.json").read_text())
        assert "character" in meta

    def test_custom_resolution(self, tmp_path: Path) -> None:
        input_dir = _setup_vroid_dir(tmp_path / "input", num_entries=1)
        entries = discover_entries(input_dir)
        output_dir = tmp_path / "output"

        convert_entry(entries[0], output_dir, resolution=256)

        example_id = _make_example_id(entries[0])
        img = Image.open(output_dir / example_id / "image.png")
        assert img.size == (256, 256)


# ---------------------------------------------------------------------------
# convert_directory
# ---------------------------------------------------------------------------


class TestConvertDirectory:
    """Test batch directory conversion."""

    def test_converts_all_entries(self, tmp_path: Path) -> None:
        input_dir = _setup_vroid_dir(tmp_path / "input", num_entries=5)
        output_dir = tmp_path / "output"

        result = convert_directory(input_dir, output_dir)
        assert isinstance(result, AdapterResult)
        assert result.images_processed == 5

    def test_max_images_limits_output(self, tmp_path: Path) -> None:
        input_dir = _setup_vroid_dir(tmp_path / "input", num_entries=10)
        output_dir = tmp_path / "output"

        result = convert_directory(input_dir, output_dir, max_images=3)
        assert result.images_processed == 3

    def test_random_sample_reproducible(self, tmp_path: Path) -> None:
        input_dir = _setup_vroid_dir(tmp_path / "input", num_entries=10)
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"

        convert_directory(input_dir, out1, max_images=3, random_sample=True, seed=42)
        convert_directory(input_dir, out2, max_images=3, random_sample=True, seed=42)

        dirs1 = sorted(d.name for d in out1.iterdir() if d.is_dir())
        dirs2 = sorted(d.name for d in out2.iterdir() if d.is_dir())
        assert dirs1 == dirs2

    def test_random_sample_different_seed(self, tmp_path: Path) -> None:
        input_dir = _setup_vroid_dir(tmp_path / "input", num_entries=10)
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"

        convert_directory(input_dir, out1, max_images=3, random_sample=True, seed=1)
        convert_directory(input_dir, out2, max_images=3, random_sample=True, seed=99)

        dirs1 = sorted(d.name for d in out1.iterdir() if d.is_dir())
        dirs2 = sorted(d.name for d in out2.iterdir() if d.is_dir())
        assert dirs1 != dirs2

    def test_only_new_skips(self, tmp_path: Path) -> None:
        input_dir = _setup_vroid_dir(tmp_path / "input", num_entries=3)
        output_dir = tmp_path / "output"

        r1 = convert_directory(input_dir, output_dir)
        assert r1.images_processed == 3

        r2 = convert_directory(input_dir, output_dir, only_new=True)
        assert r2.images_processed == 0
        assert r2.images_skipped == 3

    def test_empty_jsonl(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "empty"
        input_dir.mkdir()
        (input_dir / "metadata.jsonl").write_text("", encoding="utf-8")
        result = convert_directory(input_dir, tmp_path / "output")
        assert result.images_processed == 0

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        result = convert_directory(tmp_path / "nope", tmp_path / "output")
        assert result.images_processed == 0

    def test_handles_malformed_and_missing(self, tmp_path: Path) -> None:
        """Malformed lines and missing images are skipped gracefully."""
        input_dir = _setup_vroid_dir(
            tmp_path / "input",
            num_entries=3,
            extra_malformed=2,
            missing_images=1,
        )
        output_dir = tmp_path / "output"
        result = convert_directory(input_dir, output_dir)
        # Only the 3 valid entries should be processed.
        assert result.images_processed == 3
