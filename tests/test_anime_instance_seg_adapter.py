"""Tests for anime_instance_seg_adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ingest.anime_instance_seg_adapter import (
    ANIME_INSTANCE_SOURCE,
    STRATA_RESOLUTION,
    ConversionStats,
    _decode_rle,
    _masks_to_fg,
    _resize_to_512,
    _save_example,
    convert_split,
)


# ---------------------------------------------------------------------------
# RLE decoding
# ---------------------------------------------------------------------------


def _rle_all_fg(h: int, w: int) -> dict:
    """Return a COCO RLE dict where all pixels are foreground."""
    # bg_run=0, fg_run=h*w — encode using COCO delta encoding
    # Verified empirically: '0' encodes bg=0, then encode fg total pixels
    n = h * w
    # Encode n as COCO delta from 0: delta=n, x=n<<1
    x = n << 1
    chars = []
    while True:
        c = x & 31
        x >>= 5
        if x > 0:
            c |= 32
        chars.append(chr(c + 48))
        if x == 0:
            break
    return {"size": [h, w], "counts": "0" + "".join(chars)}


def _rle_box(h: int, w: int, r0: int, r1: int, c0: int, c1: int) -> dict:
    """Return a COCO RLE dict with a rectangular box set to foreground.

    Uses column-major ordering (COCO convention).
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[r0:r1, c0:c1] = 1
    # Build run-length list in column-major order
    flat = mask.flatten(order="F")
    runs: list[int] = []
    cur = int(flat[0])
    run = 1
    for v in flat[1:]:
        if int(v) == cur:
            run += 1
        else:
            runs.append(run)
            run = 1
            cur = int(v)
    runs.append(run)
    if flat[0] == 1:
        runs.insert(0, 0)

    # COCO encodes delta-compressed run lengths
    def _encode_delta(vals: list[int]) -> str:
        chars = []
        prev = 0
        for v in vals:
            delta = v - prev
            prev = v
            if delta < 0:
                x = (~delta << 1) | 1
            else:
                x = delta << 1
            while True:
                c = x & 31
                x >>= 5
                if x > 0:
                    c |= 32
                chars.append(chr(c + 48))
                if x == 0:
                    break
        return "".join(chars)

    return {"size": [h, w], "counts": _encode_delta(runs)}


class TestDecodeRle:
    def test_all_foreground(self):
        """A mask with all pixels set decodes correctly."""
        h, w = 4, 4
        rle = _rle_all_fg(h, w)
        decoded = _decode_rle(rle)
        assert decoded.shape == (h, w)
        assert decoded.sum() == h * w

    def test_all_background(self):
        # bg=16, no fg run — encode delta for bg=16 then stop
        # bg=16: delta=16, x=32 -> c=0|32=32 -> chr(80)='P', x=1 -> c=1 -> chr(49)='1'
        rle = {"size": [4, 4], "counts": "P1"}
        decoded = _decode_rle(rle)
        assert decoded.sum() == 0

    def test_box_foreground(self):
        """A rectangular box in the center."""
        rle = _rle_box(16, 16, 4, 12, 4, 12)
        decoded = _decode_rle(rle)
        assert decoded.shape == (16, 16)
        assert decoded.sum() == 8 * 8  # 8×8 box

    def test_output_dtype(self):
        rle = _rle_all_fg(8, 8)
        decoded = _decode_rle(rle)
        assert decoded.dtype == np.uint8

    def test_large_mask(self):
        rle = _rle_all_fg(720, 720)
        decoded = _decode_rle(rle)
        assert decoded.sum() == 720 * 720


class TestMasksToFg:
    def _make_ann(self, mask: np.ndarray) -> dict:
        """Convert a binary mask to a COCO RLE annotation dict."""
        h, w = mask.shape
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        r0, r1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
        c0, c1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
        return {"id": 1, "segmentation": _rle_box(h, w, r0, r1, c0, c1)}

    def test_single_instance(self):
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[4:12, 4:12] = 1
        anns = [self._make_ann(mask)]
        fg = _masks_to_fg(anns, 16, 16)
        assert fg.max() == 255
        assert fg[4, 4] == 255
        assert fg[0, 0] == 0

    def test_multiple_instances_merged(self):
        mask1 = np.zeros((16, 16), dtype=np.uint8)
        mask1[:8, :8] = 1
        mask2 = np.zeros((16, 16), dtype=np.uint8)
        mask2[8:, 8:] = 1
        anns = [self._make_ann(mask1), self._make_ann(mask2)]
        fg = _masks_to_fg(anns, 16, 16)
        assert fg[0, 0] == 255
        assert fg[15, 15] == 255
        assert fg[0, 15] == 0
        assert fg[15, 0] == 0

    def test_empty_annotations(self):
        fg = _masks_to_fg([], 16, 16)
        assert fg.sum() == 0

    def test_missing_segmentation_skipped(self):
        anns = [{"id": 1, "segmentation": None}]
        fg = _masks_to_fg(anns, 16, 16)
        assert fg.sum() == 0

    def test_non_dict_segmentation_skipped(self):
        # Polygon-format segmentations are not supported — should skip gracefully
        anns = [{"id": 1, "segmentation": [[0, 0, 8, 0, 8, 8, 0, 8]]}]
        fg = _masks_to_fg(anns, 16, 16)
        assert fg.sum() == 0

    def test_output_is_uint8(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[1:7, 1:7] = 1
        fg = _masks_to_fg([self._make_ann(mask)], 8, 8)
        assert fg.dtype == np.uint8

    def test_values_are_0_or_255(self):
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        fg = _masks_to_fg([self._make_ann(mask)], 16, 16)
        unique = set(fg.flatten().tolist())
        assert unique <= {0, 255}


class TestResizeTo512:
    def test_already_512(self):
        img = Image.new("RGB", (512, 512))
        result = _resize_to_512(img)
        assert result.size == (512, 512)

    def test_upscale(self):
        img = Image.new("RGB", (256, 256))
        result = _resize_to_512(img)
        assert result.size == (512, 512)

    def test_downscale(self):
        img = Image.new("RGB", (720, 720))
        result = _resize_to_512(img)
        assert result.size == (512, 512)


class TestSaveExample:
    def test_creates_expected_files(self, tmp_path):
        image = Image.new("RGB", (512, 512), color=(128, 64, 32))
        fg_mask = np.zeros((512, 512), dtype=np.uint8)
        fg_mask[100:400, 100:400] = 255
        metadata = {"source": ANIME_INSTANCE_SOURCE, "example_id": "test_001"}

        out_dir = tmp_path / "test_001"
        _save_example(image, fg_mask, out_dir, metadata)

        assert (out_dir / "image.png").is_file()
        assert (out_dir / "segmentation.png").is_file()
        assert (out_dir / "metadata.json").is_file()

    def test_image_resized_to_512(self, tmp_path):
        image = Image.new("RGB", (720, 720))
        fg_mask = np.zeros((720, 720), dtype=np.uint8)
        _save_example(image, fg_mask, tmp_path / "ex", {"source": "test"})

        saved = Image.open(tmp_path / "ex" / "image.png")
        assert saved.size == (STRATA_RESOLUTION, STRATA_RESOLUTION)

    def test_mask_resized_to_512(self, tmp_path):
        image = Image.new("RGB", (720, 720))
        fg_mask = np.full((720, 720), 255, dtype=np.uint8)
        _save_example(image, fg_mask, tmp_path / "ex", {"source": "test"})

        saved = np.array(Image.open(tmp_path / "ex" / "segmentation.png"))
        assert saved.shape == (STRATA_RESOLUTION, STRATA_RESOLUTION)

    def test_metadata_written(self, tmp_path):
        image = Image.new("RGB", (512, 512))
        fg_mask = np.zeros((512, 512), dtype=np.uint8)
        meta = {"source": ANIME_INSTANCE_SOURCE, "example_id": "xyz", "foo": 42}
        _save_example(image, fg_mask, tmp_path / "ex", meta)

        with open(tmp_path / "ex" / "metadata.json") as f:
            loaded = json.load(f)
        assert loaded["source"] == ANIME_INSTANCE_SOURCE
        assert loaded["foo"] == 42


class TestConvertSplit:
    def _make_dataset(self, tmp_path: Path, n_images: int = 3) -> Path:
        """Build a minimal fake anime_instance_dataset directory."""
        dataset_dir = tmp_path / "anime_instance_dataset"
        train_dir = dataset_dir / "train"
        ann_dir = dataset_dir / "annotations"
        train_dir.mkdir(parents=True)
        ann_dir.mkdir(parents=True)

        images = []
        annotations = []
        ann_id = 0
        for i in range(n_images):
            fname = f"{i:012d}.jpg"
            img = Image.new("RGB", (720, 720), color=(i * 30, 100, 200))
            img.save(train_dir / fname)

            # Simple box mask
            rle = _rle_box(720, 720, 100, 300, 100, 300)

            images.append({"id": i, "file_name": fname, "height": 720, "width": 720})
            annotations.append({
                "id": ann_id,
                "image_id": i,
                "category_id": 0,
                "iscrowd": 0,
                "segmentation": rle,
                "area": 40000,
                "bbox": [100.0, 100.0, 200.0, 200.0],
                "tag_string": "",
                "tag_string_character": f"char_{i}",
            })
            ann_id += 1

        coco = {
            "info": {},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 0, "name": "object", "isthing": 1}],
        }
        with open(ann_dir / "det_train.json", "w") as f:
            json.dump(coco, f)

        return dataset_dir

    def test_basic_conversion(self, tmp_path):
        dataset_dir = self._make_dataset(tmp_path, n_images=3)
        out_dir = tmp_path / "output"

        stats = convert_split(dataset_dir, out_dir, split="train")

        assert stats.converted == 3
        assert stats.skipped == 0
        assert stats.errors == 0

    def test_output_structure(self, tmp_path):
        dataset_dir = self._make_dataset(tmp_path, n_images=2)
        out_dir = tmp_path / "output"
        convert_split(dataset_dir, out_dir, split="train")

        examples = list((out_dir / "train").iterdir())
        assert len(examples) == 2
        for ex in examples:
            assert (ex / "image.png").is_file()
            assert (ex / "segmentation.png").is_file()
            assert (ex / "metadata.json").is_file()

    def test_metadata_content(self, tmp_path):
        dataset_dir = self._make_dataset(tmp_path, n_images=1)
        out_dir = tmp_path / "output"
        convert_split(dataset_dir, out_dir, split="train")

        examples = list((out_dir / "train").iterdir())
        with open(examples[0] / "metadata.json") as f:
            meta = json.load(f)

        assert meta["source"] == ANIME_INSTANCE_SOURCE
        assert meta["split"] == "train"
        assert "character_names" in meta
        assert "instance_count" in meta
        assert meta["instance_count"] >= 1

    def test_max_examples(self, tmp_path):
        dataset_dir = self._make_dataset(tmp_path, n_images=5)
        out_dir = tmp_path / "output"
        stats = convert_split(dataset_dir, out_dir, split="train", max_examples=2)
        assert stats.converted == 2

    def test_empty_mask_skipped(self, tmp_path):
        """Images where all masks decode to zero should be skipped."""
        dataset_dir = tmp_path / "anime_instance_dataset"
        train_dir = dataset_dir / "train"
        ann_dir = dataset_dir / "annotations"
        train_dir.mkdir(parents=True)
        ann_dir.mkdir(parents=True)

        img = Image.new("RGB", (720, 720))
        img.save(train_dir / "000000000000.jpg")

        # Annotation with no segmentation
        coco = {
            "info": {}, "licenses": [],
            "images": [{"id": 0, "file_name": "000000000000.jpg",
                        "height": 720, "width": 720}],
            "annotations": [{"id": 0, "image_id": 0, "category_id": 0,
                              "iscrowd": 0, "segmentation": None, "area": 0,
                              "bbox": [0, 0, 0, 0], "tag_string": "",
                              "tag_string_character": ""}],
            "categories": [{"id": 0, "name": "object"}],
        }
        with open(ann_dir / "det_train.json", "w") as f:
            json.dump(coco, f)

        stats = convert_split(dataset_dir, tmp_path / "out", split="train")
        assert stats.converted == 0
        assert stats.skipped == 1

    def test_missing_annotation_file_raises(self, tmp_path):
        dataset_dir = tmp_path / "anime_instance_dataset"
        dataset_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            convert_split(dataset_dir, tmp_path / "out", split="train")
