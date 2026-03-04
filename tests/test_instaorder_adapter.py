"""Tests for the InstaOrder adapter.

These tests exercise the pure-Python adapter logic without requiring
the actual InstaOrder dataset or COCO images.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from ingest.instaorder_adapter import (
    AdapterResult,
    _build_metadata,
    _resize_draw_order,
    _resize_to_strata,
    build_draw_order_map,
    convert_directory,
    convert_image,
    parse_depth_pairs,
    topological_sort_instances,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_test_image(path: Path, size: tuple[int, int] = (256, 256)) -> None:
    """Create a test JPEG image."""
    arr = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    img.save(path)


def _make_polygon_segmentation(x: int, y: int, w: int, h: int) -> list[list[float]]:
    """Create a simple rectangular polygon segmentation."""
    return [
        [
            float(x),
            float(y),
            float(x + w),
            float(y),
            float(x + w),
            float(y + h),
            float(x),
            float(y + h),
        ]
    ]


def _setup_instaorder_dir(
    tmp_path: Path,
    *,
    split: str = "val",
    num_images: int = 3,
    image_size: tuple[int, int] = (256, 256),
    include_images: bool = True,
    add_cyclic: bool = False,
) -> Path:
    """Create a fake InstaOrder dataset directory.

    Layout::

        dataset_dir/
        ├── annotations/
        │   ├── InstaOrder_{split}2017.json
        │   └── instances_{split}2017.json
        └── images/
            └── {split}2017/
                └── {id:012d}.jpg
    """
    dataset_dir = tmp_path / "instaorder"
    ann_dir = dataset_dir / "annotations"
    img_dir = dataset_dir / "images" / f"{split}2017"
    ann_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)

    w, h = image_size

    # Build COCO instances + InstaOrder annotations
    coco_images = []
    coco_annotations = []
    instaorder_annotations = []

    ann_id = 1
    for i in range(num_images):
        image_id = i + 1
        filename = f"{image_id:012d}.jpg"

        coco_images.append(
            {
                "id": image_id,
                "file_name": filename,
                "width": w,
                "height": h,
            }
        )

        if include_images:
            _create_test_image(img_dir / filename, size=image_size)

        # Create 3 instances per image with non-overlapping boxes
        instance_ids = []
        box_h = h // 3
        for j in range(3):
            coco_ann = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [10, j * box_h, w - 20, box_h - 5],
                "area": (w - 20) * (box_h - 5),
                "iscrowd": 0,
                "segmentation": _make_polygon_segmentation(10, j * box_h, w - 20, box_h - 5),
            }
            coco_annotations.append(coco_ann)
            instance_ids.append(ann_id)
            ann_id += 1

        # InstaOrder annotation: instance 0 in front of 1, 1 in front of 2
        if add_cyclic and i == num_images - 1:
            # Add cyclic ordering for last image
            depth = [
                {"order": "0 < 1", "overlap": True, "count": 1},
                {"order": "1 < 2", "overlap": True, "count": 1},
                {"order": "2 < 0", "overlap": True, "count": 1},
            ]
        else:
            depth = [
                {"order": "0 < 1", "overlap": True, "count": 1},
                {"order": "1 < 2", "overlap": True, "count": 1},
            ]

        instaorder_annotations.append(
            {
                "image_id": image_id,
                "instance_ids": instance_ids,
                "depth": depth,
                "occlusion": [],
            }
        )

    # Write COCO instances JSON
    coco_data = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": 1, "name": "person"}],
    }
    (ann_dir / f"instances_{split}2017.json").write_text(json.dumps(coco_data), encoding="utf-8")

    # Write InstaOrder JSON
    instaorder_data = {"annotations": instaorder_annotations}
    (ann_dir / f"InstaOrder_{split}2017.json").write_text(
        json.dumps(instaorder_data), encoding="utf-8"
    )

    return dataset_dir


# ---------------------------------------------------------------------------
# parse_depth_pairs
# ---------------------------------------------------------------------------


class TestParseDepthPairs:
    """Test depth pair extraction from InstaOrder annotations."""

    def test_basic_pairs(self) -> None:
        anns = [
            {
                "instance_ids": [101, 102, 103],
                "depth": [
                    {"order": "0 < 1", "overlap": True, "count": 1},
                    {"order": "1 < 2", "overlap": True, "count": 1},
                ],
            }
        ]
        pairs = parse_depth_pairs(anns)
        assert (101, 102) in pairs
        assert (102, 103) in pairs

    def test_equal_depth_skipped(self) -> None:
        anns = [
            {
                "instance_ids": [101, 102],
                "depth": [
                    {"order": "0 = 1", "overlap": False, "count": 1},
                ],
            }
        ]
        pairs = parse_depth_pairs(anns)
        assert len(pairs) == 0

    def test_empty_depth(self) -> None:
        anns = [{"instance_ids": [101], "depth": []}]
        assert parse_depth_pairs(anns) == []

    def test_invalid_index_skipped(self) -> None:
        anns = [
            {
                "instance_ids": [101],
                "depth": [{"order": "0 < 5", "overlap": True, "count": 1}],
            }
        ]
        pairs = parse_depth_pairs(anns)
        assert len(pairs) == 0


# ---------------------------------------------------------------------------
# topological_sort_instances
# ---------------------------------------------------------------------------


class TestTopologicalSort:
    """Test topological sorting of instance orderings."""

    def test_linear_order(self) -> None:
        ids = [1, 2, 3]
        # 1 in front of 2, 2 in front of 3
        pairs = [(1, 2), (2, 3)]
        result = topological_sort_instances(ids, pairs)
        assert result is not None
        # Back to front: 3, 2, 1
        assert result == [3, 2, 1]

    def test_single_instance(self) -> None:
        result = topological_sort_instances([42], [])
        assert result == [42]

    def test_empty(self) -> None:
        result = topological_sort_instances([], [])
        assert result == []

    def test_cycle_returns_none(self) -> None:
        ids = [1, 2, 3]
        pairs = [(1, 2), (2, 3), (3, 1)]
        result = topological_sort_instances(ids, pairs)
        assert result is None

    def test_unrelated_instances(self) -> None:
        """Instances with no depth pairs get included in some order."""
        ids = [1, 2, 3]
        pairs = [(1, 2)]
        result = topological_sort_instances(ids, pairs)
        assert result is not None
        assert set(result) == {1, 2, 3}
        # 2 should come before 1 in back-to-front order
        assert result.index(2) < result.index(1)


# ---------------------------------------------------------------------------
# build_draw_order_map
# ---------------------------------------------------------------------------


class TestBuildDrawOrderMap:
    """Test draw order map generation."""

    def test_two_instances(self) -> None:
        # Two non-overlapping instances: back (id=1) and front (id=2)
        sorted_ids = [1, 2]  # back to front
        coco_anns = {
            1: {
                "id": 1,
                "segmentation": _make_polygon_segmentation(0, 0, 50, 50),
            },
            2: {
                "id": 2,
                "segmentation": _make_polygon_segmentation(60, 60, 50, 50),
            },
        }
        result = build_draw_order_map(sorted_ids, coco_anns, 128, 128)
        assert result.shape == (128, 128)

        # Back instance should have low values, front should have 255
        back_region = result[0:50, 0:50]
        front_region = result[60:110, 60:110]
        assert back_region.max() > 0  # has content
        assert front_region.max() == 255

    def test_empty_ids(self) -> None:
        result = build_draw_order_map([], {}, 64, 64)
        assert result.shape == (64, 64)
        assert result.max() == 0


# ---------------------------------------------------------------------------
# _resize_to_strata / _resize_draw_order
# ---------------------------------------------------------------------------


class TestResize:
    """Test image and draw order map resizing."""

    def test_resize_preserves_aspect(self) -> None:
        arr = np.random.randint(0, 255, (512, 256, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        result = _resize_to_strata(img, 512)
        assert result.size == (512, 512)
        assert result.mode == "RGBA"

    def test_resize_draw_order_nearest(self) -> None:
        draw_order = np.array([[0, 127], [255, 0]], dtype=np.uint8)
        result = _resize_draw_order(draw_order, 512)
        assert result.size == (512, 512)
        assert result.mode == "L"
        # Nearest-neighbor should preserve exact values
        arr = np.array(result)
        unique = set(np.unique(arr))
        assert unique.issubset({0, 127, 255})


# ---------------------------------------------------------------------------
# _build_metadata
# ---------------------------------------------------------------------------


class TestBuildMetadata:
    """Test metadata generation."""

    def test_required_fields(self) -> None:
        meta = _build_metadata(
            "instaorder_val_000000000001",
            1,
            "val",
            512,
            original_size=(640, 480),
            n_instances=5,
            n_depth_pairs=8,
        )
        assert meta["id"] == "instaorder_val_000000000001"
        assert meta["source"] == "instaorder"
        assert meta["has_draw_order"] is True
        assert meta["has_segmentation_mask"] is False
        assert meta["has_joints"] is False
        assert meta["has_fg_mask"] is False
        assert meta["n_instances"] == 5
        assert meta["n_depth_pairs"] == 8

    def test_missing_annotations_listed(self) -> None:
        meta = _build_metadata(
            "instaorder_val_x",
            1,
            "val",
            512,
            original_size=(256, 256),
            n_instances=2,
            n_depth_pairs=1,
        )
        assert "strata_segmentation" in meta["missing_annotations"]
        assert "joints" in meta["missing_annotations"]
        assert "fg_mask" in meta["missing_annotations"]


# ---------------------------------------------------------------------------
# convert_image
# ---------------------------------------------------------------------------


class TestConvertImage:
    """Test single-image conversion."""

    def test_creates_output_structure(self, tmp_path: Path) -> None:
        dataset = _setup_instaorder_dir(tmp_path, num_images=1)
        output_dir = tmp_path / "output"

        # Load annotations
        ann_dir = dataset / "annotations"
        instaorder = json.loads((ann_dir / "InstaOrder_val2017.json").read_text())
        coco = json.loads((ann_dir / "instances_val2017.json").read_text())

        image_id = 1
        instaorder_anns = [a for a in instaorder["annotations"] if a["image_id"] == image_id]
        coco_anns = [a for a in coco["annotations"] if a["image_id"] == image_id]
        image_info = next(i for i in coco["images"] if i["id"] == image_id)

        image_dir = dataset / "images" / "val2017"
        saved = convert_image(
            image_id,
            instaorder_anns,
            coco_anns,
            image_info,
            image_dir,
            output_dir,
            "val",
        )

        assert saved is True
        example_dir = output_dir / "instaorder_val_000000000001"
        assert (example_dir / "image.png").is_file()
        assert (example_dir / "draw_order.png").is_file()
        assert (example_dir / "metadata.json").is_file()

    def test_returns_none_when_image_missing(self, tmp_path: Path) -> None:
        dataset = _setup_instaorder_dir(tmp_path, num_images=1, include_images=False)
        output_dir = tmp_path / "output"

        ann_dir = dataset / "annotations"
        instaorder = json.loads((ann_dir / "InstaOrder_val2017.json").read_text())
        coco = json.loads((ann_dir / "instances_val2017.json").read_text())

        image_id = 1
        instaorder_anns = [a for a in instaorder["annotations"] if a["image_id"] == image_id]
        coco_anns = [a for a in coco["annotations"] if a["image_id"] == image_id]
        image_info = next(i for i in coco["images"] if i["id"] == image_id)

        image_dir = dataset / "images" / "val2017"
        result = convert_image(
            image_id,
            instaorder_anns,
            coco_anns,
            image_info,
            image_dir,
            output_dir,
            "val",
        )
        assert result is None

    def test_only_new_skips_existing(self, tmp_path: Path) -> None:
        dataset = _setup_instaorder_dir(tmp_path, num_images=1)
        output_dir = tmp_path / "output"

        ann_dir = dataset / "annotations"
        instaorder = json.loads((ann_dir / "InstaOrder_val2017.json").read_text())
        coco = json.loads((ann_dir / "instances_val2017.json").read_text())

        image_id = 1
        instaorder_anns = [a for a in instaorder["annotations"] if a["image_id"] == image_id]
        coco_anns = [a for a in coco["annotations"] if a["image_id"] == image_id]
        image_info = next(i for i in coco["images"] if i["id"] == image_id)

        image_dir = dataset / "images" / "val2017"
        assert (
            convert_image(
                image_id,
                instaorder_anns,
                coco_anns,
                image_info,
                image_dir,
                output_dir,
                "val",
            )
            is True
        )
        assert (
            convert_image(
                image_id,
                instaorder_anns,
                coco_anns,
                image_info,
                image_dir,
                output_dir,
                "val",
                only_new=True,
            )
            is False
        )


# ---------------------------------------------------------------------------
# convert_directory
# ---------------------------------------------------------------------------


class TestConvertDirectory:
    """Test batch directory conversion."""

    def test_converts_all_images(self, tmp_path: Path) -> None:
        dataset = _setup_instaorder_dir(tmp_path, num_images=3)
        output_dir = tmp_path / "output"

        result = convert_directory(dataset, output_dir, split="val")
        assert isinstance(result, AdapterResult)
        assert result.images_processed == 3

    def test_max_images_limits_output(self, tmp_path: Path) -> None:
        dataset = _setup_instaorder_dir(tmp_path, num_images=5)
        output_dir = tmp_path / "output"

        result = convert_directory(dataset, output_dir, split="val", max_images=2)
        assert result.images_processed == 2

    def test_only_new_skips(self, tmp_path: Path) -> None:
        dataset = _setup_instaorder_dir(tmp_path, num_images=2)
        output_dir = tmp_path / "output"

        r1 = convert_directory(dataset, output_dir, split="val")
        assert r1.images_processed == 2

        r2 = convert_directory(dataset, output_dir, split="val", only_new=True)
        assert r2.images_processed == 0

    def test_missing_annotation_dir(self, tmp_path: Path) -> None:
        result = convert_directory(tmp_path / "nope", tmp_path / "output")
        assert result.images_processed == 0

    def test_cyclic_images_counted(self, tmp_path: Path) -> None:
        dataset = _setup_instaorder_dir(tmp_path, num_images=2, add_cyclic=True)
        output_dir = tmp_path / "output"
        result = convert_directory(dataset, output_dir, split="val")
        # Last image has cyclic ordering, should be counted
        assert result.images_cyclic >= 1
