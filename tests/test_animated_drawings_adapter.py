"""Tests for the Meta Animated Drawings ingest adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from ingest.animated_drawings_adapter import (
    _COCO_TO_STRATA_DIRECT,
    _STRATA_ID_TO_NAME,
    _SYNTHETIC_REGION_IDS,
    AdapterResult,
    _build_metadata,
    _crop_and_resize,
    convert_directory,
    convert_example,
    load_annotations,
    map_coco_to_strata_joints,
    rasterize_polygon_mask,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal COCO dataset structure for testing.
_COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def _make_test_image(w: int = 400, h: int = 600) -> Image.Image:
    """Create a simple RGB test image."""
    arr = np.full((h, w, 3), 200, dtype=np.uint8)
    # Draw a simple figure shape in the center
    cy, cx = h // 2, w // 2
    arr[cy - 50 : cy + 50, cx - 30 : cx + 30] = [100, 150, 200]
    return Image.fromarray(arr, "RGB")


def _make_coco_annotation(
    image_id: int = 0,
    bbox: list[float] | None = None,
    keypoints: list[float] | None = None,
    segmentation: list[list[float]] | None = None,
) -> dict:
    """Build a minimal COCO annotation dict."""
    if bbox is None:
        bbox = [100.0, 100.0, 200.0, 300.0]
    if keypoints is None:
        # 17 joints all visible, arranged in a rough humanoid shape.
        keypoints = _make_humanoid_keypoints(bbox)
    if segmentation is None:
        # Simple rectangle polygon covering the bbox.
        x, y, w, h = bbox
        segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]
    return {
        "id": image_id,
        "image_id": image_id,
        "category_id": 1,
        "segmentation": segmentation,
        "area": bbox[2] * bbox[3],
        "bbox": bbox,
        "iscrowd": 0,
        "keypoints": keypoints,
        "num_keypoints": 17,
    }


def _make_humanoid_keypoints(bbox: list[float]) -> list[float]:
    """Generate 17 COCO keypoints roughly arranged in a humanoid pose within bbox."""
    x, y, w, h = bbox
    cx = x + w / 2
    # Distribute joints from top to bottom.
    kps = [
        (cx, y + h * 0.05, 2),  # 0: nose
        (cx - w * 0.05, y + h * 0.03, 2),  # 1: left_eye
        (cx + w * 0.05, y + h * 0.03, 2),  # 2: right_eye
        (cx - w * 0.1, y + h * 0.05, 2),  # 3: left_ear
        (cx + w * 0.1, y + h * 0.05, 2),  # 4: right_ear
        (cx - w * 0.2, y + h * 0.2, 2),  # 5: left_shoulder
        (cx + w * 0.2, y + h * 0.2, 2),  # 6: right_shoulder
        (cx - w * 0.3, y + h * 0.35, 2),  # 7: left_elbow
        (cx + w * 0.3, y + h * 0.35, 2),  # 8: right_elbow
        (cx - w * 0.35, y + h * 0.5, 2),  # 9: left_wrist
        (cx + w * 0.35, y + h * 0.5, 2),  # 10: right_wrist
        (cx - w * 0.1, y + h * 0.55, 2),  # 11: left_hip
        (cx + w * 0.1, y + h * 0.55, 2),  # 12: right_hip
        (cx - w * 0.1, y + h * 0.7, 2),  # 13: left_knee
        (cx + w * 0.1, y + h * 0.7, 2),  # 14: right_knee
        (cx - w * 0.1, y + h * 0.9, 2),  # 15: left_ankle
        (cx + w * 0.1, y + h * 0.9, 2),  # 16: right_ankle
    ]
    flat = []
    for x_kp, y_kp, v in kps:
        flat.extend([x_kp, y_kp, v])
    return flat


def _make_coco_image_info(
    image_id: int = 0,
    filename: str = "amateur_drawings/0/test_image.png",
    width: int = 400,
    height: int = 600,
) -> dict:
    """Build a minimal COCO image info dict."""
    return {
        "id": image_id,
        "width": width,
        "height": height,
        "file_name": filename,
        "license": 1,
    }


def _write_coco_json(
    path: Path,
    images: list[dict],
    annotations: list[dict],
) -> None:
    """Write a minimal COCO annotations JSON file."""
    data = {
        "info": {"description": "Test dataset"},
        "categories": [
            {
                "supercategory": "drawing",
                "id": 1,
                "name": "human",
                "keypoints": _COCO_KEYPOINT_NAMES,
                "skeleton": [],
            }
        ],
        "images": images,
        "annotations": annotations,
        "licenses": [{"url": "", "id": 1, "name": "MIT"}],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def _setup_test_dataset(tmp_path: Path, n_images: int = 3) -> Path:
    """Create a minimal Animated Drawings dataset on disk.

    Returns the input directory path.
    """
    input_dir = tmp_path / "dataset"
    img_dir = input_dir / "amateur_drawings" / "0"
    img_dir.mkdir(parents=True)

    images = []
    annotations = []
    for i in range(n_images):
        filename = f"amateur_drawings/0/img_{i:04d}.png"
        img = _make_test_image()
        img.save(input_dir / filename)
        img_info = _make_coco_image_info(image_id=i, filename=filename, width=400, height=600)
        ann = _make_coco_annotation(image_id=i)
        images.append(img_info)
        annotations.append(ann)

    _write_coco_json(input_dir / "amateur_drawings_annotations.json", images, annotations)
    return input_dir


# ---------------------------------------------------------------------------
# Polygon rasterization tests
# ---------------------------------------------------------------------------


class TestRasterizePolygon:
    def test_rectangle_polygon(self) -> None:
        # Simple rectangle: (10,10) → (90,10) → (90,90) → (10,90).
        polygons = [[10, 10, 90, 10, 90, 90, 10, 90]]
        mask = rasterize_polygon_mask(polygons, 100, 100)
        assert mask.shape == (100, 100)
        assert mask.dtype == np.uint8
        # Interior should be 255.
        assert mask[50, 50] == 255
        # Exterior should be 0.
        assert mask[0, 0] == 0

    def test_empty_polygon(self) -> None:
        mask = rasterize_polygon_mask([], 100, 100)
        assert mask.max() == 0

    def test_multiple_polygons(self) -> None:
        poly1 = [10, 10, 40, 10, 40, 40, 10, 40]
        poly2 = [60, 60, 90, 60, 90, 90, 60, 90]
        mask = rasterize_polygon_mask([poly1, poly2], 100, 100)
        assert mask[25, 25] == 255
        assert mask[75, 75] == 255
        assert mask[50, 50] == 0


# ---------------------------------------------------------------------------
# Crop and resize tests
# ---------------------------------------------------------------------------


class TestCropAndResize:
    def test_output_size(self) -> None:
        img = _make_test_image(400, 600)
        canvas, _scale, _x0, _y0, _off_x, _off_y = _crop_and_resize(img, [100, 100, 200, 300], 512)
        assert canvas.size == (512, 512)

    def test_output_is_rgba(self) -> None:
        img = _make_test_image(400, 600)
        canvas, _scale, _x0, _y0, _off_x, _off_y = _crop_and_resize(img, [100, 100, 200, 300], 512)
        assert canvas.mode == "RGBA"

    def test_scale_is_positive(self) -> None:
        img = _make_test_image(400, 600)
        _canvas, scale, _x0, _y0, _off_x, _off_y = _crop_and_resize(img, [100, 100, 200, 300], 512)
        assert scale > 0


# ---------------------------------------------------------------------------
# Joint mapping tests
# ---------------------------------------------------------------------------


class TestJointMapping:
    def _make_joints(self, bbox: list[float] | None = None) -> list[dict]:
        """Helper to create joints from default humanoid keypoints."""
        if bbox is None:
            bbox = [100.0, 100.0, 200.0, 300.0]
        kps = _make_humanoid_keypoints(bbox)
        img = _make_test_image(400, 600)
        _, scale, x0, y0, off_x, off_y = _crop_and_resize(img, bbox, 512)
        return map_coco_to_strata_joints(kps, bbox, scale, x0, y0, off_x, off_y, 512)

    def test_returns_19_joints(self) -> None:
        joints = self._make_joints()
        assert len(joints) == 19

    def test_joint_ids_are_1_to_19(self) -> None:
        joints = self._make_joints()
        ids = [j["id"] for j in joints]
        assert ids == list(range(1, 20))

    def test_all_joints_have_required_keys(self) -> None:
        joints = self._make_joints()
        for j in joints:
            assert "id" in j
            assert "name" in j
            assert "x" in j
            assert "y" in j
            assert "visible" in j
            assert "synthetic" in j

    def test_synthetic_flag_on_interpolated_joints(self) -> None:
        joints = self._make_joints()
        for j in joints:
            if j["id"] in _SYNTHETIC_REGION_IDS:
                assert j["synthetic"] is True, f"Joint {j['name']} should be synthetic"

    def test_direct_joints_not_synthetic(self) -> None:
        joints = self._make_joints()
        direct_strata_ids = set(_COCO_TO_STRATA_DIRECT.values())
        for j in joints:
            if j["id"] in direct_strata_ids:
                assert j["synthetic"] is False, f"Joint {j['name']} should not be synthetic"

    def test_neck_is_midpoint_of_shoulders(self) -> None:
        bbox = [100.0, 100.0, 200.0, 300.0]
        kps = _make_humanoid_keypoints(bbox)
        img = _make_test_image(400, 600)
        _, scale, x0, y0, off_x, off_y = _crop_and_resize(img, bbox, 512)
        joints = map_coco_to_strata_joints(kps, bbox, scale, x0, y0, off_x, off_y, 512)

        neck = next(j for j in joints if j["id"] == 2)
        shoulder_l = next(j for j in joints if j["id"] == 6)
        shoulder_r = next(j for j in joints if j["id"] == 10)

        expected_x = (shoulder_l["x"] + shoulder_r["x"]) / 2
        expected_y = (shoulder_l["y"] + shoulder_r["y"]) / 2
        assert abs(neck["x"] - expected_x) < 0.1
        assert abs(neck["y"] - expected_y) < 0.1

    def test_region_names_match_strata(self) -> None:
        joints = self._make_joints()
        for j in joints:
            assert j["name"] == _STRATA_ID_TO_NAME[j["id"]]


# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_metadata_fields(self) -> None:
        img_info = _make_coco_image_info()
        ann = _make_coco_annotation()
        meta = _build_metadata("test_000", img_info, ann, 512)

        assert meta["id"] == "test_000"
        assert meta["source"] == "animated_drawings"
        assert meta["resolution"] == 512
        assert meta["has_fg_mask"] is True
        assert meta["has_joints"] is True
        assert meta["has_segmentation_mask"] is False
        assert meta["has_draw_order"] is False
        assert "strata_segmentation" in meta["missing_annotations"]
        assert "draw_order" in meta["missing_annotations"]


# ---------------------------------------------------------------------------
# Load annotations tests
# ---------------------------------------------------------------------------


class TestLoadAnnotations:
    def test_loads_and_indexes(self, tmp_path: Path) -> None:
        images = [_make_coco_image_info(image_id=0), _make_coco_image_info(image_id=1)]
        anns = [_make_coco_annotation(image_id=0), _make_coco_annotation(image_id=1)]
        ann_path = tmp_path / "annotations.json"
        _write_coco_json(ann_path, images, anns)

        images_by_id, anns_by_id = load_annotations(ann_path)
        assert len(images_by_id) == 2
        assert len(anns_by_id) == 2
        assert 0 in images_by_id
        assert 1 in anns_by_id


# ---------------------------------------------------------------------------
# Single example conversion tests
# ---------------------------------------------------------------------------


class TestConvertExample:
    def test_saves_all_files(self, tmp_path: Path) -> None:
        img = _make_test_image()
        src = tmp_path / "src" / "image.png"
        src.parent.mkdir()
        img.save(src)

        img_info = _make_coco_image_info(width=400, height=600)
        ann = _make_coco_annotation()
        out = tmp_path / "output"

        result = convert_example(src, img_info, ann, out, example_id="animated_drawings_000000")
        assert result is True

        example_dir = out / "animated_drawings_000000"
        assert (example_dir / "image.png").exists()
        assert (example_dir / "segmentation.png").exists()
        assert (example_dir / "joints.json").exists()
        assert (example_dir / "metadata.json").exists()

    def test_output_image_is_512x512(self, tmp_path: Path) -> None:
        img = _make_test_image()
        src = tmp_path / "src" / "image.png"
        src.parent.mkdir()
        img.save(src)

        img_info = _make_coco_image_info(width=400, height=600)
        ann = _make_coco_annotation()
        out = tmp_path / "output"

        convert_example(src, img_info, ann, out, example_id="test_000")

        output_img = Image.open(out / "test_000" / "image.png")
        assert output_img.size == (512, 512)

    def test_mask_is_binary(self, tmp_path: Path) -> None:
        img = _make_test_image()
        src = tmp_path / "src" / "image.png"
        src.parent.mkdir()
        img.save(src)

        img_info = _make_coco_image_info(width=400, height=600)
        ann = _make_coco_annotation()
        out = tmp_path / "output"

        convert_example(src, img_info, ann, out, example_id="test_000")

        mask = Image.open(out / "test_000" / "segmentation.png")
        unique = set(np.unique(np.array(mask)))
        assert unique <= {0, 255}

    def test_joints_json_has_19_entries(self, tmp_path: Path) -> None:
        img = _make_test_image()
        src = tmp_path / "src" / "image.png"
        src.parent.mkdir()
        img.save(src)

        img_info = _make_coco_image_info(width=400, height=600)
        ann = _make_coco_annotation()
        out = tmp_path / "output"

        convert_example(src, img_info, ann, out, example_id="test_000")

        joints = json.loads((out / "test_000" / "joints.json").read_text())
        assert len(joints) == 19

    def test_only_new_skips_existing(self, tmp_path: Path) -> None:
        img = _make_test_image()
        src = tmp_path / "src" / "image.png"
        src.parent.mkdir()
        img.save(src)

        img_info = _make_coco_image_info(width=400, height=600)
        ann = _make_coco_annotation()
        out = tmp_path / "output"

        convert_example(src, img_info, ann, out, example_id="test_000")
        result = convert_example(src, img_info, ann, out, example_id="test_000", only_new=True)
        assert result is False


# ---------------------------------------------------------------------------
# Directory conversion tests
# ---------------------------------------------------------------------------


class TestConvertDirectory:
    def test_processes_all_images(self, tmp_path: Path) -> None:
        input_dir = _setup_test_dataset(tmp_path, n_images=3)
        out = tmp_path / "output"

        result = convert_directory(input_dir, out)

        assert isinstance(result, AdapterResult)
        assert result.images_processed == 3
        assert result.images_skipped == 0
        assert len(result.errors) == 0

    def test_max_images_limit(self, tmp_path: Path) -> None:
        input_dir = _setup_test_dataset(tmp_path, n_images=5)
        out = tmp_path / "output"

        result = convert_directory(input_dir, out, max_images=2)
        assert result.images_processed == 2

    def test_random_sampling(self, tmp_path: Path) -> None:
        input_dir = _setup_test_dataset(tmp_path, n_images=10)
        out = tmp_path / "output"

        result = convert_directory(input_dir, out, max_images=3, random_sample=True, seed=42)
        assert result.images_processed == 3

    def test_missing_annotations_file(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "empty"
        input_dir.mkdir()
        out = tmp_path / "output"

        result = convert_directory(input_dir, out)
        assert result.images_processed == 0
        assert len(result.errors) == 1

    def test_output_directory_structure(self, tmp_path: Path) -> None:
        input_dir = _setup_test_dataset(tmp_path, n_images=1)
        out = tmp_path / "output"

        convert_directory(input_dir, out)

        # Should have one example directory.
        example_dirs = list(out.iterdir())
        assert len(example_dirs) == 1
        example_dir = example_dirs[0]

        # Each example has 4 files.
        assert (example_dir / "image.png").exists()
        assert (example_dir / "segmentation.png").exists()
        assert (example_dir / "joints.json").exists()
        assert (example_dir / "metadata.json").exists()
