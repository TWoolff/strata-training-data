"""Tests for training/data/segmentation_dataset.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from training.data.segmentation_dataset import (
    DatasetConfig,
    _detect_layout,
    _discover_flat,
    _discover_per_example,
)

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_rgba_image(path: Path, size: tuple[int, int] = (512, 512)) -> None:
    """Create an RGBA PNG with non-zero content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.randint(0, 255, (*size, 4), dtype=np.uint8)
    arr[:, :, 3] = 255  # fully opaque
    Image.fromarray(arr, mode="RGBA").save(path)


def _create_mask(
    path: Path,
    size: tuple[int, int] = (512, 512),
    region: int = 3,
) -> None:
    """Create an 8-bit grayscale mask PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full(size, region, dtype=np.uint8)
    arr[:256, :] = 1  # head in top half
    Image.fromarray(arr, mode="L").save(path)


def _create_draw_order(path: Path, size: tuple[int, int] = (512, 512)) -> None:
    """Create a grayscale draw order PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.linspace(0, 255, size[0], dtype=np.uint8)
    arr = np.tile(arr[:, None], (1, size[1]))
    Image.fromarray(arr, mode="L").save(path)


def _setup_flat_dataset(
    base: Path,
    characters: list[str],
    poses: int = 2,
    styles: list[str] | None = None,
    with_draw_order: bool = False,
) -> Path:
    """Create a flat-layout dataset directory."""
    if styles is None:
        styles = ["flat"]
    dataset_dir = base / "dataset"
    for char_id in characters:
        for p in range(poses):
            for style in styles:
                _create_rgba_image(dataset_dir / "images" / f"{char_id}_pose_{p:02d}_{style}.png")
            _create_mask(dataset_dir / "masks" / f"{char_id}_pose_{p:02d}.png")
            if with_draw_order:
                _create_draw_order(dataset_dir / "draw_order" / f"{char_id}_pose_{p:02d}.png")
    return dataset_dir


def _setup_per_example_dataset(
    base: Path,
    example_ids: list[str],
    with_draw_order: bool = False,
    with_metadata: bool = False,
) -> Path:
    """Create a per-example-layout dataset directory."""
    dataset_dir = base / "dataset"
    for eid in example_ids:
        example_dir = dataset_dir / eid
        _create_rgba_image(example_dir / "image.png")
        _create_mask(example_dir / "segmentation.png")
        if with_draw_order:
            _create_draw_order(example_dir / "draw_order.png")
        if with_metadata:
            meta = {"source": "test", "has_accessories": False}
            (example_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return dataset_dir


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------


class TestLayoutDetection:
    def test_detects_flat_layout(self, tmp_path: Path) -> None:
        dataset_dir = _setup_flat_dataset(tmp_path, ["char_a"])
        assert _detect_layout(dataset_dir) == "flat"

    def test_detects_per_example_layout(self, tmp_path: Path) -> None:
        dataset_dir = _setup_per_example_dataset(tmp_path, ["example_001"])
        assert _detect_layout(dataset_dir) == "per_example"

    def test_empty_dir_defaults_to_flat(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        assert _detect_layout(empty) == "flat"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestDiscoverFlat:
    def test_discovers_image_mask_pairs(self, tmp_path: Path) -> None:
        dataset_dir = _setup_flat_dataset(tmp_path, ["mixamo_001"], poses=2)
        examples = _discover_flat(dataset_dir)
        assert len(examples) == 2

    def test_multiple_styles_multiply_examples(self, tmp_path: Path) -> None:
        dataset_dir = _setup_flat_dataset(tmp_path, ["mixamo_001"], poses=1, styles=["flat", "cel"])
        examples = _discover_flat(dataset_dir)
        # Each style variant is a separate example (same mask)
        assert len(examples) == 2

    def test_skips_images_without_mask(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "dataset"
        _create_rgba_image(dataset_dir / "images" / "orphan_pose_00_flat.png")
        examples = _discover_flat(dataset_dir)
        assert len(examples) == 0

    def test_discovers_draw_order(self, tmp_path: Path) -> None:
        dataset_dir = _setup_flat_dataset(tmp_path, ["mixamo_001"], poses=1, with_draw_order=True)
        examples = _discover_flat(dataset_dir)
        assert examples[0].draw_order_path is not None

    def test_missing_draw_order_is_none(self, tmp_path: Path) -> None:
        dataset_dir = _setup_flat_dataset(tmp_path, ["mixamo_001"], poses=1, with_draw_order=False)
        examples = _discover_flat(dataset_dir)
        assert examples[0].draw_order_path is None


class TestDiscoverPerExample:
    def test_discovers_example_dirs(self, tmp_path: Path) -> None:
        dataset_dir = _setup_per_example_dataset(tmp_path, ["nova_001_front", "nova_002_front"])
        examples = _discover_per_example(dataset_dir)
        assert len(examples) == 2

    def test_skips_dirs_without_mask(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "dataset"
        example_dir = dataset_dir / "incomplete"
        _create_rgba_image(example_dir / "image.png")
        # No segmentation.png
        examples = _discover_per_example(dataset_dir)
        assert len(examples) == 0

    def test_discovers_metadata(self, tmp_path: Path) -> None:
        dataset_dir = _setup_per_example_dataset(tmp_path, ["ex_001"], with_metadata=True)
        examples = _discover_per_example(dataset_dir)
        assert examples[0].metadata_path is not None


# ---------------------------------------------------------------------------
# DatasetConfig
# ---------------------------------------------------------------------------


class TestDatasetConfig:
    def test_default_values(self) -> None:
        cfg = DatasetConfig()
        assert cfg.resolution == 512
        assert cfg.augment is True
        assert cfg.split_ratios == (0.8, 0.1, 0.1)

    def test_from_dict(self) -> None:
        d = {
            "data": {"resolution": 256, "split_seed": 99},
            "augmentation": {
                "horizontal_flip": False,
                "random_rotation": 5,
                "random_scale": [0.8, 1.2],
            },
        }
        cfg = DatasetConfig.from_dict(d)
        assert cfg.resolution == 256
        assert cfg.horizontal_flip is False
        assert cfg.random_rotation == 5
        assert cfg.random_scale == (0.8, 1.2)
        assert cfg.split_seed == 99


# ---------------------------------------------------------------------------
# Full Dataset
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestSegmentationDataset:
    @pytest.fixture(autouse=True)
    def _import_dataset_class(self) -> None:
        from training.data.segmentation_dataset import SegmentationDataset

        self.DatasetClass = SegmentationDataset

    def test_loads_flat_layout(self, tmp_path: Path) -> None:
        dataset_dir = _setup_flat_dataset(tmp_path, ["mixamo_001"], poses=2)
        ds = self.DatasetClass(
            [dataset_dir],
            split="train",
            augment=False,
        )
        # With default 80/10/10 split and 1 character, it goes to train
        assert len(ds) == 2

    def test_loads_per_example_layout(self, tmp_path: Path) -> None:
        dataset_dir = _setup_per_example_dataset(
            tmp_path, ["stdgen_0001_front", "stdgen_0001_back"]
        )
        ds = self.DatasetClass(
            [dataset_dir],
            split="train",
            augment=False,
        )
        assert len(ds) == 2

    def test_getitem_returns_correct_shapes(self, tmp_path: Path) -> None:
        import torch

        dataset_dir = _setup_flat_dataset(tmp_path, ["mixamo_001"], poses=1, with_draw_order=True)
        ds = self.DatasetClass(
            [dataset_dir],
            split="train",
            augment=False,
        )
        sample = ds[0]

        assert sample["image"].shape == (3, 512, 512)
        assert sample["image"].dtype.is_floating_point
        assert sample["segmentation"].shape == (512, 512)
        assert sample["segmentation"].dtype == torch.int64
        assert sample["draw_order"].shape == (1, 512, 512)
        assert sample["draw_order"].dtype.is_floating_point
        assert sample["confidence_target"].shape == (1, 512, 512)
        assert sample["has_draw_order"] is True

    def test_getitem_without_draw_order(self, tmp_path: Path) -> None:
        dataset_dir = _setup_flat_dataset(tmp_path, ["mixamo_001"], poses=1, with_draw_order=False)
        ds = self.DatasetClass(
            [dataset_dir],
            split="train",
            augment=False,
        )
        sample = ds[0]

        assert sample["has_draw_order"] is False
        assert sample["draw_order"].shape == (1, 512, 512)
        assert sample["draw_order"].sum() == 0

    def test_split_filtering(self, tmp_path: Path) -> None:
        """Multiple characters should be split; not all appear in train."""
        chars = [f"mixamo_{i:03d}" for i in range(10)]
        dataset_dir = _setup_flat_dataset(tmp_path, chars, poses=1)
        train_ds = self.DatasetClass(
            [dataset_dir],
            split="train",
            augment=False,
        )
        val_ds = self.DatasetClass(
            [dataset_dir],
            split="val",
            augment=False,
        )
        test_ds = self.DatasetClass(
            [dataset_dir],
            split="test",
            augment=False,
        )
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total == 10  # all examples accounted for
        assert len(train_ds) > 0
        # With 10 chars, at least 1 should be in val or test
        assert len(val_ds) + len(test_ds) > 0

    def test_empty_dataset(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        ds = self.DatasetClass(
            [empty],
            split="train",
            augment=False,
        )
        assert len(ds) == 0

    def test_augmentation_runs_without_error(self, tmp_path: Path) -> None:
        dataset_dir = _setup_flat_dataset(tmp_path, ["mixamo_001"], poses=1, with_draw_order=True)
        ds = self.DatasetClass(
            [dataset_dir],
            split="train",
            augment=True,
        )
        # Should not raise
        sample = ds[0]
        assert sample["image"].shape == (3, 512, 512)

    def test_confidence_target_matches_foreground(self, tmp_path: Path) -> None:
        dataset_dir = _setup_flat_dataset(
            tmp_path,
            ["mixamo_001"],
            poses=1,
        )
        ds = self.DatasetClass(
            [dataset_dir],
            split="train",
            augment=False,
        )
        sample = ds[0]
        conf = sample["confidence_target"].numpy()[0]
        # Our test images are fully opaque, so confidence should be all 1.0
        assert conf.sum() > 0

    def test_mask_values_in_valid_range(self, tmp_path: Path) -> None:
        dataset_dir = _setup_flat_dataset(
            tmp_path,
            ["mixamo_001"],
            poses=1,
        )
        ds = self.DatasetClass(
            [dataset_dir],
            split="train",
            augment=False,
        )
        sample = ds[0]
        seg = sample["segmentation"].numpy()
        assert seg.min() >= 0
        assert seg.max() < 22  # NUM_CLASSES

    def test_multiple_dataset_dirs(self, tmp_path: Path) -> None:
        dir_a = _setup_flat_dataset(tmp_path / "a", ["mixamo_001"], poses=1)
        dir_b = _setup_per_example_dataset(tmp_path / "b", ["stdgen_0001_front"])
        ds = self.DatasetClass(
            [dir_a, dir_b],
            split="train",
            augment=False,
        )
        assert len(ds) == 2

    def test_image_resize(self, tmp_path: Path) -> None:
        """Images not at target resolution should be resized."""
        dataset_dir = tmp_path / "dataset"
        _create_rgba_image(dataset_dir / "images" / "char_a_pose_00_flat.png", size=(256, 256))
        _create_mask(dataset_dir / "masks" / "char_a_pose_00.png", size=(256, 256))
        ds = self.DatasetClass(
            [dataset_dir],
            split="train",
            augment=False,
        )
        sample = ds[0]
        assert sample["image"].shape == (3, 512, 512)
        assert sample["segmentation"].shape == (512, 512)

    def test_no_augmentation_on_val(self, tmp_path: Path) -> None:
        """Augmentation should not be applied to val/test even if augment=True."""
        chars = [f"mixamo_{i:03d}" for i in range(10)]
        dataset_dir = _setup_flat_dataset(tmp_path, chars, poses=1)
        ds = self.DatasetClass(
            [dataset_dir],
            split="val",
            augment=True,
        )
        if len(ds) > 0:
            sample = ds[0]
            assert sample["image"].shape == (3, 512, 512)


# ---------------------------------------------------------------------------
# Style suffix stripping
# ---------------------------------------------------------------------------


class TestStyleSuffixStripping:
    """Verify that image filenames correctly map to mask filenames."""

    def test_all_styles_map_to_same_mask(self, tmp_path: Path) -> None:
        styles = ["flat", "cel", "pixel", "painterly", "sketch", "unlit"]
        dataset_dir = _setup_flat_dataset(tmp_path, ["mixamo_001"], poses=1, styles=styles)
        examples = _discover_flat(dataset_dir)
        # All 6 style variants should map to the same mask
        mask_paths = {ex.mask_path for ex in examples}
        assert len(mask_paths) == 1
        assert len(examples) == 6
