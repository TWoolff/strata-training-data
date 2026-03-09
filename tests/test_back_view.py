"""Tests for back view generation model, dataset, and pair extraction."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import torch

    from training.models.back_view_model import BackViewModel

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from PIL import Image

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


# ---------------------------------------------------------------------------
# BackViewModel
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestBackViewModel:
    """U-Net forward pass shape checks."""

    def test_forward_shape(self):
        model = BackViewModel(in_channels=8, out_channels=4)
        x = torch.randn(1, 8, 512, 512)
        out = model(x)
        assert "output" in out
        assert out["output"].shape == (1, 4, 512, 512)

    def test_forward_batch(self):
        model = BackViewModel()
        x = torch.randn(2, 8, 512, 512)
        out = model(x)
        assert out["output"].shape == (2, 4, 512, 512)

    def test_output_range(self):
        """Sigmoid output should be in [0, 1]."""
        model = BackViewModel()
        x = torch.randn(1, 8, 512, 512)
        out = model(x)["output"]
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_custom_channels(self):
        model = BackViewModel(in_channels=6, out_channels=3)
        x = torch.randn(1, 6, 512, 512)
        out = model(x)
        assert out["output"].shape == (1, 3, 512, 512)


# ---------------------------------------------------------------------------
# Pair extraction
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_PIL, reason="PIL not installed")
class TestPairExtraction:
    """Test prepare_back_view_pairs scanning and extraction."""

    @pytest.fixture()
    def flat_dataset(self, tmp_path: Path) -> Path:
        """Create a minimal flat-layout dataset."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        # Character with all 3 required angles
        char = "rigged_test-char"
        for angle_suffix in ["", "_three_quarter", "_back"]:
            fname = f"{char}_pose_00{angle_suffix}_textured.png"
            img = Image.new("RGBA", (512, 512), (128, 64, 32, 255))
            img.save(images_dir / fname)

        # Character missing back angle
        char2 = "rigged_incomplete"
        for angle_suffix in ["", "_three_quarter"]:
            fname = f"{char2}_pose_00{angle_suffix}_textured.png"
            img = Image.new("RGBA", (512, 512), (64, 128, 32, 255))
            img.save(images_dir / fname)

        return tmp_path

    @pytest.fixture()
    def per_example_dataset(self, tmp_path: Path) -> Path:
        """Create a minimal per-example-layout dataset."""
        char = "Meshy_AI_TestChar_0308"

        for angle in ["front", "three_quarter", "back"]:
            d = tmp_path / f"{char}_texture_{angle}"
            d.mkdir()
            img = Image.new("RGBA", (512, 512), (128, 64, 32, 255))
            img.save(d / "image.png")

        return tmp_path

    def test_detect_flat_layout(self, flat_dataset: Path):
        from training.data.prepare_back_view_pairs import _detect_layout

        assert _detect_layout(flat_dataset) == "flat"

    def test_detect_per_example_layout(self, per_example_dataset: Path):
        from training.data.prepare_back_view_pairs import _detect_layout

        assert _detect_layout(per_example_dataset) == "per_example"

    def test_scan_flat_layout(self, flat_dataset: Path):
        from training.data.prepare_back_view_pairs import _scan_flat_layout

        groups = _scan_flat_layout(flat_dataset, style_filter="textured")
        # Should find 2 groups (one complete, one incomplete)
        assert len(groups) == 2
        # Complete group should have all 3 angles
        complete = groups["rigged_test-char_pose_00"]
        assert set(complete.keys()) == {"front", "three_quarter", "back"}

    def test_scan_per_example_layout(self, per_example_dataset: Path):
        from training.data.prepare_back_view_pairs import _scan_per_example_layout

        groups = _scan_per_example_layout(per_example_dataset)
        assert len(groups) == 1
        key = next(iter(groups.keys()))
        assert set(groups[key].keys()) == {"front", "three_quarter", "back"}

    def test_extract_pairs_flat(self, flat_dataset: Path, tmp_path: Path):
        from training.data.prepare_back_view_pairs import extract_pairs

        output_dir = tmp_path / "pairs_out"
        count = extract_pairs(flat_dataset, output_dir, style_filter="textured")
        assert count == 1  # Only 1 complete triplet
        pair_dir = output_dir / "pair_00000"
        assert (pair_dir / "front.png").exists()
        assert (pair_dir / "three_quarter.png").exists()
        assert (pair_dir / "back.png").exists()

    def test_extract_pairs_per_example(self, per_example_dataset: Path, tmp_path: Path):
        from training.data.prepare_back_view_pairs import extract_pairs

        output_dir = tmp_path / "pairs_out"
        count = extract_pairs(per_example_dataset, output_dir)
        assert count == 1
        pair_dir = output_dir / "pair_00000"
        assert (pair_dir / "front.png").exists()
        assert (pair_dir / "three_quarter.png").exists()
        assert (pair_dir / "back.png").exists()

    def test_extract_pairs_resize(self, tmp_path: Path):
        """Verify images are resized to target resolution."""
        from training.data.prepare_back_view_pairs import extract_pairs

        # Create oversized images
        images_dir = tmp_path / "src" / "images"
        images_dir.mkdir(parents=True)
        for angle_suffix in ["", "_three_quarter", "_back"]:
            fname = f"char_pose_00{angle_suffix}_flat.png"
            img = Image.new("RGBA", (1024, 1024), (128, 64, 32, 255))
            img.save(images_dir / fname)

        output_dir = tmp_path / "out"
        extract_pairs(tmp_path / "src", output_dir, style_filter="flat", resolution=512)

        result = Image.open(output_dir / "pair_00000" / "front.png")
        assert result.size == (512, 512)


# ---------------------------------------------------------------------------
# BackViewDataset
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCH or not _HAS_PIL, reason="torch/PIL not installed")
class TestBackViewDataset:
    """Dataset loader tests."""

    @pytest.fixture()
    def paired_dataset(self, tmp_path: Path) -> Path:
        """Create paired directories for dataset loading."""
        for i in range(5):
            pair_dir = tmp_path / f"pair_{i:05d}"
            pair_dir.mkdir()
            for name in ("front", "three_quarter", "back"):
                img = Image.new("RGBA", (512, 512), (128, 64, 32, 255))
                img.save(pair_dir / f"{name}.png")
        return tmp_path

    def test_dataset_loading(self, paired_dataset: Path):
        from training.data.back_view_dataset import BackViewDataset, BackViewDatasetConfig

        config = BackViewDatasetConfig(
            dataset_dirs=[paired_dataset],
            split="train",
        )
        ds = BackViewDataset(config)
        assert len(ds) > 0

    def test_sample_shapes(self, paired_dataset: Path):
        from training.data.back_view_dataset import BackViewDataset, BackViewDatasetConfig

        config = BackViewDatasetConfig(
            dataset_dirs=[paired_dataset],
            split="train",
            horizontal_flip=False,
            color_jitter={},
        )
        ds = BackViewDataset(config)
        sample = ds[0]
        assert sample["image"].shape == (8, 512, 512)
        assert sample["target"].shape == (4, 512, 512)

    def test_sample_range(self, paired_dataset: Path):
        from training.data.back_view_dataset import BackViewDataset, BackViewDatasetConfig

        config = BackViewDatasetConfig(
            dataset_dirs=[paired_dataset],
            split="train",
            horizontal_flip=False,
            color_jitter={},
        )
        ds = BackViewDataset(config)
        sample = ds[0]
        assert sample["image"].min() >= 0.0
        assert sample["image"].max() <= 1.0

    def test_splits_cover_all(self, paired_dataset: Path):
        from training.data.back_view_dataset import BackViewDataset, BackViewDatasetConfig

        total = 0
        for split in ("train", "val", "test"):
            config = BackViewDatasetConfig(
                dataset_dirs=[paired_dataset],
                split=split,
            )
            ds = BackViewDataset(config)
            total += len(ds)
        assert total == 5


# ---------------------------------------------------------------------------
# Training components
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestBackViewTraining:
    """Tests for training script components."""

    def test_palette_consistency_loss(self):
        from training.train_back_view import palette_consistency_loss

        pred = torch.rand(2, 4, 64, 64)
        tq = torch.rand(2, 4, 64, 64)
        loss = palette_consistency_loss(pred, tq, bins=8)
        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_palette_loss_transparent(self):
        """Palette loss handles fully transparent images."""
        from training.train_back_view import palette_consistency_loss

        pred = torch.zeros(1, 4, 64, 64)  # fully transparent
        tq = torch.zeros(1, 4, 64, 64)
        loss = palette_consistency_loss(pred, tq)
        assert loss.item() == 0.0

    def test_perceptual_loss(self):
        from training.train_back_view import PerceptualLoss

        device = torch.device("cpu")
        ploss = PerceptualLoss(device)
        pred = torch.rand(1, 4, 64, 64)
        target = torch.rand(1, 4, 64, 64)
        loss = ploss(pred, target)
        assert loss.shape == ()
        assert loss.item() > 0.0

    def test_color_histogram(self):
        from training.train_back_view import _color_histogram

        rgb = torch.rand(2, 3, 32, 32)
        alpha = torch.ones(2, 1, 32, 32)
        hist = _color_histogram(rgb, alpha, bins=8)
        assert hist.shape == (2, 24)
        # Each sample's histogram should sum to ~3.0 (3 channels, each normalized to 1.0)
        assert torch.allclose(hist.sum(dim=1), torch.tensor([3.0, 3.0]), atol=0.01)


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestBackViewONNXExport:
    """Tests for ONNX export wrapper and registry."""

    def test_wrapper_output_shape(self):
        from training.export_onnx import BackViewWrapper
        from training.models.back_view_model import BackViewModel

        model = BackViewModel(in_channels=8, out_channels=4)
        wrapper = BackViewWrapper(model)
        x = torch.randn(1, 8, 512, 512)
        result = wrapper(x)
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert result[0].shape == (1, 4, 512, 512)

    def test_registry_entry(self):
        from training.export_onnx import MODEL_CONFIGS

        assert "back_view" in MODEL_CONFIGS
        cfg = MODEL_CONFIGS["back_view"]
        assert cfg["output_names"] == ["output"]
        assert cfg["input_shape"] == (1, 8, 512, 512)
        assert cfg["default_filename"] == "back_view_generation.onnx"

    def test_wrapper_output_range(self):
        from training.export_onnx import BackViewWrapper
        from training.models.back_view_model import BackViewModel

        model = BackViewModel()
        wrapper = BackViewWrapper(model)
        x = torch.randn(1, 8, 512, 512)
        result = wrapper(x)
        assert result[0].min() >= 0.0
        assert result[0].max() <= 1.0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestBackViewEvaluation:
    """Tests for evaluation helpers."""

    def test_rgba_to_rgb_uint8(self):
        import numpy as np

        from training.evaluate import _rgba_to_rgb_uint8

        rgba = np.array([0.5, 0.3, 0.8, 1.0]).reshape(4, 1, 1) * np.ones((4, 64, 64))
        rgb = _rgba_to_rgb_uint8(rgba.astype(np.float32))
        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype == np.uint8

    def test_compute_ssim_identical(self):
        import numpy as np

        from training.evaluate import _compute_ssim

        img = np.random.rand(64, 64, 3).astype(np.float32)
        ssim = _compute_ssim(img, img)
        assert ssim > 0.99  # identical images should have SSIM ~1.0

    def test_compute_ssim_different(self):
        import numpy as np

        from training.evaluate import _compute_ssim

        img1 = np.zeros((64, 64, 3), dtype=np.float32)
        img2 = np.ones((64, 64, 3), dtype=np.float32)
        ssim = _compute_ssim(img1, img2)
        assert ssim < 0.1  # very different images should have low SSIM
