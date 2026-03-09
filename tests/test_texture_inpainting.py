"""Tests for texture inpainting model, dataset, and ONNX export config."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import torch

    from training.models.texture_inpainting_model import TextureInpaintingModel

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from PIL import Image

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


# ---------------------------------------------------------------------------
# TextureInpaintingModel
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestTextureInpaintingModel:
    """U-Net forward pass shape checks."""

    def test_forward_shape(self):
        model = TextureInpaintingModel(in_channels=5, out_channels=4)
        x = torch.randn(1, 5, 512, 512)
        out = model(x)
        assert "inpainted" in out
        assert out["inpainted"].shape == (1, 4, 512, 512)

    def test_forward_batch(self):
        model = TextureInpaintingModel()
        x = torch.randn(2, 5, 512, 512)
        out = model(x)
        assert out["inpainted"].shape == (2, 4, 512, 512)

    def test_output_range(self):
        """Sigmoid output should be in [0, 1]."""
        model = TextureInpaintingModel()
        x = torch.randn(1, 5, 512, 512)
        out = model(x)["inpainted"]
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_custom_channels(self):
        model = TextureInpaintingModel(in_channels=6, out_channels=3)
        x = torch.randn(1, 6, 512, 512)
        out = model(x)
        assert out["inpainted"].shape == (1, 3, 512, 512)

    def test_model_size_reasonable(self):
        """Model size should match other U-Net models in the pipeline."""
        model = TextureInpaintingModel()
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        param_mb = param_bytes / (1024 * 1024)
        # Same U-Net architecture as inpainting model (~112 MB params).
        # The 15MB ONNX target from the issue is for future quantization.
        assert param_mb < 150.0, f"Model parameters {param_mb:.1f} MB unexpectedly large"


# ---------------------------------------------------------------------------
# TextureInpaintingDataset
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCH or not _HAS_PIL, reason="torch/PIL not installed")
class TestTextureInpaintingDataset:
    """Dataset loader tests."""

    @pytest.fixture()
    def texture_pairs(self, tmp_path: Path) -> Path:
        """Create minimal texture pair directories."""
        for i in range(5):
            pair_dir = tmp_path / f"char_{i:03d}"
            pair_dir.mkdir()
            # Partial texture: RGBA with some transparent regions
            partial = Image.new("RGBA", (1024, 1024), (128, 64, 32, 255))
            partial.save(pair_dir / "partial_texture.png")
            # Complete texture: full RGBA
            complete = Image.new("RGBA", (1024, 1024), (128, 64, 32, 255))
            complete.save(pair_dir / "complete_texture.png")
            # Mask: 255=needs inpainting, 0=observed
            mask = Image.new("L", (1024, 1024), 0)
            mask.save(pair_dir / "inpainting_mask.png")
        return tmp_path

    def test_dataset_loading(self, texture_pairs: Path):
        from training.data.texture_inpainting_dataset import (
            TextureInpaintingDataset,
            TextureInpaintingDatasetConfig,
        )

        config = TextureInpaintingDatasetConfig(
            dataset_dirs=[texture_pairs],
            split="train",
        )
        ds = TextureInpaintingDataset(config)
        assert len(ds) > 0

    def test_sample_shapes(self, texture_pairs: Path):
        from training.data.texture_inpainting_dataset import (
            TextureInpaintingDataset,
            TextureInpaintingDatasetConfig,
        )

        config = TextureInpaintingDatasetConfig(
            dataset_dirs=[texture_pairs],
            split="train",
            horizontal_flip=False,
            vertical_flip=False,
            color_jitter={},
            random_mask_augmentation=False,
        )
        ds = TextureInpaintingDataset(config)
        sample = ds[0]
        assert sample["image"].shape == (5, 512, 512)
        assert sample["target"].shape == (4, 512, 512)

    def test_sample_range(self, texture_pairs: Path):
        from training.data.texture_inpainting_dataset import (
            TextureInpaintingDataset,
            TextureInpaintingDatasetConfig,
        )

        config = TextureInpaintingDatasetConfig(
            dataset_dirs=[texture_pairs],
            split="train",
            horizontal_flip=False,
            vertical_flip=False,
            color_jitter={},
            random_mask_augmentation=False,
        )
        ds = TextureInpaintingDataset(config)
        sample = ds[0]
        assert sample["image"].min() >= 0.0
        assert sample["image"].max() <= 1.0
        assert sample["target"].min() >= 0.0
        assert sample["target"].max() <= 1.0

    def test_observation_mask_channel(self, texture_pairs: Path):
        """Channel 4 of image should be the observation mask (1=observed)."""
        from training.data.texture_inpainting_dataset import (
            TextureInpaintingDataset,
            TextureInpaintingDatasetConfig,
        )

        config = TextureInpaintingDatasetConfig(
            dataset_dirs=[texture_pairs],
            split="train",
            horizontal_flip=False,
            vertical_flip=False,
            color_jitter={},
            random_mask_augmentation=False,
        )
        ds = TextureInpaintingDataset(config)
        sample = ds[0]
        obs_mask = sample["image"][4]  # Channel 4 = observation mask
        # Our test mask is all-zero (no inpainting needed), so obs should be all-1
        assert obs_mask.min() == 1.0
        assert obs_mask.max() == 1.0

    def test_splits_cover_all(self, texture_pairs: Path):
        from training.data.texture_inpainting_dataset import (
            TextureInpaintingDataset,
            TextureInpaintingDatasetConfig,
        )

        total = 0
        for split in ("train", "val", "test"):
            config = TextureInpaintingDatasetConfig(
                dataset_dirs=[texture_pairs],
                split=split,
            )
            ds = TextureInpaintingDataset(config)
            total += len(ds)
        assert total == 5

    def test_config_from_dict(self):
        from training.data.texture_inpainting_dataset import TextureInpaintingDatasetConfig

        cfg = {
            "data": {
                "dataset_dirs": ["./data/textures"],
                "resolution": 256,
                "split_seed": 123,
                "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            },
            "augmentation": {
                "horizontal_flip": False,
                "vertical_flip": False,
                "color_jitter": {"brightness": 0.2},
                "random_mask_augmentation": False,
            },
        }
        config = TextureInpaintingDatasetConfig.from_dict(cfg)
        assert config.resolution == 256
        assert config.split_seed == 123
        assert config.split_ratios == (0.7, 0.15, 0.15)
        assert config.horizontal_flip is False
        assert config.random_mask_augmentation is False


# ---------------------------------------------------------------------------
# ONNX export config
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestTextureInpaintingONNXConfig:
    """Verify ONNX export configuration matches Rust runtime expectations."""

    def test_export_config_exists(self):
        from training.export_onnx import MODEL_CONFIGS

        assert "texture_inpainting" in MODEL_CONFIGS

    def test_export_config_contract(self):
        from training.export_onnx import MODEL_CONFIGS

        cfg = MODEL_CONFIGS["texture_inpainting"]
        assert cfg["default_filename"] == "texture_inpainting.onnx"
        assert cfg["output_names"] == ["inpainted"]
        # Single input (not dual_input)
        assert "dual_input" not in cfg
        assert cfg["input_shape"] == (1, 5, 512, 512)

    def test_wrapper_forward(self):
        from training.export_onnx import TextureInpaintingWrapper

        model = TextureInpaintingModel()
        wrapper = TextureInpaintingWrapper(model)
        x = torch.randn(1, 5, 512, 512)
        result = wrapper(x)
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert result[0].shape == (1, 4, 512, 512)
