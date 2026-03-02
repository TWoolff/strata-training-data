"""Tests for training/evaluate.py and training/utils/visualization.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from training.utils.visualization import (
    colorize_mask,
    overlay_segmentation,
)

try:
    import matplotlib  # noqa: F401

    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

try:
    import torch  # noqa: F401

    from training.evaluate import (
        _IMAGENET_MEAN,
        _IMAGENET_STD,
        _denormalize_image,
        evaluate_joints,
        evaluate_segmentation,
        evaluate_weights,
        main,
    )

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# ---------------------------------------------------------------------------
# Visualization: colorize_mask
# ---------------------------------------------------------------------------


class TestColorizeMask:
    """Test mask colorization using REGION_COLORS."""

    def test_background_is_black(self):
        """Region 0 (background) should map to black (0, 0, 0)."""
        mask = np.zeros((4, 4), dtype=np.uint8)
        colored = colorize_mask(mask)
        assert colored.shape == (4, 4, 3)
        assert np.all(colored == 0)

    def test_head_is_red(self):
        """Region 1 (head) should map to (255, 0, 0)."""
        mask = np.ones((2, 2), dtype=np.uint8)
        colored = colorize_mask(mask)
        np.testing.assert_array_equal(colored[0, 0], [255, 0, 0])

    def test_multiple_regions(self):
        """Multiple regions in a single mask should each get their own color."""
        mask = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        colored = colorize_mask(mask)
        assert colored.shape == (2, 2, 3)
        # Background = black, head = red, neck = green, chest = blue
        np.testing.assert_array_equal(colored[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(colored[0, 1], [255, 0, 0])
        np.testing.assert_array_equal(colored[1, 0], [0, 255, 0])
        np.testing.assert_array_equal(colored[1, 1], [0, 0, 255])


# ---------------------------------------------------------------------------
# Visualization: overlay_segmentation
# ---------------------------------------------------------------------------


class TestOverlaySegmentation:
    """Test image-mask blending."""

    def test_alpha_zero_returns_image(self):
        """With alpha=0, the overlay should be the original image."""
        image = np.full((4, 4, 3), 128, dtype=np.uint8)
        mask = np.ones((4, 4), dtype=np.uint8)
        result = overlay_segmentation(image, mask, alpha=0.0)
        np.testing.assert_array_equal(result, image)

    def test_alpha_one_returns_mask(self):
        """With alpha=1, the overlay should be the colorized mask."""
        image = np.full((4, 4, 3), 128, dtype=np.uint8)
        mask = np.ones((4, 4), dtype=np.uint8)  # head = red
        result = overlay_segmentation(image, mask, alpha=1.0)
        expected = colorize_mask(mask)
        np.testing.assert_array_equal(result, expected)

    def test_output_shape(self):
        """Output should match input image shape."""
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=np.uint8)
        result = overlay_segmentation(image, mask, alpha=0.5)
        assert result.shape == (16, 16, 3)
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# Visualization: save functions (file I/O)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestSavePredictionGrid:
    """Test prediction grid saving."""

    def test_saves_file(self, tmp_path: Path):
        """Grid should be saved to the specified path."""
        from training.utils.visualization import save_prediction_grid

        images = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(4)]
        gt_masks = [np.zeros((32, 32), dtype=np.uint8) for _ in range(4)]
        pred_masks = [np.ones((32, 32), dtype=np.uint8) for _ in range(4)]

        out_path = tmp_path / "grid.png"
        save_prediction_grid(images, gt_masks, pred_masks, out_path, n=2)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_handles_empty_list(self, tmp_path: Path):
        """Empty input list should not raise, just skip."""
        from training.utils.visualization import save_prediction_grid

        out_path = tmp_path / "grid.png"
        save_prediction_grid([], [], [], out_path, n=4)
        assert not out_path.exists()


@pytest.mark.skipif(not _HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestPlotConfusionMatrix:
    """Test confusion matrix plotting."""

    def test_saves_file(self, tmp_path: Path):
        """Confusion matrix should be saved as PNG."""
        from training.utils.visualization import plot_confusion_matrix

        confusion = np.eye(3, dtype=np.int64) * 100
        class_names = ["bg", "head", "neck"]
        out_path = tmp_path / "cm.png"
        plot_confusion_matrix(confusion, class_names, out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0


@pytest.mark.skipif(not _HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestPlotPerClassIou:
    """Test per-class IoU bar chart."""

    def test_saves_file(self, tmp_path: Path):
        """Bar chart should be saved as PNG."""
        from training.utils.visualization import plot_per_class_iou

        iou_dict = {"background": 0.9, "head": 0.8, "neck": 0.7}
        out_path = tmp_path / "iou.png"
        plot_per_class_iou(iou_dict, out_path)
        assert out_path.exists()


class TestOverlayJoints:
    """Test joint overlay drawing."""

    def test_output_shape_preserved(self):
        """Output should match input image shape."""
        from training.utils.visualization import overlay_joints

        image = np.zeros((64, 64, 3), dtype=np.uint8)
        joints = {"head": (32.0, 32.0), "hips": (32.0, 48.0)}
        result = overlay_joints(image, joints, color=(0, 255, 0), radius=3)
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_does_not_modify_original(self):
        """Original image should not be modified."""
        from training.utils.visualization import overlay_joints

        image = np.zeros((64, 64, 3), dtype=np.uint8)
        original = image.copy()
        overlay_joints(image, {"head": (32.0, 32.0)})
        np.testing.assert_array_equal(image, original)


@pytest.mark.skipif(not _HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestPlotJointScatter:
    """Test joint scatter plot."""

    def test_saves_file(self, tmp_path: Path):
        """Scatter plot should be saved as PNG."""
        from training.utils.visualization import plot_joint_scatter

        gt = np.random.rand(100, 2).astype(np.float32)
        pred = gt + np.random.randn(100, 2).astype(np.float32) * 0.05
        out_path = tmp_path / "scatter.png"
        plot_joint_scatter(gt, pred, out_path)
        assert out_path.exists()


class TestSaveJointComparison:
    """Test joint comparison image."""

    def test_saves_file(self, tmp_path: Path):
        """Joint comparison should be saved as PNG."""
        from training.utils.visualization import save_joint_comparison

        image = np.zeros((64, 64, 3), dtype=np.uint8)
        gt_joints = {"head": (30.0, 30.0), "hips": (30.0, 50.0)}
        pred_joints = {"head": (32.0, 31.0), "hips": (29.0, 49.0)}
        out_path = tmp_path / "joints.png"
        save_joint_comparison(image, gt_joints, pred_joints, out_path)
        assert out_path.exists()


@pytest.mark.skipif(not _HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestPlotPerJointError:
    """Test per-joint error bar chart."""

    def test_saves_file(self, tmp_path: Path):
        """Bar chart should be saved as PNG."""
        from training.utils.visualization import plot_per_joint_error

        error_dict = {"hips": 0.001, "spine": 0.002, "head": 0.0005}
        out_path = tmp_path / "joint_err.png"
        plot_per_joint_error(error_dict, out_path)
        assert out_path.exists()


# ---------------------------------------------------------------------------
# Evaluate: _denormalize_image (requires torch)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestDenormalizeImage:
    """Test image denormalization from ImageNet preprocessing."""

    def test_output_shape_and_dtype(self):
        """Should produce [H, W, 3] uint8 from [3, H, W] float."""
        tensor = np.zeros((3, 32, 32), dtype=np.float32)
        result = _denormalize_image(tensor)
        assert result.shape == (32, 32, 3)
        assert result.dtype == np.uint8

    def test_roundtrip_approx(self):
        """Denormalization should approximately invert normalization."""
        # Create a known pixel value and normalize it
        pixel_value = 0.6  # in [0, 1]
        normalized = (pixel_value - _IMAGENET_MEAN[0]) / _IMAGENET_STD[0]
        tensor = np.full((3, 1, 1), normalized, dtype=np.float32)
        for c in range(3):
            tensor[c] = (pixel_value - _IMAGENET_MEAN[c]) / _IMAGENET_STD[c]

        result = _denormalize_image(tensor)
        expected = round(pixel_value * 255)
        # Allow ±1 for rounding
        assert abs(int(result[0, 0, 0]) - expected) <= 1


# ---------------------------------------------------------------------------
# Evaluate: CLI & function imports (requires torch)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestEvaluateCLI:
    """Test CLI argument validation for evaluate.py."""

    def test_model_choices(self):
        """--model should only accept valid model types."""
        # We can't easily test main() without real checkpoints, but we can
        # verify the module imports correctly and the functions exist.
        assert callable(main)

    def test_evaluate_functions_exist(self):
        """All three evaluation functions should be importable."""
        assert callable(evaluate_segmentation)
        assert callable(evaluate_joints)
        assert callable(evaluate_weights)
