"""Tests for training/models/segmentation_model.py."""

from __future__ import annotations

import pytest

try:
    import torch
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch/torchvision not installed")


# Lazy import — only resolved when tests actually run (torch available).
def _make_model(**kwargs):
    from training.models.segmentation_model import SegmentationModel

    return SegmentationModel(pretrained_backbone=False, **kwargs)


# ---------------------------------------------------------------------------
# Model instantiation
# ---------------------------------------------------------------------------


def test_model_instantiates():
    """Model can be created with default parameters."""
    model = _make_model()
    assert model.num_classes == 22


def test_model_custom_num_classes():
    """Model respects a custom num_classes argument."""
    model = _make_model(num_classes=10)
    assert model.num_classes == 10


# ---------------------------------------------------------------------------
# Forward pass — output shapes
# ---------------------------------------------------------------------------


def test_forward_output_shapes():
    """Forward pass produces correct output shapes for 512x512 input."""
    model = _make_model()
    model.eval()

    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = model(x)

    assert set(out.keys()) == {"segmentation", "draw_order", "confidence"}
    assert out["segmentation"].shape == (1, 22, 512, 512)
    assert out["draw_order"].shape == (1, 1, 512, 512)
    assert out["confidence"].shape == (1, 1, 512, 512)


def test_forward_batch_size():
    """Forward pass handles batch_size > 1."""
    model = _make_model()
    model.eval()

    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = model(x)

    assert out["segmentation"].shape == (2, 22, 512, 512)
    assert out["draw_order"].shape == (2, 1, 512, 512)
    assert out["confidence"].shape == (2, 1, 512, 512)


def test_forward_non_square_input():
    """Forward pass works with non-512 resolution (upsamples to input size)."""
    model = _make_model()
    model.eval()

    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)

    assert out["segmentation"].shape == (1, 22, 256, 256)
    assert out["draw_order"].shape == (1, 1, 256, 256)
    assert out["confidence"].shape == (1, 1, 256, 256)


# ---------------------------------------------------------------------------
# Output value ranges
# ---------------------------------------------------------------------------


def test_draw_order_sigmoid_range():
    """Draw order output is in [0, 1] (sigmoid applied)."""
    model = _make_model()
    model.eval()

    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)

    assert out["draw_order"].min() >= 0.0
    assert out["draw_order"].max() <= 1.0


def test_confidence_sigmoid_range():
    """Confidence output is in [0, 1] (sigmoid applied)."""
    model = _make_model()
    model.eval()

    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)

    assert out["confidence"].min() >= 0.0
    assert out["confidence"].max() <= 1.0


def test_segmentation_raw_logits():
    """Segmentation output is raw logits (can be negative)."""
    model = _make_model()
    model.eval()

    torch.manual_seed(42)
    has_negative = False
    for _ in range(5):
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        if out["segmentation"].min() < 0:
            has_negative = True
            break

    assert has_negative, "Expected raw logits to include negative values"


# ---------------------------------------------------------------------------
# Backbone channel detection
# ---------------------------------------------------------------------------


def test_backbone_channel_detection():
    """_detect_backbone_channels returns 960 for MobileNetV3-Large."""
    model = _make_model()
    assert model._detect_backbone_channels() == 960
