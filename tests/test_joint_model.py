"""Tests for training/models/joint_model.py."""

from __future__ import annotations

import pytest

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch/torchvision not installed")


# Lazy import — only resolved when tests actually run (torch available).
def _make_model(**kwargs):
    from training.models.joint_model import JointModel

    return JointModel(pretrained_backbone=False, **kwargs)


# ---------------------------------------------------------------------------
# Model instantiation
# ---------------------------------------------------------------------------


def test_model_instantiates():
    """Model can be created with default parameters."""
    model = _make_model()
    assert model.num_joints == 20


def test_model_custom_num_joints():
    """Model respects a custom num_joints argument."""
    model = _make_model(num_joints=15)
    assert model.num_joints == 15


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

    assert set(out.keys()) == {"offsets", "confidence", "present"}
    assert out["offsets"].shape == (1, 2, 20)
    assert out["confidence"].shape == (1, 20)
    assert out["present"].shape == (1, 20)


def test_forward_batch_size():
    """Forward pass handles batch_size > 1."""
    model = _make_model()
    model.eval()

    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = model(x)

    assert out["offsets"].shape == (2, 2, 20)
    assert out["confidence"].shape == (2, 20)
    assert out["present"].shape == (2, 20)


def test_forward_smaller_input():
    """Forward pass works with non-512 resolution."""
    model = _make_model()
    model.eval()

    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)

    assert out["offsets"].shape == (1, 2, 20)
    assert out["confidence"].shape == (1, 20)
    assert out["present"].shape == (1, 20)


# ---------------------------------------------------------------------------
# Offset layout
# ---------------------------------------------------------------------------


def test_offset_layout_dx_first():
    """Offsets should be [B, 2, 20] with channel 0=dx, channel 1=dy.

    When flattened, first 20 values = dx, next 20 = dy (matching Rust contract).
    """
    model = _make_model()
    model.eval()

    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)

    offsets = out["offsets"]
    assert offsets.shape == (1, 2, 20)

    # Flatten and verify layout
    flat = offsets.flatten(1)  # [1, 40]
    assert flat.shape == (1, 40)

    # First 20 should match channel 0, next 20 match channel 1
    assert torch.allclose(flat[0, :20], offsets[0, 0, :])
    assert torch.allclose(flat[0, 20:], offsets[0, 1, :])


# ---------------------------------------------------------------------------
# Output value ranges — confidence and present are raw logits
# ---------------------------------------------------------------------------


def test_confidence_raw_logits():
    """Confidence output is raw logits (can be negative)."""
    model = _make_model()
    model.eval()

    torch.manual_seed(42)
    has_negative = False
    for _ in range(5):
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        if out["confidence"].min() < 0:
            has_negative = True
            break

    assert has_negative, "Expected raw logits to include negative values"


def test_present_raw_logits():
    """Present output is raw logits (can be negative)."""
    model = _make_model()
    model.eval()

    torch.manual_seed(42)
    has_negative = False
    for _ in range(5):
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        if out["present"].min() < 0:
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
