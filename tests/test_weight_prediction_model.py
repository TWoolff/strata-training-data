"""Tests for training/models/weight_prediction_model.py — per-vertex MLP."""

from __future__ import annotations

import pytest

try:
    import torch

    from training.models.weight_prediction_model import WeightPredictionModel

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestWeightPredictionModel:
    """Per-vertex weight prediction model (1x1 Conv2d MLP)."""

    def test_output_shapes(self):
        """Model should produce correct output shapes."""
        model = WeightPredictionModel(num_features=31, num_bones=20)
        model.eval()

        x = torch.randn(2, 31, 2048, 1)
        with torch.no_grad():
            out = model(x)

        assert out["weights"].shape == (2, 20, 2048, 1)
        assert out["confidence"].shape == (2, 1, 2048, 1)

    def test_single_vertex(self):
        """Model should work with a single vertex."""
        model = WeightPredictionModel()
        model.eval()

        x = torch.randn(1, 31, 1, 1)
        with torch.no_grad():
            out = model(x)

        assert out["weights"].shape == (1, 20, 1, 1)
        assert out["confidence"].shape == (1, 1, 1, 1)

    def test_variable_vertex_count(self):
        """Model should handle different vertex counts."""
        model = WeightPredictionModel()
        model.eval()

        for n_verts in [16, 128, 512, 2048]:
            x = torch.randn(1, 31, n_verts, 1)
            with torch.no_grad():
                out = model(x)
            assert out["weights"].shape == (1, 20, n_verts, 1)
            assert out["confidence"].shape == (1, 1, n_verts, 1)

    def test_custom_num_bones(self):
        """Model should support custom bone count."""
        model = WeightPredictionModel(num_features=31, num_bones=10)
        model.eval()

        x = torch.randn(1, 31, 64, 1)
        with torch.no_grad():
            out = model(x)

        assert out["weights"].shape == (1, 10, 64, 1)

    def test_gradient_flow(self):
        """Gradients should flow through both heads."""
        model = WeightPredictionModel()
        x = torch.randn(2, 31, 64, 1, requires_grad=True)

        out = model(x)
        loss = out["weights"].sum() + out["confidence"].sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_weights_raw_logits(self):
        """Weights output should be raw logits (not softmax)."""
        model = WeightPredictionModel()
        model.eval()

        x = torch.randn(1, 31, 64, 1)
        with torch.no_grad():
            out = model(x)

        # Raw logits can be negative
        weights = out["weights"]
        assert weights.min() < 0 or weights.max() > 1  # Not softmax/sigmoid

    def test_confidence_raw_logits(self):
        """Confidence output should be raw logits (not sigmoid)."""
        model = WeightPredictionModel()
        model.eval()

        x = torch.randn(1, 31, 64, 1)
        with torch.no_grad():
            out = model(x)

        # Raw logits can be outside [0, 1]
        conf = out["confidence"]
        assert conf.min() < 0 or conf.max() > 1

    def test_onnx_contract_shapes(self):
        """Output shapes should match the Rust ONNX contract."""
        model = WeightPredictionModel(num_features=31, num_bones=20)
        model.eval()

        # Exact input shape from weights.rs
        x = torch.randn(1, 31, 2048, 1)
        with torch.no_grad():
            out = model(x)

        # Expected: weights [1, 20, 2048, 1], confidence [1, 1, 2048, 1]
        assert out["weights"].shape == (1, 20, 2048, 1)
        assert out["confidence"].shape == (1, 1, 2048, 1)
