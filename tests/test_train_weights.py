"""Tests for training/train_weights.py — loss, collate, LR warmup."""

from __future__ import annotations

import pytest

try:
    import torch

    from training.train_weights import adjust_lr, collate_fn, compute_loss

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# compute_loss
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestComputeLoss:
    """Weight prediction multi-head loss computation."""

    @pytest.fixture()
    def default_weights(self):
        return {"weight_loss_weight": 1.0, "confidence_weight": 0.5}

    def _make_targets(self, batch_size: int = 2, n_verts: int = 64):
        gt_weights = torch.zeros(batch_size, 20, n_verts)
        # Create soft weight distributions
        gt_weights[:, 0, :] = 0.6  # hips
        gt_weights[:, 1, :] = 0.4  # spine
        conf = torch.ones(batch_size, n_verts)
        conf[:, n_verts // 2 :] = 0.0  # Half vertices without GT
        num_verts = torch.tensor([n_verts] * batch_size)
        return {
            "weights_target": gt_weights,
            "confidence_target": conf,
            "num_vertices": num_verts,
        }

    def _make_outputs(self, batch_size: int = 2, n_verts: int = 64):
        return {
            "weights": torch.randn(batch_size, 20, n_verts, 1),
            "confidence": torch.randn(batch_size, 1, n_verts, 1),
        }

    def test_loss_returns_scalar_and_components(self, default_weights):
        """compute_loss should return a scalar loss and component dict."""
        outputs = self._make_outputs()
        targets = self._make_targets()
        total, components = compute_loss(outputs, targets, default_weights)

        assert total.ndim == 0  # scalar
        assert total.item() > 0
        assert "loss/weight_kl" in components
        assert "loss/confidence" in components
        assert "loss/total" in components

    def test_no_gt_vertices(self, default_weights):
        """Weight loss should be zero when no vertices have GT data."""
        outputs = self._make_outputs()
        targets = self._make_targets()
        targets["confidence_target"] = torch.zeros(2, 64)

        _, components = compute_loss(outputs, targets, default_weights)

        assert components["loss/weight_kl"] == pytest.approx(0.0)

    def test_perfect_predictions(self, default_weights):
        """With matching predictions and targets, KL divergence should be near zero."""
        targets = self._make_targets()
        gt = targets["weights_target"]

        # Create predictions that match GT after softmax
        # Use log of GT as logits (inverse softmax)
        eps = 1e-8
        logits = torch.log(gt + eps)
        outputs = {
            "weights": logits.unsqueeze(-1),
            "confidence": torch.ones(2, 1, 64, 1) * 5.0,
        }
        _, components = compute_loss(outputs, targets, default_weights)

        # KL divergence should be very small for matching distributions
        assert components["loss/weight_kl"] < 0.1

    def test_weighted_sum(self):
        """Total loss should be weighted sum of components."""
        weights = {"weight_loss_weight": 2.0, "confidence_weight": 3.0}
        outputs = self._make_outputs()
        targets = self._make_targets()
        total, comp = compute_loss(outputs, targets, weights)

        expected = 2.0 * comp["loss/weight_kl"] + 3.0 * comp["loss/confidence"]
        assert total.item() == pytest.approx(expected, rel=1e-4)

    def test_respects_num_vertices(self, default_weights):
        """Loss should only consider vertices up to num_vertices."""
        outputs = self._make_outputs(n_verts=64)
        targets = self._make_targets(n_verts=64)
        # Only first 10 vertices are real
        targets["num_vertices"] = torch.tensor([10, 10])

        total, _ = compute_loss(outputs, targets, default_weights)
        assert total.item() >= 0.0  # Should still produce valid loss


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestCollateFn:
    """Custom collate for weight data."""

    def test_stacks_tensors(self):
        """Collate should stack all tensor fields and collect num_vertices."""
        batch = [
            {
                "features": torch.randn(31, 64, 1),
                "weights_target": torch.randn(20, 64),
                "confidence_target": torch.ones(64),
                "num_vertices": 30,
            },
            {
                "features": torch.randn(31, 64, 1),
                "weights_target": torch.randn(20, 64),
                "confidence_target": torch.ones(64),
                "num_vertices": 40,
            },
        ]
        result = collate_fn(batch)

        assert result["features"].shape == (2, 31, 64, 1)
        assert result["weights_target"].shape == (2, 20, 64)
        assert result["confidence_target"].shape == (2, 64)
        assert result["num_vertices"].shape == (2,)
        assert result["num_vertices"][0] == 30
        assert result["num_vertices"][1] == 40


# ---------------------------------------------------------------------------
# adjust_lr (warmup)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestAdjustLR:
    """Linear warmup then cosine scheduling."""

    def test_warmup_ramp(self):
        """LR should ramp linearly during warmup."""
        model = torch.nn.Linear(4, 2)
        base_lr = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        # Epoch 0: lr = base_lr * 1/5
        lr0 = adjust_lr(optimizer, epoch=0, base_lr=base_lr, warmup_epochs=5, scheduler=scheduler)
        assert lr0 == pytest.approx(base_lr * 1 / 5)

        # Epoch 4: lr = base_lr * 5/5 = base_lr
        lr4 = adjust_lr(optimizer, epoch=4, base_lr=base_lr, warmup_epochs=5, scheduler=scheduler)
        assert lr4 == pytest.approx(base_lr)

    def test_after_warmup_uses_scheduler(self):
        """After warmup, scheduler should control LR."""
        model = torch.nn.Linear(4, 2)
        base_lr = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        lr5 = adjust_lr(optimizer, epoch=5, base_lr=base_lr, warmup_epochs=5, scheduler=scheduler)
        assert lr5 <= base_lr
