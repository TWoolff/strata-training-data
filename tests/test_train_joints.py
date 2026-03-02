"""Tests for training/train_joints.py — loss, collate, LR warmup."""

from __future__ import annotations

import pytest

try:
    import torch

    from training.train_joints import adjust_lr, collate_fn, compute_loss

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# compute_loss
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestComputeLoss:
    """Joint refinement multi-head loss computation."""

    @pytest.fixture()
    def default_weights(self):
        return {"offset_weight": 1.0, "presence_weight": 1.0, "confidence_weight": 0.5}

    def _make_targets(self, batch_size: int = 2, num_joints: int = 20):
        gt_visible = torch.ones(batch_size, num_joints)
        # Mark hair_back (slot 19) as invisible
        gt_visible[:, 19] = 0.0
        gt_offsets = torch.randn(batch_size, 2, num_joints) * 0.03
        return {"gt_offsets": gt_offsets, "gt_visible": gt_visible}

    def _make_outputs(self, batch_size: int = 2, num_joints: int = 20):
        return {
            "offsets": torch.randn(batch_size, 2, num_joints) * 0.03,
            "confidence": torch.randn(batch_size, num_joints),
            "present": torch.randn(batch_size, num_joints),
        }

    def test_loss_returns_scalar_and_components(self, default_weights):
        """compute_loss should return a scalar loss and component dict."""
        outputs = self._make_outputs()
        targets = self._make_targets()
        total, components = compute_loss(outputs, targets, default_weights)

        assert total.ndim == 0  # scalar
        assert total.item() > 0
        assert "loss/offset" in components
        assert "loss/presence" in components
        assert "loss/confidence" in components
        assert "loss/total" in components

    def test_offset_loss_only_visible(self, default_weights):
        """Offset loss should only consider visible joints."""
        outputs = self._make_outputs()
        targets = self._make_targets()
        # Make all joints invisible
        targets["gt_visible"] = torch.zeros(2, 20)

        _, components = compute_loss(outputs, targets, default_weights)

        # Offset loss should be 0 when no visible joints
        assert components["loss/offset"] == pytest.approx(0.0)

    def test_perfect_offsets(self, default_weights):
        """With matching predictions and targets, offset loss should be zero."""
        targets = self._make_targets()
        outputs = {
            "offsets": targets["gt_offsets"].clone(),
            "confidence": torch.ones(2, 20) * 5.0,  # high confidence logits
            "present": torch.ones(2, 20) * 5.0,  # high presence logits
        }
        _, components = compute_loss(outputs, targets, default_weights)

        assert components["loss/offset"] < 1e-6

    def test_weighted_sum(self):
        """Total loss should be weighted sum of components."""
        weights = {"offset_weight": 2.0, "presence_weight": 3.0, "confidence_weight": 4.0}
        outputs = self._make_outputs()
        targets = self._make_targets()
        total, comp = compute_loss(outputs, targets, weights)

        expected = (
            2.0 * comp["loss/offset"] + 3.0 * comp["loss/presence"] + 4.0 * comp["loss/confidence"]
        )
        assert total.item() == pytest.approx(expected, rel=1e-4)

    def test_presence_loss_uses_all_joints(self, default_weights):
        """Presence loss uses all joints, not just visible ones."""
        outputs = self._make_outputs()
        targets = self._make_targets()

        _, _comp_full = compute_loss(outputs, targets, default_weights)

        # Make all invisible — presence loss should still be nonzero
        targets["gt_visible"] = torch.zeros(2, 20)
        _, comp_invis = compute_loss(outputs, targets, default_weights)

        assert comp_invis["loss/presence"] > 0


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestCollateFn:
    """Custom collate for joint data."""

    def test_stacks_tensors(self):
        """Collate should stack all tensor fields."""
        batch = [
            {
                "image": torch.randn(3, 8, 8),
                "gt_positions": torch.randn(20, 2),
                "gt_visible": torch.ones(20),
                "geo_positions": torch.randn(20, 2),
                "gt_offsets": torch.randn(2, 20),
            },
            {
                "image": torch.randn(3, 8, 8),
                "gt_positions": torch.randn(20, 2),
                "gt_visible": torch.ones(20),
                "geo_positions": torch.randn(20, 2),
                "gt_offsets": torch.randn(2, 20),
            },
        ]
        result = collate_fn(batch)

        assert result["image"].shape == (2, 3, 8, 8)
        assert result["gt_positions"].shape == (2, 20, 2)
        assert result["gt_visible"].shape == (2, 20)
        assert result["geo_positions"].shape == (2, 20, 2)
        assert result["gt_offsets"].shape == (2, 2, 20)


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

        # Epoch 0: lr = base_lr * 1/3
        lr0 = adjust_lr(optimizer, epoch=0, base_lr=base_lr, warmup_epochs=3, scheduler=scheduler)
        assert lr0 == pytest.approx(base_lr * 1 / 3)

        # Epoch 2: lr = base_lr * 3/3 = base_lr
        lr2 = adjust_lr(optimizer, epoch=2, base_lr=base_lr, warmup_epochs=3, scheduler=scheduler)
        assert lr2 == pytest.approx(base_lr)

    def test_after_warmup_uses_scheduler(self):
        """After warmup, scheduler should control LR."""
        model = torch.nn.Linear(4, 2)
        base_lr = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        lr3 = adjust_lr(optimizer, epoch=3, base_lr=base_lr, warmup_epochs=3, scheduler=scheduler)
        assert lr3 <= base_lr
