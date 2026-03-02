"""Tests for training/train_segmentation.py — loss, collate, LR warmup."""

from __future__ import annotations

import pytest

try:
    import torch

    from training.train_segmentation import (
        adjust_lr,
        collate_fn,
        compute_class_weights,
        compute_loss,
    )

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# compute_loss
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestComputeLoss:
    """Multi-head loss computation."""

    @pytest.fixture()
    def default_weights(self):
        return {"segmentation_weight": 1.0, "draw_order_weight": 0.5, "confidence_weight": 0.1}

    @pytest.fixture()
    def class_weights(self):
        return torch.ones(22, dtype=torch.float32)

    def _make_targets(self, batch_size: int = 2, h: int = 8, w: int = 8, has_do: bool = True):
        return {
            "segmentation": torch.randint(0, 22, (batch_size, h, w)),
            "draw_order": torch.rand(batch_size, 1, h, w),
            "has_draw_order": torch.tensor([has_do] * batch_size),
            "confidence_target": torch.ones(batch_size, 1, h, w),
        }

    def _make_outputs(self, batch_size: int = 2, h: int = 8, w: int = 8):
        return {
            "segmentation": torch.randn(batch_size, 22, h, w),
            "draw_order": torch.sigmoid(torch.randn(batch_size, 1, h, w)),
            "confidence": torch.sigmoid(torch.randn(batch_size, 1, h, w)),
        }

    def test_loss_returns_scalar_and_components(self, default_weights, class_weights):
        """compute_loss should return a scalar loss and component dict."""
        outputs = self._make_outputs()
        targets = self._make_targets()
        total, components = compute_loss(outputs, targets, class_weights, default_weights)

        assert total.ndim == 0  # scalar
        assert total.item() > 0
        assert "loss/segmentation" in components
        assert "loss/draw_order" in components
        assert "loss/confidence" in components
        assert "loss/total" in components

    def test_loss_without_draw_order(self, default_weights, class_weights):
        """When no examples have draw order, draw_order_loss should be 0."""
        outputs = self._make_outputs()
        targets = self._make_targets(has_do=False)
        total, components = compute_loss(outputs, targets, class_weights, default_weights)

        assert components["loss/draw_order"] == pytest.approx(0.0)
        assert total.item() > 0  # seg + conf still contribute

    def test_perfect_segmentation(self, default_weights, class_weights):
        """With matching predictions and targets, seg loss should be near zero."""
        b, h, w = 1, 4, 4
        targets = self._make_targets(batch_size=b, h=h, w=w)
        gt_classes = targets["segmentation"]  # [1, 4, 4]

        # Create logits with very high value for the correct class
        logits = torch.full((b, 22, h, w), -10.0)
        for i in range(h):
            for j in range(w):
                logits[0, gt_classes[0, i, j], i, j] = 10.0

        outputs = {
            "segmentation": logits,
            "draw_order": targets["draw_order"],
            "confidence": targets["confidence_target"],
        }
        _, components = compute_loss(outputs, targets, class_weights, default_weights)
        assert components["loss/segmentation"] < 0.01

    def test_weighted_sum(self, class_weights):
        """Total loss should be weighted sum of components."""
        weights = {
            "segmentation_weight": 2.0,
            "draw_order_weight": 3.0,
            "confidence_weight": 4.0,
        }
        outputs = self._make_outputs()
        targets = self._make_targets()
        total, comp = compute_loss(outputs, targets, class_weights, weights)

        expected = (
            2.0 * comp["loss/segmentation"]
            + 3.0 * comp["loss/draw_order"]
            + 4.0 * comp["loss/confidence"]
        )
        assert total.item() == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestCollateFn:
    """Custom collate for has_draw_order."""

    def test_stacks_tensors(self):
        """Collate should stack all tensor fields."""
        batch = [
            {
                "image": torch.randn(3, 8, 8),
                "segmentation": torch.randint(0, 22, (8, 8)),
                "draw_order": torch.rand(1, 8, 8),
                "has_draw_order": True,
                "confidence_target": torch.ones(1, 8, 8),
            },
            {
                "image": torch.randn(3, 8, 8),
                "segmentation": torch.randint(0, 22, (8, 8)),
                "draw_order": torch.rand(1, 8, 8),
                "has_draw_order": False,
                "confidence_target": torch.ones(1, 8, 8),
            },
        ]
        result = collate_fn(batch)

        assert result["image"].shape == (2, 3, 8, 8)
        assert result["segmentation"].shape == (2, 8, 8)
        assert result["draw_order"].shape == (2, 1, 8, 8)
        assert result["confidence_target"].shape == (2, 1, 8, 8)

    def test_has_draw_order_is_bool_tensor(self):
        """has_draw_order should be converted to a bool tensor."""
        batch = [
            {
                "image": torch.randn(3, 4, 4),
                "segmentation": torch.zeros(4, 4, dtype=torch.long),
                "draw_order": torch.zeros(1, 4, 4),
                "has_draw_order": True,
                "confidence_target": torch.ones(1, 4, 4),
            },
            {
                "image": torch.randn(3, 4, 4),
                "segmentation": torch.zeros(4, 4, dtype=torch.long),
                "draw_order": torch.zeros(1, 4, 4),
                "has_draw_order": False,
                "confidence_target": torch.ones(1, 4, 4),
            },
        ]
        result = collate_fn(batch)
        assert result["has_draw_order"].dtype == torch.bool
        assert result["has_draw_order"].tolist() == [True, False]


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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

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
        # After first scheduler.step(), LR should be slightly less than base_lr
        assert lr5 <= base_lr

    def test_no_warmup(self):
        """With warmup_epochs=0, scheduler should be used from epoch 0."""
        model = torch.nn.Linear(4, 2)
        base_lr = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        lr0 = adjust_lr(optimizer, epoch=0, base_lr=base_lr, warmup_epochs=0, scheduler=scheduler)
        assert lr0 <= base_lr  # scheduler step applied


# ---------------------------------------------------------------------------
# compute_class_weights
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestComputeClassWeights:
    """Median frequency balancing weights."""

    def test_uniform_distribution(self):
        """Uniform labels should produce roughly uniform weights."""

        class FakeDataset:
            def __len__(self):
                return 4

            def __getitem__(self, i):
                # 4 classes, equally distributed in a 2x2 mask
                mask = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
                return {"segmentation": mask}

        weights = compute_class_weights(FakeDataset(), num_classes=4)
        assert weights.shape == (4,)
        # All classes have equal frequency -> all weights should be 1.0
        for i in range(4):
            assert weights[i].item() == pytest.approx(1.0, rel=0.01)

    def test_absent_class_gets_zero_weight(self):
        """Classes with no pixels should have weight = 0."""

        class FakeDataset:
            def __len__(self):
                return 1

            def __getitem__(self, i):
                return {"segmentation": torch.zeros(4, 4, dtype=torch.long)}

        weights = compute_class_weights(FakeDataset(), num_classes=3)
        # Only class 0 present -> classes 1, 2 get weight 0
        assert weights[0].item() > 0
        assert weights[1].item() == 0.0
        assert weights[2].item() == 0.0

    def test_empty_dataset(self):
        """Empty dataset should return uniform weights."""

        class FakeDataset:
            def __len__(self):
                return 0

            def __getitem__(self, i):
                raise IndexError

        weights = compute_class_weights(FakeDataset(), num_classes=5)
        assert weights.shape == (5,)
        for i in range(5):
            assert weights[i].item() == pytest.approx(1.0)
