"""Tests for training/utils/checkpoint.py — save/load and EarlyStopping."""

from __future__ import annotations

import pytest

from training.utils.checkpoint import EarlyStopping

try:
    import torch
    import torch.nn as nn

    from training.utils.checkpoint import load_checkpoint, save_checkpoint

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Fixtures (torch-dependent)
# ---------------------------------------------------------------------------

if _HAS_TORCH:

    class _TinyModel(nn.Module):
        """Minimal model for checkpoint tests."""

        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)


@pytest.fixture()
def tiny_model():
    return _TinyModel()


@pytest.fixture()
def optimizer(tiny_model):
    return torch.optim.Adam(tiny_model.parameters(), lr=1e-3)


@pytest.fixture()
def scheduler(optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)


# ---------------------------------------------------------------------------
# save_checkpoint / load_checkpoint
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestSaveLoadCheckpoint:
    """Checkpoint round-trip tests."""

    def test_save_creates_file(self, tmp_path, tiny_model, optimizer, scheduler):
        """save_checkpoint should create the .pt file."""
        path = tmp_path / "ckpt.pt"
        save_checkpoint(tiny_model, optimizer, scheduler, epoch=5, metrics={"miou": 0.8}, path=path)
        assert path.exists()

    def test_round_trip_model_weights(self, tmp_path, tiny_model, optimizer, scheduler):
        """Model weights should survive save -> load round trip."""
        path = tmp_path / "ckpt.pt"
        original_weights = {k: v.clone() for k, v in tiny_model.state_dict().items()}
        save_checkpoint(tiny_model, optimizer, scheduler, epoch=3, metrics={}, path=path)

        # Modify weights, then restore
        with torch.no_grad():
            for p in tiny_model.parameters():
                p.fill_(0.0)

        info = load_checkpoint(path, tiny_model, optimizer, scheduler)
        for k, v in tiny_model.state_dict().items():
            assert torch.allclose(v, original_weights[k]), f"Weight mismatch for {k}"
        assert info["epoch"] == 3

    def test_round_trip_metrics(self, tmp_path, tiny_model, optimizer, scheduler):
        """Metrics should survive round trip."""
        path = tmp_path / "ckpt.pt"
        metrics = {"val/miou": 0.75, "val/loss": 0.42}
        save_checkpoint(tiny_model, optimizer, scheduler, epoch=10, metrics=metrics, path=path)

        info = load_checkpoint(path, tiny_model)
        assert info["metrics"]["val/miou"] == pytest.approx(0.75)
        assert info["metrics"]["val/loss"] == pytest.approx(0.42)
        assert info["epoch"] == 10

    def test_load_without_optimizer(self, tmp_path, tiny_model, optimizer, scheduler):
        """Loading without optimizer/scheduler should still work."""
        path = tmp_path / "ckpt.pt"
        save_checkpoint(tiny_model, optimizer, scheduler, epoch=1, metrics={}, path=path)
        info = load_checkpoint(path, tiny_model)
        assert info["epoch"] == 1

    def test_creates_parent_dirs(self, tmp_path, tiny_model, optimizer, scheduler):
        """save_checkpoint should create nested parent directories."""
        path = tmp_path / "sub" / "dir" / "ckpt.pt"
        save_checkpoint(tiny_model, optimizer, scheduler, epoch=0, metrics={}, path=path)
        assert path.exists()

    def test_load_without_scheduler_key(self, tmp_path, tiny_model, optimizer):
        """Loading a checkpoint that has no scheduler state should not error."""
        path = tmp_path / "ckpt.pt"
        save_checkpoint(tiny_model, optimizer, scheduler=None, epoch=2, metrics={}, path=path)
        info = load_checkpoint(path, tiny_model, optimizer, scheduler=None)
        assert info["epoch"] == 2


# ---------------------------------------------------------------------------
# EarlyStopping (no torch dependency)
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    """Early stopping logic."""

    def test_no_stop_while_improving(self):
        """Should not stop when metric keeps improving."""
        es = EarlyStopping(patience=3, metric_name="val/miou", mode="max")
        assert not es.step({"val/miou": 0.5})
        assert not es.step({"val/miou": 0.6})
        assert not es.step({"val/miou": 0.7})
        assert not es.step({"val/miou": 0.8})
        assert es.epochs_without_improvement == 0

    def test_stops_after_patience(self):
        """Should stop after `patience` epochs without improvement."""
        es = EarlyStopping(patience=3, metric_name="val/miou", mode="max")
        assert not es.step({"val/miou": 0.8})  # new best
        assert not es.step({"val/miou": 0.7})  # no improvement (1)
        assert not es.step({"val/miou": 0.6})  # no improvement (2)
        # 3rd epoch without improvement -> patience exhausted
        assert es.step({"val/miou": 0.5}) is True

    def test_improvement_resets_counter(self):
        """Improvement should reset the patience counter."""
        es = EarlyStopping(patience=2, metric_name="val/miou", mode="max")
        es.step({"val/miou": 0.5})
        es.step({"val/miou": 0.4})  # no improvement (1)
        assert es.epochs_without_improvement == 1
        es.step({"val/miou": 0.6})  # improvement — resets
        assert es.epochs_without_improvement == 0
        assert es.best_value == pytest.approx(0.6)

    def test_min_mode(self):
        """mode='min' should track decreasing metric."""
        es = EarlyStopping(patience=2, metric_name="val/loss", mode="min")
        assert not es.step({"val/loss": 1.0})
        assert not es.step({"val/loss": 0.8})  # improvement
        assert not es.step({"val/loss": 0.9})  # no improvement (1)
        assert es.step({"val/loss": 0.85})  # no improvement (2) -> stop

    def test_missing_metric_does_not_stop(self):
        """Missing metric in dict should not trigger stop."""
        es = EarlyStopping(patience=1, metric_name="val/miou", mode="max")
        assert not es.step({"other_metric": 0.5})
        assert not es.step({"other_metric": 0.5})

    def test_invalid_mode_raises(self):
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError, match="mode must be"):
            EarlyStopping(patience=5, mode="invalid")

    def test_patience_one(self):
        """Patience=1 should stop immediately after first non-improvement."""
        es = EarlyStopping(patience=1, metric_name="val/miou", mode="max")
        es.step({"val/miou": 0.5})
        assert es.step({"val/miou": 0.4}) is True
