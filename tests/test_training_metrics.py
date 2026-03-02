"""Tests for training/utils/metrics.py — SegmentationMetrics and JointMetrics."""

from __future__ import annotations

import numpy as np
import pytest

from training.utils.metrics import JointMetrics, SegmentationMetrics, WeightMetrics

# ---------------------------------------------------------------------------
# SegmentationMetrics
# ---------------------------------------------------------------------------


class TestSegmentationMetricsBasic:
    """Core confusion matrix and IoU computation."""

    def test_perfect_predictions(self):
        """Perfect predictions should yield mIoU = 1.0."""
        m = SegmentationMetrics(num_classes=3)
        target = np.array([[[0, 1, 2], [0, 1, 2]]])
        m.update(pred=target, target=target)
        assert m.miou() == pytest.approx(1.0)
        assert m.overall_accuracy() == pytest.approx(1.0)

    def test_completely_wrong(self):
        """All predictions wrong — IoU should be 0 for each class."""
        m = SegmentationMetrics(num_classes=3)
        target = np.array([[[0, 0, 1, 1, 2, 2]]])
        pred = np.array([[[1, 1, 2, 2, 0, 0]]])
        m.update(pred=pred, target=target)
        assert m.miou() == pytest.approx(0.0)
        assert m.overall_accuracy() == pytest.approx(0.0)

    def test_known_confusion_matrix(self):
        """Hand-computed confusion matrix → expected IoU values.

        Confusion matrix (target=row, pred=col):
            [[2, 1, 0],
             [0, 3, 0],
             [1, 0, 1]]

        IoU(0) = 2 / (2+1+0+1) = 2/4 = 0.5
        IoU(1) = 3 / (3+0+1+0) = 3/4 = 0.75
        IoU(2) = 1 / (1+0+1+0) = 1/2 = 0.5
        mIoU = (0.5 + 0.75 + 0.5) / 3 ≈ 0.5833
        """
        m = SegmentationMetrics(num_classes=3)
        # Build arrays that produce the confusion matrix above.
        targets = np.array([0, 0, 0, 1, 1, 1, 2, 2])
        preds = np.array([0, 0, 1, 1, 1, 1, 0, 2])
        m.update(pred=preds, target=targets)

        assert m.miou() == pytest.approx(0.5833, abs=1e-3)
        ious = m.per_class_iou()
        assert ious["background"] == pytest.approx(0.5)
        assert ious["head"] == pytest.approx(0.75)
        assert ious["neck"] == pytest.approx(0.5)

    def test_single_class_only(self):
        """When only one class is present, mIoU = that class's IoU."""
        m = SegmentationMetrics(num_classes=5)
        target = np.zeros((1, 4, 4), dtype=int)
        pred = np.zeros((1, 4, 4), dtype=int)
        m.update(pred=pred, target=target)
        assert m.miou() == pytest.approx(1.0)

    def test_empty_matrix(self):
        """No updates → mIoU = 0."""
        m = SegmentationMetrics(num_classes=3)
        assert m.miou() == 0.0
        assert m.overall_accuracy() == 0.0


class TestSegmentationMetricsIgnoreIndex:
    """Pixels with ignore_index should be excluded."""

    def test_ignore_index_excluded(self):
        """Ignored pixels should not affect the confusion matrix."""
        m = SegmentationMetrics(num_classes=3, ignore_index=-1)
        target = np.array([0, 1, -1, 2])
        pred = np.array([0, 1, 0, 2])
        m.update(pred=pred, target=target)
        assert m.confusion.sum() == 3  # only 3 pixels counted
        assert m.miou() == pytest.approx(1.0)

    def test_ignore_index_custom(self):
        """Custom ignore_index value should work."""
        m = SegmentationMetrics(num_classes=3, ignore_index=255)
        target = np.array([0, 255, 1])
        pred = np.array([0, 0, 1])
        m.update(pred=pred, target=target)
        assert m.confusion.sum() == 2


class TestSegmentationMetricsStreaming:
    """Multi-batch accumulation and reset."""

    def test_multi_batch_accumulation(self):
        """Results should be the same whether data is one batch or split."""
        all_target = np.array([0, 0, 1, 1, 2, 2, 0, 1])
        all_pred = np.array([0, 0, 1, 0, 2, 2, 0, 1])

        # Single batch.
        m1 = SegmentationMetrics(num_classes=3)
        m1.update(pred=all_pred, target=all_target)

        # Two batches.
        m2 = SegmentationMetrics(num_classes=3)
        m2.update(pred=all_pred[:4], target=all_target[:4])
        m2.update(pred=all_pred[4:], target=all_target[4:])

        assert m1.miou() == pytest.approx(m2.miou())
        np.testing.assert_array_equal(m1.confusion, m2.confusion)

    def test_reset_clears_state(self):
        """After reset, metrics should be as if freshly constructed."""
        m = SegmentationMetrics(num_classes=3)
        target = np.array([0, 1, 2])
        m.update(pred=target, target=target)
        assert m.confusion.sum() > 0

        m.reset()
        assert m.confusion.sum() == 0
        assert m.miou() == 0.0

    def test_batched_shape(self):
        """Accepts [B, H, W] shaped arrays."""
        m = SegmentationMetrics(num_classes=3)
        target = np.array([[[0, 1], [2, 0]], [[1, 2], [0, 1]]])  # [2, 2, 2]
        pred = target.copy()
        m.update(pred=pred, target=target)
        assert m.miou() == pytest.approx(1.0)
        assert m.confusion.sum() == 8


class TestSegmentationMetricsPerClass:
    """Per-class accuracy and overall accuracy."""

    def test_per_class_accuracy(self):
        """Per-class accuracy = TP / (TP + FN)."""
        m = SegmentationMetrics(num_classes=3)
        # Class 0: 2 correct out of 3 GT → accuracy = 2/3
        # Class 1: 1 correct out of 2 GT → accuracy = 1/2
        # Class 2: 1 correct out of 1 GT → accuracy = 1.0
        target = np.array([0, 0, 0, 1, 1, 2])
        pred = np.array([0, 0, 1, 1, 0, 2])
        m.update(pred=pred, target=target)
        acc = m.per_class_accuracy()
        assert acc["background"] == pytest.approx(2 / 3, abs=1e-4)
        assert acc["head"] == pytest.approx(0.5)
        assert acc["neck"] == pytest.approx(1.0)

    def test_overall_accuracy(self):
        """Overall pixel accuracy."""
        m = SegmentationMetrics(num_classes=3)
        target = np.array([0, 0, 0, 1, 1, 2])
        pred = np.array([0, 0, 1, 1, 0, 2])
        m.update(pred=pred, target=target)
        # 4 correct out of 6 pixels.
        assert m.overall_accuracy() == pytest.approx(4 / 6, abs=1e-4)

    def test_absent_class_zero_iou(self):
        """Classes with no GT pixels get IoU = 0 and are excluded from mIoU."""
        m = SegmentationMetrics(num_classes=5)
        target = np.array([0, 0, 1, 1])
        pred = np.array([0, 0, 1, 1])
        m.update(pred=pred, target=target)
        ious = m.per_class_iou()
        # Classes 2, 3, 4 have zero GT.
        assert ious["neck"] == 0.0
        assert ious["chest"] == 0.0
        assert ious["spine"] == 0.0
        # mIoU should only average over classes 0 and 1.
        assert m.miou() == pytest.approx(1.0)


class TestSegmentationMetrics22Classes:
    """Full 22-class setup matching the actual model."""

    def test_region_name_coverage(self):
        """per_class_iou should return all 22 region names."""
        m = SegmentationMetrics(num_classes=22)
        ious = m.per_class_iou()
        assert len(ious) == 22
        assert "background" in ious
        assert "head" in ious
        assert "shoulder_r" in ious
        assert "unused" in ious
        assert "accessory" in ious


# ---------------------------------------------------------------------------
# JointMetrics
# ---------------------------------------------------------------------------


class TestJointMetricsBasic:
    """Core joint error and presence accuracy."""

    def test_perfect_offsets(self):
        """Zero offset error when predictions match ground truth."""
        m = JointMetrics(num_joints=3)
        offsets = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])  # [1, 3, 2]
        visible = np.array([[True, True, True]])
        m.update(
            pred_offsets=offsets,
            gt_offsets=offsets,
            pred_present=visible,
            gt_visible=visible,
        )
        assert m.mean_offset_error() == pytest.approx(0.0)
        assert m.presence_accuracy() == pytest.approx(1.0)

    def test_known_offset_error(self):
        """Hand-computed MSE for known offsets.

        Joint 0: pred=(1,0) gt=(0,0) → sq_err = 1
        Joint 1: pred=(0,2) gt=(0,0) → sq_err = 4
        Mean = (1 + 4) / 2 = 2.5
        """
        m = JointMetrics(num_joints=2)
        pred = np.array([[[1.0, 0.0], [0.0, 2.0]]])
        gt = np.array([[[0.0, 0.0], [0.0, 0.0]]])
        visible = np.array([[True, True]])
        m.update(pred_offsets=pred, gt_offsets=gt, pred_present=visible, gt_visible=visible)
        assert m.mean_offset_error() == pytest.approx(2.5)

    def test_invisible_joints_excluded(self):
        """Invisible GT joints should not contribute to offset error."""
        m = JointMetrics(num_joints=2)
        pred = np.array([[[100.0, 100.0], [0.0, 0.0]]])
        gt = np.array([[[0.0, 0.0], [0.0, 0.0]]])
        visible = np.array([[False, True]])
        m.update(pred_offsets=pred, gt_offsets=gt, pred_present=visible, gt_visible=visible)
        # Only joint 1 counted, error = 0.
        assert m.mean_offset_error() == pytest.approx(0.0)

    def test_empty_metrics(self):
        """No updates → zero error, zero accuracy."""
        m = JointMetrics(num_joints=3)
        assert m.mean_offset_error() == 0.0
        assert m.presence_accuracy() == 0.0


class TestJointMetricsPresence:
    """Presence/visibility prediction accuracy."""

    def test_presence_accuracy(self):
        """Presence accuracy = correct / total."""
        m = JointMetrics(num_joints=3)
        gt_vis = np.array([[True, False, True]])
        pred_vis = np.array([[True, True, True]])  # 1 wrong (joint 1)
        offsets = np.zeros((1, 3, 2))
        m.update(
            pred_offsets=offsets,
            gt_offsets=offsets,
            pred_present=pred_vis,
            gt_visible=gt_vis,
        )
        assert m.presence_accuracy() == pytest.approx(2 / 3, abs=1e-4)

    def test_all_wrong_presence(self):
        """All presence predictions wrong."""
        m = JointMetrics(num_joints=2)
        gt_vis = np.array([[True, False]])
        pred_vis = np.array([[False, True]])
        offsets = np.zeros((1, 2, 2))
        m.update(
            pred_offsets=offsets,
            gt_offsets=offsets,
            pred_present=pred_vis,
            gt_visible=gt_vis,
        )
        assert m.presence_accuracy() == pytest.approx(0.0)


class TestJointMetricsStreaming:
    """Multi-batch accumulation and reset."""

    def test_multi_batch_accumulation(self):
        """Two batches should accumulate correctly."""
        m = JointMetrics(num_joints=2)
        offsets = np.zeros((1, 2, 2))
        visible = np.array([[True, True]])

        # Batch 1: joint 0 error = 1, joint 1 error = 0
        pred1 = np.array([[[1.0, 0.0], [0.0, 0.0]]])
        m.update(pred_offsets=pred1, gt_offsets=offsets, pred_present=visible, gt_visible=visible)

        # Batch 2: joint 0 error = 0, joint 1 error = 4
        pred2 = np.array([[[0.0, 0.0], [0.0, 2.0]]])
        m.update(pred_offsets=pred2, gt_offsets=offsets, pred_present=visible, gt_visible=visible)

        # Joint 0: mean sq err = (1 + 0) / 2 = 0.5
        # Joint 1: mean sq err = (0 + 4) / 2 = 2.0
        # Overall mean = (0.5 + 2.0) / 2 = 1.25
        assert m.mean_offset_error() == pytest.approx(1.25)

    def test_reset_clears_state(self):
        """Reset should zero all accumulators."""
        m = JointMetrics(num_joints=2)
        pred = np.array([[[1.0, 0.0], [0.0, 2.0]]])
        gt = np.zeros((1, 2, 2))
        visible = np.array([[True, True]])
        m.update(pred_offsets=pred, gt_offsets=gt, pred_present=visible, gt_visible=visible)

        m.reset()
        assert m.mean_offset_error() == 0.0
        assert m.presence_accuracy() == 0.0
        assert m._joint_counts.sum() == 0

    def test_per_joint_error_names(self):
        """per_joint_error should use BONE_ORDER names for 20-joint setup."""
        m = JointMetrics(num_joints=20)
        errors = m.per_joint_error()
        assert "hips" in errors
        assert "head" in errors
        assert "hair_back" in errors
        assert len(errors) == 20

    def test_per_joint_error_values(self):
        """Per-joint error dict should have correct values."""
        m = JointMetrics(num_joints=2)
        pred = np.array([[[1.0, 0.0], [0.0, 2.0]]])
        gt = np.zeros((1, 2, 2))
        visible = np.array([[True, True]])
        m.update(pred_offsets=pred, gt_offsets=gt, pred_present=visible, gt_visible=visible)
        errors = m.per_joint_error()
        assert errors["hips"] == pytest.approx(1.0)
        assert errors["spine"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# WeightMetrics
# ---------------------------------------------------------------------------


class TestWeightMetricsBasic:
    """Core weight MAE and confidence accuracy."""

    def test_perfect_predictions(self):
        """Zero MAE when predictions match ground truth."""
        m = WeightMetrics(num_bones=3)
        gt = np.array([[[0.6, 0.0], [0.4, 0.0], [0.0, 1.0]]])  # [1, 3, 2]
        conf = np.array([[1.0, 1.0]])  # [1, 2]
        m.update(
            pred_weights=gt,
            gt_weights=gt,
            pred_confidence=conf.astype(bool),
            gt_confidence=conf,
        )
        assert m.mae() == pytest.approx(0.0)

    def test_known_mae(self):
        """Hand-computed MAE for known weights.

        Bone 0: pred=0.5 gt=0.7 → abs_err = 0.2
        Bone 1: pred=0.5 gt=0.3 → abs_err = 0.2
        Mean = (0.2 + 0.2) / 2 = 0.2
        """
        m = WeightMetrics(num_bones=2)
        pred = np.array([[[0.5], [0.5]]])  # [1, 2, 1]
        gt = np.array([[[0.7], [0.3]]])
        conf = np.array([[1.0]])
        m.update(
            pred_weights=pred,
            gt_weights=gt,
            pred_confidence=conf.astype(bool),
            gt_confidence=conf,
        )
        assert m.mae() == pytest.approx(0.2)

    def test_zero_confidence_excluded(self):
        """Vertices without GT data should not contribute to MAE."""
        m = WeightMetrics(num_bones=2)
        pred = np.array([[[0.5, 99.0], [0.5, 99.0]]])
        gt = np.array([[[0.7, 0.0], [0.3, 0.0]]])
        conf = np.array([[1.0, 0.0]])  # Only first vertex has GT
        m.update(
            pred_weights=pred,
            gt_weights=gt,
            pred_confidence=conf.astype(bool),
            gt_confidence=conf,
        )
        # Only vertex 0 counted
        assert m.mae() == pytest.approx(0.2)

    def test_empty_metrics(self):
        """No updates → zero MAE and accuracy."""
        m = WeightMetrics(num_bones=3)
        assert m.mae() == 0.0
        assert m.confidence_accuracy() == 0.0

    def test_confidence_accuracy(self):
        """Confidence accuracy = correct / total."""
        m = WeightMetrics(num_bones=2)
        pred = np.zeros((1, 2, 3))
        gt = np.zeros((1, 2, 3))
        pred_conf = np.array([[True, False, True]])
        gt_conf = np.array([[1.0, 0.0, 0.0]])  # 2 correct, 1 wrong
        m.update(
            pred_weights=pred,
            gt_weights=gt,
            pred_confidence=pred_conf,
            gt_confidence=gt_conf,
        )
        assert m.confidence_accuracy() == pytest.approx(2 / 3, abs=1e-4)


class TestWeightMetricsStreaming:
    """Multi-batch accumulation and reset."""

    def test_reset_clears_state(self):
        """Reset should zero all accumulators."""
        m = WeightMetrics(num_bones=2)
        pred = np.array([[[0.5], [0.5]]])
        gt = np.array([[[0.7], [0.3]]])
        conf = np.array([[1.0]])
        m.update(
            pred_weights=pred,
            gt_weights=gt,
            pred_confidence=conf.astype(bool),
            gt_confidence=conf,
        )
        m.reset()
        assert m.mae() == 0.0
        assert m.confidence_accuracy() == 0.0

    def test_per_bone_mae_names(self):
        """per_bone_mae should use BONE_ORDER names for 20-bone setup."""
        m = WeightMetrics(num_bones=20)
        errors = m.per_bone_mae()
        assert "hips" in errors
        assert "head" in errors
        assert "hair_back" in errors
        assert len(errors) == 20

    def test_respects_num_vertices(self):
        """Should only consider vertices up to num_vertices."""
        m = WeightMetrics(num_bones=2)
        pred = np.zeros((1, 2, 10))
        gt = np.zeros((1, 2, 10))
        pred[0, :, 5:] = 99.0  # Junk in padded region
        gt[0, :, :5] = 0.5
        conf = np.ones((1, 10))
        num_verts = np.array([5])
        m.update(
            pred_weights=pred,
            gt_weights=gt,
            pred_confidence=conf.astype(bool),
            gt_confidence=conf,
            num_vertices=num_verts,
        )
        # Should only see error from first 5 vertices
        assert m.mae() == pytest.approx(0.5)
        assert m._conf_total == 5
