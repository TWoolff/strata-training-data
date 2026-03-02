# Issue #127: Implement training metrics — streaming mIoU and confusion matrix

## Understanding
- Need streaming metrics classes for training loop validation
- `SegmentationMetrics`: confusion matrix-based mIoU, per-class IoU, accuracy
- `JointMetrics`: per-joint MSE and presence accuracy
- Both operate on numpy arrays (decoupled from PyTorch tensors)
- Used by training scripts and referenced via `early_stopping_metric: "val/miou"`

## Approach
- Single new file: `training/utils/metrics.py`
- `SegmentationMetrics` uses a `(num_classes, num_classes)` int64 confusion matrix
  - IoU per class = TP / (TP + FP + FN) from confusion matrix diagonal and margins
  - Classes with zero GT pixels excluded from mean IoU
  - Region names from `pipeline/config.py` REGION_NAMES + "unused" (20) + "accessory" (21)
- `JointMetrics` accumulates per-joint squared errors and presence correct/total
  - 20 joints matching BONE_ORDER from `training/data/transforms.py`
  - Joint names from BONE_ORDER for per-joint error reporting
- Pure numpy — no torch dependency in metrics module itself

## Files to Modify
- **`training/utils/metrics.py`** (NEW) — Both metrics classes
- **`training/utils/__init__.py`** — Export metrics classes
- **`tests/test_training_metrics.py`** (NEW) — Unit tests

## Implementation Details

### SegmentationMetrics
- `__init__(num_classes=22, ignore_index=-1)` — 22 classes matching model output
- `update(pred, target)` — pred/target are `[B, H, W]` int arrays
  - Flatten, mask ignore_index, accumulate `confusion[target, pred] += 1`
- `miou()` — mean IoU excluding classes with zero GT pixels
- `per_class_iou()` — dict[str, float] using extended region names
- `per_class_accuracy()` — TP / (TP + FN) per class
- `overall_accuracy()` — sum(diagonal) / sum(matrix)
- `reset()` — zero the confusion matrix

### JointMetrics
- `__init__(num_joints=20)` — 20 joints matching BONE_ORDER
- `update(pred_offsets, gt_offsets, pred_present, gt_visible)` — batch arrays
  - Accumulate per-joint squared error for visible joints only
  - Track presence prediction accuracy
- `mean_offset_error()` — mean across all joints
- `per_joint_error()` — dict[str, float] using BONE_ORDER names
- `presence_accuracy()` — correct / total
- `reset()` — zero accumulators

### Region name lookup
- IDs 0–19: `REGION_NAMES[id]` from pipeline config
- ID 20: "unused"
- ID 21: "accessory"
- Build this mapping in metrics module to avoid circular imports

## Risks & Edge Cases
- Division by zero when union=0 for a class (class not in GT or pred) → return 0.0 and exclude from mean
- Empty predictions (all background) → should still work, mIoU over present classes
- Single class only → mIoU = that class's IoU
- Joints with no visible GT → exclude from per-joint error
- Large batch accumulation → int64 confusion matrix handles up to ~9.2e18 pixels

## Open Questions
- None — issue spec is clear and matches existing codebase conventions

## Implementation Notes
- Implemented as planned with no deviations from the approach
- `BONE_ORDER` imported from `training.data.transforms` (single source of truth) instead of duplicating
- Confusion matrix uses `np.bincount` trick for efficient vectorized accumulation
- 24 unit tests covering: perfect/wrong/known predictions, ignore_index, multi-batch streaming, reset, batched shapes, per-class accuracy, 22-class region names, joint offset error, invisible joint exclusion, presence accuracy, per-joint named errors
- All tests pass, lint clean, format clean
